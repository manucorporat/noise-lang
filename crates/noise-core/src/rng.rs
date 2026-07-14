//! Counter-based, keyed RNG for the random-variable hot loop (PLAN-PREGPU Track C).
//!
//! No `getrandom`, no `std::time`, no OS entropy, no threads — so `noise-core` stays
//! WASM-clean and sampling is fully deterministic for a given seed. The hot operations
//! are the keyed `fill_*` functions, each writing a whole column in one tight
//! (vectorizable) loop of independent per-lane hashes.

// ---------------------------------------------------------------------------
// Counter-based generator (PLAN-PREGPU Track C): Squares (Widynski, arXiv:2004.06278),
// keyed per draw counter.
//
// One `squares64` call per draw counter — no state chain, so any lane range can be
// computed independently (reducer chunks are just ranges, thread count can't matter)
// and every backend (interpreter, JIT, wasm, later WGSL) produces bit-identical draws.
// Squares replaced the pcg4d family after C0's criterion 8: every GPU-cheap pcg variant
// showed real sequential structure by 256 GB of PractRand, while squares (with a
// construction-compliant key) is clean at 1 TB with zero anomalies — evidence in
// tools/rng-cert/RESULTS.md.
//
// Counter layout (the whole u64 space, disjoint by construction):
//   * scalar fills:  `(source << 36) + lane` — sequential per source (the certified
//     regime), one counter per lane, 2^36 counters per source ≫ the 2^32 lane cap.
//     `normal` consumes the pair's two counters (even lane's and odd lane's) for one
//     Box–Muller pair.
//   * cell streams (`CellStream`: Knuth poisson, Permutation, Rotation — variable or
//     large per-lane consumption): `(1 << 63) | (stream << 49) | (lane << 17) | j`,
//     where `stream` is a compile-time per-program ordinal (< 2^14) and `j` counts the
//     lane's sequential draws (< 2^17 — permutation n and rotation d² cap there).
//
// Consumption contract (C0): only bits 8..31 of each u32 half are consumable (never a
// low byte). One squares64 output yields the 48-bit uniform
// `((w >> 40) << 24) | ((w >> 8) & 0xFFFFFF)`.
// ---------------------------------------------------------------------------

/// Largest Poisson `lambda` sampled by the exact Knuth loop; above this `fill_poisson` uses the
/// normal approximation (see [`fill_poisson`]). Chosen below the `(-lambda).exp()` underflow
/// point (`lambda ≈ 745`) so the Knuth path is always exact, and low enough that its `O(lambda)`
/// cost per draw stays a few hundred iterations. The Gaussian approximation is excellent well
/// before here (the Poisson is already near-Gaussian by `lambda ≈ 20`).
pub const POISSON_KNUTH_MAX: f64 = 500.0;

/// squares64 (Widynski): five middle-square rounds over `ctr * key`, 64-bit output.
#[inline(always)]
pub fn squares64(ctr: u64, key: u64) -> u64 {
    let mut x = ctr.wrapping_mul(key);
    let y = x;
    let z = y.wrapping_add(key);
    x = x.wrapping_mul(x).wrapping_add(y);
    x = x.rotate_left(32);
    x = x.wrapping_mul(x).wrapping_add(z);
    x = x.rotate_left(32);
    x = x.wrapping_mul(x).wrapping_add(y);
    x = x.rotate_left(32);
    let t = x.wrapping_mul(x).wrapping_add(z);
    x = t.rotate_left(32);
    t ^ (x.wrapping_mul(x).wrapping_add(y) >> 32)
}

/// Per-run key, built by Widynski's key construction seeded from the user seed: each
/// 8-hex-digit half uses **distinct non-zero digits** and the least significant digit is
/// **odd** — the family whose statistical certification C0's 1 TB run validated (an
/// arbitrary u64, e.g. a raw SplitMix output, is NOT a valid squares key; C0 harness
/// defect #3 measured exactly that failure). Stored split so the kernel ABI can pass two
/// 32-bit words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub k0: u32,
    pub k1: u32,
}

impl Key {
    pub fn from_seed(seed: u64) -> Self {
        // SplitMix64 stream drives the digit picks (bias from `% len` is ≤ 2^-60).
        let mut z = seed;
        let mut next = move || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        let mut key = 0u64;
        // Least significant digit: odd. Then 7 more distinct non-zero digits for the
        // low half, then 8 distinct non-zero digits for the high half.
        const ODD: [u8; 8] = [1, 3, 5, 7, 9, 11, 13, 15];
        let lsd = ODD[(next() % 8) as usize];
        key |= lsd as u64;
        let mut pool: Vec<u8> = (1..=15).filter(|&d| d != lsd).collect();
        for pos in 1..8 {
            let i = (next() % pool.len() as u64) as usize;
            key |= (pool.swap_remove(i) as u64) << (4 * pos);
        }
        let mut pool: Vec<u8> = (1..=15).collect();
        for pos in 8..16 {
            let i = (next() % pool.len() as u64) as usize;
            key |= (pool.swap_remove(i) as u64) << (4 * pos);
        }
        Key { k0: key as u32, k1: (key >> 32) as u32 }
    }

    #[inline(always)]
    pub fn as_u64(self) -> u64 {
        (self.k1 as u64) << 32 | self.k0 as u64
    }
}

/// The 48 consumed bits of one squares64 output at `ctr`: the top 24 bits of each u32
/// half (a low byte is never consumed — the C0 contract, kept uniform across generators).
#[inline(always)]
pub fn draw48(key: Key, ctr: u64) -> u64 {
    let w = squares64(ctr, key.as_u64());
    ((w >> 40) << 24) | ((w >> 8) & 0xFF_FFFF)
}

/// Scalar-fill counter: sequential per source, one per lane.
#[inline(always)]
pub fn scalar_ctr(source: u32, lane: u32) -> u64 {
    ((source as u64) << 36) + lane as u64
}

const SCALE48: f64 = 1.0 / (1u64 << 48) as f64;

/// Sequential per-cell-stream draw supply for fills that consume a variable or
/// per-lane-large number of draws (Knuth `poisson`, `Permutation`'s Fisher–Yates,
/// `Rotation`'s Gaussian seed). Counters live in the dedicated region
/// `(1 << 63) | (stream << 49) | (lane << 17) | j` — `stream` is a per-program ordinal
/// assigned at lowering (compile-time constant, like a source offset), and `j` is the
/// lane's draw index, so consumption is a pure function of `(stream, lane, j)` and any
/// lane range is independently computable.
pub struct CellStream {
    key: Key,
    base: u64,
    j: u32,
}

/// Per-lane draw budget of a [`CellStream`] (`j < 2^17`): permutation `n` and rotation
/// `d²` cap here — far above every real program, stated rather than discovered.
pub const CELL_STREAM_MAX_DRAWS: u32 = 1 << 17;

impl CellStream {
    #[inline]
    pub fn new(key: Key, stream: u32, lane: u32) -> Self {
        debug_assert!(stream < (1 << 14), "cell-stream ordinal over 2^14");
        let base = (1u64 << 63) | ((stream as u64) << 49) | ((lane as u64) << 17);
        CellStream { key, base, j: 0 }
    }

    /// The consumed 48 bits of the next draw.
    #[inline]
    pub fn next_u48(&mut self) -> u64 {
        debug_assert!(self.j < CELL_STREAM_MAX_DRAWS, "cell stream over its draw budget");
        let bits = draw48(self.key, self.base + self.j as u64);
        self.j += 1;
        bits
    }

    /// Uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn next_unit(&mut self) -> f64 {
        self.next_u48() as f64 * SCALE48
    }

    /// Integer uniform in `0..count` via Lemire multiply-high on 48 bits.
    #[inline]
    pub fn next_bounded(&mut self, count: u64) -> u64 {
        ((self.next_u48() as u128 * count as u128) >> 48) as u64
    }

    /// Bulk-fill `out` with the exact u48 stream [`next_u48`] yields, through a hot loop
    /// whose iterations are independent (one squares64 each) — this is what lets a
    /// `Rotation`'s per-lane Gaussian seed overlap its multiply chains instead of
    /// serializing through the scalar path (PLAN-PERF-3 item 1).
    pub fn fill_u48s(&mut self, out: &mut [u64]) {
        debug_assert!(
            self.j as usize + out.len() <= CELL_STREAM_MAX_DRAWS as usize,
            "cell stream over its draw budget"
        );
        for x in out.iter_mut() {
            *x = draw48(self.key, self.base + self.j as u64);
            self.j += 1;
        }
    }

    /// Bulk normals: pair `t` of `out` comes from u48s `2t`/`2t+1` — one Box–Muller
    /// evaluation yields BOTH branches (cos to the even slot, sin to the odd), so a
    /// `Rotation`'s `d²` Gaussian seed costs one `ln` + one two-branch trig per TWO
    /// entries, and the two-phase shape (u48s via [`fill_u48s`] into a reused caller
    /// `scratch`, then independent pairs) lets the draw loop and the transcendental
    /// chains overlap. `CellStream` is interpreter-only (Rotation / Permutation /
    /// Poisson never reach codegen), so this consumption schedule has no cross-backend
    /// parity constraint — but a future WGSL rotation kernel must mirror it.
    pub fn fill_normals(&mut self, out: &mut [f64], scratch: &mut Vec<u64>) {
        use std::f64::consts::TAU;
        let pairs = out.len().div_ceil(2);
        scratch.clear();
        scratch.resize(2 * pairs, 0);
        self.fill_u48s(scratch);
        for (t, uu) in scratch.chunks_exact(2).enumerate() {
            let u1 = (uu[0] as f64 + 0.5) * SCALE48;
            let u2 = uu[1] as f64 * SCALE48;
            let r = (-2.0 * crate::approx::ln(u1)).sqrt();
            let theta = TAU * u2;
            let k = 2 * t;
            out[k] = r * crate::approx::cos(theta);
            if k + 1 < out.len() {
                out[k + 1] = r * crate::approx::sin(theta);
            }
        }
    }
}

/// Uniform `f64` in `[0, 1)` for one scalar draw cell.
#[inline(always)]
pub fn unit_uniform(key: Key, lane: u32, source: u32) -> f64 {
    draw48(key, scalar_ctr(source, lane)) as f64 * SCALE48
}

/// Fill a column with `lo + (hi - lo) * u01` for lanes `lane0..`.
#[inline]
pub fn fill_uniform(key: Key, source: u32, lane0: u32, lo: f64, hi: f64, out: &mut [f64]) {
    let span = hi - lo;
    for (i, x) in out.iter_mut().enumerate() {
        *x = lo + span * unit_uniform(key, lane0.wrapping_add(i as u32), source);
    }
}

/// Integers uniform over `lo..=hi` (inclusive) as `f64`, via Lemire multiply-high on the
/// 48 consumed bits (bias ≤ `count / 2^48`; Track B's 2^24 cap makes it ≤ 2^-24).
#[inline]
pub fn fill_uniform_int(key: Key, source: u32, lane0: u32, lo: f64, hi: f64, out: &mut [f64]) {
    debug_assert!(
        hi >= lo,
        "fill_uniform_int needs hi >= lo (constant bounds are validated upstream), got ({lo}, {hi})"
    );
    let count = (hi - lo + 1.0).max(1.0) as u64;
    for (i, x) in out.iter_mut().enumerate() {
        let bits = draw48(key, scalar_ctr(source, lane0.wrapping_add(i as u32)));
        let k = ((bits as u128 * count as u128) >> 48) as u64;
        *x = lo + k as f64;
    }
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate`. Uses the shared [`crate::approx::ln`]
/// (not libm) so the JIT/wasm/WGSL lowerings — which inline exactly that polynomial —
/// produce bit-identical draws (PLAN-PREGPU draw-stream parity).
#[inline]
pub fn fill_exp(key: Key, source: u32, lane0: u32, rate: f64, out: &mut [f64]) {
    for (i, x) in out.iter_mut().enumerate() {
        let u = unit_uniform(key, lane0.wrapping_add(i as u32), source);
        *x = -crate::approx::ln(1.0 - u) / rate;
    }
}

/// `Geometric(p)` (failures before first success) via `floor(ln(u)/ln(1-p))`, `u ∈ (0, 1]`.
/// Shared-`approx` transcendentals for cross-backend bit parity (see [`fill_exp`]); the
/// compile-time `denom` stays libm (a constant, folded identically everywhere).
#[inline]
pub fn fill_geometric(key: Key, source: u32, lane0: u32, p: f64, out: &mut [f64]) {
    let denom = (1.0 - p).ln();
    for (i, x) in out.iter_mut().enumerate() {
        let u = 1.0 - unit_uniform(key, lane0.wrapping_add(i as u32), source);
        *x = (crate::approx::ln(u) / denom).floor();
    }
}

/// The Box–Muller pair for the even/odd lane pair `(lane & !1, lane | 1)`: `u1` from the
/// even lane's counter (offset half an ulp of the 48-bit grid to dodge `ln(0)`), `u2` from
/// the odd lane's; the even lane takes the cos branch, the odd lane the sin branch.
/// Shared-`approx` `ln`/`sin`/`cos` (not libm) for cross-backend bit parity — the
/// JIT/wasm/WGSL lowerings inline exactly those polynomials (and their trig kernel computes
/// both branches anyway, so a lowering that emits one lane per iteration just selects by
/// lane parity at no real cost). `TAU·u2 < 2π` is always in the exact-reduction range.
///
/// Pairing is why every fill must start on an EVEN lane: batch (1024) and reducer-chunk
/// (16384) boundaries all are, so a pair never straddles a range split — asserted in the
/// fills, stated here rather than discovered.
#[inline]
pub fn normal_pair(key: Key, even_lane: u32, source: u32) -> (f64, f64) {
    use std::f64::consts::TAU;
    debug_assert_eq!(even_lane & 1, 0);
    let bits1 = draw48(key, scalar_ctr(source, even_lane));
    let u1 = (bits1 as f64 + 0.5) * SCALE48; // (0, 1)
    let u2 = draw48(key, scalar_ctr(source, even_lane | 1)) as f64 * SCALE48;
    let r = (-2.0 * crate::approx::ln(u1)).sqrt();
    let theta = TAU * u2;
    (r * crate::approx::cos(theta), r * crate::approx::sin(theta))
}

/// `N(mu, sigma^2)` via Box–Muller over lane pairs (see [`normal_pair`]): the pair's two
/// counters, one `ln`, one two-branch trig evaluation per TWO lanes — the pair sharing
/// xoshiro had, kept under counter keying by pinning the pair to the even lane.
#[inline]
pub fn fill_normal(key: Key, source: u32, lane0: u32, mu: f64, sigma: f64, out: &mut [f64]) {
    debug_assert_eq!(lane0 & 1, 0, "fills start on even lanes (batch-aligned)");
    let mut i = 0;
    while i < out.len() {
        let (z0, z1) = normal_pair(key, lane0.wrapping_add(i as u32), source);
        out[i] = mu + sigma * z0;
        i += 1;
        if i < out.len() {
            out[i] = mu + sigma * z1;
            i += 1;
        }
    }
}

/// `Poisson(lambda)`: Knuth's product loop below [`POISSON_KNUTH_MAX`] (uniforms from the
/// per-cell [`CellStream`] chain — multiply uniforms until the running product drops below
/// `e^-lambda`; exact and `O(lambda)` per draw), the standard normal approximation
/// `round(max(0, N(lambda, sqrt(lambda))))` above it. Knuth's loop is a **hang** and silently
/// biased for large lambda (past `lambda ≈ 745` the target underflows to 0), so the switch is
/// mandatory, not an optimization; the threshold sits well inside the Knuth-exact regime and the
/// Gaussian is excellent from `lambda ≈ 20` (finding A8).
/// `source` keys the normal-approximation path (a scalar fill); `stream` is the
/// per-program cell-stream ordinal keying the Knuth loop's counters.
#[inline]
pub fn fill_poisson(key: Key, source: u32, stream: u32, lane0: u32, lambda: f64, out: &mut [f64]) {
    if lambda > POISSON_KNUTH_MAX {
        fill_normal(key, source, lane0, lambda, lambda.sqrt(), out);
        for x in out.iter_mut() {
            *x = x.round().max(0.0);
        }
        return;
    }
    let l = (-lambda).exp();
    for (i, x) in out.iter_mut().enumerate() {
        let mut s = CellStream::new(key, stream, lane0.wrapping_add(i as u32));
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= s.next_unit();
            if p <= l {
                break;
            }
        }
        *x = (k - 1) as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key construction must always yield a Widynski-compliant key: distinct non-zero
    /// hex digits within each 8-digit half, least significant digit odd — for ANY seed.
    /// (C0 harness defect #3 measured what a non-compliant key does: PractRand failures at
    /// 256 GB. This invariant is the certification's precondition.)
    #[test]
    fn key_construction_is_compliant() {
        for seed in [0u64, 1, 42, 0xC0FFEE, u64::MAX, 0x9E37_79B9_7F4A_7C15] {
            let k = Key::from_seed(seed).as_u64();
            assert_eq!(k & 1, 1, "seed {seed}: LSD must be odd (key {k:016x})");
            for half in [k & 0xFFFF_FFFF, k >> 32] {
                let digits: Vec<u64> = (0..8).map(|i| (half >> (4 * i)) & 0xF).collect();
                assert!(digits.iter().all(|&d| d != 0), "seed {seed}: zero digit in {half:08x}");
                let mut seen = [false; 16];
                for &d in &digits {
                    assert!(!seen[d as usize], "seed {seed}: repeated digit in {half:08x}");
                    seen[d as usize] = true;
                }
            }
        }
    }

    /// Known-answer test guarding the squares64 rounds, the seeded key construction, and
    /// the 48-bit consumption contract. These vectors are the cross-backend draw contract:
    /// the JIT, the wasm emitter, and later the WGSL emitter must reproduce them bit for
    /// bit. Reference: tools/rng-cert (C0).
    #[test]
    fn keyed_known_answer() {
        let key = Key::from_seed(0);
        assert_eq!(key.as_u64(), KAT_KEY0);
        assert_eq!(draw48(key, scalar_ctr(0, 0)), KAT_DRAWS[0]);
        assert_eq!(draw48(key, scalar_ctr(0, 1)), KAT_DRAWS[1]);
        assert_eq!(draw48(key, scalar_ctr(1, 0)), KAT_DRAWS[2]);
        assert_eq!(draw48(key, scalar_ctr(7, 12345)), KAT_DRAWS[3]);
        // Uniform KATs on exact bit patterns (bits, not decimal literals, are the contract).
        assert_eq!(unit_uniform(key, 0, 0).to_bits(), KAT_UNITS[0]);
        assert_eq!(unit_uniform(key, 1, 0).to_bits(), KAT_UNITS[1]);
        assert_eq!(unit_uniform(key, 0, 1).to_bits(), KAT_UNITS[2]);
        assert_eq!(unit_uniform(key, 12345, 7).to_bits(), KAT_UNITS[3]);
    }

    /// Regeneration helper for the KAT constants above (prints the current values):
    /// `cargo test -p noise-core --release -- --ignored --nocapture print_kat_vectors`
    #[test]
    #[ignore]
    fn print_kat_vectors() {
        let key = Key::from_seed(0);
        println!("KAT_KEY0 = 0x{:016X}", key.as_u64());
        for (src, lane) in [(0u32, 0u32), (0, 1), (1, 0), (7, 12345)] {
            println!("draw48(src {src}, lane {lane}) = 0x{:012X}", draw48(key, scalar_ctr(src, lane)));
        }
        for (lane, src) in [(0u32, 0u32), (1, 0), (0, 1), (12345, 7)] {
            println!("unit({lane},{src}).to_bits() = 0x{:016X}", unit_uniform(key, lane, src).to_bits());
        }
    }

    const KAT_KEY0: u64 = 0x432A_B7DF_C618_529F;
    const KAT_DRAWS: [u64; 4] = [
        0xCBA7_C589_BB1D,
        0x1C74_645A_5F00,
        0x127B_1743_B147,
        0x552C_3662_C4EC,
    ];
    const KAT_UNITS: [u64; 4] = [
        0x3FE9_74F8_B137_63A0,
        0x3FBC_7464_5A5F_0000,
        0x3FB2_7B17_43B1_4700,
        0x3FD5_4B0D_98B1_3B00,
    ];

    /// Counter keying means a fill is a pure function of the lane range: filling
    /// `[0, n)` in one call must equal filling any split of it — the property that makes
    /// reducer chunks trivially thread-count-invariant (no `chunk_seed` needed).
    #[test]
    fn keyed_fills_are_lane_range_pure() {
        let key = Key::from_seed(42);
        let n = 4096;
        let mut whole = vec![0.0; n];
        let mut split = vec![0.0; n];
        type Fill = fn(Key, u32, u32, f64, f64, &mut [f64]);
        let fills: [(Fill, f64, f64); 3] = [
            (fill_uniform, 2.0, 5.0),
            (fill_uniform_int, 1.0, 6.0),
            (fill_normal, 2.0, 3.0),
        ];
        for (fill, a, b) in fills {
            fill(key, 3, 0, a, b, &mut whole);
            // Even split point: fills start on even lanes (Box–Muller lane pairing; every real
            // range start — batch, reducer chunk — is a multiple of 1024).
            let (lo, hi) = split.split_at_mut(1026);
            fill(key, 3, 0, a, b, lo);
            fill(key, 3, 1026, a, b, hi);
            assert_eq!(whole, split);
        }
        let mut whole_p = vec![0.0; n];
        let mut split_p = vec![0.0; n];
        fill_poisson(key, 3, 0, 0, 4.5, &mut whole_p);
        let (lo, hi) = split_p.split_at_mut(1026);
        fill_poisson(key, 3, 0, 0, 4.5, lo);
        fill_poisson(key, 3, 0, 1026, 4.5, hi);
        assert_eq!(whole_p, split_p);
    }

    #[test]
    fn keyed_fills_match_requested_moments() {
        let key = Key::from_seed(7);
        let n = 200_000;
        let stats = |col: &[f64]| {
            let nf = col.len() as f64;
            let mean = col.iter().sum::<f64>() / nf;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf;
            (mean, var)
        };
        let mut col = vec![0.0; n];
        fill_normal(key, 0, 0, 2.0, 3.0, &mut col); // N(2, 9)
        let (mean, var) = stats(&col);
        assert!((mean - 2.0).abs() < 0.03, "normal mean = {mean}");
        assert!((var - 9.0).abs() < 0.15, "normal var = {var}");
        fill_uniform(key, 1, 0, 0.0, 1.0, &mut col); // mean 1/2, var 1/12
        let (mean, var) = stats(&col);
        assert!((mean - 0.5).abs() < 0.005, "uniform mean = {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.005, "uniform var = {var}");
        fill_exp(key, 2, 0, 2.0, &mut col); // mean 1/2, var 1/4
        let (mean, var) = stats(&col);
        assert!((mean - 0.5).abs() < 0.01, "exp mean = {mean}");
        assert!((var - 0.25).abs() < 0.02, "exp var = {var}");
        fill_poisson(key, 3, 0, 0, 4.5, &mut col); // mean = var = lambda
        let (mean, var) = stats(&col);
        assert!((mean - 4.5).abs() < 0.05, "poisson mean = {mean}");
        assert!((var - 4.5).abs() < 0.15, "poisson var = {var}");
        fill_geometric(key, 4, 0, 0.25, &mut col); // mean (1-p)/p = 3
        let (mean, _) = stats(&col);
        assert!((mean - 3.0).abs() < 0.05, "geometric mean = {mean}");
        fill_poisson(key, 5, 0, 0, 100_000.0, &mut col); // normal-approx regime
        let (mean, var) = stats(&col);
        assert!((mean / 1e5 - 1.0).abs() < 0.01, "poisson approx mean = {mean}");
        assert!((var / 1e5 - 1.0).abs() < 0.05, "poisson approx var = {var}");
    }

    #[test]
    fn keyed_uniform_int_is_uniform_over_range() {
        let key = Key::from_seed(99);
        let mut col = vec![0.0f64; 6_000_000];
        fill_uniform_int(key, 0, 0, 1.0, 6.0, &mut col);
        let mut counts = [0u64; 7];
        for &x in &col {
            assert!(
                (1.0..=6.0).contains(&x) && x.fract() == 0.0,
                "out-of-range face {x}"
            );
            counts[x as usize] += 1;
        }
        let expected = col.len() as f64 / 6.0;
        for face in 1..=6 {
            let dev = (counts[face] as f64 - expected).abs() / expected;
            assert!(
                dev < 0.01,
                "face {face}: count {} deviates {dev:.4} from uniform",
                counts[face]
            );
        }
    }

    #[test]
    fn poisson_large_lambda_is_fast_and_has_the_right_mean() {
        // Above POISSON_KNUTH_MAX the Knuth loop would hang (and be biased low) — the normal
        // approximation must return promptly with mean ≈ lambda and variance ≈ lambda. A huge
        // lambda (`1e12`) must not hang: this test *completing* is the proof.
        let key = Key::from_seed(3);
        let mut col = vec![0.0f64; 200_000];
        let lambda = 100_000.0;
        fill_poisson(key, 0, 0, 0, lambda, &mut col);
        let n = col.len() as f64;
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((mean / lambda - 1.0).abs() < 0.01, "mean = {mean}");
        assert!((var / lambda - 1.0).abs() < 0.05, "var = {var}");
        // The extreme case: just has to terminate, not hang.
        let mut one = [0.0f64; 8];
        fill_poisson(key, 1, 0, 1, 1e12, &mut one);
        assert!(one.iter().all(|&x| x.is_finite() && x >= 0.0));
    }

    /// Throughput of the keyed fills, single thread — the interpreter's RNG ceiling. The
    /// tools/rng-cert keyed-batch bench reaches ~236 M hashes/s (942 M u32 words/s) on this
    /// shape; fills that fall far short of it are leaving vectorization on the table. Run with:
    /// `cargo test -p noise-core --release -- --ignored --nocapture bench_keyed_fills`
    #[test]
    #[ignore]
    fn bench_keyed_fills() {
        use std::time::Instant;
        let key = Key::from_seed(1);
        let n = 1 << 22; // 4M lanes
        let mut col = vec![0.0f64; n];
        let mut time = |label: &str, f: &mut dyn FnMut(&mut [f64])| {
            f(&mut col); // warm
            let t = Instant::now();
            f(&mut col);
            let el = t.elapsed().as_secs_f64();
            std::hint::black_box(&col);
            println!("  {label:<28}{:>8.1} M draws/s", n as f64 / el / 1e6);
        };
        time("fill_uniform", &mut |c| fill_uniform(key, 0, 0, 0.0, 1.0, c));
        time("fill_uniform_int(1,6)", &mut |c| {
            fill_uniform_int(key, 1, 0, 1.0, 6.0, c)
        });
        time("fill_normal", &mut |c| fill_normal(key, 2, 0, 0.0, 1.0, c));
        time("fill_exp", &mut |c| fill_exp(key, 3, 0, 1.0, c));
        // Rotation-style consumption: d²=256 normals per lane from one CellStream (bulk).
        let d2 = 256;
        let lanes = n / d2;
        let mut scratch = Vec::new();
        let t = Instant::now();
        for lane in 0..lanes as u32 {
            let mut s = CellStream::new(key, lane, 4);
            s.fill_normals(
                &mut col[lane as usize * d2..(lane as usize + 1) * d2],
                &mut scratch,
            );
        }
        let el = t.elapsed().as_secs_f64();
        std::hint::black_box(&col);
        println!(
            "  {:<28}{:>8.1} M draws/s",
            "CellStream bulk normals",
            (lanes * d2) as f64 / el / 1e6
        );
    }

    #[test]
    fn unit_uniform_in_unit_interval() {
        let key = Key::from_seed(42);
        for lane in 0..10_000 {
            let x = unit_uniform(key, lane, 0);
            assert!((0.0..1.0).contains(&x));
        }
    }
}
