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
// and every backend (interpreter, wasm, WGSL) produces bit-identical draws.
// Squares replaced the pcg4d family after C0's criterion 8: every GPU-cheap pcg variant
// showed real sequential structure by 256 GB of PractRand, while squares (with a
// construction-compliant key) is clean at 1 TB with zero anomalies — evidence in
// tools/rng-cert/RESULTS.md.
//
// Counter layout (the whole u64 space, disjoint by construction):
//   * pair-shared scalar fills (`unif`, `normal`, `exp`, `geometric`):
//     `(source << 36) + (lane >> 1)` — sequential per source (the certified regime), ONE
//     counter per lane PAIR. With f32 lanes (Track B) a uniform needs 24 bits and a draw
//     supplies 48, so one hash feeds two lanes: even → low 24, odd → high 24.
//   * `unif_int`: `(source << 36) + lane` — one counter per lane, all 48 bits spent on
//     Lemire's multiply-high (24 would put the bias at count/2^24).
//   * cell streams (`CellStream`: Knuth poisson, Permutation, Rotation — variable or
//     large per-lane consumption): `(1 << 63) | (stream << 49) | (lane << 17) | j`,
//     where `stream` is a compile-time per-program ordinal (< 2^14) and `j` counts the
//     lane's sequential draws (< 2^17 — permutation n and rotation d² cap there).
//
// Consumption contract (C0): only bits 8..31 of each u32 half are consumable (never a
// low byte). One squares64 output yields the 48-bit uniform
// `((w >> 40) << 24) | ((w >> 8) & 0xFFFFFF)`. Every source walks its counters 0, 1, 2, …
// and spends all 48 bits of each — the byte stream C0 certified at 1 TB. Track B changed
// only which lane each half lands in, not the stream.
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

/// Scalar-fill counter: sequential per source, one per lane. Used by the fills that consume all
/// 48 bits in one lane (`unif_int`'s Lemire draw).
#[inline(always)]
pub fn scalar_ctr(source: u32, lane: u32) -> u64 {
    ((source as u64) << 36) + lane as u64
}

/// Scalar-fill counter of the LANE PAIR `lane` belongs to (PLAN-PREGPU Track B): with f32 lanes a
/// uniform needs only 24 bits, and one squares64 output carries 48 consumable ones — so **one hash
/// feeds two lanes**. The even lane takes the low 24 ([`lo24`]), the odd lane the high 24
/// ([`hi24`]).
///
/// The certified stream is untouched by this: a source still walks counters `0, 1, 2, …`
/// sequentially and every one of its 48 consumed bits is used exactly once, in order — C0's
/// criterion-8 evidence is over precisely this byte stream (tools/rng-cert `stream-sq64-engine`).
/// All Track B changed is which *lane* each half lands in.
#[inline(always)]
pub fn pair_ctr(source: u32, lane: u32) -> u64 {
    ((source as u64) << 36) + (lane >> 1) as u64
}

/// The even lane's 24 bits of a pair draw.
#[inline(always)]
pub fn lo24(bits48: u64) -> u32 {
    (bits48 & 0xFF_FFFF) as u32
}

/// The odd lane's 24 bits of a pair draw.
#[inline(always)]
pub fn hi24(bits48: u64) -> u32 {
    (bits48 >> 24) as u32
}

const SCALE48: f64 = 1.0 / (1u64 << 48) as f64;

/// f32 uniform grid: a 24-bit draw scales to `[0, 1)` with `2^-24` spacing — every value exactly
/// representable, and `1 - u` (what `ln` is fed) exactly representable too, which is what lets the
/// f32 Box–Muller drop the old half-ulp offset (see [`normal_pair`]).
const SCALE24: f32 = 1.0 / (1u32 << 24) as f32;

/// Uniform f32 in `[0, 1)` from a 24-bit draw — `bits · 2⁻²⁴`, exact.
#[inline(always)]
pub fn unit24(bits24: u32) -> f32 {
    bits24 as f32 * SCALE24
}

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
}

/// The two lanes' 24-bit draws of one pair — the shared head of every pair-shared fill.
/// `lane` may be either member of the pair; the counter is the pair's ([`pair_ctr`]).
#[inline(always)]
pub fn pair_bits(key: Key, source: u32, lane: u32) -> (u32, u32) {
    let bits = draw48(key, pair_ctr(source, lane));
    (lo24(bits), hi24(bits))
}

/// Fill a column with `lo + (hi - lo) * u01` for lanes `lane0..` — one hash per lane PAIR.
/// The `f64` bounds are rounded to f32 ONCE (`lo as f32`, `(hi - lo) as f32`), which is exactly
/// what both emitters bake in as constants, so the arithmetic agrees op-for-op.
#[inline]
pub fn fill_uniform(key: Key, source: u32, lane0: u32, lo: f64, hi: f64, out: &mut [f32]) {
    debug_assert_eq!(lane0 & 1, 0, "fills start on even lanes (pair-shared draws)");
    let (loc, span) = (lo as f32, (hi - lo) as f32);
    for (t, pair) in out.chunks_mut(2).enumerate() {
        let lane = lane0.wrapping_add(2 * t as u32);
        let (b0, b1) = pair_bits(key, source, lane);
        pair[0] = loc + span * unit24(b0);
        if let Some(x) = pair.get_mut(1) {
            *x = loc + span * unit24(b1);
        }
    }
}

/// Integers uniform over `lo..=hi` (inclusive) as `f32`, via Lemire multiply-high on the
/// 48 consumed bits (bias ≤ `count / 2^48` ≤ `2^-24` under Track B's 2^24 range cap).
///
/// The one fill that does NOT pair-share: 24 bits would put Lemire's bias at `count / 2^24` — up
/// to 1 at the cap — so this keeps its own per-lane counter and spends all 48 bits on one lane.
/// Every value in the range is exact in f32 because the range cap *is* f32's integer limit.
#[inline]
pub fn fill_uniform_int(key: Key, source: u32, lane0: u32, lo: f64, hi: f64, out: &mut [f32]) {
    debug_assert!(
        hi >= lo,
        "fill_uniform_int needs hi >= lo (constant bounds are validated upstream), got ({lo}, {hi})"
    );
    let count = (hi - lo + 1.0).max(1.0) as u64;
    let loc = lo as f32;
    for (i, x) in out.iter_mut().enumerate() {
        let bits = draw48(key, scalar_ctr(source, lane0.wrapping_add(i as u32)));
        let k = ((bits as u128 * count as u128) >> 48) as u64;
        *x = loc + k as f32;
    }
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate`, pair-shared. `1 - u ∈ [2⁻²⁴, 1]` — exact on
/// the f32 uniform grid, so `ln` is never fed 0. Uses the shared [`crate::approx::ln_f32`] (not
/// libm) so the wasm lowering — which inlines exactly that polynomial — produces
/// bit-identical draws (PLAN-PREGPU draw-stream parity).
///
/// The 24-bit grid caps a draw at `ln(2^24) / rate ≈ 16.6 / rate` (it was `33.3 / rate`): the
/// f32 uniform simply has no smaller positive value to invert. Tail mass beyond it is 6e-8.
#[inline]
pub fn fill_exp(key: Key, source: u32, lane0: u32, rate: f64, out: &mut [f32]) {
    debug_assert_eq!(lane0 & 1, 0, "fills start on even lanes (pair-shared draws)");
    let rate = rate as f32;
    let one_minus_ln = |b: u32| -crate::approx::ln_f32(1.0 - unit24(b)) / rate;
    for (t, pair) in out.chunks_mut(2).enumerate() {
        let lane = lane0.wrapping_add(2 * t as u32);
        let (b0, b1) = pair_bits(key, source, lane);
        pair[0] = one_minus_ln(b0);
        if let Some(x) = pair.get_mut(1) {
            *x = one_minus_ln(b1);
        }
    }
}

/// `Geometric(p)` (failures before first success) via `floor(ln(1 - u)/ln(1 - p))`, pair-shared;
/// `1 - u ∈ [2⁻²⁴, 1]` (see [`fill_exp`]). Shared-`approx` transcendentals for cross-backend bit
/// parity; the compile-time `denom` is the f64 `ln` rounded to f32 — the exact constant both
/// emitters bake in. Same 24-bit tail cap as [`fill_exp`].
#[inline]
pub fn fill_geometric(key: Key, source: u32, lane0: u32, p: f64, out: &mut [f32]) {
    debug_assert_eq!(lane0 & 1, 0, "fills start on even lanes (pair-shared draws)");
    let denom = (1.0 - p).ln() as f32;
    let geo = |b: u32| (crate::approx::ln_f32(1.0 - unit24(b)) / denom).floor();
    for (t, pair) in out.chunks_mut(2).enumerate() {
        let lane = lane0.wrapping_add(2 * t as u32);
        let (b0, b1) = pair_bits(key, source, lane);
        pair[0] = geo(b0);
        if let Some(x) = pair.get_mut(1) {
            *x = geo(b1);
        }
    }
}

/// The standard-normal Box–Muller pair for the lane pair containing `lane` — **one hash for both
/// lanes** (Track B): `u1` from the pair draw's low 24 bits, `u2` from its high 24; the even lane
/// takes the cos branch, the odd lane the sin branch.
///
/// `r = sqrt(-2·ln(1 - u1))`, not the f64 path's `ln(u1 + ½ulp)`: on the 24-bit grid `1 - u1` is
/// *exactly* representable in f32 and lies in `[2⁻²⁴, 1]`, so the domain guard is structural
/// rather than a nudge — and every backend computes the identical expression with no rounding
/// subtleties. (`u1 == 0` gives `ln(1) == 0` exactly, hence `r == 0` — a draw of `mu`, not a NaN.)
///
/// Shared-`approx` `ln`/`sin`/`cos` (not libm) for cross-backend bit parity; the trig kernel
/// computes both branches anyway, so a lowering that emits one lane per iteration just selects by
/// lane parity. `TAU·u2 < 2π` is always inside [`crate::approx::TRIG_MAX_F32`].
///
/// Pairing is why every fill must start on an EVEN lane: batch (1024) and reducer-chunk (16384)
/// boundaries all are, so a pair never straddles a range split — asserted in the fills.
#[inline]
pub fn normal_pair(key: Key, lane: u32, source: u32) -> (f32, f32) {
    use std::f32::consts::TAU;
    let (b1, b2) = pair_bits(key, source, lane);
    let r = (-2.0 * crate::approx::ln_f32(1.0 - unit24(b1))).sqrt();
    let theta = TAU * unit24(b2);
    (
        r * crate::approx::cos_f32(theta),
        r * crate::approx::sin_f32(theta),
    )
}

/// `N(mu, sigma^2)` via Box–Muller over lane pairs (see [`normal_pair`]): ONE hash, one `ln`, one
/// two-branch trig evaluation per TWO lanes.
#[inline]
pub fn fill_normal(key: Key, source: u32, lane0: u32, mu: f64, sigma: f64, out: &mut [f32]) {
    debug_assert_eq!(lane0 & 1, 0, "fills start on even lanes (pair-shared draws)");
    let (mu, sigma) = (mu as f32, sigma as f32);
    for (t, pair) in out.chunks_mut(2).enumerate() {
        let lane = lane0.wrapping_add(2 * t as u32);
        let (z0, z1) = normal_pair(key, lane, source);
        pair[0] = mu + sigma * z0;
        if let Some(x) = pair.get_mut(1) {
            *x = mu + sigma * z1;
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
///
/// The Knuth loop keeps its f64 `CellStream` uniforms (an `O(lambda)` product needs the headroom,
/// and this path never reaches codegen); only the *result* is an f32 lane value. Counts above 2²⁴
/// stop being exact integers in f32 — a documented Track B boundary, and one that only the
/// normal-approximation regime (`lambda > 500`) can reach.
#[inline]
pub fn fill_poisson(key: Key, source: u32, stream: u32, lane0: u32, lambda: f64, out: &mut [f32]) {
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
        *x = (k - 1) as f32;
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
    /// the wasm emitter and the WGSL emitter must reproduce them bit for
    /// bit. Reference: tools/rng-cert (C0).
    #[test]
    fn keyed_known_answer() {
        let key = Key::from_seed(0);
        assert_eq!(key.as_u64(), KAT_KEY0);
        assert_eq!(draw48(key, scalar_ctr(0, 0)), KAT_DRAWS[0]);
        assert_eq!(draw48(key, scalar_ctr(0, 1)), KAT_DRAWS[1]);
        assert_eq!(draw48(key, scalar_ctr(1, 0)), KAT_DRAWS[2]);
        assert_eq!(draw48(key, scalar_ctr(7, 12345)), KAT_DRAWS[3]);
    }

    /// The **f32 lane** contract (Track B): the pair-shared 24-bit uniforms and the Box–Muller
    /// pair, pinned to exact bit patterns. This is what the wasm emitter and the
    /// WGSL emitter must reproduce bit for bit — the pair split (even → low 24, odd → high 24) as
    /// much as the arithmetic.
    #[test]
    fn f32_lane_known_answer() {
        let key = Key::from_seed(0);
        // The pair draw is the SAME 48 bits as the f64 KAT above — split, not re-derived.
        let (b0, b1) = pair_bits(key, 0, 0);
        assert_eq!(b0, lo24(KAT_DRAWS[0]));
        assert_eq!(b1, hi24(KAT_DRAWS[0]));
        // Both members of a pair see the same draw (the odd lane does not re-hash).
        assert_eq!(pair_bits(key, 0, 1), (b0, b1));

        let mut col = [0.0f32; 4];
        fill_uniform(key, 0, 0, 0.0, 1.0, &mut col);
        assert_eq!(col.map(f32::to_bits), KAT_UNIF_F32);
        fill_normal(key, 2, 0, 0.0, 1.0, &mut col);
        assert_eq!(col.map(f32::to_bits), KAT_NORM_F32);
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
        let mut col = [0.0f32; 4];
        fill_uniform(key, 0, 0, 0.0, 1.0, &mut col);
        println!("KAT_UNIF_F32 = {:?}", col.map(|x| format!("0x{:08X}", x.to_bits())));
        fill_normal(key, 2, 0, 0.0, 1.0, &mut col);
        println!("KAT_NORM_F32 = {:?}", col.map(|x| format!("0x{:08X}", x.to_bits())));
    }

    const KAT_KEY0: u64 = 0x432A_B7DF_C618_529F;
    const KAT_DRAWS: [u64; 4] = [
        0xCBA7_C589_BB1D,
        0x1C74_645A_5F00,
        0x127B_1743_B147,
        0x552C_3662_C4EC,
    ];
    // The pair split is visible in these bits: lane 0's uniform is `0x3F09BB1D`, whose mantissa
    // ends in the low 24 bits of `KAT_DRAWS[0] = 0xCBA7C589BB1D`, and lane 1's is `0x3F4BA7C5`,
    // carrying its high 24 — the 24-bit draw scales into f32 exactly, with no rounding.
    const KAT_UNIF_F32: [u32; 4] = [0x3F09_BB1D, 0x3F4B_A7C5, 0x3EB4_BE00, 0x3DE3_A320];
    const KAT_NORM_F32: [u32; 4] = [0xBF25_F524, 0xBE36_10E6, 0x4036_2457, 0x3F7D_8F37];

    /// Counter keying means a fill is a pure function of the lane range: filling
    /// `[0, n)` in one call must equal filling any split of it — the property that makes
    /// reducer chunks trivially thread-count-invariant (no `chunk_seed` needed).
    #[test]
    fn keyed_fills_are_lane_range_pure() {
        let key = Key::from_seed(42);
        let n = 4096;
        let mut whole = vec![0.0f32; n];
        let mut split = vec![0.0f32; n];
        type Fill = fn(Key, u32, u32, f64, f64, &mut [f32]);
        let fills: [(Fill, f64, f64); 3] = [
            (fill_uniform, 2.0, 5.0),
            (fill_uniform_int, 1.0, 6.0),
            (fill_normal, 2.0, 3.0),
        ];
        for (fill, a, b) in fills {
            fill(key, 3, 0, a, b, &mut whole);
            // Even split point: fills start on even lanes (the pair-shared draw; every real range
            // start — batch, reducer chunk — is a multiple of 1024).
            let (lo, hi) = split.split_at_mut(1026);
            fill(key, 3, 0, a, b, lo);
            fill(key, 3, 1026, a, b, hi);
            assert_eq!(whole, split);
        }
        let mut whole_p = vec![0.0f32; n];
        let mut split_p = vec![0.0f32; n];
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
        // f32 lanes, f64 aggregation — the Track B split, exactly as `reduce` does it.
        let stats = |col: &[f32]| {
            let nf = col.len() as f64;
            let mean = col.iter().map(|&x| x as f64).sum::<f64>() / nf;
            let var = col.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / nf;
            (mean, var)
        };
        let mut col = vec![0.0f32; n];
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
        let mut col = vec![0.0f32; 6_000_000];
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
        let mut col = vec![0.0f32; 200_000];
        let lambda = 100_000.0;
        fill_poisson(key, 0, 0, 0, lambda, &mut col);
        let n = col.len() as f64;
        let mean = col.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = col.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!((mean / lambda - 1.0).abs() < 0.01, "mean = {mean}");
        assert!((var / lambda - 1.0).abs() < 0.05, "var = {var}");
        // The extreme case: just has to terminate, not hang. (Lane 0: the pair-shared draws need
        // every fill range to start on an even lane — every real caller is batch-aligned.)
        let mut one = [0.0f32; 8];
        fill_poisson(key, 1, 0, 0, 1e12, &mut one);
        assert!(one.iter().all(|&x| x.is_finite() && x >= 0.0));
    }

    /// Throughput of the keyed fills, single thread — the interpreter's RNG ceiling. Measured
    /// 2026-07-14 on M4 Pro after Track B (f32 lanes, pair-shared draws):
    ///
    /// | fill | f32 (Track B) | f64 (before) | note |
    /// |---|---|---|---|
    /// | `fill_uniform` | 1039 M/s | 540 M/s | **1.92×** — one hash per two lanes, as designed |
    /// | `fill_uniform_int` | 490 M/s | ~480 M/s | unchanged by design (still one 48-bit draw/lane) |
    /// | `fill_exp` | 260 M/s | — | one hash + two `ln` per pair |
    /// | `fill_normal` | 90 M/s | 86 M/s | only **1.04×**: Box–Muller is transcendental-LATENCY bound, so halving the hashing barely shows |
    ///
    /// The normal row is the honest one: `ln` + a two-branch trig is ~20 ns of dependent chain per
    /// pair and the hash was never the bottleneck there. Uniform-heavy graphs get the full 2×.
    /// Run with:
    /// `cargo test -p noise-core --release -- --ignored --nocapture bench_keyed_fills`
    #[test]
    #[ignore]
    fn bench_keyed_fills() {
        use std::time::Instant;
        let key = Key::from_seed(1);
        let n = 1 << 22; // 4M lanes
        let mut col = vec![0.0f32; n];
        let mut time = |label: &str, f: &mut dyn FnMut(&mut [f32])| {
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
        // Rotation-style consumption: d²=256 f32 normals per lane from one CellStream (the exact
        // Box–Muller fill `Inst::Rotation` runs — two u48 draws per pair, 24-bit uniforms).
        let d2 = 256;
        let lanes = n / d2;
        const SCALE24: f32 = 1.0 / (1u32 << 24) as f32;
        let mut fcol = vec![0.0f32; n];
        let t = Instant::now();
        for lane in 0..lanes as u32 {
            let mut s = CellStream::new(key, lane, 4);
            let col = &mut fcol[lane as usize * d2..(lane as usize + 1) * d2];
            let mut e = 0;
            while e < d2 {
                let hi0 = (s.next_u48() >> 24) as u32;
                let hi1 = (s.next_u48() >> 24) as u32;
                let r = (-2.0f32 * crate::approx::ln_f32((hi0 as f32 + 0.5) * SCALE24)).sqrt();
                let theta = std::f32::consts::TAU * (hi1 as f32 * SCALE24);
                col[e] = r * crate::approx::cos_f32(theta);
                if e + 1 < d2 {
                    col[e + 1] = r * crate::approx::sin_f32(theta);
                }
                e += 2;
            }
        }
        let el = t.elapsed().as_secs_f64();
        std::hint::black_box(&fcol);
        println!(
            "  {:<28}{:>8.1} M draws/s",
            "CellStream bulk normals",
            (lanes * d2) as f64 / el / 1e6
        );
    }

    /// Every 24-bit lane uniform lands in `[0, 1)` — and `1 - u`, which the `ln` paths are fed,
    /// lands in `[2⁻²⁴, 1]`, never 0 (the structural `ln(0)` guard the f32 Box–Muller relies on).
    #[test]
    fn lane_uniforms_stay_in_the_unit_interval() {
        let key = Key::from_seed(42);
        let mut col = vec![0.0f32; 10_000];
        fill_uniform(key, 0, 0, 0.0, 1.0, &mut col);
        for &u in &col {
            assert!((0.0..1.0).contains(&u), "uniform out of range: {u}");
            assert!(1.0 - u >= (1.0f32 / (1u32 << 24) as f32), "1 - u underflowed to 0");
        }
    }
}
