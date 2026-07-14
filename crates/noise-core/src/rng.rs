//! Counter-based, keyed RNG for the random-variable hot loop (PLAN-PREGPU Track C).
//!
//! No `getrandom`, no `std::time`, no OS entropy, no threads — so `noise-core` stays
//! WASM-clean and sampling is fully deterministic for a given seed. The hot operations
//! are the keyed `fill_*` functions, each writing a whole column in one tight
//! (vectorizable) loop of independent per-lane hashes.

// ---------------------------------------------------------------------------
// Counter-based generator (PLAN-PREGPU Track C): pcg4d-3r, keyed per draw cell.
//
// One hash per `(lane, source)` cell — no state chain, so any lane range can be
// computed independently (reducer chunks are just ranges, thread count can't matter)
// and every backend (interpreter, JIT, wasm, later WGSL) produces bit-identical draws
// from pure u32 arithmetic. pcg4d-3r is pcg4d (Jarzynski–Olano, JCGT 2020) with one
// extra xorshift + product round: C0 (tools/rng-cert) showed published pcg4d has
// deterministic input→output bit relations in the consumed region; the third round
// clears every consumed bit. Certification evidence lives in tools/rng-cert/RESULTS.md.
//
// Consumption contract (C0): only bits 8..31 of a word are consumable (pcg-family low
// bits are structurally weak). The f64 uniform takes 24+24 bits from words 0+1;
// `normal` takes u1 from words 0+1 and u2 from words 2+3, cos branch only (one normal
// per lane, so lane-range chunking never straddles a Box–Muller pair). Fills that need
// more than one hash per lane chain through [`CellStream`] (see its doc).
// ---------------------------------------------------------------------------

/// Largest Poisson `lambda` sampled by the exact Knuth loop; above this `fill_poisson` uses the
/// normal approximation (see [`fill_poisson`]). Chosen below the `(-lambda).exp()` underflow
/// point (`lambda ≈ 745`) so the Knuth path is always exact, and low enough that its `O(lambda)`
/// cost per draw stays a few hundred iterations. The Gaussian approximation is excellent well
/// before here (the Poisson is already near-Gaussian by `lambda ≈ 20`).
pub const POISSON_KNUTH_MAX: f64 = 500.0;

/// The pcg4d-3r hash: 4×u32 → 4×u32, pure mul/add/xor/shift.
#[inline(always)]
pub fn pcg4d_3r(mut v: [u32; 4]) -> [u32; 4] {
    for x in v.iter_mut() {
        *x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    v
}

/// Per-run key: SplitMix64 of the user seed, split into the hash's two key words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub k0: u32,
    pub k1: u32,
}

impl Key {
    pub fn from_seed(seed: u64) -> Self {
        let mut z = seed;
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Key { k0: z as u32, k1: (z >> 32) as u32 }
    }
}

/// Hash for one draw cell.
#[inline(always)]
pub fn cell(key: Key, lane: u32, source: u32) -> [u32; 4] {
    pcg4d_3r([key.k0, key.k1, lane, source])
}

/// Unbounded per-cell uniform supply for fills that consume a variable or per-lane-large
/// number of draws (Knuth `poisson`, `Permutation`'s Fisher–Yates, `Rotation`'s Gaussian
/// seed). The base cell yields one 48-bit uniform (words 0+1); its words 2+3 become the
/// *chain key*, and iteration hash `j ≥ 1` is `pcg4d_3r(chain_key, j, source)` yielding
/// two more uniforms — so the chain never reuses consumed words as key material, `j` is a
/// full u32 (no iteration cap), and distinct sources can't collide (chain keys are
/// hash-random, unlike any `source + f(j)` scheme, where some `j` would alias another
/// source's offset).
pub struct CellStream {
    ck0: u32,
    ck1: u32,
    source: u32,
    /// Next iteration index to hash (0 = the base hash, already consumed into `pending`).
    j: u32,
    /// Word pairs not yet consumed, most imminent last.
    pending: [(u32, u32); 2],
    left: u8,
    /// The sin branch of the last Box–Muller evaluation, pending consumption (see
    /// [`CellStream::next_normal`]).
    pending_normal: Option<f64>,
}

impl CellStream {
    #[inline]
    pub fn new(key: Key, lane: u32, source: u32) -> Self {
        let w = cell(key, lane, source);
        CellStream {
            ck0: w[2],
            ck1: w[3],
            source,
            j: 1,
            pending: [(w[0], w[1]), (w[0], w[1])],
            left: 1,
            pending_normal: None,
        }
    }

    #[inline]
    fn next_pair(&mut self) -> (u32, u32) {
        if self.left == 0 {
            let w = pcg4d_3r([self.ck0, self.ck1, self.j, self.source]);
            self.j = self.j.wrapping_add(1);
            self.pending = [(w[2], w[3]), (w[0], w[1])];
            self.left = 2;
        }
        self.left -= 1;
        self.pending[self.left as usize]
    }

    /// The consumed 48 bits of the next word pair.
    #[inline]
    pub fn next_u48(&mut self) -> u64 {
        let (a, b) = self.next_pair();
        ((a >> 8) as u64) << 24 | (b >> 8) as u64
    }

    /// Uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn next_unit(&mut self) -> f64 {
        self.next_u48() as f64 * (1.0 / (1u64 << 48) as f64)
    }

    /// Integer uniform in `0..count` via Lemire multiply-high on 48 bits.
    #[inline]
    pub fn next_bounded(&mut self, count: u64) -> u64 {
        ((self.next_u48() as u128 * count as u128) >> 48) as u64
    }

    /// Standard normal via Box–Muller, cos branch only (two uniforms per normal).
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_u48() as f64 + 0.5) * (1.0 / (1u64 << 48) as f64);
        let u2 = self.next_unit();
        (-2.0 * crate::approx::ln(u1)).sqrt() * crate::approx::cos(std::f64::consts::TAU * u2)
    }
}

/// The consumed 48 bits of a word pair as a uniform `f64` in `[0, 1)`.
#[inline(always)]
fn unit_f64(w0: u32, w1: u32) -> f64 {
    let bits = ((w0 >> 8) as u64) << 24 | (w1 >> 8) as u64;
    bits as f64 * (1.0 / (1u64 << 48) as f64)
}

/// Uniform `f64` in `[0, 1)` for one cell (words 0+1).
#[inline(always)]
pub fn unit_uniform(key: Key, lane: u32, source: u32) -> f64 {
    let w = cell(key, lane, source);
    unit_f64(w[0], w[1])
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
        let w = cell(key, lane0.wrapping_add(i as u32), source);
        let bits = ((w[0] >> 8) as u64) << 24 | (w[1] >> 8) as u64;
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

/// The Box–Muller pair for the even/odd lane pair `(lane & !1, lane | 1)`: one hash of the
/// even lane, `u1` from words 0+1 (offset half an ulp of the 48-bit grid to dodge `ln(0)`),
/// `u2` from words 2+3; the even lane takes the cos branch, the odd lane the sin branch.
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
    let w = cell(key, even_lane, source);
    let bits1 = ((w[0] >> 8) as u64) << 24 | (w[1] >> 8) as u64;
    let u1 = (bits1 as f64 + 0.5) * (1.0 / (1u64 << 48) as f64); // (0, 1)
    let u2 = unit_f64(w[2], w[3]);
    let r = (-2.0 * crate::approx::ln(u1)).sqrt();
    let theta = TAU * u2;
    (r * crate::approx::cos(theta), r * crate::approx::sin(theta))
}

/// `N(mu, sigma^2)` via Box–Muller over lane pairs (see [`normal_pair`]): one hash, one
/// `ln`, one two-branch trig evaluation per TWO lanes — the pair sharing xoshiro had, kept
/// under counter keying by pinning the pair to the even lane.
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
#[inline]
pub fn fill_poisson(key: Key, source: u32, lane0: u32, lambda: f64, out: &mut [f64]) {
    if lambda > POISSON_KNUTH_MAX {
        fill_normal(key, source, lane0, lambda, lambda.sqrt(), out);
        for x in out.iter_mut() {
            *x = x.round().max(0.0);
        }
        return;
    }
    let l = (-lambda).exp();
    for (i, x) in out.iter_mut().enumerate() {
        let mut s = CellStream::new(key, lane0.wrapping_add(i as u32), source);
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

    /// Known-answer test guarding the pcg4d-3r constants, the SplitMix64 key derivation,
    /// and the 48-bit consumption contract. These vectors are the cross-backend draw
    /// contract: the JIT, the wasm emitter, and later the WGSL emitter must reproduce
    /// them bit for bit. Reference: tools/rng-cert (C0).
    #[test]
    fn keyed_known_answer() {
        let key = Key::from_seed(0);
        assert_eq!((key.k0, key.k1), (0x7B1D_CDAF, 0xE220_A839));
        assert_eq!(cell(key, 0, 0), [0xF2C1_0064, 0xBCA7_79AB, 0x14D0_E6F6, 0xDEC6_9A06]);
        assert_eq!(cell(key, 1, 0), [0xCF84_DD88, 0x0F61_ED1E, 0x5970_7504, 0xDD3D_CF07]);
        assert_eq!(cell(key, 0, 1), [0xC5F1_A6F8, 0xABB4_C7DA, 0x9D21_DAFD, 0x0871_76FF]);
        assert_eq!(
            cell(key, 12345, 7),
            [0x6E0D_7242, 0x12B5_1AC5, 0xBB56_8E12, 0x0FE0_EA4A]
        );
        // Uniform KATs on exact bit patterns (bits, not decimal literals, are the contract):
        // ≈0.948257, 0.810621, 0.773219, 0.429893.
        assert_eq!(unit_uniform(key, 0, 0).to_bits(), 0x3FEE_5820_1794_EF20u64);
        assert_eq!(unit_uniform(key, 1, 0).to_bits(), 0x3FE9_F09B_A1EC_3DA0u64);
        assert_eq!(unit_uniform(key, 0, 1).to_bits(), 0x3FE8_BE34_D576_98E0u64);
        assert_eq!(unit_uniform(key, 12345, 7).to_bits(), 0x3FDB_835C_84AD_4680u64);
    }

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
        fill_poisson(key, 3, 0, 4.5, &mut whole_p);
        let (lo, hi) = split_p.split_at_mut(1026);
        fill_poisson(key, 3, 0, 4.5, lo);
        fill_poisson(key, 3, 1026, 4.5, hi);
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
        fill_poisson(key, 3, 0, 4.5, &mut col); // mean = var = lambda
        let (mean, var) = stats(&col);
        assert!((mean - 4.5).abs() < 0.05, "poisson mean = {mean}");
        assert!((var - 4.5).abs() < 0.15, "poisson var = {var}");
        fill_geometric(key, 4, 0, 0.25, &mut col); // mean (1-p)/p = 3
        let (mean, _) = stats(&col);
        assert!((mean - 3.0).abs() < 0.05, "geometric mean = {mean}");
        fill_poisson(key, 5, 0, 100_000.0, &mut col); // normal-approx regime
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
        fill_poisson(key, 0, 0, lambda, &mut col);
        let n = col.len() as f64;
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((mean / lambda - 1.0).abs() < 0.01, "mean = {mean}");
        assert!((var / lambda - 1.0).abs() < 0.05, "var = {var}");
        // The extreme case: just has to terminate, not hang.
        let mut one = [0.0f64; 8];
        fill_poisson(key, 1, 0, 1e12, &mut one);
        assert!(one.iter().all(|&x| x.is_finite() && x >= 0.0));
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
