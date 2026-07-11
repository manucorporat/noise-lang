//! Hand-rolled, seedable PRNG for the random-variable hot loop (PLAN.md "RNG").
//!
//! xoshiro256++ seeded via SplitMix64. No `getrandom`, no `std::time`, no OS entropy,
//! no threads — so `noise-core` stays WASM-clean and sampling is fully deterministic for
//! a given seed. The hot operation is `fill_uniform`, which writes a whole column in one
//! tight (vectorizable) loop.

/// xoshiro256++ state. Seed it with [`Rng::seed_from_u64`].
pub struct Rng {
    s: [u64; 4],
}

/// Largest Poisson `lambda` sampled by the exact Knuth loop; above this `fill_poisson` uses the
/// normal approximation (see [`Rng::fill_poisson`]). Chosen below the `(-lambda).exp()` underflow
/// point (`lambda ≈ 745`) so the Knuth path is always exact, and low enough that its `O(lambda)`
/// cost per draw stays a few hundred iterations. The Gaussian approximation is excellent well
/// before here (the Poisson is already near-Gaussian by `lambda ≈ 20`).
pub const POISSON_KNUTH_MAX: f64 = 500.0;

impl Rng {
    /// Expand a `u64` seed into the 256-bit state via SplitMix64. SplitMix64 never emits an
    /// all-zero run for distinct inputs, so the xoshiro state is non-zero in practice.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        let s = [next(), next(), next(), next()];
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)` using the top 53 bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Fill a whole column with `lo + (hi - lo) * u01`. One tight loop (vectorizable).
    #[inline]
    pub fn fill_uniform(&mut self, lo: f64, hi: f64, out: &mut [f64]) {
        let span = hi - lo;
        for x in out.iter_mut() {
            *x = lo + span * self.next_f64();
        }
    }

    /// Fill a column with integers uniform over `lo..=hi` (inclusive), as `f64`, via Lemire's
    /// multiply-high: the top 64 bits of `next_u64() * count` are uniform in `0..count` (bias ≤
    /// `count / 2^64`, negligible) and — unlike `floor(u01 * count)` — need no `u64→f64` convert or
    /// `floor`. `count >= 1`, so `count == 1` always yields `k == 0` (a point mass at `lo`).
    #[inline]
    pub fn fill_uniform_int(&mut self, lo: f64, hi: f64, out: &mut [f64]) {
        let count = (hi - lo + 1.0).max(1.0) as u64;
        for x in out.iter_mut() {
            let k = ((self.next_u64() as u128 * count as u128) >> 64) as u64;
            *x = lo + k as f64;
        }
    }

    /// Fill a column with `Exp(rate)` draws via inverse-CDF: `-ln(u)/rate` for `u ∈ (0, 1]`
    /// (taken as `1 - next_f64` so `ln(u)` is finite). `rate > 0` is checked at construction.
    #[inline]
    pub fn fill_exp(&mut self, rate: f64, out: &mut [f64]) {
        for x in out.iter_mut() {
            *x = -(1.0 - self.next_f64()).ln() / rate;
        }
    }

    /// Fill a column with `Poisson(lambda)` counts. For small `lambda` this is Knuth's algorithm
    /// (multiply uniforms until the running product drops below `e^-lambda`), which is `O(lambda)`
    /// per draw and exact. That algorithm is a **hang** and is **silently wrong** for large lambda:
    /// past `lambda ≈ 745` `(-lambda).exp()` underflows to `0`, so the loop can only stop when the
    /// running product itself underflows — many iterations, and biased low. So above
    /// [`POISSON_KNUTH_MAX`] we switch to the standard **normal approximation**
    /// `round(max(0, N(lambda, sqrt(lambda))))` (the CLT / de Moivre–Laplace limit): `O(1)` per
    /// draw, deterministic, mean and variance ≈ `lambda`. `poisson(1e12)` therefore returns
    /// promptly instead of hanging (finding A8). The threshold is well inside the Knuth-exact
    /// regime, so every existing (small-`lambda`) result is bit-identical.
    #[inline]
    pub fn fill_poisson(&mut self, lambda: f64, out: &mut [f64]) {
        if lambda > POISSON_KNUTH_MAX {
            // Gaussian approximation: fill with N(lambda, lambda) then snap to non-negative integers.
            self.fill_normal(lambda, lambda.sqrt(), out);
            for x in out.iter_mut() {
                *x = x.round().max(0.0);
            }
            return;
        }
        let l = (-lambda).exp();
        for x in out.iter_mut() {
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_f64();
                if p <= l {
                    break;
                }
            }
            *x = (k - 1) as f64;
        }
    }

    /// Fill a column with `Geometric(p)` draws — the number of failures before the first success
    /// (support `0, 1, 2, …`), via inverse-CDF `floor(ln(u)/ln(1-p))` for `u ∈ (0, 1]`. `p == 1`
    /// degenerates to all-zeros (`ln(0) = -inf` makes the quotient `0`). `0 < p <= 1` is checked
    /// at construction.
    #[inline]
    pub fn fill_geometric(&mut self, p: f64, out: &mut [f64]) {
        let denom = (1.0 - p).ln();
        for x in out.iter_mut() {
            let u = 1.0 - self.next_f64(); // (0, 1]
            *x = (u.ln() / denom).floor();
        }
    }

    /// Fill a column with `N(mu, sigma^2)` draws via Box–Muller, two normals per uniform pair.
    /// `u1 ∈ (0, 1]` (taken as `1 - next_f64`) keeps `ln(u1)` finite.
    #[inline]
    pub fn fill_normal(&mut self, mu: f64, sigma: f64, out: &mut [f64]) {
        use std::f64::consts::TAU;
        let mut i = 0;
        while i < out.len() {
            let u1 = 1.0 - self.next_f64(); // (0, 1]
            let u2 = self.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let (s, c) = (TAU * u2).sin_cos();
            out[i] = mu + sigma * (r * c);
            i += 1;
            if i < out.len() {
                out[i] = mu + sigma * (r * s);
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer test guarding the xoshiro256++ / SplitMix64 constants. If the algorithm
    /// or constants change, this fixed sequence changes — the test catches it.
    #[test]
    fn known_answer_sequence() {
        let mut rng = Rng::seed_from_u64(0);
        // First five outputs from seed 0; committed reference for this implementation.
        assert_eq!(rng.next_u64(), 5_987_356_902_031_041_503);
        assert_eq!(rng.next_u64(), 7_051_070_477_665_621_255);
        assert_eq!(rng.next_u64(), 6_633_766_593_972_829_180);
        assert_eq!(rng.next_u64(), 211_316_841_551_650_330);
        assert_eq!(rng.next_u64(), 9_136_120_204_379_184_874);
    }

    #[test]
    fn fill_uniform_int_is_uniform_over_range() {
        // Lemire multiply-high must hit every face of `unif_int(1,6)` with ~equal frequency and
        // never land outside `1..=6` — the properties a biased/off-by-one mapping would break.
        let mut rng = Rng::seed_from_u64(99);
        let mut col = vec![0.0f64; 6_000_000];
        rng.fill_uniform_int(1.0, 6.0, &mut col);
        let mut counts = [0u64; 7]; // index 1..=6
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

    /// Hypothesis test for the *real* RNG lever: the kernel is bound by xoshiro's serial state
    /// chain (each `next_u64` depends on the last), so the only way faster is to run **independent
    /// streams** whose chains the OoO core can overlap. This races 1/2/4/8 independent xoshiro
    /// states doing the dice-sum work. If throughput scales with stream count, multi-stream
    /// unrolling in the kernel is the win. Ignored; run with:
    /// `cargo test -p noise-core --release -- --ignored --nocapture bench_rng_multistream`
    #[test]
    #[ignore]
    fn bench_rng_multistream() {
        use std::time::Instant;
        let total = 48_000_000usize;
        let count = 6u128;
        let roll = |x: u64| ((x as u128 * count) >> 64) as u64; // Lemire dice in [0,6)

        macro_rules! run {
            ($k:literal, $rngs:expr) => {{
                let mut rngs = $rngs;
                for r in rngs.iter_mut() {
                    for _ in 0..1000 {
                        r.next_u64();
                    }
                }
                let iters = total / $k;
                let t = Instant::now();
                let mut acc = 0u64;
                for _ in 0..iters {
                    // Fixed-size array indexed by a const-bound loop → LLVM keeps the $k states in
                    // registers as independent chains (no aliasing through a slice).
                    for i in 0..$k {
                        let a = roll(rngs[i].next_u64());
                        let b = roll(rngs[i].next_u64());
                        acc = acc.wrapping_add(a + b + 2);
                    }
                }
                let mps = (iters * $k) as f64 / t.elapsed().as_secs_f64() / 1e6;
                std::hint::black_box(acc);
                println!("    streams={:2}  {mps:7.0} M dice-samples/s", $k);
            }};
        }

        println!("\n  dice-sum throughput by independent RNG stream count (single thread):");
        run!(1, [Rng::seed_from_u64(1)]);
        run!(2, [Rng::seed_from_u64(1), Rng::seed_from_u64(2)]);
        run!(
            4,
            [
                Rng::seed_from_u64(1),
                Rng::seed_from_u64(2),
                Rng::seed_from_u64(3),
                Rng::seed_from_u64(4)
            ]
        );
        run!(
            8,
            [
                Rng::seed_from_u64(1),
                Rng::seed_from_u64(2),
                Rng::seed_from_u64(3),
                Rng::seed_from_u64(4),
                Rng::seed_from_u64(5),
                Rng::seed_from_u64(6),
                Rng::seed_from_u64(7),
                Rng::seed_from_u64(8)
            ]
        );
    }

    #[test]
    fn poisson_large_lambda_is_fast_and_has_the_right_mean() {
        // Above POISSON_KNUTH_MAX the Knuth loop would hang (and be biased low) — the normal
        // approximation must return promptly with mean ≈ lambda and variance ≈ lambda. A huge
        // lambda (`1e12`, the finding's repro) must not hang: this test *completing* is the proof.
        let mut rng = Rng::seed_from_u64(3);
        let mut col = vec![0.0f64; 200_000];
        let lambda = 100_000.0;
        rng.fill_poisson(lambda, &mut col);
        let n = col.len() as f64;
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((mean / lambda - 1.0).abs() < 0.01, "mean = {mean}");
        assert!((var / lambda - 1.0).abs() < 0.05, "var = {var}");
        // The extreme case from the finding: just has to terminate, not hang.
        let mut one = [0.0f64; 8];
        rng.fill_poisson(1e12, &mut one);
        assert!(one.iter().all(|&x| x.is_finite() && x >= 0.0));
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut rng = Rng::seed_from_u64(42);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn fill_normal_matches_requested_moments() {
        let mut rng = Rng::seed_from_u64(7);
        let mut col = vec![0.0f64; 200_000];
        rng.fill_normal(2.0, 3.0, &mut col); // N(2, 9)
        let n = col.len() as f64;
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((mean - 2.0).abs() < 0.03, "mean = {mean}");
        assert!((var - 9.0).abs() < 0.15, "var = {var}");
    }
}
