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

    /// Fill a column with integers uniform over `lo..=hi` (inclusive), as `f64`. `u01 < 1`
    /// so `floor(u01 * n)` is in `0..n`, giving exactly `lo..=hi`.
    #[inline]
    pub fn fill_uniform_int(&mut self, lo: f64, hi: f64, out: &mut [f64]) {
        let n = (hi - lo + 1.0).max(1.0);
        for x in out.iter_mut() {
            *x = lo + (self.next_f64() * n).floor();
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

    /// Fill a column with `Poisson(lambda)` counts via Knuth's algorithm (multiply uniforms
    /// until the running product drops below `e^-lambda`). O(lambda) per draw — fine for the
    /// teaching-language scale; large `lambda` is slow but correct.
    #[inline]
    pub fn fill_poisson(&mut self, lambda: f64, out: &mut [f64]) {
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
