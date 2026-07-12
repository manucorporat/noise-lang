//! Inlinable polynomial approximations of `ln`/`sin`/`cos` — the reference spec the code generators
//! transcribe (PLAN.md Phase 4, "inline the transcendentals").
//!
//! The Box–Muller normal draw (and `exp`/`geometric`) spend their time in `ln`/`cos`, and the signal
//! examples in `sin`/`cos`. In the kernels those were `extern "C"`/host **calls**, which (a) cost a
//! call per draw and (b) — the bigger loss — broke the multi-stream latency-hiding, so the
//! profitability gate kept every transcendental-bound graph single-stream (see [`crate::kernel`]).
//!
//! Replacing the calls with straight-line arithmetic fixes both. Monte-Carlo sampling error
//! (~1/√N) dwarfs the ~1e-10 approximation error here, so full `libm` precision is wasted; these
//! are tuned for "indistinguishable in distribution from the interpreter oracle", which the JIT/WASM
//! parity tests check. The interpreter itself keeps using `libm` — it stays the exact oracle.
//!
//! The coefficient arrays below are the **single source of truth**: the Rust reference functions and
//! both emitters (`jit`, `wasm_emit`) evaluate the *same* numbers in the *same* Horner order, so the
//! emitted code agrees op-for-op with the reference. Constants are the standard fdlibm kernel
//! coefficients.

// Reference polynomials/constants transcribed by the JIT (`--features jit`) and WASM backends.
// Which items are live depends on the build config, so dead-code analysis is unreliable here
// (this module was previously `pub`, which masked the same warnings).
#![allow(dead_code)]

use std::f64::consts::{FRAC_2_PI, LN_2, SQRT_2};

/// `ln(m)` series: `ln(m) = 2·f·Σ cₖ·f²ᵏ` with `f = (m-1)/(m+1)` — i.e. the atanh expansion
/// `2(f + f³/3 + f⁵/5 + …)`. Low→high powers of `f²`.
pub(crate) const LN_COEFFS: [f64; 6] =
    [1.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0, 1.0 / 9.0, 1.0 / 11.0];

/// `sin(r)/r` tail on `[-π/4, π/4]`: `sin(r) = r + r·z·Σ Sₖ·zᵏ`, `z = r²` (fdlibm `__kernel_sin`).
pub(crate) const SIN_COEFFS: [f64; 6] = [
    -1.666_666_666_666_663_2e-1,
    8.333_333_333_322_49e-3,
    -1.984_126_982_985_795e-4,
    2.755_731_370_707_007e-6,
    -2.505_076_025_340_686_4e-8,
    1.589_690_995_211_55e-10,
];

/// `cos(r)` tail on `[-π/4, π/4]`: `cos(r) = 1 - z/2 + z²·Σ Cₖ·zᵏ`, `z = r²` (fdlibm `__kernel_cos`).
pub(crate) const COS_COEFFS: [f64; 6] = [
    4.166_666_666_666_602e-2,
    -1.388_888_888_887_411e-3,
    2.480_158_728_947_673e-5,
    -2.755_731_435_139_066_3e-7,
    2.087_572_321_298_175e-9,
    -1.135_964_755_778_819_5e-11,
];

/// π/2 split into a high part (low 33 bits zero) plus a tail, so `x - k·(π/2)` is computed in two
/// exact Cody–Waite steps and keeps full precision in the reduced argument for the `k` we see.
pub(crate) const PIO2_HI: f64 = 1.570_796_326_734_125_6;
pub(crate) const PIO2_LO: f64 = 6.077_100_506_506_192e-11;

/// Argument magnitude beyond which the **2-term** Cody–Waite reduction above loses accuracy (the
/// reduced-argument error grows with `k ≈ x·2/π`): measured ~1e-16 up to here, ~1e-9 at 1e7, ~1e-6
/// at 1e10, and nonsense by 1e15. So `|x| >= TRIG_MAX` is out of the polynomial's usable domain —
/// the reference [`sin`]/[`cos`] and both emitters fall back to the accurate library `sin`/`cos`
/// there (finding C3), keeping backend trig in agreement with the interpreter's libm across the
/// whole range a user can steer an RV into (e.g. `sin(1e12 * X)`). `2^20` keeps the fast inline
/// path for every ordinary argument (Box–Muller's `[0, 2π)`, signal multiples of π) while capping
/// the reduced-argument error at ~1e-16.
pub(crate) const TRIG_MAX: f64 = (1u64 << 20) as f64;

/// Scale factor lifting any subnormal into the normal range (smallest subnormal `2^-1074 · 2^54 =
/// 2^-1020` is normal), and its `ln` correction. Used to keep [`ln`] valid on subnormal inputs,
/// whose exponent field is zero and would otherwise corrupt the mantissa bit-surgery (finding C9).
pub(crate) const LN_SUBNORMAL_SCALE: f64 = (1u64 << 54) as f64;
pub(crate) const LN_SUBNORMAL_CORR: f64 = 54.0 * LN_2;

/// Horner evaluation of `Σ c[i]·z^i` (coeffs low→high) — the exact reduction both emitters mirror.
fn horner(z: f64, c: &[f64]) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * z + ci)
}

/// `ln(x)` for `x > 0`. Decompose `x = m·2^e` with `m ∈ [1,2)` by bit-surgery on the IEEE-754
/// fields, recenter `m` to `[1/√2, √2]` (one branchless halving) so the series argument is small,
/// then `ln(m) = 2·f·Σ cₖf²ᵏ` with `f = (m-1)/(m+1)`, and add `e·ln2`.
pub fn ln(x: f64) -> f64 {
    // Subnormal inputs have a zero exponent field, so the bit-surgery below would read a wrong
    // mantissa; scale them into the normal range and correct by `54·ln2` afterward (finding C9).
    // `x > 0` is guaranteed by the callers' domain guards / the draw kernels' `1 - u ∈ (0, 1]`.
    // (The `x > 0` term also stops `x == 0` from recursing forever.)
    if x > 0.0 && x < f64::MIN_POSITIVE {
        return ln(x * LN_SUBNORMAL_SCALE) - LN_SUBNORMAL_CORR;
    }
    let bits = x.to_bits();
    let e0 = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let m_bits = (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
    let m0 = f64::from_bits(m_bits); // [1, 2)

    // Recenter: if m > √2, use m/2 and bump the exponent — keeps |f| ≤ 0.172 so the series is tight.
    let big = m0 > SQRT_2;
    let m = if big { m0 * 0.5 } else { m0 };
    let e = if big { e0 + 1 } else { e0 };

    let f = (m - 1.0) / (m + 1.0);
    let f2 = f * f;
    2.0 * f * horner(f2, &LN_COEFFS) + (e as f64) * LN_2
}

/// Reduce `x` to `r ∈ [-π/4, π/4]` with quadrant `k` (so `x ≈ k·π/2 + r`), via round-to-nearest
/// and a two-part subtraction of π/2. Uses **round-ties-to-even** (`round_ties_even`) — the same
/// tie rule the emitters' `nearest` / `f64.nearest` instructions apply — rather than `f64::round`'s
/// round-half-away-from-zero, so the reference and the emitted kernels agree op-for-op at ties too
/// (finding C9).
fn reduce(x: f64) -> (f64, i64) {
    let k = (x * FRAC_2_PI).round_ties_even();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    (r, k as i64)
}

fn sin_kernel(r: f64) -> f64 {
    let z = r * r;
    r + r * z * horner(z, &SIN_COEFFS)
}

fn cos_kernel(r: f64) -> f64 {
    let z = r * r;
    1.0 - 0.5 * z + z * z * horner(z, &COS_COEFFS)
}

/// Select the right reduced-argument kernel and sign for quadrant `kq = k mod 4`. `cos` and `sin`
/// differ only by a one-quadrant phase shift, so they share this table.
fn quadrant(kq: i64, sin_r: f64, cos_r: f64, is_cos: bool) -> f64 {
    // For cos: q0=cos, q1=-sin, q2=-cos, q3=sin. For sin: q0=sin, q1=cos, q2=-sin, q3=-cos.
    let (q0, q1, q2, q3) = if is_cos {
        (cos_r, -sin_r, -cos_r, sin_r)
    } else {
        (sin_r, cos_r, -sin_r, -cos_r)
    };
    let mut res = q0;
    if kq == 1 {
        res = q1;
    }
    if kq == 2 {
        res = q2;
    }
    if kq == 3 {
        res = q3;
    }
    res
}

/// `cos(x)`. Range-reduced polynomial for `|x| < TRIG_MAX` (accurate to ~1e-16 over every ordinary
/// argument — Box–Muller's `[0, 2π)`, signal multiples of π); beyond that the 2-term reduction
/// degrades, so it defers to the library `cos` (finding C3), which is what the interpreter uses.
/// Both emitters transcribe this exact `|x| < TRIG_MAX ? poly : call` shape.
pub fn cos(x: f64) -> f64 {
    if x.abs() >= TRIG_MAX {
        return x.cos();
    }
    let (r, k) = reduce(x);
    quadrant(k & 3, sin_kernel(r), cos_kernel(r), true)
}

/// `sin(x)` (see [`cos`] — same range guard and fallback).
pub fn sin(x: f64) -> f64 {
    if x.abs() >= TRIG_MAX {
        return x.sin();
    }
    let (r, k) = reduce(x);
    quadrant(k & 3, sin_kernel(r), cos_kernel(r), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Max abs error of the approximations vs `libm` over the ranges the kernels actually exercise.
    /// The bar is ~1e-9 — far tighter than Monte-Carlo noise, so the emitted kernels are
    /// distribution-indistinguishable from the interpreter oracle.
    #[test]
    fn approximations_track_libm() {
        // ln over (0, 1] (Box–Muller / exp / geometric feed it `1 - u ∈ (0, 1]`) and up past 1.
        let mut max_ln = 0.0f64;
        for i in 1..=200_000 {
            let x = i as f64 / 100_000.0; // 1e-5 .. 2.0
            max_ln = max_ln.max((ln(x) - x.ln()).abs());
        }
        assert!(max_ln < 1e-9, "ln max abs err {max_ln:e}");

        // sin/cos over several periods (covers Box–Muller's [0,2π) and signal multiples of π).
        let (mut max_sin, mut max_cos) = (0.0f64, 0.0f64);
        for i in 0..400_000 {
            let x = (i as f64 - 200_000.0) / 2_000.0; // -100 .. 100
            max_sin = max_sin.max((sin(x) - x.sin()).abs());
            max_cos = max_cos.max((cos(x) - x.cos()).abs());
        }
        assert!(max_sin < 1e-9, "sin max abs err {max_sin:e}");
        assert!(max_cos < 1e-9, "cos max abs err {max_cos:e}");
    }

    /// `ln` accuracy over the **whole uniform-reachable range**, not just `[1e-5, 2]`: Box–Muller
    /// feeds `1 - next_f64()`, whose smallest value is `2^-53 ≈ 1.1e-16`, and `math::log` can be
    /// steered anywhere positive — down into the subnormals (finding C9). The reference must track
    /// libm as a *relative* error across all of it (absolute error grows with `|ln x|`).
    #[test]
    fn ln_tracks_libm_across_the_reachable_range() {
        let mut max_rel = 0.0f64;
        // Geometric sweep from the smallest subnormal up past 1e300, plus the Box–Muller floor.
        let mut x = f64::MIN_POSITIVE * 0.5; // a subnormal
        for _ in 0..2000 {
            let want = x.ln();
            let rel = (ln(x) - want).abs() / want.abs().max(1.0);
            max_rel = max_rel.max(rel);
            x *= 1.7;
        }
        // Exact Box–Muller floor value.
        let floor = 2f64.powi(-53);
        let rel = (ln(floor) - floor.ln()).abs() / floor.ln().abs();
        max_rel = max_rel.max(rel);
        // The atanh series is ~1e-9 absolute (denominator floored at 1), so ~1e-9 relative across
        // the whole range — far below Monte-Carlo noise. A wrong subnormal answer would be O(1).
        assert!(max_rel < 1e-8, "ln relative err {max_rel:e}");
    }

    /// Trig over **huge** arguments — where the naive 2-term Cody–Waite reduction is nonsense
    /// (~1e-1 at 1e15) — must still agree with libm because of the `|x| >= TRIG_MAX` fallback
    /// (finding C3). This is the const-graph divergence `sin(1e12 * X)` would otherwise expose.
    #[test]
    fn trig_agrees_with_libm_past_the_reduction_limit() {
        for &x in &[
            1e7,
            1e8,
            1e10,
            1e12,
            1e15,
            1e18,
            -1e12,
            -1e15,
            TRIG_MAX,
            TRIG_MAX * 4.0,
        ] {
            assert!(
                (sin(x) - x.sin()).abs() < 1e-9,
                "sin({x:e}) = {} vs libm {}",
                sin(x),
                x.sin()
            );
            assert!(
                (cos(x) - x.cos()).abs() < 1e-9,
                "cos({x:e}) = {} vs libm {}",
                cos(x),
                x.cos()
            );
        }
    }

    /// Exact-ish at the anchors the draws hit most.
    #[test]
    fn anchors() {
        assert!((ln(1.0)).abs() < 1e-12);
        assert!((ln(std::f64::consts::E) - 1.0).abs() < 1e-10);
        assert!((cos(0.0) - 1.0).abs() < 1e-12);
        assert!((sin(0.0)).abs() < 1e-12);
    }
}
