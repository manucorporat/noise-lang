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

/// Full-domain `ln(x)` — [`ln`] wrapped in the exact domain guards `jit::emit_ln_guarded` and the
/// wasm emitter lower, so the interpreter's lane path (`bytecode::apply_un`) computes the same
/// bits: `x > 0` → poly (subnormals handled inside [`ln`]), `x == 0` → `-inf`, `x < 0` / NaN →
/// NaN, `+inf` → `+inf`. This is what makes `log(RV)` bit-identical across backends
/// (PLAN-PREGPU draw-stream parity extended to the lane ops).
pub fn ln_guarded(x: f64) -> f64 {
    if x > 0.0 {
        if x == f64::INFINITY {
            f64::INFINITY
        } else {
            ln(x)
        }
    } else if x == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
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

// ---------------------------------------------------------------------------
// f32 twins (PLAN-PREGPU Track B): the lane type is f32, so these — not the f64
// references above — are what the interpreter's lane ops and both emitters compute.
// Same shapes, shorter series: f32 carries ~24 bits, so the tails that earn the last
// f64 digits are pure cost here. The f64 versions stay: deterministic values
// (`Value::Num`, const folding, the signal folder) are still f64, and the f64 `ln` is
// what `CellStream`'s interpreter-only fills use.
// ---------------------------------------------------------------------------

/// `ln(m)` series in f32 — the atanh expansion truncated where f32 runs out of bits:
/// with `|f| ≤ 0.1716` the dropped `f¹¹/11` term is ~3e-9, well under one f32 ulp of
/// `ln`'s result.
pub(crate) const LN_COEFFS_F32: [f32; 5] = [1.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0, 1.0 / 9.0];

/// `sin(r)/r` tail on `[-π/4, π/4]` (Taylor, f32): dropped `r¹¹/11!` ≈ 1.8e-9 at the range end.
pub(crate) const SIN_COEFFS_F32: [f32; 4] = [
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5040.0,
    1.0 / 362_880.0,
];

/// `cos(r)` tail on `[-π/4, π/4]` (Taylor, f32): dropped `r¹²/12!` ≈ 1.1e-10 at the range end.
pub(crate) const COS_COEFFS_F32: [f32; 4] = [
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40_320.0,
    -1.0 / 3_628_800.0,
];

/// π/2 split for the f32 Cody–Waite reduction. The high part has only 8 significant mantissa
/// bits (`0x3FC90000`), so `k · PIO2_HI_F32` is *exact* in f32 for every `k` below 2¹⁵ — far past
/// any `k` the guard admits — and the pair `HI + LO` pins π/2 to ~3e-11.
pub(crate) const PIO2_HI_F32: f32 = 1.5703125;
pub(crate) const PIO2_LO_F32: f32 =
    (std::f64::consts::FRAC_PI_2 - PIO2_HI_F32 as f64) as f32;

/// The f32 twin of [`TRIG_MAX`], and much lower: the 2-term reduction's error grows like
/// `k · 3e-11`, and an f32 ulp near 1 is only 6e-8, so the inline poly stays trustworthy to
/// `|x| < 2¹⁰` (k ≤ 652, error ≲ 2e-8 — a third of an ulp). Past it, all three backends defer to
/// the accurate library `sin`/`cos` **computed in f64 and rounded to f32** (see [`sin_f32`]) —
/// the same shape the f64 path uses, so `sin(1e12 · X)` still agrees everywhere.
pub(crate) const TRIG_MAX_F32: f32 = (1u32 << 10) as f32;

/// Subnormal lift for the f32 `ln` (smallest f32 subnormal `2^-149 · 2^25 = 2^-124` is normal).
pub(crate) const LN_SUBNORMAL_SCALE_F32: f32 = (1u32 << 25) as f32;
pub(crate) const LN_SUBNORMAL_CORR_F32: f32 = 25.0 * std::f32::consts::LN_2;

fn horner_f32(z: f32, c: &[f32]) -> f32 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * z + ci)
}

/// `ln(x)` for `x > 0`, f32 — [`ln`]'s algorithm on f32 fields (8-bit exponent, 23-bit mantissa).
pub fn ln_f32(x: f32) -> f32 {
    use std::f32::consts::{LN_2, SQRT_2};
    if x > 0.0 && x < f32::MIN_POSITIVE {
        return ln_f32(x * LN_SUBNORMAL_SCALE_F32) - LN_SUBNORMAL_CORR_F32;
    }
    let bits = x.to_bits();
    let e0 = ((bits >> 23) & 0xff) as i32 - 127;
    let m_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let m0 = f32::from_bits(m_bits); // [1, 2)
    let big = m0 > SQRT_2;
    let m = if big { m0 * 0.5 } else { m0 };
    let e = if big { e0 + 1 } else { e0 };
    let f = (m - 1.0) / (m + 1.0);
    let f2 = f * f;
    2.0 * f * horner_f32(f2, &LN_COEFFS_F32) + (e as f32) * LN_2
}

/// Full-domain f32 `ln` — the guards both emitters lower, so `log(RV)` is bit-identical
/// everywhere (see [`ln_guarded`]).
pub fn ln_guarded_f32(x: f32) -> f32 {
    if x > 0.0 {
        if x == f32::INFINITY {
            f32::INFINITY
        } else {
            ln_f32(x)
        }
    } else if x == 0.0 {
        f32::NEG_INFINITY
    } else {
        f32::NAN
    }
}

fn reduce_f32(x: f32) -> (f32, i32) {
    let k = (x * std::f32::consts::FRAC_2_PI).round_ties_even();
    let r = (x - k * PIO2_HI_F32) - k * PIO2_LO_F32;
    (r, k as i32)
}

fn sin_kernel_f32(r: f32) -> f32 {
    let z = r * r;
    r + r * z * horner_f32(z, &SIN_COEFFS_F32)
}

fn cos_kernel_f32(r: f32) -> f32 {
    let z = r * r;
    1.0 - 0.5 * z + z * z * horner_f32(z, &COS_COEFFS_F32)
}

fn quadrant_f32(kq: i32, sin_r: f32, cos_r: f32, is_cos: bool) -> f32 {
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

/// `cos(x)` in f32 — inline poly under [`TRIG_MAX_F32`], else the f64 library `cos` rounded to
/// f32. That "promote, call, demote" shape (rather than `f32::cos`, i.e. `cosf`) is the shared
/// contract: the JIT's shim and the wasm module's `Math.cos` import both compute in f64 and
/// round, so all three backends agree bit-for-bit on the fallback too.
pub fn cos_f32(x: f32) -> f32 {
    if x.abs() >= TRIG_MAX_F32 {
        return (x as f64).cos() as f32;
    }
    let (r, k) = reduce_f32(x);
    quadrant_f32(k & 3, sin_kernel_f32(r), cos_kernel_f32(r), true)
}

/// `sin(x)` in f32 (see [`cos_f32`] — same guard and f64-rounded fallback).
pub fn sin_f32(x: f32) -> f32 {
    if x.abs() >= TRIG_MAX_F32 {
        return (x as f64).sin() as f32;
    }
    let (r, k) = reduce_f32(x);
    quadrant_f32(k & 3, sin_kernel_f32(r), cos_kernel_f32(r), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f32 polynomials must track libm to within a few f32 ulps over the ranges the lane ops
    /// exercise — the Track B bar (f32 carries ~7 decimal digits; Monte-Carlo noise dwarfs the
    /// rest). `ln` is checked *relative*, trig *absolute* (their outputs are O(1)).
    #[test]
    fn f32_approximations_track_libm() {
        let mut max_ln = 0.0f32;
        // The whole f32-uniform-reachable ln domain: Box–Muller feeds `1 - u ∈ [2^-24, 1]`.
        let mut x = 2f32.powi(-24);
        while x <= 4.0 {
            let want = (x as f64).ln() as f32;
            let rel = (ln_f32(x) - want).abs() / want.abs().max(1.0);
            max_ln = max_ln.max(rel);
            x *= 1.000_01;
        }
        assert!(max_ln < 1e-6, "ln_f32 relative err {max_ln:e}");

        // Down into the subnormals and up past 1e30 (a user can steer `math::log` anywhere).
        let mut x = f32::MIN_POSITIVE * 0.5;
        let mut max_wide = 0.0f32;
        for _ in 0..300 {
            let want = (x as f64).ln() as f32;
            max_wide = max_wide.max((ln_f32(x) - want).abs() / want.abs().max(1.0));
            x *= 1.7;
        }
        assert!(max_wide < 1e-6, "ln_f32 wide-range relative err {max_wide:e}");

        let (mut max_sin, mut max_cos) = (0.0f32, 0.0f32);
        for i in 0..400_000 {
            let x = (i as f32 - 200_000.0) / 2_000.0; // -100 .. 100
            max_sin = max_sin.max((sin_f32(x) - (x as f64).sin() as f32).abs());
            max_cos = max_cos.max((cos_f32(x) - (x as f64).cos() as f32).abs());
        }
        assert!(max_sin < 1e-6, "sin_f32 max abs err {max_sin:e}");
        assert!(max_cos < 1e-6, "cos_f32 max abs err {max_cos:e}");
    }

    /// Past [`TRIG_MAX_F32`] the reduction is nonsense, so the guard must hand off to the f64
    /// library — the f32 twin of the C3 fallback (`sin(1e12 * X)`).
    #[test]
    fn f32_trig_agrees_past_the_reduction_limit() {
        for &x in &[1e4f32, 1e6, 1e12, 1e20, -1e12, TRIG_MAX_F32, TRIG_MAX_F32 * 4.0] {
            assert_eq!(sin_f32(x), (x as f64).sin() as f32, "sin_f32({x:e})");
            assert_eq!(cos_f32(x), (x as f64).cos() as f32, "cos_f32({x:e})");
        }
    }

    /// The exact anchors the Box–Muller path depends on: `ln(1) == 0` exactly (so `u1 == 0`
    /// yields `r == 0`, never a NaN), and the reduced-argument kernels are exact at 0.
    #[test]
    fn f32_anchors() {
        assert_eq!(ln_f32(1.0), 0.0);
        assert_eq!(ln_guarded_f32(0.0), f32::NEG_INFINITY);
        assert!(ln_guarded_f32(-1.0).is_nan());
        assert_eq!(ln_guarded_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(cos_f32(0.0), 1.0);
        assert_eq!(sin_f32(0.0), 0.0);
    }

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
