//! Small pure-numeric helpers shared across the interpretive paths.
//!
//! These used to be copy-pasted in several modules (findings F4/F5). Collecting them here gives
//! each ONE home so the four interpreters can't silently drift:
//!
//! - [`fold_binop`] — the scalar `BinOp` kernel used by the bytecode VM (`bytecode::apply_bin`),
//!   the signal-tree folder (`signal::scalar_binop`), the `eval` constant-fold, and the
//!   graph-simplifier (`simplify::binary`). The JIT and WASM emitters legitimately stay separate:
//!   they *emit* code, they don't compute, so their kernels live with their backends.
//! - [`floored_mod`] — floored modulo (the `%` operator), which was hand-spelled in several of the
//!   above.
//! - [`trim_float`] — the float-dust-trimming display shared by `value::format_num` and
//!   `flint::fmt_n` (they differ only in decimal places).
//! - [`quantile_sorted`] — the type-7 empirical quantile shared by `introspect` and the `Q` builtin.

use crate::ast::BinOp;

/// Floored modulo `a − b·floor(a/b)` (PLAN-COMPLEX §8): the result takes the **sign of `b`**, so
/// `x % n ∈ [0, n)` for `n > 0` — what modular/clock arithmetic wants (unlike Rust's `%`, which
/// truncates toward zero). IEEE edge cases follow `floor`: `x % 0` is `NaN` (no panic).
#[inline]
pub fn floored_mod(a: f64, b: f64) -> f64 {
    a - b * (a / b).floor()
}

/// The scalar binary kernel: apply `op` to two `f64`s, returning the numeric result. Comparisons
/// and logical ops yield `0.0`/`1.0` indicator values (the columnar representation of bools). This
/// is the single definition every interpretive path shares (finding F4), so `%`'s floored formula,
/// `^`'s `powf`, and the `!= 0.0` boolean convention are spelled exactly once.
#[inline]
pub fn fold_binop(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => floored_mod(a, b),
        BinOp::Pow => a.powf(b),
        BinOp::Lt => (a < b) as i32 as f64,
        BinOp::Gt => (a > b) as i32 as f64,
        BinOp::Le => (a <= b) as i32 as f64,
        BinOp::Ge => (a >= b) as i32 as f64,
        BinOp::Eq => (a == b) as i32 as f64,
        BinOp::Ne => (a != b) as i32 as f64,
        // Logical ops over 0/1 indicator columns.
        BinOp::And => ((a != 0.0) && (b != 0.0)) as i32 as f64,
        BinOp::Or => ((a != 0.0) || (b != 0.0)) as i32 as f64,
    }
}

/// Format a finite `f64` without floating-point dust: print to `places` decimals, then trim
/// trailing zeros (and a lone `.`), so `1.0000000000000002` prints `1` and `0.0871` stays `0.0871`.
/// `0.0`/`-0.0` collapse to `"0"`; non-finite values (`inf`/`nan`) print via their default `Display`.
/// The shared core of `value::format_num` (12 places, full precision) and `flint::fmt_n` (4 places,
/// compact chart labels) — finding F5.
pub fn trim_float(x: f64, places: usize) -> String {
    if x == 0.0 {
        return "0".to_string(); // also collapses -0
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{x:.places$}");
    let t = s.trim_end_matches('0').trim_end_matches('.');
    if t.is_empty() || t == "-0" {
        "0".to_string()
    } else {
        t.to_string()
    }
}

/// Linear-interpolated empirical quantile of a **sorted, non-empty** sample (the type-7 rule,
/// numpy's default): position `q*(len-1)`, blended between its floor/ceil order statistics. Shared
/// by [`crate::introspect`] and the `Q` builtin (finding F5).
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_binop_matches_hand_kernels() {
        // Floored mod takes the sign of b (PLAN-COMPLEX §8).
        assert_eq!(fold_binop(BinOp::Mod, 7.0, 3.0), 1.0);
        assert_eq!(fold_binop(BinOp::Mod, -1.0, 3.0), 2.0);
        assert_eq!(fold_binop(BinOp::Mod, 7.0, -3.0), -2.0);
        assert_eq!(fold_binop(BinOp::Mod, 5.5, 2.0), 1.5);
        assert!(fold_binop(BinOp::Mod, 1.0, 0.0).is_nan());
        // Comparisons and logicals are 0/1.
        assert_eq!(fold_binop(BinOp::Lt, 1.0, 2.0), 1.0);
        assert_eq!(fold_binop(BinOp::Ge, 1.0, 2.0), 0.0);
        assert_eq!(fold_binop(BinOp::And, 1.0, 0.0), 0.0);
        assert_eq!(fold_binop(BinOp::Or, 0.0, 3.0), 1.0);
        assert_eq!(fold_binop(BinOp::Pow, 2.0, 10.0), 1024.0);
    }

    #[test]
    fn trim_float_places() {
        assert_eq!(trim_float(1.000_000_000_000_2, 12), "1");
        assert_eq!(trim_float(0.0871, 4), "0.0871");
        assert_eq!(trim_float(0.0, 4), "0");
        assert_eq!(trim_float(-0.0, 12), "0");
        assert_eq!(trim_float(f64::NAN, 4), "NaN");
    }

    #[test]
    fn quantile_type7() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_sorted(&xs, 0.0), 1.0);
        assert_eq!(quantile_sorted(&xs, 1.0), 5.0);
        assert_eq!(quantile_sorted(&xs, 0.5), 3.0);
        assert_eq!(quantile_sorted(&[42.0], 0.9), 42.0);
    }
}
