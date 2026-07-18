//! Scalar / complex / signal operator folding (`binop` and friends), complex arithmetic and `^`, signal materialization, and the value-materialization helpers (`select`/`indicator`/`array_index`/`expect_array`).
//!
//! Extracted verbatim from the monolithic `eval.rs` (finding F1); an `impl Engine` block
//! whose methods reach the rest of the evaluator through `self` and the shared free
//! helpers/tables that stay in the module root.

use std::collections::HashMap;
use std::rc::Rc;

use super::*;
use crate::dist::{RvKind, RvNode};
use crate::error::{NoiseError, Result, Span};
use crate::signal::{NoiseSpec, RealizationId, SigExpr, SigUnOp};
use crate::value::Value;

impl Engine {
    /// Evaluate a prefix unary op (`-x` / `!x` / the math ufuncs). A conditioned operand pushes the
    /// op into its quantity and keeps the condition (`-(X|C)` is `(-X) | C`); otherwise the complex /
    /// RV-lift / deterministic paths apply as before. Its own method so `eval`'s frame stays small.
    pub(super) fn eval_unary_expr(&mut self, op: UnOp, rhs: &Spanned, span: Span) -> Result<Value> {
        let v = self.eval(rhs)?;
        forbid_undrawn(&v, rhs.span)?;
        if let Value::Cond {
            quantity,
            condition,
            ..
        } = v
        {
            let q = self.lift_unary(op, Value::Dist(quantity), span)?;
            let (quantity, q_kind) = self.operand_to_rv(q, span)?;
            Ok(Value::Cond {
                quantity,
                q_kind,
                condition,
            })
        } else if matches!(v, Value::Complex { .. }) {
            self.unary_complex(op, v, span)
        } else if let Value::Signal(s) = v {
            // A prefix op on a lazy signal defers into the tree (`-sine(3)` stays a signal).
            Ok(Value::Signal(Rc::new(SigExpr::Unary(SigUnOp::Un(op), s))))
        } else if let Value::Sym(s) = v {
            // A prefix op on a symbolic input defers, staying symbolic (`-barrier` is still a uniform
            // expression); it lowers to graph nodes only when it enters the RV graph as a value.
            Ok(Value::Sym(Rc::new(crate::sym::SymExpr::Unary(op, s))))
        } else if is_dist(&v) {
            self.lift_unary(op, v, span)
        } else {
            eval_unary(op, v, span) // deterministic fast path, unchanged
        }
    }

    /// A binary op with at least one conditioned operand — `2*(X|C)`, `(X|C)+1`, `(X|C) < 3`, or
    /// `(X|C)+(Y|C)`. The op pushes into the quantity and the condition rides along, so the result
    /// is the conditioned value `(quantity ⊕ other) | C`. Two conditioned operands must share the
    /// *same* condition node — conditioning on two different events at once is ill-defined, so it is
    /// a spanned error (condition once, at the end: `(X + Y) | C`).
    fn binop_cond(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        let (quantity, condition) = match (l, r) {
            (
                Value::Cond {
                    quantity: ql,
                    condition: cl,
                    ..
                },
                Value::Cond {
                    quantity: qr,
                    condition: cr,
                    ..
                },
            ) => {
                if cl != cr {
                    return Err(NoiseError::runtime(
                        "cannot combine two values conditioned on different events — condition once, \
                         at the end (e.g. `(X + Y) | C`)"
                            .to_string(),
                        span,
                    ));
                }
                (self.binop(op, Value::Dist(ql), Value::Dist(qr), span)?, cl)
            }
            (
                Value::Cond {
                    quantity,
                    condition,
                    ..
                },
                other,
            ) => (
                self.binop(op, Value::Dist(quantity), other, span)?,
                condition,
            ),
            (
                other,
                Value::Cond {
                    quantity,
                    condition,
                    ..
                },
            ) => (
                self.binop(op, other, Value::Dist(quantity), span)?,
                condition,
            ),
            _ => unreachable!("binop_cond called without a conditioned operand"),
        };
        // Re-wrap the transformed quantity, keeping the condition. The quantity is always an RV here
        // (it has a `Dist` operand), so `operand_to_rv` just reads its id/kind.
        let (quantity, q_kind) = self.operand_to_rv(quantity, span)?;
        Ok(Value::Cond {
            quantity,
            q_kind,
            condition,
        })
    }

    /// Combine two element `Value`s with a binary op — the single fold primitive the library
    /// reuses (LANG.md §0). Lifts to a graph node if either side is a `Dist`, else folds
    /// deterministically. Recipes are rejected (you can't operate on an undrawn distribution).
    pub(super) fn binop(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        forbid_undrawn(&l, span)?;
        forbid_undrawn(&r, span)?;
        // A conditioned operand pushes the op into its quantity and carries the condition along —
        // `2*(X|C)` is `(2X) | C`. Handle before the other paths so a conditioned value never folds
        // or lifts as a plain RV.
        if matches!(l, Value::Cond { .. }) || matches!(r, Value::Cond { .. }) {
            return self.binop_cond(op, l, r, span);
        }
        // A lazy signal defers ops (growing its expression tree) and materializes against a sized
        // array. Handle it before the array path so `signal ⊕ array` adopts the array's length.
        // (An undrawn noise generator never reaches here — `forbid_undrawn` above rejects it.)
        if matches!(l, Value::Signal(_)) || matches!(r, Value::Signal(_)) {
            return self.binop_signal(op, l, r, span);
        }
        // Arrays broadcast elementwise (NumPy-style): array⊕array (length-matched) and
        // array⊕scalar both map the op over the elements — so `signal + noise`, `1 + m`, and
        // `phase / kf` all work on whole signals. (A complex scalar meeting an array broadcasts
        // here, recursing into `binop` per element where the complex path below handles it.)
        if matches!(l, Value::Array(_)) || matches!(r, Value::Array(_)) {
            return self.binop_broadcast(op, l, r, span);
        }
        // A complex operand (either a constant `2 + 3i` or a complex RV) routes through the
        // complex arithmetic path: `* / ^` are true complex operations, a real operand promotes
        // to `re + 0i`, and ordering (`< > <= >=`) is a type error (no total order on ℂ). A tunable
        // input meeting a complex value folds to its current value first (P0 fallback — complex
        // signal params are PLAN-UNIFORM-INPUTS P2).
        if matches!(l, Value::Complex { .. }) || matches!(r, Value::Complex { .. }) {
            let l = self.force_sym_value(l);
            let r = self.force_sym_value(r);
            return self.binop_complex(op, l, r, span);
        }
        // A symbolic input (`input::real`) keeps the op symbolic against another scalar/symbolic
        // operand — `barrier * 2` stays a `Sym` so it later lowers as one uniform — and lowers to an
        // RV `Input` node when it meets a random variable (`path < barrier`). Handle it after the
        // Signal/Array/Complex paths (they force the Sym to a scalar) and before the plain scalar/RV
        // paths (which would otherwise reject or bake it).
        if matches!(l, Value::Sym(_)) || matches!(r, Value::Sym(_)) {
            return self.binop_sym(op, l, r, span);
        }
        // Algebraic identity folds before lifting: `0*x → 0`, `1*x → x`, `x+0/0+x → x`, `x-0 → x`.
        // These keep an RV out of the graph where it (mostly) doesn't matter — and, crucially, let
        // `math::i * x` keep a *literal* `0` real channel (`0*x`), so a complex `exp` over a random
        // angle (`e^{i·X}`) still sees a constant real part. Only fires when the non-constant side is
        // numeric, so `0 * bool_event` still type-errors rather than silently folding to 0. NOTE the
        // `0*x → 0` fold trades away non-finite propagation for discrete RVs (finding B6) — see
        // [`Self::fold_identity`] for the honest accounting.
        if let Some(folded) = self.fold_identity(op, &l, &r) {
            return Ok(folded);
        }
        if is_dist(&l) || is_dist(&r) {
            self.lift_binary(op, l, r, span)
        } else {
            eval_binary(op, l, r, span)
        }
    }

    /// Whether `v` is a `dist<number>` — the survivor guard for [`Self::fold_identity`]. The fold
    /// fires *only* on the RV path: for two constants, `eval_binary` already evaluates the identity
    /// IEEE-honestly (e.g. `0 * inf == NaN`), so there is nothing to fold and nothing to get wrong.
    /// Restricting to a numeric RV also keeps `0 * event` (a `dist<bool>`) a clean type error.
    fn is_num_dist(&self, v: &Value) -> bool {
        matches!(v, Value::Dist(id) if self.graph.kind(*id) == RvKind::Num)
    }

    /// Fold the arithmetic identities `1*x → x`, `x+0/0+x → x`, `x-0 → x`, and `0*x → 0` when the
    /// surviving operand is a numeric RV (`dist<number>`) and the other side is the literal `0`/`1`.
    /// This keeps a provably-irrelevant RV out of the graph — and, crucially, lets `math::i * x`
    /// keep a *literal* `0` real channel (`0*x`), so a complex `exp` over a random angle (`e^{i·X}`)
    /// still sees a constant real part.
    ///
    /// The `1*x`, `x+0`, `0+x`, `x-0` folds are exact for *every* IEEE value of `x` (including
    /// ±inf/NaN), so they are always sound. `0*x → 0` is NOT (finding B6): honest IEEE gives
    /// `0*inf == NaN` and `0*NaN == NaN`, and for a **discrete** RV those inputs occur with real
    /// probability, not measure zero — e.g. `X ~ unif_int(0,5); 0*(1/X)` folds to `0`, but the
    /// honest result is `NaN` on the `X == 0` lane (probability 1/6). We keep the fold deliberately:
    /// the payoff (a literal `0` complex real channel, and pruning provably-irrelevant RVs) is worth
    /// more than faithful non-finite propagation for `0*x`, and a *correct* "is `x` finite?" guard
    /// isn't local (overflow, plus `Div`/`Ln`/`Pow` reachability, would all have to be proven). The
    /// trade-off is this: `0*x` discards `x`'s non-finite lanes. For constant operands the fold never
    /// fires (`eval_binary` evaluates `0*inf` honestly). `None` falls through.
    fn fold_identity(&self, op: BinOp, l: &Value, r: &Value) -> Option<Value> {
        let is0 = |v: &Value| matches!(v, Value::Num(n) if *n == 0.0);
        let is1 = |v: &Value| matches!(v, Value::Num(n) if *n == 1.0);
        match op {
            BinOp::Mul => {
                if (is0(l) && self.is_num_dist(r)) || (is0(r) && self.is_num_dist(l)) {
                    Some(Value::Num(0.0))
                } else if is1(l) && self.is_num_dist(r) {
                    Some(r.clone())
                } else if is1(r) && self.is_num_dist(l) {
                    Some(l.clone())
                } else {
                    None
                }
            }
            BinOp::Add => {
                if is0(l) && self.is_num_dist(r) {
                    Some(r.clone())
                } else if is0(r) && self.is_num_dist(l) {
                    Some(l.clone())
                } else {
                    None
                }
            }
            BinOp::Sub if is0(r) && self.is_num_dist(l) => Some(l.clone()),
            _ => None,
        }
    }

    /// Elementwise broadcast of a binary op when at least one operand is an array.
    fn binop_broadcast(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        let out = match (l, r) {
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(length_mismatch("elementwise op", a.len(), b.len(), span));
                }
                let mut out = Vec::with_capacity(a.len());
                for (ai, bi) in a.iter().zip(b.iter()) {
                    out.push(self.binop(op, ai.clone(), bi.clone(), span)?);
                }
                out
            }
            (Value::Array(a), scalar) => {
                let mut out = Vec::with_capacity(a.len());
                for ai in a.iter() {
                    out.push(self.binop(op, ai.clone(), scalar.clone(), span)?);
                }
                out
            }
            (scalar, Value::Array(b)) => {
                let mut out = Vec::with_capacity(b.len());
                for bi in b.iter() {
                    out.push(self.binop(op, scalar.clone(), bi.clone(), span)?);
                }
                out
            }
            _ => unreachable!("binop_broadcast called without an array operand"),
        };
        Ok(Value::Array(Rc::new(out)))
    }

    /// Combine a lazy signal with another operand. A scalar or another signal **defers** (the op
    /// grows the expression tree, staying O(1) memory — `sine(3) + sine(7)` is a two-tone
    /// signal); a **complex** operand routes to the complex path (the signal promotes to
    /// `sig + 0i`); a sized **array materializes** the signal to that length and then broadcasts.
    fn binop_signal(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        // Complex first: `signal ⊕ complex` decomposes into channel ops, which land back here as
        // plain signal arithmetic.
        if matches!(l, Value::Complex { .. }) || matches!(r, Value::Complex { .. }) {
            return self.binop_complex(op, l, r, span);
        }
        match (l, r) {
            (Value::Signal(a), Value::Signal(b)) => {
                Ok(Value::Signal(Rc::new(SigExpr::Binop(op, a, b))))
            }
            // A sized array fixes the sample count: materialize the signal, then broadcast.
            (Value::Signal(s), arr @ Value::Array(_)) => {
                let mat = Value::Array(Rc::new(self.materialize_sig(&s, array_len(&arr), span)?));
                self.binop(op, mat, arr, span)
            }
            (arr @ Value::Array(_), Value::Signal(s)) => {
                let mat = Value::Array(Rc::new(self.materialize_sig(&s, array_len(&arr), span)?));
                self.binop(op, arr, mat, span)
            }
            // signal ∘ scalar-const (either order) → defer as a `Konst` leaf. `if let` computes the
            // constant once (finding F10), replacing a `scalar_const(..).is_some()` guard + re-eval.
            (Value::Signal(a), rhs) => match scalar_const(&rhs) {
                Some(k) => Ok(Value::Signal(Rc::new(SigExpr::Binop(
                    op,
                    a,
                    Rc::new(SigExpr::Konst(k)),
                )))),
                None => Err(signal_combine_error(&rhs, span)),
            },
            (lhs, Value::Signal(b)) => match scalar_const(&lhs) {
                Some(k) => Ok(Value::Signal(Rc::new(SigExpr::Binop(
                    op,
                    Rc::new(SigExpr::Konst(k)),
                    b,
                )))),
                None => Err(signal_combine_error(&lhs, span)),
            },
            // `binop_signal` is only entered with a signal operand, so one of the arms above always
            // matches; this keeps the match exhaustive.
            _ => unreachable!("binop_signal called without a signal operand"),
        }
    }

    /// Combine two operands where at least one is a **symbolic input** (`Value::Sym`,
    /// PLAN-UNIFORM-INPUTS). Two rules, mirroring `binop_signal`:
    ///   * against a **random variable** (`Value::Dist`, either order) the `Sym` lowers to an
    ///     `RvNode::Input` node and the op lifts (`operand_to_rv` does the lowering) — this is how a
    ///     threshold `path < barrier` becomes a per-lane compare against a uniform;
    ///   * against another **scalar/symbolic** operand (`Num`/`Est`/`Sym`) the op stays symbolic
    ///     (`Sym`), so `barrier * 2` defers and later lowers as a single uniform expression.
    fn binop_sym(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        // Either side an RV ⇒ lift (the Sym operand lowers to an Input node in `operand_to_rv`).
        if is_dist(&l) || is_dist(&r) {
            return self.lift_binary(op, l, r, span);
        }
        // Otherwise fold symbolically: both operands must be scalar-ish (a Sym, or a plain number).
        let (a, b) = (sym_operand(&l, span)?, sym_operand(&r, span)?);
        Ok(Value::Sym(Rc::new(crate::sym::SymExpr::Binary(op, a, b))))
    }

    /// Combine two operands where at least one is **complex** (PLAN-COMPLEX §3). Every complex op
    /// is expressed as real ops on the two channels via [`Self::binop`], so it folds for constant
    /// complex and lifts into the (real) sample-DAG when a channel is an RV — no complex value
    /// flows through the VM. A real operand promotes to `re + 0i`; ordering and logical ops are a
    /// type error (ℂ has no total order, and a complex value is not an event).
    fn binop_complex(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        use BinOp::*;
        match op {
            Add | Sub => {
                let (lr, li) = self.complex_parts(l, span)?;
                let (rr, ri) = self.complex_parts(r, span)?;
                let re = self.binop(op, lr, rr, span)?;
                let im = self.binop(op, li, ri, span)?;
                Ok(Value::complex(re, im))
            }
            Mul => {
                // (a+bi)(c+di) = (ac − bd) + (ad + bc) i
                let (a, b) = self.complex_parts(l, span)?;
                let (c, d) = self.complex_parts(r, span)?;
                let ac = self.binop(Mul, a.clone(), c.clone(), span)?;
                let bd = self.binop(Mul, b.clone(), d.clone(), span)?;
                let ad = self.binop(Mul, a, d, span)?;
                let bc = self.binop(Mul, b, c, span)?;
                let re = self.binop(Sub, ac, bd, span)?;
                let im = self.binop(Add, ad, bc, span)?;
                Ok(Value::complex(re, im))
            }
            Div => {
                // (a+bi)/(c+di) = [(ac + bd) + (bc − ad) i] / (c² + d²)
                let (a, b) = self.complex_parts(l, span)?;
                let (c, d) = self.complex_parts(r, span)?;
                let cc = self.binop(Mul, c.clone(), c.clone(), span)?;
                let dd = self.binop(Mul, d.clone(), d.clone(), span)?;
                let denom = self.binop(Add, cc, dd, span)?;
                let ac = self.binop(Mul, a.clone(), c.clone(), span)?;
                let bd = self.binop(Mul, b.clone(), d.clone(), span)?;
                let bc = self.binop(Mul, b, c, span)?;
                let ad = self.binop(Mul, a, d, span)?;
                let re_num = self.binop(Add, ac, bd, span)?;
                let im_num = self.binop(Sub, bc, ad, span)?;
                let re = self.binop(Div, re_num, denom.clone(), span)?;
                let im = self.binop(Div, im_num, denom, span)?;
                Ok(Value::complex(re, im))
            }
            Pow => self.complex_pow(l, r, span),
            // Exact (re, im) comparison: equal iff both channels are equal.
            Eq => {
                let (lr, li) = self.complex_parts(l, span)?;
                let (rr, ri) = self.complex_parts(r, span)?;
                let re_eq = self.binop(Eq, lr, rr, span)?;
                let im_eq = self.binop(Eq, li, ri, span)?;
                self.binop(And, re_eq, im_eq, span)
            }
            Ne => {
                let (lr, li) = self.complex_parts(l, span)?;
                let (rr, ri) = self.complex_parts(r, span)?;
                let re_ne = self.binop(Ne, lr, rr, span)?;
                let im_ne = self.binop(Ne, li, ri, span)?;
                self.binop(Or, re_ne, im_ne, span)
            }
            Lt | Gt | Le | Ge => Err(NoiseError::runtime(
                "complex numbers have no ordering (ℂ is not totally ordered) — compare `math::abs(z)` if you mean magnitude".to_string(),
                span,
            )),
            Mod => Err(NoiseError::runtime(
                "modulo `%` is real-only — it has no meaning on a complex number".to_string(),
                span,
            )),
            And | Or => Err(NoiseError::runtime(
                "logical operator needs two bool events, got complex".to_string(),
                span,
            )),
        }
    }

    /// `z ^ k` for a complex base and a **constant integer** exponent: repeated complex multiply
    /// (negative `k` reciprocates). Enough for the QFT/quantum path; a general complex exponent is
    /// deferred (a clear error). The exponent magnitude is capped so a stray `z ^ 1e9` can't build
    /// an unbounded graph at build time.
    fn complex_pow(&mut self, base: Value, exp: Value, span: Span) -> Result<Value> {
        let k = match scalar_const(&exp) {
            Some(k) if k.fract() == 0.0 && k.is_finite() => k,
            _ => {
                return Err(NoiseError::runtime(
                    "complex `^` needs a constant integer exponent (e.g. `z ^ 2`); a general complex power is not supported".to_string(),
                    span,
                ))
            }
        };
        if k.abs() > 4096.0 {
            return Err(NoiseError::runtime(
                format!("complex `^` exponent {k} is too large (max magnitude 4096)"),
                span,
            ));
        }
        let n = k.abs() as u32;
        let mut acc = Value::cnum(1.0, 0.0);
        for _ in 0..n {
            acc = self.binop_complex(BinOp::Mul, acc, base.clone(), span)?;
        }
        if k < 0.0 {
            acc = self.binop_complex(BinOp::Div, Value::cnum(1.0, 0.0), acc, span)?;
        }
        Ok(acc)
    }

    /// Prefix unary op on a **complex** value. Only `-` (negate both channels) is defined; `!`
    /// (logical not) needs a bool. Negation goes through `binop` so it lifts when a channel is an RV.
    fn unary_complex(&mut self, op: UnOp, v: Value, span: Span) -> Result<Value> {
        let (re, im) = self.complex_parts(v, span)?;
        match op {
            UnOp::Neg => {
                let nr = self.binop(BinOp::Sub, Value::Num(0.0), re, span)?;
                let ni = self.binop(BinOp::Sub, Value::Num(0.0), im, span)?;
                Ok(Value::complex(nr, ni))
            }
            _ => Err(NoiseError::runtime(
                format!("cannot apply {} to a complex value", unop_name(op)),
                span,
            )),
        }
    }

    /// Split an operand into its `(re, im)` real channels for complex arithmetic. A complex value
    /// hands back its stored channels; a real scalar (`Num`/`Est`/`Dist<number>`) promotes to
    /// `x + 0i`; anything else (`Bool`/`Str`/array/…) is a spanned type error.
    pub(super) fn complex_parts(&self, v: Value, span: Span) -> Result<(Value, Value)> {
        match v {
            Value::Complex { re, im } => Ok((*re, *im)),
            // A real scalar promotes to `x + 0i`. A `Sym` rides this path too, so `abs`/`re`/`im`/
            // `conj` of a slider — and everything built on them (`norm`, `normsq`, `normalize`,
            // `mse`) — stays symbolic instead of erroring.
            v @ (Value::Num(_) | Value::Est { .. } | Value::Sym(_)) => Ok((v, Value::Num(0.0))),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Num => {
                Ok((Value::Dist(id), Value::Num(0.0)))
            }
            // A real lazy signal promotes to `sig + 0i` — this is what makes a complex signal
            // pure composition (`Complex { re: Signal, im: Signal }`, PLAN-SIGNALS §1.3).
            v @ Value::Signal(_) => Ok((v, Value::Num(0.0))),
            other => Err(NoiseError::runtime(
                format!("cannot use {} in a complex expression", other.type_name()),
                span,
            )),
        }
    }

    /// Materialize a lazy signal tree to `n` element `Value`s (PLAN-SIGNALS §3). A deterministic
    /// tree folds straight to `f64`s ([`SigExpr::sample_f64`] — the fast path `nyquist.noise`
    /// rides); once a subtree carries a drawn-noise leaf the walk switches to per-element `Value`
    /// combination through the ordinary [`Self::binop`] lifting, with the **realization cache**
    /// guaranteeing every mention of the same draw yields the same RV nodes.
    pub(super) fn materialize_sig(
        &mut self,
        expr: &SigExpr,
        n: usize,
        span: Span,
    ) -> Result<Vec<Value>> {
        // The signal is an `Rc` **DAG**: `for k in 0..40 { s = s + s }` shares one child under both
        // `Binop` arms, so a naive tree walk would build `O(2^depth)` RV nodes and abort (finding
        // A3). Memoize the materialized column per node by `Rc` identity so a shared subtree is
        // built once — this is *also* more correct (a shared subtree yields the same RV nodes, so
        // `static - static == 0` holds structurally, not just via later CSE).
        let mut cache: HashMap<*const SigExpr, Rc<Vec<Value>>> = HashMap::new();
        self.materialize_sig_memo(expr, n, span, &mut cache)
    }

    fn materialize_sig_memo(
        &mut self,
        expr: &SigExpr,
        n: usize,
        span: Span,
        cache: &mut HashMap<*const SigExpr, Rc<Vec<Value>>>,
    ) -> Result<Vec<Value>> {
        let key = expr as *const SigExpr;
        if let Some(v) = cache.get(&key) {
            return Ok((**v).clone());
        }
        // A noise-free (sub)tree folds straight to `f64`s — no RV nodes needed for it (the fast
        // path `nyquist.noise` rides). `has_noise` is itself memoized, so this stays cheap.
        let out = if !expr.has_noise() {
            expr.sample_f64(n).into_iter().map(Value::Num).collect()
        } else {
            match expr {
                SigExpr::Noise { id, spec } => self.realization(*id, *spec, n, span)?,
                SigExpr::Unary(op, a) => {
                    let xs = self.materialize_sig_memo(a, n, span, cache)?;
                    let mut out = Vec::with_capacity(n);
                    for x in xs {
                        out.push(self.sig_unary(*op, x, span)?);
                    }
                    out
                }
                SigExpr::Binop(op, a, b) => {
                    let ls = self.materialize_sig_memo(a, n, span, cache)?;
                    let rs = self.materialize_sig_memo(b, n, span, cache)?;
                    let mut out = Vec::with_capacity(n);
                    for (l, r) in ls.into_iter().zip(rs) {
                        out.push(self.binop(*op, l, r, span)?);
                    }
                    out
                }
                SigExpr::Atan2(y, x) => {
                    let ys = self.materialize_sig_memo(y, n, span, cache)?;
                    let xs = self.materialize_sig_memo(x, n, span, cache)?;
                    let mut out = Vec::with_capacity(n);
                    for (yv, xv) in ys.into_iter().zip(xs) {
                        out.push(self.complex_atan2(yv, xv, span)?);
                    }
                    out
                }
                // `has_noise` is true here, so the deterministic leaves are unreachable.
                SigExpr::Wave { .. } | SigExpr::Konst(_) => {
                    unreachable!("deterministic leaf under a noise-bearing walk")
                }
            }
        };
        cache.insert(key, Rc::new(out.clone()));
        Ok(out)
    }

    /// One deferred unary step applied to a materialized element: a constant folds with the same
    /// kernel the deterministic walk uses; an RV lifts (a deferred `exp` over a noisy lane lifts
    /// to the same `UnOp::Exp` node `math::exp` of an RV builds).
    fn sig_unary(&mut self, op: SigUnOp, x: Value, span: Span) -> Result<Value> {
        match (op, x) {
            (op, Value::Num(v)) => Ok(Value::Num(crate::signal::apply_unary(op, v))),
            (op, Value::Est { val, .. }) => Ok(Value::Num(crate::signal::apply_unary(op, val))),
            (SigUnOp::Un(u), x @ Value::Dist(_)) => self.lift_unary(u, x, span),
            (SigUnOp::Exp, x @ Value::Dist(_)) => self.lift_unary(UnOp::Exp, x, span),
            (_, other) => Err(NoiseError::runtime(
                format!("cannot apply a signal op to {}", other.type_name()),
                span,
            )),
        }
    }

    /// Resolve a drawn-noise leaf through the **realization cache**: the first materialization
    /// renders the spec at length `n` and pins it; every later mention gets the SAME `Value`s
    /// (the same RV nodes — that is what makes `static - static` exactly 0 and re-rendering the
    /// same draw at two carriers a fair fight). A different `n` is a teaching error: white noise
    /// has no finer version of itself, so a silent re-render would be a lie (PLAN-SIGNALS §1.1).
    fn realization(
        &mut self,
        id: RealizationId,
        spec: NoiseSpec,
        n: usize,
        span: Span,
    ) -> Result<Vec<Value>> {
        if let Some(vals) = self.realizations.get(&id) {
            if vals.len() != n {
                return Err(NoiseError::runtime(
                    format!(
                        "this drawn noise was realized at length {} and cannot be re-rendered at {n} — \
                         noise has no finer version of itself; keep one resolution across its uses, \
                         or pin it with `~[{}]`",
                        vals.len(),
                        vals.len()
                    ),
                    span,
                ));
            }
            return Ok(vals.clone());
        }
        let vals = self.materialize_noise(spec, n);
        self.realizations.insert(id, vals.clone());
        Ok(vals)
    }

    /// `cond ? a : b` as a value: a deterministic bool picks a branch; a bool-RV builds a
    /// per-lane `Select`. The library's `max`/`min`/`vsign` reuse this (mirrors the lifted `if`).
    pub(super) fn select(&mut self, cond: Value, a: Value, b: Value, span: Span) -> Result<Value> {
        match cond {
            Value::Bool(true) => Ok(a),
            Value::Bool(false) => Ok(b),
            // A symbolic condition (`max([slider, 1])`, `if slider > 1 { … }`) is deterministic
            // *given* the inputs, so picking the branch here is exact. It is a STRUCTURAL use —
            // the branch is baked, so the program rebuilds when the slider crosses the boundary
            // (correct: the two branches can be different shapes, `Sym` has no `Select` node).
            Value::Sym(ref s) => {
                if self.force_sym(s) != 0.0 {
                    Ok(a)
                } else {
                    Ok(b)
                }
            }
            Value::Dist(c) if self.graph.kind(c) == RvKind::Bool => {
                let (aid, ak) = self.operand_to_rv(a, span)?;
                let (bid, bk) = self.operand_to_rv(b, span)?;
                if ak != bk {
                    return Err(NoiseError::runtime(
                        format!(
                            "select branches must have the same type, got {} and {}",
                            ak.type_name(),
                            bk.type_name()
                        ),
                        span,
                    ));
                }
                Ok(Value::Dist(self.graph.push(
                    RvNode::Select {
                        cond: c,
                        a: aid,
                        b: bid,
                    },
                    ak,
                )))
            }
            other => Err(NoiseError::runtime(
                format!("expected a bool condition, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Convert a boolean element (deterministic or bool-RV) to a numeric `0`/`1` indicator, so
    /// `count` can sum events. A bool-RV becomes a `Select(cond, 1, 0)` (a `Num` node).
    pub(super) fn indicator(&mut self, v: Value, span: Span) -> Result<Value> {
        match v {
            Value::Bool(b) => Ok(Value::Num(if b { 1.0 } else { 0.0 })),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Bool => {
                self.select(Value::Dist(id), Value::Num(1.0), Value::Num(0.0), span)
            }
            // A `Sym` comparison tree already evaluates to a 0/1 indicator (`num::fold_binop`), so
            // it *is* its own indicator — pass it through and it stays symbolic.
            sym @ Value::Sym(_) => Ok(sym),
            other => Err(NoiseError::runtime(
                format!("count expects boolean elements, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Resolve an index value to a usize: it must be a non-negative integer point mass — never a
    /// random variable (a random gather is the dynamics fork — PLAN-COLLECTIONS §1).
    pub(super) fn array_index(&self, v: &Value, span: Span) -> Result<usize> {
        let n = match v {
            Value::Num(n) => *n,
            Value::Est { val, .. } => *val,
            // A tunable input used as an index is a STRUCTURAL use — fold it to its current value
            // (the program legitimately recompiles when the index changes).
            Value::Sym(s) => self.force_sym(s),
            Value::Dist(_) => {
                return Err(NoiseError::runtime(
                    "array index must be a constant integer, not a random variable".to_string(),
                    span,
                ))
            }
            other => {
                return Err(NoiseError::runtime(
                    format!("array index must be a number, got {}", other.type_name()),
                    span,
                ))
            }
        };
        if n.fract() != 0.0 || n < 0.0 || !n.is_finite() {
            return Err(NoiseError::runtime(
                format!("array index must be a non-negative integer, got {n}"),
                span,
            ));
        }
        Ok(n as usize)
    }

    /// Extract the elements of an `Array` value, or a spanned error naming the actual type.
    ///
    /// This is the reducers' single funnel (PLAN-SIGNALS §1.2): a **lazy signal** handed to a
    /// reducer (`mse`/`mean`/`sum`/`dot`/…) renders here at the **ambient resolution**
    /// (`engine::set_resolution`, default [`builtins::RESOLUTION_DEFAULT`]) — the resolution is a
    /// measurement knob, so it applies at the measurement. `signal::sample(sig, n)` remains the
    /// explicit override; a drawn noise realization pins its length at first materialization, so
    /// changing the resolution between uses of one realization errors instead of lying.
    pub(super) fn expect_array(
        &mut self,
        name: &str,
        v: &Value,
        span: Span,
    ) -> Result<Rc<Vec<Value>>> {
        match v {
            Value::Array(xs) => Ok(xs.clone()),
            Value::Signal(s) => {
                let n = self.resolution;
                Ok(Rc::new(self.materialize_sig(&s.clone(), n, span)?))
            }
            other => Err(NoiseError::runtime(
                format!("{name} expects an array, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Fold a symbolic input to its **current** value (PLAN-UNIFORM-INPUTS) — the structural
    /// materialization, against this engine's live input values. Used where a concrete scalar is
    /// required (array sizes, sample counts, loop bounds, indices): the structure depends on the
    /// value, so this legitimately bakes it and the program recompiles on change.
    pub(super) fn force_sym(&self, s: &crate::sym::SymExpr) -> f64 {
        s.force_scalar(&self.inputs.borrow())
    }

    /// Fold a `Value::Sym` to its current `Value::Num`, passing every other value through unchanged
    /// — the P0 fallback where a tunable input meets a context that can't yet carry it symbolically
    /// (a complex value; see [`Self::force_sym`]).
    fn force_sym_value(&self, v: Value) -> Value {
        match v {
            Value::Sym(s) => Value::Num(self.force_sym(&s)),
            other => other,
        }
    }

    /// Resolve a non-negative integer count argument (a `~[shape]` dimension, a `rotation` size).
    pub(super) fn count_arg(&self, name: &str, v: &Value, span: Span) -> Result<usize> {
        let n = match v {
            Value::Num(n) => *n,
            Value::Est { val, .. } => *val,
            // A tunable input used as a count/size/bound is a STRUCTURAL use — fold to its current
            // value (a different size is a genuinely different program, so it recompiles; correct).
            Value::Sym(s) => self.force_sym(s),
            other => {
                return Err(NoiseError::runtime(
                    format!("{name} count must be a number, got {}", other.type_name()),
                    span,
                ))
            }
        };
        if n.fract() != 0.0 || n < 0.0 || !n.is_finite() {
            return Err(NoiseError::runtime(
                format!("{name} count must be a non-negative integer, got {n}"),
                span,
            ));
        }
        Ok(n as usize)
    }
}

/// Coerce a scalar operand to a symbolic-tree leaf: a `Sym` hands over its tree, a plain number
/// promotes to a `Const` leaf. Backs [`Engine::binop_sym`]. A non-scalar (a bool, an array, …) is a
/// spanned type error — the same shape the deterministic evaluator would give for the mixed op.
fn sym_operand(v: &Value, span: Span) -> Result<Rc<crate::sym::SymExpr>> {
    use crate::sym::SymExpr;
    match v {
        Value::Sym(s) => Ok(s.clone()),
        Value::Num(n) => Ok(Rc::new(SymExpr::Const(*n))),
        Value::Est { val, .. } => Ok(Rc::new(SymExpr::Const(*val))),
        // A bool promotes to its 0/1 indicator — the same representation a `Sym` comparison tree
        // folds to — so mixing them (`any([slider > 1, false])`) stays symbolic.
        Value::Bool(b) => Ok(Rc::new(SymExpr::Const(if *b { 1.0 } else { 0.0 }))),
        other => Err(NoiseError::type_mismatch(
            format!("cannot combine a tunable input with {}", other.type_name()),
            span,
        )),
    }
}

/// The teaching error for combining a lazy signal with a value it can't defer against (finding F10,
/// shared by the two `binop_signal` scalar arms): only another signal, a scalar constant, a complex
/// value, or a sized array is legal — anything else must be materialized with `signal::sample`.
fn signal_combine_error(other: &Value, span: Span) -> NoiseError {
    NoiseError::runtime(
        format!(
            "cannot combine a signal with {} — `signal::sample(sig, n)` it to an array first",
            other.type_name()
        ),
        span,
    )
}
