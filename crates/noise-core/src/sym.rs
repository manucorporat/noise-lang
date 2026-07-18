//! Symbolic scalar values — a host `input::real` that stays symbolic until its context forces it
//! one way or the other (PLAN-UNIFORM-INPUTS).
//!
//! An `input::real` evaluates to a [`Value::Sym`](crate::value::Value::Sym) carrying a [`SymExpr`]:
//! a lazy `Rc`-DAG over an `Input` leaf, exactly the [`SigExpr`](crate::signal::SigExpr) playbook one
//! dimension simpler (a scalar, not a per-index window). Arithmetic keeps it symbolic; it then
//! materializes **two ways**:
//!
//!   * **`force_scalar`** — when the *structure* needs a concrete value (an array size, a sample
//!     count, a loop bound, an index): fold the tree with the current input values. The structure
//!     then legitimately depends on the value → a **structural** input, which recompiles on change
//!     (correct — a different size is a genuinely different program).
//!   * **lowering to `RvNode::Input` nodes** (in `eval`) — when the `Sym` enters the RV graph as a
//!     *value* (a threshold, an arithmetic operand, a distribution parameter): the value is never
//!     baked, so the compiled kernel is stable across input values → a **value** input, the uniform
//!     that makes a slider drag re-dispatch without recompiling.
//!
//! Only `input::real` becomes a `Sym`; `input::int`/`bool` stay concrete (they are structural /
//! control and recompiling on their change is correct). See the plan for the full rationale.

use std::rc::Rc;

use crate::ast::{BinOp, UnOp};

/// A lazy symbolic scalar: an expression tree over host-input leaves and promoted constants. `Rc`
/// keeps the builder cheap; [`Value::Sym`](crate::value::Value::Sym) holds `Rc<SymExpr>`. It stays
/// symbolic through arithmetic and either folds to `f64` ([`force_scalar`](SymExpr::force_scalar)) or
/// lowers to `RvNode::Input` graph nodes (in `eval`), mirroring `SigExpr`'s two-mode materialization.
#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr {
    /// A host input slot — the leaf that must NOT fold into the compiled kernel. `idx` is the
    /// input's declaration-order slot (its position in the run's input manifest / the runtime
    /// `input_values` slice).
    Input(u32),
    /// A promoted scalar constant (`barrier * 2` carries a `Const(2.0)`).
    Const(f64),
    /// A deferred unary op.
    Unary(UnOp, Rc<SymExpr>),
    /// A deferred binary op — `Sym ∘ Sym` or `Sym ∘ const`.
    Binary(BinOp, Rc<SymExpr>, Rc<SymExpr>),
}

impl SymExpr {
    /// A leaf reading input slot `idx`.
    pub fn input(idx: u32) -> Rc<Self> {
        Rc::new(SymExpr::Input(idx))
    }

    /// Defer a unary op, staying symbolic. Every real-valued math ufunc (`sqrt`/`sin`/`exp`/…)
    /// routes its `Sym` arm through here, so a slider keeps its **value**-input status (no
    /// recompile on drag) instead of folding — the whole point of `Sym` (PLAN-UNIFORM-INPUTS).
    pub fn unary(op: UnOp, a: Rc<Self>) -> Rc<Self> {
        Rc::new(SymExpr::Unary(op, a))
    }

    /// Fold this tree to a single `f64` against the current input `values` (`values[idx]` for an
    /// `Input` leaf). This is the **structural** materialization — the caller uses the concrete value
    /// as an array size / sample count / loop bound / index, so the structure depends on it and the
    /// program legitimately recompiles when the value changes. An out-of-range slot (a value not yet
    /// resolved) reads `0.0` — defensive; in practice an input is declared before it is used.
    pub fn force_scalar(&self, values: &[f64]) -> f64 {
        match self {
            SymExpr::Input(idx) => values.get(*idx as usize).copied().unwrap_or(0.0),
            SymExpr::Const(c) => *c,
            SymExpr::Unary(op, a) => crate::eval::apply_unop_f64(*op, a.force_scalar(values)),
            SymExpr::Binary(op, a, b) => {
                crate::num::fold_binop(*op, a.force_scalar(values), b.force_scalar(values))
            }
        }
    }
}
