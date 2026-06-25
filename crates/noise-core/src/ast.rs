//! Abstract syntax tree. A program is a list of statements; everything is an expression
//! (see GOAL.md "everything is an expression"). Spans are attached for diagnostics.

use crate::error::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow, // **
    Eq,  // ==
    Ne,  // !=
    Lt,
    Gt,
    Le,
    Ge,
    And, // &&
    Or,  // ||
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg, // -x
    Not, // !x
    // Math ufuncs (created by the `sin`/`cos`/`atan`/`sign` builtins, never parsed as prefix
    // operators). They lift over random variables and map over arrays like any other unary node.
    Sin,
    Cos,
    Atan,
    Sign, // -1 / 0 / +1
    // Round-to-nearest-integer. Internal-only: synthesized by the `_int` distribution recipes
    // (`normal_int`/`exp_int`), which draw a continuous source then round each lane. Not parsed
    // as a surface operator/ufunc.
    Round,
}

/// `=` binds a deterministic value; `~` binds a random variable / distribution.
/// In Phase 0 both behave deterministically; `~` gains distribution semantics in Phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindKind {
    Assign, // =
    Sample, // ~
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    Str(String),
    Ident(String),
    Unary(UnOp, Box<Spanned>),
    Binary(BinOp, Box<Spanned>, Box<Spanned>),
    /// `name <bind> value` — an assignment/sample binding, itself an expression.
    Bind(BindKind, String, Box<Spanned>),
    /// `f(params) = body` (deterministic) or `f(params) ~ body` (stochastic — each call draws).
    /// Defines a user function; evaluates to `unit`. See LANG.md core model §4.
    FnDef {
        kind: BindKind,
        name: String,
        params: Vec<String>,
        body: Box<Spanned>,
    },
    /// `f(args...)` — call. Resolves to a user function first, then a builtin.
    Call(String, Vec<Spanned>),
    /// `{ stmts... }` — a block; evaluates to its last statement's value.
    Block(Vec<Spanned>),
    /// `if cond { .. } else { .. }` — else is optional.
    If(Box<Spanned>, Box<Spanned>, Option<Box<Spanned>>),
    /// `[a, b, c]` — an array literal (fixed length, known at build time). See PLAN-COLLECTIONS.
    Array(Vec<Spanned>),
    /// `a..b` — a half-open integer range (Rust-style): the array `[a, a+1, …, b-1]`. Bounds must
    /// be deterministic integers; `a >= b` is the empty array. Replaces the old `range` builtin.
    Range(Box<Spanned>, Box<Spanned>),
    /// `xs[i]` — indexing. `i` must resolve to a constant integer in range (no random gather).
    Index(Box<Spanned>, Box<Spanned>),
    /// `for x in xs { body }` — build-time unroll: bind `x` to each element of `xs` (in the
    /// *current* frame, so accumulators leak across iterations) and evaluate `body`. Yields unit.
    For {
        var: String,
        iter: Box<Spanned>,
        body: Box<Spanned>,
    },
    /// `use module;` — bring a module's items into unqualified scope (Rust-style). Evaluates to
    /// unit. `builtin` is always active; `rand`/`math`/`vec` need a `use` (or a `mod::name` path).
    Use(String),
}

/// A qualified or bare name (`rand::unif`, `pi`). Modules are single-level for now. Qualified
/// names ride inside the existing `Ident`/`Call` name string with a `::` separator; this splits
/// one back into `(Some(module), base)` or `(None, base)`.
pub fn split_path(name: &str) -> (Option<&str>, &str) {
    match name.split_once("::") {
        Some((module, base)) => (Some(module), base),
        None => (None, name),
    }
}

/// An expression paired with its source span.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned {
    pub expr: Expr,
    pub span: Span,
}

impl Spanned {
    pub fn new(expr: Expr, span: Span) -> Self {
        Spanned { expr, span }
    }
}

/// A whole program: a sequence of statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub stmts: Vec<Spanned>,
}
