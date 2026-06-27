//! Abstract syntax tree. A program is a list of statements; everything is an expression
//! (see GOAL.md "everything is an expression"). Spans are attached for diagnostics.

use crate::error::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod, // % — floored modulo: a − b·floor(a/b)
    Pow, // ^ — exponentiation (right-associative)
    Eq,  // ==
    Ne,  // !=
    Lt,
    Gt,
    Le,
    Ge,
    And, // &&
    Or,  // ||
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    // Floor / ceiling (`math::floor`/`math::ceil` ufuncs, PLAN-COMPLEX §8). Also the building block
    // the `%` operator desugars to (`a − b·floor(a/b)`). Native in every backend (cranelift/wasm
    // both have a floor/ceil instruction; the interpreter uses `f64::floor`/`ceil`).
    Floor,
    Ceil,
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
    /// `~expr` / `~[n, m, …] expr` — the prefix **draw** operator (LANG.md §2: `~` is the only
    /// thing that draws). With an empty shape it draws a single sample of the recipe `expr`; with
    /// a shape it builds a nested array of that shape, each leaf an *independent* draw. A
    /// non-recipe operand is repeated as-is. The statement form `x ~ rhs` is sugar for
    /// `x = ~rhs`; `x ~[8] rhs` for `x = ~[8] rhs`.
    Sample {
        shape: Vec<Spanned>,
        body: Box<Spanned>,
    },
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
    /// `[for var in iter { body }]` — a comprehension (PLAN-COMPLEX §8). Exactly the
    /// `for var in iter { body }` loop statement wrapped in `[ ]` so each `body` value is
    /// *collected* into a new array (rather than discarded). Build-time unrolled: `var` binds to
    /// each element in the *current* frame (so the body closes over outer variables — no closures
    /// needed). A pure 1-to-1 map: the result always has `Len(iter)` elements.
    Comprehension {
        body: Box<Spanned>,
        var: String,
        iter: Box<Spanned>,
    },
    /// `a @ b` — the **matrix product** (Python/NumPy `@`). Dispatches on operand shape at build
    /// time: vector·vector → scalar dot, matrix·vector → matrix–vector product, matrix·matrix →
    /// matrix–matrix product. Lowers to sums of `*` (so it lifts over random variables like any
    /// arithmetic). Distinct from `*`, which stays elementwise/broadcast.
    MatMul(Box<Spanned>, Box<Spanned>),
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
    /// `event | given` — a **conditioning** expression (`P(A | C)`, `E(X | C)`, …). It is *only*
    /// meaningful as the first argument of a query (`P`/`E`/`Var`/`Q`): it restricts that single
    /// query to the worlds where `given` holds (Bayes' rule, scoped to the query — no side effect,
    /// no global state). `given` must be an event (bool); `event` is the quantity being measured.
    /// Evaluated in any other position it is a spanned error.
    Cond {
        event: Box<Spanned>,
        given: Box<Spanned>,
    },
    /// `continue` — skip the rest of the enclosing loop body (PLAN-COMPLEX §8). Evaluating it
    /// short-circuits the current `{ block }` (later statements don't run); a `for` loop discards
    /// that iteration's side effects, and a comprehension *omits* that element. This is how a
    /// comprehension expresses a filter (`if bad(x) { continue }; f(x)`) without special syntax.
    Continue,
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
