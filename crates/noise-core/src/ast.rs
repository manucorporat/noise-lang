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
    // Real exponential / natural log ufuncs (`math::exp`/`math::log`, PLAN-FINANCE F1) — the
    // lognormal/Kelly unlock. `Exp` lowers to a library-`exp` call in both code generators
    // (finding C9 — the old `pow(e, x)` could differ in the last bit from `f64::exp`); `Ln` reuses
    // the inlined `approx::ln` polynomial behind a full-domain guard (x > 0 → poly, 0 → -inf,
    // < 0 → NaN, ±inf/NaN propagate) so it matches `f64::ln` semantics.
    Exp,
    Ln,
    // IEEE square root (`math::sqrt` of an RV, `vec::norm`, complex `abs`). Its own node — NOT
    // `Pow(x, 0.5)` — because both Cranelift and wasm have a single correctly-rounded `f64.sqrt`
    // instruction, so `Sqrt` is *fusible* in the codegen cost model where non-integer `pow` is a
    // libcall (PLAN-PERF-2 §5). Semantics are `f64::sqrt`, which differs from `powf(x, 0.5)` at
    // exactly two inputs: `sqrt(-0.0) = -0.0` vs `powf(-0.0, 0.5) = +0.0`, and `sqrt(-inf) = NaN`
    // vs `powf(-inf, 0.5) = +inf` (C99 pow). Every backend implements it as native sqrt, so all
    // three stay bit-identical.
    Sqrt,
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
    /// `f(args...)` — call. Resolves to a user function first, then a builtin. Arguments are
    /// **either all positional or all named, never mixed** (see [`CallArgs`]).
    Call(String, CallArgs),
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
    /// A fenced **template** (PLAN-LITERATE §D3): raw multi-line text with `${expr}` interpolation.
    /// Evaluates to a `string` (each hole rendered via its display form, like `Print`). At **root
    /// statement position** it emits an `Output::Note` instead of becoming the program value; nested
    /// anywhere else it is just a string value. `syntax` is the triple-fence info tag (e.g. `md`),
    /// carried so a host can render the note as markdown vs preformatted text.
    Template {
        parts: Vec<TemplatePart>,
        syntax: Option<String>,
    },
    /// `continue` — skip the rest of the enclosing loop body (PLAN-COMPLEX §8). Evaluating it
    /// short-circuits the current `{ block }` (later statements don't run); a `for` loop discards
    /// that iteration's side effects, and a comprehension *omits* that element. This is how a
    /// comprehension expresses a filter (`if bad(x) { continue }; f(x)`) without special syntax.
    Continue,
}

/// A call's argument list. A call is **either all positional or all named — never mixed**
/// (`f(x, y)` or `f(a: x, b: y)`, but not `f(x, b: y)`). The two forms are disjoint at the AST
/// level so positional calls — the overwhelming majority — stay a plain `Vec<Spanned>` on the hot
/// path, exactly as before named arguments existed. Named args bind to parameters *by name* at
/// eval time (a thin name→slot reorder layered before argument evaluation). See PLAN-INPUTS §2.
#[derive(Debug, Clone, PartialEq)]
pub enum CallArgs {
    /// `f(a, b)` — arguments in parameter order.
    Positional(Vec<Spanned>),
    /// `f(x: a, y: b)` — `(name, value)` pairs in any order; each parameter named at most once.
    Named(Vec<(String, Spanned)>),
}

impl CallArgs {
    /// The number of arguments, regardless of form.
    pub fn len(&self) -> usize {
        match self {
            CallArgs::Positional(a) => a.len(),
            CallArgs::Named(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One segment of a [`Template`](Expr::Template): literal text, or an interpolation hole. Holes are
/// full expressions carrying their **original source span** (so an error inside `${…}` points at the
/// real byte location, not a re-based one).
#[derive(Debug, Clone, PartialEq)]
pub enum TemplatePart {
    Lit(String),
    Hole(Spanned),
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
