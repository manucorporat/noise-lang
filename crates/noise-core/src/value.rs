//! Runtime values.
//!
//! Phase 0/1 are deterministic: `Num`, `Bool`, `Str`, `Unit`. Phase 2 adds `Dist(RvId)`, a
//! lightweight handle into the engine-owned sample-DAG (`RvGraph`) — a random variable.
//! `Est` is a Monte Carlo *estimate*: a number carrying a standard error, produced by `P`.
//! It behaves like a number in arithmetic (propagating its error) and **displays rounded to
//! the precision its error justifies** — so more samples reveal more digits, and the error
//! correctly inflates through derived expressions like `4 * P(C)`.

use std::fmt;
use std::rc::Rc;

use crate::dist::{Recipe, RvId, RvKind};
use crate::introspect::Summary;
use crate::signal::{NoiseSpec, SignalSpec};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Bool(bool),
    Str(String),
    Unit,
    /// An **undrawn distribution** — a recipe (LANG.md core model §2). `unif(0,1)` yields one.
    /// It is *not* a number: you can bind it (`Die = unif(0,1)`) or draw it (`X ~ Die`), but
    /// arithmetic on it is an error ("draw it with `~` first"). Drawing instantiates fresh
    /// graph node(s) and yields a `Dist`.
    Recipe(Recipe),
    /// A random variable / lazy sample-DAG handle (Phase 2). `PartialEq` compares ids
    /// (structural identity); user-level `==` over RVs lifts in `eval_binary` and never
    /// reaches `Value::eq`.
    Dist(RvId),
    /// A Monte Carlo estimate: `val` with standard error `se` (`se == 0` ⇒ exact). Arithmetic
    /// propagates `se` (first order); `Display` rounds `val` to the digits `se` justifies.
    Est { val: f64, se: f64 },
    /// A fixed-length array (PLAN-COLLECTIONS). Length is known at build time; elements are
    /// arbitrary `Value`s (`Num`, `Dist`, `Bool`, or nested `Array` for matrices). A vector of
    /// random variables is just an array of `Dist`. `Rc` keeps `push`/clone cheap.
    Array(Rc<Vec<Value>>),
    /// A **lazy signal generator** (a continuous waveform described by frequency). It stays
    /// symbolic — O(1) memory — through scalar/trig ops, and materializes to an `Array` only when
    /// combined with a sized array (adopting its length) or via `signal::sample(sig, n)`.
    Signal(Rc<SignalSpec>),
    /// A **lazy noise generator** (`signal::noise_white`/`noise_brown`/`noise_ou`/`noise_pink`):
    /// zero-mean noise of a given spectral color, with no length yet. Like a signal it stays
    /// symbolic until it meets a sized context, but it is *random* — it materializes into fresh
    /// `normal` RV nodes (so `E` can average over realizations), one independent draw stream per
    /// leaf-vector lane. The `NoiseSpec` carries the strength and color.
    Noise(NoiseSpec),
    /// A **complex scalar** (PLAN-COMPLEX). `re`/`im` are each a *real* scalar `Value` —
    /// `Num`/`Est` for a constant complex (`2 + 3i`), or `Dist` for a complex random variable
    /// (e.g. a `rand::normal_complex` draw). Representing the two channels as ordinary real
    /// `Value`s lets complex arithmetic reuse the whole real lifting/folding machinery
    /// (`binop_complex` decomposes `* / ^` into real ops on the channels), so the sample-DAG and
    /// VM stay strictly `f64` — no complex value-channel was added there. A complex value emerges
    /// only from `math::i`/`j` (the unit `0 + 1i`) or a complex distribution; pure-real
    /// expressions stay `Num`. Invariant: `re`/`im` are always real scalars (`Num`/`Est`/`Dist`).
    Complex { re: Box<Value>, im: Box<Value> },
    /// A **conditioned value** — `event | given` (Bayes, scoped to a query). `quantity` is the RV
    /// being measured (an event for `P`, any number for `E`/`Var`/`Q`; its kind is `q_kind`),
    /// `condition` is the bool RV it is conditioned on. The two are kept *separate* (not yet fused)
    /// so operations compose: `2*(X|C)+1` pushes the arithmetic into `quantity` and carries the same
    /// `condition` along — it is `(2X+1) | C`. Two values conditioned on *different* events cannot be
    /// combined (the one rule that keeps conditioning consistent). Like a `Recipe`, a conditioned
    /// value is *consumed* by `P`/`E`/`Var`/`Q` (which fuse it into `select(condition, quantity, NaN)`
    /// and sample the subpopulation where the condition holds), never operated on past that.
    Cond { quantity: RvId, q_kind: RvKind, condition: RvId },
    /// An **introspection summary** — what `describe`/`hist`/`samples`/`corr`/`scatter`/`explain`
    /// evaluate to (see [`crate::introspect`]). It is a *value*: it binds, flows through, and
    /// `Display`s as an ASCII block in the CLI (the playground serializes its `payload` instead).
    /// `Rc` keeps it cheap to clone.
    Summary(Rc<Summary>),
    /// The **`continue` control sentinel** (PLAN-COMPLEX §8). Produced by evaluating `continue`; it
    /// short-circuits the enclosing `{ block }` (the evaluator stops at the statement that yields
    /// it) and signals the surrounding loop to skip — a `for` discards the iteration, a
    /// comprehension omits the element. Not a data value: using it in arithmetic/arrays is a type
    /// error, like any other misuse.
    Continue,
}

impl Value {
    /// Build a complex value from two real-scalar channels (the single constructor, so the
    /// `re`/`im`-are-real-scalars invariant lives in one place).
    pub fn complex(re: Value, im: Value) -> Value {
        Value::Complex { re: Box::new(re), im: Box::new(im) }
    }
    /// A constant complex from two `f64`s — `re + im·i`.
    pub fn cnum(re: f64, im: f64) -> Value {
        Value::complex(Value::Num(re), Value::Num(im))
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Num(_) => "number",
            Value::Bool(_) => "bool",
            Value::Str(_) => "string",
            Value::Unit => "unit",
            Value::Recipe(_) => "distribution",
            Value::Dist(_) => "dist",
            Value::Est { .. } => "number",
            Value::Array(_) => "array",
            Value::Signal(_) => "signal",
            Value::Noise(_) => "noise",
            Value::Complex { .. } => "complex",
            Value::Cond { .. } => "conditioned value",
            Value::Summary(_) => "summary",
            Value::Continue => "continue",
        }
    }
}

/// Format a constant complex `re + im·i` the way a mathematician writes it: `"2 + 3i"`,
/// `"2 - 3i"`, `"3i"`, `"-1i"`, `"0"` for `0 + 0i`, `"2"` for a real-valued `2 + 0i`. Only used
/// for the constant case (both channels `Num`/`Est`); a complex *random variable* prints via the
/// channel `Display` fallback in [`Value::fmt`].
fn format_complex_const(re: f64, im: f64) -> String {
    if im == 0.0 {
        return format_num(re);
    }
    if re == 0.0 {
        return format!("{}i", format_num(im));
    }
    let sign = if im < 0.0 { "-" } else { "+" };
    format!("{} {} {}i", format_num(re), sign, format_num(im.abs()))
}

/// Round `val` to the decimal places justified by standard error `se`: `digits =
/// floor(-log10(se))`. `se <= 0` or non-finite ⇒ exact (return `val` unchanged). This is the
/// single source of truth for "confidence precision".
pub fn round_to_se(val: f64, se: f64) -> f64 {
    if se <= 0.0 || !se.is_finite() {
        return val;
    }
    let digits = (-se.log10()).floor().clamp(0.0, 12.0) as i32;
    let factor = 10f64.powi(digits);
    (val * factor).round() / factor
}

/// Format a deterministic number without floating-point dust: round to 12 decimal places and trim
/// trailing zeros, so `1.0000000000000002` prints `1`, `1.49e-30` prints `0`, and `0.0871` stays
/// `0.0871`. Non-finite values (`inf`/`nan`) print as-is. (Estimates use `round_to_se` instead.)
pub fn format_num(n: f64) -> String {
    if n == 0.0 {
        return "0".to_string(); // also collapses -0
    }
    if !n.is_finite() {
        return format!("{n}");
    }
    let s = format!("{n:.12}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Num(n) => write!(f, "{}", format_num(*n)),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Unit => write!(f, "()"),
            // An undrawn distribution prints as its recipe, e.g. `unif(0, 1)`.
            Value::Recipe(r) => write!(f, "{r}"),
            // No sampling on Display — keep it pure/cheap.
            Value::Dist(id) => write!(f, "<dist #{}>", id.0),
            // Show only the digits the standard error justifies.
            Value::Est { val, se } => write!(f, "{}", round_to_se(*val, *se)),
            // `[a, b, c]` — comma-joined elements via their own Display.
            Value::Array(xs) => {
                write!(f, "[")?;
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, "]")
            }
            Value::Signal(s) => write!(f, "{s}"),
            Value::Noise(spec) => write!(f, "{spec}"),
            // Constant channels render as `2 + 3i`; random channels fall back to `<re> + <im>i`.
            Value::Complex { re, im } => {
                let as_const = |v: &Value| match v {
                    Value::Num(n) => Some(*n),
                    Value::Est { val, .. } => Some(*val),
                    _ => None,
                };
                match (as_const(re), as_const(im)) {
                    (Some(a), Some(b)) => write!(f, "{}", format_complex_const(a, b)),
                    _ => write!(f, "{re} + {im}i"),
                }
            }
            Value::Cond { quantity, condition, .. } => {
                write!(f, "<conditioned #{} | #{}>", quantity.0, condition.0)
            }
            Value::Summary(s) => write!(f, "{s}"),
            Value::Continue => write!(f, "continue"),
        }
    }
}
