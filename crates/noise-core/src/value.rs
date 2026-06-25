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

use crate::dist::{Recipe, RvId};
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
        }
    }
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
        }
    }
}
