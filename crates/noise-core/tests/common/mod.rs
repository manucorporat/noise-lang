//! Shared helpers for the noise-core integration test suite.
//!
//! These tests were relocated out of `lib.rs`'s giant in-crate `mod tests` (finding E3): as
//! integration tests they compile against ONLY the exported crate surface, which is what keeps the
//! public API honest (finding E1). Helpers that used to be free functions inside `mod tests` live
//! here and are shared via `mod common;` from each themed test file.
//!
//! `#![allow(dead_code, unused_imports)]` because each themed test binary pulls in this whole
//! module but uses only the helpers and re-exports it needs.
#![allow(dead_code, unused_imports)]

pub use noise_core::introspect::{Dist1, Payload, Summary, View};
pub use noise_core::{
    Emission, Engine, InputKind, InputSpec, InputValue, Moments, NoiseError, Output, ResolvedInput,
    Result, RunStats, Value,
};

/// The distribution/math/collections/signal modules are strictly scoped (a `use` or a `mod::name`
/// path is required). Most of the corpus predates modules and uses bare names, so [`run`] runs
/// programs with all four non-default modules pre-`use`d. Tests that need the *raw* strict
/// behaviour (the module-system tests) call [`run_raw`] instead.
pub fn with_prelude(src: &str) -> String {
    format!("use rand; use math; use vec; use signal;\n{src}")
}
pub fn run(src: &str) -> Result<Value> {
    noise_core::run(&with_prelude(src))
}
/// Run a program with NO prelude — bare `rand`/`math`/`vec` names are out of scope.
pub fn run_raw(src: &str) -> Result<Value> {
    noise_core::run(src)
}

pub fn num(src: &str) -> f64 {
    match run(src).unwrap() {
        Value::Num(n) => n,
        // An estimate's central value (used by assertions that check accuracy/tolerance;
        // display precision is checked separately via `display_of`).
        Value::Est { val, .. } => val,
        other => panic!("expected number, got {other:?} for {src:?}"),
    }
}
/// The user-visible string a program prints (e.g. an estimate rounded to its precision).
pub fn display_of(src: &str) -> String {
    run(src).unwrap().to_string()
}
pub fn boolean(src: &str) -> bool {
    match run(src).unwrap() {
        Value::Bool(b) => b,
        other => panic!("expected bool, got {other:?} for {src:?}"),
    }
}

pub fn run_num(src: &str) -> f64 {
    num(src)
}

pub fn moments_of(src: &str, n: usize, seed: u64) -> Moments {
    let mut eng = Engine::new();
    let rv = eng.run_rv(&with_prelude(src)).unwrap();
    eng.moments(&rv, n, seed).unwrap()
}

pub fn draws_of(src: &str, n: usize, seed: u64) -> Vec<f64> {
    let mut eng = Engine::new();
    let rv = eng.run_rv(&with_prelude(src)).unwrap();
    eng.sample(&rv, n, seed).unwrap()
}

pub fn graph_len(src: &str) -> usize {
    let mut eng = Engine::new();
    eng.run(&with_prelude(src)).unwrap();
    eng.graph().len()
}

pub fn string_of(src: &str) -> String {
    match run(src).unwrap() {
        Value::Str(s) => s,
        other => panic!("expected string, got {other:?} for {src:?}"),
    }
}

pub fn as_num(v: Value) -> f64 {
    match v {
        Value::Num(n) => n,
        Value::Est { val, .. } => val,
        other => panic!("expected number, got {other:?}"),
    }
}

/// Pull `(re, im)` out of a constant `Value::Complex` (or promote a real scalar).
pub fn complex_of(src: &str) -> (f64, f64) {
    match run(src).unwrap() {
        Value::Complex { re, im } => (as_num(*re), as_num(*im)),
        Value::Num(n) => (n, 0.0),
        other => panic!("expected complex, got {other:?} for {src:?}"),
    }
}

/// Run a program whose last expression is an introspection and return its summary.
pub fn summary_of(src: &str) -> std::rc::Rc<Summary> {
    match run(src).unwrap() {
        Value::Summary(s) => s,
        other => panic!("expected a summary, got {other:?} for {src:?}"),
    }
}
pub fn one(src: &str) -> Dist1 {
    match &summary_of(src).payload {
        Payload::One(d) => d.clone(),
        other => panic!("expected a one-variable summary, got {other:?}"),
    }
}

pub fn fan_plot_of(src: &str) -> std::rc::Rc<Summary> {
    let mut eng = Engine::new();
    eng.run(&with_prelude(src)).unwrap();
    for o in eng.take_output() {
        if let Output::Plot(s) = o.output {
            if matches!(s.payload, Payload::Fan(_)) {
                return s;
            }
        }
    }
    panic!("expected a fan plot in the output stream for {src:?}");
}

pub fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(xs) => xs
            .iter()
            .map(|x| match x {
                Value::Num(n) => *n,
                other => panic!("expected a number, got {other:?}"),
            })
            .collect(),
        other => panic!("expected an array, got {other:?}"),
    }
}

/// The rows of a matrix value.
pub fn rows(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(rs) => rs.iter().map(nums).collect(),
        other => panic!("expected a matrix, got {other:?}"),
    }
}

/// Run a program, then hand back its engine so introspection can resolve against the retained scope
/// (the same trick the playground sidecar uses) — so a `plot::` call and its `stats::` twin see the
/// identical graph and roots.
pub fn engine_after(src: &str) -> Engine {
    let mut eng = Engine::new();
    eng.run(&with_prelude(src)).unwrap();
    eng
}

pub fn plot_payload(eng: &mut Engine) -> Payload {
    for o in eng.take_output() {
        if let Output::Plot(s) = &o.output {
            return s.payload.clone();
        }
    }
    panic!("expected a plot in the output stream");
}
