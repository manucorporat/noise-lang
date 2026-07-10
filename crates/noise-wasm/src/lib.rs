//! WebAssembly bindings for Noise — the bridge the browser playground runs against.
//!
//! One entry point, [`run`], parses and evaluates a Noise program with a fresh [`Engine`] and
//! returns a JSON string the JS side parses into `{ ok, value, output, error, stats }`. `Print`
//! output is captured (the engine buffers it; see `Engine::drain_output`) rather than written to a
//! stdout that does not exist on `wasm32`. `stats` carries the run-time counters the playground
//! shows (samples / operations / random draws); the JS side measures wall-clock to derive ops/sec.

use noise_core::flint::to_flint;
use noise_core::introspect::Summary;
use noise_core::{Engine, Output, RunStats, Value};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// One entry in the program's output stream, tagged for the JS renderer: a `Print` line (`text`) or
/// a `plot::*` chart (`plot`). Keeping these in one ordered list is what lets the playground show
/// text and charts interleaved exactly as the program emitted them.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum LogItem {
    Text { text: String },
    Plot(PlotOut),
}

/// Split an engine output stream into the text-only log (for simple demos that want a plain string)
/// and the ordered, tagged `log` (for the playground's interleaved text+chart rendering).
fn split_output(items: Vec<Output>) -> (String, Vec<LogItem>) {
    let mut text = String::new();
    let mut log = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Output::Text(line) => {
                text.push_str(&line);
                text.push('\n');
                log.push(LogItem::Text { text: line });
            }
            Output::Plot(s) => log.push(LogItem::Plot(PlotOut::from(&*s))),
        }
    }
    (text, log)
}

/// Run-time counters surfaced in the playground's "engine" readout. Mirrors [`RunStats`]; the JS
/// side pairs these with a wall-clock time it measures around [`run`] to show ops/second.
#[derive(Serialize)]
struct Stats {
    /// Number of forcing operations (`P`/`E`/`Var`/`Q`/`sample`) the program ran.
    forcings: u64,
    /// Total Monte-Carlo draws across all forcings.
    samples: u64,
    /// Total per-lane operations executed (Σ draws × cone-node-count).
    ops: u64,
    /// Total random source draws (Σ draws × source-node-count).
    rng_draws: u64,
}

impl From<RunStats> for Stats {
    fn from(s: RunStats) -> Self {
        Stats { forcings: s.forcings, samples: s.samples, ops: s.ops, rng_draws: s.rng_draws }
    }
}

/// The result of running a program, serialized to JSON for the JS playground.
#[derive(Serialize)]
struct RunResult {
    /// `true` if the program evaluated without error.
    ok: bool,
    /// The display form of the last statement's value, or `null` for `unit` / on error.
    value: Option<String>,
    /// Everything `Print` emitted, in source order (may be present even on error).
    output: String,
    /// The error message (with source span), or `null` on success.
    error: Option<String>,
    /// Run-time counters for the engine readout (partial work still counts on error).
    stats: Stats,
    /// The output stream in source order: `Print` lines and `plot::*` charts, interleaved.
    log: Vec<LogItem>,
}

/// Run a Noise program. Always returns a JSON string (never throws); the JS side reads
/// `ok`/`value`/`output`/`error`/`stats`/`log`.
#[wasm_bindgen]
pub fn run(src: &str) -> String {
    let mut engine = Engine::new();
    let result = engine.run(src);
    let (output, log) = split_output(engine.take_output());
    let stats: Stats = engine.stats().into();
    let payload = match result {
        Ok(Value::Unit) => RunResult { ok: true, value: None, output, error: None, stats, log },
        Ok(value) => {
            RunResult { ok: true, value: Some(value.to_string()), output, error: None, stats, log }
        }
        Err(e) => RunResult { ok: false, value: None, output, error: Some(e.to_string()), stats, log },
    };
    // Serialization of this fixed, string-only struct cannot fail; fall back defensively anyway.
    serde_json::to_string(&payload).unwrap_or_else(|_| {
        r#"{"ok":false,"value":null,"output":"","error":"internal serialization error","log":[]}"#.into()
    })
}

/// The crate version, surfaced in the playground footer ("Noise vX.Y.Z").
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// === variable introspection (the playground "inspect without editing the code" path) ============
//
// The whole sidecar rests on one fact: `Engine`'s scope persists across `run` calls. So we run the
// user's program once (building the graph + scope), then resolve each introspection *request* by
// evaluating one more expression — `describe(x)` / `corr(a, b)` / `explain(y)` — against the SAME
// engine. Those builtins are exactly the in-source ones; here they're driven by an external request
// list instead of source text, so a variable can be inspected without touching the program.

/// A live top-level binding, surfaced so the playground can list what's introspectable and offer
/// only random variables (`kind` starting `dist<…>`) for `describe`/`corr`/`explain`.
#[derive(Serialize)]
struct Binding {
    name: String,
    kind: String,
}

/// One introspection request from the playground. `vars` holds one expression (a `describe`/
/// `explain` target) or two (`corr`). `given` is an optional condition expression; `explain` flips
/// the one-variable case to the driver fan-out. Targets/conditions are full Noise expressions
/// evaluated in the program's scope — so the request can reference `X + Y` or a fresh condition the
/// source never names.
#[derive(Deserialize)]
struct Request {
    vars: Vec<String>,
    #[serde(default)]
    given: Option<String>,
    #[serde(default)]
    explain: bool,
    /// With one (array) variable: the element×element correlation heatmap (`corr(vec)`).
    #[serde(default)]
    correlate: bool,
}

impl Request {
    /// Lower a request to the Noise expression we evaluate in the retained scope. Arity picks the
    /// operation; `given` wraps the target in a conditioning bar (parenthesized so precedence holds).
    /// `describe` is polymorphic on the target's type (scalar/vector/matrix/dist), so one mapping
    /// covers value cards, histograms, series, and heatmaps.
    fn to_call(&self) -> Option<String> {
        let target = self.vars.first()?;
        let cond = |e: &str| match &self.given {
            Some(g) => format!("({e}) | ({g})"),
            None => e.to_string(),
        };
        Some(if self.explain {
            format!("explain({})", cond(target))
        } else if self.correlate {
            format!("corr({target})") // one array → correlation matrix
        } else if self.vars.len() >= 2 {
            // `corr` doesn't take a condition yet; the front-end keeps `given` off two-var requests.
            format!("corr({}, {})", self.vars[0], self.vars[1])
        } else {
            format!("describe({})", cond(target))
        })
    }
}

/// A plot, as the browser receives it: a heading, a one-line text fallback, and the Flint
/// `ChartAssemblyInput` specs to render. There is exactly **one** plot shape — no per-payload union
/// — because the engine already decided which chart a histogram, a fan, or a correlation matrix is
/// (see `noise_core::flint`). The renderer compiles `charts` with a stock Flint backend and, when
/// there is more than one, layers them (that is the fan: two bands + a median line).
///
/// `error` is set instead of `charts` when an introspection *request* failed — one bad request from
/// the sidecar never sinks the batch.
#[derive(Serialize)]
struct PlotOut {
    title: String,
    text: String,
    charts: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl From<&Summary> for PlotOut {
    fn from(s: &Summary) -> PlotOut {
        let p = to_flint(s);
        PlotOut { title: p.title, text: p.text, charts: p.charts, error: None }
    }
}

impl PlotOut {
    /// A failed introspection request, rendered as a card that says why.
    fn failed(error: String) -> PlotOut {
        PlotOut { title: "—".into(), text: error.clone(), charts: vec![], error: Some(error) }
    }
}

/// The run result plus the live bindings and the resolved introspections.
#[derive(Serialize)]
struct IntrospectResult {
    ok: bool,
    value: Option<String>,
    output: String,
    error: Option<String>,
    stats: Stats,
    /// The program's live top-level variables (name + kind), for the picker.
    bindings: Vec<Binding>,
    /// One entry per request, in request order.
    introspections: Vec<PlotOut>,
    /// The output stream in source order: `Print` lines and `plot::*` charts, interleaved.
    log: Vec<LogItem>,
}

/// Run a program, then resolve a list of introspection requests against its (retained) scope.
/// `requests_json` is a JSON array of [`Request`]. Always returns a JSON string. The bindings and
/// per-request results let the playground show a variable's distribution, two variables' relationship,
/// or what drives a variable — all without the source containing a single `describe`/`corr` call.
#[wasm_bindgen]
pub fn run_with_introspection(src: &str, requests_json: &str) -> String {
    let mut engine = Engine::new();
    let result = engine.run(src);
    // Capture the program's own output stream (Print + plot::*) now — before the follow-up runs.
    let (output, log) = split_output(engine.take_output());
    let stats: Stats = engine.stats().into();
    let bindings: Vec<Binding> =
        engine.bindings().into_iter().map(|(name, kind)| Binding { name, kind: kind.to_string() }).collect();

    let requests: Vec<Request> = serde_json::from_str(requests_json).unwrap_or_default();
    let mut introspections = Vec::with_capacity(requests.len());
    // Only resolve requests if the program itself ran (a failed program has no scope to inspect).
    if result.is_ok() {
        for req in &requests {
            let out = match req.to_call() {
                None => PlotOut::failed("empty request".into()),
                Some(call) => match engine.run(&call) {
                    Ok(Value::Summary(s)) => PlotOut::from(&*s),
                    Ok(_) => PlotOut::failed("not a random variable".into()),
                    Err(e) => PlotOut::failed(e.to_string()),
                },
            };
            engine.take_output(); // discard any stray output from a resolved expression
            introspections.push(out);
        }
    }

    let payload = match result {
        Ok(Value::Unit) => IntrospectResult {
            ok: true,
            value: None,
            output,
            error: None,
            stats,
            bindings,
            introspections,
            log,
        },
        Ok(value) => IntrospectResult {
            ok: true,
            value: Some(value.to_string()),
            output,
            error: None,
            stats,
            bindings,
            introspections,
            log,
        },
        Err(e) => IntrospectResult {
            ok: false,
            value: None,
            output,
            error: Some(e.to_string()),
            stats,
            bindings,
            introspections,
            log,
        },
    };
    serde_json::to_string(&payload).unwrap_or_else(|_| {
        r#"{"ok":false,"value":null,"output":"","error":"internal serialization error","bindings":[],"introspections":[],"log":[]}"#.into()
    })
}
