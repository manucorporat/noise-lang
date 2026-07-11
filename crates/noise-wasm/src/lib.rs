//! WebAssembly bindings for Noise — the bridge the browser playground runs against.
//!
//! One contract (PLAN-LITERATE §D5): [`run`] parses and evaluates a program and returns **one**
//! `Document` as JSON — meta (frontmatter) + a flat, ordered array of typed blocks (code, notes,
//! plots, inputs) + the comment layer + the result (which carries the input manifest). Every
//! playground view (Preview, plots-only, output-only) is a pure filter over that one array.
//! [`run_with_introspection`] returns the same `Document` next to the hover sidecar (live bindings +
//! per-request plots). [`meta`] reads the frontmatter without running, for the pre-run paper header.

use noise_core::flint::to_flint;
use noise_core::introspect::Summary;
use noise_core::{Engine, InputValue, Value};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Run options a host may pass: input overrides (`{ "inputs": { name: value, … } }`), sample
/// budget/seed later. Absent or empty → each input uses its declared default.
#[derive(Deserialize, Default)]
struct Opts {
    #[serde(default)]
    inputs: serde_json::Map<String, serde_json::Value>,
}

/// Parse the optional `opts_json` into engine input overrides. Returns an error string (surfaced as
/// a failed document) if the JSON or an input value is malformed.
fn parse_opts(opts_json: Option<String>) -> Result<Vec<(String, InputValue)>, String> {
    let json = match opts_json {
        Some(j) if !j.trim().is_empty() => j,
        _ => return Ok(Vec::new()),
    };
    let opts: Opts = serde_json::from_str(&json).map_err(|e| format!("invalid opts JSON: {e}"))?;
    let mut out = Vec::new();
    for (name, v) in opts.inputs {
        let iv = match v {
            serde_json::Value::Bool(b) => InputValue::Bool(b),
            serde_json::Value::Number(n) => {
                InputValue::Num(n.as_f64().ok_or_else(|| format!("input `{name}` is not a number"))?)
            }
            _ => return Err(format!("input `{name}` override must be a number or bool")),
        };
        out.push((name, iv));
    }
    Ok(out)
}

/// A `Document`-shaped error payload for a failure *before* running (bad opts) — same shape hosts
/// always receive, so there is never a second error channel.
fn doc_error(message: String) -> serde_json::Value {
    serde_json::json!({
        "meta": { "tags": [], "extra": {} },
        "blocks": [],
        "comments": [],
        "result": {
            "value": null,
            "error": { "message": message, "span": { "start": 0, "end": 0 } },
            "stats": { "forcings": 0, "samples": 0, "ops": 0, "rng_draws": 0 },
            "truncated": null,
            "inputs": [],
        },
    })
}

fn to_json_string(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| {
        r#"{"meta":{"tags":[],"extra":{}},"blocks":[],"comments":[],"result":{"value":null,"error":{"message":"internal serialization error","span":{"start":0,"end":0}},"stats":{"forcings":0,"samples":0,"ops":0,"rng_draws":0},"truncated":null,"inputs":[]}}"#.into()
    })
}

/// Run a Noise program → one `Document` (PLAN-LITERATE §D5) as JSON. `opts_json` (optional) carries
/// input overrides. Never throws; a lex/parse/runtime failure comes back as a document with
/// `result.error` set (and whatever blocks ran before the failure).
#[wasm_bindgen]
pub fn run(src: &str, opts_json: Option<String>) -> String {
    let overrides = match parse_opts(opts_json) {
        Ok(o) => o,
        Err(e) => return to_json_string(&doc_error(e)),
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(overrides);
    let document = engine.run_to_document(src);
    to_json_string(&document.to_json())
}

/// Read a program's frontmatter *without running it* — the pure entry a host calls to build the
/// pre-run paper header. Returns `{ ok, title, abstract?, tags, extra }` on success, or
/// `{ ok: false, error }` if the fence is malformed. A file with no frontmatter is `ok: true` with a
/// null title and empty tags. Inputs are discovered from the run (`Document.result.inputs`), not
/// here (PLAN-INPUTS §3).
#[wasm_bindgen]
pub fn meta(src: &str) -> String {
    let value = match noise_core::frontmatter::parse(src) {
        Ok(Some((fm, _span))) => {
            let mut m = match fm.to_json() {
                serde_json::Value::Object(m) => m,
                _ => serde_json::Map::new(),
            };
            m.insert("ok".into(), serde_json::json!(true));
            serde_json::Value::Object(m)
        }
        Ok(None) => serde_json::json!({ "ok": true, "tags": [], "extra": {} }),
        Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
    };
    serde_json::to_string(&value)
        .unwrap_or_else(|_| r#"{"ok":false,"error":"internal serialization error"}"#.into())
}

/// The crate version, surfaced in the playground footer ("Noise vX.Y.Z").
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// === variable introspection (the playground "inspect without editing the code" path) ============
//
// The sidecar rests on one fact: `Engine`'s scope persists across `run` calls. We run the user's
// program once (building the `Document` + retained scope), then resolve each introspection *request*
// by evaluating one more expression — `describe(x)` / `corr(a, b)` / `explain(y)` — against the SAME
// engine. Those builtins are exactly the in-source ones, driven by an external request list.

/// A live top-level binding, surfaced so the playground can list what's introspectable and offer
/// only random variables (`kind` starting `dist<…>`) for `describe`/`corr`/`explain`.
#[derive(serde::Serialize)]
struct Binding {
    name: String,
    kind: String,
}

/// One introspection request from the playground.
#[derive(Deserialize)]
struct Request {
    vars: Vec<String>,
    #[serde(default)]
    given: Option<String>,
    #[serde(default)]
    explain: bool,
    #[serde(default)]
    correlate: bool,
}

impl Request {
    fn to_call(&self) -> Option<String> {
        let target = self.vars.first()?;
        let cond = |e: &str| match &self.given {
            Some(g) => format!("({e}) | ({g})"),
            None => e.to_string(),
        };
        Some(if self.explain {
            format!("explain({})", cond(target))
        } else if self.correlate {
            format!("corr({target})")
        } else if self.vars.len() >= 2 {
            format!("corr({}, {})", self.vars[0], self.vars[1])
        } else {
            format!("describe({})", cond(target))
        })
    }
}

/// A plot, as the browser receives it for an introspection request: a heading, a one-line text
/// fallback, and the Flint chart specs to render. `error` is set (with empty `charts`) when a
/// request failed — one bad request never sinks the batch.
#[derive(serde::Serialize)]
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
    fn failed(error: String) -> PlotOut {
        PlotOut { title: "—".into(), text: error.clone(), charts: vec![], error: Some(error) }
    }
}

/// Run a program → one `Document`, then resolve introspection requests against its retained scope.
/// Returns `{ document, bindings, introspections }`: the `Document` (PLAN-LITERATE §D5), the live
/// top-level bindings (for a variable picker), and one `PlotOut` per request in request order. The
/// sidecar lets the playground show a variable's distribution / relationship / drivers without the
/// source containing a single `describe`/`corr` call.
#[wasm_bindgen]
pub fn run_with_introspection(src: &str, requests_json: &str, opts_json: Option<String>) -> String {
    let overrides = match parse_opts(opts_json) {
        Ok(o) => o,
        Err(e) => {
            let payload =
                serde_json::json!({ "document": doc_error(e), "bindings": [], "introspections": [] });
            return to_json_string(&payload);
        }
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(overrides);
    let document = engine.run_to_document(src);
    let ran_ok = document.result.error.is_none();

    let bindings: Vec<Binding> = engine
        .bindings()
        .into_iter()
        .map(|(name, kind)| Binding { name, kind: kind.to_string() })
        .collect();

    let requests: Vec<Request> = serde_json::from_str(requests_json).unwrap_or_default();
    let mut introspections = Vec::with_capacity(requests.len());
    if ran_ok {
        for req in &requests {
            let out = match req.to_call() {
                None => PlotOut::failed("empty request".into()),
                Some(call) => match engine.run(&call) {
                    Ok(Value::Summary(s)) => PlotOut::from(&*s),
                    Ok(_) => PlotOut::failed("not a random variable".into()),
                    Err(e) => PlotOut::failed(e.to_string()),
                },
            };
            engine.take_output(); // discard stray output from a resolved expression
            introspections.push(out);
        }
    }

    let payload = serde_json::json!({
        "document": document.to_json(),
        "bindings": bindings,
        "introspections": introspections,
    });
    to_json_string(&payload)
}
