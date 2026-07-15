//! WebAssembly bindings for Noise — the bridge the browser playground runs against.
//!
//! One contract (PLAN-LITERATE §D5): [`run`] parses and evaluates a program and returns **one**
//! `Document` as JSON — meta (frontmatter) + a flat, ordered array of typed blocks (code, notes,
//! plots, inputs) + the comment layer + the result (which carries the input manifest). Every
//! playground view (Preview, plots-only, output-only) is a pure filter over that one array.
//! [`run_with_introspection`] returns the same `Document` next to the hover sidecar (live bindings +
//! per-request plots). [`meta`] reads the frontmatter without running, for the pre-run paper header.

use noise_core::introspect::Summary;
use noise_core::to_flint;
use noise_core::{Engine, InputValue, Value};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Re-exported as `initThreadPool(n)` — the JS host **must** call and await it before running a
/// program, or rayon has no workers and the reduction stays sequential (correct, just single-core).
///
/// This is the whole reason a JS host is involved: WebAssembly has no instruction that spawns a
/// thread. The threads proposal gives wasm shared memory and atomics and explicitly leaves thread
/// *creation* to the embedder, so the workers can only come from `new Worker()` on the JS side.
/// `wasm-bindgen-rayon` generates that glue: it spawns N workers, has each instantiate *this same
/// module* against *this same* `SharedArrayBuffer` memory, and hands the set to rayon as its pool.
#[cfg(feature = "wasm-threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Run options a host may pass: input overrides (`{ "inputs": { name: value, … } }`), a `profile`
/// flag to capture per-phase timings into the document, sample budget/seed later. Absent or empty →
/// each input uses its declared default and profiling is off.
#[derive(Deserialize, Default)]
struct Opts {
    #[serde(default)]
    inputs: serde_json::Map<String, serde_json::Value>,
    /// When `true`, the engine times each forcing's phases and returns them in `result.profile`.
    #[serde(default)]
    profile: bool,
}

/// The engine-facing options parsed out of `opts_json`: the input overrides and whether to profile.
#[derive(Debug, Default, PartialEq)]
struct ParsedOpts {
    overrides: Vec<(String, InputValue)>,
    profile: bool,
}

/// Parse the optional `opts_json` into engine options (input overrides + the profile flag). Returns
/// an error string (surfaced as a failed document) if the JSON or an input value is malformed. Takes
/// `&str` (borrowed) rather than an owned `Option<String>` so the host's JSON isn't copied just to be
/// parsed (finding G4).
fn parse_opts(opts_json: Option<&str>) -> Result<ParsedOpts, String> {
    let json = match opts_json {
        Some(j) if !j.trim().is_empty() => j,
        _ => return Ok(ParsedOpts::default()),
    };
    let opts: Opts = serde_json::from_str(json).map_err(|e| format!("invalid opts JSON: {e}"))?;
    let mut overrides = Vec::new();
    for (name, v) in opts.inputs {
        let iv = match v {
            serde_json::Value::Bool(b) => InputValue::Bool(b),
            serde_json::Value::Number(n) => InputValue::Num(
                n.as_f64()
                    .ok_or_else(|| format!("input `{name}` is not a number"))?,
            ),
            _ => return Err(format!("input `{name}` override must be a number or bool")),
        };
        overrides.push((name, iv));
    }
    Ok(ParsedOpts {
        overrides,
        profile: opts.profile,
    })
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
            // Same error shape core emits (finding D1/D2): span + 1-based line/col + a stable code.
            // A pre-run failure has no source location, so it points at the start of the file.
            "error": { "message": message, "span": { "start": 0, "end": 0 }, "line": 1, "col": 1, "code": "runtime_error" },
            "stats": { "forcings": 0, "samples": 0, "ops": 0, "rng_draws": 0 },
            "profile": null,
            "truncated": null,
            "inputs": [],
        },
    })
}

/// The hand-written last-resort document returned if serializing a real `Document` ever fails. It
/// must stay shape-compatible with [`doc_error`] (asserted by a test) so the host never receives an
/// unreadable payload.
const SERIALIZATION_FALLBACK: &str = r#"{"meta":{"tags":[],"extra":{}},"blocks":[],"comments":[],"result":{"value":null,"error":{"message":"internal serialization error","span":{"start":0,"end":0},"line":1,"col":1,"code":"runtime_error"},"stats":{"forcings":0,"samples":0,"ops":0,"rng_draws":0},"profile":null,"truncated":null,"inputs":[]}}"#;

fn to_json_string(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| SERIALIZATION_FALLBACK.into())
}

/// Turn a caught panic payload into a human-readable message for the `doc_error` document. Keeps the
/// "never throws" contract honest even if some deep code path still `panic!`s / `unreachable!`s.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    let detail = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string());
    format!("internal error (please report): {detail}")
}

/// Run a Noise program → one `Document` (PLAN-LITERATE §D5) as JSON. `opts_json` (optional) carries
/// input overrides. Never throws; a lex/parse/runtime failure comes back as a document with
/// `result.error` set (and whatever blocks ran before the failure). A stray Rust panic (an
/// `unreachable!` we missed) is also caught — via [`std::panic::catch_unwind`] on unwinding targets
/// — and routed into the same `doc_error` shape rather than surfacing as an opaque JS
/// `RuntimeError: unreachable` that poisons the instance (finding A9).
#[wasm_bindgen]
pub fn run(src: &str, opts_json: Option<String>) -> String {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_impl(src, opts_json)))
        .unwrap_or_else(|payload| to_json_string(&doc_error(panic_message(payload))))
}

fn run_impl(src: &str, opts_json: Option<String>) -> String {
    let opts = match parse_opts(opts_json.as_deref()) {
        Ok(o) => o,
        Err(e) => return to_json_string(&doc_error(e)),
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(opts.overrides);
    engine.set_profiling(opts.profile);
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
    /// Validate the host-supplied binding **names** against the engine's live bindings before they
    /// are string-interpolated into source (finding G5). The `vars` are binding names; an unknown
    /// one otherwise produces a baffling downstream *parse/undefined-name* error from the synthesized
    /// call — this returns a clean, direct message instead. (The `given` field is a full condition
    /// expression, not a bare name, so it is left for the evaluator to check.)
    fn validate_names(&self, known: &std::collections::HashSet<&str>) -> Result<(), String> {
        for name in &self.vars {
            if !known.contains(name.as_str()) {
                return Err(format!(
                    "unknown variable `{name}` (not a binding in this program)"
                ));
            }
        }
        Ok(())
    }

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
        PlotOut {
            title: p.title,
            text: p.text,
            charts: p.charts,
            error: None,
        }
    }
}

impl PlotOut {
    fn failed(error: String) -> PlotOut {
        PlotOut {
            title: "—".into(),
            text: error.clone(),
            charts: vec![],
            error: Some(error),
        }
    }
}

/// Run a program → one `Document`, then resolve introspection requests against its retained scope.
/// Returns `{ document, bindings, introspections }`: the `Document` (PLAN-LITERATE §D5), the live
/// top-level bindings (for a variable picker), and one `PlotOut` per request in request order. The
/// sidecar lets the playground show a variable's distribution / relationship / drivers without the
/// source containing a single `describe`/`corr` call.
#[wasm_bindgen]
pub fn run_with_introspection(src: &str, requests_json: &str, opts_json: Option<String>) -> String {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_with_introspection_impl(src, requests_json, opts_json)
    }))
    .unwrap_or_else(|payload| {
        let payload =
            serde_json::json!({ "document": doc_error(panic_message(payload)), "bindings": [], "introspections": [] });
        to_json_string(&payload)
    })
}

fn run_with_introspection_impl(
    src: &str,
    requests_json: &str,
    opts_json: Option<String>,
) -> String {
    let opts = match parse_opts(opts_json.as_deref()) {
        Ok(o) => o,
        Err(e) => {
            let payload = serde_json::json!({ "document": doc_error(e), "bindings": [], "introspections": [] });
            return to_json_string(&payload);
        }
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(opts.overrides);
    engine.set_profiling(opts.profile);
    let document = engine.run_to_document(src);
    let ran_ok = document.result.error.is_none();

    let live = engine.bindings();
    let bindings: Vec<Binding> = live
        .iter()
        .map(|(name, kind)| Binding {
            name: name.clone(),
            kind: kind.to_string(),
        })
        .collect();
    // The set of valid binding names, for validating each request's `vars` before interpolation.
    let known: std::collections::HashSet<&str> = live.iter().map(|(n, _)| n.as_str()).collect();

    let requests: Vec<Request> = serde_json::from_str(requests_json).unwrap_or_default();
    let mut introspections = Vec::with_capacity(requests.len());
    if ran_ok {
        for req in &requests {
            // Validate host-supplied names first (finding G5): an unknown binding gets a clean
            // error rather than a baffling downstream parse error from the synthesized call.
            let out = match req.validate_names(&known) {
                Err(e) => PlotOut::failed(e),
                Ok(()) => match req.to_call() {
                    None => PlotOut::failed("empty request".into()),
                    Some(call) => match engine.run(&call) {
                        Ok(Value::Summary(s)) => PlotOut::from(&*s),
                        Ok(_) => PlotOut::failed("not a random variable".into()),
                        Err(e) => PlotOut::failed(e.to_string()),
                    },
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

#[cfg(test)]
mod tests {
    //! Finding G6: the wasm host had zero tests. These cover the pure, host-facing seams — opts
    //! parsing/clamping, the `Request` protocol, and the `doc_error` JSON contract *including* the
    //! hand-written serialization-failure fallback that must stay shape-compatible with a real
    //! `doc_error`. They run natively (no wasm needed) since every helper here is plain Rust.
    use super::*;

    #[test]
    fn parse_opts_reads_numbers_and_bools() {
        // absent / empty → no overrides, profiling off
        assert_eq!(parse_opts(None).unwrap(), ParsedOpts::default());
        assert_eq!(parse_opts(Some("   ")).unwrap(), ParsedOpts::default());
        // a well-formed opts object → typed overrides
        let got = parse_opts(Some(r#"{"inputs":{"n":6,"flag":true}}"#)).unwrap();
        assert!(got.overrides.contains(&("n".to_string(), InputValue::Num(6.0))));
        assert!(got
            .overrides
            .contains(&("flag".to_string(), InputValue::Bool(true))));
        assert!(!got.profile, "profile defaults off when absent");
    }

    #[test]
    fn parse_opts_reads_the_profile_flag() {
        // `profile: true` opts the run into per-phase timing (surfaced as `result.profile`); the
        // inputs map may be absent alongside it.
        let got = parse_opts(Some(r#"{"profile":true}"#)).unwrap();
        assert!(got.profile);
        assert!(got.overrides.is_empty());
        // still off when explicitly false
        assert!(!parse_opts(Some(r#"{"inputs":{"n":1},"profile":false}"#))
            .unwrap()
            .profile);
    }

    #[test]
    fn parse_opts_rejects_malformed() {
        // invalid JSON
        assert!(parse_opts(Some("{not json")).is_err());
        // an input override that is neither a number nor a bool
        assert!(parse_opts(Some(r#"{"inputs":{"s":"hi"}}"#)).is_err());
    }

    #[test]
    fn request_parses_and_builds_calls() {
        // a bare describe
        let r: Request = serde_json::from_str(r#"{"vars":["x"]}"#).unwrap();
        assert_eq!(r.to_call().as_deref(), Some("describe(x)"));
        // conditioned describe
        let r: Request = serde_json::from_str(r#"{"vars":["x"],"given":"y > 0"}"#).unwrap();
        assert_eq!(r.to_call().as_deref(), Some("describe((x) | (y > 0))"));
        // two-var correlation
        let r: Request = serde_json::from_str(r#"{"vars":["a","b"]}"#).unwrap();
        assert_eq!(r.to_call().as_deref(), Some("corr(a, b)"));
        // explain
        let r: Request = serde_json::from_str(r#"{"vars":["z"],"explain":true}"#).unwrap();
        assert_eq!(r.to_call().as_deref(), Some("explain(z)"));
        // empty request → no call
        let r: Request = serde_json::from_str(r#"{"vars":[]}"#).unwrap();
        assert_eq!(r.to_call(), None);
    }

    #[test]
    fn request_validates_names_against_bindings() {
        let known: std::collections::HashSet<&str> = ["x", "y"].into_iter().collect();
        let ok: Request = serde_json::from_str(r#"{"vars":["x"]}"#).unwrap();
        assert!(ok.validate_names(&known).is_ok());
        let bad: Request = serde_json::from_str(r#"{"vars":["nope"]}"#).unwrap();
        let e = bad.validate_names(&known).unwrap_err();
        assert!(e.contains("nope"), "message was: {e}");
        // `given` is an expression, not a name — it is not validated here.
        let cond: Request = serde_json::from_str(r#"{"vars":["y"],"given":"x > 0"}"#).unwrap();
        assert!(cond.validate_names(&known).is_ok());
    }

    #[test]
    fn doc_error_has_the_stable_shape() {
        let v = doc_error("boom".into());
        assert_eq!(v["result"]["error"]["message"], "boom");
        // The Phase-4 line/col/code fields must be present (the JS host reads them).
        assert_eq!(v["result"]["error"]["line"], 1);
        assert_eq!(v["result"]["error"]["col"], 1);
        assert_eq!(v["result"]["error"]["code"], "runtime_error");
        assert_eq!(v["result"]["error"]["span"]["start"], 0);
        assert!(v["blocks"].is_array());
        assert!(v["result"]["stats"].is_object());
    }

    /// The hand-written [`SERIALIZATION_FALLBACK`] string must parse to the SAME shape as a real
    /// `doc_error`, or a serialization failure would hand the host a document it can't read.
    #[test]
    fn serialization_fallback_matches_doc_error_shape() {
        let real = doc_error("internal serialization error".into());
        let fallback: serde_json::Value = serde_json::from_str(SERIALIZATION_FALLBACK).unwrap();
        let keys = |v: &serde_json::Value| {
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        // Same top-level keys, same result keys, same error sub-shape (message differs, not shape).
        assert_eq!(keys(&fallback), keys(&real));
        assert_eq!(keys(&fallback["result"]), keys(&real["result"]));
        assert_eq!(
            keys(&fallback["result"]["error"]),
            keys(&real["result"]["error"])
        );
        assert_eq!(fallback["result"]["error"]["code"], "runtime_error");
        assert_eq!(fallback, real); // message is identical too, so the whole payload matches
    }
}
