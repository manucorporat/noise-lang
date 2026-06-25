//! WebAssembly bindings for Noise — the bridge the browser playground runs against.
//!
//! One entry point, [`run`], parses and evaluates a Noise program with a fresh [`Engine`] and
//! returns a JSON string the JS side parses into `{ ok, value, output, error }`. `Print` output
//! is captured (the engine buffers it; see `Engine::drain_output`) rather than written to a
//! stdout that does not exist on `wasm32`.

use noise_core::{Engine, Value};
use serde::Serialize;
use wasm_bindgen::prelude::*;

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
}

/// Run a Noise program. Always returns a JSON string (never throws); the JS side reads
/// `ok`/`value`/`output`/`error`.
#[wasm_bindgen]
pub fn run(src: &str) -> String {
    let mut engine = Engine::new();
    let result = engine.run(src);
    let output = engine.drain_output();
    let payload = match result {
        Ok(Value::Unit) => RunResult { ok: true, value: None, output, error: None },
        Ok(value) => RunResult { ok: true, value: Some(value.to_string()), output, error: None },
        Err(e) => RunResult { ok: false, value: None, output, error: Some(e.to_string()) },
    };
    // Serialization of this fixed, string-only struct cannot fail; fall back defensively anyway.
    serde_json::to_string(&payload).unwrap_or_else(|_| {
        r#"{"ok":false,"value":null,"output":"","error":"internal serialization error"}"#.into()
    })
}

/// The crate version, surfaced in the playground footer ("Noise vX.Y.Z").
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
