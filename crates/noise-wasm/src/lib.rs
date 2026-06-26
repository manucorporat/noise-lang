//! WebAssembly bindings for Noise — the bridge the browser playground runs against.
//!
//! One entry point, [`run`], parses and evaluates a Noise program with a fresh [`Engine`] and
//! returns a JSON string the JS side parses into `{ ok, value, output, error, stats }`. `Print`
//! output is captured (the engine buffers it; see `Engine::drain_output`) rather than written to a
//! stdout that does not exist on `wasm32`. `stats` carries the run-time counters the playground
//! shows (samples / operations / random draws); the JS side measures wall-clock to derive ops/sec.

use noise_core::{Engine, RunStats, Value};
use serde::Serialize;
use wasm_bindgen::prelude::*;

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
}

/// Run a Noise program. Always returns a JSON string (never throws); the JS side reads
/// `ok`/`value`/`output`/`error`/`stats`.
#[wasm_bindgen]
pub fn run(src: &str) -> String {
    let mut engine = Engine::new();
    let result = engine.run(src);
    let output = engine.drain_output();
    let stats: Stats = engine.stats().into();
    let payload = match result {
        Ok(Value::Unit) => RunResult { ok: true, value: None, output, error: None, stats },
        Ok(value) => {
            RunResult { ok: true, value: Some(value.to_string()), output, error: None, stats }
        }
        Err(e) => RunResult { ok: false, value: None, output, error: Some(e.to_string()), stats },
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
