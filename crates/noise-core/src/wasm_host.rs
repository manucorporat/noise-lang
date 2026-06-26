//! Browser host seam for the WASM emitter (PLAN.md "Browser note") — wasm32-only.
//!
//! [`crate::wasm_emit`] turns a graph into a WebAssembly module (`Vec<u8>`); this module is what
//! *runs* it in the browser. A wasm sandbox can't run a child module by itself — only the JS host
//! can `WebAssembly.instantiate` — so the kernel is driven through a tiny inline-JS shim
//! (`nz_kernel_*`): compile+instantiate once, seed the xoshiro state into the child's own linear
//! memory, then run a batch per call, copying only the output column back across the boundary.
//!
//! It plugs into the exact same [`Backend`]/[`Program`]/[`Runner`] seam the interpreter and native
//! JIT use, so on `wasm32` [`crate::backend::compile_root`] transparently routes profitable graphs
//! here (and falls back to the interpreter for everything the gate rejects, or if instantiation
//! fails — e.g. the main-thread sync-compile size limit). Correctness is never at stake, only speed.
//!
//! Single-runner assumption: `reduce::run_reduction` is sequential on `wasm32` (its threaded path is
//! `#[cfg(not(target_arch = "wasm32"))]`), so exactly one [`Runner`] drives a program at a time. The
//! xoshiro state therefore lives in the child instance's memory across batches — only `reseed`
//! writes it — which keeps the per-batch boundary crossing to a single output copy.

use std::sync::Arc;

use wasm_bindgen::prelude::wasm_bindgen;

use crate::backend::{Backend, InterpBackend, Program, Runner};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId};
use crate::kernel::seed_state;
use crate::wasm_emit::emit_for;

// The JS host. An ES module (wasm-bindgen `inline_js`) whose module-level state persists across
// calls. Instances are **content-addressed**: `nz_kernel_new` hashes the kernel bytes and reuses a
// live instance for identical kernels (re-running the same program is the common case — no recompile,
// no leak), with an LRU cap bounding the registry. It returns a handle (>= 0), or -1 if the module
// can't be instantiated (e.g. the main-thread sync-compile size limit) — the caller then falls back
// to the interpreter. We intentionally don't tie instance lifetime to Rust `Drop` (that FFI call gets
// eliminated, and the LRU is both simpler and sound). The kernel reads/writes its xoshiro state at
// byte 0 and writes its output column at byte 4096 of its OWN memory — the convention `wasm_emit` is
// built around (state low, output at 4096; one 64 KiB page is plenty). Single-runner on wasm32, so a
// reused instance is never driven concurrently; every reduction `reseed`s before its first batch.
#[wasm_bindgen(inline_js = r#"
const _CAP = 64;
const _byHash = new Map(); // hash -> id
const _byId = new Map();   // id (insertion-ordered) -> instance
let _next = 1;
// round-half-away-from-zero, matching Rust's f64::round (Math.round rounds half toward +inf).
const _round = (x) => { const a = Math.floor(Math.abs(x) + 0.5); return x < 0 ? -a : a; };
// `ln`/`sin`/`cos` are inlined in the kernel now; only these three remain host calls.
const _imports = { m: { atan: Math.atan, round: _round, pow: Math.pow } };

function _hash(bytes) { // FNV-1a over the kernel bytes (cheap; bytes are small and run once per program)
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) { h = (h ^ bytes[i]) >>> 0; h = Math.imul(h, 0x01000193) >>> 0; }
  return (h >>> 0) + ":" + bytes.length;
}

export function nz_kernel_new(bytes) {
  const key = _hash(bytes);
  const cached = _byHash.get(key);
  if (cached !== undefined && _byId.has(cached)) return cached;
  let instance;
  try {
    instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), _imports);
  } catch (_e) {
    return -1;
  }
  const id = _next++;
  _byHash.set(key, id);
  _byId.set(id, instance);
  if (_byId.size > _CAP) { const oldest = _byId.keys().next().value; _byId.delete(oldest); }
  return id;
}

export function nz_kernel_seed(handle, state) {
  const inst = _byId.get(handle);
  // `state` is a BigUint64Array view over wasm memory (from Rust `&[u64]`); copy it into the child.
  new BigUint64Array(inst.exports.memory.buffer, 0, state.length).set(state);
}

export function nz_kernel_run(handle, out, n) {
  const inst = _byId.get(handle);
  inst.exports.kernel(4096, n, 0); // kernel(out_ptr, n, state_ptr) over the child's own memory
  // `out` is a live Float64Array view over wasm memory (from Rust `&mut [f64]`); fill it directly.
  out.set(new Float64Array(inst.exports.memory.buffer, 4096, n));
}
"#)]
extern "C" {
    fn nz_kernel_new(bytes: &[u8]) -> i32;
    fn nz_kernel_seed(handle: i32, state: &[u64]);
    fn nz_kernel_run(handle: i32, out: &mut [f64], n: u32);
}

/// The browser backend: emit a kernel for the graph and instantiate it in the JS host. Falls back to
/// the interpreter when the graph isn't profitable to emit, or the host can't instantiate the module.
#[derive(Default)]
pub struct WasmHostBackend;

impl WasmHostBackend {
    pub fn new() -> Self {
        WasmHostBackend
    }
}

impl Backend for WasmHostBackend {
    fn compile(&self, graph: &RvGraph, root: RvId) -> Box<dyn Program> {
        let Some((bytes, streams)) = emit_for(graph, root) else {
            return InterpBackend.compile(graph, root); // gate rejected it (e.g. Poisson)
        };
        let handle = nz_kernel_new(&bytes);
        if handle < 0 {
            return InterpBackend.compile(graph, root); // instantiation failed; stay correct
        }
        Box::new(WasmProgram { inner: Arc::new(KernelHandle { handle, streams }) })
    }
}

/// A handle to a JS-host instance (kept alive by the host's content-addressed LRU, not by this
/// struct). Shared behind an `Arc` so runners can clone it cheaply. Plain values → `Send + Sync`.
struct KernelHandle {
    handle: i32,
    streams: usize,
}

/// A compiled browser program: the shared kernel handle. (Spun into a single runner on wasm32.)
struct WasmProgram {
    inner: Arc<KernelHandle>,
}

impl Program for WasmProgram {
    fn runner(&self) -> Box<dyn Runner> {
        Box::new(WasmRunner { inner: self.inner.clone(), buf: vec![0.0; BATCH] })
    }
}

/// Per-runner state: a clone of the shared handle and an output column. The RNG state lives in the
/// child instance's memory (written by `reseed`, advanced in place by each `next_batch`).
struct WasmRunner {
    inner: Arc<KernelHandle>,
    buf: Vec<f64>,
}

impl Runner for WasmRunner {
    fn reseed(&mut self, seed: u64) {
        let state = seed_state(seed, self.inner.streams);
        nz_kernel_seed(self.inner.handle, &state);
    }

    fn next_batch(&mut self, len: usize) -> &[f64] {
        // Fill the full BATCH (constant RNG consumption per call), then slice to `len`.
        nz_kernel_run(self.inner.handle, &mut self.buf, BATCH as u32);
        &self.buf[..len]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}
