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
//! One instance per runner, not per program. With the `wasm-threads` feature, `reduce::run_reduction`
//! fans out across Web Workers, so several [`Runner`]s drive one program concurrently — and each must
//! own its own child instance, because the xoshiro state lives *in that instance's memory* across
//! batches (only `reseed` writes it, which is what keeps the per-batch crossing to a single output
//! copy). Sharing one instance between workers would have them stomping on each other's RNG state.
//!
//! That is why [`Program`] holds the kernel *bytes* and instantiates in [`Program::runner`], on the
//! thread that will drive it: a JS host handle is not portable across workers (each worker is a
//! separate JS agent with its own `nz_kernel_*` registry — linear memory is shared, JS globals are
//! not). See [`WasmProgram`].

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
// to the interpreter.
//
// **Liveness (finding C5).** `nz_kernel_new` inserts/refreshes the returned handle as the
// most-recent entry and evicts strictly oldest-first, so a kernel that was just created or reused —
// i.e. one a `Runner` is about to drive — is never the eviction victim (single-runner on wasm32
// means it is used before `_CAP` other distinct kernels can be created). As defense-in-depth,
// `nz_kernel_seed`/`nz_kernel_run` are **status-returning**: they return -1 (instead of
// dereferencing `undefined` and throwing) if their instance is ever gone, and the Rust caller then
// transparently falls back to the interpreter — so an evicted live handle degrades to correct-slow
// rather than a poisoned instance. We still intentionally don't tie instance lifetime to Rust `Drop`
// (that FFI call can be elided; the LRU + status-return is simpler and sound).
//
// The kernel reads/writes its xoshiro state at byte 0 and writes its output column at byte 4096 of
// its OWN memory — the convention `wasm_emit` is built around (state low, output at 4096; one 64 KiB
// page is plenty). `Program::runner` seeds a placeholder state on creation (matching the JIT path)
// so a reused instance never runs a batch on a previous program's leftover state; the driver
// `reseed`s again before its first batch.
#[wasm_bindgen(inline_js = r#"
const _CAP = 64;
const _byHash = new Map(); // hash -> id
const _byId = new Map();   // id (recency-ordered) -> instance
let _next = 1;
// round-half-away-from-zero, matching Rust's f64::round (Math.round rounds half toward +inf).
const _round = (x) => { const a = Math.floor(Math.abs(x) + 0.5); return x < 0 ? -a : a; };
// `ln` is inlined; `sin`/`cos` are inlined but re-imported for the large-argument fallback
// (finding C3); `exp` is imported to match the interpreter (finding C9).
const _imports = { m: { atan: Math.atan, round: _round, pow: Math.pow, sin: Math.sin, cos: Math.cos, exp: Math.exp } };

function _hash(bytes) { // FNV-1a over the kernel bytes (cheap; bytes are small and run once per program)
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) { h = (h ^ bytes[i]) >>> 0; h = Math.imul(h, 0x01000193) >>> 0; }
  return (h >>> 0) + ":" + bytes.length;
}

export function nz_kernel_new(bytes) {
  const key = _hash(bytes);
  const cached = _byHash.get(key);
  if (cached !== undefined && _byId.has(cached)) {
    // Cache hit — refresh recency (delete + re-insert) so a hot kernel moves to the most-recent
    // slot and can't be the eviction victim while it's in use.
    const inst = _byId.get(cached);
    _byId.delete(cached);
    _byId.set(cached, inst);
    return cached;
  }
  let instance;
  try {
    instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), _imports);
  } catch (_e) {
    return -1;
  }
  const id = _next++;
  _byHash.set(key, id);
  _byId.set(id, instance); // most-recent
  // Evict least-recently-used once over the cap. The just-inserted id is most-recent, so a live
  // handle (just created / just refreshed) is never evicted here.
  if (_byId.size > _CAP) { const oldest = _byId.keys().next().value; _byId.delete(oldest); }
  return id;
}

// Status-returning: 0 on success, -1 if the instance is gone (evicted). The Rust caller falls back
// to the interpreter on -1 instead of the host throwing (finding C5).
export function nz_kernel_seed(handle, state) {
  const inst = _byId.get(handle);
  if (inst === undefined) return -1;
  // `state` is a BigUint64Array view over wasm memory (from Rust `&[u64]`); copy it into the child.
  new BigUint64Array(inst.exports.memory.buffer, 0, state.length).set(state);
  return 0;
}

export function nz_kernel_run(handle, out, n) {
  const inst = _byId.get(handle);
  if (inst === undefined) return -1;
  inst.exports.kernel(4096, n, 0); // kernel(out_ptr, n, state_ptr) over the child's own memory
  // `out` is a live Float64Array view over wasm memory (from Rust `&mut [f64]`); fill it directly.
  out.set(new Float64Array(inst.exports.memory.buffer, 4096, n));
  return 0;
}
"#)]
extern "C" {
    fn nz_kernel_new(bytes: &[u8]) -> i32;
    fn nz_kernel_seed(handle: i32, state: &[u64]) -> i32;
    fn nz_kernel_run(handle: i32, out: &mut [f64], n: u32) -> i32;
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
    fn compile(&self, graph: &RvGraph, root: RvId, draws: usize) -> Box<dyn Program> {
        let Some((bytes, streams)) = emit_for(graph, root, draws) else {
            // Gate rejected it: unsupported (Poisson / Gather), libcall-bound, or too few draws to
            // pay back the emit + instantiate (`kernel::break_even_draws`).
            return InterpBackend.compile(graph, root, draws);
        };
        // Interpreter program for the *same* graph, kept as a fallback: used if this thread can't
        // instantiate the kernel, or if its instance is evicted mid-run (a status -1 from
        // `nz_kernel_seed`/`nz_kernel_run`), so either degrades to correct-slow rather than throwing
        // (finding C5). Cheap to compile.
        let fallback: Arc<dyn Program> = Arc::from(InterpBackend.compile(graph, root, draws));
        Box::new(WasmProgram {
            bytes: Arc::new(bytes),
            streams,
            fallback,
        })
    }
}

/// A compiled browser program: the emitted kernel's **bytes** plus an interpreter fallback program
/// for the same graph.
///
/// We keep the bytes, not a host handle, because the handle is not portable across threads. Under
/// wasm threads every thread is a separate JS agent (a Web Worker) with its **own** module-level
/// `nz_kernel_*` registry — linear memory is shared, JS globals are not. A handle minted while
/// compiling on the driver thread would simply not exist in a worker's registry, `nz_kernel_run`
/// would return -1, and every worker would silently degrade to the interpreter — losing exactly the
/// emitter win that threads were added to multiply.
///
/// So instantiation moves to [`Program::runner`], which runs *on* the thread that will drive it. The
/// bytes live in shared linear memory, so each worker instantiates from the same kernel, and the
/// host's content-addressing makes the second and later calls for a given kernel a cache hit.
struct WasmProgram {
    bytes: Arc<Vec<u8>>,
    streams: usize,
    fallback: Arc<dyn Program>,
}

impl Program for WasmProgram {
    fn runner(&self) -> Box<dyn Runner> {
        // Instantiate in *this* thread's host registry (see the type docs). A negative handle means
        // this thread can't run the kernel (e.g. the main-thread sync-compile size limit) — the
        // interpreter fallback keeps it correct.
        let handle = nz_kernel_new(&self.bytes);
        if handle < 0 {
            return self.fallback.runner();
        }
        // Seed a placeholder state into the (possibly reused, content-addressed) instance now —
        // matching the JIT path (`jit::JitProgram::runner`) — so a reused instance never carries a
        // previous program's xoshiro state into a batch, even before the driver's first `reseed`
        // (finding C5). If the instance is already gone, start on the interpreter fallback.
        let state = seed_state(0, self.streams);
        let fell_back = if nz_kernel_seed(handle, &state) < 0 {
            Some(self.fallback.runner())
        } else {
            None
        };
        Box::new(WasmRunner {
            handle,
            streams: self.streams,
            buf: vec![0.0; BATCH],
            fallback: self.fallback.clone(),
            fell_back,
            last_seed: 0,
        })
    }
}

/// Per-runner state: a clone of the shared handle and an output column. The RNG state lives in the
/// child instance's memory (written by `reseed`, advanced in place by each `next_batch`). If the
/// host instance is lost (evicted), `fell_back` holds an interpreter runner that drives the rest of
/// the run (finding C5); `last_seed` lets that fallback resume from the right seed.
struct WasmRunner {
    handle: i32,
    streams: usize,
    buf: Vec<f64>,
    fallback: Arc<dyn Program>,
    fell_back: Option<Box<dyn Runner>>,
    last_seed: u64,
}

impl WasmRunner {
    /// Switch to the interpreter fallback, seeded to `seed`, for this and every subsequent batch.
    fn switch_to_fallback(&mut self, seed: u64) {
        let mut r = self.fallback.runner();
        r.reseed(seed);
        self.fell_back = Some(r);
    }
}

impl Runner for WasmRunner {
    fn reseed(&mut self, seed: u64) {
        self.last_seed = seed;
        if let Some(fb) = self.fell_back.as_mut() {
            fb.reseed(seed);
            return;
        }
        let state = seed_state(seed, self.streams);
        if nz_kernel_seed(self.handle, &state) < 0 {
            self.switch_to_fallback(seed); // instance evicted; degrade to the interpreter
        }
    }

    fn next_batch(&mut self, len: usize) -> &[f64] {
        // Fill the full BATCH (constant RNG consumption per call), then slice to `len`.
        if self.fell_back.is_none() {
            if nz_kernel_run(self.handle, &mut self.buf, BATCH as u32) >= 0 {
                return &self.buf[..len];
            }
            // Instance evicted mid-run: fall back for this and all future batches (finding C5).
            let seed = self.last_seed;
            self.switch_to_fallback(seed);
        }
        self.fell_back.as_mut().unwrap().next_batch(len)
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}
