//! Browser host seam for the WASM emitter (PLAN.md "Browser note") — wasm32-only.
//!
//! [`crate::wasm_emit`] turns a graph into a WebAssembly module (`Vec<u8>`); this module is what
//! *runs* it in the browser. A wasm sandbox can't run a child module by itself — only the JS host
//! can `WebAssembly.instantiate` — so the kernel is driven through a tiny inline-JS shim
//! (`nz_kernel_*`): compile+instantiate once, then run a batch per call — the kernel is stateless
//! (counter-keyed, PLAN-PREGPU Track C): the runner passes the key words and starting lane as
//! arguments, and only the output column crosses the boundary.
//!
//! It plugs into the exact same [`Backend`]/[`Program`]/[`Runner`] seam the interpreter and the
//! other codegen backends use, so on `wasm32` [`crate::backend::compile_root`] transparently routes profitable graphs
//! here (and falls back to the interpreter for everything the gate rejects, or if instantiation
//! fails — e.g. the main-thread sync-compile size limit). Correctness is never at stake, only speed.
//!
//! One instance per runner, not per program. With the `wasm-threads` feature, `reduce::run_reduction`
//! fans out across Web Workers, so several [`Runner`]s drive one program concurrently — and each
//! instantiates in its own JS agent (the kernel itself is stateless, so instance sharing would be
//! sound; the per-agent registry is what forces per-thread instantiation).
//!
//! That is why [`Program`] holds the kernel *bytes* and instantiates in [`Program::runner`], on the
//! thread that will drive it: a JS host handle is not portable across workers (each worker is a
//! separate JS agent with its own `nz_kernel_*` registry — linear memory is shared, JS globals are
//! not). See [`WasmProgram`].

use std::sync::Arc;

use wasm_bindgen::prelude::wasm_bindgen;

use crate::backend::{Backend, InterpBackend, JointProgram, JointRunner, Program, Runner};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId};
use crate::rng::Key;
use crate::wasm_emit::{emit_for, emit_for_roots};

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
// `nz_kernel_run*` are **status-returning**: they return -1 (instead of dereferencing `undefined`
// and throwing) if their instance is ever gone, and the Rust caller then transparently falls back
// to the interpreter — so an evicted live handle degrades to correct-slow rather than a poisoned
// instance. We still intentionally don't tie instance lifetime to Rust `Drop` (that FFI call can
// be elided; the LRU + status-return is simpler and sound).
//
// The kernel writes its output column (f32 lanes) at byte 4096 of its OWN memory — the convention `wasm_emit`
// is built around. It is stateless (counter-keyed): the key words and starting lane arrive as call
// arguments, so a content-addressed, reused instance can never leak one program's draws into
// another's — there is nothing to seed and nothing left behind.
#[wasm_bindgen(inline_js = r#"
const _CAP = 64;
const _byHash = new Map(); // hash -> id
const _byId = new Map();   // id (recency-ordered) -> instance
let _next = 1;
// round-half-away-from-zero, matching Rust's f64::round (Math.round rounds half toward +inf).
const _round = (x) => { const a = Math.floor(Math.abs(x) + 0.5); return x < 0 ? -a : a; };
// `ln` is inlined; `sin`/`cos` are inlined but re-imported for the large-argument fallback
// (finding C3); `exp` is imported to match the interpreter (finding C9); `sqrt` is imported
// because V8/arm64 regresses ~30% on large kernel bodies with inline `f64.sqrt` (2026-07-14).
const _imports = { m: { atan: Math.atan, round: _round, pow: Math.pow, sin: Math.sin, cos: Math.cos, exp: Math.exp, sqrt: Math.sqrt } };

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
// to the interpreter on -1 instead of the host throwing (finding C5). The kernel is stateless:
// key words + starting lane arrive per call (i32 wrap of the u32 values is fine — the kernel
// treats them as raw 32-bit words).
export function nz_kernel_run(handle, out, n, k0, k1, lane0) {
  const inst = _byId.get(handle);
  if (inst === undefined) return -1;
  inst.exports.kernel(4096, n, k0, k1, lane0); // kernel(out_ptr, n, key_lo, key_hi, lane0)
  // `out` is a live Float32Array view over wasm memory (from Rust `&mut [f32]` — lanes are f32,
  // PLAN-PREGPU Track B); fill it directly.
  out.set(new Float32Array(inst.exports.memory.buffer, 4096, n));
  return 0;
}

// Multi-column twin of nz_kernel_run for a joint (multi-root) kernel: the kernel fills `n` lanes
// of each BATCH-strided column at 4096, and `out` (length k*BATCH) receives all columns in one
// copy. Same status contract: -1 if the instance is gone (finding C5).
export function nz_kernel_run_cols(handle, out, n, k0, k1, lane0) {
  const inst = _byId.get(handle);
  if (inst === undefined) return -1;
  inst.exports.kernel(4096, n, k0, k1, lane0);
  out.set(new Float32Array(inst.exports.memory.buffer, 4096, out.length));
  return 0;
}
"#)]
extern "C" {
    fn nz_kernel_new(bytes: &[u8]) -> i32;
    fn nz_kernel_run(handle: i32, out: &mut [f32], n: u32, k0: u32, k1: u32, lane0: u32) -> i32;
    fn nz_kernel_run_cols(
        handle: i32,
        out: &mut [f32],
        n: u32,
        k0: u32,
        k1: u32,
        lane0: u32,
    ) -> i32;
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
        let Some(bytes) = emit_for(graph, root, draws) else {
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
            fallback,
        })
    }

    /// The joint (multi-root) path: one emitted multi-column kernel, gated on the union cone —
    /// this is what puts the emitter (2–7× over the wasm bytecode interpreter) under the plot /
    /// introspection drivers, which were previously interpreter-only. Falls back to the multi-root
    /// interpreter when the gate declines or the host can't instantiate.
    fn compile_joint(
        &self,
        graph: &RvGraph,
        roots: &[RvId],
        draws: usize,
    ) -> Box<dyn JointProgram> {
        let Some(bytes) = emit_for_roots(graph, roots, draws) else {
            return InterpBackend.compile_joint(graph, roots, draws);
        };
        // Multi-root interpreter fallback for the same graph, for the same C5 degradation story as
        // the single-root path (instantiate failure / eviction → correct-slow, never a throw).
        let fallback: Arc<dyn JointProgram> =
            Arc::from(InterpBackend.compile_joint(graph, roots, draws));
        Box::new(WasmJointProgram {
            bytes: Arc::new(bytes),
            k: roots.len(),
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
    fallback: Arc<dyn Program>,
}

impl Program for WasmProgram {
    fn runner(&self) -> Box<dyn Runner> {
        // Instantiate in *this* thread's host registry (see the type docs). A negative handle means
        // this thread can't run the kernel (e.g. the main-thread sync-compile size limit) — the
        // interpreter fallback keeps it correct. Nothing to seed: the kernel is stateless, so a
        // reused (content-addressed) instance can't carry another program's draws (old finding C5
        // seeding hazard, now structurally gone).
        let handle = nz_kernel_new(&self.bytes);
        if handle < 0 {
            return self.fallback.runner();
        }
        Box::new(WasmRunner {
            handle,
            key: Key::from_seed(0),
            lane: 0,
            seed: 0,
            buf: vec![0.0f32; BATCH],
            fallback: self.fallback.clone(),
            fell_back: None,
        })
    }
}

/// Per-runner state: the host handle, this runner's draw key + lane cursor (the kernel is
/// stateless — both are call arguments), and an output column. If the host instance is lost
/// (evicted), `fell_back` holds an interpreter runner that drives the rest of the run (finding
/// C5); `seed`/`lane` let it resume from exactly the right position — with counter keying the
/// fallback's draws are bit-identical to what the kernel would have produced.
struct WasmRunner {
    handle: i32,
    key: Key,
    lane: u32,
    seed: u64,
    buf: Vec<f32>,
    fallback: Arc<dyn Program>,
    fell_back: Option<Box<dyn Runner>>,
}

impl WasmRunner {
    /// Switch to the interpreter fallback, positioned at (`seed`, `lane`), for this and every
    /// subsequent batch.
    fn switch_to_fallback(&mut self) {
        let mut r = self.fallback.runner();
        r.position(self.seed, self.lane);
        self.fell_back = Some(r);
    }
}

impl Runner for WasmRunner {
    fn position(&mut self, seed: u64, lane: u32) {
        self.seed = seed;
        self.key = Key::from_seed(seed);
        self.lane = lane;
        if let Some(fb) = self.fell_back.as_mut() {
            fb.position(seed, lane);
        }
    }

    fn next_batch(&mut self, len: usize) -> &[f32] {
        // Fill the full BATCH (constant lane consumption per call), then slice to `len`.
        if self.fell_back.is_none() {
            let ok = nz_kernel_run(
                self.handle,
                &mut self.buf,
                BATCH as u32,
                self.key.k0,
                self.key.k1,
                self.lane,
            ) >= 0;
            if ok {
                self.lane = self.lane.wrapping_add(BATCH as u32);
                return &self.buf[..len];
            }
            // Instance evicted mid-run: fall back for this and all future batches (finding C5).
            self.switch_to_fallback();
        }
        self.lane = self.lane.wrapping_add(BATCH as u32);
        self.fell_back.as_mut().unwrap().next_batch(len)
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}

/// A compiled multi-root browser program: the multi-column kernel's **bytes** plus a multi-root
/// interpreter fallback. Bytes-not-handle for the same cross-worker reason as [`WasmProgram`].
struct WasmJointProgram {
    bytes: Arc<Vec<u8>>,
    /// Number of roots (BATCH-strided output columns) the kernel writes.
    k: usize,
    fallback: Arc<dyn JointProgram>,
}

impl JointProgram for WasmJointProgram {
    fn runner(&self) -> Box<dyn JointRunner> {
        // Same instantiate-on-the-driving-thread story as `WasmProgram::runner` (stateless kernel:
        // nothing to seed).
        let handle = nz_kernel_new(&self.bytes);
        if handle < 0 {
            return self.fallback.runner();
        }
        Box::new(WasmJointRunner {
            handle,
            key: Key::from_seed(0),
            lane: 0,
            seed: 0,
            buf: vec![0.0f32; self.k * BATCH],
            fallback: self.fallback.clone(),
            fell_back: None,
        })
    }
}

/// Per-runner joint state: the host handle, the draw key + lane cursor, and one flat `k×BATCH`
/// column buffer. Fallback story mirrors [`WasmRunner`] (finding C5): if the instance is ever
/// gone, the rest of the run degrades to the multi-root interpreter positioned at (`seed`,
/// `lane`) — bit-identical draws under counter keying.
struct WasmJointRunner {
    handle: i32,
    key: Key,
    lane: u32,
    seed: u64,
    buf: Vec<f32>,
    fallback: Arc<dyn JointProgram>,
    fell_back: Option<Box<dyn JointRunner>>,
}

impl WasmJointRunner {
    fn switch_to_fallback(&mut self) {
        let mut r = self.fallback.runner();
        r.position(self.seed, self.lane);
        self.fell_back = Some(r);
    }
}

impl JointRunner for WasmJointRunner {
    fn position(&mut self, seed: u64, lane: u32) {
        self.seed = seed;
        self.key = Key::from_seed(seed);
        self.lane = lane;
        if let Some(fb) = self.fell_back.as_mut() {
            fb.position(seed, lane);
        }
    }

    fn next_batch(&mut self) {
        if self.fell_back.is_none() {
            let ok = nz_kernel_run_cols(
                self.handle,
                &mut self.buf,
                BATCH as u32,
                self.key.k0,
                self.key.k1,
                self.lane,
            ) >= 0;
            if ok {
                self.lane = self.lane.wrapping_add(BATCH as u32);
                return;
            }
            // Instance evicted mid-run: fall back for this and all future batches (finding C5).
            self.switch_to_fallback();
        }
        self.lane = self.lane.wrapping_add(BATCH as u32);
        self.fell_back.as_mut().unwrap().next_batch();
    }

    fn col(&self, j: usize) -> &[f32] {
        match self.fell_back.as_ref() {
            Some(fb) => fb.col(j),
            None => &self.buf[j * BATCH..(j + 1) * BATCH],
        }
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}
