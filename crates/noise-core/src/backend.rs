//! Execution-backend seam (PLAN.md Phase 4, steps "B0" + compile-once-share).
//!
//! Two levels:
//!   * [`Program`] — the *immutable* compiled artifact (interpreter bytecode, or a JIT'd kernel).
//!     `Send + Sync`, so ONE compile is shared across every worker thread.
//!   * [`Runner`]  — *per-worker* mutable execution state (the column register file / output
//!     buffer, plus the RNG). Spun up cheaply from a `&Program` via [`Program::runner`].
//!
//! Splitting these is what lets the parallel reducer compile once and fan out cheap runners, rather
//! than recompiling on every thread (the old single-level `Sampler` forced per-thread compiles —
//! the cost ceiling on multicore). The columnar interpreter ([`InterpBackend`]) is the default and
//! the correctness oracle; a Cranelift native JIT and a future WASM emitter sit behind this seam,
//! each consuming the same [`RvGraph`] IR.

use std::sync::Arc;

use crate::bytecode::{compile, run_batch, Program as ByteProgram, BATCH};
use crate::dist::{RvGraph, RvId};
use crate::kernel::NodeCost;
use crate::rng::Rng;

/// Compiles the transitive cone of a root RV into an immutable, shareable [`Program`].
pub trait Backend {
    fn compile(&self, graph: &RvGraph, root: RvId) -> Box<dyn Program>;
}

/// The default forcing path: compile `root` with the best available backend. Three mutually
/// exclusive targets sit behind this one seam, each lowering the same simplified `RvGraph`:
///   * native + `jit` → the Cranelift JIT (machine code),
///   * `wasm32` → the WASM-emitter host backend (an emitted wasm kernel driven by the JS host),
///   * otherwise → the columnar interpreter.
///
/// Each codegen path falls back to the interpreter for any graph it can't profitably emit, so the
/// choice only ever affects speed, never results.
///
/// Returns the compiled program alongside the simplified cone's [`NodeCost`] — the per-draw
/// operation/source counts the playground multiplies by the draw count for its run-time readout.
/// Computing it here (on the post-simplify graph, once) keeps it backend-independent and exact.
pub fn compile_root(graph: &RvGraph, root: RvId) -> (Box<dyn Program>, NodeCost) {
    // Simplify once (fold constants, apply identities, CSE) so the backend lowers a smaller DAG.
    // The rewritten graph is local — backends copy what they need, retaining no reference to it.
    let (graph, root) = crate::simplify::simplify(graph, root);
    let cost = crate::kernel::cost(&graph, root);
    #[cfg(feature = "jit")]
    let program = crate::jit::JitBackend::new().compile(&graph, root);
    #[cfg(all(not(feature = "jit"), target_arch = "wasm32"))]
    let program = crate::wasm_host::WasmHostBackend::new().compile(&graph, root);
    #[cfg(all(not(feature = "jit"), not(target_arch = "wasm32")))]
    let program = InterpBackend.compile(&graph, root);
    (program, cost)
}

/// An immutable compiled program. `Send + Sync` so a single compilation is shared by reference
/// across all worker threads; each worker calls [`runner`](Program::runner) for its own state.
pub trait Program: Send + Sync {
    /// Create a fresh per-worker [`Runner`] (own scratch + RNG). Cheap — no recompilation.
    fn runner(&self) -> Box<dyn Runner>;
}

/// Per-worker execution state. Used by exactly one thread, so it need not be `Send`/`Sync`.
pub trait Runner {
    /// (Re)initialize the RNG from `seed`. Must be called before [`next_batch`](Runner::next_batch).
    fn reseed(&mut self, seed: u64);

    /// Produce the next batch of draws and return the **first `len`** root samples (`len <=
    /// batch_cap()`). Advances the RNG by a *fixed* amount per call (independent of `len`), so the
    /// draw stream doesn't depend on how a final partial batch is sliced.
    fn next_batch(&mut self, len: usize) -> &[f64];

    /// Maximum `len` accepted by [`next_batch`](Runner::next_batch) — the backend's column width.
    fn batch_cap(&self) -> usize;
}

/// The default backend: the batched, columnar bytecode interpreter (`bytecode` module).
#[derive(Debug, Default, Clone, Copy)]
pub struct InterpBackend;

impl Backend for InterpBackend {
    fn compile(&self, graph: &RvGraph, root: RvId) -> Box<dyn Program> {
        Box::new(InterpProgram {
            inner: Arc::new(compile(graph, root)),
        })
    }
}

/// Interpreter program: the shared, immutable bytecode (`Arc` so runners keep it alive).
struct InterpProgram {
    inner: Arc<ByteProgram>,
}

impl Program for InterpProgram {
    fn runner(&self) -> Box<dyn Runner> {
        // Allocate this worker's column file once; reused across its batches.
        let regs = (0..self.inner.n_regs)
            .map(|_| vec![0.0f64; BATCH].into_boxed_slice())
            .collect();
        // Placeholder RNG; the driver calls `reseed` before the first batch.
        Box::new(InterpRunner {
            prog: self.inner.clone(),
            regs,
            rng: Rng::seed_from_u64(0),
        })
    }
}

/// Interpreter runner: a clone of the shared bytecode `Arc`, this worker's column file, and its RNG.
struct InterpRunner {
    prog: Arc<ByteProgram>,
    regs: Vec<Box<[f64]>>,
    rng: Rng,
}

impl Runner for InterpRunner {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    fn next_batch(&mut self, len: usize) -> &[f64] {
        // Fill the full BATCH (so RNG consumption is constant per call), then slice to `len`.
        run_batch(&self.prog, &mut self.regs, &mut self.rng);
        &self.regs[self.prog.root as usize][..len]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}
