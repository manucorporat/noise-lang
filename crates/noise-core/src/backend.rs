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

use crate::bytecode::{
    compile, compile_roots as byte_compile_roots, run_batch, Program as ByteProgram, BATCH,
};
use crate::dist::{RvGraph, RvId};
use crate::kernel::NodeCost;
use crate::rng::Rng;

/// Compiles the transitive cone of a root RV into an immutable, shareable [`Program`].
///
/// `draws` is how many samples the caller is about to take from this program. Codegen backends need
/// it: compiling is a fixed cost paid once, fusion is a saving earned per draw, so a query that
/// draws too few samples is faster interpreted no matter how good the kernel would be (see
/// [`crate::kernel::break_even_draws`]). The interpreter ignores it.
pub trait Backend {
    fn compile(&self, graph: &RvGraph, root: RvId, draws: usize) -> Box<dyn Program>;

    /// Compile several roots into ONE shared [`JointProgram`] whose per-lane draws are *joint*
    /// (every root reads the same per-lane draw of any shared source) — the forcing path behind the
    /// joint introspection drivers (`scatter`/`describe`/`corr`/`fan`). The default is the columnar
    /// interpreter's multi-root lowering ([`crate::bytecode::compile_roots`]) — always correct — so
    /// a backend only overrides this when it can emit a faster joint kernel.
    fn compile_joint(
        &self,
        graph: &RvGraph,
        roots: &[RvId],
        _draws: usize,
    ) -> Box<dyn JointProgram> {
        let (prog, regs) = byte_compile_roots(graph, roots);
        Box::new(InterpJointProgram {
            inner: Arc::new(prog),
            regs: regs.iter().map(|&r| r as usize).collect(),
        })
    }
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
pub fn compile_root(graph: &RvGraph, root: RvId, draws: usize) -> (Box<dyn Program>, NodeCost) {
    // Simplify once (fold constants, apply identities, CSE) so the backend lowers a smaller DAG.
    // The rewritten graph is local — backends copy what they need, retaining no reference to it.
    let (graph, root) = crate::simplify::simplify(graph, root);
    let cost = crate::kernel::cost(&graph, root);
    // The Cranelift JIT is native-only: `not(target_arch = "wasm32")` guards against feature
    // unification turning `jit` on for a wasm32 build, which would otherwise select an impossible
    // backend (finding C7). On wasm32 the WASM-host backend always wins (the `jit` arm can't match
    // there); the interpreter is the remaining native, non-`jit` case. The three cfgs are mutually
    // exclusive and exhaustive over the {wasm32?} × {jit?} matrix.
    #[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
    let program = crate::jit::JitBackend::new().compile(&graph, root, draws);
    #[cfg(target_arch = "wasm32")]
    let program = crate::wasm_host::WasmHostBackend::new().compile(&graph, root, draws);
    #[cfg(all(not(feature = "jit"), not(target_arch = "wasm32")))]
    let program = InterpBackend.compile(&graph, root, draws);
    (program, cost)
}

/// The multi-root twin of [`compile_root`] — the forcing path behind every joint introspection
/// pass (`plot::scatter`/`plot::fan`/`plot::corr`, `describe` of an array). One simplify over the
/// *union* of the roots' cones (cross-root sharing preserved — see
/// [`crate::simplify::simplify_roots`]), then the best available backend lowers all roots into ONE
/// shared kernel: native + `jit` → a multi-output Cranelift kernel, `wasm32` → a multi-column
/// emitted wasm kernel, otherwise → the multi-root bytecode interpreter. Codegen paths decline
/// unprofitable graphs exactly like [`compile_root`] (same per-backend draw thresholds), falling
/// back to the interpreter — the choice affects speed, never correctness.
///
/// Returns the program alongside the union cone's [`NodeCost`] (per-draw ops/sources on the
/// simplified graph), which the caller records into the engine's run stats.
pub fn compile_roots(
    graph: &RvGraph,
    roots: &[RvId],
    draws: usize,
) -> (Box<dyn JointProgram>, NodeCost) {
    let (graph, roots) = crate::simplify::simplify_roots(graph, roots);
    let cost = crate::kernel::cost_roots(&graph, &roots);
    // Same three-way cfg dispatch as `compile_root` (see there for why the cfgs are exclusive and
    // exhaustive over the {wasm32?} × {jit?} matrix).
    #[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
    let program = crate::jit::JitBackend::new().compile_joint(&graph, &roots, draws);
    #[cfg(target_arch = "wasm32")]
    let program = crate::wasm_host::WasmHostBackend::new().compile_joint(&graph, &roots, draws);
    #[cfg(all(not(feature = "jit"), not(target_arch = "wasm32")))]
    let program = InterpBackend.compile_joint(&graph, &roots, draws);
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
    /// The interpreter has no meaningful compile cost, so it never declines: `draws` is unused.
    fn compile(&self, graph: &RvGraph, root: RvId, _draws: usize) -> Box<dyn Program> {
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

/// The multi-root twin of [`Program`]: an immutable compiled artifact whose runners produce one
/// *column per root* per batch, with all roots drawn jointly on shared lanes.
pub trait JointProgram: Send + Sync {
    /// Create a fresh per-worker [`JointRunner`] (own scratch + RNG). Cheap — no recompilation.
    fn runner(&self) -> Box<dyn JointRunner>;
}

/// Per-worker execution state of a [`JointProgram`]. Used by exactly one thread.
pub trait JointRunner {
    /// (Re)initialize the RNG from `seed`. Must be called before [`next_batch`](JointRunner::next_batch).
    fn reseed(&mut self, seed: u64);

    /// Produce the next batch of joint draws. Advances the RNG by a *fixed* amount per call, so the
    /// draw stream doesn't depend on how a final partial batch is consumed.
    fn next_batch(&mut self);

    /// Root `j`'s column of the current batch (`batch_cap()` lanes; the driver slices a final
    /// partial batch itself). Lane `i` across all columns is one *joint* draw.
    fn col(&self, j: usize) -> &[f64];

    /// Lanes per batch — the backend's column width.
    fn batch_cap(&self) -> usize;
}

/// Interpreter joint program: the shared multi-root bytecode plus the register holding each root.
struct InterpJointProgram {
    inner: Arc<ByteProgram>,
    regs: Vec<usize>,
}

impl JointProgram for InterpJointProgram {
    fn runner(&self) -> Box<dyn JointRunner> {
        let buf = (0..self.inner.n_regs)
            .map(|_| vec![0.0f64; BATCH].into_boxed_slice())
            .collect();
        // Placeholder RNG; the driver calls `reseed` before the first batch.
        Box::new(InterpJointRunner {
            prog: self.inner.clone(),
            regs: self.regs.clone(),
            buf,
            rng: Rng::seed_from_u64(0),
        })
    }
}

/// Interpreter joint runner: this worker's full column file plus the root-register map.
struct InterpJointRunner {
    prog: Arc<ByteProgram>,
    regs: Vec<usize>,
    buf: Vec<Box<[f64]>>,
    rng: Rng,
}

impl JointRunner for InterpJointRunner {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    fn next_batch(&mut self) {
        run_batch(&self.prog, &mut self.buf, &mut self.rng);
    }

    fn col(&self, j: usize) -> &[f64] {
        &self.buf[self.regs[j]]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}
