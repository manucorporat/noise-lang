//! Execution-backend seam (PLAN.md Phase 4, steps "B0" + compile-once-share).
//!
//! Two levels:
//!   * [`Program`] — the *immutable* compiled artifact (interpreter bytecode, or an emitted wasm
//!     kernel). `Send + Sync`, so ONE compile is shared across every worker thread.
//!   * [`Runner`]  — *per-worker* mutable execution state (the column register file / output
//!     buffer, plus the RNG). Spun up cheaply from a `&Program` via [`Program::runner`].
//!
//! Splitting these is what lets the parallel reducer compile once and fan out cheap runners, rather
//! than recompiling on every thread (the old single-level `Sampler` forced per-thread compiles —
//! the cost ceiling on multicore). The columnar interpreter ([`InterpBackend`]) is the default and
//! the correctness oracle; the browser's WASM emitter sits behind this seam consuming the same
//! [`RvGraph`] IR, and the native WebGPU backend hooks a level up in `reduce` (see `gpu.rs`). (A
//! Cranelift native JIT was a third lowering here until PLAN-DROP-JIT retired it.)

use std::sync::Arc;

use crate::bytecode::{
    compile, compile_roots as byte_compile_roots, run_batch, Program as ByteProgram, BATCH,
};
use crate::dist::{RvGraph, RvId};
use crate::kernel::NodeCost;
use crate::rng::Key;

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

/// The default forcing path: compile `root` with the best available backend. Two targets sit behind
/// this one seam, each lowering the same simplified `RvGraph`:
///   * `wasm32` → the WASM-emitter host backend (an emitted wasm kernel driven by the JS host),
///   * otherwise → the columnar interpreter.
///
/// The wasm codegen path falls back to the interpreter for any graph it can't profitably emit, so
/// the choice only ever affects speed, never results. (The native performance backend is the GPU —
/// it hooks a level up in `reduce::run_reduction`, not through this seam.)
///
/// Returns the compiled program alongside the simplified cone's [`NodeCost`] — the per-draw
/// operation/source counts the playground multiplies by the draw count for its run-time readout.
/// Computing it here (on the post-simplify graph, once) keeps it backend-independent and exact.
///
/// Returns `Arc`, not `Box`: identical forcings share ONE compile through the per-engine cache
/// ([`crate::compile_cache`]) — a repeated query, an introspection pass forcing one root several
/// times, a playground re-run on a persistent engine all hit instead of recompiling. Simplify runs
/// *before* the lookup (it's
/// cheap relative to codegen, and the simplified cone is the correct cache key — see
/// [`crate::compile_cache::key`]); with no cache installed this compiles exactly as before.
pub fn compile_root(graph: &RvGraph, root: RvId, draws: usize) -> (Arc<dyn Program>, NodeCost) {
    // Expand any re-rollable loop (G4c) BEFORE simplify, so the CPU backends lower exactly the flat
    // DAG eval built before this feature — bit-for-bit the same answer and draw stream. (The GPU took
    // its shot earlier in `run_reduction`, rolling the Scan into a real loop; this is the fallback.)
    let unrolled = crate::simplify::unroll_scans(graph, root);
    let (graph, root) = unrolled.as_ref().map_or((graph, root), |(g, r)| (g, *r));
    // Simplify once (fold constants, apply identities, CSE) so the backend lowers a smaller DAG.
    // The rewritten graph is local — backends copy what they need, retaining no reference to it.
    let (graph, root) = crate::simplify::simplify(graph, root);
    let key = crate::compile_cache::key(&graph, &[root], gate_bucket(draws));
    if let Some(hit) = crate::compile_cache::lookup_single(&key) {
        return hit;
    }
    let cost = crate::kernel::cost(&graph, root);
    #[cfg(test)]
    probe::record_compile();
    // Two lowerings behind this seam: on wasm32 the WASM-host backend (an emitted kernel driven by the
    // JS host), otherwise the columnar interpreter. The native *performance* backend is the GPU, which
    // hooks a level up in `reduce::run_reduction` (a dispatch wants ≥256k lanes, where a `Runner`
    // pulls 1024), not here — so on native the interpreter is the floor the GPU falls back to. (The
    // Cranelift JIT that was the native arm was retired in PLAN-DROP-JIT; git keeps `jit.rs`.)
    #[cfg(target_arch = "wasm32")]
    let program = crate::wasm_host::WasmHostBackend::new().compile(&graph, root, draws);
    #[cfg(not(target_arch = "wasm32"))]
    let program = InterpBackend.compile(&graph, root, draws);
    let program: Arc<dyn Program> = Arc::from(program);
    crate::compile_cache::store_single(key, &program, cost);
    (program, cost)
}

/// The single `draws`-dependent input to the WASM codegen gate (`kernel::profitable[_roots]` tests
/// `draws >= MIN_DRAWS_WASM`; nothing else about the artifact depends on the count), reduced to the
/// decision bit so the compile cache keys on the DECISION, not the raw count: on wasm 200k and 300k
/// draws share one entry, while a count below the gate compiles (and caches) the interpreter artifact
/// under its own key rather than returning a stale kernel — or vice versa. **Native has no CPU
/// codegen gate** now the JIT is gone (the interpreter always compiles; the GPU decides for itself in
/// `reduce`), so everything shares one bucket there.
fn gate_bucket(draws: usize) -> bool {
    #[cfg(target_arch = "wasm32")]
    let bucket = draws >= crate::kernel::MIN_DRAWS_WASM;
    #[cfg(not(target_arch = "wasm32"))]
    let bucket = {
        let _ = draws;
        false
    };
    bucket
}

/// Test-only compile counter, proving the cache hits (or correctly misses). Thread-local, not a
/// process global: the test harness runs tests concurrently, and compiles always happen on the
/// forcing test's own (driver) thread, so a thread-local counts exactly that test's compiles.
#[cfg(test)]
pub(crate) mod probe {
    use std::cell::Cell;

    thread_local! {
        static COMPILES: Cell<u64> = const { Cell::new(0) };
    }

    /// Count one actual backend compile (a cache miss or an uncached call).
    pub(crate) fn record_compile() {
        COMPILES.with(|c| c.set(c.get() + 1));
    }

    /// Total compiles on this thread so far (tests take deltas, so no reset is needed).
    pub(crate) fn compiles() -> u64 {
        COMPILES.with(Cell::get)
    }
}

/// The multi-root twin of [`compile_root`] — the forcing path behind every joint introspection
/// pass (`plot::scatter`/`plot::fan`/`plot::corr`, `describe` of an array). One simplify over the
/// *union* of the roots' cones (cross-root sharing preserved — see
/// [`crate::simplify::simplify_roots`]), then the best available backend lowers all roots into ONE
/// shared kernel: `wasm32` → a multi-column emitted wasm kernel, otherwise → the multi-root bytecode
/// interpreter. The wasm codegen path declines unprofitable graphs exactly like [`compile_root`]
/// (same draw threshold), falling back to the interpreter — the choice affects speed, never
/// correctness.
///
/// Returns the program alongside the union cone's [`NodeCost`] (per-draw ops/sources on the
/// simplified graph), which the caller records into the engine's run stats.
///
/// `Arc` + cached like [`compile_root`]. The joint key additionally carries the roots' count and
/// order (already part of [`crate::compile_cache::key`]) because column order is baked into the
/// kernel — `[a, b]` and `[b, a]` are different artifacts over the same union cone.
pub fn compile_roots(
    graph: &RvGraph,
    roots: &[RvId],
    draws: usize,
) -> (Arc<dyn JointProgram>, NodeCost) {
    let unrolled = crate::simplify::unroll_scans_roots(graph, roots);
    let (graph, roots): (&RvGraph, &[RvId]) =
        unrolled.as_ref().map_or((graph, roots), |(g, r)| (g, r));
    let (graph, roots) = crate::simplify::simplify_roots(graph, roots);
    let key = crate::compile_cache::key(&graph, &roots, gate_bucket(draws));
    if let Some(hit) = crate::compile_cache::lookup_joint(&key) {
        return hit;
    }
    let cost = crate::kernel::cost_roots(&graph, &roots);
    #[cfg(test)]
    probe::record_compile();
    // Same two-way dispatch as `compile_root`: wasm32 → the multi-column emitted wasm kernel,
    // otherwise → the multi-root bytecode interpreter.
    #[cfg(target_arch = "wasm32")]
    let program = crate::wasm_host::WasmHostBackend::new().compile_joint(&graph, &roots, draws);
    #[cfg(not(target_arch = "wasm32"))]
    let program = InterpBackend.compile_joint(&graph, &roots, draws);
    let program: Arc<dyn JointProgram> = Arc::from(program);
    crate::compile_cache::store_joint(key, &program, cost);
    (program, cost)
}

/// An immutable compiled program. `Send + Sync` so a single compilation is shared by reference
/// across all worker threads; each worker calls [`runner`](Program::runner) for its own state.
pub trait Program: Send + Sync {
    /// Create a fresh per-worker [`Runner`] (own scratch + RNG). Cheap — no recompilation. `inputs`
    /// is the forcing's snapshot of host input values (PLAN-UNIFORM-INPUTS) — a shared `Arc` so every
    /// worker's runner reads the same slot values; empty for an input-free program.
    fn runner(&self, inputs: Arc<[f64]>) -> Box<dyn Runner>;
}

/// Per-worker execution state. Used by exactly one thread, so it need not be `Send`/`Sync`.
pub trait Runner {
    /// Position this runner: derive the draw key from `seed` and set the next batch's starting
    /// global lane. Counter keying (PLAN-PREGPU Track C) makes this O(1) and makes any lane range
    /// independently computable — a reducer chunk is just a range. Must be called before
    /// [`next_batch`](Runner::next_batch).
    fn position(&mut self, seed: u64, lane: u32);

    /// Produce the next batch of draws and return the **first `len`** root samples (`len <=
    /// batch_cap()`). Advances the lane cursor by a *fixed* `batch_cap()` per call (independent of
    /// `len`), so the draw stream doesn't depend on how a final partial batch is sliced.
    ///
    /// Lanes are **f32** (PLAN-PREGPU Track B). This one signature is the whole type change: a
    /// draw is a Monte-Carlo sample whose `O(1/√N)` standard error dwarfs f32's ~1e-7 rounding, so
    /// the transport type can be narrow — while everything that becomes a *reported estimate*
    /// (`reduce`'s Σx/Σx² accumulators, quantile interpolation, the public draw vectors) widens to
    /// f64 at this boundary and stays there. Narrow lanes, wide sums.
    fn next_batch(&mut self, len: usize) -> &[f32];

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
    fn runner(&self, inputs: Arc<[f64]>) -> Box<dyn Runner> {
        // Allocate this worker's column file once; reused across its batches. Array registers
        // (`Inst::Permutation`) get their own `n × BATCH` buffers, sized by `Program::arrays`.
        let regs = (0..self.inner.n_regs)
            .map(|_| vec![0.0f32; BATCH].into_boxed_slice())
            .collect();
        let arrs = alloc_arrays(&self.inner);
        // Placeholder key/lane; the driver calls `position` before the first batch.
        Box::new(InterpRunner {
            prog: self.inner.clone(),
            regs,
            arrs,
            inputs,
            key: Key::from_seed(0),
            lane: 0,
        })
    }
}

/// Allocate a worker's array-register file (one lane-major `n × BATCH` buffer per array register;
/// see `bytecode::Program::arrays`). Empty for the common no-array program.
fn alloc_arrays(prog: &ByteProgram) -> Vec<Box<[f32]>> {
    prog.arrays
        .iter()
        .map(|&n| vec![0.0f32; n as usize * BATCH].into_boxed_slice())
        .collect()
}

/// Interpreter runner: a clone of the shared bytecode `Arc`, this worker's column file, and its
/// draw key + lane cursor.
struct InterpRunner {
    prog: Arc<ByteProgram>,
    regs: Vec<Box<[f32]>>,
    arrs: Vec<Box<[f32]>>,
    /// This forcing's host input values (`Inst::Input` reads a slot by index). Shared across workers.
    inputs: Arc<[f64]>,
    key: Key,
    lane: u32,
}

impl Runner for InterpRunner {
    fn position(&mut self, seed: u64, lane: u32) {
        self.key = Key::from_seed(seed);
        self.lane = lane;
    }

    fn next_batch(&mut self, len: usize) -> &[f32] {
        // Fill the full BATCH (lane consumption is constant per call), then slice to `len`.
        run_batch(&self.prog, &mut self.regs, &mut self.arrs, self.key, self.lane, &self.inputs);
        self.lane = self.lane.wrapping_add(BATCH as u32);
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
    /// `inputs` is the forcing's host-input-values snapshot (see [`Program::runner`]).
    fn runner(&self, inputs: Arc<[f64]>) -> Box<dyn JointRunner>;
}

/// Per-worker execution state of a [`JointProgram`]. Used by exactly one thread.
pub trait JointRunner {
    /// Position this runner: derive the draw key from `seed` and set the next batch's starting
    /// global lane (see [`Runner::position`]). Must be called before
    /// [`next_batch`](JointRunner::next_batch).
    fn position(&mut self, seed: u64, lane: u32);

    /// Produce the next batch of joint draws. Advances the lane cursor by a *fixed* `batch_cap()`
    /// per call, so the draw stream doesn't depend on how a final partial batch is consumed.
    fn next_batch(&mut self);

    /// Root `j`'s column of the current batch (`batch_cap()` lanes; the driver slices a final
    /// partial batch itself). Lane `i` across all columns is one *joint* draw. f32, like
    /// [`Runner::next_batch`].
    fn col(&self, j: usize) -> &[f32];

    /// Lanes per batch — the backend's column width.
    fn batch_cap(&self) -> usize;
}

/// Interpreter joint program: the shared multi-root bytecode plus the register holding each root.
struct InterpJointProgram {
    inner: Arc<ByteProgram>,
    regs: Vec<usize>,
}

impl JointProgram for InterpJointProgram {
    fn runner(&self, inputs: Arc<[f64]>) -> Box<dyn JointRunner> {
        let buf = (0..self.inner.n_regs)
            .map(|_| vec![0.0f32; BATCH].into_boxed_slice())
            .collect();
        let arrs = alloc_arrays(&self.inner);
        // Placeholder key/lane; the driver calls `position` before the first batch.
        Box::new(InterpJointRunner {
            prog: self.inner.clone(),
            regs: self.regs.clone(),
            buf,
            arrs,
            inputs,
            key: Key::from_seed(0),
            lane: 0,
        })
    }
}

/// Interpreter joint runner: this worker's full column file plus the root-register map.
struct InterpJointRunner {
    prog: Arc<ByteProgram>,
    regs: Vec<usize>,
    buf: Vec<Box<[f32]>>,
    arrs: Vec<Box<[f32]>>,
    inputs: Arc<[f64]>,
    key: Key,
    lane: u32,
}

impl JointRunner for InterpJointRunner {
    fn position(&mut self, seed: u64, lane: u32) {
        self.key = Key::from_seed(seed);
        self.lane = lane;
    }

    fn next_batch(&mut self) {
        run_batch(&self.prog, &mut self.buf, &mut self.arrs, self.key, self.lane, &self.inputs);
        self.lane = self.lane.wrapping_add(BATCH as u32);
    }

    fn col(&self, j: usize) -> &[f32] {
        &self.buf[self.regs[j]]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}
