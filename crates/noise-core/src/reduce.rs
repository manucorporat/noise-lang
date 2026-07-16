//! Parallel, deterministic Monte-Carlo reduction (PLAN.md Phase 4 "multicore").
//!
//! Sampling is an embarrassingly parallel reduction: draws are independent and the quantities we
//! force (`P`, `E`, `Var`) are *monoid folds* over the sample stream. This module captures that
//! with a [`Reducer`] trait — a commutative monoid over sample columns — and a driver that runs it
//! over **fixed lane-range chunks** (counter keying, PLAN-PREGPU Track C: a chunk IS a lane range,
//! so there is nothing to seed per chunk — and the chunked stream is the sequential one).
//!
//! Determinism is the whole point: the chunk set and each chunk's seed depend only on `(seed, n)`,
//! never on thread scheduling, and per-chunk accumulators are merged in **chunk-index order**. So
//! the result is identical for any thread count — bit for bit. Threads change only wall-clock.
//!
//! **Resumable ranges** (PLAN-PRECISION Track A): the driver's real entry point is
//! [`run_reduction_range`], which reduces any chunk-aligned lane range `a..b` *into a carried
//! accumulator*. Because a chunk is a pure lane range and the fold is chunk-ordered, a reduction
//! over `0..n₁` extended by `n₁..n₂` is **bit-identical** to a single run over `0..n₂` — the
//! adaptive precision driver ([`crate::sampler`]) is built on exactly this property.
//!
//! **Epochs** (PLAN-PRECISION Track E): the lane index is `u32` end-to-end below this driver, which
//! caps a single stream at 2³² draws. Ranges beyond it split into **epochs** of 2³² lanes; epoch
//! `e` draws under a seed derived from `(seed, e)` (statistically independent streams, epoch 0
//! keeping the raw seed so every existing stream is unchanged), each epoch is an ordinary ≤ 2³²-lane
//! reduction, and accumulators fold in epoch order. Deterministic, and nothing below the driver
//! changes.
//!
//! **Soft-stop** (PLAN-PRECISION Track H): beside the hard [`CancelToken`] abort (everything
//! discarded, `Err(cancelled)`), the driver honors the token's *soft* flag and the run's `max_time`
//! deadline (see [`crate::exec`]): workers stop claiming chunks, the chunks that completed fold
//! normally, and the caller gets `Ok` with the accumulator's true count plus a [`StopCause`]
//! marker. Which chunks completed is timing-dependent, so a stopped run is explicitly *not*
//! bit-reproducible — but it is statistically honest (iid chunks, timing never depends on drawn
//! values), and the se downstream is computed from the true folded count.
//!
//! Executor — two of them, one monoid. Native uses `std::thread::scope` (zero-dependency
//! work-stealing). In the browser, the `wasm-threads` feature fans the same chunks out over rayon,
//! whose pool is Web Workers sharing one linear memory (bootstrapped by the JS host; see
//! `wasm_bindgen_rayon`). Without that feature wasm32 runs the chunks sequentially.
//!
//! The [`Reducer`] monoid is what makes this safe to have two of: because the chunk set, each
//! chunk's seed, and the merge order are all fixed by `(seed, n)` alone, **every executor produces
//! bit-identical results** — including a single thread. The executor is free to be whatever is
//! fastest on the target; it can never be the reason an answer changed.

use std::ops::Range;

use crate::backend::{compile_root, Runner};
use crate::error::{NoiseError, Result};
use crate::exec::{CancelToken, StopCause};
// `Program` (the `&dyn` seam threads hand around) only exists on a threaded executor.
#[cfg(threaded)]
use crate::backend::Program;
use crate::dist::{RvGraph, RvId};
use crate::sampler::Moments;
// The deadline `Instant` travels to workers explicitly (thread-locals are invisible inside a
// scope), so the import is only needed where workers exist — and in the native tests.
#[cfg(any(threaded, test))]
use web_time::Instant;

/// Samples per chunk: 16 batches. Small enough that even a default `N=1e6` run yields ~60 chunks
/// — enough to keep a many-core machine load-balanced — yet large enough that per-chunk setup and
/// accumulator-merge stay negligible against the sampling work. **Fixed** (not thread-derived) so
/// chunking is deterministic: the load-bearing choice for core-count-invariant results.
pub const CHUNK_SAMPLES: usize = 16 * crate::bytecode::BATCH; // 16_384

/// Lanes per **epoch** (Track E): the `u32` lane index below the driver caps a single keyed stream
/// at 2³² draws. Ranges beyond it run as consecutive epochs under per-epoch derived seeds.
const EPOCH_LANES: u64 = 1 << 32;

/// Below this many draws, thread-spawn + per-thread compile overhead outweighs the win, so we run
/// the (identical) sequential path. Determinism is unaffected — same chunks either way.
#[cfg(threaded)]
const PAR_MIN_SAMPLES: u64 = 1 << 18; // 262_144

/// A monoid over sample columns. Parallel soundness reduces to a single local property: `merge` is
/// associative, and `absorb` depends only on the column (not global order). Commutativity is NOT
/// required — the driver always folds in chunk-index order ([`fold_in_order`]) — so an
/// order-preserving merge like concatenation ([`CollectReducer`]) is exactly as deterministic as a
/// commutative one.
pub trait Reducer: Sync {
    type Acc: Send;
    fn identity(&self) -> Self::Acc;
    /// Fold one batch column of draws into `acc`. The column is **f32** (the lane type, Track B);
    /// every accumulator here widens to f64 on the way in, so *aggregation* — the thing that
    /// becomes a reported estimate — never loses precision to the narrow lanes.
    fn absorb(&self, acc: &mut Self::Acc, col: &[f32]);
    /// Combine two partial accumulators. Must be associative (with `identity` as the unit); the
    /// index-ordered fold supplies the ordering, so commutativity is optional.
    fn merge(&self, a: Self::Acc, b: Self::Acc) -> Self::Acc;

    /// If this reducer is a plain moments fold (count, Σx, Σx²), say so — the GPU backend then
    /// folds **on the device** and reads back per-workgroup partial sums instead of every lane
    /// (PLAN-PRECISION Track F). `None` (the default) keeps the full-column readback path; a
    /// reducer that returns `Some` must also implement [`Reducer::absorb_moments`].
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn moments_mode(&self) -> Option<MomentsMode> {
        None
    }

    /// Fold one on-GPU partial — `count` lanes whose draws summed to `sum` / `sum_sq` — into `acc`.
    /// Only called when [`Reducer::moments_mode`] returned `Some`.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn absorb_moments(&self, _acc: &mut Self::Acc, _count: u64, _sum: f64, _sum_sq: f64) {
        unreachable!("absorb_moments requires moments_mode() == Some(_)")
    }
}

/// How a reduce-mode GPU shader treats each lane's value (PLAN-PRECISION Track F). Mirrors the two
/// moments reducers, which are the only reductions a shader can fold on-device. (Only the `gpu`
/// feature consumes it — the trait hooks exist unconditionally so reducers don't fork on the
/// feature.)
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MomentsMode {
    /// Every lane folds ([`MomentsReducer`]) — a NaN draw poisons the sums, exactly as on the CPU.
    All,
    /// NaN lanes are skipped and not counted ([`CondMomentsReducer`]) — the conditioning sentinel.
    SkipNan,
}

/// The result of a (possibly soft-stopped) reduction: the folded accumulator plus why it stopped
/// short, if it did. `stopped == None` means the whole requested range was folded.
pub struct Reduced<A> {
    pub acc: A,
    pub stopped: Option<StopCause>,
}

/// The derived seed of epoch `e` (Track E). Epoch 0 keeps the raw seed — so every stream that
/// exists today is bit-unchanged — and later epochs hash `(seed, e)` through `squares64` (keyed by
/// the run's own [`crate::rng::Key`]), giving statistically independent, deterministic streams.
fn epoch_seed(seed: u64, e: u64) -> u64 {
    if e == 0 {
        seed
    } else {
        crate::rng::squares64(e, crate::rng::Key::from_seed(seed).as_u64())
    }
}

/// Run chunk `chunk` (global index; covering epoch-local lanes `[chunk*CHUNK - epoch_base ..)`,
/// clamped to `end`) into a fresh accumulator, reusing this worker's `runner`. Counter keying
/// (PLAN-PREGPU Track C) makes a chunk just a lane range: positioning the runner at the chunk's
/// epoch-local start IS the per-chunk isolation — no derived per-chunk seed, and thread-count
/// invariance is by construction (the draw at lane `i` is a pure function of `(seed, i, source)`).
fn reduce_chunk<R: Reducer>(
    runner: &mut dyn Runner,
    r: &R,
    epoch_base: u64,
    end: u64,
    chunk: u64,
    seed: u64,
) -> R::Acc {
    let start = chunk * CHUNK_SAMPLES as u64;
    let len = (CHUNK_SAMPLES as u64).min(end - start) as usize;
    // Epoch-local lane (Track E): the driver split the range at 2³² boundaries, so this always fits.
    let lane = u32::try_from(start - epoch_base).expect("epoch-local lane exceeds 2^32");
    runner.position(seed, lane);
    let cap = runner.batch_cap();
    let mut acc = r.identity();
    let mut rem = len;
    while rem > 0 {
        let take = rem.min(cap);
        r.absorb(&mut acc, runner.next_batch(take));
        rem -= take;
    }
    acc
}

/// Merge per-chunk accumulators into `carry` in **chunk-index order**, so the fold is identical
/// regardless of how chunks were distributed across threads (the determinism guarantee) — and, for
/// a resumed range, identical to a single run over the union (the carry is the running left fold).
/// (Only the parallel drivers need it — the sequential path folds in order by construction.)
#[cfg(threaded)]
fn fold_in_order<R: Reducer>(r: &R, mut carry: R::Acc, mut indexed: Vec<(u64, R::Acc)>) -> R::Acc {
    indexed.sort_by_key(|(i, _)| *i);
    for (_, a) in indexed {
        carry = r.merge(carry, a);
    }
    carry
}

/// Drive a reduction over `n` draws of `root` — the fixed-`n` entry point ([`run_reduction_range`]
/// over `0..n` from a fresh accumulator). Parallel on native for large `n`, otherwise sequential —
/// always over the same deterministic chunks, so the result never depends on the thread count.
///
/// **Cancellable** (PLAN-PREGPU Track A): a tripped hard token aborts with
/// [`NoiseError::cancelled`] — that partial answer must never escape. **Soft-stoppable**
/// (PLAN-PRECISION Track H): a tripped soft flag or a passed `max_time` deadline folds the chunks
/// that completed and returns them — the accumulator's own count is the honest sample size, and
/// the run-level warning is raised by the engine (which sees the tripped flag). Callers that need
/// the stop marker itself use [`run_reduction_range`].
pub fn run_reduction<R: Reducer>(
    graph: &RvGraph,
    root: RvId,
    n: usize,
    seed: u64,
    r: &R,
) -> Result<R::Acc> {
    Ok(run_reduction_range(graph, root, 0..n as u64, seed, r, r.identity())?.acc)
}

/// Drive a reduction over the lane range `lanes` of `root`, folding into `carry` — the resumable
/// entry point (PLAN-PRECISION Track A). `lanes.start` must be chunk-aligned (a multiple of
/// [`CHUNK_SAMPLES`]); callers extend a previous range by passing its end and the accumulator it
/// returned. Because chunks are pure lane ranges merged in index order into the running carry, the
/// staged fold is **bit-identical** to a single run over the union range.
///
/// Ranges crossing a 2³² boundary split into epochs (Track E) — consecutive ≤ 2³²-lane reductions
/// under per-epoch seeds, folded in epoch order.
pub fn run_reduction_range<R: Reducer>(
    graph: &RvGraph,
    root: RvId,
    lanes: Range<u64>,
    seed: u64,
    r: &R,
    carry: R::Acc,
) -> Result<Reduced<R::Acc>> {
    debug_assert_eq!(
        lanes.start % CHUNK_SAMPLES as u64,
        0,
        "range starts must be chunk-aligned"
    );
    let mut acc = carry;
    if lanes.start >= lanes.end {
        return Ok(Reduced { acc, stopped: None });
    }
    // Split at epoch boundaries (Track E). The common (< 2³² lanes) case is a single pass.
    let (e0, e1) = (lanes.start / EPOCH_LANES, (lanes.end - 1) / EPOCH_LANES);
    for e in e0..=e1 {
        let base = e * EPOCH_LANES;
        let sub = lanes.start.max(base)..lanes.end.min(base + EPOCH_LANES);
        let out = run_epoch(graph, root, base, sub, epoch_seed(seed, e), r, acc)?;
        acc = out.acc;
        if out.stopped.is_some() {
            return Ok(Reduced { acc, stopped: out.stopped });
        }
    }
    Ok(Reduced { acc, stopped: None })
}

/// One epoch's reduction: global lanes `sub` (within the epoch starting at `epoch_base`), under
/// this epoch's `seed`. This is the driver proper — GPU hook, compile-once, thread fan-out.
fn run_epoch<R: Reducer>(
    graph: &RvGraph,
    root: RvId,
    epoch_base: u64,
    sub: Range<u64>,
    seed: u64,
    r: &R,
    carry: R::Acc,
) -> Result<Reduced<R::Acc>> {
    let n = sub.end - sub.start;
    // Per-forcing phase timing (NOISE_PROFILE=1, PLAN-DROP-JIT D0). Inert otherwise.
    let _prof = crate::profile::forcing("run_reduction", usize::try_from(n).unwrap_or(usize::MAX));
    let (c0, c1) = (
        sub.start / CHUNK_SAMPLES as u64,
        sub.end.div_ceil(CHUNK_SAMPLES as u64),
    );

    // Read token + deadline ONCE here, on the driver thread: `exec`'s thread-locals are invisible
    // inside `thread::scope`/`rayon::scope`, so both have to travel to the workers explicitly.
    let token = crate::exec::current();
    let deadline = crate::exec::deadline();
    // Cancelled before we even start (a host that aborted during parse/eval): don't compile.
    if is_cancelled(&token) {
        return Err(NoiseError::cancelled());
    }
    // Already soft-stopped (a deadline passed between statements): don't draw either.
    if let Some(cause) = crate::exec::stop_cause_of(token.as_ref(), deadline) {
        return Ok(Reduced { acc: carry, stopped: Some(cause) });
    }

    // The GPU takes the WHOLE epoch range or none of it (PLAN-WEBGPU G2). It hooks here rather than
    // at `Runner` because a dispatch wants >=256k lanes to be worth its ~1.2ms fixed cost, where a
    // `Runner` pulls 1024 at a time. A decline (no adapter, a cone it can't lower, or a gate saying
    // the CPU finishes first) hands the carry back and falls through to the code below, so this can
    // only change speed.
    #[cfg(feature = "gpu")]
    let carry = {
        let local = (sub.start - epoch_base)..(sub.end - epoch_base);
        match crate::gpu::try_reduce(graph, root, local, seed, r, token.as_ref(), deadline, carry)? {
            // `try_reduce` records the run stats itself, off the SIMPLIFIED cone.
            crate::gpu::GpuReduce::Done(out) => return Ok(out),
            crate::gpu::GpuReduce::Declined(c) => c,
        }
    };

    // Compile ONCE; the resulting program is shared (by reference) across all workers. Record the
    // run-time counters here on the driver thread (before fan-out) so workers stay lock-free.
    let (program, cost) = {
        let _s = crate::profile::span("compile");
        compile_root(graph, root, usize::try_from(n).unwrap_or(usize::MAX))
    };
    crate::stats::record(usize::try_from(n).unwrap_or(usize::MAX), cost.ops, cost.sources);
    crate::profile::set_ops(cost.ops);
    crate::profile::note("backend=cpu (gpu declined or absent)");
    let _reduce = crate::profile::span("reduce");

    // Snapshot this forcing's host input values ONCE, on the driver thread (like `token`); the shared
    // `Arc` travels to each worker's runner (PLAN-UNIFORM-INPUTS).
    let inputs = crate::input_rt::current();

    #[cfg(threaded)]
    {
        let threads = chosen_threads(n, c1 - c0);
        if threads > 1 {
            return run_parallel(
                &*program,
                epoch_base,
                sub.end,
                seed,
                r,
                c0..c1,
                threads,
                token.as_ref(),
                deadline,
                inputs,
                carry,
            );
        }
    }

    // Sequential: one runner, every chunk in order.
    let mut runner = program.runner(inputs);
    let mut acc = carry;
    let mut stopped = None;
    for i in c0..c1 {
        if is_cancelled(&token) {
            return Err(NoiseError::cancelled());
        }
        if let Some(cause) = crate::exec::stop_cause_of(token.as_ref(), deadline) {
            stopped = Some(cause);
            break;
        }
        acc = r.merge(acc, reduce_chunk(&mut *runner, r, epoch_base, sub.end, i, seed));
    }
    Ok(Reduced { acc, stopped })
}

/// One relaxed load — the whole per-chunk cancellation cost.
#[inline]
fn is_cancelled(token: &Option<CancelToken>) -> bool {
    token.as_ref().is_some_and(CancelToken::is_cancelled)
}

/// Worker count: clamp available cores to the chunk count, and stay single-threaded below the
/// parallel threshold. Only the *speed* depends on this — never the result.
#[cfg(threaded)]
fn chosen_threads(n: u64, n_chunks: u64) -> usize {
    if n < PAR_MIN_SAMPLES || n_chunks < 2 {
        return 1;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1);
    // In the browser the pool size is whatever the JS host passed to `initThreadPool` — asking rayon
    // is the only way to know, since `available_parallelism` is meaningless inside a worker.
    #[cfg(target_arch = "wasm32")]
    let cores = rayon::current_num_threads().max(1);
    cores.min(usize::try_from(n_chunks).unwrap_or(usize::MAX))
}

/// Native parallel driver: the already-compiled `program` is shared by reference; each worker
/// spins up a cheap [`Runner`] (no recompile) and steals chunks via an atomic counter. Per-chunk
/// accumulators are collected and merged in index order into the carry.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn run_parallel<R: Reducer>(
    program: &dyn Program,
    epoch_base: u64,
    end: u64,
    seed: u64,
    r: &R,
    chunks: Range<u64>,
    threads: usize,
    token: Option<&CancelToken>,
    deadline: Option<Instant>,
    inputs: std::sync::Arc<[f64]>,
    carry: R::Acc,
) -> Result<Reduced<R::Acc>> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let next = AtomicU64::new(chunks.start);
    // Borrow the shared work counter so each `move` worker closure captures the reference (Copy),
    // not the counter itself (the `inputs` clone is what each worker moves).
    let next = &next;
    let per_thread: Vec<Vec<(u64, R::Acc)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let inputs = inputs.clone();
                scope.spawn(move || {
                    // Cheap per-worker runner (scratch + RNG) over the SHARED compiled program.
                    let mut runner = program.runner(inputs);
                    let mut local: Vec<(u64, R::Acc)> = Vec::new();
                    loop {
                        // Every worker drops out on cancel, so the scope joins promptly instead of
                        // finishing the whole chunk list. The claimed-but-unrun chunks simply never
                        // land — the caller is about to discard the lot anyway.
                        if token.is_some_and(CancelToken::is_cancelled) {
                            break;
                        }
                        // Soft stop / deadline (Track H): stop CLAIMING chunks; everything already
                        // folded is kept. One flag load + (rarely) one clock read per 16,384 samples.
                        if crate::exec::stop_cause_of(token, deadline).is_some() {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= chunks.end {
                            break;
                        }
                        local.push((i, reduce_chunk(&mut *runner, r, epoch_base, end, i, seed)));
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("reduction worker panicked"))
            .collect()
    });
    // Checked AFTER the join, not inside it: a worker that saw the hard flag left its chunks
    // unfolded, so the collected set is partial and must not be combined into an answer.
    if token.is_some_and(CancelToken::is_cancelled) {
        return Err(NoiseError::cancelled());
    }
    let stopped = crate::exec::stop_cause_of(token, deadline);
    Ok(Reduced {
        acc: fold_in_order(r, carry, per_thread.into_iter().flatten().collect()),
        stopped,
    })
}

/// Browser parallel driver (`wasm-threads`): the same work-stealing shape as the native one, run on
/// rayon's pool — which, on wasm32, *is* a set of Web Workers sharing this module's linear memory
/// (the JS host bootstraps it via `wasm_bindgen_rayon`'s `initThreadPool`; wasm itself cannot spawn a
/// thread, so the pool must come from the host).
///
/// `rayon::scope` rather than a parallel iterator, so each worker builds its [`Runner`] *once* and
/// reuses it across chunks — matching the native driver. That matters more here than natively: a
/// wasm `Runner` instantiates the emitted kernel in its own worker's JS registry
/// (see [`crate::wasm_host`]), so a per-chunk runner would re-instantiate per chunk.
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
#[allow(clippy::too_many_arguments)]
fn run_parallel<R: Reducer>(
    program: &dyn Program,
    epoch_base: u64,
    end: u64,
    seed: u64,
    r: &R,
    chunks: Range<u64>,
    threads: usize,
    token: Option<&CancelToken>,
    deadline: Option<Instant>,
    inputs: std::sync::Arc<[f64]>,
    carry: R::Acc,
) -> Result<Reduced<R::Acc>> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    let next = AtomicU64::new(chunks.start);
    // `rayon::scope`'s spawns return `()`, so workers deposit their local runs here. Contended only
    // once per worker (at the end), not per chunk.
    let collected: Mutex<Vec<(u64, R::Acc)>> = Mutex::new(Vec::new());
    // Borrow the shared counter/sink so each `move` worker captures references (the per-worker
    // `inputs` clone is what it moves).
    let (next, collected) = (&next, &collected);

    rayon::scope(|scope| {
        for _ in 0..threads {
            let inputs = inputs.clone();
            scope.spawn(move |_| {
                // Cheap per-worker runner (scratch + RNG) over the SHARED compiled program.
                let mut runner = program.runner(inputs);
                let mut local: Vec<(u64, R::Acc)> = Vec::new();
                loop {
                    if token.is_some_and(CancelToken::is_cancelled) {
                        break;
                    }
                    if crate::exec::stop_cause_of(token, deadline).is_some() {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= chunks.end {
                        break;
                    }
                    local.push((i, reduce_chunk(&mut *runner, r, epoch_base, end, i, seed)));
                }
                collected.lock().expect("reduction mutex poisoned").append(&mut local);
            });
        }
    });

    if token.is_some_and(CancelToken::is_cancelled) {
        return Err(NoiseError::cancelled());
    }
    let stopped = crate::exec::stop_cause_of(token, deadline);
    // Chunks come back in completion order; `fold_in_order` sorts by index, which is what makes
    // the answer identical to the sequential run bit for bit.
    // `collected` is a shared `&Mutex` here (reborrowed above so the `move` workers capture it by
    // reference), so take the Vec out through the lock rather than consuming the Mutex. Bind it to a
    // local first so the `MutexGuard` temporary drops here, not at the end of the block (where it
    // would outlive the owned `collected` — E0597).
    let runs = std::mem::take(&mut *collected.lock().expect("reduction mutex poisoned"));
    Ok(Reduced {
        acc: fold_in_order(r, carry, runs),
        stopped,
    })
}

// --- the moments reducer (mean + population variance), powering P / E / Var ---

/// Raw power sums: count, Σx, Σx². We deliberately accumulate sums rather than running Welford
/// state — the per-element divide Welford needs (`delta / count`) is the hot-loop bottleneck for
/// cheap graphs *and* blocks vectorization. Σx / Σx² is a plain reduction the compiler turns into
/// SIMD (see [`MomentsReducer::absorb`]), and componentwise addition is an exactly-associative
/// monoid — so merging is trivial and the chunk-order fold stays bit-for-bit deterministic.
///
/// The tradeoff is numerical: variance via `E[X²] − E[X]²` can cancel when the mean dwarfs the
/// variance (`mean² / variance ≫ 1`). For this language's scales (dice, uniforms, modest normals)
/// that's immaterial; if huge-magnitude data ever matters, accumulate deviations from a provisional
/// mean per chunk. Note the merge is mathematically identical to one sequential accumulation, so
/// parallelism adds *no* extra error over the single-threaded path.
#[derive(Clone, Copy)]
pub struct MomentAcc {
    count: u64,
    sum: f64,
    sum_sq: f64,
}

impl MomentAcc {
    /// Number of draws folded in. For the conditional reducer this is the *in-condition* count
    /// `m ≈ n·P(C)` — the effective sample size a conditional estimate's standard error uses. For a
    /// soft-stopped run it is the true folded count — what makes the reported se honest (Track H).
    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn into_moments(self) -> Moments {
        if self.count == 0 {
            return Moments {
                mean: 0.0,
                variance: 0.0,
            };
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        // Population variance E[X²] − E[X]²; clamp away tiny negative rounding when var ≈ 0.
        let variance = (self.sum_sq / n - mean * mean).max(0.0);
        Moments { mean, variance }
    }
}

/// Number of independent accumulator lanes in [`MomentsReducer::absorb`]. Eight `f64` lanes give
/// LLVM enough parallel reduction chains to fill the vector units (and break the serial
/// `sum += x` dependency) while keeping the lane→element mapping a fixed function of position — so
/// the fold is identical regardless of thread count.
const REDUCE_LANES: usize = 8;

/// Σx / Σx² over a chunk, then componentwise-add to merge chunks. Both are exactly associative, so
/// the merge is a commutative monoid (the parallel-correctness requirement) and the index-ordered
/// fold is bit-reproducible.
pub struct MomentsReducer;

impl Reducer for MomentsReducer {
    type Acc = MomentAcc;

    fn identity(&self) -> MomentAcc {
        MomentAcc {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn absorb(&self, acc: &mut MomentAcc, col: &[f32]) {
        // Element `i` always lands in lane `i % REDUCE_LANES` — a fixed mapping, so the partial
        // sums (and thus the final value) don't depend on scheduling. The `chunks_exact` + const
        // inner loop is the idiom LLVM autovectorizes into NEON/AVX reduction code; the f32→f64
        // widen is itself a vector convert, and the accumulators stay f64 (Track B: narrow lanes,
        // wide sums — `x·x` in f32 would lose half the bits of Σx² before it is ever added).
        let mut lane_sum = [0.0f64; REDUCE_LANES];
        let mut lane_sq = [0.0f64; REDUCE_LANES];
        let mut chunks = col.chunks_exact(REDUCE_LANES);
        for c in chunks.by_ref() {
            for k in 0..REDUCE_LANES {
                let x = c[k] as f64;
                lane_sum[k] += x;
                lane_sq[k] += x * x;
            }
        }
        // Tail (fewer than REDUCE_LANES elements), then fold the lanes — fixed order both times.
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for &x in chunks.remainder() {
            let x = x as f64;
            sum += x;
            sum_sq += x * x;
        }
        for k in 0..REDUCE_LANES {
            sum += lane_sum[k];
            sum_sq += lane_sq[k];
        }
        acc.count += col.len() as u64;
        acc.sum += sum;
        acc.sum_sq += sum_sq;
    }

    fn merge(&self, a: MomentAcc, b: MomentAcc) -> MomentAcc {
        MomentAcc {
            count: a.count + b.count,
            sum: a.sum + b.sum,
            sum_sq: a.sum_sq + b.sum_sq,
        }
    }

    fn moments_mode(&self) -> Option<MomentsMode> {
        Some(MomentsMode::All)
    }

    fn absorb_moments(&self, acc: &mut MomentAcc, count: u64, sum: f64, sum_sq: f64) {
        acc.count += count;
        acc.sum += sum;
        acc.sum_sq += sum_sq;
    }
}

/// Like [`MomentsReducer`], but lanes whose draw is **NaN are skipped** rather than folded. This is
/// the conditioning sentinel: a query `P(A | C)` / `E(X | C)` compiles to the single root
/// `select(C, quantity, NaN)` — `quantity` on the lanes where `C` holds, `NaN` elsewhere — so one
/// sampling pass draws event and condition *jointly* (shared upstream draws), and this reducer
/// averages over exactly the in-condition lanes. `count` comes out as the in-condition sample size
/// `m`, which the standard error uses. (No SIMD lanes here: the per-element `is_nan` branch already
/// breaks vectorization, and conditioning is not the hot path.)
///
/// KNOWN HOLE (finding B2): skipping *all* NaN lanes conflates "condition false" with "the quantity
/// is NaN on an in-condition lane". An in-condition NaN quantity (e.g. `math::log(X)` where `X < 0`
/// but the condition still holds) is silently dropped instead of propagated, biasing the estimate
/// and tightening `m`'s standard error dishonestly. The fix is a dedicated condition column (see the
/// note in `Engine::query_cond`); deferred because it would ripple through the single-column
/// `Reducer`/`Runner` interface (the wasm backend included).
pub struct CondMomentsReducer;

impl Reducer for CondMomentsReducer {
    type Acc = MomentAcc;

    fn identity(&self) -> MomentAcc {
        MomentAcc {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn absorb(&self, acc: &mut MomentAcc, col: &[f32]) {
        for &x in col {
            if !x.is_nan() {
                let x = x as f64;
                acc.count += 1;
                acc.sum += x;
                acc.sum_sq += x * x;
            }
        }
    }

    fn merge(&self, a: MomentAcc, b: MomentAcc) -> MomentAcc {
        MomentAcc {
            count: a.count + b.count,
            sum: a.sum + b.sum,
            sum_sq: a.sum_sq + b.sum_sq,
        }
    }

    fn moments_mode(&self) -> Option<MomentsMode> {
        Some(MomentsMode::SkipNan)
    }

    fn absorb_moments(&self, acc: &mut MomentAcc, count: u64, sum: f64, sum_sq: f64) {
        acc.count += count;
        acc.sum += sum;
        acc.sum_sq += sum_sq;
    }
}

// --- the collect reducer (raw draws), powering Q / quantiles ---

/// Collects the raw draws themselves — the reduction behind `Q` (quantiles, PLAN-PERF-2 item 6). A
/// quantile is an order statistic, not a monoid fold, so nothing compresses: the accumulator is the
/// chunk's draw vector and `merge` is concatenation. Only the *sampling* parallelizes; the caller
/// sorts/selects centrally. Concatenation is associative but not commutative — deterministic anyway
/// because the driver folds in chunk-index order (see [`Reducer::merge`]), and the downstream sort
/// erases concatenation order entirely. So the collected vector, like the moments, is bit-identical
/// for any thread count — and, since counter keying made a chunk a pure lane range, it IS
/// `sample_n`'s stream, element for element (see `sampler::sample_n_par`).
///
/// `skip_nan` drops NaN lanes instead of collecting them — the conditioning sentinel of a
/// `select(C, x, NaN)` root (`Q(x | C, q)`), mirroring [`CondMomentsReducer`] including its KNOWN
/// HOLE (finding B2): an in-condition NaN quantity is dropped, not propagated.
pub struct CollectReducer {
    pub skip_nan: bool,
}

impl Reducer for CollectReducer {
    type Acc = Vec<f64>;

    fn identity(&self) -> Vec<f64> {
        Vec::new()
    }

    fn absorb(&self, acc: &mut Vec<f64>, col: &[f32]) {
        // First batch of a chunk: reserve the whole chunk up front so a chunk costs exactly one
        // allocation (a chunk never exceeds CHUNK_SAMPLES draws — no doubling churn per batch).
        // The collected vector is f64 (the public draw type — quantiles/plots/stats upstream are
        // untouched by Track B); widening happens here, at the copy that was already happening.
        if acc.capacity() == 0 {
            acc.reserve(CHUNK_SAMPLES);
        }
        if self.skip_nan {
            acc.extend(col.iter().filter(|x| !x.is_nan()).map(|&x| x as f64));
        } else {
            acc.extend(col.iter().map(|&x| x as f64));
        }
    }

    fn merge(&self, mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
        // Index-ordered concatenation. The empty fast path hands chunk 0's buffer straight to the
        // fold instead of copying it into the identity; thereafter `append` doubles, so the whole
        // n-element concat is O(n) amortized — noise against the sampling it follows.
        if a.is_empty() {
            return b;
        }
        a.append(&mut b);
        a
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::Value;

    fn pi_graph() -> (crate::eval::Engine, RvId) {
        let mut eng = crate::eval::Engine::new();
        let rv = eng
            .run_rv("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X*X + Y*Y < 1")
            .unwrap();
        let id = match rv {
            Value::Dist(id) => id,
            other => panic!("expected dist, got {other:?}"),
        };
        (eng, id)
    }

    /// Drive the parallel driver over `0..n` from a fresh accumulator (the old test surface).
    fn par<R: Reducer>(
        program: &dyn Program,
        n: usize,
        seed: u64,
        r: &R,
        threads: usize,
    ) -> R::Acc {
        let n_chunks = (n as u64).div_ceil(CHUNK_SAMPLES as u64);
        run_parallel(
            program,
            0,
            n as u64,
            seed,
            r,
            0..n_chunks,
            threads,
            None,
            None,
            std::sync::Arc::from(&[] as &[f64]),
            r.identity(),
        )
        .unwrap()
        .acc
    }

    /// The determinism guarantee: the reduced moments are **bit-for-bit identical** for any thread
    /// count, because chunks are fixed-seeded and merged in chunk-index order. This is the property
    /// that makes the answer independent of the machine's core count.
    #[test]
    fn parallel_result_is_bit_identical_across_thread_counts() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let (n, seed) = (500_000usize, 12_345u64);
        let r = MomentsReducer;

        // Compile once, share the program across all the thread-count variations.
        let (program, _cost) = compile_root(g, id, n);
        let base = par(&*program, n, seed, &r, 1).into_moments();
        for t in [2usize, 3, 5, 8] {
            let m = par(&*program, n, seed, &r, t).into_moments();
            assert_eq!(
                m.mean.to_bits(),
                base.mean.to_bits(),
                "mean differs at {t} threads"
            );
            assert_eq!(
                m.variance.to_bits(),
                base.variance.to_bits(),
                "variance differs at {t} threads"
            );
        }
        // ...and it actually estimates π/4 ≈ 0.785.
        assert!(
            (base.mean - std::f64::consts::FRAC_PI_4).abs() < 2e-3,
            "mean = {}",
            base.mean
        );
    }

    /// The Track A invariant everything hangs off: a reduction over `0..n₁` extended by `n₁..n₂`
    /// (through the carried accumulator) is **bit-identical** to the single run over `0..n₂` — on
    /// the public driver, at stage boundaries that are and are not thread-count-parallel.
    #[test]
    fn staged_range_reduction_is_bit_identical_to_single_run() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let (n1, n2, seed) = (65_536u64, 500_000u64, 7u64);
        let r = MomentsReducer;

        let single = run_reduction(g, id, n2 as usize, seed, &r).unwrap();

        let stage1 = run_reduction_range(g, id, 0..n1, seed, &r, r.identity()).unwrap();
        assert!(stage1.stopped.is_none());
        let stage2 = run_reduction_range(g, id, n1..n2, seed, &r, stage1.acc).unwrap();
        assert!(stage2.stopped.is_none());
        let staged = stage2.acc;

        assert_eq!(staged.count(), single.count());
        assert_eq!(staged.sum.to_bits(), single.sum.to_bits());
        assert_eq!(staged.sum_sq.to_bits(), single.sum_sq.to_bits());
    }

    /// Epoch seeds: epoch 0 must keep the raw seed (every existing stream unchanged); later epochs
    /// must differ from it and from each other (independent streams).
    #[test]
    fn epoch_seeds_are_stable_and_distinct() {
        assert_eq!(epoch_seed(42, 0), 42);
        let e1 = epoch_seed(42, 1);
        let e2 = epoch_seed(42, 2);
        assert_ne!(e1, 42);
        assert_ne!(e1, e2);
        // Deterministic: same inputs, same seed.
        assert_eq!(e1, epoch_seed(42, 1));
    }

    /// Track E: a range crossing the 2³² lane boundary — where the old driver panicked — runs as
    /// two epochs, folds the full requested count, and a staged split at the boundary is
    /// bit-identical to the single crossing run. Epoch 1 must also actually draw a *different*
    /// stream than epoch 0's opening lanes (the derived seed at work).
    #[test]
    fn ranges_across_the_epoch_boundary_fold_and_stage_identically() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let r = MomentsReducer;
        let seed = 5u64;
        let lo = EPOCH_LANES - 2 * CHUNK_SAMPLES as u64;
        let hi = EPOCH_LANES + 2 * CHUNK_SAMPLES as u64;

        let single = run_reduction_range(g, id, lo..hi, seed, &r, r.identity()).unwrap();
        assert!(single.stopped.is_none());
        assert_eq!(single.acc.count(), hi - lo);

        let stage1 = run_reduction_range(g, id, lo..EPOCH_LANES, seed, &r, r.identity()).unwrap();
        let stage2 = run_reduction_range(g, id, EPOCH_LANES..hi, seed, &r, stage1.acc).unwrap();
        assert_eq!(stage2.acc.count(), single.acc.count());
        assert_eq!(stage2.acc.sum.to_bits(), single.acc.sum.to_bits());
        assert_eq!(stage2.acc.sum_sq.to_bits(), single.acc.sum_sq.to_bits());

        // Epoch 1's lanes 0.. are NOT epoch 0's lanes 0.. — the derived per-epoch seed gives an
        // independent stream, not a replay.
        let n = 2 * CHUNK_SAMPLES as u64;
        let epoch0 = run_reduction_range(g, id, 0..n, seed, &r, r.identity()).unwrap();
        let epoch1 = run_reduction_range(g, id, EPOCH_LANES..EPOCH_LANES + n, seed, &r, r.identity())
            .unwrap();
        assert_ne!(
            epoch0.acc.sum.to_bits(),
            epoch1.acc.sum.to_bits(),
            "epoch 1 must draw an independent stream"
        );
    }

    /// A soft-stopped reduction folds what completed and reports the cause — and the accumulator's
    /// count is the true folded count (what makes the downstream se honest, Track H).
    #[test]
    fn soft_stop_returns_partial_fold_with_true_count() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let token = crate::exec::CancelToken::new();
        token.stop();
        let _g = crate::exec::install(&token);
        let r = MomentsReducer;
        let out = run_reduction_range(g, id, 0..1_000_000, 3, &r, r.identity()).unwrap();
        assert_eq!(out.stopped, Some(StopCause::User));
        // Stopped before the first chunk was claimed: zero draws, honestly reported.
        assert_eq!(out.acc.count(), 0);
    }

    /// A passed deadline soft-stops the reduction with `StopCause::Time` and keeps the completed
    /// chunks (here: none, since the deadline is already past when the run starts).
    #[test]
    fn deadline_soft_stops_with_time_cause() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let token = crate::exec::CancelToken::new();
        let _g = crate::exec::install(&token);
        let _d = crate::exec::install_deadline(Some(
            Instant::now() - std::time::Duration::from_millis(1),
        ));
        let r = MomentsReducer;
        let out = run_reduction_range(g, id, 0..1_000_000, 3, &r, r.identity()).unwrap();
        assert_eq!(out.stopped, Some(StopCause::Time));
        assert_eq!(out.acc.count(), 0);
    }

    /// Same-process A/B of the new Σx/Σx² absorb vs the old streaming-Welford absorb over an
    /// identical large column — isolates the reduction speedup (no generation / multicore noise).
    /// Ignored by default; run with:
    /// `cargo test -p noise-core --release -- --ignored --nocapture bench_reduce`
    #[test]
    #[ignore]
    fn bench_reduce_absorb() {
        use std::time::Instant;

        // A representative column (small-magnitude draws, like dice/uniform outputs).
        let n = 4_000_000usize;
        let mut col = vec![0.0f32; n]; // lanes are f32 (Track B); the accumulators stay f64
        let mut z = 0x1234_5678u64;
        for x in col.iter_mut() {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((z >> 40) as f32) * (1.0 / (1u32 << 24) as f32) * 12.0;
        }

        // New: Σx/Σx² (the shipped MomentsReducer).
        let r = MomentsReducer;
        let t = Instant::now();
        let mut acc = r.identity();
        r.absorb(&mut acc, &col);
        let new_mps = n as f64 / t.elapsed().as_secs_f64() / 1e6;
        let new_m = acc.into_moments();

        // Old: streaming Welford with the per-element divide.
        let t = Instant::now();
        let (mut count, mut mean, mut m2) = (0u64, 0.0f64, 0.0f64);
        for &x in &col {
            count += 1;
            let x = x as f64;
            let delta = x - mean;
            mean += delta / count as f64;
            m2 += delta * (x - mean);
        }
        let old_mps = n as f64 / t.elapsed().as_secs_f64() / 1e6;

        println!("\n  reduce absorb (single thread, M elem/s):");
        println!(
            "    welford(old) {old_mps:7.0}   sum/sumsq(new) {new_mps:7.0}   speedup {:.2}x",
            new_mps / old_mps
        );
        // Sanity: both estimate the same moments.
        assert!(
            (new_m.mean - mean).abs() < 1e-6,
            "mean mismatch {} vs {}",
            new_m.mean,
            mean
        );
        assert!(
            (new_m.variance - m2 / count as f64).abs() < 1e-6,
            "var mismatch"
        );
    }

    /// The quantile collection shares the moments' determinism guarantee: the collected draw
    /// vector is **bit-identical, element for element,** for any thread count (fixed chunk seeds +
    /// index-ordered concatenation), and the public driver — whatever thread count it picks,
    /// including the sequential below-threshold branch — returns exactly that vector. This is the
    /// invariant that makes a `Q(...)` answer independent of the machine's core count.
    #[test]
    fn collected_draws_are_bit_identical_across_thread_counts() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let (n, seed) = (500_000usize, 9_876u64);
        let r = CollectReducer { skip_nan: false };

        let (program, _cost) = compile_root(g, id, n);
        let base = par(&*program, n, seed, &r, 1);
        assert_eq!(base.len(), n);
        for t in [2usize, 3, 5, 8] {
            let v = par(&*program, n, seed, &r, t);
            assert!(
                v.len() == n && v.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
                "collected draws differ at {t} threads"
            );
        }
        // The public entry (its own compile + threshold + thread choice) agrees bit for bit.
        let public = run_reduction(g, id, n, seed, &r).unwrap();
        assert!(
            public.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
            "run_reduction disagrees with the pinned 1-thread collection"
        );
    }

    /// Before/after readout for PLAN-PERF-2 item 6: the OLD sequential quantile collection
    /// (`sample_n` — one runner, one ordered stream) vs the NEW parallel chunked collection
    /// (`sample_n_par`), each followed by the (shared, central) sort — a 5M-draw `Q` over a normal
    /// cone. Ignored; run with:
    /// `cargo test -p noise-core [--features gpu] --release -- --ignored --nocapture bench_quantile_collect`
    #[test]
    #[ignore]
    fn bench_quantile_collect() {
        use std::time::Instant;

        let mut eng = crate::eval::Engine::new();
        let rv = eng.run_rv("use rand; Z ~ normal(0, 1); Z + Z * Z").unwrap();
        let id = match rv {
            Value::Dist(id) => id,
            other => panic!("expected dist, got {other:?}"),
        };
        let g = eng.graph();
        let (n, seed) = (5_000_000usize, 0u64);

        let time = |f: &dyn Fn() -> Vec<f64>| {
            let _ = f(); // warm (compile cache, allocator)
            let t = Instant::now();
            let mut draws = f();
            draws.sort_by(f64::total_cmp);
            std::hint::black_box(crate::num::quantile_sorted(&draws, 0.9));
            t.elapsed().as_secs_f64() * 1e3
        };
        let old_ms = time(&|| crate::sampler::sample_n(g, id, n, seed).unwrap());
        let new_ms = time(&|| crate::sampler::sample_n_par(g, id, n, seed).unwrap());
        println!("\n  Q collection + sort, 5M draws of a normal cone (ms):");
        println!(
            "    sequential(old) {old_ms:8.1}   parallel(new) {new_ms:8.1}   speedup {:.2}x",
            old_ms / new_ms
        );
    }

    /// Repeated `moments` calls are bit-identical despite work-stealing (index-ordered merge).
    #[test]
    fn moments_is_bit_reproducible_across_runs() {
        let (eng, id) = pi_graph();
        let a = crate::sampler::moments(eng.graph(), id, 600_000, 7).unwrap();
        let b = crate::sampler::moments(eng.graph(), id, 600_000, 7).unwrap();
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        assert_eq!(a.variance.to_bits(), b.variance.to_bits());
    }

    /// End-to-end multicore scaling: time a full `moments` workload (generate + reduce) at 1 thread
    /// vs all cores, at a sample count large enough to amortize thread-spawn overhead. This is the
    /// "the simplest program uses every core for free" number. Ignored; run with:
    /// `cargo test -p noise-core [--features gpu] --release -- --ignored --nocapture bench_parallel_scaling`
    #[test]
    #[ignore]
    fn bench_parallel_scaling() {
        use std::time::Instant;

        let (eng, id) = pi_graph();
        let g = eng.graph();
        let n = 64_000_000usize; // big enough that per-call thread spawn is negligible
        let seed = 0xC0FFEE;
        let r = MomentsReducer;
        let (program, _cost) = compile_root(g, id, n); // compile ONCE, shared across thread counts

        let drive = |threads: usize| {
            let _ = par(&*program, n, seed, &r, threads); // warm up
            let t = Instant::now();
            let m = par(&*program, n, seed, &r, threads);
            std::hint::black_box(m);
            n as f64 / t.elapsed().as_secs_f64() / 1e6
        };

        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1);
        let one = drive(1);
        let all = drive(cores);
        println!("\n  moments(pi) end-to-end (generate + reduce), M samples/s:");
        println!(
            "    1 thread {one:8.0}   {cores} threads {all:8.0}   scaling {:.1}x",
            all / one
        );
    }
}
