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
//! Executor — two of them, one monoid. Native uses `std::thread::scope` (zero-dependency
//! work-stealing). In the browser, the `wasm-threads` feature fans the same chunks out over rayon,
//! whose pool is Web Workers sharing one linear memory (bootstrapped by the JS host; see
//! `wasm_bindgen_rayon`). Without that feature wasm32 runs the chunks sequentially.
//!
//! The [`Reducer`] monoid is what makes this safe to have two of: because the chunk set, each
//! chunk's seed, and the merge order are all fixed by `(seed, n)` alone, **every executor produces
//! bit-identical results** — including a single thread. The executor is free to be whatever is
//! fastest on the target; it can never be the reason an answer changed.

use crate::backend::{compile_root, Runner};
// `Program` (the `&dyn` seam threads hand around) only exists on a threaded executor.
#[cfg(threaded)]
use crate::backend::Program;
use crate::dist::{RvGraph, RvId};
use crate::sampler::Moments;

/// Samples per chunk: 16 batches. Small enough that even a default `N=1e6` run yields ~60 chunks
/// — enough to keep a many-core machine load-balanced — yet large enough that per-chunk setup and
/// accumulator-merge stay negligible against the sampling work. **Fixed** (not thread-derived) so
/// chunking is deterministic: the load-bearing choice for core-count-invariant results.
const CHUNK_SAMPLES: usize = 16 * crate::bytecode::BATCH; // 16_384

/// Below this many draws, thread-spawn + per-thread compile overhead outweighs the win, so we run
/// the (identical) sequential path. Determinism is unaffected — same chunks either way.
#[cfg(threaded)]
const PAR_MIN_SAMPLES: usize = 1 << 18; // 262_144

/// A monoid over sample columns. Parallel soundness reduces to a single local property: `merge` is
/// associative, and `absorb` depends only on the column (not global order). Commutativity is NOT
/// required — the driver always folds in chunk-index order ([`combine_in_order`]) — so an
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
}

/// Run chunk `chunk` (covering `[chunk*CHUNK .. )`, clamped to `n`) into a fresh accumulator,
/// reusing this worker's `runner`. Counter keying (PLAN-PREGPU Track C) makes a chunk just a
/// lane range: positioning the runner at `chunk * CHUNK_SAMPLES` IS the per-chunk isolation —
/// no derived per-chunk seed, and thread-count invariance is by construction (the draw at lane
/// `i` is a pure function of `(seed, i, source)`).
fn reduce_chunk<R: Reducer>(
    runner: &mut dyn Runner,
    r: &R,
    n: usize,
    chunk: usize,
    seed: u64,
) -> R::Acc {
    let start = chunk * CHUNK_SAMPLES;
    let len = CHUNK_SAMPLES.min(n - start);
    // One u32 of lane index caps a forcing at 2^32 draws (documented language boundary;
    // far above the op budget's practical caps).
    let lane = u32::try_from(start).expect("forcing exceeds 2^32 lanes");
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

/// Merge per-chunk accumulators in **chunk-index order**, so the fold is identical regardless of
/// how chunks were distributed across threads (the determinism guarantee).
fn combine_in_order<R: Reducer>(r: &R, mut indexed: Vec<(usize, R::Acc)>) -> R::Acc {
    indexed.sort_by_key(|(i, _)| *i);
    let mut acc = r.identity();
    for (_, a) in indexed {
        acc = r.merge(acc, a);
    }
    acc
}

/// Drive a reduction over `n` draws of `root`. Parallel on native for large `n`, otherwise
/// sequential — always over the same deterministic chunks, so the result never depends on the
/// thread count.
pub fn run_reduction<R: Reducer>(
    graph: &RvGraph,
    root: RvId,
    n: usize,
    seed: u64,
    r: &R,
) -> R::Acc {
    if n == 0 {
        return r.identity();
    }
    let n_chunks = n.div_ceil(CHUNK_SAMPLES);

    // Compile ONCE; the resulting program is shared (by reference) across all workers. Record the
    // run-time counters here on the driver thread (before fan-out) so workers stay lock-free.
    let (program, cost) = compile_root(graph, root, n);
    crate::stats::record(n, cost.ops, cost.sources);

    #[cfg(threaded)]
    {
        let threads = chosen_threads(n, n_chunks);
        if threads > 1 {
            return run_parallel(&*program, n, seed, r, n_chunks, threads);
        }
    }

    // Sequential: one runner, every chunk in order.
    let mut runner = program.runner();
    let indexed: Vec<(usize, R::Acc)> = (0..n_chunks)
        .map(|i| (i, reduce_chunk(&mut *runner, r, n, i, seed)))
        .collect();
    combine_in_order(r, indexed)
}

/// Worker count: clamp available cores to the chunk count, and stay single-threaded below the
/// parallel threshold. Only the *speed* depends on this — never the result.
#[cfg(threaded)]
fn chosen_threads(n: usize, n_chunks: usize) -> usize {
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
    cores.min(n_chunks)
}

/// Native parallel driver: the already-compiled `program` is shared by reference; each worker
/// spins up a cheap [`Runner`] (no recompile) and steals chunks via an atomic counter. Per-chunk
/// accumulators are collected and merged in index order.
#[cfg(not(target_arch = "wasm32"))]
fn run_parallel<R: Reducer>(
    program: &dyn Program,
    n: usize,
    seed: u64,
    r: &R,
    n_chunks: usize,
    threads: usize,
) -> R::Acc {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let per_thread: Vec<Vec<(usize, R::Acc)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    // Cheap per-worker runner (scratch + RNG) over the SHARED compiled program.
                    let mut runner = program.runner();
                    let mut local: Vec<(usize, R::Acc)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            break;
                        }
                        local.push((i, reduce_chunk(&mut *runner, r, n, i, seed)));
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
    combine_in_order(r, per_thread.into_iter().flatten().collect())
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
fn run_parallel<R: Reducer>(
    program: &dyn Program,
    n: usize,
    seed: u64,
    r: &R,
    n_chunks: usize,
    threads: usize,
) -> R::Acc {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let next = AtomicUsize::new(0);
    // `rayon::scope`'s spawns return `()`, so workers deposit their local runs here. Contended only
    // once per worker (at the end), not per chunk.
    let collected: Mutex<Vec<(usize, R::Acc)>> = Mutex::new(Vec::with_capacity(n_chunks));

    rayon::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|_| {
                // Cheap per-worker runner (scratch + RNG) over the SHARED compiled program.
                let mut runner = program.runner();
                let mut local: Vec<(usize, R::Acc)> = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_chunks {
                        break;
                    }
                    local.push((i, reduce_chunk(&mut *runner, r, n, i, seed)));
                }
                collected.lock().expect("reduction mutex poisoned").append(&mut local);
            });
        }
    });

    // Chunks come back in completion order; `combine_in_order` sorts by index, which is what makes
    // the answer identical to the sequential run bit for bit.
    combine_in_order(r, collected.into_inner().expect("reduction mutex poisoned"))
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
    /// `m ≈ n·P(C)` — the effective sample size a conditional estimate's standard error uses.
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
/// `Reducer`/`Runner` interface (JIT/wasm included).
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

    /// The determinism guarantee: the reduced moments are **bit-for-bit identical** for any thread
    /// count, because chunks are fixed-seeded and merged in chunk-index order. This is the property
    /// that makes the answer independent of the machine's core count.
    #[test]
    fn parallel_result_is_bit_identical_across_thread_counts() {
        let (eng, id) = pi_graph();
        let g = eng.graph();
        let (n, seed) = (500_000usize, 12_345u64);
        let n_chunks = n.div_ceil(CHUNK_SAMPLES);
        let r = MomentsReducer;

        // Compile once, share the program across all the thread-count variations.
        let (program, _cost) = compile_root(g, id, n);
        let base = run_parallel(&*program, n, seed, &r, n_chunks, 1).into_moments();
        for t in [2usize, 3, 5, 8] {
            let m = run_parallel(&*program, n, seed, &r, n_chunks, t).into_moments();
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
        let n_chunks = n.div_ceil(CHUNK_SAMPLES);
        let r = CollectReducer { skip_nan: false };

        let (program, _cost) = compile_root(g, id, n);
        let base = run_parallel(&*program, n, seed, &r, n_chunks, 1);
        assert_eq!(base.len(), n);
        for t in [2usize, 3, 5, 8] {
            let v = run_parallel(&*program, n, seed, &r, n_chunks, t);
            assert!(
                v.len() == n && v.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
                "collected draws differ at {t} threads"
            );
        }
        // The public entry (its own compile + threshold + thread choice) agrees bit for bit.
        let public = run_reduction(g, id, n, seed, &r);
        assert!(
            public.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
            "run_reduction disagrees with the pinned 1-thread collection"
        );
    }

    /// Before/after readout for PLAN-PERF-2 item 6: the OLD sequential quantile collection
    /// (`sample_n` — one runner, one ordered stream) vs the NEW parallel chunked collection
    /// (`sample_n_par`), each followed by the (shared, central) sort — a 5M-draw `Q` over a normal
    /// cone. Ignored; run with:
    /// `cargo test -p noise-core [--features jit] --release -- --ignored --nocapture bench_quantile_collect`
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
        let old_ms = time(&|| crate::sampler::sample_n(g, id, n, seed));
        let new_ms = time(&|| crate::sampler::sample_n_par(g, id, n, seed));
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
        let a = crate::sampler::moments(eng.graph(), id, 600_000, 7);
        let b = crate::sampler::moments(eng.graph(), id, 600_000, 7);
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        assert_eq!(a.variance.to_bits(), b.variance.to_bits());
    }

    /// End-to-end multicore scaling: time a full `moments` workload (generate + reduce) at 1 thread
    /// vs all cores, at a sample count large enough to amortize thread-spawn overhead. This is the
    /// "the simplest program uses every core for free" number. Ignored; run with:
    /// `cargo test -p noise-core [--features jit] --release -- --ignored --nocapture bench_parallel_scaling`
    #[test]
    #[ignore]
    fn bench_parallel_scaling() {
        use std::time::Instant;

        let (eng, id) = pi_graph();
        let g = eng.graph();
        let n = 64_000_000usize; // big enough that per-call thread spawn is negligible
        let seed = 0xC0FFEE;
        let n_chunks = n.div_ceil(CHUNK_SAMPLES);
        let r = MomentsReducer;
        let (program, _cost) = compile_root(g, id, n); // compile ONCE, shared across thread counts

        let drive = |threads: usize| {
            run_parallel(&*program, n, seed, &r, n_chunks, threads); // warm up
            let t = Instant::now();
            let m = run_parallel(&*program, n, seed, &r, n_chunks, threads);
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
