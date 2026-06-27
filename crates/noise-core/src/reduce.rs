//! Parallel, deterministic Monte-Carlo reduction (PLAN.md Phase 4 "multicore").
//!
//! Sampling is an embarrassingly parallel reduction: draws are independent and the quantities we
//! force (`P`, `E`, `Var`) are *monoid folds* over the sample stream. This module captures that
//! with a [`Reducer`] trait — a commutative monoid over sample columns — and a driver that runs it
//! over **fixed, deterministically-seeded chunks**.
//!
//! Determinism is the whole point: the chunk set and each chunk's seed depend only on `(seed, n)`,
//! never on thread scheduling, and per-chunk accumulators are merged in **chunk-index order**. So
//! the result is identical for any thread count — bit for bit. Threads change only wall-clock.
//!
//! Executor: native uses `std::thread::scope` (zero-dependency work-stealing); wasm32 always runs
//! the same chunks sequentially (browser parallelism is web-workers, handled above this layer).
//! The [`Reducer`] monoid is executor-agnostic, so a rayon backend could drop in unchanged.

use crate::backend::{compile_root, Program, Runner};
use crate::dist::{RvGraph, RvId};
use crate::sampler::Moments;

/// Samples per chunk: 16 batches. Small enough that even a default `N=1e6` run yields ~60 chunks
/// — enough to keep a many-core machine load-balanced — yet large enough that per-chunk reseed and
/// accumulator-merge stay negligible against the sampling work. **Fixed** (not thread-derived) so
/// chunking is deterministic: the load-bearing choice for core-count-invariant results.
const CHUNK_SAMPLES: usize = 16 * crate::bytecode::BATCH; // 16_384

/// Below this many draws, thread-spawn + per-thread compile overhead outweighs the win, so we run
/// the (identical) sequential path. Determinism is unaffected — same chunks either way.
#[cfg(not(target_arch = "wasm32"))]
const PAR_MIN_SAMPLES: usize = 1 << 18; // 262_144

/// A commutative monoid over sample columns. Parallel soundness reduces to a single local property:
/// `merge` is associative + commutative, and `absorb` depends only on the column (not global order).
pub trait Reducer: Sync {
    type Acc: Send;
    fn identity(&self) -> Self::Acc;
    /// Fold one batch column of draws into `acc`.
    fn absorb(&self, acc: &mut Self::Acc, col: &[f64]);
    /// Combine two partial accumulators. Must be associative + commutative.
    fn merge(&self, a: Self::Acc, b: Self::Acc) -> Self::Acc;
}

/// Deterministic per-chunk seed: a SplitMix64 mix of `(seed, chunk_index)`, giving each chunk a
/// well-separated xoshiro substream regardless of run order or core count.
fn chunk_seed(seed: u64, chunk: usize) -> u64 {
    let mut z = seed.wrapping_add((chunk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Run chunk `chunk` (covering `[chunk*CHUNK .. )`, clamped to `n`) into a fresh accumulator,
/// reusing this worker's `runner`.
fn reduce_chunk<R: Reducer>(
    runner: &mut dyn Runner,
    r: &R,
    n: usize,
    chunk: usize,
    seed: u64,
) -> R::Acc {
    let start = chunk * CHUNK_SAMPLES;
    let len = CHUNK_SAMPLES.min(n - start);
    runner.reseed(chunk_seed(seed, chunk));
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
    let (program, cost) = compile_root(graph, root);
    crate::stats::record(n, cost.ops, cost.sources);

    #[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
fn chosen_threads(n: usize, n_chunks: usize) -> usize {
    if n < PAR_MIN_SAMPLES || n_chunks < 2 {
        return 1;
    }
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
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
        handles.into_iter().map(|h| h.join().expect("reduction worker panicked")).collect()
    });
    combine_in_order(r, per_thread.into_iter().flatten().collect())
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
            return Moments { mean: 0.0, variance: 0.0 };
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
        MomentAcc { count: 0, sum: 0.0, sum_sq: 0.0 }
    }

    fn absorb(&self, acc: &mut MomentAcc, col: &[f64]) {
        // Element `i` always lands in lane `i % REDUCE_LANES` — a fixed mapping, so the partial
        // sums (and thus the final value) don't depend on scheduling. The `chunks_exact` + const
        // inner loop is the idiom LLVM autovectorizes into NEON/AVX reduction code.
        let mut lane_sum = [0.0f64; REDUCE_LANES];
        let mut lane_sq = [0.0f64; REDUCE_LANES];
        let mut chunks = col.chunks_exact(REDUCE_LANES);
        for c in chunks.by_ref() {
            for k in 0..REDUCE_LANES {
                let x = c[k];
                lane_sum[k] += x;
                lane_sq[k] += x * x;
            }
        }
        // Tail (fewer than REDUCE_LANES elements), then fold the lanes — fixed order both times.
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for &x in chunks.remainder() {
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
        MomentAcc { count: a.count + b.count, sum: a.sum + b.sum, sum_sq: a.sum_sq + b.sum_sq }
    }
}

/// Like [`MomentsReducer`], but lanes whose draw is **NaN are skipped** rather than folded. This is
/// the conditioning sentinel: a query `P(A | C)` / `E(X | C)` compiles to the single root
/// `select(C, quantity, NaN)` — `quantity` on the lanes where `C` holds, `NaN` elsewhere — so one
/// sampling pass draws event and condition *jointly* (shared upstream draws), and this reducer
/// averages over exactly the in-condition lanes. `count` comes out as the in-condition sample size
/// `m`, which the standard error uses. (No SIMD lanes here: the per-element `is_nan` branch already
/// breaks vectorization, and conditioning is not the hot path.)
pub struct CondMomentsReducer;

impl Reducer for CondMomentsReducer {
    type Acc = MomentAcc;

    fn identity(&self) -> MomentAcc {
        MomentAcc { count: 0, sum: 0.0, sum_sq: 0.0 }
    }

    fn absorb(&self, acc: &mut MomentAcc, col: &[f64]) {
        for &x in col {
            if !x.is_nan() {
                acc.count += 1;
                acc.sum += x;
                acc.sum_sq += x * x;
            }
        }
    }

    fn merge(&self, a: MomentAcc, b: MomentAcc) -> MomentAcc {
        MomentAcc { count: a.count + b.count, sum: a.sum + b.sum, sum_sq: a.sum_sq + b.sum_sq }
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
        let (program, _cost) = compile_root(g, id);
        let base = run_parallel(&*program, n, seed, &r, n_chunks, 1).into_moments();
        for t in [2usize, 3, 5, 8] {
            let m = run_parallel(&*program, n, seed, &r, n_chunks, t).into_moments();
            assert_eq!(m.mean.to_bits(), base.mean.to_bits(), "mean differs at {t} threads");
            assert_eq!(
                m.variance.to_bits(),
                base.variance.to_bits(),
                "variance differs at {t} threads"
            );
        }
        // ...and it actually estimates π/4 ≈ 0.785.
        assert!((base.mean - std::f64::consts::FRAC_PI_4).abs() < 2e-3, "mean = {}", base.mean);
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
        let mut col = vec![0.0f64; n];
        let mut z = 0x1234_5678u64;
        for x in col.iter_mut() {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((z >> 11) as f64) * (1.0 / (1u64 << 53) as f64) * 12.0;
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
            let delta = x - mean;
            mean += delta / count as f64;
            m2 += delta * (x - mean);
        }
        let old_mps = n as f64 / t.elapsed().as_secs_f64() / 1e6;

        println!("\n  reduce absorb (single thread, M elem/s):");
        println!("    welford(old) {old_mps:7.0}   sum/sumsq(new) {new_mps:7.0}   speedup {:.2}x", new_mps / old_mps);
        // Sanity: both estimate the same moments.
        assert!((new_m.mean - mean).abs() < 1e-6, "mean mismatch {} vs {}", new_m.mean, mean);
        assert!((new_m.variance - m2 / count as f64).abs() < 1e-6, "var mismatch");
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
        let (program, _cost) = compile_root(g, id); // compile ONCE, shared across thread counts

        let drive = |threads: usize| {
            run_parallel(&*program, n, seed, &r, n_chunks, threads); // warm up
            let t = Instant::now();
            let m = run_parallel(&*program, n, seed, &r, n_chunks, threads);
            std::hint::black_box(m);
            n as f64 / t.elapsed().as_secs_f64() / 1e6
        };

        let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
        let one = drive(1);
        let all = drive(cores);
        println!("\n  moments(pi) end-to-end (generate + reduce), M samples/s:");
        println!("    1 thread {one:8.0}   {cores} threads {all:8.0}   scaling {:.1}x", all / one);
    }
}
