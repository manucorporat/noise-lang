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

    // Compile ONCE; the resulting program is shared (by reference) across all workers.
    let program = compile_root(graph, root);

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

/// Welford state: count, running mean, and sum of squared deviations.
#[derive(Clone, Copy)]
pub struct MomentAcc {
    count: u64,
    mean: f64,
    m2: f64,
}

impl MomentAcc {
    pub fn into_moments(self) -> Moments {
        let variance = if self.count > 0 { self.m2 / self.count as f64 } else { 0.0 };
        Moments { mean: self.mean, variance }
    }
}

/// Streaming Welford within a chunk; Chan's parallel formula to merge chunks. Both are numerically
/// stable, and the merge is associative + commutative (the parallel-correctness requirement).
pub struct MomentsReducer;

impl Reducer for MomentsReducer {
    type Acc = MomentAcc;

    fn identity(&self) -> MomentAcc {
        MomentAcc { count: 0, mean: 0.0, m2: 0.0 }
    }

    fn absorb(&self, acc: &mut MomentAcc, col: &[f64]) {
        for &x in col {
            acc.count += 1;
            let delta = x - acc.mean;
            acc.mean += delta / acc.count as f64;
            let delta2 = x - acc.mean;
            acc.m2 += delta * delta2;
        }
    }

    fn merge(&self, a: MomentAcc, b: MomentAcc) -> MomentAcc {
        if a.count == 0 {
            return b;
        }
        if b.count == 0 {
            return a;
        }
        let count = a.count + b.count;
        let delta = b.mean - a.mean;
        let cf = count as f64;
        let mean = a.mean + delta * (b.count as f64 / cf);
        let m2 = a.m2 + b.m2 + delta * delta * (a.count as f64 * b.count as f64 / cf);
        MomentAcc { count, mean, m2 }
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
        let program = compile_root(g, id);
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

    /// Repeated `moments` calls are bit-identical despite work-stealing (index-ordered merge).
    #[test]
    fn moments_is_bit_reproducible_across_runs() {
        let (eng, id) = pi_graph();
        let a = crate::sampler::moments(eng.graph(), id, 600_000, 7);
        let b = crate::sampler::moments(eng.graph(), id, 600_000, 7);
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        assert_eq!(a.variance.to_bits(), b.variance.to_bits());
    }
}
