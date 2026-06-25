//! Sampling driver — the only forcing path in Phase 2 (PLAN.md "batch sampler").
//!
//! Backend-independent: compile the RV cone into a [`Sampler`] once (via [`InterpBackend`], the
//! default), seed the RNG once, then loop `ceil(N / batch_cap)` batches. The final partial batch
//! is sliced to the true remaining length so over-count never biases moments. Swapping the
//! backend (e.g. a Cranelift JIT) changes only how a batch is produced, not this loop.

use crate::backend::compile_root;
use crate::dist::{RvGraph, RvId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Moments {
    pub mean: f64,
    pub variance: f64,
}

/// Run `n` draws of `root`, calling `sink` with each batch's filled root column slice.
/// The last batch may be partial (`len < BATCH`); the slice carries the true length.
pub fn for_each_batch(
    graph: &RvGraph,
    root: RvId,
    n: usize,
    seed: u64,
    mut sink: impl FnMut(&[f64]),
) {
    if n == 0 {
        return;
    }
    let program = compile_root(graph, root);
    let mut runner = program.runner();
    runner.reseed(seed);
    let cap = runner.batch_cap();

    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(cap);
        sink(runner.next_batch(take));
        remaining -= take;
    }
}

/// Collect raw draws (small N / tests that need the full vector).
pub fn sample_n(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for_each_batch(graph, root, n, seed, |col| out.extend_from_slice(col));
    out
}

/// Empirical mean + population variance over `n` draws — the forcing path behind `P`/`E`/`Var`.
///
/// Delegates to the parallel, deterministic monoid reduction in [`crate::reduce`]: draws are split
/// into fixed, independently-seeded chunks, folded with Welford, and merged in chunk-index order.
/// The result is identical for any thread count (native multicore for large `n`, sequential
/// otherwise and on wasm). Note this uses per-chunk seeding, so it does not consume the RNG stream
/// identically to [`sample_n`] (which stays a single ordered stream); both are deterministic in
/// `seed` and converge to the same values within Monte-Carlo error.
pub fn moments(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Moments {
    crate::reduce::run_reduction(graph, root, n, seed, &crate::reduce::MomentsReducer).into_moments()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{RvKind, RvNode, Source};

    fn const_graph(v: f64) -> (RvGraph, RvId) {
        let mut g = RvGraph::default();
        let id = g.push(RvNode::ConstNum(v), RvKind::Num);
        (g, id)
    }

    #[test]
    fn sample_n_zero_is_empty() {
        let (g, id) = const_graph(1.0);
        assert!(sample_n(&g, id, 0, 0).is_empty());
        // moments over zero draws is well-defined (no NaN), not a panic.
        assert_eq!(moments(&g, id, 0, 0), Moments { mean: 0.0, variance: 0.0 });
    }

    #[test]
    fn sample_n_returns_exactly_n_across_a_partial_final_batch() {
        let (g, id) = const_graph(7.0);
        // 1500 is not a multiple of BATCH (1024) — exercises the sliced final batch.
        let draws = sample_n(&g, id, 1500, 0);
        assert_eq!(draws.len(), 1500);
        assert!(draws.iter().all(|&x| x == 7.0), "constant column must stay constant");
    }

    #[test]
    fn moments_of_a_constant_have_zero_variance() {
        let (g, id) = const_graph(3.5);
        let m = moments(&g, id, 10_000, 1);
        assert_eq!(m.mean, 3.5);
        assert_eq!(m.variance, 0.0);
    }

    #[test]
    fn unif_int_degenerate_range_is_a_point_mass() {
        // unif_int(5,5): n = max(5-5+1, 1) = 1, so every draw is exactly 5 (under the hood).
        let mut g = RvGraph::default();
        let id = g.push(RvNode::Src(Source::UniformInt { lo: 5.0, hi: 5.0 }), RvKind::Num);
        let draws = sample_n(&g, id, 4096, 123);
        assert!(draws.iter().all(|&x| x == 5.0), "unif_int(5,5) must be exactly 5");
    }
}
