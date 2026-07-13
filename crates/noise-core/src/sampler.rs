//! Sampling driver — the only forcing path in Phase 2 (PLAN.md "batch sampler").
//!
//! Backend-independent: compile the RV cone into a [`Sampler`] once (via [`InterpBackend`], the
//! default), seed the RNG once, then loop `ceil(N / batch_cap)` batches. The final partial batch
//! is sliced to the true remaining length so over-count never biases moments. Swapping the
//! backend (e.g. a Cranelift JIT) changes only how a batch is produced, not this loop.

use crate::backend::compile_root;
use crate::bytecode::{compile_roots, run_batch, BATCH};
use crate::dist::{RvGraph, RvId};
use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use = "Moments carries the sampled mean/variance; computing them without using them wastes the sampling pass (finding F10)"]
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
    let (program, cost) = compile_root(graph, root, n);
    crate::stats::record(n, cost.ops, cost.sources);
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
    crate::reduce::run_reduction(graph, root, n, seed, &crate::reduce::MomentsReducer)
        .into_moments()
}

/// Conditional moments — the forcing path behind `P(· | C)` / `E(· | C)` / `Var(· | C)`. `root`
/// must be the conditioning root `select(C, quantity, NaN)`: NaN-valued lanes (where `C` is false)
/// are skipped, so the moments are over the subpopulation where `C` holds. Returns those moments
/// alongside the in-condition count `m` (≈ `n·P(C)`), which the caller uses for the standard error.
/// `m == 0` means the condition never occurred in `n` draws (the conditional is undefined upstream).
///
/// KNOWN HOLE (finding B2): like [`cond_sample_n`], the NaN filter can't distinguish "condition
/// false" from "the quantity is itself NaN on an in-condition lane" — an in-condition NaN quantity is
/// *dropped* rather than propagated. So a conditional estimate over a quantity that can go NaN inside
/// the condition (e.g. `E(math::log(X) | X > -1)` with `X ~ unif(-1, 1)`) is silently biased and
/// reports a *tighter* SE than the honest one. The fix is a dedicated condition column (see
/// `Engine::query_cond`); deferred for the interface-ripple reason noted on `cond_sample_n`.
pub fn cond_moments(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> (Moments, u64) {
    let acc =
        crate::reduce::run_reduction(graph, root, n, seed, &crate::reduce::CondMomentsReducer);
    (acc.into_moments(), acc.count())
}

/// Collect the in-condition draws of a conditioning root `select(C, quantity, NaN)` — every
/// non-NaN lane, in stream order — for a conditional quantile `Q(· | C)`. NaN lanes (where `C` is
/// false) are dropped, so the returned vector holds exactly the `m ≈ n·P(C)` draws from the
/// subpopulation where `C` holds (unsorted; the caller sorts).
///
/// KNOWN HOLE (finding B2): like [`cond_moments`], the NaN filter can't tell "condition false" from
/// "the quantity is NaN on an in-condition lane" — an in-condition NaN quantity is dropped rather
/// than propagated, biasing the conditional quantile sample. The fix is a dedicated condition column
/// (see `Engine::query_cond`); deferred for the same interface-ripple reason.
pub fn cond_sample_n(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Vec<f64> {
    let mut out = Vec::new();
    for_each_batch(graph, root, n, seed, |col| {
        out.extend(col.iter().copied().filter(|x| !x.is_nan()));
    });
    out
}

/// Drive a **joint** pass over `roots` — the single batch loop the four joint drivers
/// (`sample_pairs`/`grid_moments`/`grid_draws`/`corr_matrix`) share (finding B8). Compile the roots
/// into ONE shared instruction stream ([`compile_roots`]) so every lane draws them jointly, **record
/// the run-time cost** here (so a `describe`/`hist`/`corr`/`fan` pass is no longer invisible in the
/// engine's stats readout), then loop `ceil(n / BATCH)` batches invoking `sink(buf, idx, take)`:
/// `idx[j]` is the register holding `roots[j]`, and `take` is this (possibly partial final) batch's
/// true length. A single ordered RNG stream (not the per-chunk reseed of [`moments`]) keeps lanes
/// paired across roots; this is the interpreter path (the introspection budget is modest).
fn for_each_joint_batch(
    graph: &RvGraph,
    roots: &[RvId],
    n: usize,
    seed: u64,
    mut sink: impl FnMut(&[Box<[f64]>], &[usize], usize),
) {
    let (prog, regs) = compile_roots(graph, roots);
    let cost = crate::kernel::cost_roots(graph, roots);
    crate::stats::record(n, cost.ops, cost.sources);
    let idx: Vec<usize> = regs.iter().map(|&r| r as usize).collect();
    let mut buf: Vec<Box<[f64]>> = (0..prog.n_regs)
        .map(|_| vec![0.0f64; BATCH].into_boxed_slice())
        .collect();
    let mut rng = Rng::seed_from_u64(seed);
    let mut remaining = n;
    while remaining > 0 {
        run_batch(&prog, &mut buf, &mut rng);
        let take = remaining.min(BATCH);
        sink(&buf, &idx, take);
        remaining -= take;
    }
}

/// Joint draws of two roots — the forcing path behind `corr`/`scatter` (relationship
/// introspection). `a` and `b` are sampled in **one** pass over a shared instruction stream
/// ([`compile_roots`]), so each returned `(aᵢ, bᵢ)` is one lane's *paired* draw (shared upstream
/// randomness). When `cond` is `Some(c)`, only the lanes where `c` holds (its draw ≠ 0) are kept —
/// the conditional relationship `corr(A, B | C)`. A single ordered RNG stream (not the per-chunk
/// reseed of [`moments`]) keeps the pairing exact; this is the interpreter path (the introspection
/// budget is modest, so the JIT/wasm fast paths aren't needed here).
pub fn sample_pairs(
    graph: &RvGraph,
    a: RvId,
    b: RvId,
    cond: Option<RvId>,
    n: usize,
    seed: u64,
) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }
    let mut roots = vec![a, b];
    if let Some(c) = cond {
        roots.push(c);
    }
    let has_cond = cond.is_some();
    let mut out = Vec::with_capacity(n);
    for_each_joint_batch(graph, &roots, n, seed, |buf, idx, take| {
        let (ra, rb) = (idx[0], idx[1]);
        let rc = if has_cond { Some(idx[2]) } else { None };
        for k in 0..take {
            if rc.is_none_or(|c| buf[c][k] != 0.0) {
                out.push((buf[ra][k], buf[rb][k]));
            }
        }
    });
    out
}

/// Per-element moments of a whole set of roots in ONE joint pass — the forcing path behind a
/// vector/matrix summary (`describe` of an array). `roots` are the element RVs (row-major for a
/// matrix); every lane draws them jointly ([`compile_roots`]), and we accumulate `Σx`/`Σx²` per
/// element. Returns one [`Moments`] per root, in input order. Marginal moments don't *need* the
/// joint pass, but sampling once for all elements is far cheaper than one pass each.
pub fn grid_moments(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> Vec<Moments> {
    let k = roots.len();
    if k == 0 || n == 0 {
        return vec![
            Moments {
                mean: 0.0,
                variance: 0.0
            };
            k
        ];
    }
    let (mut sum, mut sum_sq) = (vec![0.0f64; k], vec![0.0f64; k]);
    let mut count = 0u64;
    for_each_joint_batch(graph, roots, n, seed, |buf, idx, take| {
        for j in 0..k {
            let col = &buf[idx[j]];
            let (mut s, mut sq) = (0.0f64, 0.0f64);
            for &x in &col[..take] {
                s += x;
                sq += x * x;
            }
            sum[j] += s;
            sum_sq[j] += sq;
        }
        count += take as u64;
    });
    let nf = count as f64;
    (0..k)
        .map(|j| {
            let mean = sum[j] / nf;
            Moments {
                mean,
                variance: (sum_sq[j] / nf - mean * mean).max(0.0),
            }
        })
        .collect()
}

/// Raw per-element draws of a whole set of roots in ONE joint pass — the forcing path behind a fan
/// chart (per-index *quantiles* need the full sample column, not just `Σx`/`Σx²`). `roots` are the
/// element RVs of a path; every lane draws them jointly ([`compile_roots`]), so column `j` holds the
/// `n` lane-aligned draws of `roots[j]` — the bands a caller derives are consistent across the
/// index. Memory is `k×n` f64s; the caller budgets `n` accordingly.
pub fn grid_draws(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> Vec<Vec<f64>> {
    let k = roots.len();
    if k == 0 || n == 0 {
        return vec![Vec::new(); k];
    }
    let mut cols: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(n)).collect();
    for_each_joint_batch(graph, roots, n, seed, |buf, idx, take| {
        for j in 0..k {
            cols[j].extend_from_slice(&buf[idx[j]][..take]);
        }
    });
    cols
}

/// The full `k×k` correlation matrix over a set of roots in ONE joint pass — the forcing path behind
/// the element-vs-element heatmap (`corr` of a vector). Accumulates per-element `Σx`/`Σx²` and the
/// pairwise `Σxᵢxⱼ`, then forms Pearson correlations. Row-major `k*k`; the diagonal is 1 (a constant
/// element, zero variance, correlates 0 with everything). `O(k²)` per lane, so the caller caps `k`.
pub fn corr_matrix(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> Vec<f64> {
    let k = roots.len();
    if k == 0 || n == 0 {
        return vec![0.0; k * k];
    }
    let mut sum = vec![0.0f64; k];
    let mut cross = vec![0.0f64; k * k]; // upper triangle filled; symmetrized at the end
    let mut vals = vec![0.0f64; k];
    let mut count = 0u64;
    for_each_joint_batch(graph, roots, n, seed, |buf, idx, take| {
        for lane in 0..take {
            for i in 0..k {
                vals[i] = buf[idx[i]][lane];
            }
            for i in 0..k {
                sum[i] += vals[i];
                let row = i * k;
                for j in i..k {
                    cross[row + j] += vals[i] * vals[j];
                }
            }
        }
        count += take as u64;
    });
    let nf = count as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / nf).collect();
    let var: Vec<f64> = (0..k)
        .map(|i| (cross[i * k + i] / nf - mean[i] * mean[i]).max(0.0))
        .collect();
    let mut corr = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let cov = cross[i * k + j] / nf - mean[i] * mean[j];
            let denom = (var[i] * var[j]).sqrt();
            let c = if denom > 0.0 {
                (cov / denom).clamp(-1.0, 1.0)
            } else {
                f64::from(i == j)
            };
            corr[i * k + j] = c;
            corr[j * k + i] = c; // symmetric
        }
    }
    corr
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
        assert_eq!(
            moments(&g, id, 0, 0),
            Moments {
                mean: 0.0,
                variance: 0.0
            }
        );
    }

    #[test]
    fn sample_n_returns_exactly_n_across_a_partial_final_batch() {
        let (g, id) = const_graph(7.0);
        // 1500 is not a multiple of BATCH (1024) — exercises the sliced final batch.
        let draws = sample_n(&g, id, 1500, 0);
        assert_eq!(draws.len(), 1500);
        assert!(
            draws.iter().all(|&x| x == 7.0),
            "constant column must stay constant"
        );
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
        let id = g.push(
            RvNode::Src(Source::UniformInt { lo: 5.0, hi: 5.0 }),
            RvKind::Num,
        );
        let draws = sample_n(&g, id, 4096, 123);
        assert!(
            draws.iter().all(|&x| x == 5.0),
            "unif_int(5,5) must be exactly 5"
        );
    }
}
