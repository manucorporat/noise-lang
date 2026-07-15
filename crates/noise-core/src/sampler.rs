//! Sampling driver — the only forcing path in Phase 2 (PLAN.md "batch sampler").
//!
//! Backend-independent: compile the RV cone into a [`Sampler`] once (via [`InterpBackend`], the
//! default), seed the RNG once, then loop `ceil(N / batch_cap)` batches. The final partial batch
//! is sliced to the true remaining length so over-count never biases moments. Swapping the
//! backend (e.g. the emitted wasm kernel) changes only how a batch is produced, not this loop.

use crate::backend::{compile_root, compile_roots};
use crate::dist::{RvGraph, RvId};
use crate::error::{NoiseError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use = "Moments carries the sampled mean/variance; computing them without using them wastes the sampling pass (finding F10)"]
pub struct Moments {
    pub mean: f64,
    pub variance: f64,
}

/// Run `n` draws of `root`, calling `sink` with each batch's filled root column slice.
/// The last batch may be partial (`len < BATCH`); the slice carries the true length.
///
/// Columns are **f32** (the lane type — PLAN-PREGPU Track B); every caller here widens to f64 as
/// it copies out, so the public draw vectors stay f64.
/// Cancellable (PLAN-PREGPU Track A): the thread's installed token is checked once per batch, so a
/// long single-stream collect aborts promptly. Like [`crate::reduce::run_reduction`], it returns
/// `Err` rather than a short vector — a truncated sample is not a smaller sample, it's a wrong one.
pub fn for_each_batch(
    graph: &RvGraph,
    root: RvId,
    n: usize,
    seed: u64,
    mut sink: impl FnMut(&[f32]),
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    // Per-forcing phase timing (NOISE_PROFILE=1, PLAN-DROP-JIT D0). This sequential path is the
    // plot/introspect collector — it never touches the GPU, which D0 wants to make visible.
    let _prof = crate::profile::forcing("for_each_batch", n);
    let (program, cost) = {
        let _s = crate::profile::span("compile");
        compile_root(graph, root, n)
    };
    crate::stats::record(n, cost.ops, cost.sources);
    crate::profile::set_ops(cost.ops);
    let _sample = crate::profile::span("sample");
    let mut runner = program.runner(crate::input_rt::current());
    runner.position(seed, 0);
    let cap = runner.batch_cap();

    let mut remaining = n;
    while remaining > 0 {
        if crate::exec::cancelled() {
            return Err(NoiseError::cancelled());
        }
        let take = remaining.min(cap);
        sink(runner.next_batch(take));
        remaining -= take;
    }
    Ok(())
}

/// Collect raw draws (small N / tests that need the full vector).
pub fn sample_n(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    for_each_batch(graph, root, n, seed, |col| {
        out.extend(col.iter().map(|&x| x as f64))
    })?;
    Ok(out)
}

/// Collect `n` raw draws **in parallel** — the forcing path behind `Q` (quantiles, PLAN-PERF-2
/// item 6). A quantile is an order statistic, so unlike `P`/`E`/`Var` it can't fold; but the
/// *sampling* is still embarrassingly parallel, so this delegates to the same chunked reduction as
/// [`moments`] ([`crate::reduce`], [`crate::reduce::CollectReducer`]): fixed, independently-seeded
/// chunks drawn in parallel and concatenated in chunk-index order, giving a vector that is
/// bit-identical for any thread count (and native multicore above the reduction's threshold —
/// sequential below it and on non-threaded wasm, over the *same* chunks). The caller sorts/selects
/// centrally.
///
/// **It reproduces [`sample_n`]'s stream exactly** — element for element, at any thread count.
/// Counter keying (PLAN-PREGPU Track C) made a chunk *just a lane range*, so "chunk `i`" and
/// "lanes `i·CHUNK ..`" are the same draws by construction; there is no per-chunk reseed left to
/// diverge. (Before Track C this was a documented trade-off: same distribution, different draws.
/// The guarantee got strictly stronger and the caveat is gone.) Pinned by
/// `par_and_seq_collect_the_identical_stream`.
pub fn sample_n_par(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Result<Vec<f64>> {
    crate::reduce::run_reduction(
        graph,
        root,
        n,
        seed,
        &crate::reduce::CollectReducer { skip_nan: false },
    )
}

/// Empirical mean + population variance over `n` draws — the forcing path behind `P`/`E`/`Var`.
///
/// Delegates to the parallel, deterministic monoid reduction in [`crate::reduce`]: draws are split
/// into fixed lane-range chunks, folded into Σx/Σx², and merged in chunk-index order. The result is
/// identical for any thread count (native multicore for large `n`, sequential otherwise and on
/// wasm) — and, since counter keying made a chunk a pure lane range, it folds over exactly the
/// draws [`sample_n`] would produce, in the same order. One stream, one seeding story.
pub fn moments(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Result<Moments> {
    Ok(
        crate::reduce::run_reduction(graph, root, n, seed, &crate::reduce::MomentsReducer)?
            .into_moments(),
    )
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
pub fn cond_moments(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Result<(Moments, u64)> {
    let acc =
        crate::reduce::run_reduction(graph, root, n, seed, &crate::reduce::CondMomentsReducer)?;
    Ok((acc.into_moments(), acc.count()))
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
pub fn cond_sample_n(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for_each_batch(graph, root, n, seed, |col| {
        out.extend(col.iter().filter(|x| !x.is_nan()).map(|&x| x as f64));
    })?;
    Ok(out)
}

/// Parallel twin of [`cond_sample_n`] — the forcing path behind `Q(x | C, q)`. Same chunked
/// reduction as [`sample_n_par`], with each chunk dropping its NaN (condition-false) lanes before
/// the index-ordered concatenation — the kept draws per chunk are fixed by `(seed, n)` alone, so
/// the result stays bit-identical for any thread count. Same KNOWN HOLE (finding B2) as
/// [`cond_sample_n`]: an in-condition NaN quantity is dropped rather than propagated.
pub fn cond_sample_n_par(graph: &RvGraph, root: RvId, n: usize, seed: u64) -> Result<Vec<f64>> {
    crate::reduce::run_reduction(
        graph,
        root,
        n,
        seed,
        &crate::reduce::CollectReducer { skip_nan: true },
    )
}

/// Drive a **joint** pass over `roots` — the single batch loop the four joint drivers
/// (`sample_pairs`/`grid_moments`/`grid_draws`/`corr_matrix`) share (finding B8). Compile the roots
/// through the backend seam ([`compile_roots`]) into ONE shared kernel so every lane draws them
/// jointly — the WASM emitter lowers a multi-output kernel where profitable, the multi-root
/// bytecode interpreter otherwise — **record the run-time cost** here (so a
/// `describe`/`hist`/`corr`/`fan` pass is no longer invisible in the engine's stats readout), then
/// loop `ceil(n / cap)` batches invoking `sink(cols, take)`: `cols[j]` is `roots[j]`'s column and
/// `take` is this (possibly partial final) batch's true length. The batch loop walks lanes `0..n`
/// in order — the same lane-keyed stream every other driver consumes (counter keying, PLAN-PREGPU
/// Track C); lane pairing across roots holds on every backend by construction (one shared
/// instruction stream).
fn for_each_joint_batch(
    graph: &RvGraph,
    roots: &[RvId],
    n: usize,
    seed: u64,
    mut sink: impl FnMut(&[&[f32]], usize),
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    // Per-forcing phase timing (NOISE_PROFILE=1, PLAN-DROP-JIT D0).
    let _prof = crate::profile::forcing("joint", n);
    // GPU joint driver (PLAN-DROP-JIT D4b): dispatch every root in ONE multi-column shader and fold
    // per column — the introspection/plot passes (describe/hist/corr/fan/scatter, plot::line/fan)
    // were the biggest CPU pool the corpus profile surfaced, and never touched the GPU before this. A
    // decline (thin/unsupported cone, no adapter, gate) falls through to the CPU interpreter below, so
    // this only ever changes speed. `try_joint` records its own run stats off the simplified cone.
    #[cfg(feature = "gpu")]
    {
        let token = crate::exec::current();
        if crate::gpu::try_joint(graph, roots, n, seed, &mut sink, token.as_ref())?.is_some() {
            return Ok(());
        }
    }
    let (prog, cost) = {
        let _s = crate::profile::span("compile");
        compile_roots(graph, roots, n)
    };
    // A union cone with zero RNG sources is *deterministic*: every lane of every batch is the same
    // value, so drawing the full budget is pure waste — and it is common, not a corner case. A plot
    // of an already-computed vector (`plot::line(signal::sample(...))`, a curve of forced `P()`
    // results) reaches here as k constant roots, which the interpreter would otherwise re-evaluate
    // `n × k` times (200k draws × 60 elements in `birthday`). One batch fully characterizes the
    // result (quantiles, moments, correlations of a constant are that constant), so clamp to it.
    let mut runner = prog.runner(crate::input_rt::current());
    runner.position(seed, 0);
    let cap = runner.batch_cap();
    let n = if cost.sources == 0 { n.min(cap) } else { n };
    crate::stats::record(n, cost.ops, cost.sources);
    crate::profile::set_ops(cost.ops);
    let _sample = crate::profile::span("sample");
    let mut remaining = n;
    while remaining > 0 {
        if crate::exec::cancelled() {
            return Err(NoiseError::cancelled());
        }
        runner.next_batch();
        let take = remaining.min(cap);
        let cols: Vec<&[f32]> = (0..roots.len()).map(|j| runner.col(j)).collect();
        sink(&cols, take);
        remaining -= take;
    }
    Ok(())
}

/// Joint draws of two roots — the forcing path behind `corr`/`scatter` (relationship
/// introspection). `a` and `b` are sampled in **one** pass over a shared instruction stream
/// ([`compile_roots`]), so each returned `(aᵢ, bᵢ)` is one lane's *paired* draw (shared upstream
/// randomness). When `cond` is `Some(c)`, only the lanes where `c` holds (its draw ≠ 0) are kept —
/// the conditional relationship `corr(A, B | C)`. Lane-keyed draws keep the batch loop
/// deterministic; pairing holds on every backend by construction (see [`for_each_joint_batch`]).
pub fn sample_pairs(
    graph: &RvGraph,
    a: RvId,
    b: RvId,
    cond: Option<RvId>,
    n: usize,
    seed: u64,
) -> Result<Vec<(f64, f64)>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut roots = vec![a, b];
    if let Some(c) = cond {
        roots.push(c);
    }
    let has_cond = cond.is_some();
    let mut out = Vec::with_capacity(n);
    for_each_joint_batch(graph, &roots, n, seed, |cols, take| {
        let (ca, cb) = (cols[0], cols[1]);
        let cc = if has_cond { Some(cols[2]) } else { None };
        for k in 0..take {
            if cc.is_none_or(|c| c[k] != 0.0) {
                out.push((ca[k] as f64, cb[k] as f64));
            }
        }
    })?;
    Ok(out)
}

/// Per-element moments of a whole set of roots in ONE joint pass — the forcing path behind a
/// vector/matrix summary (`describe` of an array). `roots` are the element RVs (row-major for a
/// matrix); every lane draws them jointly ([`compile_roots`]), and we accumulate `Σx`/`Σx²` per
/// element. Returns one [`Moments`] per root, in input order. Marginal moments don't *need* the
/// joint pass, but sampling once for all elements is far cheaper than one pass each.
pub fn grid_moments(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> Result<Vec<Moments>> {
    let k = roots.len();
    if k == 0 || n == 0 {
        return Ok(vec![
            Moments {
                mean: 0.0,
                variance: 0.0
            };
            k
        ]);
    }
    let (mut sum, mut sum_sq) = (vec![0.0f64; k], vec![0.0f64; k]);
    let mut count = 0u64;
    for_each_joint_batch(graph, roots, n, seed, |cols, take| {
        for j in 0..k {
            let col = cols[j];
            let (mut s, mut sq) = (0.0f64, 0.0f64);
            for &x in &col[..take] {
                let x = x as f64;
                s += x;
                sq += x * x;
            }
            sum[j] += s;
            sum_sq[j] += sq;
        }
        count += take as u64;
    })?;
    let nf = count as f64;
    Ok((0..k)
        .map(|j| {
            let mean = sum[j] / nf;
            Moments {
                mean,
                variance: (sum_sq[j] / nf - mean * mean).max(0.0),
            }
        })
        .collect())
}

/// Raw per-element draws of a whole set of roots in ONE joint pass — the forcing path behind a fan
/// chart (per-index *quantiles* need the full sample column, not just `Σx`/`Σx²`). `roots` are the
/// element RVs of a path; every lane draws them jointly ([`compile_roots`]), so column `j` holds the
/// `n` lane-aligned draws of `roots[j]` — the bands a caller derives are consistent across the
/// index. Memory is `k×n` f64s; the caller budgets `n` accordingly.
pub fn grid_draws(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> Result<Vec<Vec<f64>>> {
    let k = roots.len();
    if k == 0 || n == 0 {
        return Ok(vec![Vec::new(); k]);
    }
    let mut out: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(n)).collect();
    for_each_joint_batch(graph, roots, n, seed, |cols, take| {
        for j in 0..k {
            out[j].extend(cols[j][..take].iter().map(|&x| x as f64));
        }
    })?;
    Ok(out)
}

/// The full `k×k` correlation matrix over a set of roots in ONE joint pass — the forcing path behind
/// the element-vs-element heatmap (`corr` of a vector). Accumulates per-element `Σx`/`Σx²` and the
/// pairwise `Σxᵢxⱼ`, then forms Pearson correlations. Row-major `k*k`; the diagonal is 1 (a constant
/// element, zero variance, correlates 0 with everything). `O(k²)` per lane, so the caller caps `k`.
pub fn corr_matrix(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> Result<Vec<f64>> {
    let k = roots.len();
    if k == 0 || n == 0 {
        return Ok(vec![0.0; k * k]);
    }
    let mut sum = vec![0.0f64; k];
    let mut cross = vec![0.0f64; k * k]; // upper triangle filled; symmetrized at the end
    let mut vals = vec![0.0f64; k];
    let mut count = 0u64;
    for_each_joint_batch(graph, roots, n, seed, |cols, take| {
        for lane in 0..take {
            for i in 0..k {
                vals[i] = cols[i][lane] as f64;
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
    })?;
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
    Ok(corr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::dist::{RvKind, RvNode, Source, Uniform};

    fn const_graph(v: f64) -> (RvGraph, RvId) {
        let mut g = RvGraph::default();
        let id = g.push(RvNode::ConstNum(v), RvKind::Num);
        (g, id)
    }

    /// Counter keying (PLAN-PREGPU Track C) made a reducer chunk *just a lane range*, so the
    /// parallel collector must now draw the SAME stream as the sequential one — element for
    /// element, at any thread count. If this ever fails, `Runner::position` is no longer a pure
    /// function of `(seed, lane)` and the whole determinism story is broken.
    #[test]
    fn par_and_seq_collect_the_identical_stream() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let z = g.push(
            RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }),
            RvKind::Num,
        );
        let sum = g.push(RvNode::Binary(BinOp::Add, x, z), RvKind::Num);
        // Above the parallel threshold, so the par path really does fan out over threads.
        let n = 500_000;
        let seq = sample_n(&g, sum, n, 7).unwrap();
        let par = sample_n_par(&g, sum, n, 7).unwrap();
        assert_eq!(seq.len(), par.len());
        assert_eq!(
            seq.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            par.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "sample_n and sample_n_par must draw the identical stream"
        );
    }

    /// The joint drivers now route through the backend seam (`backend::compile_roots`), so this
    /// pins THE joint invariant on whichever backend this build selects (the multi-root bytecode
    /// interpreter on native, or the emitted wasm joint kernel on wasm — the draw count is above
    /// `MIN_DRAWS_WASM` on purpose): two roots sharing a source must read the *same* per-lane draw.
    /// `b = 2a` exactly, every lane.
    #[test]
    fn sample_pairs_share_draws_on_every_backend() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let two = g.push(RvNode::ConstNum(2.0), RvKind::Num);
        let x2 = g.push(RvNode::Binary(BinOp::Mul, x, two), RvKind::Num);
        let n = 150_000; // ≥ MIN_DRAWS_WASM: exercises the codegen joint path where available
        let pairs = sample_pairs(&g, x, x2, None, n, 42).unwrap();
        assert_eq!(pairs.len(), n);
        for &(a, b) in &pairs {
            assert!((0.0..1.0).contains(&a), "unif(0,1) draw out of range: {a}");
            assert_eq!(b, 2.0 * a, "roots must share the lane's draw of X");
        }
        // And the draws must actually vary (a broken column stride could repeat one lane).
        assert!(pairs.windows(2).any(|w| w[0].0 != w[1].0));
    }

    /// Same invariant on a transcendental-bearing cone (`normal` → single-stream codegen kernels),
    /// so both stream layouts of the joint kernel are pinned.
    #[test]
    fn sample_pairs_share_draws_single_stream_cone() {
        let mut g = RvGraph::default();
        let z = g.push(
            RvNode::Src(Source::Normal {
                mu: 0.0,
                sigma: 1.0,
            }),
            RvKind::Num,
        );
        let one = g.push(RvNode::ConstNum(1.0), RvKind::Num);
        let z1 = g.push(RvNode::Binary(BinOp::Add, z, one), RvKind::Num);
        let n = 150_000;
        let pairs = sample_pairs(&g, z, z1, None, n, 43).unwrap();
        assert_eq!(pairs.len(), n);
        for &(a, b) in &pairs {
            // The lane arithmetic is f32 (Track B), so the pairing identity must be checked in the
            // lane type: `z + 1` is one f32 add, not an f64 one over the widened draw.
            assert_eq!(
                b as f32,
                a as f32 + 1.0f32,
                "roots must share the lane's draw of Z"
            );
        }
        let mean = pairs.iter().map(|p| p.0).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "E[Z] should be ~0, got {mean}");
    }

    /// `grid_moments` through the seam: marginal moments of jointly-drawn roots must match each
    /// root's own distribution (catches a column-stride/ordering bug in a multi-output kernel).
    #[test]
    fn grid_moments_match_marginals_through_the_seam() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let y = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 2.0, hi: 4.0 })),
            RvKind::Num,
        );
        let s = g.push(RvNode::Binary(BinOp::Add, x, y), RvKind::Num);
        let ms = grid_moments(&g, &[x, y, s], 200_000, 7).unwrap();
        assert!(
            (ms[0].mean - 0.5).abs() < 0.01,
            "E[X]≈0.5, got {}",
            ms[0].mean
        );
        assert!(
            (ms[1].mean - 3.0).abs() < 0.01,
            "E[Y]≈3.0, got {}",
            ms[1].mean
        );
        assert!(
            (ms[2].mean - 3.5).abs() < 0.01,
            "E[X+Y]≈3.5, got {}",
            ms[2].mean
        );
        assert!(
            (ms[0].variance - 1.0 / 12.0).abs() < 0.01,
            "Var[X]≈1/12, got {}",
            ms[0].variance
        );
    }

    /// The joint driver on a **fat** cone — one that clears the GPU joint gate (PLAN-DROP-JIT D4b), so
    /// `--features gpu` exercises `gpu::try_joint` (multi-column shader → per-column fold) and the
    /// default build the CPU joint loop; both must agree with the analytic truth. Two properties a
    /// column-layout, stride, or pairing bug would break: (1) each element's mean equals its **distinct**
    /// offset `i` (a swapped/mislaid column would read the wrong offset), and (2) the elements are
    /// independent, so every off-diagonal correlation is ~0 (a shared-stride bug would fabricate
    /// correlation). Each root is `i + z_i + 0.1·z_i³` over its own `N(0,1)` — a fat odd cone (mean `i`),
    /// fat enough (≈7 ops/root, union ≈70) that the joint gate accepts it where a GPU is present.
    #[test]
    fn joint_driver_matches_analytic_on_a_fat_iid_cone() {
        use crate::dist::RvKind;
        let k = 10usize;
        let mut g = RvGraph::default();
        let tenth = g.push(RvNode::ConstNum(0.1), RvKind::Num);
        let roots: Vec<RvId> = (0..k)
            .map(|i| {
                let z = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
                let z2 = g.push(RvNode::Binary(BinOp::Mul, z, z), RvKind::Num);
                let z3 = g.push(RvNode::Binary(BinOp::Mul, z2, z), RvKind::Num);
                let sc = g.push(RvNode::Binary(BinOp::Mul, z3, tenth), RvKind::Num);
                let sum = g.push(RvNode::Binary(BinOp::Add, z, sc), RvKind::Num); // z + 0.1 z³, mean 0
                let off = g.push(RvNode::ConstNum(i as f64), RvKind::Num);
                g.push(RvNode::Binary(BinOp::Add, sum, off), RvKind::Num) // i + z + 0.1 z³, mean i
            })
            .collect();

        let n = 300_000;
        let ms = grid_moments(&g, &roots, n, 7).unwrap();
        for (i, m) in ms.iter().enumerate() {
            assert!(
                (m.mean - i as f64).abs() < 0.05,
                "element {i}: mean {} should be ~{i} (a column-layout bug reads the wrong element)",
                m.mean
            );
        }
        // Independence: every off-diagonal correlation ~0. Monte-Carlo SE ~1/√n ≈ 0.002, so the max
        // over 90 pairs sits comfortably under 0.03; a stride/pairing bug would blow this up.
        let corr = corr_matrix(&g, &roots, n, 7).unwrap();
        let max_off = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| corr[i * k + j].abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_off < 0.03,
            "off-diagonal |corr| max = {max_off} — independent elements must not correlate"
        );
    }

    #[test]
    fn sample_n_zero_is_empty() {
        let (g, id) = const_graph(1.0);
        assert!(sample_n(&g, id, 0, 0).unwrap().is_empty());
        // moments over zero draws is well-defined (no NaN), not a panic.
        assert_eq!(
            moments(&g, id, 0, 0).unwrap(),
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
        let draws = sample_n(&g, id, 1500, 0).unwrap();
        assert_eq!(draws.len(), 1500);
        assert!(
            draws.iter().all(|&x| x == 7.0),
            "constant column must stay constant"
        );
    }

    #[test]
    fn moments_of_a_constant_have_zero_variance() {
        let (g, id) = const_graph(3.5);
        let m = moments(&g, id, 10_000, 1).unwrap();
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
        let draws = sample_n(&g, id, 4096, 123).unwrap();
        assert!(
            draws.iter().all(|&x| x == 5.0),
            "unif_int(5,5) must be exactly 5"
        );
    }
}
