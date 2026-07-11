//! Variable introspection — the single engine-side core behind *looking at* a random variable.
//!
//! The rest of the language collapses a distribution to a scalar (`P`/`E`/`Var`/`Q`) before you
//! can see anything. Introspection does the opposite: it hands back the **whole** shape. There are
//! exactly **two operations, picked by arity** — everything user-facing is a *view* or a
//! *composition* of these two:
//!
//!   * one variable  → [`Dist1`]: moments, quantiles, a histogram, a sample head. The views
//!     `describe` / `hist` / `samples` are three renderings of one `Dist1`.
//!   * two variables → [`Dist2`]: correlation, covariance, scatter points. The views `corr` /
//!     `scatter` render one `Dist2`.
//!   * `explain(Y)` is **not** a third operation: it is a fan-out of `Dist2` over the named
//!     variables upstream of `Y`, ranked by how much each moves `Y`.
//!
//! This is the seam two front-ends share (the "one core, two adapters" design): the in-source
//! builtins (`describe(x)`, …) call it and wrap the result in a [`Value::Summary`](crate::Value)
//! that renders in the CLI; the sidecar `run_with_introspection` will call the very same functions
//! against a captured scope and serialize the result for the playground. The measurement lives here
//! once; the adapters only differ in how they obtain the `RvId`. Neither draws anything: rendering
//! is [`crate::flint`]'s job, and it only *describes* charts.

use std::fmt;

use crate::dist::{RvGraph, RvId};
use crate::sampler;

/// Monte Carlo budget for an introspection pass. Deliberately below `P`'s `1e6`: a histogram, a set
/// of quantiles, or a scatter is a *visual* — it doesn't need a probability's last digit, and the
/// interactive playground issues many of these per run. Capped (not the engine's full
/// `max_samples`) so a `describe` stays snappy and memory-light regardless of the configured budget.
pub const INTROSPECT_N: usize = 200_000;

/// Fixed seed — an introspection summary is reproducible like every other forcing path.
pub const INTROSPECT_SEED: u64 = 0;

/// Bins in a numeric histogram (a boolean quantity uses 2). ~30 is enough resolution for a bar
/// chart without over-fragmenting a modest in-condition sample. Also the default bin count of
/// `stats::histogram(x)`, so its numbers *are* the ones `plot::hist(x)` draws.
pub const NUM_BINS: usize = 30;

/// Default scatter points kept for a [`Dist2`] (the full pair sample drives the *statistics*; the
/// point list is a subsample for plotting, so a chart isn't asked to draw 200k dots).
const SCATTER_POINTS: usize = 800;

/// Longest path a [`fan`] accepts. A fan holds every column's full sample, so this bounds the
/// [`FAN_CELLS`] trade (see there) and keeps a stray `cumsum` over 10⁵ steps from hanging the
/// engine. Shared by `plot::fan` and `stats::fan` — the same computation, so the same limit.
pub const FAN_MAX: usize = 1024;

/// Largest vector a [`corr_grid`] accepts: the matrix is `n²` correlations, each over the full
/// joint sample. Shared by `corr(vec)` / `plot::corr` / `stats::corr`.
pub const CORR_MAX: usize = 64;

/// A histogram: `bins.len()` equal-width buckets spanning `[lo, hi]` (a boolean quantity is the two
/// buckets `false`/`true` with `lo=0, hi=1`). `bins[i]` is the count in bucket `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    pub lo: f64,
    pub hi: f64,
    pub bins: Vec<u64>,
}

/// A one-variable summary: the whole distribution, not a single number. `describe`/`hist`/`samples`
/// are three views of this. `boolean` marks an event quantity (draws are 0/1, so `mean` is its
/// probability and the histogram is two buckets).
#[derive(Debug, Clone, PartialEq)]
pub struct Dist1 {
    pub n: u64,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub max: f64,
    pub q05: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub q95: f64,
    pub hist: Histogram,
    /// A short prefix of raw draws — what `samples(x, k)` shows (demystifies "what's under a name").
    pub head: Vec<f64>,
    pub boolean: bool,
}

impl Dist1 {
    /// Compute the summary from raw draws (already filtered to the in-condition lanes for a
    /// conditional summary). `head_k` raw draws are kept verbatim for the `samples` view. Empty
    /// `draws` (a condition that never held) is the caller's error to raise — this assumes non-empty.
    pub fn from_draws(draws: &[f64], boolean: bool, head_k: usize) -> Dist1 {
        let n = draws.len() as u64;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &x in draws {
            sum += x;
            sum_sq += x * x;
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
        let nf = n as f64;
        let mean = sum / nf;
        let sd = (sum_sq / nf - mean * mean).max(0.0).sqrt();
        let mut sorted = draws.to_vec();
        sorted.sort_by(f64::total_cmp);
        let head = draws.iter().take(head_k).copied().collect();
        Dist1 {
            n,
            mean,
            sd,
            min,
            max,
            q05: quantile_sorted(&sorted, 0.05),
            q25: quantile_sorted(&sorted, 0.25),
            q50: quantile_sorted(&sorted, 0.50),
            q75: quantile_sorted(&sorted, 0.75),
            q95: quantile_sorted(&sorted, 0.95),
            hist: histogram(draws, boolean, NUM_BINS),
            head,
            boolean,
        }
    }
}

/// A two-variable summary: how `a` and `b` move together. `corr`/`scatter` are views of this. The
/// statistics use every paired draw; `points` is a subsample for plotting.
#[derive(Debug, Clone, PartialEq)]
pub struct Dist2 {
    pub n: u64,
    pub corr: f64,
    pub cov: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub sd_a: f64,
    pub sd_b: f64,
    pub points: Vec<(f64, f64)>,
}

impl Dist2 {
    /// Compute correlation / covariance / means from paired draws, plus an evenly-strided subsample
    /// of `max_points` for a scatter plot. Zero variance on either side ⇒ correlation `0` (a constant
    /// has no linear relationship to anything), never a NaN.
    pub fn from_pairs(pairs: &[(f64, f64)], max_points: usize) -> Dist2 {
        let n = pairs.len() as u64;
        let nf = n as f64;
        let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(x, y) in pairs {
            sa += x;
            sb += y;
            saa += x * x;
            sbb += y * y;
            sab += x * y;
        }
        let mean_a = sa / nf;
        let mean_b = sb / nf;
        let var_a = (saa / nf - mean_a * mean_a).max(0.0);
        let var_b = (sbb / nf - mean_b * mean_b).max(0.0);
        let cov = sab / nf - mean_a * mean_b;
        let (sd_a, sd_b) = (var_a.sqrt(), var_b.sqrt());
        let corr = if sd_a > 0.0 && sd_b > 0.0 {
            (cov / (sd_a * sd_b)).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let stride = (pairs.len() / max_points.max(1)).max(1);
        let points = pairs
            .iter()
            .step_by(stride)
            .take(max_points)
            .copied()
            .collect();
        Dist2 {
            n,
            corr,
            cov,
            mean_a,
            mean_b,
            sd_a,
            sd_b,
            points,
        }
    }
}

/// A scalar value with its uncertainty — what `describe`/`plot::value` of a `number`/estimate shows.
/// `se == 0` is an exact point (a constant); otherwise the 95% CI is `val ± 1.96·se`. This is why a
/// query result like `pi = 4*P(…)` is still worth looking at: it carries a confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueCard {
    pub val: f64,
    pub se: f64,
}

/// Per-cell moments of an array of random variables — a vector (`rows == 1`) or a matrix. `mean`/`sd`
/// are row-major, length `rows*cols`. A vector renders as a per-index mean±sd series; a matrix as a
/// heatmap of the means. Computed in one joint pass ([`crate::sampler::grid_moments`]).
#[derive(Debug, Clone, PartialEq)]
pub struct DistGrid {
    pub rows: usize,
    pub cols: usize,
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
}

impl DistGrid {
    /// `true` when the grid is one-dimensional (a vector) — the renderer draws a series, not a heatmap.
    pub fn is_series(&self) -> bool {
        self.rows <= 1 || self.cols <= 1
    }
}

/// Per-index quantile bands of a *path* — a vector of random variables sampled **jointly** (one
/// pass, shared lanes), so the bands are consistent across the index: the data behind
/// `plot::fan(path)`. Each band vector has length `cols`; `q05[t] … q95[t]` bracket the simulated
/// value at index `t`, and `mean` rides along (it falls out of the same draws for free). A
/// deterministic path is the degenerate fan: every band equals the values.
#[derive(Debug, Clone, PartialEq)]
pub struct FanChart {
    pub cols: usize,
    pub n: u64,
    pub q05: Vec<f64>,
    pub q25: Vec<f64>,
    pub q50: Vec<f64>,
    pub q75: Vec<f64>,
    pub q95: Vec<f64>,
    pub mean: Vec<f64>,
}

/// The full element×element correlation matrix of a vector of random variables (`n×n`, row-major) —
/// the dependence heatmap behind `corr(vec)` / `plot::corr`. Diagonal is 1; iid elements are ~0
/// off-diagonal. Computed in one joint pass ([`crate::sampler::corr_matrix`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CorrMatrix {
    pub n: usize,
    pub corr: Vec<f64>,
}

/// One contributor in an [`Explain`]: a named variable upstream of the target, its correlation with
/// the target, and `share` — a rough fraction of the explained spread it accounts for (`corr²`,
/// normalized over the candidates). `share` is a *first-cut* attribution: it ranks drivers honestly
/// but, because the candidates can overlap, the shares are not a true variance decomposition (that
/// is the freeze/Sobol upgrade, same surface, better numbers).
#[derive(Debug, Clone, PartialEq)]
pub struct Driver {
    pub name: String,
    pub corr: f64,
    pub share: f64,
}

/// Why a variable is uncertain: its total spread plus the upstream named variables that drive it,
/// ranked. A fan-out of [`Dist2`] — the structurally-unique introspection (it needs the sample-DAG
/// to know which variables are upstream), surfaced as one call.
#[derive(Debug, Clone, PartialEq)]
pub struct Explain {
    pub sd: f64,
    pub drivers: Vec<Driver>,
}

impl Explain {
    /// Rank `candidates` (name, corr-with-target) into drivers by |corr|, attaching the `corr²`
    /// share. `sd` is the target's standard deviation (its total spread).
    pub fn from_candidates(sd: f64, candidates: Vec<(String, f64)>) -> Explain {
        let total: f64 = candidates.iter().map(|(_, c)| c * c).sum();
        let mut drivers: Vec<Driver> = candidates
            .into_iter()
            .map(|(name, corr)| {
                let share = if total > 0.0 {
                    corr * corr / total
                } else {
                    0.0
                };
                Driver { name, corr, share }
            })
            .collect();
        drivers.sort_by(|a, b| b.corr.abs().total_cmp(&a.corr.abs()));
        Explain { sd, drivers }
    }
}

// --- forcing entry points (graph + root → summary) -------------------------------------------

/// One-variable summary of `root`. `conditional` selects the forcing path: `false` samples `root`
/// directly (a marginal distribution); `true` treats `root` as the conditioning root
/// `select(C, quantity, NaN)` and summarizes only the in-condition (non-NaN) draws — so
/// `describe(X | C)` shows the conditional distribution. Returns `None` when a conditional draws
/// zero in-condition lanes (the condition never held — the caller raises a spanned error).
pub fn dist1(
    graph: &RvGraph,
    root: RvId,
    boolean: bool,
    conditional: bool,
    n: usize,
    seed: u64,
    head_k: usize,
) -> Option<Dist1> {
    let draws = draws(graph, root, conditional, n, seed);
    if draws.is_empty() {
        return None;
    }
    Some(Dist1::from_draws(&draws, boolean, head_k))
}

/// The raw draws behind a one-variable summary — the sampling half of [`dist1`], split out so the
/// `stats::` builtins can bin or rank the very same numbers a `plot::` chart shows. `conditional`
/// selects the forcing path (see [`dist1`]); an empty result means the condition never held.
pub fn draws(graph: &RvGraph, root: RvId, conditional: bool, n: usize, seed: u64) -> Vec<f64> {
    if conditional {
        sampler::cond_sample_n(graph, root, n, seed)
    } else {
        sampler::sample_n(graph, root, n, seed)
    }
}

/// Two-variable summary of `(a, b)`, optionally within the worlds where `cond` holds. `None` when a
/// conditional pass keeps zero lanes.
pub fn dist2(
    graph: &RvGraph,
    a: RvId,
    b: RvId,
    cond: Option<RvId>,
    n: usize,
    seed: u64,
) -> Option<Dist2> {
    let pairs = sampler::sample_pairs(graph, a, b, cond, n, seed);
    if pairs.is_empty() {
        return None;
    }
    Some(Dist2::from_pairs(&pairs, SCATTER_POINTS))
}

/// Per-cell moments of an array of `roots` (row-major, `rows×cols`) in one joint pass — the data
/// behind a vector/matrix plot. A vector passes `rows = 1`.
pub fn grid(
    graph: &RvGraph,
    roots: &[RvId],
    rows: usize,
    cols: usize,
    n: usize,
    seed: u64,
) -> DistGrid {
    let moments = sampler::grid_moments(graph, roots, n, seed);
    DistGrid {
        rows,
        cols,
        mean: moments.iter().map(|m| m.mean).collect(),
        sd: moments.iter().map(|m| m.variance.sqrt()).collect(),
    }
}

/// The element×element correlation matrix of a vector of `roots` (one joint pass).
pub fn corr_grid(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> CorrMatrix {
    CorrMatrix {
        n: roots.len(),
        corr: sampler::corr_matrix(graph, roots, n, seed),
    }
}

/// Memory budget for a fan chart's joint draw matrix, in `f64` cells (`cols × n` ≈ 32 MB at the
/// cap). A fan holds every column's *full sample* (quantiles need it, unlike [`grid`]'s running
/// moments), so a wide path trims the per-index draw count instead of ballooning memory: a 252-step
/// year still gets ~15k draws per index — plenty to pin a q05/q95 band for a chart.
const FAN_CELLS: usize = 4_000_000;

/// Per-index quantile bands of a path of `roots` in ONE joint pass — the data behind
/// `plot::fan(path)`. All indices are drawn on shared lanes ([`sampler::grid_draws`]), then each
/// column is sorted and read at the same five quantiles a scalar [`Dist1`] reports
/// (q05/q25/q50/q75/q95, [`quantile_sorted`]). `n` is a *request*; it is clamped to the
/// [`FAN_CELLS`] budget (see there) and the effective count is returned in the chart.
pub fn fan(graph: &RvGraph, roots: &[RvId], n: usize, seed: u64) -> FanChart {
    let cols = roots.len();
    let n = n.min(FAN_CELLS / cols.max(1)).max(1);
    let draws = sampler::grid_draws(graph, roots, n, seed);
    let mut fc = FanChart {
        cols,
        n: n as u64,
        q05: Vec::with_capacity(cols),
        q25: Vec::with_capacity(cols),
        q50: Vec::with_capacity(cols),
        q75: Vec::with_capacity(cols),
        q95: Vec::with_capacity(cols),
        mean: Vec::with_capacity(cols),
    };
    for mut col in draws {
        fc.mean
            .push(col.iter().sum::<f64>() / col.len().max(1) as f64);
        col.sort_by(f64::total_cmp);
        fc.q05.push(quantile_sorted(&col, 0.05));
        fc.q25.push(quantile_sorted(&col, 0.25));
        fc.q50.push(quantile_sorted(&col, 0.50));
        fc.q75.push(quantile_sorted(&col, 0.75));
        fc.q95.push(quantile_sorted(&col, 0.95));
    }
    fc
}

/// Linear-interpolated empirical quantile of a **sorted, non-empty** sample (numpy's type-7 rule —
/// the same rule `builtins::Q` uses, redefined here so introspection has no cross-module dep).
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Bin `draws` into a [`Histogram`] of `nbins` equal-width buckets over `[min, max]` (a degenerate
/// point mass is one bucket). A boolean quantity ignores `nbins`: it is the two buckets
/// `false`/`true`, and there is no third thing an event can be.
///
/// The one binning in the codebase. `Dist1::from_draws` calls it with [`NUM_BINS`], and
/// `stats::histogram(x, bins)` calls it with whatever the program asked for — so the array a
/// program reads and the bars a chart draws are the same computation, never two that agree.
pub fn histogram(draws: &[f64], boolean: bool, nbins: usize) -> Histogram {
    if boolean {
        let mut bins = vec![0u64; 2];
        for &x in draws {
            bins[(x != 0.0) as usize] += 1;
        }
        return Histogram {
            lo: 0.0,
            hi: 1.0,
            bins,
        };
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in draws {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    let nbins = nbins.max(1);
    if !lo.is_finite() || !hi.is_finite() || lo == hi {
        return Histogram {
            lo,
            hi,
            bins: vec![draws.len() as u64],
        };
    }
    let span = hi - lo;
    let mut bins = vec![0u64; nbins];
    for &x in draws {
        let mut idx = (((x - lo) / span) * nbins as f64) as usize;
        if idx >= nbins {
            idx = nbins - 1; // the max draw lands in the last bucket, not one past it
        }
        bins[idx] += 1;
    }
    Histogram { lo, hi, bins }
}

impl Histogram {
    /// The center of each bucket — what a bar sits on, and what `stats::histogram` returns as its
    /// first row. A `boolean` histogram's two buckets are the *points* 0 and 1, not the midpoints
    /// 0.25 / 0.75 of the halves of `[0, 1]`.
    pub fn midpoints(&self, boolean: bool) -> Vec<f64> {
        if boolean {
            return vec![0.0, 1.0];
        }
        let width = (self.hi - self.lo) / self.bins.len() as f64;
        (0..self.bins.len())
            .map(|i| self.lo + (i as f64 + 0.5) * width)
            .collect()
    }
}

// --- the user-facing summary value + its CLI rendering ----------------------------------------

/// Which rendering an in-source builtin asked for. The *engine* always computes the full payload;
/// the view only picks how [`crate::flint`] titles and words it. So `describe`/`hist`/`samples`
/// differ only here, never in what was measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Describe,
    Hist,
    Samples,
    Corr,
    Scatter,
    Explain,
    Value,
    Grid,
    CorrMatrix,
    Fan,
}

/// The payload behind a [`Summary`] — one of the data ops' results. `One`/`Two` are the scalar
/// distribution / relationship; `Value` is a scalar+CI; `Grid` is a vector/matrix; `CorrMatrix` is
/// an element×element heatmap; `Explain` is the driver fan-out.
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    One(Dist1),
    Two(Dist2),
    Explain(Explain),
    Value(ValueCard),
    Grid(DistGrid),
    CorrMatrix(CorrMatrix),
    Fan(FanChart),
}

/// A first-class introspection result — what `describe(x)` / `corr(a, b)` / `explain(y)` evaluate
/// to. Carried in a [`Value::Summary`](crate::Value) so it composes as an ordinary value; every
/// host renders it through [`crate::flint`], which turns this `payload` into chart specs plus a
/// text card. `label`/`label_b` are the source names of the introspected variable(s), for the
/// heading.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub view: View,
    pub label: String,
    pub label_b: Option<String>,
    pub payload: Payload,
}

/// The CLI / REPL rendering. Noise owns no chart code: a summary renders as the one-line text card
/// [`crate::flint::text_card`] builds — the same line a graphical host shows when it cannot draw.
/// The picture is the host's job (`plot::*` also emits Flint specs; see [`crate::flint`]).
impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&crate::flint::text_card(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist1_of_a_flat_sample_is_uniform_shaped() {
        let draws: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let d = Dist1::from_draws(&draws, false, 5);
        assert!((d.mean - 0.5).abs() < 0.01);
        assert!((d.q50 - 0.5).abs() < 0.01);
        assert_eq!(d.head.len(), 5);
        // a roughly flat histogram: no bucket dominates
        let max = *d.hist.bins.iter().max().unwrap();
        let min = *d.hist.bins.iter().min().unwrap();
        assert!(
            max - min <= 2,
            "flat sample should bin evenly: {:?}",
            d.hist.bins
        );
    }

    #[test]
    fn dist2_recovers_a_known_correlation() {
        // y = x exactly ⇒ corr 1; y independent ⇒ corr ~0.
        let same: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, i as f64)).collect();
        assert!((Dist2::from_pairs(&same, 100).corr - 1.0).abs() < 1e-9);
        let anti: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, -(i as f64))).collect();
        assert!((Dist2::from_pairs(&anti, 100).corr + 1.0).abs() < 1e-9);
    }

    #[test]
    fn explain_ranks_by_absolute_correlation() {
        let e = Explain::from_candidates(
            1.0,
            vec![("a".into(), 0.2), ("b".into(), -0.9), ("c".into(), 0.5)],
        );
        assert_eq!(e.drivers[0].name, "b"); // strongest |corr| first
        assert_eq!(e.drivers[1].name, "c");
        assert_eq!(e.drivers[2].name, "a");
        // shares sum to ~1
        let s: f64 = e.drivers.iter().map(|d| d.share).sum();
        assert!((s - 1.0).abs() < 1e-9);
    }
}
