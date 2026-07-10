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
//! once; the adapters only differ in how they obtain the `RvId` and how they render the payload.

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

/// Bins in a numeric histogram (a boolean quantity uses 2). ~30 is enough resolution for an ASCII
/// sparkline or a playground bar chart without over-fragmenting a modest in-condition sample.
const NUM_BINS: usize = 30;

/// Default scatter points kept for a [`Dist2`] (the full pair sample drives the *statistics*; the
/// point list is a subsample for plotting, so a chart isn't asked to draw 200k dots).
const SCATTER_POINTS: usize = 800;

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
            hist: histogram(draws, boolean),
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
        let corr = if sd_a > 0.0 && sd_b > 0.0 { (cov / (sd_a * sd_b)).clamp(-1.0, 1.0) } else { 0.0 };
        let stride = (pairs.len() / max_points.max(1)).max(1);
        let points = pairs.iter().step_by(stride).take(max_points).copied().collect();
        Dist2 { n, corr, cov, mean_a, mean_b, sd_a, sd_b, points }
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
                let share = if total > 0.0 { corr * corr / total } else { 0.0 };
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
    let draws = if conditional {
        sampler::cond_sample_n(graph, root, n, seed)
    } else {
        sampler::sample_n(graph, root, n, seed)
    };
    if draws.is_empty() {
        return None;
    }
    Some(Dist1::from_draws(&draws, boolean, head_k))
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
pub fn grid(graph: &RvGraph, roots: &[RvId], rows: usize, cols: usize, n: usize, seed: u64) -> DistGrid {
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
    CorrMatrix { n: roots.len(), corr: sampler::corr_matrix(graph, roots, n, seed) }
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
        fc.mean.push(col.iter().sum::<f64>() / col.len().max(1) as f64);
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
/// the same rule `builtins::Q` uses, kept private here so introspection has no cross-module dep).
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Bin `draws` into a [`Histogram`]. A boolean quantity is the two buckets `false`/`true`; a numeric
/// one is `NUM_BINS` equal-width buckets over `[min, max]` (a degenerate point mass is one bucket).
fn histogram(draws: &[f64], boolean: bool) -> Histogram {
    if boolean {
        let mut bins = vec![0u64; 2];
        for &x in draws {
            bins[(x != 0.0) as usize] += 1;
        }
        return Histogram { lo: 0.0, hi: 1.0, bins };
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
    if !lo.is_finite() || !hi.is_finite() || lo == hi {
        return Histogram { lo, hi, bins: vec![draws.len() as u64] };
    }
    let span = hi - lo;
    let mut bins = vec![0u64; NUM_BINS];
    for &x in draws {
        let mut idx = (((x - lo) / span) * NUM_BINS as f64) as usize;
        if idx >= NUM_BINS {
            idx = NUM_BINS - 1; // the max draw lands in the last bucket, not one past it
        }
        bins[idx] += 1;
    }
    Histogram { lo, hi, bins }
}

// --- the user-facing summary value + its CLI rendering ----------------------------------------

/// Which rendering an in-source builtin asked for. The *engine* always computes the full payload;
/// the view only picks what the CLI `Display` shows (the playground gets the whole payload and picks
/// its own). So `describe`/`hist`/`samples` differ only here, never in what was measured.
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
/// to. Carried in a [`Value::Summary`](crate::Value) so it composes as an ordinary value and
/// `Display`s as an ASCII block in the CLI; the sidecar serializes the same `payload` for the
/// playground. `label`/`label_b` are the source names of the introspected variable(s), for the
/// heading.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub view: View,
    pub label: String,
    pub label_b: Option<String>,
    pub payload: Payload,
}

/// Unicode block ramp for sparklines / bars (8 levels, empty handled by the caller).
const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// A sparkline string for histogram bin counts (each bin → one block glyph scaled to the tallest).
fn sparkline(bins: &[u64]) -> String {
    let max = bins.iter().copied().max().unwrap_or(0);
    if max == 0 {
        return String::new();
    }
    bins.iter()
        .map(|&c| {
            if c == 0 {
                ' '
            } else {
                let lvl = ((c as f64 / max as f64) * (BLOCKS.len() - 1) as f64).round() as usize;
                BLOCKS[lvl]
            }
        })
        .collect()
}

/// Trim float dust for compact display (mirrors `value::format_num`'s intent, local to avoid a dep).
fn fmt_n(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let s = format!("{x:.4}");
    let t = s.trim_end_matches('0').trim_end_matches('.');
    if t.is_empty() || t == "-0" { "0".to_string() } else { t.to_string() }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.payload, self.view) {
            (Payload::One(d), View::Samples) => {
                let body =
                    d.head.iter().map(|x| fmt_n(*x)).collect::<Vec<_>>().join(", ");
                write!(f, "{} samples: [{}]", self.label, body)
            }
            (Payload::One(d), View::Hist) => fmt_hist(f, &self.label, d),
            (Payload::One(d), _) => fmt_describe(f, &self.label, d),
            (Payload::Two(d), View::Scatter) => fmt_scatter(f, self, d),
            (Payload::Two(d), _) => write!(
                f,
                "corr({}, {}) = {}   cov = {}",
                self.label,
                self.label_b.as_deref().unwrap_or("?"),
                fmt_n(d.corr),
                fmt_n(d.cov),
            ),
            (Payload::Explain(e), _) => fmt_explain(f, &self.label, e),
            (Payload::Value(v), _) => fmt_value(f, &self.label, v),
            (Payload::Grid(g), _) => fmt_grid(f, &self.label, g),
            (Payload::CorrMatrix(c), _) => fmt_corr_matrix(f, &self.label, c),
            (Payload::Fan(c), _) => fmt_fan(f, &self.label, c),
        }
    }
}

fn fmt_value(f: &mut fmt::Formatter<'_>, label: &str, v: &ValueCard) -> fmt::Result {
    if v.se > 0.0 {
        let half = 1.96 * v.se;
        write!(f, "{} = {} ± {}   95% CI {} … {}", label, fmt_n(v.val), fmt_n(v.se), fmt_n(v.val - half), fmt_n(v.val + half))
    } else {
        write!(f, "{} = {}", label, fmt_n(v.val))
    }
}

fn fmt_grid(f: &mut fmt::Formatter<'_>, label: &str, g: &DistGrid) -> fmt::Result {
    if g.mean.is_empty() {
        return write!(f, "{} (empty)", label);
    }
    if g.is_series() {
        // A vector: the per-index means as a sparkline, with the range.
        let lo = g.mean.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = g.mean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bins: Vec<u64> = scale_to_levels(&g.mean, lo, hi);
        return write!(f, "{} [{}]  ({} elems, mean {}…{})", label, sparkline(&bins), g.mean.len(), fmt_n(lo), fmt_n(hi));
    }
    // A matrix: a heatmap of the means, one row per line.
    writeln!(f, "{} ({}×{} mean)", label, g.rows, g.cols)?;
    let lo = g.mean.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = g.mean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for r in 0..g.rows {
        let row: String = (0..g.cols).map(|c| heat_glyph(g.mean[r * g.cols + c], lo, hi)).collect();
        writeln!(f, "  {row}")?;
    }
    write!(f, "  low {} ░▒▓█ {} high", fmt_n(lo), fmt_n(hi))
}

fn fmt_corr_matrix(f: &mut fmt::Formatter<'_>, label: &str, c: &CorrMatrix) -> fmt::Result {
    writeln!(f, "corr({}) — {}×{}", label, c.n, c.n)?;
    for i in 0..c.n {
        // Key the ASCII ramp on |corr| (strength of dependence): near-zero stays faint, ±1 solid —
        // so iid elements read as a clean diagonal rather than noisy mid-tones. (Sign shows via color
        // on the web renderer.)
        let row: String = (0..c.n).map(|j| heat_glyph(c.corr[i * c.n + j].abs(), 0.0, 1.0)).collect();
        writeln!(f, "  {row}")?;
    }
    write!(f, "  |corr|: 0 ░▒▓█ 1")
}

fn fmt_fan(f: &mut fmt::Formatter<'_>, label: &str, c: &FanChart) -> fmt::Result {
    if c.q50.is_empty() {
        return write!(f, "fan {label} (empty)");
    }
    // ONE global scale (the outermost band's extremes), so the stacked rows read as a cone: the q95
    // row visibly rides high and the q05 row low. Each row is a sparkline of that quantile's path.
    let lo = c.q05.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = c.q95.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    writeln!(f, "fan {label}  [{} … {}]  ({} elems, n={})", fmt_n(lo), fmt_n(hi), c.cols, c.n)?;
    let bands: [(&str, &[f64]); 5] =
        [("q95", &c.q95), ("q75", &c.q75), ("q50", &c.q50), ("q25", &c.q25), ("q05", &c.q05)];
    for (i, (name, row)) in bands.iter().enumerate() {
        // Directly level→glyph (NOT `sparkline`, which re-normalizes each row to its own max and
        // would collapse the shared scale). Level 0 still draws ▁ so a flat-low band stays visible.
        let line: String = scale_to_levels(row, lo, hi)
            .iter()
            .map(|&l| BLOCKS[(l as usize).min(BLOCKS.len() - 1)])
            .collect();
        write!(f, "  {name} {line}")?;
        if i + 1 < bands.len() {
            writeln!(f)?;
        }
    }
    Ok(())
}

/// Map values onto the 8 sparkline levels, given a `[lo, hi]` range.
fn scale_to_levels(xs: &[f64], lo: f64, hi: f64) -> Vec<u64> {
    let span = (hi - lo).max(1e-12);
    xs.iter().map(|&x| (((x - lo) / span) * 7.0).round() as u64).collect()
}

/// A single heatmap glyph for `x` within `[lo, hi]` (4 solid levels ░ ▒ ▓ █ — no blank, so an empty
/// cell never reads as "missing").
fn heat_glyph(x: f64, lo: f64, hi: f64) -> char {
    let span = (hi - lo).max(1e-12);
    let t = ((x - lo) / span).clamp(0.0, 1.0);
    ['░', '▒', '▓', '█'][((t * 3.0).round() as usize).min(3)]
}

fn fmt_describe(f: &mut fmt::Formatter<'_>, label: &str, d: &Dist1) -> fmt::Result {
    if d.boolean {
        // An event: the informative number is P(true), not quantiles.
        return write!(f, "{} (n={})  P(true) = {}", label, d.n, fmt_n(d.mean));
    }
    write!(
        f,
        "{} (n={})  mean {}  sd {}  [{} {}]\n  q05 {}  q25 {}  med {}  q75 {}  q95 {}  {}",
        label,
        d.n,
        fmt_n(d.mean),
        fmt_n(d.sd),
        fmt_n(d.min),
        fmt_n(d.max),
        fmt_n(d.q05),
        fmt_n(d.q25),
        fmt_n(d.q50),
        fmt_n(d.q75),
        fmt_n(d.q95),
        sparkline(&d.hist.bins),
    )
}

fn fmt_hist(f: &mut fmt::Formatter<'_>, label: &str, d: &Dist1) -> fmt::Result {
    writeln!(f, "{} (n={})", label, d.n)?;
    let total: u64 = d.hist.bins.iter().sum();
    let total = total.max(1);
    let max = d.hist.bins.iter().copied().max().unwrap_or(1).max(1);
    let nb = d.hist.bins.len();
    let span = d.hist.hi - d.hist.lo;
    for (i, &c) in d.hist.bins.iter().enumerate() {
        let edge = if d.boolean {
            (if i == 0 { "false" } else { "true " }).to_string()
        } else if nb <= 1 {
            fmt_n(d.hist.lo)
        } else {
            format!("{:>8}", fmt_n(d.hist.lo + span * (i as f64 / nb as f64)))
        };
        let width = ((c as f64 / max as f64) * 40.0).round() as usize;
        let pct = 100.0 * c as f64 / total as f64;
        writeln!(f, "  {} │{}{} {:.1}%", edge, "█".repeat(width), " ".repeat(40 - width), pct)?;
    }
    Ok(())
}

fn fmt_scatter(f: &mut fmt::Formatter<'_>, s: &Summary, d: &Dist2) -> fmt::Result {
    let b = s.label_b.as_deref().unwrap_or("?");
    write!(f, "{} vs {}   corr = {}\n", s.label, b, fmt_n(d.corr))?;
    // A small ASCII grid (width × height) of the subsampled points, axes auto-scaled.
    const W: usize = 40;
    const H: usize = 12;
    let (mut x0, mut x1, mut y0, mut y1) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in &d.points {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let (dx, dy) = ((x1 - x0).max(1e-12), (y1 - y0).max(1e-12));
    let mut grid = vec![vec![' '; W]; H];
    for &(x, y) in &d.points {
        let cx = (((x - x0) / dx) * (W - 1) as f64).round() as usize;
        let cy = (((y - y0) / dy) * (H - 1) as f64).round() as usize;
        grid[H - 1 - cy][cx] = '•';
    }
    for row in &grid {
        writeln!(f, "  {}", row.iter().collect::<String>())?;
    }
    Ok(())
}

fn fmt_explain(f: &mut fmt::Formatter<'_>, label: &str, e: &Explain) -> fmt::Result {
    writeln!(f, "explain {}   sd = {}", label, fmt_n(e.sd))?;
    if e.drivers.is_empty() {
        return write!(f, "  (no named upstream variables)");
    }
    for d in &e.drivers {
        let width = (d.share * 20.0).round() as usize;
        writeln!(
            f,
            "  {:<12} {}{} {:>4.0}%  (corr {})",
            d.name,
            "█".repeat(width),
            " ".repeat(20 - width),
            d.share * 100.0,
            fmt_n(d.corr),
        )?;
    }
    Ok(())
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
        assert!(max - min <= 2, "flat sample should bin evenly: {:?}", d.hist.bins);
    }

    #[test]
    fn dist2_recovers_a_known_correlation() {
        // y = x exactly ⇒ corr 1; y independent ⇒ corr ~0.
        let same: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, i as f64)).collect();
        assert!((Dist2::from_pairs(&same, 100).corr - 1.0).abs() < 1e-9);
        let anti: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, -(i as f64))).collect();
        assert!((Dist2::from_pairs(&anti, 100).corr + 1.0).abs() < 1e-9);
    }

    /// The fan rendering keeps ONE global scale: a q05 row pinned at the global low must render at
    /// the bottom glyph even though its own row-max is far below the global high (i.e. rows are
    /// level-mapped directly, never re-normalized per row like `sparkline`).
    #[test]
    fn fan_display_shares_one_global_scale() {
        let c = FanChart {
            cols: 3,
            n: 100,
            q05: vec![0.0, 0.0, 0.0],
            q25: vec![1.0, 1.0, 1.0],
            q50: vec![2.0, 2.0, 2.0],
            q75: vec![3.0, 3.0, 3.0],
            q95: vec![8.0, 8.0, 8.0],
            mean: vec![2.0, 2.0, 2.0],
        };
        let s = Summary { view: View::Fan, label: "path".into(), label_b: None, payload: Payload::Fan(c) };
        let shown = s.to_string();
        assert!(shown.starts_with("fan path"), "{shown}");
        assert!(shown.contains("0 … 8"), "global range legend: {shown}");
        assert!(shown.contains("q95 ███"), "q95 at the global high renders full blocks: {shown}");
        assert!(shown.contains("q05 ▁▁▁"), "q05 at the global low renders bottom blocks: {shown}");
        // q25 sits at 1/8 of the range — a low block, NOT a full one (no per-row re-normalizing).
        assert!(shown.contains("q25 ▂▂▂"), "q25 keeps the shared scale: {shown}");
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
