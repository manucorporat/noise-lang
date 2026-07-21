//! Sampling driver — the only forcing path in Phase 2 (PLAN.md "batch sampler").
//!
//! Backend-independent: compile the RV cone into a [`Sampler`] once (via [`InterpBackend`], the
//! default), seed the RNG once, then loop `ceil(N / batch_cap)` batches. The final partial batch
//! is sliced to the true remaining length so over-count never biases moments. Swapping the
//! backend (e.g. the emitted wasm kernel) changes only how a batch is produced, not this loop.

use crate::backend::{compile_root, compile_roots};
use crate::dist::{RvGraph, RvId};
use crate::error::{NoiseError, Result};
use crate::exec::StopCause;
use crate::reduce::{
    run_reduction_range, CondMomentsReducer, MomentAcc, MomentsReducer, Reducer, CHUNK_SAMPLES,
};
use web_time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use = "Moments carries the sampled mean/variance; computing them without using them wastes the sampling pass (finding F10)"]
pub struct Moments {
    pub mean: f64,
    pub variance: f64,
}

// === the precision-targeted query driver (PLAN-PRECISION Tracks B + D) ==========================

/// Which scalar a `P`/`E`/`Var` query estimates from the sampled moments — decides both the
/// estimate and its standard error (the adaptive loop's stopping rule reads the se).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    /// `P`: the mean of a 0/1 indicator; binomial se with the rule-of-three floor.
    Prob,
    /// `E`: the mean; se = `sqrt(var / m)`.
    Mean,
    /// `Var`: the population variance; asymptotic se ≈ `var · sqrt(2/m)` (exact for Gaussian).
    Variance,
}

/// How many draws a query takes: an explicit **count** (adaptivity off — exactly `n` lanes), a
/// **precision target** (keep drawing until `se ≤ max(abs, rel·|est|)`, bounded by the run
/// deadline — the standard numerical-integration contract, QUADPACK/GSL), or the **auto default**
/// (PLAN-PRECISION Phase 2 — see [`DEFAULT_REL`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawBudget {
    Fixed(u64),
    ToPrecision {
        rel: f64,
        abs: f64,
    },
    /// The untargeted default: `se ≤ DEFAULT_REL · max(|est|, sd)`, where `sd = se·√m` is the
    /// quantity's per-draw spread. The `sd` floor is what a bare `rel` target lacks — it makes the
    /// default **scale-free and always reachable** (a pure relative target never terminates on an
    /// estimate of ≈ 0). In the sd-floored regime it solves to `m ≥ DEFAULT_REL⁻²` effective
    /// draws; when `|est|` dominates the spread it stops earlier. Deliberately NOT expressible via
    /// `set_precision`: a declared target is a pure demand ("this many digits"), the default is a
    /// bounded-effort heuristic.
    Auto,
}

/// The auto default's relative target (Phase 2: precision default-on). 5e-3 ≈ 2–3 significant
/// digits; in the sd-floored regime it solves to `m ≥ rel⁻² = 40,000` effective draws — so an easy
/// untargeted query costs one pilot (65,536 lanes, ~25× less than the old fixed 1M default), while
/// a conditional/rare query honestly *extends* until its effective `m` gets there.
pub const DEFAULT_REL: f64 = 5e-3;

/// Why a query stopped short of its ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capped {
    /// The run's soft-stop tripped: a user `stop()` or the `max_time` deadline.
    Stopped(StopCause),
    /// The adaptive loop hit its absolute lane ceiling ([`LANE_CAP`]) without meeting the target
    /// (an unreachable target, e.g. a relative target on an estimate of exactly 0).
    Lanes,
}

/// One query's outcome: the estimate, its honest standard error (from the **true folded count**,
/// Track H), and how it ended. `count` is the effective sample size (the in-condition `m` for a
/// conditional query); `drawn` is the lanes actually swept.
pub struct QueryRun {
    pub est: f64,
    pub se: f64,
    pub count: u64,
    pub drawn: u64,
    pub capped: Option<Capped>,
    /// The se the stopping rule was targeting, when a precision target was in force (for the
    /// capped-run warning: "target ±X needs ~N draws").
    pub target_se: Option<f64>,
}

/// The adaptive pilot: 4 reducer chunks. Cheap (se-of-se ~0.3%), and already above the wasm
/// codegen gate (`MIN_DRAWS_WASM`), so pilot and growth stages share one compiled artifact.
const PILOT_N: u64 = 4 * CHUNK_SAMPLES as u64; // 65_536

/// Per-stage growth ceiling (×64): bounds the damage of a noisy pilot se (a lucky pilot under-
/// estimates variance → the CLT extrapolation overshoots). Two or three stages reach any realistic
/// target.
const GROWTH_CAP: u64 = 64;

/// CLT extrapolation margin: ask for 10% more than `n · (se/target)²` so the common case finishes
/// in one growth stage instead of a tail of tiny top-ups.
const GROWTH_MARGIN: f64 = 1.1;

/// Absolute lane ceiling of the adaptive loop — the backstop for an *unreachable* target (e.g. a
/// purely relative target on an estimate of exactly 0, whose target se is 0). The run deadline
/// (`max_time`) is the real guard; this only bounds a deadline-less run. 2⁴⁰ lanes ≈ 1.1e12 draws.
const LANE_CAP: u64 = 1 << 40;

/// The estimate and standard error `quantity` reads off a moments accumulator. `se = ∞` when no
/// draws have been folded (a target can never be met by an empty accumulator).
fn est_se(acc: &MomentAcc, quantity: Quantity) -> (f64, f64) {
    let m = acc.count();
    if m == 0 {
        return (0.0, f64::INFINITY);
    }
    let mf = m as f64;
    let mom = acc.into_moments();
    match quantity {
        Quantity::Prob => {
            let p = mom.mean;
            // Standard error of a Monte Carlo probability estimate; rule-of-three floor keeps it
            // finite (and extrapolating) when p̂ is 0 or 1.
            let se = (p * (1.0 - p) / mf).sqrt().max(0.5 / mf);
            (p, se)
        }
        Quantity::Mean => (mom.mean, (mom.variance / mf).sqrt()),
        Quantity::Variance => (mom.variance, mom.variance.abs() * (2.0 / mf).sqrt()),
    }
}

/// Round `n` up to a whole number of reducer chunks — stage boundaries must be chunk-aligned so
/// the staged fold is bit-identical to a single run (see [`run_reduction_range`]).
fn chunk_align(n: u64) -> u64 {
    n.div_ceil(CHUNK_SAMPLES as u64) * CHUNK_SAMPLES as u64
}

/// Drive one `P`/`E`/`Var` query to its budget — the single forcing path behind all three
/// (PLAN-PRECISION Track B). `cond` selects the conditioning reducer (NaN lanes skipped, `count`
/// = the in-condition `m`); the adaptive path needs no special casing for it, because the se is
/// computed from `m` and the CLT extrapolation scales `drawn` (lanes) by the same factor as `m`.
///
/// * [`DrawBudget::Fixed`] — one reduction over exactly `0..n`: bit-identical to the old fixed-n
///   behavior, including under a soft stop (partial fold, honest count).
/// * [`DrawBudget::ToPrecision`] / [`DrawBudget::Auto`] — pilot, then extrapolate-and-extend
///   (`n_need = n·(se/target)²·1.1`, per-stage growth ≤ ×64, chunk-aligned): each stage *extends*
///   the previous accumulator over fresh lanes, so the final result is bit-identical to a fixed
///   run at the same final n. The two differ only in the target se (pure demand vs the sd-floored
///   default). The stage size is additionally capped by measured throughput against the run
///   deadline (Track D): a stage is sized to roughly land on the deadline rather than overshoot it
///   64×, and the in-stage per-chunk deadline check bounds any misprediction to one check
///   granularity.
pub fn query_moments(
    graph: &RvGraph,
    root: RvId,
    cond: bool,
    quantity: Quantity,
    budget: DrawBudget,
    seed: u64,
) -> Result<QueryRun> {
    // One reducer per conditioning mode; both accumulate `MomentAcc`, so the loop is shared.
    let plain = MomentsReducer;
    let condr = CondMomentsReducer;
    let reduce = |lanes: std::ops::Range<u64>, carry: MomentAcc| {
        if cond {
            run_reduction_range(graph, root, lanes, seed, &condr, carry)
        } else {
            run_reduction_range(graph, root, lanes, seed, &plain, carry)
        }
    };
    let identity = plain.identity();

    if let DrawBudget::Fixed(n) = budget {
        let out = reduce(0..n, identity)?;
        let (est, se) = est_se(&out.acc, quantity);
        return Ok(QueryRun {
            est,
            se,
            count: out.acc.count(),
            drawn: n,
            capped: out.stopped.map(Capped::Stopped),
            target_se: None,
        });
    }

    // The stopping rule's target se, from the current estimate. A declared target is the pure
    // QUADPACK contract; the auto default adds the sd floor (see [`DrawBudget::Auto`]) — `se·√m`
    // reconstructs the estimator's per-draw spread whatever the quantity (binomial, mean, or the
    // variance's own asymptotic se), so one formula serves all three.
    let target_of = |est: f64, se: f64, m: u64| match budget {
        DrawBudget::ToPrecision { rel, abs } => abs.max(rel * est.abs()),
        DrawBudget::Auto => {
            let sd = if m > 0 { se * (m as f64).sqrt() } else { 0.0 };
            DEFAULT_REL * est.abs().max(sd)
        }
        DrawBudget::Fixed(_) => unreachable!("handled above"),
    };

    // Adaptive: pilot, then extrapolate-and-extend until the target, the deadline, or the cap.
    let mut acc = identity;
    let mut drawn = 0u64;
    let mut next = PILOT_N;
    loop {
        let stage_t = Instant::now();
        let out = reduce(drawn..next, acc)?;
        let stage_secs = stage_t.elapsed().as_secs_f64();
        acc = out.acc;
        let stage_lanes = next - drawn;
        drawn = next;
        let (est, se) = est_se(&acc, quantity);
        let target = target_of(est, se, acc.count());
        if let Some(cause) = out.stopped {
            return Ok(QueryRun {
                est,
                se,
                count: acc.count(),
                drawn,
                capped: Some(Capped::Stopped(cause)),
                target_se: Some(target),
            });
        }
        if se <= target {
            crate::profile::note(format!(
                "precision: target ±{target:.2e} met at n={drawn} (se ±{se:.2e})"
            ));
            return Ok(QueryRun {
                est,
                se,
                count: acc.count(),
                drawn,
                capped: None,
                target_se: Some(target),
            });
        }
        if drawn >= LANE_CAP {
            return Ok(QueryRun {
                est,
                se,
                count: acc.count(),
                drawn,
                capped: Some(Capped::Lanes),
                target_se: Some(target),
            });
        }
        // A conditional whose condition has not held ONCE by a generous sweep is (almost surely)
        // structurally never-true — bail out so the caller raises the teaching error instead of
        // sampling toward the lane cap. 2²⁴ lanes ≈ 16.8M draws, 16× the old fixed default.
        const COND_NEVER_CAP: u64 = 1 << 24;
        if cond && acc.count() == 0 && drawn >= COND_NEVER_CAP {
            return Ok(QueryRun {
                est,
                se,
                count: 0,
                drawn,
                capped: None,
                target_se: Some(target),
            });
        }

        // CLT sizing: se ∝ 1/√n (the rule-of-three floor regime extrapolates linearly, which this
        // also covers — each stage re-derives from the *current* se, whichever regime it is in).
        let need = if target > 0.0 && se.is_finite() {
            (drawn as f64 * (se / target) * (se / target) * GROWTH_MARGIN).min(LANE_CAP as f64)
        } else {
            LANE_CAP as f64
        };
        let mut n_next = (need as u64)
            .min(drawn.saturating_mul(GROWTH_CAP))
            .min(LANE_CAP);

        // Track D (deadline-aware sizing): from THIS stage's measured throughput, cap the next
        // stage at what the remaining deadline is predicted to afford. Throughput is per-backend
        // state implicitly — the gate re-decides per stage, and a backend switch mispredicts one
        // stage at most, which the in-stage per-chunk deadline check turns into "late by one chunk"
        // rather than "late by a stage".
        if let Some(deadline) = crate::exec::deadline() {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .map_or(0.0, |d| d.as_secs_f64());
            if stage_secs > 0.0 && stage_lanes > 0 {
                let rate = stage_lanes as f64 / stage_secs;
                let affordable = drawn + (rate * remaining) as u64;
                // Never size below one chunk: if the deadline is that close, the in-stage check
                // stops the stage almost immediately anyway, and a zero-length stage would spin.
                n_next = n_next.min(affordable.max(drawn + CHUNK_SAMPLES as u64));
            }
        }

        next = chunk_align(n_next.max(drawn + CHUNK_SAMPLES as u64));
        crate::profile::note(format!(
            "precision: stage n={drawn} se ±{se:.2e} target ±{target:.2e} → extend to n={next}"
        ));
    }
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
        // Soft stop / deadline (PLAN-PRECISION Track H): keep what the sink already consumed — a
        // sparser collect is a valid smaller-sample pass; the engine raises the run-level warning.
        if crate::exec::stop_cause().is_some() {
            break;
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

// (The old `cond_moments` fixed-n helper was absorbed into [`query_moments`] — the conditional
// reducer rides the same driver, fixed or adaptive, and its KNOWN HOLE (finding B2: an
// in-condition NaN quantity is dropped, not propagated) is unchanged and documented on
// [`crate::reduce::CondMomentsReducer`].)

/// Collect the in-condition draws of a conditioning root `select(C, quantity, NaN)` — every
/// non-NaN lane, in stream order — for a conditional quantile `Q(· | C)` or a conditional
/// introspection. NaN lanes (where `C` is false) are dropped, so the returned vector holds exactly
/// the `m ≈ n·P(C)` draws from the subpopulation where `C` holds (unsorted; the caller sorts).
/// Chunked like [`sample_n_par`], and the kept draws per chunk are fixed by `(seed, n)` alone, so
/// the result stays bit-identical for any thread count.
///
/// KNOWN HOLE (finding B2): the NaN filter can't tell "condition false" from "the quantity is NaN
/// on an in-condition lane" — an in-condition NaN quantity is dropped rather than propagated,
/// biasing the conditional quantile sample. The fix is a dedicated condition column (see
/// `Engine::query_cond`); deferred for the same interface-ripple reason.
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
        // Soft stop / deadline (Track H): a stopped plot/introspection pass renders what it
        // collected rather than being dropped.
        if crate::exec::stop_cause().is_some() {
            break;
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
            RvNode::Src(Source::Normal {
                mu: 0.0,
                sigma: 1.0,
            }),
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
                let z = g.push(
                    RvNode::Src(Source::Normal {
                        mu: 0.0,
                        sigma: 1.0,
                    }),
                    RvKind::Num,
                );
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

    /// The precision loop's correctness invariant (PLAN-PRECISION validation 1): a target-stopped
    /// adaptive run is **bit-identical** to a fixed-n run at the same final n — the staged fold
    /// walked exactly the chunks the single run would have.
    #[test]
    fn adaptive_run_is_bit_identical_to_fixed_run_at_its_final_n() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let half = g.push(RvNode::ConstNum(0.5), RvKind::Num);
        let hit = g.push(RvNode::Binary(BinOp::Lt, x, half), RvKind::Bool);

        let adaptive = query_moments(
            &g,
            hit,
            false,
            Quantity::Prob,
            DrawBudget::ToPrecision {
                rel: 2e-3,
                abs: 0.0,
            },
            7,
        )
        .unwrap();
        assert!(
            adaptive.capped.is_none(),
            "no deadline set — must hit the target"
        );
        assert!(
            adaptive.se <= 2e-3 * adaptive.est.abs(),
            "target missed: se {} est {}",
            adaptive.se,
            adaptive.est
        );
        assert!(
            adaptive.drawn > PILOT_N,
            "rel=2e-3 on p=0.5 needs more than the pilot, drew {}",
            adaptive.drawn
        );

        let fixed = query_moments(
            &g,
            hit,
            false,
            Quantity::Prob,
            DrawBudget::Fixed(adaptive.drawn),
            7,
        )
        .unwrap();
        assert_eq!(adaptive.est.to_bits(), fixed.est.to_bits());
        assert_eq!(adaptive.se.to_bits(), fixed.se.to_bits());
        assert_eq!(adaptive.count, fixed.count);
    }

    /// Validation 6 (PLAN-PRECISION): a tight conditioning no longer silently gets a fraction of
    /// the budget — the loop reads the se off the in-condition count m, so a ~3%-acceptance
    /// conditional reaches the same relative target as an unconditional query, by drawing
    /// proportionally more lanes.
    #[test]
    fn conditional_query_reaches_the_target_by_drawing_more() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        // Condition: X < 0.03 (~3% acceptance). Quantity: X < 0.015 given that (p ≈ 0.5).
        let c_thresh = g.push(RvNode::ConstNum(0.03), RvKind::Num);
        let cond = g.push(RvNode::Binary(BinOp::Lt, x, c_thresh), RvKind::Bool);
        let q_thresh = g.push(RvNode::ConstNum(0.015), RvKind::Num);
        let event = g.push(RvNode::Binary(BinOp::Lt, x, q_thresh), RvKind::Bool);
        let nan = g.push(RvNode::ConstNum(f64::NAN), RvKind::Num);
        // Numeric event indicator under the condition sentinel: select(cond, event, NaN).
        let root = g.push(
            RvNode::Select {
                cond,
                a: event,
                b: nan,
            },
            RvKind::Num,
        );

        let rel = 5e-3;
        let out = query_moments(
            &g,
            root,
            true,
            Quantity::Prob,
            DrawBudget::ToPrecision { rel, abs: 0.0 },
            7,
        )
        .unwrap();
        assert!(out.capped.is_none());
        assert!((out.est - 0.5).abs() < 0.02, "P(A|C) = {}", out.est);
        assert!(
            out.se <= rel * out.est.abs(),
            "target missed: se {}",
            out.se
        );
        // m is the in-condition count; the lanes swept must be ~1/0.03 ≈ 33× that.
        assert!(
            out.drawn > 10 * out.count,
            "drawn {} should dwarf in-condition m {}",
            out.drawn,
            out.count
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
