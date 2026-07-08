//! Builtin dispatch for `Expr::Call`.
//!
//! Phase 2 shipped `unif`. Phase 3 (cheap correctness wins) adds the discrete uniform
//! `unif_int`, `bernoulli`, and the probability query `P`. Args are already evaluated to
//! `Value`s by the engine. Distribution constructors take deterministic numbers and return a
//! `Dist`; `P` forces sampling and returns a probability as a plain `Num` (so it composes in
//! arithmetic, e.g. `4 * P(C)`).

use std::rc::Rc;

use crate::dist::{DistArg, Recipe, RvGraph, RvId, RvKind};
use crate::error::{NoiseError, Result, Span};
use crate::sampler;
use crate::signal::{SigExpr, Wave};
use crate::value::Value;

/// Default Monte Carlo budget for a language-surface `P`/`E`/`Var`/`Q` when the call carries no
/// explicit sample count. Large enough that a probability estimate's standard error
/// (`sqrt(p(1-p)/N) ≈ 5e-4`) is tight. A program can lower (or raise) this for the whole run with
/// `engine::set_max_samples(N)`; an explicit per-call count (`P(event, n)`) still overrides both.
pub const P_DEFAULT_N: usize = 1_000_000;
/// Default per-query **operation** ceiling (`engine::set_max_opts`): a built-in cap so no single
/// `P`/`E`/`Var`/`Q` spends more than ~1e9 per-lane ops, keeping worst-case runtime bounded out of
/// the box. A query over a cone of `C` distinct nodes auto-clamps to `1e9 / C` draws, so the cap
/// only bites for `C > 1000` (with the default `P_DEFAULT_N` draws) — ordinary small-cone queries,
/// even at millions of draws, are never clamped; only genuinely large models are. A program raises
/// it with `engine::set_max_samples`'s sibling `engine::set_max_opts(N)`.
pub const MAX_OPS_DEFAULT: u64 = 1_000_000_000;
/// Default **sampling resolution** (`engine::set_resolution`, PLAN-SIGNALS §1.2): the length at
/// which reducers (`mse`/`mean`/`sum`/…) render a lazy signal that never met an explicit length.
/// The time-axis twin of [`P_DEFAULT_N`] — a measurement knob, not part of the question — sized
/// so toy programs never need to mention it.
pub const RESOLUTION_DEFAULT: usize = 256;
/// A fixed seed keeps a run reproducible; threading an explicit seed through `P` is a later
/// refinement.
const P_DEFAULT_SEED: u64 = 0;

/// Called from `eval.rs`'s `Expr::Call` arm.
///
/// Distribution constructors (`unif`, `unif_int`, `bernoulli`) return a **recipe** — an undrawn
/// `Value::Recipe`, *not* a graph node. They never touch the graph: drawing is `~`'s job (see
/// `Engine::draw`). `P` reads the graph to estimate a probability.
///
/// `default_n` is the Monte Carlo budget the sampling builtins (`P`/`E`/`Var`/`Q`) use when the
/// call carries no explicit sample count — the engine threads its current setting (default
/// [`P_DEFAULT_N`], adjustable via `engine::set_max_samples`) through here. `max_opts` is the
/// optional per-query operation budget (`engine::set_max_opts`, `0` = unlimited): each query
/// auto-clamps its draw count so `draws × cone-node-count ≤ max_opts` (see [`clamp_to_op_budget`]).
pub fn call(
    name: &str,
    arg_vals: &[Value],
    graph: &RvGraph,
    default_n: usize,
    max_opts: u64,
    span: Span,
    check: bool,
) -> Result<Value> {
    match name {
        "unif" => {
            let [lo, hi] = two_args(name, arg_vals, graph, span)?;
            match both_const(lo, hi) {
                Some([lo, hi]) => Ok(Value::Recipe(Recipe::Uniform { lo, hi })),
                None => Ok(Value::Recipe(Recipe::UniformDyn { lo, hi })),
            }
        }
        "unif_int" => {
            let [lo, hi] = two_args(name, arg_vals, graph, span)?;
            match both_const(lo, hi) {
                Some([lo, hi]) => {
                    if lo > hi {
                        return Err(NoiseError::runtime(
                            format!("unif_int needs lo <= hi, got ({lo}, {hi})"),
                            span,
                        ));
                    }
                    // Treat bounds as inclusive integers (normalize once, at recipe construction).
                    Ok(Value::Recipe(Recipe::UniformInt { lo: lo.round(), hi: hi.round() }))
                }
                // A random bound can't be range-checked here; the draw transform handles it per lane.
                None => Ok(Value::Recipe(Recipe::UniformIntDyn { lo, hi })),
            }
        }
        "bernoulli" => {
            let p = one_arg(name, arg_vals, graph, span)?;
            match p {
                DistArg::Const(p) => {
                    if !(0.0..=1.0).contains(&p) {
                        return Err(NoiseError::runtime(
                            format!("bernoulli(p) needs 0 <= p <= 1, got {p}"),
                            span,
                        ));
                    }
                    Ok(Value::Recipe(Recipe::Bernoulli { p }))
                }
                // A random p isn't range-checked: `(U < p)` is true with the lane's clamped p.
                rv => Ok(Value::Recipe(Recipe::BernoulliDyn { p: rv })),
            }
        }
        "normal" | "normal_int" => {
            let [mu, sigma] = two_args(name, arg_vals, graph, span)?;
            let int = name == "normal_int";
            match both_const(mu, sigma) {
                Some([mu, sigma]) => {
                    if sigma < 0.0 || !sigma.is_finite() || !mu.is_finite() {
                        return Err(NoiseError::runtime(
                            format!("{name}(mu, sigma) needs finite mu and sigma >= 0, got ({mu}, {sigma})"),
                            span,
                        ));
                    }
                    Ok(Value::Recipe(if int {
                        Recipe::NormalInt { mu, sigma }
                    } else {
                        Recipe::Normal { mu, sigma }
                    }))
                }
                None => Ok(Value::Recipe(Recipe::NormalDyn { mu, sigma, int })),
            }
        }
        "normal_complex" => {
            // Circularly-symmetric complex Gaussian: re/im each ~ N(0, sigma/√2), so E|z|² = sigma².
            let sigma = one_num(name, arg_vals, span)?;
            if sigma < 0.0 || !sigma.is_finite() {
                return Err(NoiseError::runtime(
                    format!("normal_complex(sigma) needs a finite sigma >= 0, got {sigma}"),
                    span,
                ));
            }
            Ok(Value::Recipe(Recipe::NormalComplex { sigma }))
        }
        "exponential" | "exponential_int" => {
            let rate = one_arg(name, arg_vals, graph, span)?;
            let int = name == "exponential_int";
            match rate {
                DistArg::Const(rate) => {
                    if rate <= 0.0 || !rate.is_finite() {
                        return Err(NoiseError::runtime(
                            format!("{name}(rate) needs a finite rate > 0, got {rate}"),
                            span,
                        ));
                    }
                    Ok(Value::Recipe(if int { Recipe::ExpInt { rate } } else { Recipe::Exp { rate } }))
                }
                rv => Ok(Value::Recipe(Recipe::ExpDyn { rate: rv, int })),
            }
        }
        "poisson" => {
            let lambda = one_num(name, arg_vals, span)?;
            if lambda <= 0.0 || !lambda.is_finite() {
                return Err(NoiseError::runtime(
                    format!("poisson(lambda) needs a finite lambda > 0, got {lambda}"),
                    span,
                ));
            }
            Ok(Value::Recipe(Recipe::Poisson { lambda }))
        }
        "geometric" => {
            let p = one_num(name, arg_vals, span)?;
            if !(0.0..=1.0).contains(&p) || p <= 0.0 {
                return Err(NoiseError::runtime(
                    format!("geometric(p) needs 0 < p <= 1, got {p}"),
                    span,
                ));
            }
            Ok(Value::Recipe(Recipe::Geometric { p }))
        }
        "rotation" => {
            // A random d×d orthonormal matrix — a *recipe* like any other distribution, drawn with
            // `~` (the actual Gram–Schmidt source draws happen in `Engine::draw_rotation`). `d` must
            // be a non-negative integer (the matrix dimension).
            let d = one_num(name, arg_vals, span)?;
            if d.fract() != 0.0 || d < 0.0 || !d.is_finite() {
                return Err(NoiseError::runtime(
                    format!("rotation(d) needs a non-negative integer dimension, got {d}"),
                    span,
                ));
            }
            Ok(Value::Recipe(Recipe::Rotation { d: d as usize }))
        }
        "permutation" => {
            // A uniform random permutation of `0..n` — a *recipe* drawn with `~` (the iid uniform
            // key draws and their argsort happen in `Engine::draw_permutation`). `n` must be a
            // non-negative integer (the permutation length).
            let n = one_num(name, arg_vals, span)?;
            if n.fract() != 0.0 || n < 0.0 || !n.is_finite() {
                return Err(NoiseError::runtime(
                    format!("permutation(n) needs a non-negative integer length, got {n}"),
                    span,
                ));
            }
            Ok(Value::Recipe(Recipe::Permutation { n: n as usize }))
        }
        "sqrt" => {
            let x = one_num(name, arg_vals, span)?;
            if x < 0.0 {
                return Err(NoiseError::runtime(format!("sqrt needs x >= 0, got {x}"), span));
            }
            Ok(Value::Num(x.sqrt()))
        }
        "log" | "log10" => {
            // Natural / base-10 logarithm. Domain x > 0 (log of 0 or negative is undefined).
            let x = one_num(name, arg_vals, span)?;
            if x <= 0.0 {
                return Err(NoiseError::runtime(format!("{name} needs x > 0, got {x}"), span));
            }
            Ok(Value::Num(if name == "log" { x.ln() } else { x.log10() }))
        }
        "gcd" => {
            // Greatest common divisor (Euclid), defined on integers via |a|, |b|; gcd(0, 0) = 0.
            // Deterministic integer op (a `Dist` argument errors in `as_num`).
            let [a, b] = two_nums(name, arg_vals, span)?;
            let mut a = as_int(name, a, span)?.unsigned_abs();
            let mut b = as_int(name, b, span)?.unsigned_abs();
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            Ok(Value::Num(a as f64))
        }
        "modpow" => {
            // Modular exponentiation base^exp mod modulus, by square-and-multiply in `i128` — exact
            // (no `base^exp` overflow) and O(log exp). exp >= 0, modulus > 0. The explicit form of
            // the `(base ^ exp) % modulus` idiom (which silently loses precision past 2^53).
            let [base, exp, modulus] = three_nums(name, arg_vals, span)?;
            let base = as_int(name, base, span)?;
            let exp = as_int(name, exp, span)?;
            let modulus = as_int(name, modulus, span)?;
            if exp < 0 {
                return Err(NoiseError::runtime(
                    format!("modpow exponent must be >= 0, got {exp}"),
                    span,
                ));
            }
            if modulus <= 0 {
                return Err(NoiseError::runtime(
                    format!("modpow modulus must be > 0, got {modulus}"),
                    span,
                ));
            }
            Ok(Value::Num(modpow_int(base, exp as u64, modulus) as f64))
        }
        "E" | "Var" => moment(name, arg_vals, graph, default_n, max_opts, span, check),
        "round" => {
            // round(x, digits) — round x to `digits` decimal places. Handy for messages.
            let [x, digits] = two_nums(name, arg_vals, span)?;
            let factor = 10f64.powi(digits as i32);
            Ok(Value::Num((x * factor).round() / factor))
        }
        "P" => prob(arg_vals, graph, default_n, max_opts, span, check),
        "Q" => quantile(arg_vals, graph, default_n, max_opts, span, check),
        // --- pure collection builtins (no graph access; PLAN-COLLECTIONS §3) ---
        "Len" => len(name, arg_vals, span),
        // pure vector constructors / rearrangers (graph-free; just move `Value`s around)
        "iota" => iota(name, arg_vals, span),
        "ones" => filled(name, arg_vals, 1.0, span),
        "zeros" => filled(name, arg_vals, 0.0, span),
        "transpose" => transpose(name, arg_vals, span),
        // signal generators (lazy). `sample` is intercepted in `eval.rs` (materializing may draw
        // noise RV nodes and read the realization cache, so it needs `&mut Engine`).
        "sine" | "cosine" => wave(name, arg_vals, span),
        // `Print` is intercepted in `eval.rs` (it needs `&mut self` to append to the capture
        // buffer), so it never reaches here; this arm is unreachable and kept only for clarity.
        "Print" => Ok(Value::Unit),
        other => Err(NoiseError::runtime(
            format!("unknown function '{other}'"),
            span,
        )),
    }
}

/// `P(event)` or `P(event, n)` — the probability that `event` is true, in `[0, 1]`. Accepts a
/// bool random variable (estimated by Monte Carlo over `n` samples, default `1e6`) or a
/// deterministic bool (0 or 1 exactly). A numeric argument is an error: a probability is only
/// defined for an event.
///
/// The estimate is **auto-rounded to its confidence precision**: a Monte Carlo estimate from
/// `n` samples has standard error `sqrt(p(1-p)/n)`, so only the digits above that error are
/// meaningful. More samples → smaller error → more digits, with no manual `round`.
/// Auto-clamp a query's draw count to the per-query operation budget (`engine::set_max_opts`).
/// A forcing over `root`'s cone costs `n × C` per-lane ops, where `C` is the cone's distinct-node
/// count ([`crate::kernel::cost`]). With a budget `max_opts > 0`, cap `n` at `max_opts / C` so the
/// query never spends more than the budget — but never below 1 (a query always draws at least once,
/// even if a single draw already exceeds the budget). `max_opts == 0` means unlimited (no clamp).
/// This makes each query's complexity deterministic in the model size: a heavier cone simply draws
/// fewer samples (a looser estimate) instead of doing unbounded work.
fn clamp_to_op_budget(n: usize, graph: &RvGraph, root: RvId, max_opts: u64) -> usize {
    if max_opts == 0 {
        return n;
    }
    let cone = crate::kernel::cost(graph, root).ops.max(1);
    let cap = (max_opts / cone).max(1) as usize;
    n.min(cap)
}

fn prob(arg_vals: &[Value], graph: &RvGraph, default_n: usize, max_opts: u64, span: Span, check: bool) -> Result<Value> {
    if arg_vals.is_empty() || arg_vals.len() > 2 {
        return Err(NoiseError::runtime(
            format!(
                "P expects 1 or 2 arguments (event, optional sample count), got {}",
                arg_vals.len()
            ),
            span,
        ));
    }
    let n = match arg_vals.get(1) {
        Some(v) => {
            let n = as_num(v, span)?;
            if n < 1.0 || !n.is_finite() {
                return Err(NoiseError::runtime(
                    format!("P sample count must be a finite number >= 1, got {n}"),
                    span,
                ));
            }
            n as usize
        }
        None => default_n,
    };
    match &arg_vals[0] {
        // A deterministic event is exact — no sampling, no error.
        Value::Bool(b) => Ok(Value::Est { val: if *b { 1.0 } else { 0.0 }, se: 0.0 }),
        Value::Dist(id) if graph.kind(*id) == RvKind::Bool => {
            // Validate-only mode: the event graph is built and type-checked; skip the Monte Carlo
            // estimate and hand back a neutral, in-range placeholder (a valid probability, so it's
            // safe if it flows into a range-checked constructor downstream).
            if check {
                return Ok(Value::Est { val: 0.5, se: 0.0 });
            }
            // A finite-discrete event is a finite sum — answer exactly (se = 0), no sampling.
            if let Some(ex) = crate::enumerate::try_enumerate(graph, *id, max_opts) {
                return Ok(Value::Est { val: ex.mean(), se: 0.0 });
            }
            let n = clamp_to_op_budget(n, graph, *id, max_opts);
            let p_hat = sampler::moments(graph, *id, n, P_DEFAULT_SEED).mean;
            // Standard error of a Monte Carlo probability estimate; rule-of-three floor keeps
            // it finite when p_hat is 0 or 1. Display rounds to the digits this justifies.
            let nf = n as f64;
            let se = (p_hat * (1.0 - p_hat) / nf).sqrt().max(0.5 / nf);
            Ok(Value::Est { val: p_hat, se })
        }
        Value::Dist(id) => Err(NoiseError::runtime(
            format!(
                "P expects an event (bool), got {} — did you mean a comparison like `X < 0`?",
                graph.kind(*id).type_name()
            ),
            span,
        )),
        other => Err(NoiseError::runtime(
            format!("P expects an event (bool), got {}", other.type_name()),
            span,
        )),
    }
}

/// `E(x)` / `Var(x)` — the Monte Carlo expectation or variance of a *numeric* quantity, with a
/// standard error (`E` ↔ mean, `Var` ↔ population variance). `P` is the special case for
/// events; `E`/`Var` are total over any number or numeric RV (a bool RV works too — its mean is
/// `P`). A deterministic number is exact: `E(5) = 5 ± 0`, `Var(5) = 0 ± 0`.
fn moment(name: &str, arg_vals: &[Value], graph: &RvGraph, default_n: usize, max_opts: u64, span: Span, check: bool) -> Result<Value> {
    if arg_vals.is_empty() || arg_vals.len() > 2 {
        return Err(NoiseError::runtime(
            format!("{name} expects 1 or 2 arguments (quantity, optional sample count), got {}", arg_vals.len()),
            span,
        ));
    }
    let n = match arg_vals.get(1) {
        Some(v) => {
            let n = as_num(v, span)?;
            if n < 1.0 || !n.is_finite() {
                return Err(NoiseError::runtime(
                    format!("{name} sample count must be a finite number >= 1, got {n}"),
                    span,
                ));
            }
            n as usize
        }
        None => default_n,
    };
    let want_var = name == "Var";
    match &arg_vals[0] {
        // A deterministic value is exact: mean = it, variance = 0, no sampling.
        Value::Num(x) => Ok(Value::Est { val: if want_var { 0.0 } else { *x }, se: 0.0 }),
        Value::Est { val, .. } => Ok(Value::Est { val: if want_var { 0.0 } else { *val }, se: 0.0 }),
        // A bool is a Bernoulli point mass (LANG.md §1): `true` ≡ Bernoulli(1), `false` ≡
        // Bernoulli(0). So E(true) = 1, E(false) = 0, var = 0 — exactly as for a bool RV.
        Value::Bool(b) => {
            Ok(Value::Est { val: if want_var { 0.0 } else { f64::from(*b) }, se: 0.0 })
        }
        Value::Dist(id) => {
            // Validate-only mode: the quantity's graph is built and type-checked; skip sampling.
            if check {
                return Ok(Value::Est { val: 0.0, se: 0.0 });
            }
            // A finite-discrete quantity has exact moments — a finite sum, no sampling.
            if let Some(ex) = crate::enumerate::try_enumerate(graph, *id, max_opts) {
                let val = if want_var { ex.variance() } else { ex.mean() };
                return Ok(Value::Est { val, se: 0.0 });
            }
            let n = clamp_to_op_budget(n, graph, *id, max_opts);
            let m = sampler::moments(graph, *id, n, P_DEFAULT_SEED);
            let nf = n as f64;
            if want_var {
                // Asymptotic SE of a variance estimate ≈ var·sqrt(2/n) (exact for Gaussian).
                Ok(Value::Est { val: m.variance, se: m.variance.abs() * (2.0 / nf).sqrt() })
            } else {
                // Standard error of the mean: sqrt(Var / n). A point mass (Var=0) is exact.
                Ok(Value::Est { val: m.mean, se: (m.variance / nf).sqrt() })
            }
        }
        // Complex `E`/`Var` are deferred (PLAN-COMPLEX §5): query the channels separately.
        Value::Complex { .. } => Err(NoiseError::runtime(
            format!("{name} of a complex value is not supported yet — query the channels separately, e.g. `{name}(math::re(z))` and `{name}(math::im(z))`"),
            span,
        )),
        other => Err(NoiseError::runtime(
            format!("{name} expects a number or numeric random variable, got {}", other.type_name()),
            span,
        )),
    }
}

/// `Q(x, q)` / `Q(x, q, n)` — the **quantile** (inverse CDF) of a *numeric* quantity at level
/// `q ∈ [0, 1]`: `Q(X, 0.5)` is the median, `Q(X, 0.95)` the 95th percentile, `Q(X, 0)`/`Q(X, 1)`
/// the min/max draw. The companion to `E`/`Var` for tail/spread questions. Estimated by Monte
/// Carlo: draw `n` samples (default `1e6`, fixed seed), sort, and linearly interpolate between the
/// two bracketing order statistics. A deterministic value is its own quantile at every level.
///
/// Returns a plain `Num` (not an `Est`): a sample quantile's standard error depends on the density
/// at that point, so we don't claim auto-rounded precision the way `P`/`E` do.
fn quantile(arg_vals: &[Value], graph: &RvGraph, default_n: usize, max_opts: u64, span: Span, check: bool) -> Result<Value> {
    if arg_vals.len() < 2 || arg_vals.len() > 3 {
        return Err(NoiseError::runtime(
            format!(
                "Q expects 2 or 3 arguments (quantity, q in [0,1], optional sample count), got {}",
                arg_vals.len()
            ),
            span,
        ));
    }
    let q = as_num(&arg_vals[1], span)?;
    if !(0.0..=1.0).contains(&q) {
        return Err(NoiseError::runtime(
            format!("Q needs a quantile level q in [0, 1], got {q}"),
            span,
        ));
    }
    let n = match arg_vals.get(2) {
        Some(v) => {
            let n = as_num(v, span)?;
            if n < 1.0 || !n.is_finite() {
                return Err(NoiseError::runtime(
                    format!("Q sample count must be a finite number >= 1, got {n}"),
                    span,
                ));
            }
            n as usize
        }
        None => default_n,
    };
    match &arg_vals[0] {
        // A deterministic value is a point mass: its quantile is itself at every level.
        Value::Num(x) => Ok(Value::Num(*x)),
        Value::Est { val, .. } => Ok(Value::Num(*val)),
        Value::Bool(b) => Ok(Value::Num(f64::from(*b))),
        Value::Dist(id) => {
            // Validate-only mode: the quantity's graph is built and type-checked; skip sampling.
            if check {
                return Ok(Value::Num(0.0));
            }
            // A finite-discrete quantity has an exact inverse CDF — no sampling, no interpolation.
            if let Some(ex) = crate::enumerate::try_enumerate(graph, *id, max_opts) {
                return Ok(Value::Num(ex.quantile(q)));
            }
            let n = clamp_to_op_budget(n, graph, *id, max_opts);
            let mut draws = sampler::sample_n(graph, *id, n, P_DEFAULT_SEED);
            draws.sort_by(f64::total_cmp);
            Ok(Value::Num(empirical_quantile(&draws, q)))
        }
        other => Err(NoiseError::runtime(
            format!("Q expects a number or numeric random variable, got {}", other.type_name()),
            span,
        )),
    }
}

// --- conditional queries: `P(A | C)`, `E(X | C)`, `Var(X | C)`, `Q(X | C, q)` (Bayes, scoped to
//     one query — no side effect). `eval.rs` builds the single conditioning root
//     `select(C, quantity, NaN)` so event and condition are sampled *jointly* in one pass; these
//     reduce over the non-NaN (in-condition) lanes via `sampler::cond_*`. The standard error uses
//     the in-condition sample size `m ≈ n·P(C)`, not `n` — a rarer condition gives a looser estimate.

/// The condition after `|` never held in any of the `n` draws, so `P(·|C)` / `E(·|C)` is undefined
/// (a 0/0). Point the user at the two fixes rather than returning a silent NaN.
fn cond_never(n: usize, span: Span) -> NoiseError {
    NoiseError::runtime(
        format!(
            "the condition after `|` never occurred in {n} samples, so the conditional is undefined \
             — use a more likely condition or raise the sample count"
        ),
        span,
    )
}

/// Exact enumeration proved the condition after `|` has probability 0 — not "rare", impossible.
/// More samples can't fix this one, so the message doesn't suggest them.
fn cond_impossible(span: Span) -> NoiseError {
    NoiseError::runtime(
        "the condition after `|` has probability 0 (it can never hold), so the conditional is \
         undefined"
            .to_string(),
        span,
    )
}

/// Resolve an optional trailing sample-count `Value` for a conditional query (mirrors the validation
/// in `prob`/`moment`/`quantile`). `None` → `default_n`.
fn opt_count(name: &str, v: Option<&Value>, default_n: usize, span: Span) -> Result<usize> {
    match v {
        Some(v) => {
            let n = as_num(v, span)?;
            if n < 1.0 || !n.is_finite() {
                return Err(NoiseError::runtime(
                    format!("{name} sample count must be a finite number >= 1, got {n}"),
                    span,
                ));
            }
            Ok(n as usize)
        }
        None => Ok(default_n),
    }
}

/// `P(event | cond)` — the conditional probability over the worlds where `cond` holds. `root` is the
/// conditioning root `select(cond, event_indicator, NaN)` (built in `eval.rs`); `tail` is the
/// optional `[n]` sample count after the `|`-expression. The estimate is a probability with the
/// standard error of `m ≈ n·P(cond)` in-condition draws (so it self-rounds like an unconditional `P`).
pub fn prob_cond(
    graph: &RvGraph,
    root: RvId,
    tail: &[Value],
    default_n: usize,
    max_opts: u64,
    span: Span,
    check: bool,
) -> Result<Value> {
    if tail.len() > 1 {
        return Err(NoiseError::runtime(
            format!("P(event | cond) takes an optional sample count, got {} extra argument(s)", tail.len()),
            span,
        ));
    }
    if check {
        return Ok(Value::Est { val: 0.5, se: 0.0 });
    }
    let n = opt_count("P", tail.first(), default_n, span)?;
    // A finite-discrete conditional is exact Bayes: renormalize the non-NaN (in-condition) mass.
    if let Some(ex) = crate::enumerate::try_enumerate(graph, root, max_opts) {
        let given = ex.condition().ok_or_else(|| cond_impossible(span))?;
        return Ok(Value::Est { val: given.mean(), se: 0.0 });
    }
    let n = clamp_to_op_budget(n, graph, root, max_opts);
    let (m, count) = sampler::cond_moments(graph, root, n, P_DEFAULT_SEED);
    if count == 0 {
        return Err(cond_never(n, span));
    }
    let p_hat = m.mean;
    let cf = count as f64;
    let se = (p_hat * (1.0 - p_hat) / cf).sqrt().max(0.5 / cf);
    Ok(Value::Est { val: p_hat, se })
}

/// `E(x | cond)` / `Var(x | cond)` — the conditional mean/variance over the worlds where `cond`
/// holds. `root` is `select(cond, x, NaN)`; `tail` is the optional `[n]`. The standard error uses the
/// in-condition sample size.
pub fn moment_cond(
    name: &str,
    graph: &RvGraph,
    root: RvId,
    tail: &[Value],
    default_n: usize,
    max_opts: u64,
    span: Span,
    check: bool,
) -> Result<Value> {
    if tail.len() > 1 {
        return Err(NoiseError::runtime(
            format!("{name}(x | cond) takes an optional sample count, got {} extra argument(s)", tail.len()),
            span,
        ));
    }
    if check {
        return Ok(Value::Est { val: 0.0, se: 0.0 });
    }
    let n = opt_count(name, tail.first(), default_n, span)?;
    // A finite-discrete conditional has exact moments over the renormalized in-condition mass.
    if let Some(ex) = crate::enumerate::try_enumerate(graph, root, max_opts) {
        let given = ex.condition().ok_or_else(|| cond_impossible(span))?;
        let val = if name == "Var" { given.variance() } else { given.mean() };
        return Ok(Value::Est { val, se: 0.0 });
    }
    let n = clamp_to_op_budget(n, graph, root, max_opts);
    let (m, count) = sampler::cond_moments(graph, root, n, P_DEFAULT_SEED);
    if count == 0 {
        return Err(cond_never(n, span));
    }
    let cf = count as f64;
    if name == "Var" {
        Ok(Value::Est { val: m.variance, se: m.variance.abs() * (2.0 / cf).sqrt() })
    } else {
        Ok(Value::Est { val: m.mean, se: (m.variance / cf).sqrt() })
    }
}

/// `Q(x | cond, q)` / `Q(x | cond, q, n)` — the conditional quantile over the worlds where `cond`
/// holds. `root` is `select(cond, x, NaN)`; `tail` is `[q]` or `[q, n]`. Estimated by collecting the
/// in-condition draws (NaN lanes dropped), sorting, and interpolating — like the unconditional `Q`.
pub fn quantile_cond(
    graph: &RvGraph,
    root: RvId,
    tail: &[Value],
    default_n: usize,
    max_opts: u64,
    span: Span,
    check: bool,
) -> Result<Value> {
    if tail.is_empty() || tail.len() > 2 {
        return Err(NoiseError::runtime(
            format!(
                "Q(x | cond, q[, n]) needs a quantile level q in [0,1] (and an optional sample count), got {} argument(s) after the condition",
                tail.len()
            ),
            span,
        ));
    }
    let q = as_num(&tail[0], span)?;
    if !(0.0..=1.0).contains(&q) {
        return Err(NoiseError::runtime(
            format!("Q needs a quantile level q in [0, 1], got {q}"),
            span,
        ));
    }
    if check {
        return Ok(Value::Num(0.0));
    }
    let n = opt_count("Q", tail.get(1), default_n, span)?;
    // A finite-discrete conditional has an exact inverse CDF over the in-condition mass.
    if let Some(ex) = crate::enumerate::try_enumerate(graph, root, max_opts) {
        let given = ex.condition().ok_or_else(|| cond_impossible(span))?;
        return Ok(Value::Num(given.quantile(q)));
    }
    let n = clamp_to_op_budget(n, graph, root, max_opts);
    let mut draws = sampler::cond_sample_n(graph, root, n, P_DEFAULT_SEED);
    if draws.is_empty() {
        return Err(cond_never(n, span));
    }
    draws.sort_by(f64::total_cmp);
    Ok(Value::Num(empirical_quantile(&draws, q)))
}

/// Linear-interpolated empirical quantile of a **sorted, non-empty** sample (the `type-7` rule,
/// numpy's default): position `q*(len-1)`, blended between its floor/ceil order statistics.
fn empirical_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// `iota(n)` — `[0, 1, …, n-1]` (the same array as `0..n`; a handy alias).
fn iota(name: &str, args: &[Value], span: Span) -> Result<Value> {
    let n = count_arg(name, args, span)?;
    Ok(Value::Array(Rc::new((0..n).map(|i| Value::Num(i as f64)).collect())))
}

/// `ones(n)` / `zeros(n)` — an `n`-vector filled with `fill`.
fn filled(name: &str, args: &[Value], fill: f64, span: Span) -> Result<Value> {
    let n = count_arg(name, args, span)?;
    Ok(Value::Array(Rc::new(vec![Value::Num(fill); n])))
}

/// `sine(freq)` / `cosine(freq)` — a **lazy signal** of `freq` cycles over its (not yet chosen)
/// window. It costs O(1) until materialized — by meeting a sized array (adopting its length),
/// `signal::sample(sig, n)`, or a reducer rendering at the ambient resolution. The two-argument
/// form `sine(n, freq)` is the **eager** shorthand: it materializes `n` samples immediately
/// (handy for a quick concrete waveform).
fn wave(name: &str, args: &[Value], span: Span) -> Result<Value> {
    let w = if name == "cosine" { Wave::Cosine } else { Wave::Sine };
    match args {
        [freq] => Ok(Value::Signal(SigExpr::wave(w, as_num(freq, span)?))),
        [_n, freq] => {
            let n = count_arg(name, &args[..1], span)?;
            Ok(Value::Array(Rc::new(
                SigExpr::wave(w, as_num(freq, span)?).sample_f64(n).into_iter().map(Value::Num).collect(),
            )))
        }
        _ => Err(NoiseError::runtime(
            format!("{name} expects (freq) for a lazy signal or (samples, freq) for an array, got {} args", args.len()),
            span,
        )),
    }
}

/// `transpose(M)` — swap rows and columns of a rectangular matrix (array of equal-length rows).
/// Pure: it only rearranges the existing element `Value`s (no graph nodes built).
fn transpose(name: &str, args: &[Value], span: Span) -> Result<Value> {
    if args.len() != 1 {
        return Err(NoiseError::runtime(
            format!("{name} expects 1 argument, got {}", args.len()),
            span,
        ));
    }
    let rows = as_array(name, &args[0], span)?;
    if rows.is_empty() {
        return Ok(Value::Array(Rc::new(Vec::new())));
    }
    let cols = as_array(name, &rows[0], span)?.len();
    // Every row must be an array of the same length (a rectangular matrix).
    let mut grid = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let r = as_array(name, row, span)?;
        if r.len() != cols {
            return Err(NoiseError::runtime(
                format!("{name} needs a rectangular matrix; rows have lengths {cols} and {}", r.len()),
                span,
            ));
        }
        grid.push(r.clone());
    }
    let out: Vec<Value> = (0..cols)
        .map(|j| Value::Array(Rc::new(grid.iter().map(|r| r[j].clone()).collect())))
        .collect();
    Ok(Value::Array(Rc::new(out)))
}

/// Resolve a single non-negative integer count argument (for `iota`/`ones`/`zeros`).
fn count_arg(name: &str, args: &[Value], span: Span) -> Result<usize> {
    let n = one_num(name, args, span)?;
    if n.fract() != 0.0 || n < 0.0 || !n.is_finite() {
        return Err(NoiseError::runtime(
            format!("{name} size must be a non-negative integer, got {n}"),
            span,
        ));
    }
    Ok(n as usize)
}

/// `Len(xs)` — the number of elements (a build-time constant).
fn len(name: &str, args: &[Value], span: Span) -> Result<Value> {
    if args.len() != 1 {
        return Err(NoiseError::runtime(
            format!("{name} expects 1 argument, got {}", args.len()),
            span,
        ));
    }
    let xs = as_array(name, &args[0], span)?;
    Ok(Value::Num(xs.len() as f64))
}

/// Borrow the elements of an `Array` argument, or a spanned type error.
fn as_array<'a>(name: &str, v: &'a Value, span: Span) -> Result<&'a Rc<Vec<Value>>> {
    match v {
        Value::Array(xs) => Ok(xs),
        other => Err(NoiseError::runtime(
            format!("{name} expects an array, got {}", other.type_name()),
            span,
        )),
    }
}

fn one_num(name: &str, args: &[Value], span: Span) -> Result<f64> {
    if args.len() != 1 {
        return Err(NoiseError::runtime(
            format!("{name} expects 1 argument, got {}", args.len()),
            span,
        ));
    }
    as_num(&args[0], span)
}

fn two_nums(name: &str, args: &[Value], span: Span) -> Result<[f64; 2]> {
    if args.len() != 2 {
        return Err(NoiseError::runtime(
            format!("{name} expects 2 arguments, got {}", args.len()),
            span,
        ));
    }
    Ok([as_num(&args[0], span)?, as_num(&args[1], span)?])
}

fn three_nums(name: &str, args: &[Value], span: Span) -> Result<[f64; 3]> {
    if args.len() != 3 {
        return Err(NoiseError::runtime(
            format!("{name} expects 3 arguments, got {}", args.len()),
            span,
        ));
    }
    Ok([as_num(&args[0], span)?, as_num(&args[1], span)?, as_num(&args[2], span)?])
}

/// Largest exactly-representable `f64` integer (`2^53`). The integer builtins (`gcd`/`modpow`)
/// reject anything beyond it, so every value round-trips through `i64` without precision loss.
const MAX_EXACT_INT: f64 = 9_007_199_254_740_992.0;

/// Coerce a deterministic number to an exact integer (`i64`) for `gcd`/`modpow`, or a spanned
/// error: it must be a finite whole number with magnitude `<= 2^53`.
fn as_int(name: &str, n: f64, span: Span) -> Result<i64> {
    if n.fract() != 0.0 || !n.is_finite() || n.abs() > MAX_EXACT_INT {
        return Err(NoiseError::runtime(
            format!("{name} needs integer arguments (whole, |x| <= 2^53), got {n}"),
            span,
        ));
    }
    Ok(n as i64)
}

/// `base^exp mod modulus` by square-and-multiply in `i128` (so intermediate products are exact).
/// `modulus > 0`; the result is normalized to `[0, modulus)`, so a negative `base` is handled.
fn modpow_int(base: i64, mut exp: u64, modulus: i64) -> i64 {
    let m = modulus as i128;
    let mut result: i128 = 1 % m; // 0 when modulus == 1
    let mut b = (base as i128 % m + m) % m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % m;
        }
        b = b * b % m;
        exp >>= 1;
    }
    result as i64
}

/// Extract a deterministic `Value::Num` or return a spanned runtime error. (Distribution
/// parameters must be deterministic in this phase; a `Dist` argument is an error.)
fn as_num(v: &Value, span: Span) -> Result<f64> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Est { val, .. } => Ok(*val),
        other => Err(NoiseError::runtime(
            format!("expected a number, got {}", other.type_name()),
            span,
        )),
    }
}

/// Resolve a distribution parameter that may be **deterministic or random** — the seam that lets
/// `bernoulli(p)`/`normal(mu, sigma)`/… accept an RV parameter (hierarchical models). A `Num`/`Est`
/// is a constant; a numeric `Dist` (`RvKind::Num`) is the parameter's per-lane draws. A bool RV, a
/// recipe, or any other value is a spanned error (a probability/rate must be a number).
fn dist_arg(name: &str, v: &Value, graph: &RvGraph, span: Span) -> Result<DistArg> {
    match v {
        Value::Num(n) => Ok(DistArg::Const(*n)),
        Value::Est { val, .. } => Ok(DistArg::Const(*val)),
        Value::Dist(id) if graph.kind(*id) == RvKind::Num => Ok(DistArg::Rv(*id)),
        Value::Dist(id) => Err(NoiseError::runtime(
            format!("{name} parameter must be a number or numeric random variable, got {}", graph.kind(*id).type_name()),
            span,
        )),
        Value::Recipe(_) => Err(NoiseError::runtime(
            format!("{name} parameter is an undrawn distribution — draw it with `~` first (e.g. `m ~ unif(0,1); {name}(…, m)`)"),
            span,
        )),
        other => Err(NoiseError::runtime(
            format!("{name} parameter must be a number or numeric random variable, got {}", other.type_name()),
            span,
        )),
    }
}

/// Arity-1 distribution parameter (constant or RV).
fn one_arg(name: &str, args: &[Value], graph: &RvGraph, span: Span) -> Result<DistArg> {
    if args.len() != 1 {
        return Err(NoiseError::runtime(
            format!("{name} expects 1 argument, got {}", args.len()),
            span,
        ));
    }
    dist_arg(name, &args[0], graph, span)
}

/// Arity-2 distribution parameters (each constant or RV).
fn two_args(name: &str, args: &[Value], graph: &RvGraph, span: Span) -> Result<[DistArg; 2]> {
    if args.len() != 2 {
        return Err(NoiseError::runtime(
            format!("{name} expects 2 arguments, got {}", args.len()),
            span,
        ));
    }
    Ok([dist_arg(name, &args[0], graph, span)?, dist_arg(name, &args[1], graph, span)?])
}

/// If both parameters are constants, return them as plain `f64`s (the fast path that keeps the
/// existing constant-parameter recipe + single source instruction); `None` if either is random.
fn both_const(a: DistArg, b: DistArg) -> Option<[f64; 2]> {
    match (a, b) {
        (DistArg::Const(a), DistArg::Const(b)) => Some([a, b]),
        _ => None,
    }
}
