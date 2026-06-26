//! Builtin dispatch for `Expr::Call`.
//!
//! Phase 2 shipped `unif`. Phase 3 (cheap correctness wins) adds the discrete uniform
//! `unif_int`, `bernoulli`, and the probability query `P`. Args are already evaluated to
//! `Value`s by the engine. Distribution constructors take deterministic numbers and return a
//! `Dist`; `P` forces sampling and returns a probability as a plain `Num` (so it composes in
//! arithmetic, e.g. `4 * P(C)`).

use std::rc::Rc;

use crate::dist::{Recipe, RvGraph, RvId, RvKind};
use crate::error::{NoiseError, Result, Span};
use crate::sampler;
use crate::signal::{SignalSpec, Wave};
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
) -> Result<Value> {
    match name {
        "unif" => {
            let [lo, hi] = two_nums(name, arg_vals, span)?;
            Ok(Value::Recipe(Recipe::Uniform { lo, hi }))
        }
        "unif_int" => {
            let [lo, hi] = two_nums(name, arg_vals, span)?;
            if lo > hi {
                return Err(NoiseError::runtime(
                    format!("unif_int needs lo <= hi, got ({lo}, {hi})"),
                    span,
                ));
            }
            // Treat bounds as inclusive integers (normalize once, at recipe construction).
            Ok(Value::Recipe(Recipe::UniformInt { lo: lo.round(), hi: hi.round() }))
        }
        "bernoulli" => {
            let p = one_num(name, arg_vals, span)?;
            if !(0.0..=1.0).contains(&p) {
                return Err(NoiseError::runtime(
                    format!("bernoulli(p) needs 0 <= p <= 1, got {p}"),
                    span,
                ));
            }
            Ok(Value::Recipe(Recipe::Bernoulli { p }))
        }
        "normal" | "normal_int" => {
            let [mu, sigma] = two_nums(name, arg_vals, span)?;
            if sigma < 0.0 || !sigma.is_finite() || !mu.is_finite() {
                return Err(NoiseError::runtime(
                    format!("{name}(mu, sigma) needs finite mu and sigma >= 0, got ({mu}, {sigma})"),
                    span,
                ));
            }
            Ok(Value::Recipe(if name == "normal_int" {
                Recipe::NormalInt { mu, sigma }
            } else {
                Recipe::Normal { mu, sigma }
            }))
        }
        "exp" | "exp_int" => {
            let rate = one_num(name, arg_vals, span)?;
            if rate <= 0.0 || !rate.is_finite() {
                return Err(NoiseError::runtime(
                    format!("{name}(rate) needs a finite rate > 0, got {rate}"),
                    span,
                ));
            }
            Ok(Value::Recipe(if name == "exp_int" {
                Recipe::ExpInt { rate }
            } else {
                Recipe::Exp { rate }
            }))
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
        "E" | "Var" => moment(name, arg_vals, graph, default_n, max_opts, span),
        "round" => {
            // round(x, digits) — round x to `digits` decimal places. Handy for messages.
            let [x, digits] = two_nums(name, arg_vals, span)?;
            let factor = 10f64.powi(digits as i32);
            Ok(Value::Num((x * factor).round() / factor))
        }
        "P" => prob(arg_vals, graph, default_n, max_opts, span),
        "Q" => quantile(arg_vals, graph, default_n, max_opts, span),
        // --- pure collection builtins (no graph access; PLAN-COLLECTIONS §3) ---
        "Len" => len(name, arg_vals, span),
        // pure vector constructors / rearrangers (graph-free; just move `Value`s around)
        "iota" => iota(name, arg_vals, span),
        "ones" => filled(name, arg_vals, 1.0, span),
        "zeros" => filled(name, arg_vals, 0.0, span),
        "transpose" => transpose(name, arg_vals, span),
        // signal generators (lazy) + materialization
        "sine" | "cosine" => wave(name, arg_vals, span),
        "sample" => sample(name, arg_vals, span),
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

fn prob(arg_vals: &[Value], graph: &RvGraph, default_n: usize, max_opts: u64, span: Span) -> Result<Value> {
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
fn moment(name: &str, arg_vals: &[Value], graph: &RvGraph, default_n: usize, max_opts: u64, span: Span) -> Result<Value> {
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
fn quantile(arg_vals: &[Value], graph: &RvGraph, default_n: usize, max_opts: u64, span: Span) -> Result<Value> {
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

/// `sine(freq)` / `cosine(freq)` — a **lazy signal generator** of `freq` cycles over its (not yet
/// chosen) window. It costs O(1) until materialized — by meeting a sized array (adopting its
/// length) or `sample(sig, n)`. The two-argument form `sine(n, freq)` is the **eager** shorthand:
/// it materializes `n` samples immediately (handy for a quick concrete waveform).
fn wave(name: &str, args: &[Value], span: Span) -> Result<Value> {
    let w = if name == "cosine" { Wave::Cosine } else { Wave::Sine };
    match args {
        [freq] => Ok(Value::Signal(Rc::new(SignalSpec::base(w, as_num(freq, span)?)))),
        [_n, freq] => {
            let n = count_arg(name, &args[..1], span)?;
            Ok(Value::Array(Rc::new(
                SignalSpec::base(w, as_num(freq, span)?).sample(n).into_iter().map(Value::Num).collect(),
            )))
        }
        _ => Err(NoiseError::runtime(
            format!("{name} expects (freq) for a lazy signal or (samples, freq) for an array, got {} args", args.len()),
            span,
        )),
    }
}

/// `sample(sig, n)` — materialize a lazy signal to a concrete `n`-sample array. This is the knob
/// the Nyquist–Shannon theorem turns: sample a `freq`-cycle signal below vs. above `2*freq` points.
fn sample(name: &str, args: &[Value], span: Span) -> Result<Value> {
    if args.len() != 2 {
        return Err(NoiseError::runtime(
            format!("{name} expects 2 arguments (signal, samples), got {}", args.len()),
            span,
        ));
    }
    let n = count_arg(name, &args[1..], span)?;
    match &args[0] {
        Value::Signal(s) => {
            Ok(Value::Array(Rc::new(s.sample(n).into_iter().map(Value::Num).collect())))
        }
        other => Err(NoiseError::runtime(
            format!("{name} expects a signal, got {}", other.type_name()),
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
