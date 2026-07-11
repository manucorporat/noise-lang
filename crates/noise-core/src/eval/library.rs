//! The builtin library dispatched from `lib_call`: reducers, scans, linear algebra, ufuncs, the bootstrap / rotation / permutation draws, quantization, and matmul.
//!
//! Extracted verbatim from the monolithic `eval.rs` (finding F1); an `impl Engine` block
//! whose methods reach the rest of the evaluator through `self` and the shared free
//! helpers/tables that stay in the module root.

use std::rc::Rc;

use super::*;
use crate::builtins;
use crate::dist::{DataId, Recipe, RvId, RvKind, RvNode, Source, Uniform};
use crate::error::{NoiseError, Result, Span};
use crate::signal::{NoiseKind, NoiseSpec, SigExpr, SigUnOp};
use crate::value::Value;

impl Engine {
    /// `engine::set_max_samples(N)` — set the default Monte Carlo budget (the sample count `P`/`E`/
    /// `Var`/`Q` use when called without an explicit count) for the rest of the run. Lets a program
    /// trade accuracy for speed once, up front, instead of threading `n` through every query; an
    /// explicit per-call count still overrides it. Returns unit (it's a setting, not a value).
    fn lib_set_max_samples(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [n] = arity1("set_max_samples", args, span)?;
        let n = self.count_arg("set_max_samples", n, span)?;
        if n < 1 {
            return Err(NoiseError::runtime(
                "set_max_samples(N) needs N >= 1 (the Monte Carlo budget must draw at least once)"
                    .to_string(),
                span,
            ));
        }
        self.max_samples = n;
        Ok(Value::Unit)
    }

    /// `engine::set_max_ops(N)` — cap the *operations* each `P`/`E`/`Var`/`Q` query may spend for
    /// the rest of the run. A forcing over a cone of `C` distinct nodes costs `n×C` per-lane ops, so
    /// the query auto-clamps its draw count to `N/C` (never below 1): a heavy cone simply draws
    /// fewer samples (looser estimate) instead of doing unbounded work. This bounds each query's
    /// complexity *deterministically*, independent of the model's size — a budget, not an error.
    /// Pairs with `set_max_samples`, which caps draws; the query uses the smaller of the two.
    /// Returns unit (it's a setting, not a value).
    ///
    /// `name` is the invoked spelling: `set_max_ops` (correct — it caps *operations*, matching
    /// `MAX_OPS_DEFAULT`) or the retained back-compat alias `set_max_opts` (finding F9), so error
    /// messages name whichever the program actually wrote.
    fn lib_set_max_ops(&mut self, name: &str, args: &[Value], span: Span) -> Result<Value> {
        let [n] = arity1(name, args, span)?;
        let n = self.count_arg(name, n, span)?;
        if n < 1 {
            return Err(NoiseError::runtime(
                format!("{name}(N) needs N >= 1 (the operation budget must allow at least one op)"),
                span,
            ));
        }
        self.max_opts = n as u64;
        Ok(Value::Unit)
    }

    /// Dispatch a library call (collections / linear algebra). Returns `None` if `name` is not a
    /// library function, so the caller falls through to the pure builtins. These live here (not in
    /// `builtins.rs`) because they build graph nodes and/or draw — they need `&mut self` (§0).
    pub(super) fn lib_call(
        &mut self,
        name: &str,
        args: &[Value],
        span: Span,
    ) -> Option<Result<Value>> {
        let r = match name {
            "set_max_samples" => self.lib_set_max_samples(args, span),
            // `set_max_ops` is the correct name (it caps operations); `set_max_opts` is a retained
            // back-compat alias (finding F9). Both set the same knob.
            "set_max_ops" | "set_max_opts" => self.lib_set_max_ops(name, args, span),
            "set_resolution" => self.lib_set_resolution(args, span),
            // materializing a signal may draw noise RV nodes / read the realization cache, so
            // `sample` needs `&mut self` and can't live in the pure `builtins::call`.
            "sample" => self.lib_sample(args, span),
            "quantize" => self.lib_quantize(args, span),
            "sum" => self.lib_sum(args, span),
            "prod" => self.lib_prod(args, span),
            // prefix scans (PLAN-FINANCE F3): array in → same-length array of running reductions.
            "cumsum" | "cumprod" => self.lib_cum_fold(name, args, span),
            "cummax" | "cummin" => self.lib_cum_extreme(name, args, span),
            "count" => self.lib_count(args, span),
            "any" => self.lib_any(args, span),
            "all" => self.lib_all(args, span),
            "max" => self.lib_extreme(name, args, span),
            "min" => self.lib_extreme(name, args, span),
            "mean" => self.lib_mean(args, span),
            "dot" => self.lib_dot(args, span),
            "vdot" => self.lib_vdot(args, span),
            "normsq" => self.lib_normsq(args, span),
            "norm" => self.lib_norm(args, span),
            "normalize" => self.lib_normalize(args, span),
            "adjoint" => self.lib_adjoint(args, span),
            "outer" => self.lib_outer(args, span),
            "has_duplicates" => self.lib_has_duplicates(args, span),
            "count_duplicates" => self.lib_count_duplicates(args, span),
            "mse" => self.lib_mse(args, span),
            "sin" => self.lib_ufunc(UnOp::Sin, args, span),
            "cos" => self.lib_ufunc(UnOp::Cos, args, span),
            "atan" => self.lib_ufunc(UnOp::Atan, args, span),
            "sign" => self.lib_ufunc(UnOp::Sign, args, span),
            // RV-aware logarithms (PLAN-FINANCE F1) — intercepted here (not `builtins.rs`)
            // because the Dist path builds `UnOp::Ln` graph nodes.
            "log" => self.lib_log("log", args, span),
            "log10" => self.lib_log("log10", args, span),
            // complex-aware math ufuncs (PLAN-COMPLEX §4) + the real rounding family (§8).
            "exp" | "abs" | "sqrt" | "arg" | "conj" | "re" | "im" | "floor" | "ceil" => {
                self.math_ufunc(name, args, span)
            }
            "categorical" => self.lib_categorical(args, span),
            "empirical" => self.lib_empirical(args, span),
            "block_bootstrap" => self.lib_block_bootstrap(args, span),
            "noise_white" => self.lib_noise(NoiseKind::White, args, span),
            "noise_white_complex" => self.lib_noise(NoiseKind::WhiteComplex, args, span),
            "noise_brown" => self.lib_noise(NoiseKind::Brown, args, span),
            "noise_pink" => self.lib_noise(NoiseKind::Pink, args, span),
            "noise_ou" => self.lib_noise_ou(args, span),
            _ => return None,
        };
        Some(r)
    }

    /// `signal::noise_white|white_complex|brown|pink(sigma)` — an **undrawn** zero-mean noise
    /// generator of a given spectral color (`Value::Noise`). A recipe, not a value: `~` draws one
    /// lazy realization, `~[n]` a length-pinned one; anything else is the undrawn-distribution
    /// error (PLAN-SIGNALS §1.1). Lives in `lib_call` (not `builtins.rs`) so it sits beside the
    /// noise materialization paths.
    fn lib_noise(&mut self, kind: NoiseKind, args: &[Value], span: Span) -> Result<Value> {
        let [s] = arity1("noise", args, span)?;
        let sigma = self.noise_sigma(s, span)?;
        Ok(Value::Noise(NoiseSpec { sigma, kind }))
    }

    /// `signal::noise_ou(sigma, tau)` — colored noise with correlation time `tau` samples (`tau > 0`;
    /// lag-1 autocorrelation `exp(-1/tau)`).
    fn lib_noise_ou(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [s, t] = arity2("noise_ou", args, span)?;
        let sigma = self.noise_sigma(s, span)?;
        // `tau` is a correlation *time*, not a strength — extract it with its own message rather
        // than reusing `noise_sigma`'s "noise strength must be a number" text (finding F9).
        let tau = match t {
            Value::Num(n) | Value::Est { val: n, .. } => *n,
            other => {
                return Err(NoiseError::runtime(
                    format!(
                        "noise_ou correlation time tau must be a number, got {}",
                        other.type_name()
                    ),
                    span,
                ))
            }
        };
        if tau <= 0.0 || !tau.is_finite() {
            return Err(NoiseError::runtime(
                format!("noise_ou(sigma, tau) needs a finite correlation time tau > 0, got {tau}"),
                span,
            ));
        }
        Ok(Value::Noise(NoiseSpec {
            sigma,
            kind: NoiseKind::Ou { tau },
        }))
    }

    /// Extract a finite `sigma >= 0` from a noise argument.
    fn noise_sigma(&self, v: &Value, span: Span) -> Result<f64> {
        let n = match v {
            Value::Num(n) | Value::Est { val: n, .. } => *n,
            other => {
                return Err(NoiseError::runtime(
                    format!("noise strength must be a number, got {}", other.type_name()),
                    span,
                ))
            }
        };
        if n < 0.0 || !n.is_finite() {
            return Err(NoiseError::runtime(
                format!("noise strength must be a finite number >= 0, got {n}"),
                span,
            ));
        }
        Ok(n)
    }

    /// `signal::sample(sig, n)` — the explicit resolution override: materialize a lazy signal to
    /// a concrete `n`-sample array. A **complex signal** samples each channel (through the
    /// realization cache, so the quadratures stay consistent) and zips into an array of complex.
    /// An **undrawn noise generator** is rejected — `~` is the only way in (PLAN-SIGNALS §1.1).
    fn lib_sample(&mut self, args: &[Value], span: Span) -> Result<Value> {
        if args.len() != 2 {
            return Err(NoiseError::runtime(
                format!(
                    "sample expects 2 arguments (signal, samples), got {}",
                    args.len()
                ),
                span,
            ));
        }
        let n = self.count_arg("sample", &args[1], span)?;
        match &args[0] {
            Value::Signal(s) => Ok(Value::Array(Rc::new(self.materialize_sig(
                &s.clone(),
                n,
                span,
            )?))),
            Value::Complex { re, im }
                if matches!(&**re, Value::Signal(_)) || matches!(&**im, Value::Signal(_)) =>
            {
                let res = self.sample_channel(re, n, span)?;
                let ims = self.sample_channel(im, n, span)?;
                Ok(Value::Array(Rc::new(
                    res.into_iter()
                        .zip(ims)
                        .map(|(a, b)| Value::complex(a, b))
                        .collect(),
                )))
            }
            noise @ Value::Noise(_) => {
                // The undrawn-distribution error — `sample` was the old sanctioned drawer.
                forbid_undrawn(noise, span)?;
                unreachable!("forbid_undrawn rejects every Noise")
            }
            other => Err(NoiseError::runtime(
                format!("sample expects a signal, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Materialize one channel of a complex signal: a lazy `Signal` renders at `n`; a real scalar
    /// broadcasts (a constant channel is `n` copies).
    fn sample_channel(&mut self, v: &Value, n: usize, span: Span) -> Result<Vec<Value>> {
        match v {
            Value::Signal(s) => self.materialize_sig(&s.clone(), n, span),
            Value::Num(_) | Value::Est { .. } | Value::Dist(_) => Ok(vec![v.clone(); n]),
            other => Err(NoiseError::runtime(
                format!(
                    "cannot sample a complex signal with a {} channel",
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// `engine::set_resolution(N)` — set the ambient **sampling resolution**: the length reducers
    /// (`mse`/`mean`/`sum`/…) use to render a lazy signal that never met an explicit length. The
    /// time-axis twin of `set_max_samples` (PLAN-SIGNALS §1.2): a measurement knob, set once, not
    /// threaded through the math. `signal::sample(sig, n)` remains the explicit override.
    fn lib_set_resolution(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [n] = arity1("set_resolution", args, span)?;
        let n = self.count_arg("set_resolution", n, span)?;
        if n < 1 {
            return Err(NoiseError::runtime(
                "set_resolution(N) needs N >= 1 (a signal renders to at least one sample)"
                    .to_string(),
                span,
            ));
        }
        self.resolution = n;
        Ok(Value::Unit)
    }

    /// Materialize a noise generator to `n` zero-mean `normal` RV nodes of the requested color.
    /// White is iid; the colored kinds build a recurrence over fresh white draws (so they stay
    /// inside the RV graph — no FFT needed): Brown is a cumulative sum, OU an AR(1), Pink a sum of
    /// octave-spaced OU processes.
    pub(super) fn materialize_noise(&mut self, spec: NoiseSpec, n: usize) -> Vec<Value> {
        let ids = match spec.kind {
            NoiseKind::White => (0..n).map(|_| self.normal_src(spec.sigma)).collect(),
            NoiseKind::Brown => self.brown_ids(spec.sigma, n),
            NoiseKind::Ou { tau } => self.ou_ids(spec.sigma, tau, n),
            NoiseKind::Pink => self.pink_ids(spec.sigma, n),
            // The complex generator splits into two real White lanes at draw time
            // (`draw_noise`/`draw_noise_shaped`), so it never reaches a single-lane render.
            NoiseKind::WhiteComplex => {
                unreachable!("white_complex splits into two real lanes at draw")
            }
        };
        ids.into_iter().map(Value::Dist).collect()
    }

    /// A fresh `normal(0, sigma)` source node.
    fn normal_src(&mut self, sigma: f64) -> RvId {
        self.graph
            .push(RvNode::Src(Source::Normal { mu: 0.0, sigma }), RvKind::Num)
    }

    /// Brownian / red noise: `x_k = x_{k-1} + ε_k` with `ε ~ normal(0, sigma)` — a random walk
    /// (cumulative sum of white). Non-stationary; its variance grows with `k`.
    fn brown_ids(&mut self, sigma: f64, n: usize) -> Vec<RvId> {
        let mut out = Vec::with_capacity(n);
        let mut prev = self.normal_src(sigma);
        if n > 0 {
            out.push(prev);
        }
        for _ in 1..n {
            let step = self.normal_src(sigma);
            prev = self
                .graph
                .push(RvNode::Binary(BinOp::Add, prev, step), RvKind::Num);
            out.push(prev);
        }
        out
    }

    /// Ornstein–Uhlenbeck / AR(1) colored noise: `x_k = φ·x_{k-1} + innov_k`, `φ = exp(-1/tau)`,
    /// `innov ~ normal(0, sigma·√(1-φ²))`. Stationary with variance `sigma²` and lag-1
    /// autocorrelation `φ`; `tau → 0` ⇒ white, larger `tau` ⇒ longer memory.
    fn ou_ids(&mut self, sigma: f64, tau: f64, n: usize) -> Vec<RvId> {
        let phi = (-1.0 / tau).exp();
        let innov = sigma * (1.0 - phi * phi).sqrt();
        let phi_c = self.graph.push(RvNode::ConstNum(phi), RvKind::Num);
        let mut out = Vec::with_capacity(n);
        let mut prev = self.normal_src(sigma); // stationary marginal
        if n > 0 {
            out.push(prev);
        }
        for _ in 1..n {
            let eps = self.normal_src(innov);
            let scaled = self
                .graph
                .push(RvNode::Binary(BinOp::Mul, phi_c, prev), RvKind::Num);
            prev = self
                .graph
                .push(RvNode::Binary(BinOp::Add, scaled, eps), RvKind::Num);
            out.push(prev);
        }
        out
    }

    /// Pink (`~1/f`) noise as a sum of octave-spaced OU processes (`tau = 1, 2, 4, …`), each with
    /// equal variance — geometrically-spaced Lorentzians tile to a `1/f` envelope (a clean,
    /// in-graph alternative to FFT spectral synthesis). The per-octave strength `sigma/√M` keeps
    /// the total marginal variance `≈ sigma²`.
    fn pink_ids(&mut self, sigma: f64, n: usize) -> Vec<RvId> {
        if n == 0 {
            return Vec::new();
        }
        // Octaves spanning timescales from 1 up to ~n (capped so node count stays bounded).
        let octaves = (usize::BITS - n.leading_zeros()).clamp(1, 16) as usize;
        let sigma_oct = sigma / (octaves as f64).sqrt();
        let mut acc = self.ou_ids(sigma_oct, 1.0, n);
        for i in 1..octaves {
            let tau = (1u64 << i) as f64;
            let oct = self.ou_ids(sigma_oct, tau, n);
            for k in 0..n {
                acc[k] = self
                    .graph
                    .push(RvNode::Binary(BinOp::Add, acc[k], oct[k]), RvKind::Num);
            }
        }
        acc
    }

    /// Draw a `rotation(d)` recipe: a fresh `d`×`d` random **orthonormal** matrix per Monte Carlo
    /// sample (a Haar rotation: the random rotation `Π` of TurboQuant Algorithm 1, so `Π·x` is
    /// uniform on the unit sphere and each coordinate is `≈ N(0, 1/d)`). Built by drawing a Gaussian
    /// seed matrix and orthonormalizing its rows with (modified) Gram–Schmidt, lowered into the RV
    /// graph — it reuses `dot`/`-`/`normalize`, so every entry is an ordinary RV node sampled per
    /// lane. The cost is `O(d³)` graph nodes, so keep `d` modest (≤ ~32 for interactive runs). The
    /// inner reducers can't actually fail here (we control the shapes), so the span is synthetic.
    pub(super) fn draw_rotation(&mut self, d: usize) -> Result<Value> {
        let span = Span::default();
        // Gaussian seed: `d` rows of `d` iid N(0,1) draws (a fresh source node per entry).
        let mut seed = Vec::with_capacity(d);
        for _ in 0..d {
            let mut row = Vec::with_capacity(d);
            for _ in 0..d {
                row.push(self.draw(Recipe::Normal {
                    mu: 0.0,
                    sigma: 1.0,
                })?);
            }
            seed.push(Value::Array(Rc::new(row)));
        }
        // Modified Gram–Schmidt over the rows: subtract the projection onto each previously
        // orthonormalized row (which, being unit, has projection coefficient `dot(u, qⱼ)`), then
        // normalize. The resulting rows are orthonormal, hence the whole matrix is orthogonal.
        let mut q: Vec<Value> = Vec::with_capacity(d);
        for v in seed.into_iter() {
            let mut u = v;
            for qj in q.iter() {
                let coeff = self.lib_dot(&[u.clone(), qj.clone()], span)?;
                let proj = self.binop(BinOp::Mul, qj.clone(), coeff, span)?;
                u = self.binop(BinOp::Sub, u, proj, span)?;
            }
            q.push(self.lib_normalize(&[u], span)?);
        }
        Ok(Value::Array(Rc::new(q)))
    }

    /// Draw a `permutation(n)` recipe: a fresh uniform random permutation of `0..n` per Monte
    /// Carlo sample, returned as a length-`n` array. Built the same way `rotation` is — as
    /// arithmetic over shared iid sources, so each element is an ordinary RV node and the entries
    /// are jointly a permutation per lane. We draw `n` iid uniform keys and take their **argsort**:
    /// `deck[k] = rank(keyₖ) = #{ j : keyⱼ < keyₖ }`. With continuous keys there are no ties (prob
    /// 0), so the ranks are a permutation of `0..n`. Cost is `O(n²)` comparison nodes, so keep `n`
    /// modest. The inner ops can't fail here (we control the shapes), so the span is synthetic.
    pub(super) fn draw_permutation(&mut self, n: usize) -> Result<Value> {
        let span = Span::default();
        // `n` iid uniform keys (a fresh source node each).
        let mut keys = Vec::with_capacity(n);
        for _ in 0..n {
            keys.push(self.draw(Recipe::Uniform { lo: 0.0, hi: 1.0 })?);
        }
        // Each element is the rank of its key: count how many other keys are strictly smaller.
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut rank = Value::Num(0.0);
            for (j, key_j) in keys.iter().enumerate() {
                if j == k {
                    continue;
                }
                let lt = self.binop(BinOp::Lt, key_j.clone(), keys[k].clone(), span)?;
                let ind = self.indicator(lt, span)?;
                rank = self.binop(BinOp::Add, rank, ind, span)?;
            }
            out.push(rank);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Draw an `empirical(xs)` recipe (the iid bootstrap, PLAN-FINANCE F2): one fresh
    /// `unif_int(0, n-1)` index source, gathered over the constant data elements — exactly the
    /// manual idiom `i ~ unif_int(0, Len(xs)-1); xs[i]`. Each `~` (and each leaf of a shaped
    /// `~[n]` draw) builds a fresh index, so repeated draws resample independently. Gather is
    /// interpreter-only (`kernel::walk_cost`) — accepted for bootstrap workloads.
    pub(super) fn draw_empirical(&mut self, data: DataId) -> Value {
        let xs = self.datasets[data.0 as usize].clone();
        let hi = (xs.len() - 1) as f64; // non-empty by construction (`bootstrap_data`)
        let index = self
            .graph
            .push(RvNode::Src(Source::UniformInt { lo: 0.0, hi }), RvKind::Num);
        let elems = self.const_elems(&xs);
        Value::Dist(
            self.graph
                .push(RvNode::Gather { elems, index }, RvKind::Num),
        )
    }

    /// Draw a `block_bootstrap(xs, b)` recipe: a length-`n` series assembled from `⌈n/b⌉` random
    /// contiguous blocks of the data (the moving-block bootstrap). Element `j` is
    /// `xs[start_{⌊j/b⌋} + (j mod b)]` with each block start an independent `unif_int(0, n-b)` —
    /// non-wrapping, so every block is a real run from the history and within-block
    /// autocorrelation survives; the last block truncates when `b ∤ n`. Like `Permutation` this
    /// is a *structured* draw: it returns a whole `Value::Array` of RV elements. With `b == n`
    /// the only start is 0 (the series is the data); with `b == 1` it degenerates to `n` iid
    /// `empirical` draws.
    pub(super) fn draw_block_bootstrap(&mut self, data: DataId, b: usize) -> Value {
        let xs = self.datasets[data.0 as usize].clone();
        let n = xs.len();
        let elems = self.const_elems(&xs);
        // One independent start per block, uniform over the valid (non-wrapping) positions.
        let start_hi = (n - b) as f64; // 1 <= b <= n by construction (`lib_block_bootstrap`)
        let starts: Vec<RvId> = (0..n.div_ceil(b))
            .map(|_| {
                self.graph.push(
                    RvNode::Src(Source::UniformInt {
                        lo: 0.0,
                        hi: start_hi,
                    }),
                    RvKind::Num,
                )
            })
            .collect();
        // Offset constants are shared across blocks; offset 0 reuses the start node directly.
        let mut offsets: Vec<Option<RvId>> = vec![None; b];
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let (start, off) = (starts[j / b], j % b);
            let index = if off == 0 {
                start
            } else {
                let off_id = match offsets[off] {
                    Some(id) => id,
                    None => {
                        let id = self.graph.push(RvNode::ConstNum(off as f64), RvKind::Num);
                        offsets[off] = Some(id);
                        id
                    }
                };
                self.graph
                    .push(RvNode::Binary(BinOp::Add, start, off_id), RvKind::Num)
            };
            out.push(Value::Dist(self.graph.push(
                RvNode::Gather {
                    elems: elems.clone(),
                    index,
                },
                RvKind::Num,
            )));
        }
        Value::Array(Rc::new(out))
    }

    /// Lift a constant dataset to `ConstNum` element nodes (the gather targets of the bootstrap
    /// draws).
    fn const_elems(&mut self, xs: &[f64]) -> Box<[RvId]> {
        xs.iter()
            .map(|&x| self.graph.push(RvNode::ConstNum(x), RvKind::Num))
            .collect()
    }

    /// `quantize(v, centroids)` — snap each coordinate of `v` to its **nearest** value in `centroids`
    /// (the optimal scalar quantizer of TurboQuant Algorithm 1: a Voronoi/Lloyd–Max decision rule
    /// whose cell boundaries are the midpoints between consecutive sorted centroids). `centroids`
    /// must be constants. Each coordinate lowers to a chain of `select(v < midpoint, …)` nodes, so a
    /// random `v` stays a random variable. With a 1-element codebook this is a constant map.
    fn lib_quantize(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [v, c] = arity2("quantize", args, span)?;
        let xs = self.expect_array("quantize", v, span)?;
        let cs = self.expect_array("quantize", c, span)?;
        if cs.is_empty() {
            return Err(NoiseError::runtime(
                "quantize needs a non-empty codebook".to_string(),
                span,
            ));
        }
        let mut levels: Vec<f64> = Vec::with_capacity(cs.len());
        for e in cs.iter() {
            match scalar_const(e) {
                // Reject non-finite centroids like `lib_categorical` does: a NaN centroid has no
                // order, so `sort_by(partial_cmp().unwrap())` used to panic (finding A7); an inf
                // centroid would poison the midpoint thresholds.
                Some(n) if n.is_finite() => levels.push(n),
                Some(_) => {
                    return Err(NoiseError::runtime(
                        "quantize centroids must be finite numbers (no NaN/inf)".to_string(),
                        span,
                    ))
                }
                None => {
                    return Err(NoiseError::runtime(
                        "quantize centroids must be constant numbers, not random variables"
                            .to_string(),
                        span,
                    ))
                }
            }
        }
        // `total_cmp` is a total order on all `f64`, so the sort never panics even if a stray
        // non-finite slipped through; combined with the finite check above the result is clean.
        levels.sort_by(f64::total_cmp);
        let mut out = Vec::with_capacity(xs.len());
        for x in xs.iter() {
            out.push(self.nearest_centroid(x.clone(), &levels, span)?);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Snap a single value to the nearest of `levels` (sorted ascending) via a nested `select`
    /// over the midpoint thresholds. Outermost test wins, so `x < t₀ → levels[0]`, then
    /// `x < t₁ → levels[1]`, …, else the top level.
    fn nearest_centroid(&mut self, x: Value, levels: &[f64], span: Span) -> Result<Value> {
        let mut result = Value::Num(levels[levels.len() - 1]);
        for i in (0..levels.len() - 1).rev() {
            let t = 0.5 * (levels[i] + levels[i + 1]);
            let cond = self.binop(BinOp::Lt, x.clone(), Value::Num(t), span)?;
            result = self.select(cond, Value::Num(levels[i]), result, span)?;
        }
        Ok(result)
    }

    /// `sum(xs)` — fold `+` over the elements (`0` for an empty array).
    fn lib_sum(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("sum", args, span)?;
        let xs = self.expect_array("sum", xs, span)?;
        let mut acc = Value::Num(0.0);
        for x in xs.iter() {
            acc = self.binop(BinOp::Add, acc, x.clone(), span)?;
        }
        Ok(acc)
    }

    /// `prod(xs)` — fold `*` over the elements (`1` for an empty array, the multiplicative
    /// identity, mirroring `sum`'s `0`). Folds from `xs[0]` so no spurious `1*x` node is built
    /// and non-numeric element types behave exactly like the underlying `*`.
    fn lib_prod(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("prod", args, span)?;
        let xs = self.expect_array("prod", xs, span)?;
        let Some(first) = xs.first() else {
            return Ok(Value::Num(1.0));
        };
        let mut acc = first.clone();
        for x in xs.iter().skip(1) {
            acc = self.binop(BinOp::Mul, acc, x.clone(), span)?;
        }
        Ok(acc)
    }

    /// `cumsum(xs)` / `cumprod(xs)` — prefix scan (PLAN-FINANCE F3): `out[t]` is the sum/product
    /// of `xs[0..=t]`, so the output has the same length as the input. `out[0]` is `xs[0]` itself
    /// (the scan starts from the first element — no spurious `+0`/`*1` node, and every element
    /// type behaves exactly like the underlying `+`/`*`). The scan of an empty array is the empty
    /// array. Constants fold and RVs lift uniformly through `binop`, so
    /// `path = cumprod(1 + steps)` over shaped draws builds the same DAG the hand-written
    /// for-loop accumulator would.
    fn lib_cum_fold(&mut self, name: &str, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1(name, args, span)?;
        let xs = self.expect_array(name, xs, span)?;
        let op = if name == "cumsum" {
            BinOp::Add
        } else {
            BinOp::Mul
        };
        let mut out: Vec<Value> = Vec::with_capacity(xs.len());
        for x in xs.iter() {
            let acc = match out.last().cloned() {
                None => x.clone(),
                Some(prev) => self.binop(op, prev, x.clone(), span)?,
            };
            out.push(acc);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// `cummax(xs)` / `cummin(xs)` — running-extremum scan via the select (lifted `if`) path,
    /// mirroring `max`/`min` element-for-element: `out[t]` is the max/min of `xs[0..=t]`. Unlike
    /// the reducers (which error on empty — there is no extremum of nothing), the scan of an
    /// empty array is the empty array: every prefix an output element summarizes is nonempty.
    fn lib_cum_extreme(&mut self, name: &str, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1(name, args, span)?;
        let xs = self.expect_array(name, xs, span)?;
        let cmp = if name == "cummax" {
            BinOp::Gt
        } else {
            BinOp::Lt
        };
        let mut out: Vec<Value> = Vec::with_capacity(xs.len());
        for x in xs.iter() {
            let m = match out.last().cloned() {
                None => x.clone(),
                Some(prev) => {
                    let cond = self.binop(cmp, x.clone(), prev.clone(), span)?;
                    self.select(cond, x.clone(), prev, span)?
                }
            };
            out.push(m);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// `count(xs)` — number of true elements (sum of `0/1` indicators).
    fn lib_count(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("count", args, span)?;
        let xs = self.expect_array("count", xs, span)?;
        let mut acc = Value::Num(0.0);
        for x in xs.iter() {
            let ind = self.indicator(x.clone(), span)?;
            acc = self.binop(BinOp::Add, acc, ind, span)?;
        }
        Ok(acc)
    }

    /// `any(xs)` — `||` over the elements (`false` for empty).
    fn lib_any(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("any", args, span)?;
        let xs = self.expect_array("any", xs, span)?;
        let mut acc = Value::Bool(false);
        for x in xs.iter() {
            acc = self.binop(BinOp::Or, acc, x.clone(), span)?;
        }
        Ok(acc)
    }

    /// `all(xs)` — `&&` over the elements (`true` for empty).
    fn lib_all(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("all", args, span)?;
        let xs = self.expect_array("all", xs, span)?;
        let mut acc = Value::Bool(true);
        for x in xs.iter() {
            acc = self.binop(BinOp::And, acc, x.clone(), span)?;
        }
        Ok(acc)
    }

    /// `max(xs)` / `min(xs)` — running extremum via the select (lifted `if`) path.
    fn lib_extreme(&mut self, name: &str, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1(name, args, span)?;
        let xs = self.expect_array(name, xs, span)?;
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                format!("{name} of an empty array"),
                span,
            ));
        }
        let cmp = if name == "max" { BinOp::Gt } else { BinOp::Lt };
        let mut m = xs[0].clone();
        for x in xs.iter().skip(1) {
            let cond = self.binop(cmp, x.clone(), m.clone(), span)?;
            m = self.select(cond, x.clone(), m, span)?;
        }
        Ok(m)
    }

    /// `mean(xs)` — `sum(xs) / len(xs)`.
    fn lib_mean(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("mean", args, span)?;
        let arr = self.expect_array("mean", xs, span)?;
        if arr.is_empty() {
            return Err(NoiseError::runtime(
                "mean of an empty array".to_string(),
                span,
            ));
        }
        let s = self.lib_sum(args, span)?;
        self.binop(BinOp::Div, s, Value::Num(arr.len() as f64), span)
    }

    /// `dot(a, b)` — inner product (Add-chain of per-index products). Lengths must match.
    fn lib_dot(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [a, b] = arity2("dot", args, span)?;
        let a = self.expect_array("dot", a, span)?;
        let b = self.expect_array("dot", b, span)?;
        if a.len() != b.len() {
            return Err(length_mismatch("dot", a.len(), b.len(), span));
        }
        let mut acc = Value::Num(0.0);
        for (ai, bi) in a.iter().zip(b.iter()) {
            let prod = self.binop(BinOp::Mul, ai.clone(), bi.clone(), span)?;
            acc = self.binop(BinOp::Add, acc, prod, span)?;
        }
        Ok(acc)
    }

    /// `vdot(a, b)` — the **Hermitian** inner product `Σ conj(aᵢ)·bᵢ` (PLAN-COMPLEX §7, numpy's
    /// `vdot`). Conjugates the *first* argument, unlike the bilinear [`Self::lib_dot`] / `@`. For
    /// real vectors it coincides with `dot`; for complex vectors it is the physically-meaningful
    /// inner product (so `vdot(z, z) == normsq(z)`, a real).
    fn lib_vdot(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [a, b] = arity2("vdot", args, span)?;
        let a = self.expect_array("vdot", a, span)?;
        let b = self.expect_array("vdot", b, span)?;
        if a.len() != b.len() {
            return Err(length_mismatch("vdot", a.len(), b.len(), span));
        }
        let mut acc = Value::Num(0.0);
        for (ai, bi) in a.iter().zip(b.iter()) {
            let ca = self.math_ufunc("conj", std::slice::from_ref(ai), span)?;
            let prod = self.binop(BinOp::Mul, ca, bi.clone(), span)?;
            acc = self.binop(BinOp::Add, acc, prod, span)?;
        }
        Ok(acc)
    }

    /// `outer(a, b)` — the outer product `M[i][j] = aᵢ·bⱼ` (an `len(a)×len(b)` matrix). The general
    /// linear-algebra primitive behind building a QFT matrix (`outer(iota, iota)` through complex
    /// `exp`) and one-hot encodings. Lifts/folds elementwise like every product.
    fn lib_outer(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [a, b] = arity2("outer", args, span)?;
        let a = self.expect_array("outer", a, span)?;
        let b = self.expect_array("outer", b, span)?;
        let mut rows = Vec::with_capacity(a.len());
        for ai in a.iter() {
            let mut row = Vec::with_capacity(b.len());
            for bj in b.iter() {
                row.push(self.binop(BinOp::Mul, ai.clone(), bj.clone(), span)?);
            }
            rows.push(Value::Array(Rc::new(row)));
        }
        Ok(Value::Array(Rc::new(rows)))
    }

    /// `adjoint(M)` — the conjugate transpose `Mᴴ` (`conj` ∘ `transpose`, PLAN-COMPLEX §7): the
    /// quantum/linear-algebra "dagger". For a real matrix it is the plain transpose.
    fn lib_adjoint(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [m] = arity1("adjoint", args, span)?;
        let t = builtins::call("transpose", std::slice::from_ref(m), &self.query_ctx(span))?;
        self.math_ufunc("conj", &[t], span)
    }

    /// `categorical(weights)` — sample an index `0..len(weights)` with probability proportional to
    /// `weights` (PLAN-COMPLEX §9; the honest measurement op `y ~ categorical(|ψ|²)`). The weights
    /// must be constant, non-negative numbers summing to a positive total. Built by inverse-CDF: a
    /// single `unif(0, total)` draw `u`, then `index = #{k : prefix_k ≤ u}` (each prefix threshold a
    /// lifted indicator). Returns a `Dist<number>` directly — like a gather, *not* a recipe — so
    /// each call builds an independent draw (bind-and-redraw shares the one draw, as with `gather`).
    fn lib_categorical(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [w] = arity1("categorical", args, span)?;
        let xs = self.expect_array("categorical", w, span)?;
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                "categorical needs a non-empty weight vector".to_string(),
                span,
            ));
        }
        let mut weights = Vec::with_capacity(xs.len());
        for e in xs.iter() {
            match scalar_const(e) {
                Some(v) if v >= 0.0 && v.is_finite() => weights.push(v),
                _ => {
                    return Err(NoiseError::runtime(
                        "categorical weights must be constant, non-negative numbers".to_string(),
                        span,
                    ))
                }
            }
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(NoiseError::runtime(
                "categorical weights must sum to a positive value".to_string(),
                span,
            ));
        }
        let u = self.graph.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: total })),
            RvKind::Num,
        );
        let mut prefix = 0.0;
        let mut index = Value::Num(0.0);
        for &wk in &weights {
            prefix += wk;
            let cond = self.binop(BinOp::Le, Value::Num(prefix), Value::Dist(u), span)?;
            let ind = self.indicator(cond, span)?;
            index = self.binop(BinOp::Add, index, ind, span)?;
        }
        Ok(index)
    }

    /// `rand::empirical(xs)` — the **iid bootstrap** constructor (PLAN-FINANCE F2): a *recipe*
    /// whose every `~` draw is a uniformly-random element of the constant data array `xs`
    /// ("don't fit a distribution — resample history"). Sugar for
    /// `i ~ unif_int(0, Len(xs)-1); xs[i]`, but a true recipe — so `a ~ …; b ~ …` are
    /// independent resamples and `~[n]` yields `n` iid resamples per leaf (unlike
    /// `categorical`, which returns its one draw directly and shares it under `~[n]`).
    fn lib_empirical(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("empirical", args, span)?;
        let data = self.bootstrap_data("empirical", xs, span)?;
        Ok(Value::Recipe(Recipe::Empirical { data }))
    }

    /// `rand::block_bootstrap(xs, block_len)` — the **moving-block bootstrap** constructor
    /// (PLAN-FINANCE F2): a recipe whose `~` draw is a whole `Len(xs)`-long series glued from
    /// random contiguous blocks of `xs` (block starts iid `unif_int(0, n - block_len)`), so
    /// streaks/autocorrelation inside a block survive the resampling. `block_len` must be an
    /// integer with `1 <= block_len <= Len(xs)`.
    fn lib_block_bootstrap(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs, b] = arity2("block_bootstrap", args, span)?;
        let data = self.bootstrap_data("block_bootstrap", xs, span)?;
        let n = self.datasets[data.0 as usize].len();
        let bl = match scalar_const(b) {
            Some(x) => x,
            None => {
                return Err(NoiseError::runtime(
                    format!(
                        "block_bootstrap(xs, block_len) needs a constant number for the block \
                         length, got {}",
                        b.type_name()
                    ),
                    span,
                ))
            }
        };
        if bl.fract() != 0.0 || !bl.is_finite() || bl < 1.0 || bl > n as f64 {
            return Err(NoiseError::runtime(
                format!(
                    "block_bootstrap(xs, block_len) needs an integer block length with \
                     1 <= block_len <= Len(xs) = {n}, got {bl}"
                ),
                span,
            ));
        }
        Ok(Value::Recipe(Recipe::BlockBootstrap {
            data,
            block_len: bl as usize,
        }))
    }

    /// Validate and intern a bootstrap data array (`empirical` / `block_bootstrap`): a
    /// non-empty **flat** array of constant numbers (paste your data as a literal array).
    /// Returns the engine dataset handle the recipe carries, so the recipe stays `Copy`.
    fn bootstrap_data(&mut self, name: &str, v: &Value, span: Span) -> Result<DataId> {
        let xs = self.expect_array(name, v, span)?;
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                format!("{name} needs a non-empty data array — an empty history has nothing to resample"),
                span,
            ));
        }
        let mut data = Vec::with_capacity(xs.len());
        for e in xs.iter() {
            match scalar_const(e) {
                Some(x) => data.push(x),
                None => {
                    return Err(NoiseError::runtime(
                        format!(
                            "{name} resamples a flat array of constant numbers, but an element \
                             is {} — paste the data as plain numbers",
                            e.type_name()
                        ),
                        span,
                    ))
                }
            }
        }
        let id = DataId(self.datasets.len() as u32);
        self.datasets.push(Rc::new(data));
        Ok(id)
    }

    /// `normsq(a)` — `Σ |aᵢ|²`, the squared Euclidean norm. **Magnitude-based**, so it always
    /// returns a *real* (PLAN-COMPLEX §7): for a real vector this is `Σ aᵢ²` (unchanged); for a
    /// complex vector it is `Σ (reᵢ² + imᵢ²)` — exactly what measurement probabilities and signal
    /// error need. Nested arrays (matrices) sum over all leaves (Frobenius).
    fn lib_normsq(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [a] = arity1("normsq", args, span)?;
        let xs = self.expect_array("normsq", a, span)?;
        let mut acc = Value::Num(0.0);
        for x in xs.iter() {
            let mag = if matches!(x, Value::Array(_)) {
                self.lib_normsq(std::slice::from_ref(x), span)?
            } else {
                let (re, im) = self.complex_parts(x.clone(), span)?;
                let rr = self.binop(BinOp::Mul, re.clone(), re, span)?;
                let ii = self.binop(BinOp::Mul, im.clone(), im, span)?;
                self.binop(BinOp::Add, rr, ii, span)?
            };
            acc = self.binop(BinOp::Add, acc, mag, span)?;
        }
        Ok(acc)
    }

    /// `norm(a)` — Euclidean length, `normsq(a) ^ 0.5` (so it lifts over RVs, and folds to
    /// `sqrt` for constant vectors).
    fn lib_norm(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let ns = self.lib_normsq(args, span)?;
        self.binop(BinOp::Pow, ns, Value::Num(0.5), span)
    }

    /// `mat @ vec` — matrix-vector product (`out[i] = dot(M[i], v)`). Private helper for the `@`
    /// operator's `(mat, vec)` case; there is no standalone builtin (use `@`).
    fn lib_matvec(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [m, v] = arity2("@", args, span)?;
        let rows = self.expect_array("@", m, span)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            out.push(self.lib_dot(&[row.clone(), v.clone()], span)?);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Evaluate both operands of `@`, then dispatch by shape. Split out of the `eval` match so its
    /// locals don't enlarge the (deeply recursive) `eval` stack frame — see `eval_sample`.
    pub(super) fn eval_matmul(&mut self, l: &Spanned, r: &Spanned, span: Span) -> Result<Value> {
        let lv = self.eval(l)?;
        let rv = self.eval(r)?;
        self.matmul(lv, rv, span)
    }

    /// The matrix-product operator `@` (`Expr::MatMul`). Dispatches on operand rank (0 = scalar,
    /// 1 = vector, 2 = matrix), lowering to the existing dot / broadcast machinery so every result
    /// element is an ordinary value/RV node:
    ///   - `vec @ vec` → scalar dot product
    ///   - `mat @ vec` → matrix–vector product (`out[i] = dot(M[i], v)`)
    ///   - `vec @ mat` → row-vector × matrix (`out[j] = Σ_p v[p]·M[p][j]`)
    ///   - `mat @ mat` → matrix–matrix product (each result row is `A[i] @ B`)
    ///
    /// A scalar operand is an error — `@` is for linear algebra; use `*` for scaling.
    fn matmul(&mut self, l: Value, r: Value, span: Span) -> Result<Value> {
        match (value_rank(&l), value_rank(&r)) {
            (1, 1) => self.lib_dot(&[l, r], span),
            (2, 1) => self.lib_matvec(&[l, r], span),
            (1, 2) => {
                let rows = self.expect_array("@", &r, span)?;
                let weights = self.expect_array("@", &l, span)?;
                self.weighted_row_sum(&weights, &rows, span)
            }
            (2, 2) => {
                let arows = self.expect_array("@", &l, span)?;
                let brows = self.expect_array("@", &r, span)?;
                let mut out = Vec::with_capacity(arows.len());
                for a in arows.iter() {
                    let w = self.expect_array("@", a, span)?;
                    out.push(self.weighted_row_sum(&w, &brows, span)?);
                }
                Ok(Value::Array(Rc::new(out)))
            }
            _ => Err(NoiseError::runtime(
                format!(
                    "`@` (matrix product) needs vector/matrix operands, got {} @ {} — use `*` to scale",
                    l.type_name(),
                    r.type_name()
                ),
                span,
            )),
        }
    }

    /// `Σ_p weights[p] · rows[p]` — the weighted sum of the matrix `rows` by a vector of `weights`,
    /// returning a single row vector. The scalar·row product broadcasts and the row sum is
    /// elementwise, so this lifts over random variables. Backs `vec @ mat` and each row of
    /// `mat @ mat`. Requires the inner dimensions to match and be non-empty.
    fn weighted_row_sum(&mut self, weights: &[Value], rows: &[Value], span: Span) -> Result<Value> {
        if weights.len() != rows.len() {
            return Err(length_mismatch("@", weights.len(), rows.len(), span));
        }
        if weights.is_empty() {
            return Err(NoiseError::runtime(
                "`@` needs a non-empty inner dimension".to_string(),
                span,
            ));
        }
        let mut acc = self.binop(BinOp::Mul, weights[0].clone(), rows[0].clone(), span)?;
        for (w, row) in weights.iter().zip(rows.iter()).skip(1) {
            let term = self.binop(BinOp::Mul, w.clone(), row.clone(), span)?;
            acc = self.binop(BinOp::Add, acc, term, span)?;
        }
        Ok(acc)
    }

    /// `normalize(v)` — `v / norm(v)` (unit vector). The division broadcasts the scalar `1/norm`
    /// across the array, the same way `c * v` would.
    fn lib_normalize(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [v] = arity1("normalize", args, span)?;
        let norm = self.lib_norm(args, span)?;
        self.binop(BinOp::Div, v.clone(), norm, span)
    }

    /// A transcendental ufunc (`sin`/`cos`/`atan`) applied uniformly across value kinds: a scalar
    /// computes directly, a random variable lifts to a graph node (sampled per lane), and an array
    /// maps elementwise (so it works on whole signals). This is what lets `cos(phase)` build a
    /// waveform and `atan(rQ / rI)` demodulate a noisy one with the same function.
    fn lib_ufunc(&mut self, op: UnOp, args: &[Value], span: Span) -> Result<Value> {
        let [x] = arity1(unop_name(op), args, span)?;
        match x {
            Value::Num(n) => Ok(Value::Num(apply_unop_f64(op, *n))),
            Value::Est { val, .. } => Ok(Value::Num(apply_unop_f64(op, *val))),
            Value::Dist(_) => self.lift_unary(op, x.clone(), span),
            // A lazy signal defers the ufunc into its expression tree (stays a signal).
            Value::Signal(s) => Ok(Value::Signal(Rc::new(SigExpr::Unary(
                SigUnOp::Un(op),
                s.clone(),
            )))),
            Value::Array(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                for e in xs.iter() {
                    out.push(self.lib_ufunc(op, std::slice::from_ref(e), span)?);
                }
                Ok(Value::Array(Rc::new(out)))
            }
            other => Err(NoiseError::runtime(
                format!(
                    "{} expects a number, array, or random variable, got {}",
                    unop_name(op),
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// `math::log` (natural) / `math::log10` — RV-aware logarithms (PLAN-FINANCE F1). A
    /// deterministic scalar keeps the friendly build-time domain error (`x > 0`); a random
    /// variable lifts to a per-lane `UnOp::Ln` node with IEEE `f64::ln` semantics (a lane's value
    /// can't be inspected at build time, so `0 → -inf` and `negative → NaN` per lane); `log10`
    /// rides on the same node via `log10(x) = ln(x)/ln(10)`. Arrays map elementwise and a lazy
    /// signal defers the op into its tree, like every other ufunc.
    fn lib_log(&mut self, name: &'static str, args: &[Value], span: Span) -> Result<Value> {
        let [x] = arity1(name, args, span)?;
        let base10 = name == "log10";
        match x {
            Value::Num(n) | Value::Est { val: n, .. } => {
                let n = *n;
                if n <= 0.0 {
                    return Err(NoiseError::runtime(
                        format!("{name} needs x > 0, got {n}"),
                        span,
                    ));
                }
                Ok(Value::Num(if base10 { n.log10() } else { n.ln() }))
            }
            Value::Array(xs) => {
                let xs = xs.clone();
                let mut out = Vec::with_capacity(xs.len());
                for e in xs.iter() {
                    out.push(self.lib_log(name, std::slice::from_ref(e), span)?);
                }
                Ok(Value::Array(Rc::new(out)))
            }
            Value::Dist(_) => {
                let lnx = self.lift_unary(UnOp::Ln, x.clone(), span)?;
                if base10 {
                    self.binop(BinOp::Div, lnx, Value::Num(std::f64::consts::LN_10), span)
                } else {
                    Ok(lnx)
                }
            }
            Value::Signal(s) => {
                let lns = Value::Signal(Rc::new(SigExpr::Unary(SigUnOp::Un(UnOp::Ln), s.clone())));
                if base10 {
                    self.binop(BinOp::Div, lns, Value::Num(std::f64::consts::LN_10), span)
                } else {
                    Ok(lns)
                }
            }
            other => Err(NoiseError::runtime(
                format!(
                    "{name} expects a number, array, or random variable, got {}",
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// The complex-aware `math::` ufuncs (PLAN-COMPLEX §4): `exp`/`abs`/`sqrt`/`arg`/`conj`/`re`/`im`
    /// and the real-only rounding family `floor`/`ceil` (§8). Each branches by input type — real in →
    /// real semantics; complex in → complex semantics — and maps elementwise over arrays (so it works
    /// on a whole complex signal). Like every ufunc it folds for constants and lifts into the (real)
    /// sample-DAG when a channel is an RV.
    fn math_ufunc(&mut self, name: &str, args: &[Value], span: Span) -> Result<Value> {
        let [x] = arity1(name, args, span)?;
        // Map over arrays uniformly (a complex array is an array of `Complex`/real elements).
        if let Value::Array(xs) = x {
            let mut out = Vec::with_capacity(xs.len());
            for e in xs.iter() {
                out.push(self.math_ufunc(name, std::slice::from_ref(e), span)?);
            }
            return Ok(Value::Array(Rc::new(out)));
        }
        let x = x.clone();
        match name {
            "exp" => self.cufunc_exp(x, span),
            // |z| = √(re² + im²); for a real `x` this is √(x²) = |x| (and lifts/folds the same way).
            "abs" => {
                let (a, b) = self.complex_parts(x, span)?;
                let aa = self.binop(BinOp::Mul, a.clone(), a, span)?;
                let bb = self.binop(BinOp::Mul, b.clone(), b, span)?;
                let s = self.binop(BinOp::Add, aa, bb, span)?;
                self.binop(BinOp::Pow, s, Value::Num(0.5), span)
            }
            "sqrt" => self.cufunc_sqrt(x, span),
            "arg" => self.cufunc_arg(x, span),
            "conj" => match x {
                Value::Complex { re, im } => {
                    let neg_im = self.binop(BinOp::Sub, Value::Num(0.0), *im, span)?;
                    Ok(Value::complex(*re, neg_im))
                }
                real => {
                    // `conj` of a real is itself — but reject non-numerics with a clear message.
                    let _ = self.complex_parts(real.clone(), span)?;
                    Ok(real)
                }
            },
            "re" => {
                let (a, _) = self.complex_parts(x, span)?;
                Ok(a)
            }
            "im" => {
                let (_, b) = self.complex_parts(x, span)?;
                Ok(b)
            }
            "floor" => self.cufunc_round_fam(UnOp::Floor, x, span),
            "ceil" => self.cufunc_round_fam(UnOp::Ceil, x, span),
            _ => unreachable!("math_ufunc dispatched on an unknown name '{name}'"),
        }
    }

    /// `exp(z)` — `e^z`. Real `x` → `e^x` (a lazy signal **defers** `exp` into its tree — legal
    /// on the deterministic part of a chain); a real RV lifts to a per-lane `UnOp::Exp` node
    /// (PLAN-FINANCE F1 — `exp(normal(...))` is the lognormal construction); complex `z = a + bi`
    /// → Euler `e^a·(cos b + i·sin b)`, where both channels may be RVs (exp/cos/sin all lift).
    fn cufunc_exp(&mut self, x: Value, span: Span) -> Result<Value> {
        match x {
            Value::Num(n) => Ok(Value::Num(n.exp())),
            Value::Est { val, .. } => Ok(Value::Num(val.exp())),
            Value::Signal(s) => Ok(Value::Signal(Rc::new(SigExpr::Unary(SigUnOp::Exp, s)))),
            x @ Value::Dist(_) => self.lift_unary(UnOp::Exp, x, span),
            Value::Complex { re, im } => {
                // The magnitude factor e^a: folds for a constant, defers for a signal, and
                // errors for a random real part (the recursive call enforces all three).
                let ea = self.cufunc_exp(*re, span)?;
                let cos = self.lib_ufunc(UnOp::Cos, std::slice::from_ref(&im), span)?;
                let sin = self.lib_ufunc(UnOp::Sin, std::slice::from_ref(&im), span)?;
                let re_out = self.binop(BinOp::Mul, ea.clone(), cos, span)?;
                let im_out = self.binop(BinOp::Mul, ea, sin, span)?;
                Ok(Value::complex(re_out, im_out))
            }
            other => Err(NoiseError::runtime(
                format!(
                    "math::exp expects a number or complex value, got {}",
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// `sqrt(z)`. Real `x` → IEEE `√x` (so `sqrt(-1.0)` stays `NaN`; a real RV lifts via `x ^ 0.5`).
    /// **Constant** complex → the principal square root. A complex *random variable* square root is
    /// not supported (exotic; would need per-lane branch logic) — a clear error.
    fn cufunc_sqrt(&mut self, x: Value, span: Span) -> Result<Value> {
        match x {
            Value::Num(n) => Ok(Value::Num(n.sqrt())),
            Value::Est { val, .. } => Ok(Value::Num(val.sqrt())),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Num => {
                self.binop(BinOp::Pow, Value::Dist(id), Value::Num(0.5), span)
            }
            // A lazy signal defers as `x ^ 0.5` (the same lowering the RV path uses).
            sig @ Value::Signal(_) => self.binop(BinOp::Pow, sig, Value::Num(0.5), span),
            Value::Complex { re, im } => match (scalar_const(&re), scalar_const(&im)) {
                (Some(a), Some(b)) => {
                    let r = (a * a + b * b).sqrt();
                    let re_out = ((r + a) / 2.0).max(0.0).sqrt();
                    let mut im_out = ((r - a) / 2.0).max(0.0).sqrt();
                    if b < 0.0 {
                        im_out = -im_out;
                    }
                    Ok(Value::cnum(re_out, im_out))
                }
                _ => Err(NoiseError::runtime(
                    "math::sqrt of a complex random variable is not supported".to_string(),
                    span,
                )),
            },
            other => Err(NoiseError::runtime(
                format!(
                    "math::sqrt expects a number or complex value, got {}",
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// `arg(z)` — the phase angle. Complex `z` → `atan2(im, re)`. Real `x` → `0` for `x ≥ 0`, `π`
    /// for `x < 0` (the real restriction of `atan2`). Both branches lift over RVs.
    fn cufunc_arg(&mut self, x: Value, span: Span) -> Result<Value> {
        match x {
            Value::Complex { re, im } => self.complex_atan2(*im, *re, span),
            Value::Num(n) => Ok(Value::Num(if n < 0.0 { std::f64::consts::PI } else { 0.0 })),
            Value::Est { val, .. } => Ok(Value::Num(if val < 0.0 {
                std::f64::consts::PI
            } else {
                0.0
            })),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Num => {
                let neg = self.binop(BinOp::Lt, Value::Dist(id), Value::Num(0.0), span)?;
                self.select(neg, Value::Num(std::f64::consts::PI), Value::Num(0.0), span)
            }
            // A real lazy signal defers as `π·(x < 0)` (signal comparisons are 0/1).
            sig @ Value::Signal(_) => {
                let neg = self.binop(BinOp::Lt, sig, Value::Num(0.0), span)?;
                self.binop(BinOp::Mul, Value::Num(std::f64::consts::PI), neg, span)
            }
            other => Err(NoiseError::runtime(
                format!(
                    "math::arg expects a number or complex value, got {}",
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// `atan2(y, x)` over real channels (constant, RV, or lazy signal). Folds to `f64::atan2`
    /// when both are constant; a **signal** channel defers to a [`SigExpr::Atan2`] node (the
    /// phase read-out of a complex signal stays lazy); otherwise builds the quadrant-correct form
    /// `atan(y/x) + adj`, where `adj` shifts by `±π` in the left half-plane (`x < 0`). The `x = 0`
    /// axis is measure-zero for the continuous RVs this is used on, so it isn't special-cased in
    /// the lifted path.
    pub(super) fn complex_atan2(&mut self, y: Value, x: Value, span: Span) -> Result<Value> {
        if let (Some(yc), Some(xc)) = (scalar_const(&y), scalar_const(&x)) {
            return Ok(Value::Num(yc.atan2(xc)));
        }
        if matches!(y, Value::Signal(_)) || matches!(x, Value::Signal(_)) {
            let ye = sig_operand(&y, span)?;
            let xe = sig_operand(&x, span)?;
            return Ok(Value::Signal(Rc::new(SigExpr::Atan2(ye, xe))));
        }
        let pi = std::f64::consts::PI;
        let yx = self.binop(BinOp::Div, y.clone(), x.clone(), span)?;
        let core = self.lib_ufunc(UnOp::Atan, std::slice::from_ref(&yx), span)?;
        let x_neg = self.binop(BinOp::Lt, x, Value::Num(0.0), span)?;
        let y_nonneg = self.binop(BinOp::Ge, y, Value::Num(0.0), span)?;
        let pick = self.select(y_nonneg, Value::Num(pi), Value::Num(-pi), span)?;
        let adj = self.select(x_neg, pick, Value::Num(0.0), span)?;
        self.binop(BinOp::Add, core, adj, span)
    }

    /// `floor(x)` / `ceil(x)` — real-only (PLAN-COMPLEX §8). Scalars fold; a real RV lifts to a
    /// `Floor`/`Ceil` node; a complex input is a type error (no Gaussian-integer rounding).
    fn cufunc_round_fam(&mut self, op: UnOp, x: Value, span: Span) -> Result<Value> {
        match x {
            Value::Num(n) => Ok(Value::Num(apply_unop_f64(op, n))),
            Value::Est { val, .. } => Ok(Value::Num(apply_unop_f64(op, val))),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Num => {
                self.lift_unary(op, Value::Dist(id), span)
            }
            // A lazy signal defers the rounding into its tree.
            Value::Signal(s) => Ok(Value::Signal(Rc::new(SigExpr::Unary(SigUnOp::Un(op), s)))),
            Value::Complex { .. } => Err(NoiseError::runtime(
                format!(
                    "{} is real-only — it has no meaning on a complex number",
                    unop_name(op)
                ),
                span,
            )),
            other => Err(NoiseError::runtime(
                format!(
                    "{} expects a number, got {}",
                    unop_name(op),
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// `mse(a, b)` — mean squared error between two equal-length signals: `mean((aᵢ-bᵢ)²)`. A
    /// general "how different are these two signals" measure (e.g. recovered vs. transmitted).
    fn lib_mse(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [a, b] = arity2("mse", args, span)?;
        // `a - b` via `binop` so it broadcasts and materializes a lazy signal against the other
        // operand's length (so `mse(recovered_array, lazy_signal)` just works).
        let diff = self.binop(BinOp::Sub, a.clone(), b.clone(), span)?;
        let n = self.expect_array("mse", &diff, span)?.len();
        if n == 0 {
            return Err(NoiseError::runtime(
                "mse of empty signals".to_string(),
                span,
            ));
        }
        let ss = self.lib_normsq(&[diff], span)?; // Σ (aᵢ-bᵢ)²
        self.binop(BinOp::Div, ss, Value::Num(n as f64), span)
    }

    /// `has_duplicates(xs)` — true iff some pair of elements is equal (the birthday predicate).
    /// Thin wrapper over [`Self::lib_count_duplicates`]: a collision exists iff the count is `> 0`.
    fn lib_has_duplicates(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let count = self.lib_count_duplicates(args, span)?;
        self.binop(BinOp::Gt, count, Value::Num(0.0), span)
    }

    /// `count_duplicates(xs)` — number of equal pairs `i<j` (the count of birthday collisions).
    /// `O(n²)` comparison nodes; fine at the small `n` the headline examples use.
    fn lib_count_duplicates(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [xs] = arity1("count_duplicates", args, span)?;
        let xs = self.expect_array("count_duplicates", xs, span)?;
        let mut count = Value::Num(0.0);
        for i in 0..xs.len() {
            for j in (i + 1)..xs.len() {
                let eq = self.binop(BinOp::Eq, xs[i].clone(), xs[j].clone(), span)?;
                let ind = self.indicator(eq, span)?;
                count = self.binop(BinOp::Add, count, ind, span)?;
            }
        }
        Ok(count)
    }
}
