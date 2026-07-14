//! Conditioning (`X | C`) and the introspection surface: `describe`/`corr`/`explain`, the `plot::*` chart dispatch, and the `stats::*` raw-data twins.
//!
//! Extracted verbatim from the monolithic `eval.rs` (finding F1); an `impl Engine` block
//! whose methods reach the rest of the evaluator through `self` and the shared free
//! helpers/tables that stay in the module root.

use std::rc::Rc;

use super::*;
use crate::builtins;
use crate::dist::{RvId, RvKind, RvNode};
use crate::error::{NoiseError, Result, Span};
use crate::value::Value;

impl Engine {
    /// Build a conditioned value from `event | given` (LANG.md "conditioning"). It records the
    /// quantity and the condition *separately* in a `Value::Cond` (the fusion into
    /// `select(condition, quantity, NaN)` is deferred to the query) so operations compose:
    /// `2*(X|C)+1` is `(2X+1) | C`. `given` must be an event (bool); `event` may be an event (for
    /// `P`) or any numeric quantity (for `E`/`Var`/`Q`, checked at the query). A conditioned value
    /// can be bound and queried; combining two that are conditioned on *different* events is rejected
    /// (`binop_cond`).
    pub(super) fn eval_cond(&mut self, event: &Spanned, given: &Spanned) -> Result<Value> {
        // The condition: a bool RV (or deterministic bool). A constant `true` is the unconditional
        // query; a constant `false` makes the condition never hold (a 0/0, caught at the query).
        let cond_v = self.eval(given)?;
        forbid_undrawn(&cond_v, given.span)?;
        let (condition, cond_kind) = self.operand_to_rv(cond_v, given.span)?;
        if cond_kind != RvKind::Bool {
            return Err(NoiseError::runtime(
                "the condition after `|` must be an event (bool), e.g. `X > 0`".to_string(),
                given.span,
            ));
        }
        // The quantity: an event (bool) or a number — the query (`P` vs `E`/`Var`/`Q`) decides which
        // it accepts, using the `q_kind` recorded here.
        let quantity_v = self.eval(event)?;
        forbid_undrawn(&quantity_v, event.span)?;
        let (quantity, q_kind) = self.operand_to_rv(quantity_v, event.span)?;
        Ok(Value::Cond {
            quantity,
            q_kind,
            condition,
        })
    }

    /// Query a conditioned value — `P(a)`, `E(a)`, `Var(a)`, `Q(a, q)` where `a = X | C`. Fuse the
    /// condition and quantity into the single root `select(C, quantity, NaN)` — the quantity on the
    /// lanes where `C` holds, the NaN sentinel elsewhere — so one sampling pass draws quantity and
    /// condition *jointly* (shared upstream draws). The conditional estimators then average / sort
    /// over the non-NaN (in-condition) lanes. Sampling jointly is what makes `Q(a, q)` correct: two
    /// separate passes would mis-pair the lanes. `arg_vals[0]` is the conditioned value; the rest are
    /// the ordinary trailing arguments (an optional sample count, plus `q` for `Q`).
    pub(super) fn query_cond(
        &mut self,
        qname: &str,
        arg_vals: &[Value],
        span: Span,
    ) -> Result<Value> {
        let (quantity, q_kind, condition) = match arg_vals[0] {
            Value::Cond {
                quantity,
                q_kind,
                condition,
            } => (quantity, q_kind, condition),
            _ => unreachable!("query_cond called without a conditioned first argument"),
        };
        if qname == "P" && q_kind != RvKind::Bool {
            return Err(NoiseError::runtime(
                "P expects an event (bool) — a conditioned number works with E/Var/Q, e.g. `E(X | C)`"
                    .to_string(),
                span,
            ));
        }
        // KNOWN HOLE (finding B2): the NaN sentinel conflates "condition is false" with "the
        // quantity itself is NaN on an in-condition lane". The conditional reducers/collectors drop
        // *every* NaN lane, so an in-condition NaN quantity is silently discarded rather than
        // propagated — biasing the estimate and reporting a falsely-tight SE (the dropped lanes
        // don't count against `m`). Example: `E(math::log(X) | X > -1)` with `X ~ unif(-1,1)` returns
        // ≈ -1 (the mean over the `X > 0` lanes) when the honest answer is NaN — `log(X)` is NaN on
        // the `X < 0` lanes, which are *in condition* with probability ~1/2. The correct fix is a
        // dedicated condition column (as `sample_pairs` carries one), keeping the quantity's NaN
        // distinct from the sentinel; it is deferred because the single-root reduce/backend path
        // (`reduce`/`Runner`, incl. JIT/wasm) produces one column per batch, and re-routing to a
        // two-root interpreter pass would change RNG consumption order. See the matching notes on
        // `reduce::CondMomentsReducer` and `sampler::cond_sample_n`.
        let nan = self.graph.push(RvNode::ConstNum(f64::NAN), RvKind::Num);
        let root = self.graph.push(
            RvNode::Select {
                cond: condition,
                a: quantity,
                b: nan,
            },
            RvKind::Num,
        );
        let tail = &arg_vals[1..];
        let ctx = self.query_ctx(span);
        match qname {
            "P" => builtins::prob_cond(root, tail, &ctx),
            "E" | "Var" => builtins::moment_cond(qname, root, tail, &ctx),
            "Q" => builtins::quantile_cond(root, tail, &ctx),
            _ => unreachable!("query_cond dispatched with an unknown name"),
        }
    }

    /// Dispatch a variable-introspection call (`describe`/`hist`/`samples`/`corr`/`scatter`/
    /// `explain`) to the [`crate::introspect`] core, returning a [`Value::Summary`]. `args` is the
    /// un-evaluated argument list (for labelling a summary by its source name); `arg_vals` are the
    /// evaluated operands. Kept its own method so `eval`'s frame stays small (recursion budget). All
    /// six are *views/compositions* of two operations: a one-variable [`Dist1`](crate::introspect::Dist1)
    /// and a two-variable [`Dist2`](crate::introspect::Dist2).
    pub(super) fn introspect_call(
        &mut self,
        name: &str,
        args: &[Spanned],
        arg_vals: &[Value],
        span: Span,
    ) -> Result<Value> {
        use crate::introspect::{
            dist1, dist2, Dist1, Dist2, Explain, Payload, Summary, View, INTROSPECT_N,
            INTROSPECT_SEED,
        };
        // Introspection runs at its own modest, capped budget (a visual, not a probability).
        let n = self.max_samples.min(INTROSPECT_N);
        let seed = INTROSPECT_SEED;
        let summary = |view, label, label_b, payload| {
            Ok(Value::Summary(Rc::new(Summary {
                view,
                label,
                label_b,
                payload,
            })))
        };
        match name {
            "describe" | "hist" | "samples" => {
                if arg_vals.is_empty()
                    || arg_vals.len() > 2
                    || (name != "samples" && arg_vals.len() != 1)
                {
                    return Err(NoiseError::runtime(
                        format!(
                            "{name} expects 1 argument (a variable to inspect){}",
                            if name == "samples" {
                                " and an optional count"
                            } else {
                                ""
                            }
                        ),
                        span,
                    ));
                }
                let head_k = match arg_vals.get(1) {
                    Some(v) => introspect_count(v, span)?,
                    None => {
                        if name == "samples" {
                            10
                        } else {
                            8
                        }
                    }
                };
                let label = label_of(&args[0]);
                // `describe` is polymorphic: a scalar number/estimate → a value+CI card, an array →
                // a per-cell grid (vector series / matrix heatmap). `hist`/`samples` stay scalar-RV.
                if name == "describe" {
                    match &arg_vals[0] {
                        Value::Num(_) | Value::Est { .. } | Value::Bool(_) => {
                            return self.value_card(&arg_vals[0], label);
                        }
                        Value::Array(xs) => {
                            let xs = xs.clone();
                            return self.grid_summary(&xs, label, span);
                        }
                        // A lazy signal renders at the ambient resolution — `plot::line(msg)`
                        // works without an inline length, like the reducers.
                        Value::Signal(s) => {
                            let n = self.resolution;
                            let xs = self.materialize_sig(&s.clone(), n, span)?;
                            return self.grid_summary(&xs, label, span);
                        }
                        Value::Complex { .. } => {
                            return Err(complex_has_no_trace(span));
                        }
                        _ => {}
                    }
                }
                let view = match name {
                    "hist" => View::Hist,
                    "samples" => View::Samples,
                    _ => View::Describe,
                };
                let (root, conditional, boolean) = self.introspect_root(&arg_vals[0], span)?;
                if self.check_mode {
                    return summary(
                        view,
                        label,
                        None,
                        Payload::One(Dist1::from_draws(&[0.0], boolean, 0)),
                    );
                }
                match dist1(&self.graph, root, boolean, conditional, n, seed, head_k)? {
                    Some(d) => summary(view, label, None, Payload::One(d)),
                    None => Err(condition_never(n, span)),
                }
            }
            "corr" | "scatter" => {
                // `corr(vec)` — one array argument → the element×element correlation heatmap.
                if name == "corr" && arg_vals.len() == 1 {
                    let label = label_of(&args[0]);
                    let xs = match &arg_vals[0] {
                        Value::Array(xs) => xs.clone(),
                        other => {
                            return Err(NoiseError::runtime(
                                format!("corr needs two variables to compare, or one vector to correlate its elements — got {}", other.type_name()),
                                span,
                            ))
                        }
                    };
                    return self.corr_matrix_summary(&xs, label, span);
                }
                if arg_vals.len() != 2 {
                    return Err(NoiseError::runtime(
                        format!(
                            "{name} expects 2 variables to compare, got {}",
                            arg_vals.len()
                        ),
                        span,
                    ));
                }
                let (la, lb) = (label_of(&args[0]), label_of(&args[1]));
                let a = self.introspect_plain_root(&arg_vals[0], name, span)?;
                let b = self.introspect_plain_root(&arg_vals[1], name, span)?;
                let view = if name == "scatter" {
                    View::Scatter
                } else {
                    View::Corr
                };
                if self.check_mode {
                    return summary(
                        view,
                        la,
                        Some(lb),
                        Payload::Two(Dist2::from_pairs(&[(0.0, 0.0)], 1)),
                    );
                }
                match dist2(&self.graph, a, b, None, n, seed)? {
                    Some(d) => summary(view, la, Some(lb), Payload::Two(d)),
                    None => Err(condition_never(n, span)),
                }
            }
            "explain" => {
                if arg_vals.len() != 1 {
                    return Err(NoiseError::runtime(
                        format!(
                            "explain expects 1 variable to explain, got {}",
                            arg_vals.len()
                        ),
                        span,
                    ));
                }
                let label = label_of(&args[0]);
                // Target quantity (+ optional condition for `explain(Y | C)`).
                let (target, cond) = match &arg_vals[0] {
                    Value::Cond {
                        quantity,
                        condition,
                        ..
                    } => (*quantity, Some(*condition)),
                    other => (self.operand_to_rv(other.clone(), span)?.0, None),
                };
                if self.check_mode {
                    return summary(
                        View::Explain,
                        label,
                        None,
                        Payload::Explain(Explain::from_candidates(0.0, vec![])),
                    );
                }
                // Candidate drivers: named random variables that are upstream of the target (and not
                // the target itself). Collected (owned) before any `&mut self` so the scope borrow ends.
                let anc = ancestors(&self.graph, target);
                let mut cands: Vec<(String, RvId)> = self
                    .vars
                    .iter()
                    .filter_map(|(k, v)| match v {
                        Value::Dist(id) if *id != target && anc.contains(id) => {
                            Some((k.clone(), *id))
                        }
                        _ => None,
                    })
                    .collect();
                cands.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic ranking on ties
                                                     // Total spread of the target (conditional sd if `explain(Y | C)`).
                let target_root = match cond {
                    Some(c) => {
                        let nan = self.graph.push(RvNode::ConstNum(f64::NAN), RvKind::Num);
                        self.graph.push(
                            RvNode::Select {
                                cond: c,
                                a: target,
                                b: nan,
                            },
                            RvKind::Num,
                        )
                    }
                    None => target,
                };
                let sd = dist1(&self.graph, target_root, false, cond.is_some(), n, seed, 0)?
                    .map_or(f64::NAN, |d| d.sd);
                let mut corrs = Vec::with_capacity(cands.len());
                for (cname, cid) in &cands {
                    if let Some(d) = dist2(&self.graph, target, *cid, cond, n, seed)? {
                        corrs.push((cname.clone(), d.corr));
                    }
                }
                summary(
                    View::Explain,
                    label,
                    None,
                    Payload::Explain(Explain::from_candidates(sd, corrs)),
                )
            }
            _ => unreachable!("introspect_call dispatched with an unknown name"),
        }
    }

    /// Build the sampling root for a one-variable introspection. A plain value lifts to its RV (a
    /// `dist`, or a folded constant); a conditioned value `X | C` fuses to `select(C, X, NaN)` and
    /// marks the summary `conditional` (its NaN lanes are dropped). Returns `(root, conditional,
    /// boolean)` where `boolean` flags an event quantity (draws are 0/1).
    fn introspect_root(&mut self, v: &Value, span: Span) -> Result<(RvId, bool, bool)> {
        match v {
            Value::Cond {
                quantity,
                q_kind,
                condition,
            } => {
                let nan = self.graph.push(RvNode::ConstNum(f64::NAN), RvKind::Num);
                let root = self.graph.push(
                    RvNode::Select {
                        cond: *condition,
                        a: *quantity,
                        b: nan,
                    },
                    RvKind::Num,
                );
                Ok((root, true, *q_kind == RvKind::Bool))
            }
            other => {
                let (id, kind) = self.operand_to_rv(other.clone(), span)?;
                Ok((id, false, kind == RvKind::Bool))
            }
        }
    }

    /// Build the root for a `corr`/`scatter` operand. These need a *joint* pass over two plain roots
    /// (the paired-sampling machinery); a conditioned operand isn't supported yet, so it's a clear
    /// spanned error rather than a wrong answer (condition upstream, or use `explain`).
    fn introspect_plain_root(&mut self, v: &Value, name: &str, span: Span) -> Result<RvId> {
        if matches!(v, Value::Cond { .. }) {
            return Err(NoiseError::runtime(
                format!("{name} doesn't support a conditioned value yet — condition upstream, or use `explain`"),
                span,
            ));
        }
        Ok(self.operand_to_rv(v.clone(), span)?.0)
    }

    /// `describe` of a scalar `number`/`bool`/estimate → a value-with-uncertainty card. A plain
    /// number is an exact point; an `Est` (e.g. `4*P(…)`) carries its standard error, so even a
    /// query result has something to look at (its confidence interval).
    fn value_card(&self, v: &Value, label: String) -> Result<Value> {
        use crate::introspect::{Payload, Summary, ValueCard, View};
        let card = match v {
            Value::Num(x) => ValueCard { val: *x, se: 0.0 },
            Value::Est { val, se } => ValueCard { val: *val, se: *se },
            Value::Bool(b) => ValueCard {
                val: f64::from(*b),
                se: 0.0,
            },
            _ => unreachable!("value_card only reached for a scalar"),
        };
        Ok(Value::Summary(Rc::new(Summary {
            view: View::Value,
            label,
            label_b: None,
            payload: Payload::Value(card),
        })))
    }

    /// `describe` of an array → a per-cell grid summary (vector series / matrix heatmap), sampled in
    /// one joint pass.
    fn grid_summary(&mut self, xs: &[Value], label: String, span: Span) -> Result<Value> {
        use crate::introspect::{
            grid, DistGrid, Payload, Summary, View, INTROSPECT_N, INTROSPECT_SEED,
        };
        let wrap = |g| {
            Value::Summary(Rc::new(Summary {
                view: View::Grid,
                label,
                label_b: None,
                payload: Payload::Grid(g),
            }))
        };
        let (roots, rows, cols) = self.array_roots(xs, span)?;
        if self.check_mode {
            return Ok(wrap(DistGrid {
                rows,
                cols,
                mean: vec![],
                sd: vec![],
            }));
        }
        let n = self.max_samples.min(INTROSPECT_N);
        Ok(wrap(grid(
            &self.graph,
            &roots,
            rows,
            cols,
            n,
            INTROSPECT_SEED,
        )?))
    }

    /// `corr(vec)` → the element×element correlation heatmap, sampled in one joint pass.
    fn corr_matrix_summary(&mut self, xs: &[Value], label: String, span: Span) -> Result<Value> {
        use crate::introspect::{Payload, Summary, View};
        let c = self.corr_matrix_of(xs, span)?;
        Ok(Value::Summary(Rc::new(Summary {
            view: View::CorrMatrix,
            label,
            label_b: None,
            payload: Payload::CorrMatrix(c),
        })))
    }

    /// The element×element correlation matrix of a vector, in one joint pass — the computation
    /// behind `corr(vec)` / `plot::corr(vec)` / `stats::corr(vec)`. Restricted to a vector (1-D) and
    /// capped in length, since the cost is O(n²) per lane. In check mode the matrix is empty (no
    /// sampling), and the callers fill the shape.
    fn corr_matrix_of(
        &mut self,
        xs: &[Value],
        span: Span,
    ) -> Result<crate::introspect::CorrMatrix> {
        use crate::introspect::{corr_grid, CorrMatrix, CORR_MAX, INTROSPECT_SEED};
        let roots = self.vector_roots(xs, span)?;
        if roots.len() > CORR_MAX {
            return Err(NoiseError::runtime(
                format!(
                    "corr supports up to {CORR_MAX} elements, got {} — slice the vector first",
                    roots.len()
                ),
                span,
            ));
        }
        if self.check_mode {
            return Ok(CorrMatrix {
                n: roots.len(),
                corr: vec![],
            });
        }
        // The pairwise matrix needs fewer draws than a single estimate; cap it to stay snappy.
        let n = self.max_samples.min(100_000);
        corr_grid(&self.graph, &roots, n, INTROSPECT_SEED)
    }

    /// Lower a vector of scalar RVs/constants to element roots (a nested array — a matrix — is a
    /// spanned error here; `array_roots` handles the 2-D case).
    fn vector_roots(&mut self, xs: &[Value], span: Span) -> Result<Vec<RvId>> {
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                "cannot inspect an empty array".to_string(),
                span,
            ));
        }
        let mut roots = Vec::with_capacity(xs.len());
        for x in xs {
            if matches!(x, Value::Array(_)) {
                return Err(NoiseError::runtime(
                    "expected a vector of variables (a 1-D array)".to_string(),
                    span,
                ));
            }
            if matches!(x, Value::Complex { .. }) {
                return Err(complex_has_no_trace(span));
            }
            roots.push(self.operand_to_rv(x.clone(), span)?.0);
        }
        Ok(roots)
    }

    /// Lower an array to `(roots, rows, cols)` (row-major) — a vector (`rows == 1`) or a rectangular
    /// matrix of scalar RVs. Ragged/3-D/oversized arrays are spanned errors.
    fn array_roots(&mut self, xs: &[Value], span: Span) -> Result<(Vec<RvId>, usize, usize)> {
        const GRID_MAX: usize = 1024;
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                "cannot inspect an empty array".to_string(),
                span,
            ));
        }
        if let Value::Array(first) = &xs[0] {
            let (rows, cols) = (xs.len(), first.len());
            let mut roots = Vec::with_capacity(rows.saturating_mul(cols));
            for row in xs {
                let r = match row {
                    Value::Array(r) => r,
                    _ => {
                        return Err(NoiseError::runtime(
                            "a matrix needs array rows (this array mixes rows and scalars)"
                                .to_string(),
                            span,
                        ))
                    }
                };
                if r.len() != cols {
                    return Err(NoiseError::runtime(
                        format!(
                            "a matrix must be rectangular; rows have lengths {cols} and {}",
                            r.len()
                        ),
                        span,
                    ));
                }
                for cell in r.iter() {
                    if matches!(cell, Value::Array(_)) {
                        return Err(NoiseError::runtime(
                            "3-D arrays aren't supported".to_string(),
                            span,
                        ));
                    }
                    roots.push(self.operand_to_rv(cell.clone(), span)?.0);
                }
            }
            if roots.len() > GRID_MAX {
                return Err(NoiseError::runtime(
                    format!(
                        "array too large to inspect ({} cells, max {GRID_MAX})",
                        roots.len()
                    ),
                    span,
                ));
            }
            Ok((roots, rows, cols))
        } else {
            let roots = self.vector_roots(xs, span)?;
            if roots.len() > GRID_MAX {
                return Err(NoiseError::runtime(
                    format!(
                        "array too large to inspect ({} elems, max {GRID_MAX})",
                        roots.len()
                    ),
                    span,
                ));
            }
            let cols = roots.len();
            Ok((roots, 1, cols))
        }
    }

    /// `plot::*` — the example-facing charting surface. Each maps to the introspection core (a plot
    /// is just a summary the program asked to *see*), captures the result in the `plots` buffer (so
    /// the CLI/playground render it), and evaluates to unit — a statement, like `Print`. The chart
    /// kind comes from the resulting payload, so the names are intent-revealing sugar:
    /// `histogram`/`hist` (a distribution), `line`/`heatmap`/`value`/`show` (polymorphic `describe`
    /// of a vector/matrix/scalar), `scatter`, `corr` (pair or element-matrix), `explain`, `samples`.
    pub(super) fn plot_call(
        &mut self,
        base: &str,
        args: &[Spanned],
        arg_vals: &[Value],
        span: Span,
    ) -> Result<Value> {
        // `fan` has no scalar-introspection counterpart (it's a whole-path quantile chart), so it
        // dispatches to its own summary builder rather than the `introspect_call` core.
        if base == "fan" {
            let summary = self.fan_summary(args, arg_vals, span)?;
            if let Value::Summary(s) = &summary {
                self.emit(Output::Plot(s.clone()));
            }
            return Ok(Value::Unit);
        }
        let inner = match base {
            "histogram" | "hist" => "hist",
            "line" | "heatmap" | "value" | "show" | "dist" | "describe" => "describe",
            "scatter" => "scatter",
            "corr" => "corr",
            "explain" => "explain",
            "samples" => "samples",
            other => {
                return Err(NoiseError::runtime(
                    format!("unknown plot 'plot::{other}' (try {})", PLOT_FNS.join(", ")),
                    span,
                ))
            }
        };
        let summary = self.introspect_call(inner, args, arg_vals, span)?;
        if let Value::Summary(s) = &summary {
            self.emit(Output::Plot(s.clone()));
        }
        // A plot is captured into the output stream (rendered by the host), not a value to fold — it
        // yields unit like `Print`, and interleaves with `Print` lines in source order.
        Ok(Value::Unit)
    }

    // === `stats::*` — the numbers behind the pictures ==========================================
    //
    // Every `plot::` shortcut is a computation plus a chart. `stats::` is the same computation
    // without the chart: `stats::histogram(x)` returns the very bins `plot::hist(x)` draws,
    // `stats::fan(path)` the very bands `plot::fan(path)` shades. Not a re-implementation — both
    // call the same functions in `crate::introspect`, at the same budget and the same seed, so the
    // two agree exactly rather than approximately. That is what makes a chart auditable: you can
    // always ask for its data.
    //
    // These force sampling (like `P`/`E`/`Var`/`Q`), which is why they live in `stats::` and not in
    // the never-sampling `math::`. Their budget is the *introspection* budget (`INTROSPECT_N`), not
    // `P`'s — a chart's numbers, not a probability's last digit. So `Q(x, 0.5)` and
    // `stats::quantiles(x, [0.5])` may differ in the final places: `Q` is the estimator,
    // `stats::quantiles` is what the picture is made of.

    /// Dispatch a `stats::` call. Each arm returns plain numbers/arrays, so the result composes with
    /// the rest of the language (index it, plot it, feed it back in).
    pub(super) fn stats_call(
        &mut self,
        base: &str,
        arg_vals: &[Value],
        span: Span,
    ) -> Result<Value> {
        match base {
            "histogram" => self.stats_histogram(arg_vals, span),
            "quantiles" => self.stats_quantiles(arg_vals, span),
            "moments" => self.stats_moments(arg_vals, span),
            "fan" => self.stats_fan(arg_vals, span),
            "corr" => self.stats_corr(arg_vals, span),
            other => Err(NoiseError::runtime(
                format!("unknown 'stats::{other}' (try histogram, quantiles, moments, fan, corr)"),
                span,
            )),
        }
    }

    /// `stats::histogram(x)` / `stats::histogram(x, bins)` → `[[midpoints], [counts]]`, the two rows
    /// a bar chart needs. Accepts a conditioned variable (`stats::histogram(x | c)`), like `hist`.
    /// An event has exactly two buckets, `false` and `true`, whatever `bins` says.
    fn stats_histogram(&mut self, arg_vals: &[Value], span: Span) -> Result<Value> {
        use crate::introspect::{histogram, Histogram, NUM_BINS};
        if arg_vals.is_empty() || arg_vals.len() > 2 {
            return Err(NoiseError::runtime(
                format!("stats::histogram expects a variable and an optional bin count, got {} arguments", arg_vals.len()),
                span,
            ));
        }
        let nbins = match arg_vals.get(1) {
            Some(v) => introspect_count(v, span)?,
            None => NUM_BINS,
        };
        if nbins == 0 {
            return Err(NoiseError::runtime(
                "stats::histogram needs at least 1 bin".to_string(),
                span,
            ));
        }
        let (root, conditional, boolean) = self.introspect_root(&arg_vals[0], span)?;
        let nbins = if boolean { 2 } else { nbins };
        let h = match self.stats_draws(root, conditional, span)? {
            None => Histogram {
                lo: 0.0,
                hi: 1.0,
                bins: vec![0; nbins],
            }, // check mode: shape only
            Some(draws) => histogram(&draws, boolean, nbins),
        };
        Ok(matrix_of([
            h.midpoints(boolean),
            h.bins.iter().map(|&c| c as f64).collect(),
        ]))
    }

    /// `stats::quantiles(x, [q…])` → one value per requested quantile, in the order asked. The same
    /// empirical rule `Q` uses, over the histogram's sample.
    fn stats_quantiles(&mut self, arg_vals: &[Value], span: Span) -> Result<Value> {
        use crate::introspect::quantile_sorted;
        if arg_vals.len() != 2 {
            return Err(NoiseError::runtime(
                "stats::quantiles expects a variable and an array of quantiles, e.g. `stats::quantiles(x, [0.05, 0.5, 0.95])`".to_string(),
                span,
            ));
        }
        let qs = match &arg_vals[1] {
            Value::Array(qs) => qs.clone(),
            other => {
                return Err(NoiseError::runtime(
                    format!("stats::quantiles wants an array of quantiles, got {} — for a single one, `Q(x, 0.5)`", other.type_name()),
                    span,
                ))
            }
        };
        let mut levels = Vec::with_capacity(qs.len());
        for q in qs.iter() {
            let q = match q {
                Value::Num(q) | Value::Est { val: q, .. } => *q,
                other => {
                    return Err(NoiseError::runtime(
                        format!("a quantile must be a number, got {}", other.type_name()),
                        span,
                    ))
                }
            };
            if !(0.0..=1.0).contains(&q) {
                return Err(NoiseError::runtime(
                    format!("a quantile must lie in [0, 1], got {q}"),
                    span,
                ));
            }
            levels.push(q);
        }
        let (root, conditional, _) = self.introspect_root(&arg_vals[0], span)?;
        let Some(mut draws) = self.stats_draws(root, conditional, span)? else {
            return Ok(row_of(vec![0.0; levels.len()])); // check mode: shape only
        };
        draws.sort_by(f64::total_cmp);
        Ok(row_of(
            levels
                .into_iter()
                .map(|q| quantile_sorted(&draws, q))
                .collect(),
        ))
    }

    /// `stats::moments(x)` → `[n, mean, sd, min, max]` — the header line of `describe(x)`, as data.
    /// `n` is the number of draws that *counted* (a conditioned variable keeps only its in-condition
    /// lanes), so it is the honest denominator behind the other four.
    fn stats_moments(&mut self, arg_vals: &[Value], span: Span) -> Result<Value> {
        use crate::introspect::Dist1;
        if arg_vals.len() != 1 {
            return Err(NoiseError::runtime(
                format!("stats::moments expects 1 variable, got {}", arg_vals.len()),
                span,
            ));
        }
        let (root, conditional, boolean) = self.introspect_root(&arg_vals[0], span)?;
        let Some(draws) = self.stats_draws(root, conditional, span)? else {
            return Ok(row_of(vec![0.0; 5])); // check mode: shape only
        };
        let d = Dist1::from_draws(&draws, boolean, 0);
        Ok(row_of(vec![d.n as f64, d.mean, d.sd, d.min, d.max]))
    }

    /// `stats::fan(path)` → a 6×cols matrix: the rows `q05, q25, q50, q75, q95, mean`, one column
    /// per index. Exactly the bands `plot::fan(path)` shades, drawn in one joint pass.
    fn stats_fan(&mut self, arg_vals: &[Value], span: Span) -> Result<Value> {
        let c = self.fan_chart("stats::fan", arg_vals, span)?;
        if c.q50.is_empty() {
            return Ok(matrix_of([(); 6].map(|_| vec![0.0; c.cols]))); // check mode: shape only
        }
        Ok(matrix_of([c.q05, c.q25, c.q50, c.q75, c.q95, c.mean]))
    }

    /// `stats::corr(a, b)` → their correlation, one number. `stats::corr(v)` → the element×element
    /// matrix of a vector (`n×n`, diagonal 1) — the heatmap `plot::corr(v)` draws.
    fn stats_corr(&mut self, arg_vals: &[Value], span: Span) -> Result<Value> {
        use crate::introspect::{dist2, INTROSPECT_N, INTROSPECT_SEED};
        if let [Value::Array(xs)] = arg_vals {
            let c = self.corr_matrix_of(&xs.clone(), span)?;
            if c.corr.is_empty() {
                return Ok(matrix_of((0..c.n).map(|_| vec![0.0; c.n]))); // check mode: shape only
            }
            return Ok(matrix_of(c.corr.chunks(c.n).map(<[f64]>::to_vec)));
        }
        if arg_vals.len() != 2 {
            return Err(NoiseError::runtime(
                format!("stats::corr expects two variables, or one vector to correlate its elements — got {} arguments", arg_vals.len()),
                span,
            ));
        }
        let a = self.introspect_plain_root(&arg_vals[0], "stats::corr", span)?;
        let b = self.introspect_plain_root(&arg_vals[1], "stats::corr", span)?;
        if self.check_mode {
            return Ok(Value::Num(0.0));
        }
        let n = self.max_samples.min(INTROSPECT_N);
        match dist2(&self.graph, a, b, None, n, INTROSPECT_SEED)? {
            Some(d) => Ok(Value::Num(d.corr)),
            None => Err(condition_never(n, span)),
        }
    }

    /// Draw the sample behind a one-variable `stats::` call, at the introspection budget and seed —
    /// the same draws `describe`/`hist` see. `None` means check mode (no sampling happened, and the
    /// caller returns a correctly-shaped placeholder); an empty in-condition sample is an error, as
    /// it is for `describe`.
    fn stats_draws(
        &mut self,
        root: RvId,
        conditional: bool,
        span: Span,
    ) -> Result<Option<Vec<f64>>> {
        use crate::introspect::{draws, INTROSPECT_N, INTROSPECT_SEED};
        if self.check_mode {
            return Ok(None);
        }
        let n = self.max_samples.min(INTROSPECT_N);
        let d = draws(&self.graph, root, conditional, n, INTROSPECT_SEED)?;
        if d.is_empty() {
            return Err(condition_never(n, span));
        }
        Ok(Some(d))
    }

    /// `plot::fan(path)` — the cone chart: per-index quantile bands (q05/q25/q50/q75/q95) of a
    /// vector of random variables, sampled **jointly** in one pass (shared lanes,
    /// [`crate::introspect::fan`]) so the bands are consistent across the index. Restricted to a
    /// vector — a path. A deterministic numeric array is the degenerate fan (every band equals the
    /// values); a scalar or matrix is a friendly spanned error.
    fn fan_summary(&mut self, args: &[Spanned], arg_vals: &[Value], span: Span) -> Result<Value> {
        use crate::introspect::{Payload, Summary, View};
        // `fan_chart` validates the arity first (`plot::fan()` with no args must be a clean error,
        // not an `args[0]` panic — the no-panics contract). After it succeeds there is exactly one
        // argument, so labelling by `args[0]` is safe.
        let c = self.fan_chart("plot::fan", arg_vals, span)?;
        let label = label_of(&args[0]);
        Ok(Value::Summary(Rc::new(Summary {
            view: View::Fan,
            label,
            label_b: None,
            payload: Payload::Fan(c),
        })))
    }

    /// The per-index quantile bands of a path, in ONE joint pass — the computation behind
    /// `plot::fan(path)` and `stats::fan(path)`. `who` names the caller so the errors stay honest
    /// about which surface the program used. In check mode the bands are empty (no sampling); the
    /// callers fill the shape from `cols`.
    fn fan_chart(
        &mut self,
        who: &str,
        arg_vals: &[Value],
        span: Span,
    ) -> Result<crate::introspect::FanChart> {
        use crate::introspect::{fan, FanChart, FAN_MAX, INTROSPECT_N, INTROSPECT_SEED};
        if arg_vals.len() != 1 {
            return Err(NoiseError::runtime(
                format!(
                    "{who} expects 1 argument (a path — an array of random values), got {}",
                    arg_vals.len()
                ),
                span,
            ));
        }
        let not_a_path = || {
            NoiseError::runtime(
                format!("{who} wants a vector — a path of random values"),
                span,
            )
        };
        let xs = match &arg_vals[0] {
            Value::Array(xs) => xs.clone(),
            _ => return Err(not_a_path()),
        };
        if xs.iter().any(|x| matches!(x, Value::Array(_))) {
            return Err(not_a_path()); // a matrix (nested rows) has no single index to fan over
        }
        if xs.len() > FAN_MAX {
            return Err(NoiseError::runtime(
                format!(
                    "{who} supports up to {FAN_MAX} elements, got {} — slice the path first",
                    xs.len()
                ),
                span,
            ));
        }
        let roots = self.vector_roots(&xs, span)?;
        if self.check_mode {
            let empty = FanChart {
                cols: roots.len(),
                n: 0,
                q05: vec![],
                q25: vec![],
                q50: vec![],
                q75: vec![],
                q95: vec![],
                mean: vec![],
            };
            return Ok(empty);
        }
        // The introspection budget, further clamped inside `fan` to its cols×n memory cap.
        let n = self.max_samples.min(INTROSPECT_N);
        fan(&self.graph, &roots, n, INTROSPECT_SEED)
    }
}
