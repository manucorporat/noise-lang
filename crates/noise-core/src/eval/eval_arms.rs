//! AST evaluation arms: identifiers, module resolution, calls (with named-arg reordering), arrays, ranges, indexing/gather, loops, comprehensions, bindings, and `input::*`.
//!
//! Extracted verbatim from the monolithic `eval.rs` (finding F1); an `impl Engine` block
//! whose methods reach the rest of the evaluator through `self` and the shared free
//! helpers/tables that stay in the module root.

use std::rc::Rc;

use super::*;
use crate::builtins;
use crate::dist::{RvKind, RvNode};
use crate::error::{NoiseError, Result, Span};
use crate::input::{InputKind, InputSpec, InputValue, ResolvedInput};
use crate::value::Value;

impl Engine {
    /// Render a template's parts to a string: literal text verbatim, each hole via its value's
    /// display form (an `Est` self-rounds to its standard error, exactly like `Print`). Holes are
    /// deterministic-only — an undrawn recipe is the same error string `+` raises.
    pub(super) fn render_template(&mut self, parts: &[crate::ast::TemplatePart]) -> Result<String> {
        use crate::ast::TemplatePart;
        let mut out = String::new();
        for part in parts {
            match part {
                TemplatePart::Lit(s) => out.push_str(s),
                TemplatePart::Hole(expr) => {
                    let v = self.eval(expr)?;
                    forbid_undrawn(&v, expr.span)?;
                    out.push_str(&v.to_string());
                }
            }
        }
        Ok(out)
    }

    /// Resolve an identifier — a variable, or a constant (`pi`/`e`) gated by module scope. A
    /// qualified `math::pi` always resolves; a bare `pi` needs `math` active (`use math;`).
    pub(super) fn eval_ident(&self, name: &str, span: Span) -> Result<Value> {
        let (module, base) = split_path(name);
        match module {
            // Qualified: only `math`'s constants are *values*; everything else is a function
            // (must be called) or unknown.
            Some(m) => {
                if !is_module(m) {
                    return Err(NoiseError::runtime(format!("unknown module '{m}'"), span));
                }
                if let Some(c) = math_const(base) {
                    if m == "math" {
                        return Ok(Value::Num(c));
                    }
                    return Err(NoiseError::runtime(
                        format!("'{base}' is in module 'math', not '{m}'"),
                        span,
                    ));
                }
                if math_const_complex(base) {
                    if m == "math" {
                        return Ok(Value::cnum(0.0, 1.0));
                    }
                    return Err(NoiseError::runtime(
                        format!("'{base}' is in module 'math', not '{m}'"),
                        span,
                    ));
                }
                Err(match module_of(base) {
                    Some(real) if real == m => NoiseError::runtime(
                        format!("'{m}::{base}' is a function — call it, e.g. `{m}::{base}(...)`"),
                        span,
                    ),
                    Some(real) => NoiseError::runtime(
                        format!("'{base}' is in module '{real}', not '{m}'"),
                        span,
                    ),
                    None => NoiseError::runtime(format!("module '{m}' has no item '{base}'"), span),
                })
            }
            // Bare: a variable wins; then a math constant if `math` is in scope.
            None => {
                if let Some(v) = self.vars.get(name) {
                    return Ok(v.clone());
                }
                if let Some(c) = math_const(name) {
                    if self.module_active("math") {
                        return Ok(Value::Num(c));
                    }
                    return Err(NoiseError::runtime(
                        format!("'{name}' is in module 'math' — add `use math;` or write `math::{name}`"),
                        span,
                    ));
                }
                if math_const_complex(name) {
                    if self.module_active("math") {
                        return Ok(Value::cnum(0.0, 1.0));
                    }
                    return Err(NoiseError::runtime(
                        format!("'{name}' is in module 'math' — add `use math;` or write `math::{name}`"),
                        span,
                    ));
                }
                Err(NoiseError::undefined_name(name, span))
            }
        }
    }

    /// Whether a module's items are reachable unqualified. `builtin` is always active; the rest
    /// require a `use`.
    fn module_active(&self, m: &str) -> bool {
        m == "builtin" || self.used.contains(m)
    }

    /// Strict module-access check for a call (Rust-style scoping). Validates that `base` is
    /// reachable under `module` (a `mod::base` path) or unqualified (`base` in an active module).
    /// A bare name not in any module is allowed through so dispatch can report "unknown function".
    fn resolve_call(&self, module: Option<&str>, base: &str, span: Span) -> Result<()> {
        match module {
            Some(m) => {
                if !is_module(m) {
                    return Err(NoiseError::runtime(
                        format!("unknown module '{m}' (known modules: {})", MODULES.join(", ")),
                        span,
                    ));
                }
                match module_of(base) {
                    Some(real) if real == m => Ok(()),
                    Some(real) => Err(NoiseError::runtime(
                        format!("'{base}' is in module '{real}', not '{m}'"),
                        span,
                    )),
                    None => Err(NoiseError::runtime(
                        format!("module '{m}' has no function '{base}'"),
                        span,
                    )),
                }
            }
            None => match module_of(base) {
                Some(m) if self.module_active(m) => Ok(()),
                Some(m) => Err(NoiseError::runtime(
                    format!("'{base}' is in module '{m}' — add `use {m};` or write `{m}::{base}`"),
                    span,
                )),
                // `stats` has no unqualified form to enable, so point at the path itself rather
                // than let the name fall through to "unknown function".
                None if STATS_FNS.contains(&base) => Err(NoiseError::runtime(
                    format!("'{base}' is in module 'stats' — write `stats::{base}(...)` (always qualified, like `plot::`)"),
                    span,
                )),
                None => Ok(()),
            },
        }
    }

    pub(super) fn eval_array(&mut self, elems: &[Spanned]) -> Result<Value> {
        let mut out = Vec::with_capacity(elems.len());
        for e in elems {
            let v = self.eval(e)?;
            forbid_continue(&v, e.span)?; // `[1, continue, 3]` is a misuse (finding F8)
            forbid_undrawn(&v, e.span)?;
            out.push(v);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// `a..b` — the half-open integer range `[a, b)` as an array of numbers (`0..n` has `n`
    /// elements `0 … n-1`). Bounds must be deterministic integers; `a >= b` yields the empty
    /// array. This is the syntax that replaced the old `range(a, b)` builtin.
    pub(super) fn eval_range(&mut self, lo: &Spanned, hi: &Spanned, span: Span) -> Result<Value> {
        let a = self.range_bound(lo)?;
        let b = self.range_bound(hi)?;
        if a.fract() != 0.0 || b.fract() != 0.0 {
            return Err(NoiseError::runtime(
                format!("range bounds must be integers, got {a}..{b}"),
                span,
            ));
        }
        // Compute the integer length up front — never iterate an `f64` counter. `0..1e12` would
        // otherwise allocate/iterate a trillion elements (OOM), and for `a >= 2^53` the old
        // `i += 1.0` is a no-op (float precision), so `while i < b` never advances → a *true*
        // infinite loop. `RANGE_MAX` caps the materialized length with a teaching error, in the
        // spirit of `CORR_MAX`: a range this large is a mistake, not a workload.
        let len = if b > a { b - a } else { 0.0 };
        if len > RANGE_MAX as f64 {
            return Err(NoiseError::runtime(
                format!(
                    "range {a}..{b} has {len} elements, over the {RANGE_MAX} cap — a range \
                     materializes every element as an array; use a smaller range"
                ),
                span,
            ));
        }
        let len = len as usize;
        let mut out = Vec::with_capacity(len);
        for k in 0..len {
            out.push(Value::Num(a + k as f64));
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Evaluate one range bound to a finite deterministic number (a `Dist`/recipe/array bound is
    /// an error — a range needs concrete endpoints).
    fn range_bound(&mut self, e: &Spanned) -> Result<f64> {
        let v = self.eval(e)?;
        forbid_undrawn(&v, e.span)?;
        match v {
            Value::Num(n) | Value::Est { val: n, .. } if n.is_finite() => Ok(n),
            other => Err(NoiseError::runtime(
                format!(
                    "range bound must be a finite number, got {}",
                    other.type_name()
                ),
                e.span,
            )),
        }
    }

    pub(super) fn eval_index(&mut self, arr: &Spanned, idx: &Spanned) -> Result<Value> {
        let av = self.eval(arr)?;
        let iv = self.eval(idx)?;
        let xs = match av {
            Value::Array(xs) => xs,
            other => {
                return Err(NoiseError::runtime(
                    format!("cannot index {} — not an array", other.type_name()),
                    arr.span,
                ))
            }
        };
        // A random-variable index is a per-lane *gather*: each sample selects its own element.
        if is_dist(&iv) {
            return self.gather(&xs, iv, arr.span, idx.span);
        }
        let i = self.array_index(&iv, idx.span)?;
        if i >= xs.len() {
            return Err(NoiseError::runtime(
                format!("array index {i} out of bounds (len {})", xs.len()),
                idx.span,
            ));
        }
        Ok(xs[i].clone())
    }

    /// Build a per-lane gather node for `xs[index]` where `index` is a random variable. Every
    /// element is lifted to an RV (constants fold into `ConstNum`/`ConstBool` nodes); they must be
    /// scalars of one kind — gathering a matrix row into a single lane is out of scope. At sample
    /// time each lane rounds its `index` to the nearest integer and **clamps** it into range (a
    /// permutation index is always valid, so the clamp only guards malformed inputs).
    fn gather(
        &mut self,
        xs: &[Value],
        index: Value,
        arr_span: Span,
        idx_span: Span,
    ) -> Result<Value> {
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                "cannot gather from an empty array".to_string(),
                arr_span,
            ));
        }
        let (index_id, index_kind) = self.operand_to_rv(index, idx_span)?;
        if index_kind != RvKind::Num {
            return Err(NoiseError::runtime(
                format!(
                    "array index must be a number, got {}",
                    index_kind.type_name()
                ),
                idx_span,
            ));
        }
        let mut elems = Vec::with_capacity(xs.len());
        let mut elem_kind: Option<RvKind> = None;
        for x in xs.iter() {
            let (id, k) = self.operand_to_rv(x.clone(), arr_span)?;
            match elem_kind {
                None => elem_kind = Some(k),
                Some(k0) if k0 != k => {
                    return Err(NoiseError::runtime(
                        "a gathered array must have elements of a single type".to_string(),
                        arr_span,
                    ))
                }
                _ => {}
            }
            elems.push(id);
        }
        let kind = elem_kind.expect("non-empty array has a kind");
        let id = self.graph.push(
            RvNode::Gather {
                elems: elems.into_boxed_slice(),
                index: index_id,
            },
            kind,
        );
        Ok(Value::Dist(id))
    }

    pub(super) fn eval_for(&mut self, var: &str, iter: &Spanned, body: &Spanned) -> Result<Value> {
        let iv = self.eval(iter)?;
        let xs = match iv {
            Value::Array(xs) => xs,
            other => {
                return Err(NoiseError::runtime(
                    format!(
                        "`for` expects an array to iterate, got {}",
                        other.type_name()
                    ),
                    iter.span,
                ))
            }
        };
        // Build-time unroll: bind the loop var in the *current* frame and run the body once per
        // element. Bindings leak (Noise blocks don't scope), which is exactly how accumulators
        // persist across iterations. Each `~` inside is a fresh node, giving independence.
        for x in xs.iter() {
            self.vars.insert(var.to_string(), x.clone());
            self.eval(body)?;
        }
        Ok(Value::Unit)
    }

    /// `[for var in iter { body }]` — a comprehension (PLAN-COMPLEX §8). Build-time unrolled exactly
    /// like a leaking `for`: bind `var` to each element in the *current* frame (so the body closes
    /// over outer variables — this is why a higher-order `map(xs, f)` is unnecessary and Noise needs
    /// no closures), evaluate `body`, and collect the results. A pure 1-to-1 map: the result always
    /// has `Len(iter)` elements.
    pub(super) fn eval_comprehension(
        &mut self,
        body: &Spanned,
        var: &str,
        iter: &Spanned,
    ) -> Result<Value> {
        let iv = self.eval(iter)?;
        let xs = match iv {
            Value::Array(xs) => xs,
            other => {
                return Err(NoiseError::runtime(
                    format!(
                        "a comprehension needs an array to iterate, got {}",
                        other.type_name()
                    ),
                    iter.span,
                ))
            }
        };
        let mut out = Vec::with_capacity(xs.len());
        for x in xs.iter() {
            self.vars.insert(var.to_string(), x.clone());
            let v = self.eval(body)?;
            // `continue` in the body omits this element — that's how a comprehension filters.
            if matches!(v, Value::Continue) {
                continue;
            }
            forbid_undrawn(&v, body.span)?;
            out.push(v);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Evaluate a user-function call: bind args to params in a *fresh* frame (params-only scope
    /// — functions are pure in their arguments and may call other functions, but do not capture
    /// outer variables), evaluate the body, then restore the caller's frame. A `~` function
    /// additionally draws its result, so each call is an independent draw.
    /// Evaluate a binding `name <bind> rhs`. Split out of [`eval`](Engine::eval) to keep that
    /// function's frame small (recursion-depth budget). Handles input name inference (§2).
    pub(super) fn eval_bind(&mut self, kind: BindKind, name: &str, rhs: &Spanned) -> Result<Value> {
        // Name inference (PLAN-INPUTS §2): when `input::…` is the *direct* RHS of a bind, the input's
        // name defaults to the binding's LHS identifier — so `dice_sides = input::real(min: 1, max:
        // 100)` needs no explicit `name:`. Only a direct RHS qualifies; a nested `input::` elsewhere
        // still requires its own `name:`.
        let v = match as_input_call(&rhs.expr) {
            Some((base, call_args)) => self.input_call(base, call_args, Some(name), rhs.span)?,
            None => self.eval(rhs)?,
        };
        // `x = continue` binds a loop-control sentinel into a data position (finding F8) — reject it
        // here, at the binding, instead of letting it surface later as "arithmetic on continue".
        forbid_continue(&v, rhs.span)?;
        // The core-model split (LANG.md §2): `~` is the *only* thing that draws.
        //   `~` on a recipe instantiates a FRESH random variable (a sample-DAG node); on a point
        //        mass / already-drawn value it binds as-is (a Dirac draw is just the constant).
        //   `=` binds the evaluated value verbatim — a recipe STAYS a recipe, so a later `~` on it
        //        draws an independent copy.
        let bound = match kind {
            BindKind::Sample => self.draw_if_recipe(v)?,
            BindKind::Assign => v,
        };
        self.vars.insert(name.to_string(), bound.clone());
        Ok(bound)
    }

    /// Evaluate a call expression `name(args)`. Split out of [`eval`](Engine::eval) to keep that
    /// function's stack frame small (the recursion-depth budget, `MAX_CALL_DEPTH`). Resolves
    /// `input::`/`plot::`/`stats::` namespaces, user functions, queries, introspection, and the
    /// builtin library — after reordering any named arguments into positional order (§2).
    pub(super) fn eval_call(
        &mut self,
        name: &str,
        call_args: &CallArgs,
        span: Span,
    ) -> Result<Value> {
        let (module, base) = split_path(name);
        // `input::*` — an inline host-tunable parameter (PLAN-INPUTS). It consumes its own named
        // arguments (the spec), so it is intercepted before the generic named→positional reorder. A
        // standalone call has no binding LHS to infer a name from (`None`); the `x = input::…` form
        // routes through `Expr::Bind` with the name inferred.
        if module == Some("input") {
            return self.input_call(base, call_args, None, span);
        }
        // Named arguments bind to parameters by name (PLAN-INPUTS §2). Reorder them into positional
        // order here — before evaluating anything — so the rest of dispatch runs on the exact
        // positional code path. Positional calls skip this entirely.
        let reordered;
        let args: &[Spanned] = match call_args {
            CallArgs::Positional(a) => a,
            CallArgs::Named(named) => {
                reordered = self.reorder_named_args(module, base, named, span)?;
                &reordered
            }
        };
        let mut arg_vals = Vec::with_capacity(args.len());
        for a in args {
            arg_vals.push(self.eval(a)?);
        }
        // Reject `continue` in any argument position (finding F8) — a separate pass so the hot,
        // deeply-recursive arg-eval loop above keeps its lean stack frame (the recursion budget).
        for (a, v) in args.iter().zip(&arg_vals) {
            forbid_continue(v, a.span)?; // `f(continue)` is a misuse
        }
        // A user function (unqualified) shadows a builtin of the same name. This is the *only*
        // recursive tail of `eval_call` (`f() = f()`), so it stays here — everything else routes
        // through the `#[inline(never)]` `dispatch_call`, which keeps this frame small (and off the
        // `MAX_CALL_DEPTH` recursion path) by not merging the builtin-dispatch locals into it.
        if module.is_none() {
            if let Some(f) = self.funcs.get(base).cloned() {
                return self.call_user_fn(base, &f, arg_vals, span);
            }
        }
        self.dispatch_call(module, base, args, arg_vals, span)
    }

    /// The non-user-function tail of [`eval_call`](Self::eval_call): `plot::`/`stats::` charts,
    /// conditioned queries, variable introspection, the `&mut`-needing library reducers, `Print`,
    /// and the pure `builtins::call`. Marked `#[inline(never)]` on purpose: it holds the widest
    /// locals in call dispatch, and keeping them in their own stack frame (rather than merged into
    /// the recursive `eval_call`) preserves the `MAX_CALL_DEPTH` budget on a small test stack.
    #[inline(never)]
    fn dispatch_call(
        &mut self,
        module: Option<&str>,
        base: &str,
        args: &[Spanned],
        arg_vals: Vec<Value>,
        span: Span,
    ) -> Result<Value> {
        // `plot::*` — the charting surface for examples. Computes a summary (reusing the
        // introspection core) and *captures* it like `Print`, returning unit.
        if module == Some("plot") {
            return self.plot_call(base, args, &arg_vals, span);
        }
        // `stats::*` — the same computations, handed back as numbers instead of a chart.
        if module == Some("stats") {
            return self.stats_call(base, &arg_vals, span);
        }
        if module.is_none() {
            // A query over a conditioned value — `P(a)` / `E(a)` / `Var(a)` / `Q(a, q)` where
            // `a` is `X | C`. Fuse the condition into `select(C, quantity, NaN)` here (needs
            // `&mut` graph) and hand the root to the conditional estimators.
            if matches!(base, "P" | "E" | "Var" | "Q")
                && matches!(arg_vals.first(), Some(Value::Cond { .. }))
            {
                return self.query_cond(base, &arg_vals, span);
            }
            // Variable introspection — `describe`/`hist`/`samples`/`corr`/`scatter`/`explain`.
            // Always-on builtins that build sampling roots (so they need `&mut` graph) and,
            // for `explain`, read the variable scope; routed here before module resolution.
            if is_introspection(base) {
                return self.introspect_call(base, args, &arg_vals, span);
            }
        }
        // Strict module scoping: a `rand`/`math`/`vec` name needs `use` or a `mod::` path.
        self.resolve_call(module, base, span)?;
        if let Some(result) = self.lib_call(base, &arg_vals, span) {
            // Library reducers/constructors build graph nodes and/or draw, so they need
            // `&mut self` — intercepted here before the pure-builtin fallback (§0).
            result
        } else if base == "Print" {
            // `Print` needs `&mut self` to append to the capture buffer, so it can't live
            // in the pure `builtins::call`. (A user `Print` would have been resolved above.)
            let line = arg_vals
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            self.emit(Output::Text(line));
            Ok(Value::Unit)
        } else {
            builtins::call(base, &arg_vals, &self.query_ctx(span))
        }
    }

    /// Reorder a **named** argument list into positional order for the callee (PLAN-INPUTS §2).
    /// Named arguments are supported for **user functions** only: each parameter must be named
    /// exactly once, an unknown name is an error, and a missing parameter is an error. The result
    /// is a fresh `Vec<Spanned>` in parameter order — dispatch then proceeds on the positional path.
    /// (`input::` intercepts its own named args before reaching here; every other builtin accepts
    /// positional arguments only.)
    fn reorder_named_args(
        &self,
        module: Option<&str>,
        base: &str,
        named: &[(String, Spanned)],
        span: Span,
    ) -> Result<Vec<Spanned>> {
        let params = match (module, self.funcs.get(base)) {
            (None, Some(f)) => f.params.clone(),
            _ => {
                let full = match module {
                    Some(m) => format!("{m}::{base}"),
                    None => base.to_string(),
                };
                return Err(NoiseError::runtime(
                    format!(
                        "`{full}` does not accept named arguments — only user-defined functions and \
                         `input::` take `name: value` arguments; call it positionally"
                    ),
                    span,
                ));
            }
        };
        let mut slots: Vec<Option<Spanned>> = vec![None; params.len()];
        for (arg_name, value) in named {
            let idx = params.iter().position(|p| p == arg_name).ok_or_else(|| {
                NoiseError::runtime(
                    format!("`{base}` has no parameter named `{arg_name}`"),
                    value.span,
                )
            })?;
            // A duplicate name is already rejected by the parser, but guard anyway.
            if slots[idx].is_some() {
                return Err(NoiseError::runtime(
                    format!("parameter `{arg_name}` of `{base}` bound more than once"),
                    value.span,
                ));
            }
            slots[idx] = Some(value.clone());
        }
        let mut ordered = Vec::with_capacity(params.len());
        for (p, slot) in params.iter().zip(slots) {
            match slot {
                Some(v) => ordered.push(v),
                None => {
                    return Err(NoiseError::runtime(
                        format!("missing argument `{p}` in named call to `{base}`"),
                        span,
                    ))
                }
            }
        }
        Ok(ordered)
    }

    /// Evaluate an `input::{real,int,bool}(…)` call (PLAN-INPUTS §1). Reads the spec from named
    /// arguments, resolves the current value (host override, else default — clamped/snapped), records
    /// the input in the run manifest, and emits an inline control. Returns the value as a point mass
    /// so downstream code reads it like any number. `inferred_name` is the binding LHS when the call
    /// is the direct RHS of `x = input::…` (name inference, §2); `None` for a standalone call.
    ///
    /// First evaluation of a given name registers + emits; a later mention of the same name returns
    /// the same value without re-emitting. Re-declaring a name with a *different* spec is an error.
    fn input_call(
        &mut self,
        base: &str,
        call_args: &CallArgs,
        inferred_name: Option<&str>,
        span: Span,
    ) -> Result<Value> {
        let kind = InputKind::from_base(base).ok_or_else(|| {
            NoiseError::runtime(
                format!("unknown input type `input::{base}` (want input::real / input::int / input::bool)"),
                span,
            )
        })?;

        // The spec arrives as named arguments; an empty argument list is allowed (`default` may be
        // inferred to be required below). A positional argument list is a usage error.
        let named: &[(String, Spanned)] = match call_args {
            CallArgs::Named(n) => n,
            CallArgs::Positional(p) if p.is_empty() => &[],
            CallArgs::Positional(_) => {
                return Err(NoiseError::runtime(
                    format!(
                        "input::{base} takes named arguments, e.g. \
                         `input::{base}(min: 1, max: 10, default: 5)`"
                    ),
                    span,
                ))
            }
        };

        // Collect the recognized spec fields; an unknown field name is an error.
        let mut name_field: Option<String> = None;
        let mut label: Option<String> = None;
        let mut min: Option<f64> = None;
        let mut max: Option<f64> = None;
        let mut step: Option<f64> = None;
        let mut default: Option<InputValue> = None;
        for (key, value_expr) in named {
            match key.as_str() {
                "name" => name_field = Some(self.eval_input_str(base, "name", value_expr)?),
                "label" => label = Some(self.eval_input_str(base, "label", value_expr)?),
                "min" => min = Some(self.eval_input_num(base, "min", value_expr)?),
                "max" => max = Some(self.eval_input_num(base, "max", value_expr)?),
                "step" => step = Some(self.eval_input_num(base, "step", value_expr)?),
                "default" => default = Some(self.eval_input_value(base, value_expr)?),
                other => {
                    return Err(NoiseError::runtime(
                        format!(
                            "input::{base} has no field `{other}` \
                             (fields: name, min, max, step, default, label)"
                        ),
                        value_expr.span,
                    ))
                }
            }
        }

        // The name: an explicit `name:` wins; otherwise the binding LHS (name inference). A
        // standalone `input::…` with neither is an error.
        let name = match (name_field, inferred_name) {
            (Some(n), _) => {
                if !crate::input::is_ident(&n) {
                    return Err(NoiseError::runtime(
                        format!("input name `{n}` is not a valid identifier"),
                        span,
                    ));
                }
                n
            }
            (None, Some(n)) => n.to_string(),
            (None, None) => {
                return Err(NoiseError::runtime(
                    format!(
                        "input::{base} needs a name — bind it (`x = input::{base}(…)`) or pass \
                         `name: \"x\"`"
                    ),
                    span,
                ))
            }
        };

        let default = default.ok_or_else(|| {
            NoiseError::runtime(format!("input `{name}` needs a `default`"), span)
        })?;

        let spec = InputSpec {
            name: name.clone(),
            kind,
            min,
            max,
            step,
            default,
            label,
        };
        spec.validate(span)?;

        // Dedup by name. A repeat with the same spec reuses the resolved value; a repeat with a
        // different spec is a redeclaration conflict.
        if let Some(existing) = self.input_manifest.iter().find(|r| r.spec.name == name) {
            if existing.spec != spec {
                return Err(NoiseError::runtime(
                    format!("input `{name}` redeclared with a different spec"),
                    span,
                ));
            }
            return Ok(input_value_to_value(existing.value));
        }

        let over = self
            .input_overrides
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v);
        let value = spec.resolve(over, span)?;
        self.emit(Output::Input {
            spec: spec.clone(),
            value,
        });
        self.input_manifest.push(ResolvedInput {
            spec,
            value,
            stmt_span: self.current_stmt_span,
        });
        Ok(input_value_to_value(value))
    }

    /// Evaluate an `input::` spec field expected to be a number (`min`/`max`/`step`).
    fn eval_input_num(&mut self, base: &str, field: &str, e: &Spanned) -> Result<f64> {
        match self.eval(e)? {
            Value::Num(n) => Ok(n),
            other => Err(NoiseError::runtime(
                format!(
                    "input::{base} field `{field}` must be a number, got {}",
                    other.type_name()
                ),
                e.span,
            )),
        }
    }

    /// Evaluate an `input::` spec field expected to be a string (`name`/`label`).
    fn eval_input_str(&mut self, base: &str, field: &str, e: &Spanned) -> Result<String> {
        match self.eval(e)? {
            Value::Str(s) => Ok(s),
            other => Err(NoiseError::runtime(
                format!(
                    "input::{base} field `{field}` must be a string, got {}",
                    other.type_name()
                ),
                e.span,
            )),
        }
    }

    /// Evaluate an `input::` `default` — a number or a bool (the two input value kinds).
    fn eval_input_value(&mut self, base: &str, e: &Spanned) -> Result<InputValue> {
        match self.eval(e)? {
            Value::Num(n) => Ok(InputValue::Num(n)),
            Value::Bool(b) => Ok(InputValue::Bool(b)),
            other => Err(NoiseError::runtime(
                format!(
                    "input::{base} `default` must be a number or bool, got {}",
                    other.type_name()
                ),
                e.span,
            )),
        }
    }
}
