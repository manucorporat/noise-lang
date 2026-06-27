//! Tree-walking evaluator for the deterministic core (PLAN.md Phase 1).
//!
//! This is the correctness reference. Phase 2 introduces the bytecode + batched sampler
//! for the *random-variable* hot path; deterministic evaluation can stay here.
//!
//! Scoping note: blocks currently evaluate in the enclosing environment (no new scope),
//! mirroring the legacy semantics where block-local bindings leak outward. Revisit when
//! user-defined functions land (Phase 3). Documented in LANG.md.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::ast::*;
use crate::builtins;
use crate::dist::{Recipe, RvGraph, RvId, RvKind, RvNode, Source, Uniform};
use crate::error::{NoiseError, Result, Span};
use crate::parser::parse;
use crate::sampler::{self, Moments};
use crate::signal::{NoiseKind, NoiseSpec, SigOp, SignalSpec};
use crate::value::Value;

/// A user-defined function (LANG.md §4). `Assign` is deterministic (`f(a)=…`); `Sample` draws a
/// fresh RV on each call (`f()~dist`). Stored behind `Rc` so a call can clone the handle out of
/// the function table without borrowing the engine for the duration of the call.
#[derive(Debug)]
struct UserFn {
    kind: BindKind,
    params: Vec<String>,
    body: Spanned,
}

/// Build-time call-depth limit. Noise unrolls calls/loops at build time, so calls must
/// terminate; this converts accidental infinite recursion into a clean error instead of a
/// process-aborting stack overflow. Kept conservative so even a small (2 MiB) thread stack
/// can't be blown before the limit trips — deep recursion isn't a target use (loops unroll).
const MAX_CALL_DEPTH: usize = 256;

pub struct Engine {
    vars: HashMap<String, Value>,
    /// User functions live in their own namespace (a call resolves here before builtins).
    funcs: HashMap<String, Rc<UserFn>>,
    /// Append-only sample-DAG arena (Phase 2). Built during `run`; read-only when sampling.
    graph: RvGraph,
    /// Current user-function call depth (guarded by `MAX_CALL_DEPTH`).
    call_depth: usize,
    /// Modules brought into unqualified scope via `use` (Rust-style). `builtin` is always active
    /// and is *not* stored here; `rand`/`math`/`vec` must be `use`d (or accessed as `mod::name`).
    used: HashSet<String>,
    /// Captured `Print` output (newline-terminated lines). The CLI/REPL drain and print it; the
    /// WASM playground reads it to show program output in the browser. Buffering (instead of a
    /// bare `println!`) keeps `Print` portable to `wasm32`, where stdout goes nowhere.
    output: String,
    /// Default Monte Carlo budget for `P`/`E`/`Var`/`Q` when a call carries no explicit sample
    /// count. Starts at [`builtins::P_DEFAULT_N`]; a program tunes it for the whole run with
    /// `engine::set_max_samples(N)`. An explicit per-call count (`P(event, n)`) still wins over this.
    max_samples: usize,
    /// Per-query operation budget (`engine::set_max_opts(N)`; `0` = unlimited). A forcing over a
    /// cone of `C` distinct nodes costs `n×C` per-lane ops, so each `P`/`E`/`Var`/`Q` query
    /// auto-clamps its draw count to `N/C` (never below 1). Bounds each query's work so a program's
    /// worst-case complexity stays deterministic regardless of cone size — without ever erroring.
    /// Defaults to [`builtins::MAX_OPS_DEFAULT`] (a built-in safety ceiling), not unlimited.
    max_opts: u64,
    /// Validate-only mode (set by [`Engine::check`]). When on, the sampling estimators
    /// (`P`/`E`/`Var`/`Q`) skip their Monte Carlo loop and return a neutral placeholder — the
    /// program is still parsed, evaluated, and graph-built (so type/shape/scope errors surface),
    /// but no draws happen, so a check finishes fast regardless of the configured sample budget.
    check_mode: bool,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            graph: RvGraph::default(),
            call_depth: 0,
            used: HashSet::new(),
            output: String::new(),
            max_samples: builtins::P_DEFAULT_N,
            max_opts: builtins::MAX_OPS_DEFAULT,
            check_mode: false,
        }
    }

    /// Take everything `Print` has emitted so far, clearing the buffer. The CLI/REPL call this
    /// after a run to flush to stdout; the WASM playground calls it to display program output.
    pub fn drain_output(&mut self) -> String {
        std::mem::take(&mut self.output)
    }

    /// Read-only access to the sample-DAG (tests assert it stays empty for deterministic
    /// programs).
    pub fn graph(&self) -> &RvGraph {
        &self.graph
    }

    /// Run-time counters (samples drawn, operations, random draws) accumulated since the last
    /// [`run`](Engine::run). The CLI/playground read this to show how much Monte-Carlo work the
    /// program triggered. See [`crate::stats`].
    pub fn stats(&self) -> crate::stats::RunStats {
        crate::stats::snapshot()
    }

    /// Parse and evaluate a whole program, returning the value of the last statement
    /// (or `Unit` for an empty program).
    pub fn run(&mut self, src: &str) -> Result<Value> {
        // Fresh run-time counters for this program (the playground reads them after `run`).
        crate::stats::reset();
        let program = parse(src)?;
        let mut last = Value::Unit;
        for stmt in &program.stmts {
            last = self.eval(stmt)?;
        }
        Ok(last)
    }

    /// Convenience alias of [`Engine::run`]: the last statement's value is the RV.
    /// Tests do `let rv = eng.run_rv("X ~ unif(-1,1); X ^ 2")?;`.
    pub fn run_rv(&mut self, src: &str) -> Result<Value> {
        self.run(src)
    }

    /// Validate a program without running its Monte Carlo: parse it, evaluate every statement, and
    /// build the sample-DAG — surfacing syntax, scope, type, and shape errors — but skip the actual
    /// sampling in `P`/`E`/`Var`/`Q` (see [`Engine::check_mode`]). Returns the last value (whose
    /// estimator results are placeholders) so callers can just check for `Ok`. Fast regardless of
    /// the configured sample budget.
    pub fn check(&mut self, src: &str) -> Result<Value> {
        self.check_mode = true;
        let result = self.run(src);
        self.check_mode = false;
        result
    }

    fn eval(&mut self, node: &Spanned) -> Result<Value> {
        match &node.expr {
            Expr::Number(n) => Ok(Value::Num(*n)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Ident(name) => self.eval_ident(name, node.span),
            Expr::Unary(op, rhs) => {
                let v = self.eval(rhs)?;
                forbid_recipe(&v, rhs.span)?;
                if matches!(v, Value::Complex { .. }) {
                    self.unary_complex(*op, v, node.span)
                } else if is_dist(&v) {
                    self.lift_unary(*op, v, node.span)
                } else {
                    eval_unary(*op, v, node.span) // deterministic fast path, unchanged
                }
            }
            Expr::Binary(op, l, r) => {
                let lv = self.eval(l)?;
                let rv = self.eval(r)?;
                forbid_recipe(&lv, l.span)?;
                forbid_recipe(&rv, r.span)?;
                self.binop(*op, lv, rv, node.span)
            }
            Expr::Bind(kind, name, rhs) => {
                let v = self.eval(rhs)?;
                // The core-model split (LANG.md §2): `~` is the *only* thing that draws.
                //   `~` on a recipe instantiates a FRESH random variable (a sample-DAG node);
                //        on a point mass / already-drawn value it binds as-is (a Dirac draw is
                //        just the constant — no new randomness, nothing to instantiate).
                //   `=` binds the evaluated value verbatim — crucially, a recipe STAYS a recipe,
                //        so a later `~` on it draws an independent copy.
                let bound = match kind {
                    BindKind::Sample => self.draw_if_recipe(v)?,
                    BindKind::Assign => v,
                };
                self.vars.insert(name.clone(), bound.clone());
                Ok(bound)
            }
            Expr::Sample { shape, body } => self.eval_sample(shape, body),
            Expr::MatMul(l, r) => self.eval_matmul(l, r, node.span),
            Expr::FnDef { kind, name, params, body } => {
                // Defining a function registers it (cloning the body out of the AST so it
                // outlives this `run`) and evaluates to unit.
                let f = Rc::new(UserFn {
                    kind: *kind,
                    params: params.clone(),
                    body: (**body).clone(),
                });
                self.funcs.insert(name.clone(), f);
                Ok(Value::Unit)
            }
            Expr::Block(stmts) => {
                let mut last = Value::Unit;
                for s in stmts {
                    last = self.eval(s)?;
                    // `continue` short-circuits the block: the remaining statements don't run, and
                    // the sentinel propagates up to the enclosing loop (see `eval_comprehension`).
                    if matches!(last, Value::Continue) {
                        return Ok(Value::Continue);
                    }
                }
                Ok(last)
            }
            Expr::If(cond, then_b, else_b) => {
                let c = self.eval(cond)?;
                forbid_recipe(&c, cond.span)?;
                match c {
                    // Deterministic condition: take exactly one branch (short-circuit).
                    Value::Bool(true) => self.eval(then_b),
                    Value::Bool(false) => match else_b {
                        Some(eb) => self.eval(eb),
                        None => Ok(Value::Unit),
                    },
                    // Random-variable condition: lift to a per-lane select. BOTH branches are
                    // evaluated (to build the select node), then blended lane-by-lane.
                    Value::Dist(cid) if self.graph.kind(cid) == RvKind::Bool => {
                        self.lift_if(cid, then_b, else_b.as_deref(), node.span)
                    }
                    Value::Dist(_) => Err(NoiseError::runtime(
                        "if condition is a dist<number>, expected an event (bool)".to_string(),
                        cond.span,
                    )),
                    other => Err(NoiseError::runtime(
                        format!("if condition must be a bool, got {}", other.type_name()),
                        cond.span,
                    )),
                }
            }
            Expr::Call(name, args) => {
                let mut arg_vals = Vec::with_capacity(args.len());
                for a in args {
                    arg_vals.push(self.eval(a)?);
                }
                let (module, base) = split_path(name);
                // A user function (unqualified) shadows a builtin of the same name.
                if module.is_none() {
                    if let Some(f) = self.funcs.get(base).cloned() {
                        return self.call_user_fn(base, &f, arg_vals, node.span);
                    }
                }
                // Strict module scoping: a `rand`/`math`/`vec` name needs `use` or a `mod::` path.
                self.resolve_call(module, base, node.span)?;
                if let Some(result) = self.lib_call(base, &arg_vals, node.span) {
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
                    self.output.push_str(&line);
                    self.output.push('\n');
                    Ok(Value::Unit)
                } else {
                    builtins::call(base, &arg_vals, &self.graph, self.max_samples, self.max_opts, node.span, self.check_mode)
                }
            }
            // Extracted into methods to keep `eval`'s stack frame small (recursion-depth budget).
            Expr::Array(elems) => self.eval_array(elems),
            Expr::Comprehension { body, var, iter } => self.eval_comprehension(body, var, iter),
            Expr::Range(lo, hi) => self.eval_range(lo, hi, node.span),
            Expr::Index(arr, idx) => self.eval_index(arr, idx),
            Expr::For { var, iter, body } => self.eval_for(var, iter, body),
            Expr::Use(module) => {
                if !is_module(module) {
                    return Err(NoiseError::runtime(
                        format!("unknown module '{module}' (known modules: rand, math, vec, signal, engine, builtin)"),
                        node.span,
                    ));
                }
                // `use builtin;` is a harmless no-op (it's always active).
                self.used.insert(module.clone());
                Ok(Value::Unit)
            }
            // The `continue` control sentinel — short-circuits the enclosing block / loop body.
            Expr::Continue => Ok(Value::Continue),
        }
    }

    /// Resolve an identifier — a variable, or a constant (`pi`/`e`) gated by module scope. A
    /// qualified `math::pi` always resolves; a bare `pi` needs `math` active (`use math;`).
    fn eval_ident(&self, name: &str, span: Span) -> Result<Value> {
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
                Err(NoiseError::runtime(format!("undefined variable '{name}'"), span))
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
                        format!("unknown module '{m}' (known modules: rand, math, vec, signal, engine, builtin)"),
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
                None => Ok(()),
            },
        }
    }

    fn eval_array(&mut self, elems: &[Spanned]) -> Result<Value> {
        let mut out = Vec::with_capacity(elems.len());
        for e in elems {
            let v = self.eval(e)?;
            forbid_recipe(&v, e.span)?;
            out.push(v);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// `a..b` — the half-open integer range `[a, b)` as an array of numbers (`0..n` has `n`
    /// elements `0 … n-1`). Bounds must be deterministic integers; `a >= b` yields the empty
    /// array. This is the syntax that replaced the old `range(a, b)` builtin.
    fn eval_range(&mut self, lo: &Spanned, hi: &Spanned, span: Span) -> Result<Value> {
        let a = self.range_bound(lo)?;
        let b = self.range_bound(hi)?;
        if a.fract() != 0.0 || b.fract() != 0.0 {
            return Err(NoiseError::runtime(
                format!("range bounds must be integers, got {a}..{b}"),
                span,
            ));
        }
        let mut out = Vec::new();
        let mut i = a;
        while i < b {
            out.push(Value::Num(i));
            i += 1.0;
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Evaluate one range bound to a finite deterministic number (a `Dist`/recipe/array bound is
    /// an error — a range needs concrete endpoints).
    fn range_bound(&mut self, e: &Spanned) -> Result<f64> {
        let v = self.eval(e)?;
        forbid_recipe(&v, e.span)?;
        match v {
            Value::Num(n) | Value::Est { val: n, .. } if n.is_finite() => Ok(n),
            other => Err(NoiseError::runtime(
                format!("range bound must be a finite number, got {}", other.type_name()),
                e.span,
            )),
        }
    }

    fn eval_index(&mut self, arr: &Spanned, idx: &Spanned) -> Result<Value> {
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
    fn gather(&mut self, xs: &[Value], index: Value, arr_span: Span, idx_span: Span) -> Result<Value> {
        if xs.is_empty() {
            return Err(NoiseError::runtime(
                "cannot gather from an empty array".to_string(),
                arr_span,
            ));
        }
        let (index_id, index_kind) = self.operand_to_rv(index, idx_span)?;
        if index_kind != RvKind::Num {
            return Err(NoiseError::runtime(
                format!("array index must be a number, got {}", index_kind.type_name()),
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
        let id = self
            .graph
            .push(RvNode::Gather { elems: elems.into_boxed_slice(), index: index_id }, kind);
        Ok(Value::Dist(id))
    }

    fn eval_for(&mut self, var: &str, iter: &Spanned, body: &Spanned) -> Result<Value> {
        let iv = self.eval(iter)?;
        let xs = match iv {
            Value::Array(xs) => xs,
            other => {
                return Err(NoiseError::runtime(
                    format!("`for` expects an array to iterate, got {}", other.type_name()),
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
    fn eval_comprehension(&mut self, body: &Spanned, var: &str, iter: &Spanned) -> Result<Value> {
        let iv = self.eval(iter)?;
        let xs = match iv {
            Value::Array(xs) => xs,
            other => {
                return Err(NoiseError::runtime(
                    format!("a comprehension needs an array to iterate, got {}", other.type_name()),
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
            forbid_recipe(&v, body.span)?;
            out.push(v);
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Evaluate a user-function call: bind args to params in a *fresh* frame (params-only scope
    /// — functions are pure in their arguments and may call other functions, but do not capture
    /// outer variables), evaluate the body, then restore the caller's frame. A `~` function
    /// additionally draws its result, so each call is an independent draw.
    fn call_user_fn(
        &mut self,
        name: &str,
        f: &UserFn,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value> {
        if args.len() != f.params.len() {
            return Err(NoiseError::runtime(
                format!("{name} expects {} argument(s), got {}", f.params.len(), args.len()),
                span,
            ));
        }
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(NoiseError::runtime(
                format!(
                    "call stack too deep (limit {MAX_CALL_DEPTH}) calling '{name}' — Noise \
                     unrolls calls at build time, so recursion must terminate"
                ),
                span,
            ));
        }
        self.call_depth += 1;
        // Swap in a fresh frame holding only the parameters; restore on the way out.
        let mut frame = HashMap::with_capacity(f.params.len());
        for (p, a) in f.params.iter().zip(args) {
            frame.insert(p.clone(), a);
        }
        let saved = std::mem::replace(&mut self.vars, frame);
        let result = self.eval(&f.body);
        self.vars = saved;
        self.call_depth -= 1;
        // A stochastic (`~`) function draws on each call (recipe → fresh RV); a deterministic
        // (`=`) function returns its body value verbatim.
        match f.kind {
            BindKind::Sample => self.draw_if_recipe(result?),
            BindKind::Assign => result,
        }
    }

    /// `~` semantics in one place: a recipe is drawn into a fresh RV; anything else (a point
    /// mass, an already-drawn RV) binds as-is, since there is nothing new to draw. Fallible because
    /// a structured recipe (`rotation`) builds a whole matrix and could surface a shape error;
    /// scalar recipe draws never fail.
    fn draw_if_recipe(&mut self, v: Value) -> Result<Value> {
        match v {
            Value::Recipe(r) => self.draw(r),
            other => Ok(other),
        }
    }

    /// The prefix draw operator `~[shape]? body` (LANG.md §2). Evaluate the operand once to a
    /// recipe (or any value), then materialize: a bare `~` draws a single sample; a shape draws a
    /// nested array with an *independent* draw at every leaf. Kept out of the `eval` match so that
    /// arm's locals don't inflate the (deeply recursive) `eval` stack frame.
    fn eval_sample(&mut self, shape: &[Spanned], body: &Spanned) -> Result<Value> {
        let v = self.eval(body)?;
        if shape.is_empty() {
            return self.draw_if_recipe(v);
        }
        let mut dims = Vec::with_capacity(shape.len());
        for dim in shape {
            let dv = self.eval(dim)?;
            dims.push(self.count_arg("~", &dv, dim.span)?);
        }
        self.draw_shaped(&dims, &v)
    }

    /// Build a nested array of the given shape, drawing the recipe independently at every leaf
    /// (`draw_if_recipe` instantiates fresh source nodes each call, so the leaves are iid). A
    /// non-recipe operand is repeated as-is. Backs the shaped prefix draw `~[n, m, …] recipe`.
    fn draw_shaped(&mut self, dims: &[usize], recipe: &Value) -> Result<Value> {
        let (n, rest) = (dims[0], &dims[1..]);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if rest.is_empty() {
                out.push(self.draw_if_recipe(recipe.clone())?);
            } else {
                out.push(self.draw_shaped(rest, recipe)?);
            }
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Draw a fresh random variable from a recipe — the *only* place sampling-DAG source nodes
    /// are created (LANG.md §2: `~` is the only thing that draws). Each call instantiates new
    /// source node(s), so two `~` on the same recipe are independent. The scalar recipes return a
    /// `Value::Dist`; the structured `rotation` recipe returns a `Value::Array` (a matrix of RVs).
    fn draw(&mut self, r: Recipe) -> Result<Value> {
        // The multivariate recipes: drawing them builds a whole array of correlated draws.
        if let Recipe::Rotation { d } = r {
            return self.draw_rotation(d);
        }
        if let Recipe::Permutation { n } = r {
            return self.draw_permutation(n);
        }
        // A complex draw yields a `Value::Complex` (two independent real channels), not a scalar id.
        if let Recipe::NormalComplex { sigma } = r {
            let s = sigma / std::f64::consts::SQRT_2;
            let re = self.graph.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: s }), RvKind::Num);
            let im = self.graph.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: s }), RvKind::Num);
            return Ok(Value::complex(Value::Dist(re), Value::Dist(im)));
        }
        let id = match r {
            Recipe::Uniform { lo, hi } => {
                self.graph.push(RvNode::Src(Source::Uniform(Uniform { lo, hi })), RvKind::Num)
            }
            Recipe::UniformInt { lo, hi } => {
                self.graph.push(RvNode::Src(Source::UniformInt { lo, hi }), RvKind::Num)
            }
            Recipe::Normal { mu, sigma } => {
                self.graph.push(RvNode::Src(Source::Normal { mu, sigma }), RvKind::Num)
            }
            Recipe::Exp { rate } => {
                self.graph.push(RvNode::Src(Source::Exp { rate }), RvKind::Num)
            }
            Recipe::Poisson { lambda } => {
                self.graph.push(RvNode::Src(Source::Poisson { lambda }), RvKind::Num)
            }
            Recipe::Geometric { p } => {
                self.graph.push(RvNode::Src(Source::Geometric { p }), RvKind::Num)
            }
            // The `_int` family draws a continuous source then rounds each lane to an integer.
            Recipe::NormalInt { mu, sigma } => {
                let z = self.graph.push(RvNode::Src(Source::Normal { mu, sigma }), RvKind::Num);
                self.graph.push(RvNode::Unary(UnOp::Round, z), RvKind::Num)
            }
            Recipe::ExpInt { rate } => {
                let z = self.graph.push(RvNode::Src(Source::Exp { rate }), RvKind::Num);
                self.graph.push(RvNode::Unary(UnOp::Round, z), RvKind::Num)
            }
            Recipe::Bernoulli { p } => {
                // bernoulli(p) ≡ (unif(0,1) < p): a bool-RV that is 1 with probability p.
                let u = self
                    .graph
                    .push(RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })), RvKind::Num);
                let c = self.graph.push(RvNode::ConstNum(p), RvKind::Num);
                self.graph.push(RvNode::Binary(BinOp::Lt, u, c), RvKind::Bool)
            }
            // Handled above with an early return (they yield arrays/complex, not a scalar `id`).
            Recipe::Rotation { .. } => unreachable!("rotation drawn via draw_rotation"),
            Recipe::Permutation { .. } => unreachable!("permutation drawn via draw_permutation"),
            Recipe::NormalComplex { .. } => unreachable!("normal_complex drawn via the complex path"),
        };
        Ok(Value::Dist(id))
    }

    /// Lift a unary op over a random variable. The operand is a `Value::Dist` (the caller's
    /// pre-check guarantees it). Type-checked by `RvKind` with spanned errors before sampling.
    fn lift_unary(&mut self, op: UnOp, v: Value, span: Span) -> Result<Value> {
        let id = match v {
            Value::Dist(id) => id,
            _ => unreachable!("lift_unary only reached with a Dist operand"),
        };
        let kind = self.graph.kind(id);
        let result_kind = match op {
            UnOp::Neg => {
                if kind != RvKind::Num {
                    return Err(NoiseError::runtime(
                        format!("cannot apply Neg to {}", kind.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
            UnOp::Not => {
                if kind != RvKind::Bool {
                    return Err(NoiseError::runtime(
                        format!("cannot apply Not to {}", kind.type_name()),
                        span,
                    ));
                }
                RvKind::Bool
            }
            // Math ufuncs need a numeric RV and yield a numeric RV.
            UnOp::Sin | UnOp::Cos | UnOp::Atan | UnOp::Sign | UnOp::Round | UnOp::Floor
            | UnOp::Ceil => {
                if kind != RvKind::Num {
                    return Err(NoiseError::runtime(
                        format!("cannot apply {} to {}", unop_name(op), kind.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
        };
        Ok(Value::Dist(self.graph.push(RvNode::Unary(op, id), result_kind)))
    }

    /// Lift a binary op over random variables. At least one operand is a `Value::Dist`;
    /// deterministic operands are folded into `ConstNum`/`ConstBool` graph nodes. Type rules
    /// mirror the deterministic evaluator, on `RvKind`, with spanned errors before sampling.
    fn lift_binary(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        use BinOp::*;
        let (lid, lk) = self.operand_to_rv(l, span)?;
        let (rid, rk) = self.operand_to_rv(r, span)?;
        let result_kind = match op {
            Add | Sub | Mul | Div | Mod | Pow => {
                if lk != RvKind::Num || rk != RvKind::Num {
                    return Err(NoiseError::runtime(
                        format!("arithmetic on {} and {}", lk.type_name(), rk.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
            Lt | Gt | Le | Ge => {
                if lk != RvKind::Num || rk != RvKind::Num {
                    return Err(NoiseError::runtime(
                        format!("cannot compare {} and {}", lk.type_name(), rk.type_name()),
                        span,
                    ));
                }
                RvKind::Bool
            }
            Eq | Ne => {
                if lk != rk {
                    return Err(NoiseError::runtime(
                        format!("cannot compare {} and {}", lk.type_name(), rk.type_name()),
                        span,
                    ));
                }
                RvKind::Bool
            }
            And | Or => {
                if lk != RvKind::Bool || rk != RvKind::Bool {
                    return Err(NoiseError::runtime(
                        format!(
                            "logical operator needs two bool events, got {} and {}",
                            lk.type_name(),
                            rk.type_name()
                        ),
                        span,
                    ));
                }
                RvKind::Bool
            }
        };
        Ok(Value::Dist(self.graph.push(RvNode::Binary(op, lid, rid), result_kind)))
    }

    /// Lift `if cond { then } else { else }` where `cond` is a bool random variable. Builds a
    /// per-lane `Select` RV: `cond ? then : else`. An `else` branch is REQUIRED (every lane
    /// needs a value), and the two branches must have the same kind.
    fn lift_if(
        &mut self,
        cond: RvId,
        then_b: &Spanned,
        else_b: Option<&Spanned>,
        span: Span,
    ) -> Result<Value> {
        let else_b = else_b.ok_or_else(|| {
            NoiseError::runtime(
                "an `if` over a random variable needs an `else` branch (every sample needs a value)"
                    .to_string(),
                span,
            )
        })?;
        // Both branches are evaluated: a lifted `if` is a value-select, not control flow.
        let then_v = self.eval(then_b)?;
        let else_v = self.eval(else_b)?;
        let (a, ak) = self.operand_to_rv(then_v, then_b.span)?;
        let (b, bk) = self.operand_to_rv(else_v, else_b.span)?;
        if ak != bk {
            return Err(NoiseError::runtime(
                format!(
                    "`if` branches must have the same type, got {} and {}",
                    ak.type_name(),
                    bk.type_name()
                ),
                span,
            ));
        }
        Ok(Value::Dist(self.graph.push(RvNode::Select { cond, a, b }, ak)))
    }

    /// Coerce an operand to an `(RvId, RvKind)`. `Dist` reuses its id (structural sharing);
    /// `Num`/`Bool` fold into a const node; `Str`/`Unit` are spanned errors (preserving the
    /// deterministic type-error contract, e.g. for `X + "a"`).
    fn operand_to_rv(&mut self, v: Value, span: Span) -> Result<(RvId, RvKind)> {
        match v {
            Value::Dist(id) => Ok((id, self.graph.kind(id))),
            Value::Num(n) => Ok((self.graph.push(RvNode::ConstNum(n), RvKind::Num), RvKind::Num)),
            // An estimate folds in as its central value (its error is dropped inside the RV).
            Value::Est { val, .. } => {
                Ok((self.graph.push(RvNode::ConstNum(val), RvKind::Num), RvKind::Num))
            }
            Value::Bool(b) => {
                Ok((self.graph.push(RvNode::ConstBool(b), RvKind::Bool), RvKind::Bool))
            }
            other => Err(NoiseError::runtime(
                format!("cannot use {} in a random-variable expression", other.type_name()),
                span,
            )),
        }
    }

    /// Combine two element `Value`s with a binary op — the single fold primitive the library
    /// reuses (LANG.md §0). Lifts to a graph node if either side is a `Dist`, else folds
    /// deterministically. Recipes are rejected (you can't operate on an undrawn distribution).
    fn binop(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        forbid_recipe(&l, span)?;
        forbid_recipe(&r, span)?;
        // A lazy signal defers scalar/trig ops and materializes against a sized array. Handle it
        // before the array path so `signal ⊕ array` adopts the array's length.
        if matches!(l, Value::Signal(_)) || matches!(r, Value::Signal(_)) {
            return self.binop_signal(op, l, r, span);
        }
        // Lazy white noise materializes against a sized array (into iid normal RV nodes), one
        // independent draw per leaf-vector lane. Handle it before the array path, like a signal.
        if matches!(l, Value::Noise(_)) || matches!(r, Value::Noise(_)) {
            return self.binop_noise(op, l, r, span);
        }
        // Arrays broadcast elementwise (NumPy-style): array⊕array (length-matched) and
        // array⊕scalar both map the op over the elements — so `signal + noise`, `1 + m`, and
        // `phase / kf` all work on whole signals. (A complex scalar meeting an array broadcasts
        // here, recursing into `binop` per element where the complex path below handles it.)
        if matches!(l, Value::Array(_)) || matches!(r, Value::Array(_)) {
            return self.binop_broadcast(op, l, r, span);
        }
        // A complex operand (either a constant `2 + 3i` or a complex RV) routes through the
        // complex arithmetic path: `* / ^` are true complex operations, a real operand promotes
        // to `re + 0i`, and ordering (`< > <= >=`) is a type error (no total order on ℂ).
        if matches!(l, Value::Complex { .. }) || matches!(r, Value::Complex { .. }) {
            return self.binop_complex(op, l, r, span);
        }
        // Algebraic identity folds before lifting: `0*x → 0`, `1*x → x`, `x+0/0+x → x`, `x-0 → x`.
        // These keep an RV out of the graph where it provably doesn't matter — and, crucially, let
        // `math::i * x` keep a *literal* `0` real channel (`0*x`), so a complex `exp` over a random
        // angle (`e^{i·X}`) still sees a constant real part. Only fires when the non-constant side is
        // numeric, so `0 * bool_event` still type-errors rather than silently folding to 0.
        if let Some(folded) = self.fold_identity(op, &l, &r) {
            return Ok(folded);
        }
        if is_dist(&l) || is_dist(&r) {
            self.lift_binary(op, l, r, span)
        } else {
            eval_binary(op, l, r, span)
        }
    }

    /// Whether `v` is a `dist<number>` — the survivor guard for [`Self::fold_identity`]. The fold
    /// fires *only* on the RV path: for two constants, `eval_binary` already evaluates the identity
    /// IEEE-honestly (e.g. `0 * inf == NaN`), so there is nothing to fold and nothing to get wrong.
    /// Restricting to a numeric RV also keeps `0 * event` (a `dist<bool>`) a clean type error.
    fn is_num_dist(&self, v: &Value) -> bool {
        matches!(v, Value::Dist(id) if self.graph.kind(*id) == RvKind::Num)
    }

    /// Fold the arithmetic identities `1*x → x`, `x+0/0+x → x`, `x-0 → x`, and `0*x → 0` when the
    /// surviving operand is a numeric RV (`dist<number>`) and the other side is the literal `0`/`1`.
    /// This keeps a provably-irrelevant RV out of the graph — and, crucially, lets `math::i * x`
    /// keep a *literal* `0` real channel (`0*x`), so a complex `exp` over a random angle (`e^{i·X}`)
    /// still sees a constant real part. (`0*x → 0` discards `x`'s inf/NaN propagation, but only for a
    /// measure-zero set of draws; for constant operands the fold never fires.) `None` falls through.
    fn fold_identity(&self, op: BinOp, l: &Value, r: &Value) -> Option<Value> {
        let is0 = |v: &Value| matches!(v, Value::Num(n) if *n == 0.0);
        let is1 = |v: &Value| matches!(v, Value::Num(n) if *n == 1.0);
        match op {
            BinOp::Mul => {
                if (is0(l) && self.is_num_dist(r)) || (is0(r) && self.is_num_dist(l)) {
                    Some(Value::Num(0.0))
                } else if is1(l) && self.is_num_dist(r) {
                    Some(r.clone())
                } else if is1(r) && self.is_num_dist(l) {
                    Some(l.clone())
                } else {
                    None
                }
            }
            BinOp::Add => {
                if is0(l) && self.is_num_dist(r) {
                    Some(r.clone())
                } else if is0(r) && self.is_num_dist(l) {
                    Some(l.clone())
                } else {
                    None
                }
            }
            BinOp::Sub if is0(r) && self.is_num_dist(l) => Some(l.clone()),
            _ => None,
        }
    }

    /// Elementwise broadcast of a binary op when at least one operand is an array.
    fn binop_broadcast(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        let out = match (l, r) {
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Err(length_mismatch("elementwise op", a.len(), b.len(), span));
                }
                let mut out = Vec::with_capacity(a.len());
                for (ai, bi) in a.iter().zip(b.iter()) {
                    out.push(self.binop(op, ai.clone(), bi.clone(), span)?);
                }
                out
            }
            (Value::Array(a), scalar) => {
                let mut out = Vec::with_capacity(a.len());
                for ai in a.iter() {
                    out.push(self.binop(op, ai.clone(), scalar.clone(), span)?);
                }
                out
            }
            (scalar, Value::Array(b)) => {
                let mut out = Vec::with_capacity(b.len());
                for bi in b.iter() {
                    out.push(self.binop(op, scalar.clone(), bi.clone(), span)?);
                }
                out
            }
            _ => unreachable!("binop_broadcast called without an array operand"),
        };
        Ok(Value::Array(Rc::new(out)))
    }

    /// Combine a lazy signal with another operand. A scalar **defers** (the op joins the signal's
    /// pipeline, staying O(1) memory); a sized **array materializes** the signal to that length and
    /// then broadcasts. Two bare signals (no length) is an error — `sample` one first.
    fn binop_signal(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        match (l, r) {
            (Value::Signal(s), rhs) if scalar_const(&rhs).is_some() => {
                let c = scalar_const(&rhs).unwrap();
                Ok(Value::Signal(Rc::new(s.push(SigOp::Scalar { op, c, scalar_on_left: false }))))
            }
            (lhs, Value::Signal(s)) if scalar_const(&lhs).is_some() => {
                let c = scalar_const(&lhs).unwrap();
                Ok(Value::Signal(Rc::new(s.push(SigOp::Scalar { op, c, scalar_on_left: true }))))
            }
            // A sized array fixes the sample count: materialize the signal, then broadcast.
            (Value::Signal(s), arr @ Value::Array(_)) => {
                let n = array_len(&arr);
                let mat = materialize_signal(&s, n);
                self.binop(op, mat, arr, span)
            }
            (arr @ Value::Array(_), Value::Signal(s)) => {
                let n = array_len(&arr);
                let mat = materialize_signal(&s, n);
                self.binop(op, arr, mat, span)
            }
            (Value::Signal(_), Value::Signal(_)) => Err(NoiseError::runtime(
                "cannot combine two lazy signals without a sample length — `signal::sample(s, n)` one first"
                    .to_string(),
                span,
            )),
            (l, r) => {
                let other = if matches!(l, Value::Signal(_)) { r } else { l };
                Err(NoiseError::runtime(
                    format!(
                        "cannot combine a signal with {} — `signal::sample(sig, n)` it to an array first",
                        other.type_name()
                    ),
                    span,
                ))
            }
        }
    }

    /// Combine lazy white noise with another operand. Only a **sized array** gives it a length:
    /// the noise materializes into a matching shape of fresh iid normal RV nodes (independent per
    /// leaf lane — so I/Q get distinct noise), then broadcasts. A scalar/signal/another-noise has
    /// no length — `signal::sample(noise_white(s), n)` it first.
    fn binop_noise(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        match (l, r) {
            (Value::Noise(spec), arr @ Value::Array(_)) => {
                let mat = self.materialize_noise_like(spec, &arr, span)?;
                self.binop(op, mat, arr, span)
            }
            (arr @ Value::Array(_), Value::Noise(spec)) => {
                let mat = self.materialize_noise_like(spec, &arr, span)?;
                self.binop(op, arr, mat, span)
            }
            (l, r) => {
                let other = if matches!(l, Value::Noise(_)) { r } else { l };
                Err(NoiseError::runtime(
                    format!(
                        "noise needs a sized context, but met {} — add it to a sized signal/array, \
                         or `signal::sample(noise_*(…), n)` it first",
                        other.type_name()
                    ),
                    span,
                ))
            }
        }
    }

    /// Combine two operands where at least one is **complex** (PLAN-COMPLEX §3). Every complex op
    /// is expressed as real ops on the two channels via [`Self::binop`], so it folds for constant
    /// complex and lifts into the (real) sample-DAG when a channel is an RV — no complex value
    /// flows through the VM. A real operand promotes to `re + 0i`; ordering and logical ops are a
    /// type error (ℂ has no total order, and a complex value is not an event).
    fn binop_complex(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        use BinOp::*;
        match op {
            Add | Sub => {
                let (lr, li) = self.complex_parts(l, span)?;
                let (rr, ri) = self.complex_parts(r, span)?;
                let re = self.binop(op, lr, rr, span)?;
                let im = self.binop(op, li, ri, span)?;
                Ok(Value::complex(re, im))
            }
            Mul => {
                // (a+bi)(c+di) = (ac − bd) + (ad + bc) i
                let (a, b) = self.complex_parts(l, span)?;
                let (c, d) = self.complex_parts(r, span)?;
                let ac = self.binop(Mul, a.clone(), c.clone(), span)?;
                let bd = self.binop(Mul, b.clone(), d.clone(), span)?;
                let ad = self.binop(Mul, a, d, span)?;
                let bc = self.binop(Mul, b, c, span)?;
                let re = self.binop(Sub, ac, bd, span)?;
                let im = self.binop(Add, ad, bc, span)?;
                Ok(Value::complex(re, im))
            }
            Div => {
                // (a+bi)/(c+di) = [(ac + bd) + (bc − ad) i] / (c² + d²)
                let (a, b) = self.complex_parts(l, span)?;
                let (c, d) = self.complex_parts(r, span)?;
                let cc = self.binop(Mul, c.clone(), c.clone(), span)?;
                let dd = self.binop(Mul, d.clone(), d.clone(), span)?;
                let denom = self.binop(Add, cc, dd, span)?;
                let ac = self.binop(Mul, a.clone(), c.clone(), span)?;
                let bd = self.binop(Mul, b.clone(), d.clone(), span)?;
                let bc = self.binop(Mul, b, c, span)?;
                let ad = self.binop(Mul, a, d, span)?;
                let re_num = self.binop(Add, ac, bd, span)?;
                let im_num = self.binop(Sub, bc, ad, span)?;
                let re = self.binop(Div, re_num, denom.clone(), span)?;
                let im = self.binop(Div, im_num, denom, span)?;
                Ok(Value::complex(re, im))
            }
            Pow => self.complex_pow(l, r, span),
            // Exact (re, im) comparison: equal iff both channels are equal.
            Eq => {
                let (lr, li) = self.complex_parts(l, span)?;
                let (rr, ri) = self.complex_parts(r, span)?;
                let re_eq = self.binop(Eq, lr, rr, span)?;
                let im_eq = self.binop(Eq, li, ri, span)?;
                self.binop(And, re_eq, im_eq, span)
            }
            Ne => {
                let (lr, li) = self.complex_parts(l, span)?;
                let (rr, ri) = self.complex_parts(r, span)?;
                let re_ne = self.binop(Ne, lr, rr, span)?;
                let im_ne = self.binop(Ne, li, ri, span)?;
                self.binop(Or, re_ne, im_ne, span)
            }
            Lt | Gt | Le | Ge => Err(NoiseError::runtime(
                "complex numbers have no ordering (ℂ is not totally ordered) — compare `math::abs(z)` if you mean magnitude".to_string(),
                span,
            )),
            Mod => Err(NoiseError::runtime(
                "modulo `%` is real-only — it has no meaning on a complex number".to_string(),
                span,
            )),
            And | Or => Err(NoiseError::runtime(
                "logical operator needs two bool events, got complex".to_string(),
                span,
            )),
        }
    }

    /// `z ^ k` for a complex base and a **constant integer** exponent: repeated complex multiply
    /// (negative `k` reciprocates). Enough for the QFT/quantum path; a general complex exponent is
    /// deferred (a clear error). The exponent magnitude is capped so a stray `z ^ 1e9` can't build
    /// an unbounded graph at build time.
    fn complex_pow(&mut self, base: Value, exp: Value, span: Span) -> Result<Value> {
        let k = match scalar_const(&exp) {
            Some(k) if k.fract() == 0.0 && k.is_finite() => k,
            _ => {
                return Err(NoiseError::runtime(
                    "complex `^` needs a constant integer exponent (e.g. `z ^ 2`); a general complex power is not supported".to_string(),
                    span,
                ))
            }
        };
        if k.abs() > 4096.0 {
            return Err(NoiseError::runtime(
                format!("complex `^` exponent {k} is too large (max magnitude 4096)"),
                span,
            ));
        }
        let n = k.abs() as u32;
        let mut acc = Value::cnum(1.0, 0.0);
        for _ in 0..n {
            acc = self.binop_complex(BinOp::Mul, acc, base.clone(), span)?;
        }
        if k < 0.0 {
            acc = self.binop_complex(BinOp::Div, Value::cnum(1.0, 0.0), acc, span)?;
        }
        Ok(acc)
    }

    /// Prefix unary op on a **complex** value. Only `-` (negate both channels) is defined; `!`
    /// (logical not) needs a bool. Negation goes through `binop` so it lifts when a channel is an RV.
    fn unary_complex(&mut self, op: UnOp, v: Value, span: Span) -> Result<Value> {
        let (re, im) = self.complex_parts(v, span)?;
        match op {
            UnOp::Neg => {
                let nr = self.binop(BinOp::Sub, Value::Num(0.0), re, span)?;
                let ni = self.binop(BinOp::Sub, Value::Num(0.0), im, span)?;
                Ok(Value::complex(nr, ni))
            }
            _ => Err(NoiseError::runtime(
                format!("cannot apply {} to a complex value", unop_name(op)),
                span,
            )),
        }
    }

    /// Split an operand into its `(re, im)` real channels for complex arithmetic. A complex value
    /// hands back its stored channels; a real scalar (`Num`/`Est`/`Dist<number>`) promotes to
    /// `x + 0i`; anything else (`Bool`/`Str`/array/…) is a spanned type error.
    fn complex_parts(&self, v: Value, span: Span) -> Result<(Value, Value)> {
        match v {
            Value::Complex { re, im } => Ok((*re, *im)),
            v @ (Value::Num(_) | Value::Est { .. }) => Ok((v, Value::Num(0.0))),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Num => {
                Ok((Value::Dist(id), Value::Num(0.0)))
            }
            other => Err(NoiseError::runtime(
                format!("cannot use {} in a complex expression", other.type_name()),
                span,
            )),
        }
    }

    /// Build a value with the *same nested shape* as `arr` but every leaf vector replaced by a fresh
    /// length-matched draw of `spec` (its color). Nesting recurses (a 2×n carrier yields two
    /// independent n-vectors), so broadcasting noise onto an `[I, Q]` pair gives I and Q distinct
    /// noise — each an independent realization of the same generator.
    fn materialize_noise_like(&mut self, spec: NoiseSpec, arr: &Value, span: Span) -> Result<Value> {
        let xs = match arr {
            Value::Array(xs) => xs,
            other => return Err(NoiseError::runtime(
                format!("internal: noise can only materialize against an array, got {}", other.type_name()),
                span,
            )),
        };
        // A vector of sub-arrays (a matrix) → recurse per row; a leaf vector → draw `len` samples.
        if matches!(xs.first(), Some(Value::Array(_))) {
            let mut out = Vec::with_capacity(xs.len());
            for e in xs.iter() {
                out.push(self.materialize_noise_like(spec, e, span)?);
            }
            Ok(Value::Array(Rc::new(out)))
        } else {
            Ok(Value::Array(Rc::new(self.materialize_noise(spec, xs.len()))))
        }
    }

    /// `cond ? a : b` as a value: a deterministic bool picks a branch; a bool-RV builds a
    /// per-lane `Select`. The library's `max`/`min`/`vsign` reuse this (mirrors the lifted `if`).
    fn select(&mut self, cond: Value, a: Value, b: Value, span: Span) -> Result<Value> {
        match cond {
            Value::Bool(true) => Ok(a),
            Value::Bool(false) => Ok(b),
            Value::Dist(c) if self.graph.kind(c) == RvKind::Bool => {
                let (aid, ak) = self.operand_to_rv(a, span)?;
                let (bid, bk) = self.operand_to_rv(b, span)?;
                if ak != bk {
                    return Err(NoiseError::runtime(
                        format!(
                            "select branches must have the same type, got {} and {}",
                            ak.type_name(),
                            bk.type_name()
                        ),
                        span,
                    ));
                }
                Ok(Value::Dist(self.graph.push(RvNode::Select { cond: c, a: aid, b: bid }, ak)))
            }
            other => Err(NoiseError::runtime(
                format!("expected a bool condition, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Convert a boolean element (deterministic or bool-RV) to a numeric `0`/`1` indicator, so
    /// `count` can sum events. A bool-RV becomes a `Select(cond, 1, 0)` (a `Num` node).
    fn indicator(&mut self, v: Value, span: Span) -> Result<Value> {
        match v {
            Value::Bool(b) => Ok(Value::Num(if b { 1.0 } else { 0.0 })),
            Value::Dist(id) if self.graph.kind(id) == RvKind::Bool => {
                self.select(Value::Dist(id), Value::Num(1.0), Value::Num(0.0), span)
            }
            other => Err(NoiseError::runtime(
                format!("count expects boolean elements, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Resolve an index value to a usize: it must be a non-negative integer point mass — never a
    /// random variable (a random gather is the dynamics fork — PLAN-COLLECTIONS §1).
    fn array_index(&self, v: &Value, span: Span) -> Result<usize> {
        let n = match v {
            Value::Num(n) => *n,
            Value::Est { val, .. } => *val,
            Value::Dist(_) => {
                return Err(NoiseError::runtime(
                    "array index must be a constant integer, not a random variable".to_string(),
                    span,
                ))
            }
            other => {
                return Err(NoiseError::runtime(
                    format!("array index must be a number, got {}", other.type_name()),
                    span,
                ))
            }
        };
        if n.fract() != 0.0 || n < 0.0 || !n.is_finite() {
            return Err(NoiseError::runtime(
                format!("array index must be a non-negative integer, got {n}"),
                span,
            ));
        }
        Ok(n as usize)
    }

    /// Extract the elements of an `Array` value, or a spanned error naming the actual type.
    fn expect_array(&self, name: &str, v: &Value, span: Span) -> Result<Rc<Vec<Value>>> {
        match v {
            Value::Array(xs) => Ok(xs.clone()),
            other => Err(NoiseError::runtime(
                format!("{name} expects an array, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// Resolve a non-negative integer count argument (a `~[shape]` dimension, a `rotation` size).
    fn count_arg(&self, name: &str, v: &Value, span: Span) -> Result<usize> {
        let n = match v {
            Value::Num(n) => *n,
            Value::Est { val, .. } => *val,
            other => {
                return Err(NoiseError::runtime(
                    format!("{name} count must be a number, got {}", other.type_name()),
                    span,
                ))
            }
        };
        if n.fract() != 0.0 || n < 0.0 || !n.is_finite() {
            return Err(NoiseError::runtime(
                format!("{name} count must be a non-negative integer, got {n}"),
                span,
            ));
        }
        Ok(n as usize)
    }

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

    /// `engine::set_max_opts(N)` — cap the *operations* each `P`/`E`/`Var`/`Q` query may spend for
    /// the rest of the run. A forcing over a cone of `C` distinct nodes costs `n×C` per-lane ops, so
    /// the query auto-clamps its draw count to `N/C` (never below 1): a heavy cone simply draws
    /// fewer samples (looser estimate) instead of doing unbounded work. This bounds each query's
    /// complexity *deterministically*, independent of the model's size — a budget, not an error.
    /// Pairs with `set_max_samples`, which caps draws; the query uses the smaller of the two.
    /// Returns unit (it's a setting, not a value).
    fn lib_set_max_opts(&mut self, args: &[Value], span: Span) -> Result<Value> {
        let [n] = arity1("set_max_opts", args, span)?;
        let n = self.count_arg("set_max_opts", n, span)?;
        if n < 1 {
            return Err(NoiseError::runtime(
                "set_max_opts(N) needs N >= 1 (the operation budget must allow at least one op)"
                    .to_string(),
                span,
            ));
        }
        self.max_opts = n as u64;
        Ok(Value::Unit)
    }

    /// Dispatch a library call (collections / linear algebra). Returns `None` if `name` is not a
    /// library function, so the caller falls through to the pure builtins. These live here (not in
    /// `builtins.rs`) because they build graph nodes and/or draw — they need `&mut self` (§0).
    fn lib_call(&mut self, name: &str, args: &[Value], span: Span) -> Option<Result<Value>> {
        // `sample(noise_white(s), n)` draws RV nodes (needs `&mut self`), so it can't ride the pure
        // signal `sample` in `builtins::call`. Intercept just the noise case; signals fall through.
        if name == "sample" {
            return match args.first() {
                Some(Value::Noise(sigma)) => Some(self.lib_sample_noise(*sigma, args, span)),
                _ => None,
            };
        }
        let r = match name {
            "set_max_samples" => self.lib_set_max_samples(args, span),
            "set_max_opts" => self.lib_set_max_opts(args, span),
            "quantize" => self.lib_quantize(args, span),
            "sum" => self.lib_sum(args, span),
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
            // complex-aware math ufuncs (PLAN-COMPLEX §4) + the real rounding family (§8).
            "exp" | "abs" | "sqrt" | "arg" | "conj" | "re" | "im" | "floor" | "ceil" => {
                self.math_ufunc(name, args, span)
            }
            "categorical" => self.lib_categorical(args, span),
            "noise_white" => self.lib_noise(NoiseKind::White, args, span),
            "noise_brown" => self.lib_noise(NoiseKind::Brown, args, span),
            "noise_pink" => self.lib_noise(NoiseKind::Pink, args, span),
            "noise_ou" => self.lib_noise_ou(args, span),
            _ => return None,
        };
        Some(r)
    }

    /// `signal::noise_white|brown|pink(sigma)` — a **lazy** zero-mean noise generator of a given
    /// spectral color (`Value::Noise`). Pure here (no draw): it only becomes `normal` RV nodes when
    /// it meets a sized array (see `binop_noise`) or `sample(...)`. Lives in `lib_call` (not
    /// `builtins.rs`) so it sits beside the noise materialization paths.
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
        let tau = self.noise_sigma(t, span)?; // reuse the number-extractor, then range-check
        if tau <= 0.0 || !tau.is_finite() {
            return Err(NoiseError::runtime(
                format!("noise_ou(sigma, tau) needs a finite correlation time tau > 0, got {tau}"),
                span,
            ));
        }
        Ok(Value::Noise(NoiseSpec { sigma, kind: NoiseKind::Ou { tau } }))
    }

    /// Extract a finite `sigma >= 0` from a noise argument.
    fn noise_sigma(&self, v: &Value, span: Span) -> Result<f64> {
        let n = match v {
            Value::Num(n) | Value::Est { val: n, .. } => *n,
            other => return Err(NoiseError::runtime(
                format!("noise strength must be a number, got {}", other.type_name()),
                span,
            )),
        };
        if n < 0.0 || !n.is_finite() {
            return Err(NoiseError::runtime(
                format!("noise strength must be a finite number >= 0, got {n}"),
                span,
            ));
        }
        Ok(n)
    }

    /// `sample(noise, n)` — materialize a lazy noise generator to a length-`n` array of RV nodes.
    /// (Deterministic signals materialize via the pure `builtins::call` path; noise draws, so it
    /// lands here.)
    fn lib_sample_noise(&mut self, spec: NoiseSpec, args: &[Value], span: Span) -> Result<Value> {
        if args.len() != 2 {
            return Err(arity_err("sample", 2, args.len(), span));
        }
        let n = self.count_arg("sample", &args[1], span)?;
        Ok(Value::Array(Rc::new(self.materialize_noise(spec, n))))
    }

    /// Materialize a noise generator to `n` zero-mean `normal` RV nodes of the requested color.
    /// White is iid; the colored kinds build a recurrence over fresh white draws (so they stay
    /// inside the RV graph — no FFT needed): Brown is a cumulative sum, OU an AR(1), Pink a sum of
    /// octave-spaced OU processes.
    fn materialize_noise(&mut self, spec: NoiseSpec, n: usize) -> Vec<Value> {
        let ids = match spec.kind {
            NoiseKind::White => (0..n).map(|_| self.normal_src(spec.sigma)).collect(),
            NoiseKind::Brown => self.brown_ids(spec.sigma, n),
            NoiseKind::Ou { tau } => self.ou_ids(spec.sigma, tau, n),
            NoiseKind::Pink => self.pink_ids(spec.sigma, n),
        };
        ids.into_iter().map(Value::Dist).collect()
    }

    /// A fresh `normal(0, sigma)` source node.
    fn normal_src(&mut self, sigma: f64) -> RvId {
        self.graph.push(RvNode::Src(Source::Normal { mu: 0.0, sigma }), RvKind::Num)
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
            prev = self.graph.push(RvNode::Binary(BinOp::Add, prev, step), RvKind::Num);
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
            let scaled = self.graph.push(RvNode::Binary(BinOp::Mul, phi_c, prev), RvKind::Num);
            prev = self.graph.push(RvNode::Binary(BinOp::Add, scaled, eps), RvKind::Num);
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
                acc[k] = self.graph.push(RvNode::Binary(BinOp::Add, acc[k], oct[k]), RvKind::Num);
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
    fn draw_rotation(&mut self, d: usize) -> Result<Value> {
        let span = Span::default();
        // Gaussian seed: `d` rows of `d` iid N(0,1) draws (a fresh source node per entry).
        let mut seed = Vec::with_capacity(d);
        for _ in 0..d {
            let mut row = Vec::with_capacity(d);
            for _ in 0..d {
                row.push(self.draw(Recipe::Normal { mu: 0.0, sigma: 1.0 })?);
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
    fn draw_permutation(&mut self, n: usize) -> Result<Value> {
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
            return Err(NoiseError::runtime("quantize needs a non-empty codebook".to_string(), span));
        }
        let mut levels: Vec<f64> = Vec::with_capacity(cs.len());
        for e in cs.iter() {
            match scalar_const(e) {
                Some(n) => levels.push(n),
                None => {
                    return Err(NoiseError::runtime(
                        "quantize centroids must be constant numbers, not random variables".to_string(),
                        span,
                    ))
                }
            }
        }
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
            return Err(NoiseError::runtime(format!("{name} of an empty array"), span));
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
            return Err(NoiseError::runtime("mean of an empty array".to_string(), span));
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
        let t = builtins::call("transpose", std::slice::from_ref(m), &self.graph, self.max_samples, self.max_opts, span, self.check_mode)?;
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
            return Err(NoiseError::runtime("categorical needs a non-empty weight vector".to_string(), span));
        }
        let mut weights = Vec::with_capacity(xs.len());
        for e in xs.iter() {
            match scalar_const(e) {
                Some(v) if v >= 0.0 && v.is_finite() => weights.push(v),
                _ => return Err(NoiseError::runtime(
                    "categorical weights must be constant, non-negative numbers".to_string(),
                    span,
                )),
            }
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(NoiseError::runtime("categorical weights must sum to a positive value".to_string(), span));
        }
        let u = self.graph.push(RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: total })), RvKind::Num);
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
    fn eval_matmul(&mut self, l: &Spanned, r: &Spanned, span: Span) -> Result<Value> {
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
            // A lazy signal defers the ufunc into its pipeline (stays a generator).
            Value::Signal(s) => Ok(Value::Signal(Rc::new(s.push(SigOp::Unary(op))))),
            Value::Array(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                for e in xs.iter() {
                    out.push(self.lib_ufunc(op, std::slice::from_ref(e), span)?);
                }
                Ok(Value::Array(Rc::new(out)))
            }
            other => Err(NoiseError::runtime(
                format!("{} expects a number, array, or random variable, got {}", unop_name(op), other.type_name()),
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

    /// `exp(z)` — `e^z`. Real `x` → `e^x`; complex `z = a + bi` → Euler `e^a·(cos b + i·sin b)`.
    /// The imaginary part may be an RV (cos/sin lift); the **real part must be a constant** — there
    /// is no `exp` node in the VM, so a random real part (`math::exp` of a real RV) is rejected. In
    /// practice the complex `exp` is always applied to a purely-imaginary angle (`e^{iθ}`), where
    /// `a = 0`.
    fn cufunc_exp(&mut self, x: Value, span: Span) -> Result<Value> {
        match x {
            Value::Num(n) => Ok(Value::Num(n.exp())),
            Value::Est { val, .. } => Ok(Value::Num(val.exp())),
            Value::Dist(_) => Err(NoiseError::runtime(
                "math::exp of a random real value is not supported (the VM has no exp node); exp of a complex angle e^{i·θ} works because it lowers to cos/sin".to_string(),
                span,
            )),
            Value::Complex { re, im } => {
                let a = match scalar_const(&re) {
                    Some(a) => a,
                    None => {
                        return Err(NoiseError::runtime(
                            "math::exp needs a constant real part (e^a); only the imaginary angle may be random".to_string(),
                            span,
                        ))
                    }
                };
                let ea = a.exp();
                let cos = self.lib_ufunc(UnOp::Cos, std::slice::from_ref(&im), span)?;
                let sin = self.lib_ufunc(UnOp::Sin, std::slice::from_ref(&im), span)?;
                let re_out = self.binop(BinOp::Mul, Value::Num(ea), cos, span)?;
                let im_out = self.binop(BinOp::Mul, Value::Num(ea), sin, span)?;
                Ok(Value::complex(re_out, im_out))
            }
            other => Err(NoiseError::runtime(
                format!("math::exp expects a number or complex value, got {}", other.type_name()),
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
                format!("math::sqrt expects a number or complex value, got {}", other.type_name()),
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
            Value::Est { val, .. } => {
                Ok(Value::Num(if val < 0.0 { std::f64::consts::PI } else { 0.0 }))
            }
            Value::Dist(id) if self.graph.kind(id) == RvKind::Num => {
                let neg = self.binop(BinOp::Lt, Value::Dist(id), Value::Num(0.0), span)?;
                self.select(neg, Value::Num(std::f64::consts::PI), Value::Num(0.0), span)
            }
            other => Err(NoiseError::runtime(
                format!("math::arg expects a number or complex value, got {}", other.type_name()),
                span,
            )),
        }
    }

    /// `atan2(y, x)` over real-scalar channels (constant or RV). Folds to `f64::atan2` when both are
    /// constant; otherwise builds the quadrant-correct form `atan(y/x) + adj`, where `adj` shifts by
    /// `±π` in the left half-plane (`x < 0`). The `x = 0` axis is measure-zero for the continuous RVs
    /// this is used on, so it isn't special-cased in the lifted path.
    fn complex_atan2(&mut self, y: Value, x: Value, span: Span) -> Result<Value> {
        if let (Some(yc), Some(xc)) = (scalar_const(&y), scalar_const(&x)) {
            return Ok(Value::Num(yc.atan2(xc)));
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
            Value::Complex { .. } => Err(NoiseError::runtime(
                format!("{} is real-only — it has no meaning on a complex number", unop_name(op)),
                span,
            )),
            other => Err(NoiseError::runtime(
                format!("{} expects a number, got {}", unop_name(op), other.type_name()),
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
            return Err(NoiseError::runtime("mse of empty signals".to_string(), span));
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

    /// Resolve a `Value::Dist` to its `RvId`, else a spanned runtime error.
    fn expect_dist(&self, v: &Value) -> Result<RvId> {
        match v {
            Value::Dist(id) => Ok(*id),
            other => Err(NoiseError::runtime(
                format!("expected a random variable, got {}", other.type_name()),
                Span::default(),
            )),
        }
    }

    /// Rust sampling API (for tests, NOT a language builtin). Draws `n` samples of the RV
    /// `v` under `seed`. The graph is read-only here — building/lifting already happened.
    pub fn sample(&self, v: &Value, n: usize, seed: u64) -> Result<Vec<f64>> {
        let id = self.expect_dist(v)?;
        Ok(sampler::sample_n(&self.graph, id, n, seed))
    }

    /// Rust sampling API (for tests): empirical mean + population variance of the RV `v`.
    pub fn moments(&self, v: &Value, n: usize, seed: u64) -> Result<Moments> {
        let id = self.expect_dist(v)?;
        Ok(sampler::moments(&self.graph, id, n, seed))
    }
}

/// Whether a value is a random variable (cheap pre-check for the lifting decision).
#[inline]
fn is_dist(v: &Value) -> bool {
    matches!(v, Value::Dist(_))
}

/// The built-in modules. `builtin` is always active; the others need a `use`.
const MODULES: [&str; 6] = ["rand", "math", "vec", "signal", "engine", "builtin"];

/// Whether `m` names a known module.
#[inline]
fn is_module(m: &str) -> bool {
    MODULES.contains(&m)
}

/// The module each builtin / constant belongs to — the single source of truth for scoping. The
/// *implementation* (lib_call vs builtins::call) is orthogonal; this only governs name access.
///
/// Module builtins are **lowercase** (`sum`, `mse`, `normal`). The always-on core (`P`, `Q`, `E`,
/// `Var`, `Print`, `Len`) is the lone exception — it is **capital-only**. The two math
/// **constants** `pi`/`e` are lowercase — note that `E` (capital) is the expectation builtin while
/// `e` (lowercase) is Euler's number, so these two are intentionally distinct and never aliased.
fn module_of(name: &str) -> Option<&'static str> {
    Some(match name {
        // distribution constructors, including `rotation` (a recipe for a random orthonormal matrix,
        // drawn with `~` like any distribution). Batched sampling is the prefix `~[shape]` operator,
        // not a builtin — see the `Sample` AST node.
        "unif" | "unif_int" | "bernoulli" | "normal" | "normal_int" | "normal_complex"
        | "exponential" | "exponential_int" | "poisson" | "geometric" | "categorical"
        | "rotation" | "permutation" => "rand",
        // math constants (lowercase only): pi/e (real), i/j (the imaginary unit, complex)
        "pi" | "e" | "i" | "j" => "math",
        "sqrt" | "round" | "log" | "log10" | "sin" | "cos" | "atan" | "sign" => "math",
        // deterministic integer number theory (modular-arithmetic core)
        "gcd" | "modpow" => "math",
        // complex-aware math ufuncs (PLAN-COMPLEX §4) + the real rounding family (§8). `exp` is the
        // exponential *function* here; the exponential *distribution* was renamed `rand::exponential`
        // precisely so this name is free.
        "exp" | "abs" | "arg" | "conj" | "re" | "im" | "floor" | "ceil" => "math",
        // collections / linear algebra (vector add/sub/matvec are the `+`/`-`/`@` operators)
        "sum" | "count" | "any" | "all" | "max" | "min" | "mean" | "dot" | "vdot" | "normsq"
        | "norm" | "transpose" | "adjoint" | "normalize" | "has_duplicates" | "count_duplicates"
        | "mse" | "ones" | "zeros" | "iota" | "outer" | "quantize" => "vec",
        // signal generation (DSP waveforms) + colored noise + materialization
        "sine" | "cosine" | "sample" | "noise_white"
        | "noise_brown" | "noise_pink" | "noise_ou" => "signal",
        // run-time knobs: tune the evaluator itself (e.g. the Monte Carlo budget). Capital-only,
        // no lowercase alias — these are imperative settings, not value-producing builtins.
        "set_max_samples" | "set_max_opts" => "engine",
        // always-on core: probability/expectation, IO, array length. These are **capital-only** (no
        // lowercase alias). Arrays are fixed-size: the half-open range has dedicated `a..b` syntax
        // (not a `range` builtin), and there is no `Push` (arrays are never grown).
        "P" | "Q" | "E" | "Var" | "Print" | "Len" => "builtin",
        _ => return None,
    })
}

/// Built-in math constants, resolved as a fallback after variable lookup. Unlike a global
/// `vars` entry, this is visible inside function bodies (which have a params-only frame).
#[inline]
fn math_const(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        _ => None,
    }
}

/// The complex math constant `math::i` (alias `math::j`) — the imaginary unit `0 + 1i`. Like
/// `pi`/`e` it is a *value*, resolved as a fallback after variable lookup (so a user's loop
/// variable `i` still wins). `j` is provided because the AM/FM (electrical-engineering) example
/// uses `j` for the imaginary unit, where `i` clashes with current/index.
#[inline]
fn math_const_complex(name: &str) -> bool {
    matches!(name, "i" | "j")
}

/// Enforce the load-bearing rule "you cannot do arithmetic on an undrawn distribution"
/// (LANG.md §2). A recipe (`unif(0,1)`, or a name bound to one with `=`) has no draw to operate
/// on; using it in an expression is an error that points at `~`.
#[inline]
fn forbid_recipe(v: &Value, span: Span) -> Result<()> {
    if let Value::Recipe(r) = v {
        return Err(NoiseError::runtime(
            format!(
                "`{r}` is an undrawn distribution, not a value — draw it first with `~` \
                 (e.g. `X ~ {r}`) and use `X`"
            ),
            span,
        ));
    }
    Ok(())
}

/// Extract a constant scalar (`Num`/`Est`) for deferring into a signal's pipeline; `None` for
/// anything else (an array materializes the signal, an RV is rejected — sample it first).
fn scalar_const(v: &Value) -> Option<f64> {
    match v {
        Value::Num(n) => Some(*n),
        Value::Est { val, .. } => Some(*val),
        _ => None,
    }
}

/// Tensor rank for the `@` operator: `0` = scalar, `1` = vector (array of non-arrays), `2` =
/// matrix (array whose first element is an array). An empty array is treated as a rank-1 vector.
/// Higher ranks and jaggedness aren't distinguished here — `@` only needs to tell these three
/// apart, and the dot/broadcast helpers it delegates to validate shape from there.
fn value_rank(v: &Value) -> u8 {
    match v {
        Value::Array(xs) => match xs.first() {
            Some(Value::Array(_)) => 2,
            _ => 1,
        },
        _ => 0,
    }
}

/// Length of an `Array` value (caller guarantees the variant).
fn array_len(v: &Value) -> usize {
    match v {
        Value::Array(xs) => xs.len(),
        _ => unreachable!("array_len on a non-array"),
    }
}

/// Materialize a lazy signal to a concrete `n`-sample array of numbers.
fn materialize_signal(s: &SignalSpec, n: usize) -> Value {
    Value::Array(Rc::new(s.sample(n).into_iter().map(Value::Num).collect()))
}

/// Display name for a transcendental unary op (for errors and dispatch).
fn unop_name(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "neg",
        UnOp::Not => "not",
        UnOp::Sin => "sin",
        UnOp::Cos => "cos",
        UnOp::Atan => "atan",
        UnOp::Sign => "sign",
        UnOp::Round => "round",
        UnOp::Floor => "floor",
        UnOp::Ceil => "ceil",
    }
}

/// Deterministic scalar evaluation of a transcendental unary op (the `Num` fast path; the RV path
/// uses the identical kernel in `bytecode::apply_un`).
fn apply_unop_f64(op: UnOp, x: f64) -> f64 {
    match op {
        UnOp::Neg => -x,
        UnOp::Sin => x.sin(),
        UnOp::Cos => x.cos(),
        UnOp::Atan => x.atan(),
        // sign: -1 / 0 / +1 (0 at exactly zero, unlike f64::signum which is ±1 at ±0.0).
        UnOp::Sign => (x > 0.0) as i32 as f64 - (x < 0.0) as i32 as f64,
        UnOp::Round => x.round(),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Not => unreachable!("Not is a boolean op, not a numeric ufunc"),
    }
}

/// Spanned arity error shared by the library methods.
fn arity_err(name: &str, want: usize, got: usize, span: Span) -> NoiseError {
    NoiseError::runtime(format!("{name} expects {want} argument(s), got {got}"), span)
}

/// Spanned vector-length mismatch shared by `dot` and elementwise broadcast.
fn length_mismatch(name: &str, a: usize, b: usize, span: Span) -> NoiseError {
    NoiseError::runtime(format!("{name} needs equal-length vectors, got {a} and {b}"), span)
}

/// Borrow exactly one argument, or an arity error.
fn arity1<'a>(name: &str, args: &'a [Value], span: Span) -> Result<[&'a Value; 1]> {
    match args {
        [a] => Ok([a]),
        _ => Err(arity_err(name, 1, args.len(), span)),
    }
}

/// Borrow exactly two arguments, or an arity error.
fn arity2<'a>(name: &str, args: &'a [Value], span: Span) -> Result<[&'a Value; 2]> {
    match args {
        [a, b] => Ok([a, b]),
        _ => Err(arity_err(name, 2, args.len(), span)),
    }
}

fn eval_unary(op: UnOp, v: Value, span: Span) -> Result<Value> {
    match (op, v) {
        (UnOp::Neg, Value::Num(n)) => Ok(Value::Num(-n)),
        (UnOp::Neg, Value::Est { val, se }) => Ok(Value::Est { val: -val, se }),
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (op, v) => Err(NoiseError::runtime(
            format!("cannot apply {:?} to {}", op, v.type_name()),
            span,
        )),
    }
}

/// First-order error propagation for arithmetic on estimates. Comparisons use the central
/// values (deterministic bools). Independence is assumed across operands.
fn eval_est_binary(op: BinOp, a: f64, sa: f64, b: f64, sb: f64, span: Span) -> Result<Value> {
    use BinOp::*;
    let est = |val: f64, se: f64| Ok(Value::Est { val, se });
    let quad = |x: f64, y: f64| (x * x + y * y).sqrt();
    match op {
        Add => est(a + b, quad(sa, sb)),
        Sub => est(a - b, quad(sa, sb)),
        Mul => est(a * b, quad(b * sa, a * sb)),
        Div => est(a / b, quad(sa / b, a * sb / (b * b))),
        // floored modulo is locally a translation in `a` (slope 1 a.e.), so the error rides along.
        Mod => est(floored_mod(a, b), sa),
        // d/da(a^b) = b·a^(b-1); exponent error is usually negligible, so propagate via the base.
        Pow => est(a.powf(b), (b * a.powf(b - 1.0)).abs() * sa),
        Lt => Ok(Value::Bool(a < b)),
        Gt => Ok(Value::Bool(a > b)),
        Le => Ok(Value::Bool(a <= b)),
        Ge => Ok(Value::Bool(a >= b)),
        Eq => Ok(Value::Bool(a == b)),
        Ne => Ok(Value::Bool(a != b)),
        And | Or => Err(NoiseError::runtime(
            "logical operator needs two bools, got numbers".to_string(),
            span,
        )),
    }
}

/// Floored modulo `a − b·floor(a/b)` (PLAN-COMPLEX §8): the result takes the **sign of `b`**, so
/// `x % n ∈ [0, n)` for `n > 0` — what modular/clock arithmetic wants (unlike Rust's `%`, which
/// truncates toward zero). IEEE edge cases follow `floor`: `x % 0` is `NaN` (no panic).
#[inline]
fn floored_mod(a: f64, b: f64) -> f64 {
    a - b * (a / b).floor()
}

/// View a `Num`/`Est` as `(value, standard_error)`; anything else is `None`.
fn as_est(v: &Value) -> Option<(f64, f64)> {
    match v {
        Value::Num(n) => Some((*n, 0.0)),
        Value::Est { val, se } => Some((*val, *se)),
        _ => None,
    }
}

fn eval_binary(op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
    use BinOp::*;
    // String concatenation: `+` with at least one string operand stringifies both via Display
    // (e.g. `"pi = " + 3.14` → `"pi = 3.14"`). This is deterministic only — a string can't enter
    // a random-variable expression (that path errors in `operand_to_rv`).
    if op == Add && (matches!(l, Value::Str(_)) || matches!(r, Value::Str(_))) {
        return Ok(Value::Str(format!("{l}{r}")));
    }
    // Uncertainty propagation: if either operand carries a standard error, propagate it
    // (first order) so e.g. `4 * P(C)` inflates the error and shows fewer digits.
    if matches!(l, Value::Est { .. }) || matches!(r, Value::Est { .. }) {
        if let (Some((a, sa)), Some((b, sb))) = (as_est(&l), as_est(&r)) {
            return eval_est_binary(op, a, sa, b, sb, span);
        }
    }
    match op {
        Add | Sub | Mul | Div | Mod | Pow => match (l, r) {
            (Value::Num(a), Value::Num(b)) => Ok(Value::Num(match op {
                Add => a + b,
                Sub => a - b,
                Mul => a * b,
                Div => a / b,
                Mod => floored_mod(a, b),
                Pow => a.powf(b),
                _ => unreachable!(),
            })),
            (a, b) => Err(NoiseError::runtime(
                format!("arithmetic on {} and {}", a.type_name(), b.type_name()),
                span,
            )),
        },
        Lt | Gt | Le | Ge => match (l, r) {
            (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(match op {
                Lt => a < b,
                Gt => a > b,
                Le => a <= b,
                Ge => a >= b,
                _ => unreachable!(),
            })),
            (a, b) => Err(NoiseError::runtime(
                format!("cannot compare {} and {}", a.type_name(), b.type_name()),
                span,
            )),
        },
        Eq | Ne => {
            let equal = match (&l, &r) {
                (Value::Num(a), Value::Num(b)) => a == b,
                (Value::Bool(a), Value::Bool(b)) => a == b,
                (Value::Str(a), Value::Str(b)) => a == b,
                (a, b) => {
                    return Err(NoiseError::runtime(
                        format!("cannot compare {} and {}", a.type_name(), b.type_name()),
                        span,
                    ))
                }
            };
            Ok(Value::Bool(if op == Eq { equal } else { !equal }))
        }
        And | Or => match (l, r) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(match op {
                And => a && b,
                Or => a || b,
                _ => unreachable!(),
            })),
            (a, b) => Err(NoiseError::runtime(
                format!(
                    "logical operator needs two bools, got {} and {}",
                    a.type_name(),
                    b.type_name()
                ),
                span,
            )),
        },
    }
}
