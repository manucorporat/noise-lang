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
use crate::dist::{DataId, DistArg, Recipe, RvGraph, RvId, RvKind, RvNode, Source, Uniform};
use crate::error::{NoiseError, Result, Span};
use crate::input::{InputKind, InputSpec, InputValue, ResolvedInput};
use crate::parser::parse;
use crate::sampler::{self, Moments};
use crate::signal::{NoiseKind, NoiseSpec, RealizationId, SigExpr, SigUnOp};
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
    /// The program's **output stream**, in source order: `Print` lines and `plot::*` charts
    /// interleaved (a plot is just another kind of output). The CLI/REPL drain and render it; the
    /// WASM playground reads it to show program output (text + charts) in the browser. Buffering
    /// (instead of a bare `println!`) keeps `Print` portable to `wasm32`, where stdout goes nowhere.
    outputs: Vec<Emission>,
    /// The span of the top-level statement currently executing — stamped onto every [`Emission`] so
    /// the doc model can attribute an output to its producing statement (PLAN-LITERATE §D5).
    current_stmt_span: Span,
    /// Emissions dropped after the [`MAX_EMISSIONS`] cap, and the first dropped statement's span.
    dropped: usize,
    first_dropped_span: Option<Span>,
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
    /// Drawn-noise **realization cache** (PLAN-SIGNALS §3). A `~`-drawn noise pins its rendered
    /// RV lanes here at first materialization; every later mention gets the SAME nodes (so
    /// `static - static` is exactly 0 and two `sample`s see one noise), and a different requested
    /// length is a teaching error. Lives next to `vars` so the playground's introspection sidecar
    /// (which relies on Engine state persisting across `run()`) keeps working unchanged.
    realizations: HashMap<RealizationId, Vec<Value>>,
    /// Next fresh [`RealizationId`] (unique per engine; allocated by the `~` draw).
    next_realization: usize,
    /// Ambient **sampling resolution** (`engine::set_resolution(N)`, default
    /// [`builtins::RESOLUTION_DEFAULT`]): the length at which reducers render a lazy signal that
    /// never met an explicit length — the time-axis twin of `max_samples`.
    resolution: usize,
    /// Constant data arrays interned by the bootstrap constructors (`rand::empirical` /
    /// `rand::block_bootstrap`), referenced by [`DataId`] so the recipes stay `Copy`.
    /// Append-only, like the graph.
    datasets: Vec<Rc<Vec<f64>>>,
    /// Host-supplied **input overrides** (PLAN-INPUTS §5): `name -> value` set before a run via
    /// [`Engine::set_input_overrides`]. Each feeds `input::` resolution by name (clamped/snapped
    /// like a knob). An override naming an input the program never declares is a post-run error.
    input_overrides: Vec<(String, InputValue)>,
    /// The run's **input manifest**: every `input::` declaration in evaluation order, resolved to a
    /// value. First evaluation of a name registers here; later mentions reuse it. Drives host
    /// discovery (`Document.result.inputs`) and dedup / redeclaration checks (PLAN-INPUTS §3).
    input_manifest: Vec<ResolvedInput>,
}

/// One item in a program's output stream — the unit `take_output` returns, in source order. `Print`
/// pushes a [`Text`](Output::Text) line; `plot::*` pushes a [`Plot`](Output::Plot). Keeping them in
/// one ordered vector is what lets text and charts interleave exactly as the program emitted them.
#[derive(Debug, Clone)]
pub enum Output {
    /// A `Print` line (no trailing newline; the renderer adds line breaks).
    Text(String),
    /// An emitted **template** (PLAN-LITERATE §D3): a root-position template statement renders to
    /// this rather than becoming the program value. `syntax` is the triple-fence info tag (`md`, …)
    /// so a host renders markdown vs preformatted text; the CLI prints the raw text either way.
    Note {
        text: String,
        syntax: Option<String>,
    },
    /// A `plot::*` chart — the same summary the introspection core produces.
    Plot(Rc<crate::introspect::Summary>),
    /// An `input::*` control (PLAN-INPUTS §4): a host-tunable parameter declared inline. Carries the
    /// spec and its resolved current value so the doc model can render a slider/checkbox right after
    /// the code that declares it. Emitted once per input (at its first declaration).
    Input { spec: InputSpec, value: InputValue },
}

/// One emitted output tagged with the **top-level statement** that produced it (PLAN-LITERATE §D5).
/// `stmt_span` is many-to-one by design: a root `for` that prints five times yields five emissions
/// sharing one `stmt_span`; a plotting function called from three root statements attributes each
/// emission to its *call site's* root statement. The doc model interleaves emissions after the code
/// block whose span contains their `stmt_span`.
#[derive(Debug, Clone)]
pub struct Emission {
    pub stmt_span: Span,
    pub output: Output,
}

/// The most output blocks a single run records (PLAN-LITERATE §D5). Past the cap the engine keeps
/// running but stops recording and reports `truncated`, so a runaway loop can't melt a host with a
/// 10⁴-block payload — and nothing is *silently* dropped.
pub const MAX_EMISSIONS: usize = 200;

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
            outputs: Vec::new(),
            current_stmt_span: Span::new(0, 0),
            dropped: 0,
            first_dropped_span: None,
            max_samples: builtins::P_DEFAULT_N,
            max_opts: builtins::MAX_OPS_DEFAULT,
            check_mode: false,
            realizations: HashMap::new(),
            next_realization: 0,
            resolution: builtins::RESOLUTION_DEFAULT,
            datasets: Vec::new(),
            input_overrides: Vec::new(),
            input_manifest: Vec::new(),
        }
    }

    /// Set the host **input overrides** for the next run (PLAN-INPUTS §5): `name -> value`, applied
    /// to matching `input::` declarations by name. Values are clamped/snapped to each input's spec
    /// exactly like the default. Overriding a name the program never declares is a post-run error
    /// (surfaced by [`run_with_inputs`](Engine::run_with_inputs) / embedded in the document).
    pub fn set_input_overrides(&mut self, overrides: Vec<(String, InputValue)>) {
        self.input_overrides = overrides;
    }

    /// The input manifest accumulated by the most recent run (PLAN-INPUTS §3) — every declared
    /// input, resolved, in declaration order. Hosts read this to render controls and prune stale
    /// overrides.
    pub fn input_manifest(&self) -> &[ResolvedInput] {
        &self.input_manifest
    }

    /// Take the program's whole output stream so far (`Print` lines and `plot::*` charts, in source
    /// order), clearing the buffer. The CLI/REPL render it to the terminal; the WASM playground
    /// serializes it to show interleaved text and charts.
    pub fn take_output(&mut self) -> Vec<Emission> {
        std::mem::take(&mut self.outputs)
    }

    /// The text-only output so far (the concatenation of `Print`/`Note` lines, charts omitted),
    /// without clearing — a convenience for callers that only want the textual log (e.g. simple demos).
    pub fn output_text(&self) -> String {
        let mut s = String::new();
        for item in &self.outputs {
            match &item.output {
                Output::Text(line) | Output::Note { text: line, .. } => {
                    s.push_str(line);
                    s.push('\n');
                }
                Output::Plot(_) | Output::Input { .. } => {}
            }
        }
        s
    }

    /// Record an emission, stamped with the current top-level statement, enforcing the
    /// [`MAX_EMISSIONS`] cap. Past the cap the output is dropped (counted, not recorded) so a
    /// runaway loop keeps running without producing an unbounded payload.
    fn emit(&mut self, output: Output) {
        if self.outputs.len() >= MAX_EMISSIONS {
            self.dropped += 1;
            if self.first_dropped_span.is_none() {
                self.first_dropped_span = Some(self.current_stmt_span);
            }
            return;
        }
        self.outputs.push(Emission {
            stmt_span: self.current_stmt_span,
            output,
        });
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

    /// The live top-level bindings after a [`run`](Engine::run), as `(name, kind)` pairs sorted by
    /// name. `kind` is the value's type tag — `"dist<number>"` / `"dist<bool>"` for random variables,
    /// else `"number"` / `"bool"` / `"array"` / … — so a UI (the playground variable picker) can list
    /// what's introspectable and offer only random variables for `describe`/`corr`/`explain`. The
    /// scope persists across `run` calls (a later `run("describe(x)")` resolves against it), which is
    /// exactly what lets introspection requests reference a program's variables without editing it.
    pub fn bindings(&self) -> Vec<(String, &'static str)> {
        let mut out: Vec<(String, &'static str)> = self
            .vars
            .iter()
            .map(|(name, v)| {
                let kind = match v {
                    Value::Dist(id) => self.graph.kind(*id).type_name(),
                    other => other.type_name(),
                };
                (name.clone(), kind)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Parse and evaluate a whole program, returning the value of the last statement (or `Unit` for
    /// an empty program). Inline `input::` declarations resolve against any host overrides set with
    /// [`set_input_overrides`](Engine::set_input_overrides); with none, each input uses its default.
    pub fn run(&mut self, src: &str) -> Result<Value> {
        // Fresh run-time counters for this program (the playground reads them after `run`).
        crate::stats::reset();
        self.dropped = 0;
        self.first_dropped_span = None;
        self.input_manifest.clear();
        let program = parse(src)?;
        let mut last = Value::Unit;
        for stmt in &program.stmts {
            last = self.eval_top(stmt)?;
        }
        self.check_input_overrides()?;
        Ok(last)
    }

    /// Like [`run`](Engine::run), but with host-supplied **input overrides** (PLAN-INPUTS §5). Sets
    /// the overrides, runs, and reports an override that names an input the program never declared.
    pub fn run_with_inputs(
        &mut self,
        src: &str,
        overrides: Vec<(String, InputValue)>,
    ) -> Result<Value> {
        self.set_input_overrides(overrides);
        self.run(src)
    }

    /// Verify every host input override named a declared input. The manifest is known only *after*
    /// the run (inputs are discovered dynamically, §3), so this is a post-run check.
    fn check_input_overrides(&self) -> Result<()> {
        for (name, _) in &self.input_overrides {
            if !self.input_manifest.iter().any(|r| &r.spec.name == name) {
                return Err(NoiseError::runtime(
                    format!(
                        "override for unknown input `{name}` (the program declares no such input)"
                    ),
                    Span::new(0, 0),
                ));
            }
        }
        Ok(())
    }

    /// Evaluate one **top-level** statement: stamp its span for emission attribution, and handle the
    /// root-position template emit (PLAN-LITERATE §D3/§D5). Anywhere below the top level a template
    /// is an ordinary string value (`eval`'s `Expr::Template` arm).
    fn eval_top(&mut self, stmt: &Spanned) -> Result<Value> {
        self.current_stmt_span = stmt.span;
        if let Expr::Template { parts, syntax } = &stmt.expr {
            let text = self.render_template(parts)?;
            self.emit(Output::Note {
                text,
                syntax: syntax.clone(),
            });
            Ok(Value::Unit)
        } else {
            self.eval(stmt)
        }
    }

    /// Run a program and assemble the single [`Document`](crate::doc::Document) contract
    /// (PLAN-LITERATE §D5) — meta + a flat, ordered block array (code / notes / plots / inputs) +
    /// the comment layer + the result. Never returns `Err`: a lex/parse failure yields a best-effort
    /// document (meta if the frontmatter parsed, empty blocks, spanned error); a runtime failure
    /// returns all blocks emitted up to the failing statement, with the error spanned. Inline
    /// `input::` declarations resolve against overrides set with
    /// [`set_input_overrides`](Engine::set_input_overrides). Both hosts (CLI, wasm) render *this*.
    pub fn run_to_document(&mut self, src: &str) -> crate::doc::Document {
        use crate::doc::Document;
        crate::stats::reset();
        self.dropped = 0;
        self.first_dropped_span = None;
        self.input_manifest.clear();

        // Frontmatter first — a malformed fence still yields a shaped document (no meta, spanned error).
        let fm = match crate::frontmatter::parse(src) {
            Ok(fm) => fm.map(|(fm, _)| fm),
            Err(e) => return Document::error_only(None, e, self.stats()),
        };
        let program = match parse(src) {
            Ok(p) => p,
            Err(e) => return Document::error_only(fm, e, self.stats()),
        };

        // Pure segmentation + comment attachment (no evaluation).
        let segs = crate::doc::segment(src, &program.stmts);
        let comments = match crate::doc::comment_layer(src, &program.stmts) {
            Ok(c) => c,
            // A trivia re-lex can only fail the same way `parse` already succeeded, but stay total.
            Err(e) => return Document::error_only(fm, e, self.stats()),
        };

        // Evaluate, catching the first runtime error (the document still carries all prior blocks).
        let mut last = Value::Unit;
        let mut error: Option<NoiseError> = None;
        for stmt in &program.stmts {
            match self.eval_top(stmt) {
                Ok(v) => last = v,
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        // A stray input override is a post-run error (the manifest exists only now); it never
        // discards the blocks that ran, and defers to a real runtime error if one occurred first.
        if error.is_none() {
            if let Err(e) = self.check_input_overrides() {
                error = Some(e);
            }
        }
        let emissions = self.take_output();
        let inputs = std::mem::take(&mut self.input_manifest);
        let truncated = self.first_dropped_span.map(|span| (self.dropped, span));
        crate::doc::assemble(
            src,
            fm,
            segs,
            comments,
            emissions,
            inputs,
            last,
            error,
            self.stats(),
            truncated,
        )
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
            // Extracted into a method (like the other arms) to keep `eval`'s stack frame small for
            // the recursion-depth budget — see `MAX_CALL_DEPTH`.
            Expr::Unary(op, rhs) => self.eval_unary_expr(*op, rhs, node.span),
            Expr::Binary(op, l, r) => {
                let lv = self.eval(l)?;
                let rv = self.eval(r)?;
                forbid_undrawn(&lv, l.span)?;
                forbid_undrawn(&rv, r.span)?;
                self.binop(*op, lv, rv, node.span)
            }
            // Extracted into a method (like the other arms) to keep `eval`'s stack frame small for
            // the recursion-depth budget — see `MAX_CALL_DEPTH`.
            Expr::Bind(kind, name, rhs) => self.eval_bind(*kind, name, rhs),
            Expr::Sample { shape, body } => self.eval_sample(shape, body),
            Expr::MatMul(l, r) => self.eval_matmul(l, r, node.span),
            Expr::FnDef {
                kind,
                name,
                params,
                body,
            } => {
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
                forbid_undrawn(&c, cond.span)?;
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
            // Extracted into a method (like the other arms) to keep `eval`'s stack frame small for
            // the recursion-depth budget — see `MAX_CALL_DEPTH`.
            Expr::Call(name, call_args) => self.eval_call(name, call_args, node.span),
            // Extracted into methods to keep `eval`'s stack frame small (recursion-depth budget).
            Expr::Array(elems) => self.eval_array(elems),
            Expr::Comprehension { body, var, iter } => self.eval_comprehension(body, var, iter),
            Expr::Range(lo, hi) => self.eval_range(lo, hi, node.span),
            Expr::Index(arr, idx) => self.eval_index(arr, idx),
            Expr::For { var, iter, body } => self.eval_for(var, iter, body),
            Expr::Use(module) => {
                if !is_module(module) {
                    return Err(NoiseError::runtime(
                        format!(
                            "unknown module '{module}' (known modules: {})",
                            MODULES.join(", ")
                        ),
                        node.span,
                    ));
                }
                // `use builtin;` is a harmless no-op (it's always active).
                self.used.insert(module.clone());
                Ok(Value::Unit)
            }
            // `event | given` — a conditioned value (Bayes, scoped to a query). Builds a
            // `Value::Cond` you can bind (`a = X | C`) and later query (`P(a)`), but not do
            // arithmetic on — like a `Recipe`, it is consumed by `P`/`E`/`Var`/`Q`, not operated on.
            Expr::Cond { event, given } => self.eval_cond(event, given),
            // A template (PLAN-LITERATE §D3). Evaluated as a *value* here — it renders to a string
            // (each hole via its display form). Root-position emission (pushing `Output::Note`) is
            // handled in the run loop, which sees the top-level statement; nested anywhere else, a
            // template is just this string.
            Expr::Template { parts, .. } => Ok(Value::Str(self.render_template(parts)?)),
            // The `continue` control sentinel — short-circuits the enclosing block / loop body.
            Expr::Continue => Ok(Value::Continue),
        }
    }

    /// Render a template's parts to a string: literal text verbatim, each hole via its value's
    /// display form (an `Est` self-rounds to its standard error, exactly like `Print`). Holes are
    /// deterministic-only — an undrawn recipe is the same error string `+` raises.
    fn render_template(&mut self, parts: &[crate::ast::TemplatePart]) -> Result<String> {
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
                Err(NoiseError::runtime(
                    format!("undefined variable '{name}'"),
                    span,
                ))
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

    fn eval_array(&mut self, elems: &[Spanned]) -> Result<Value> {
        let mut out = Vec::with_capacity(elems.len());
        for e in elems {
            let v = self.eval(e)?;
            forbid_undrawn(&v, e.span)?;
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

    fn eval_for(&mut self, var: &str, iter: &Spanned, body: &Spanned) -> Result<Value> {
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
    fn eval_comprehension(&mut self, body: &Spanned, var: &str, iter: &Spanned) -> Result<Value> {
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
    fn eval_bind(&mut self, kind: BindKind, name: &str, rhs: &Spanned) -> Result<Value> {
        // Name inference (PLAN-INPUTS §2): when `input::…` is the *direct* RHS of a bind, the input's
        // name defaults to the binding's LHS identifier — so `dice_sides = input::real(min: 1, max:
        // 100)` needs no explicit `name:`. Only a direct RHS qualifies; a nested `input::` elsewhere
        // still requires its own `name:`.
        let v = match as_input_call(&rhs.expr) {
            Some((base, call_args)) => self.input_call(base, call_args, Some(name), rhs.span)?,
            None => self.eval(rhs)?,
        };
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
    fn eval_call(&mut self, name: &str, call_args: &CallArgs, span: Span) -> Result<Value> {
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
        // `plot::*` — the charting surface for examples. Computes a summary (reusing the
        // introspection core) and *captures* it like `Print`, returning unit.
        if module == Some("plot") {
            return self.plot_call(base, args, &arg_vals, span);
        }
        // `stats::*` — the same computations, handed back as numbers instead of a chart.
        if module == Some("stats") {
            return self.stats_call(base, &arg_vals, span);
        }
        // A user function (unqualified) shadows a builtin of the same name.
        if module.is_none() {
            if let Some(f) = self.funcs.get(base).cloned() {
                return self.call_user_fn(base, &f, arg_vals, span);
            }
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
            builtins::call(
                base,
                &arg_vals,
                &self.graph,
                self.max_samples,
                self.max_opts,
                span,
                self.check_mode,
            )
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
        let value = spec.resolve(over)?;
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

    fn call_user_fn(
        &mut self,
        name: &str,
        f: &UserFn,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value> {
        if args.len() != f.params.len() {
            return Err(NoiseError::runtime(
                format!(
                    "{name} expects {} argument(s), got {}",
                    f.params.len(),
                    args.len()
                ),
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

    /// `~` semantics in one place: a recipe is drawn into a fresh RV; an undrawn noise generator
    /// is drawn into ONE lazy **realization** (a `Signal` noise leaf — length still lazy);
    /// anything else (a point mass, an already-drawn RV) binds as-is, since there is nothing new
    /// to draw. Fallible because a structured recipe (`rotation`) builds a whole matrix and could
    /// surface a shape error; scalar recipe and noise draws never fail.
    fn draw_if_recipe(&mut self, v: Value) -> Result<Value> {
        match v {
            Value::Recipe(r) => self.draw(r),
            Value::Noise(spec) => Ok(self.draw_noise(spec)),
            other => Ok(other),
        }
    }

    /// Draw one lazy noise **realization** (PLAN-SIGNALS §1.1): allocate a fresh
    /// [`RealizationId`] and wrap it in a signal leaf. The length stays lazy — the realization
    /// pins it at first materialization (see [`Engine::realization`]). The complex generator
    /// splits into two independent real lanes of strength `sigma/√2` (per-quadrature CSCG, so
    /// `E|z|² = sigma²` like `rand::normal_complex`).
    fn draw_noise(&mut self, spec: NoiseSpec) -> Value {
        if let NoiseKind::WhiteComplex = spec.kind {
            let lane = NoiseSpec {
                sigma: spec.sigma / std::f64::consts::SQRT_2,
                kind: NoiseKind::White,
            };
            let re = self.draw_noise(lane);
            let im = self.draw_noise(lane);
            return Value::complex(re, im);
        }
        let id = RealizationId(self.next_realization);
        self.next_realization += 1;
        Value::Signal(Rc::new(SigExpr::Noise { id, spec }))
    }

    /// `~[n] noise` / `~[m, n] noise` — an **eager** realization pinned up front: the last
    /// dimension is the realization length, outer dimensions draw independent realizations. This
    /// is the old `sample(noise_*(…), n)`, now spelled as a draw; it materializes directly to an
    /// ordinary array of RVs (no cache entry needed — the value IS the realization).
    fn draw_noise_shaped(&mut self, dims: &[usize], spec: NoiseSpec) -> Value {
        if let [n] = dims {
            let vals = match spec.kind {
                NoiseKind::WhiteComplex => {
                    let lane = NoiseSpec {
                        sigma: spec.sigma / std::f64::consts::SQRT_2,
                        kind: NoiseKind::White,
                    };
                    let re = self.materialize_noise(lane, *n);
                    let im = self.materialize_noise(lane, *n);
                    re.into_iter()
                        .zip(im)
                        .map(|(a, b)| Value::complex(a, b))
                        .collect()
                }
                _ => self.materialize_noise(spec, *n),
            };
            return Value::Array(Rc::new(vals));
        }
        let (m, rest) = (dims[0], &dims[1..]);
        Value::Array(Rc::new(
            (0..m).map(|_| self.draw_noise_shaped(rest, spec)).collect(),
        ))
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
        // `~[n]` on a noise generator pins ONE realization to length `n` (the shape is the time
        // axis), not `n` independent realizations — so it gets its own arm.
        if let Value::Noise(spec) = v {
            return Ok(self.draw_noise_shaped(&dims, spec));
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
        if let Recipe::Empirical { data } = r {
            return Ok(self.draw_empirical(data));
        }
        if let Recipe::BlockBootstrap { data, block_len } = r {
            return Ok(self.draw_block_bootstrap(data, block_len));
        }
        // A complex draw yields a `Value::Complex` (two independent real channels), not a scalar id.
        if let Recipe::NormalComplex { sigma } = r {
            let s = sigma / std::f64::consts::SQRT_2;
            let re = self.graph.push(
                RvNode::Src(Source::Normal { mu: 0.0, sigma: s }),
                RvKind::Num,
            );
            let im = self.graph.push(
                RvNode::Src(Source::Normal { mu: 0.0, sigma: s }),
                RvKind::Num,
            );
            return Ok(Value::complex(Value::Dist(re), Value::Dist(im)));
        }
        let id = match r {
            Recipe::Uniform { lo, hi } => self.graph.push(
                RvNode::Src(Source::Uniform(Uniform { lo, hi })),
                RvKind::Num,
            ),
            Recipe::UniformInt { lo, hi } => self
                .graph
                .push(RvNode::Src(Source::UniformInt { lo, hi }), RvKind::Num),
            Recipe::Normal { mu, sigma } => self
                .graph
                .push(RvNode::Src(Source::Normal { mu, sigma }), RvKind::Num),
            Recipe::Exp { rate } => self
                .graph
                .push(RvNode::Src(Source::Exp { rate }), RvKind::Num),
            Recipe::Poisson { lambda } => self
                .graph
                .push(RvNode::Src(Source::Poisson { lambda }), RvKind::Num),
            Recipe::Geometric { p } => self
                .graph
                .push(RvNode::Src(Source::Geometric { p }), RvKind::Num),
            // The `_int` family draws a continuous source then rounds each lane to an integer.
            Recipe::NormalInt { mu, sigma } => {
                let z = self
                    .graph
                    .push(RvNode::Src(Source::Normal { mu, sigma }), RvKind::Num);
                self.graph.push(RvNode::Unary(UnOp::Round, z), RvKind::Num)
            }
            Recipe::ExpInt { rate } => {
                let z = self
                    .graph
                    .push(RvNode::Src(Source::Exp { rate }), RvKind::Num);
                self.graph.push(RvNode::Unary(UnOp::Round, z), RvKind::Num)
            }
            Recipe::Bernoulli { p } => {
                // bernoulli(p) ≡ (unif(0,1) < p): a bool-RV that is 1 with probability p.
                let u = self.graph.push(
                    RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                    RvKind::Num,
                );
                let c = self.graph.push(RvNode::ConstNum(p), RvKind::Num);
                self.graph
                    .push(RvNode::Binary(BinOp::Lt, u, c), RvKind::Bool)
            }
            // --- distributions with a (possibly) random parameter: lower to a standard base draw +
            //     a deterministic transform, so the VM/RNG never change (LANG.md "Hierarchical
            //     distributions"). A fresh base draw per `~`, the SAME parameter node reused, gives
            //     conditional independence given the parameter (`a ~ bernoulli(p); b ~ bernoulli(p)`
            //     are independent given `p`). The transform nodes simplify/CSE/lower like any other.
            Recipe::UniformDyn { lo, hi } => {
                // lo + (hi − lo)·U,  U ~ unif(0,1).
                let u = self.graph.push(
                    RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                    RvKind::Num,
                );
                let (lo, hi) = (self.arg_id(lo), self.arg_id(hi));
                let width = self
                    .graph
                    .push(RvNode::Binary(BinOp::Sub, hi, lo), RvKind::Num);
                let scaled = self
                    .graph
                    .push(RvNode::Binary(BinOp::Mul, width, u), RvKind::Num);
                self.graph
                    .push(RvNode::Binary(BinOp::Add, lo, scaled), RvKind::Num)
            }
            Recipe::UniformIntDyn { lo, hi } => {
                // lo + floor((hi − lo + 1)·U),  U ~ unif(0,1) → inclusive integers lo..=hi.
                let u = self.graph.push(
                    RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                    RvKind::Num,
                );
                let (lo, hi) = (self.arg_id(lo), self.arg_id(hi));
                let diff = self
                    .graph
                    .push(RvNode::Binary(BinOp::Sub, hi, lo), RvKind::Num);
                let one = self.graph.push(RvNode::ConstNum(1.0), RvKind::Num);
                let width = self
                    .graph
                    .push(RvNode::Binary(BinOp::Add, diff, one), RvKind::Num);
                let scaled = self
                    .graph
                    .push(RvNode::Binary(BinOp::Mul, u, width), RvKind::Num);
                let floored = self
                    .graph
                    .push(RvNode::Unary(UnOp::Floor, scaled), RvKind::Num);
                self.graph
                    .push(RvNode::Binary(BinOp::Add, lo, floored), RvKind::Num)
            }
            Recipe::NormalDyn { mu, sigma, int } => {
                // mu + sigma·Z,  Z ~ N(0,1); `int` rounds each lane (normal_int).
                let z = self.graph.push(
                    RvNode::Src(Source::Normal {
                        mu: 0.0,
                        sigma: 1.0,
                    }),
                    RvKind::Num,
                );
                let (mu, sigma) = (self.arg_id(mu), self.arg_id(sigma));
                let scaled = self
                    .graph
                    .push(RvNode::Binary(BinOp::Mul, sigma, z), RvKind::Num);
                let val = self
                    .graph
                    .push(RvNode::Binary(BinOp::Add, mu, scaled), RvKind::Num);
                if int {
                    self.graph
                        .push(RvNode::Unary(UnOp::Round, val), RvKind::Num)
                } else {
                    val
                }
            }
            Recipe::ExpDyn { rate, int } => {
                // E / rate,  E ~ Exp(1) → Exp(rate); `int` rounds each lane (exponential_int).
                let e = self
                    .graph
                    .push(RvNode::Src(Source::Exp { rate: 1.0 }), RvKind::Num);
                let rate = self.arg_id(rate);
                let val = self
                    .graph
                    .push(RvNode::Binary(BinOp::Div, e, rate), RvKind::Num);
                if int {
                    self.graph
                        .push(RvNode::Unary(UnOp::Round, val), RvKind::Num)
                } else {
                    val
                }
            }
            Recipe::BernoulliDyn { p } => {
                // (U < p),  U ~ unif(0,1): a bool-RV true with the lane's probability p.
                let u = self.graph.push(
                    RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                    RvKind::Num,
                );
                let p = self.arg_id(p);
                self.graph
                    .push(RvNode::Binary(BinOp::Lt, u, p), RvKind::Bool)
            }
            // Handled above with an early return (they yield arrays/complex, not a scalar `id`).
            Recipe::Rotation { .. } => unreachable!("rotation drawn via draw_rotation"),
            Recipe::Permutation { .. } => unreachable!("permutation drawn via draw_permutation"),
            Recipe::Empirical { .. } => unreachable!("empirical drawn via draw_empirical"),
            Recipe::BlockBootstrap { .. } => {
                unreachable!("block_bootstrap drawn via draw_block_bootstrap")
            }
            Recipe::NormalComplex { .. } => {
                unreachable!("normal_complex drawn via the complex path")
            }
        };
        Ok(Value::Dist(id))
    }

    /// Materialize a (possibly random) distribution parameter as a sample-DAG node: a constant folds
    /// to a `ConstNum`; a random parameter reuses its existing node, so every `~` draw of the recipe
    /// shares the SAME per-lane parameter value (with a fresh base draw) — conditional independence
    /// given the parameter.
    fn arg_id(&mut self, a: DistArg) -> RvId {
        match a {
            DistArg::Const(x) => self.graph.push(RvNode::ConstNum(x), RvKind::Num),
            DistArg::Rv(id) => id,
        }
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
            UnOp::Sin
            | UnOp::Cos
            | UnOp::Atan
            | UnOp::Sign
            | UnOp::Round
            | UnOp::Floor
            | UnOp::Ceil
            | UnOp::Exp
            | UnOp::Ln => {
                if kind != RvKind::Num {
                    return Err(NoiseError::runtime(
                        format!("cannot apply {} to {}", unop_name(op), kind.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
        };
        Ok(Value::Dist(
            self.graph.push(RvNode::Unary(op, id), result_kind),
        ))
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
        Ok(Value::Dist(
            self.graph.push(RvNode::Binary(op, lid, rid), result_kind),
        ))
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
        Ok(Value::Dist(
            self.graph.push(RvNode::Select { cond, a, b }, ak),
        ))
    }

    /// Coerce an operand to an `(RvId, RvKind)`. `Dist` reuses its id (structural sharing);
    /// `Num`/`Bool` fold into a const node; `Str`/`Unit` are spanned errors (preserving the
    /// deterministic type-error contract, e.g. for `X + "a"`).
    fn operand_to_rv(&mut self, v: Value, span: Span) -> Result<(RvId, RvKind)> {
        match v {
            Value::Dist(id) => Ok((id, self.graph.kind(id))),
            Value::Num(n) => Ok((
                self.graph.push(RvNode::ConstNum(n), RvKind::Num),
                RvKind::Num,
            )),
            // An estimate folds in as its central value (its error is dropped inside the RV).
            Value::Est { val, .. } => Ok((
                self.graph.push(RvNode::ConstNum(val), RvKind::Num),
                RvKind::Num,
            )),
            Value::Bool(b) => Ok((
                self.graph.push(RvNode::ConstBool(b), RvKind::Bool),
                RvKind::Bool,
            )),
            other => Err(NoiseError::runtime(
                format!(
                    "cannot use {} in a random-variable expression",
                    other.type_name()
                ),
                span,
            )),
        }
    }

    /// Build a conditioned value from `event | given` (LANG.md "conditioning"). It records the
    /// quantity and the condition *separately* in a `Value::Cond` (the fusion into
    /// `select(condition, quantity, NaN)` is deferred to the query) so operations compose:
    /// `2*(X|C)+1` is `(2X+1) | C`. `given` must be an event (bool); `event` may be an event (for
    /// `P`) or any numeric quantity (for `E`/`Var`/`Q`, checked at the query). A conditioned value
    /// can be bound and queried; combining two that are conditioned on *different* events is rejected
    /// (`binop_cond`).
    fn eval_cond(&mut self, event: &Spanned, given: &Spanned) -> Result<Value> {
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
    fn query_cond(&mut self, qname: &str, arg_vals: &[Value], span: Span) -> Result<Value> {
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
        let (default_n, max_opts, check) = (self.max_samples, self.max_opts, self.check_mode);
        match qname {
            "P" => builtins::prob_cond(&self.graph, root, tail, default_n, max_opts, span, check),
            "E" | "Var" => builtins::moment_cond(
                qname,
                &self.graph,
                root,
                tail,
                default_n,
                max_opts,
                span,
                check,
            ),
            "Q" => {
                builtins::quantile_cond(&self.graph, root, tail, default_n, max_opts, span, check)
            }
            _ => unreachable!("query_cond dispatched with an unknown name"),
        }
    }

    /// Dispatch a variable-introspection call (`describe`/`hist`/`samples`/`corr`/`scatter`/
    /// `explain`) to the [`crate::introspect`] core, returning a [`Value::Summary`]. `args` is the
    /// un-evaluated argument list (for labelling a summary by its source name); `arg_vals` are the
    /// evaluated operands. Kept its own method so `eval`'s frame stays small (recursion budget). All
    /// six are *views/compositions* of two operations: a one-variable [`Dist1`](crate::introspect::Dist1)
    /// and a two-variable [`Dist2`](crate::introspect::Dist2).
    fn introspect_call(
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
                match dist1(&self.graph, root, boolean, conditional, n, seed, head_k) {
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
                match dist2(&self.graph, a, b, None, n, seed) {
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
                let sd = dist1(&self.graph, target_root, false, cond.is_some(), n, seed, 0)
                    .map_or(f64::NAN, |d| d.sd);
                let mut corrs = Vec::with_capacity(cands.len());
                for (cname, cid) in &cands {
                    if let Some(d) = dist2(&self.graph, target, *cid, cond, n, seed) {
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
        )))
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
        Ok(corr_grid(&self.graph, &roots, n, INTROSPECT_SEED))
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
    fn plot_call(
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
                    format!("unknown plot 'plot::{other}' (try histogram, line, heatmap, fan, scatter, corr, explain, value)"),
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
    fn stats_call(&mut self, base: &str, arg_vals: &[Value], span: Span) -> Result<Value> {
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
        match dist2(&self.graph, a, b, None, n, INTROSPECT_SEED) {
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
        let d = draws(&self.graph, root, conditional, n, INTROSPECT_SEED);
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
        let label = label_of(&args[0]);
        let c = self.fan_chart("plot::fan", arg_vals, span)?;
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
        Ok(fan(&self.graph, &roots, n, INTROSPECT_SEED))
    }

    /// Evaluate a prefix unary op (`-x` / `!x` / the math ufuncs). A conditioned operand pushes the
    /// op into its quantity and keeps the condition (`-(X|C)` is `(-X) | C`); otherwise the complex /
    /// RV-lift / deterministic paths apply as before. Its own method so `eval`'s frame stays small.
    fn eval_unary_expr(&mut self, op: UnOp, rhs: &Spanned, span: Span) -> Result<Value> {
        let v = self.eval(rhs)?;
        forbid_undrawn(&v, rhs.span)?;
        if let Value::Cond {
            quantity,
            condition,
            ..
        } = v
        {
            let q = self.lift_unary(op, Value::Dist(quantity), span)?;
            let (quantity, q_kind) = self.operand_to_rv(q, span)?;
            Ok(Value::Cond {
                quantity,
                q_kind,
                condition,
            })
        } else if matches!(v, Value::Complex { .. }) {
            self.unary_complex(op, v, span)
        } else if let Value::Signal(s) = v {
            // A prefix op on a lazy signal defers into the tree (`-sine(3)` stays a signal).
            Ok(Value::Signal(Rc::new(SigExpr::Unary(SigUnOp::Un(op), s))))
        } else if is_dist(&v) {
            self.lift_unary(op, v, span)
        } else {
            eval_unary(op, v, span) // deterministic fast path, unchanged
        }
    }

    /// A binary op with at least one conditioned operand — `2*(X|C)`, `(X|C)+1`, `(X|C) < 3`, or
    /// `(X|C)+(Y|C)`. The op pushes into the quantity and the condition rides along, so the result
    /// is the conditioned value `(quantity ⊕ other) | C`. Two conditioned operands must share the
    /// *same* condition node — conditioning on two different events at once is ill-defined, so it is
    /// a spanned error (condition once, at the end: `(X + Y) | C`).
    fn binop_cond(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        let (quantity, condition) = match (l, r) {
            (
                Value::Cond {
                    quantity: ql,
                    condition: cl,
                    ..
                },
                Value::Cond {
                    quantity: qr,
                    condition: cr,
                    ..
                },
            ) => {
                if cl != cr {
                    return Err(NoiseError::runtime(
                        "cannot combine two values conditioned on different events — condition once, \
                         at the end (e.g. `(X + Y) | C`)"
                            .to_string(),
                        span,
                    ));
                }
                (self.binop(op, Value::Dist(ql), Value::Dist(qr), span)?, cl)
            }
            (
                Value::Cond {
                    quantity,
                    condition,
                    ..
                },
                other,
            ) => (
                self.binop(op, Value::Dist(quantity), other, span)?,
                condition,
            ),
            (
                other,
                Value::Cond {
                    quantity,
                    condition,
                    ..
                },
            ) => (
                self.binop(op, other, Value::Dist(quantity), span)?,
                condition,
            ),
            _ => unreachable!("binop_cond called without a conditioned operand"),
        };
        // Re-wrap the transformed quantity, keeping the condition. The quantity is always an RV here
        // (it has a `Dist` operand), so `operand_to_rv` just reads its id/kind.
        let (quantity, q_kind) = self.operand_to_rv(quantity, span)?;
        Ok(Value::Cond {
            quantity,
            q_kind,
            condition,
        })
    }

    /// Combine two element `Value`s with a binary op — the single fold primitive the library
    /// reuses (LANG.md §0). Lifts to a graph node if either side is a `Dist`, else folds
    /// deterministically. Recipes are rejected (you can't operate on an undrawn distribution).
    fn binop(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        forbid_undrawn(&l, span)?;
        forbid_undrawn(&r, span)?;
        // A conditioned operand pushes the op into its quantity and carries the condition along —
        // `2*(X|C)` is `(2X) | C`. Handle before the other paths so a conditioned value never folds
        // or lifts as a plain RV.
        if matches!(l, Value::Cond { .. }) || matches!(r, Value::Cond { .. }) {
            return self.binop_cond(op, l, r, span);
        }
        // A lazy signal defers ops (growing its expression tree) and materializes against a sized
        // array. Handle it before the array path so `signal ⊕ array` adopts the array's length.
        // (An undrawn noise generator never reaches here — `forbid_undrawn` above rejects it.)
        if matches!(l, Value::Signal(_)) || matches!(r, Value::Signal(_)) {
            return self.binop_signal(op, l, r, span);
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

    /// Combine a lazy signal with another operand. A scalar or another signal **defers** (the op
    /// grows the expression tree, staying O(1) memory — `sine(3) + sine(7)` is a two-tone
    /// signal); a **complex** operand routes to the complex path (the signal promotes to
    /// `sig + 0i`); a sized **array materializes** the signal to that length and then broadcasts.
    fn binop_signal(&mut self, op: BinOp, l: Value, r: Value, span: Span) -> Result<Value> {
        // Complex first: `signal ⊕ complex` decomposes into channel ops, which land back here as
        // plain signal arithmetic.
        if matches!(l, Value::Complex { .. }) || matches!(r, Value::Complex { .. }) {
            return self.binop_complex(op, l, r, span);
        }
        match (l, r) {
            (Value::Signal(a), Value::Signal(b)) => {
                Ok(Value::Signal(Rc::new(SigExpr::Binop(op, a, b))))
            }
            (Value::Signal(a), rhs) if scalar_const(&rhs).is_some() => {
                let c = Rc::new(SigExpr::Konst(scalar_const(&rhs).unwrap()));
                Ok(Value::Signal(Rc::new(SigExpr::Binop(op, a, c))))
            }
            (lhs, Value::Signal(b)) if scalar_const(&lhs).is_some() => {
                let c = Rc::new(SigExpr::Konst(scalar_const(&lhs).unwrap()));
                Ok(Value::Signal(Rc::new(SigExpr::Binop(op, c, b))))
            }
            // A sized array fixes the sample count: materialize the signal, then broadcast.
            (Value::Signal(s), arr @ Value::Array(_)) => {
                let mat = Value::Array(Rc::new(self.materialize_sig(&s, array_len(&arr), span)?));
                self.binop(op, mat, arr, span)
            }
            (arr @ Value::Array(_), Value::Signal(s)) => {
                let mat = Value::Array(Rc::new(self.materialize_sig(&s, array_len(&arr), span)?));
                self.binop(op, arr, mat, span)
            }
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
            // A real lazy signal promotes to `sig + 0i` — this is what makes a complex signal
            // pure composition (`Complex { re: Signal, im: Signal }`, PLAN-SIGNALS §1.3).
            v @ Value::Signal(_) => Ok((v, Value::Num(0.0))),
            other => Err(NoiseError::runtime(
                format!("cannot use {} in a complex expression", other.type_name()),
                span,
            )),
        }
    }

    /// Materialize a lazy signal tree to `n` element `Value`s (PLAN-SIGNALS §3). A deterministic
    /// tree folds straight to `f64`s ([`SigExpr::sample_f64`] — the fast path `nyquist.noise`
    /// rides); once a subtree carries a drawn-noise leaf the walk switches to per-element `Value`
    /// combination through the ordinary [`Self::binop`] lifting, with the **realization cache**
    /// guaranteeing every mention of the same draw yields the same RV nodes.
    fn materialize_sig(&mut self, expr: &SigExpr, n: usize, span: Span) -> Result<Vec<Value>> {
        if !expr.has_noise() {
            return Ok(expr.sample_f64(n).into_iter().map(Value::Num).collect());
        }
        match expr {
            SigExpr::Noise { id, spec } => self.realization(*id, *spec, n, span),
            SigExpr::Unary(op, a) => {
                let xs = self.materialize_sig(a, n, span)?;
                let mut out = Vec::with_capacity(n);
                for x in xs {
                    out.push(self.sig_unary(*op, x, span)?);
                }
                Ok(out)
            }
            SigExpr::Binop(op, a, b) => {
                let ls = self.materialize_sig(a, n, span)?;
                let rs = self.materialize_sig(b, n, span)?;
                let mut out = Vec::with_capacity(n);
                for (l, r) in ls.into_iter().zip(rs) {
                    out.push(self.binop(*op, l, r, span)?);
                }
                Ok(out)
            }
            SigExpr::Atan2(y, x) => {
                let ys = self.materialize_sig(y, n, span)?;
                let xs = self.materialize_sig(x, n, span)?;
                let mut out = Vec::with_capacity(n);
                for (yv, xv) in ys.into_iter().zip(xs) {
                    out.push(self.complex_atan2(yv, xv, span)?);
                }
                Ok(out)
            }
            // `has_noise` is true here, so the deterministic leaves are unreachable.
            SigExpr::Wave { .. } | SigExpr::Konst(_) => {
                unreachable!("deterministic leaf under a noise-bearing walk")
            }
        }
    }

    /// One deferred unary step applied to a materialized element: a constant folds with the same
    /// kernel the deterministic walk uses; an RV lifts (a deferred `exp` over a noisy lane lifts
    /// to the same `UnOp::Exp` node `math::exp` of an RV builds).
    fn sig_unary(&mut self, op: SigUnOp, x: Value, span: Span) -> Result<Value> {
        match (op, x) {
            (op, Value::Num(v)) => Ok(Value::Num(crate::signal::apply_unary(op, v))),
            (op, Value::Est { val, .. }) => Ok(Value::Num(crate::signal::apply_unary(op, val))),
            (SigUnOp::Un(u), x @ Value::Dist(_)) => self.lift_unary(u, x, span),
            (SigUnOp::Exp, x @ Value::Dist(_)) => self.lift_unary(UnOp::Exp, x, span),
            (_, other) => Err(NoiseError::runtime(
                format!("cannot apply a signal op to {}", other.type_name()),
                span,
            )),
        }
    }

    /// Resolve a drawn-noise leaf through the **realization cache**: the first materialization
    /// renders the spec at length `n` and pins it; every later mention gets the SAME `Value`s
    /// (the same RV nodes — that is what makes `static - static` exactly 0 and re-rendering the
    /// same draw at two carriers a fair fight). A different `n` is a teaching error: white noise
    /// has no finer version of itself, so a silent re-render would be a lie (PLAN-SIGNALS §1.1).
    fn realization(
        &mut self,
        id: RealizationId,
        spec: NoiseSpec,
        n: usize,
        span: Span,
    ) -> Result<Vec<Value>> {
        if let Some(vals) = self.realizations.get(&id) {
            if vals.len() != n {
                return Err(NoiseError::runtime(
                    format!(
                        "this drawn noise was realized at length {} and cannot be re-rendered at {n} — \
                         noise has no finer version of itself; keep one resolution across its uses, \
                         or pin it with `~[{}]`",
                        vals.len(),
                        vals.len()
                    ),
                    span,
                ));
            }
            return Ok(vals.clone());
        }
        let vals = self.materialize_noise(spec, n);
        self.realizations.insert(id, vals.clone());
        Ok(vals)
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
                Ok(Value::Dist(self.graph.push(
                    RvNode::Select {
                        cond: c,
                        a: aid,
                        b: bid,
                    },
                    ak,
                )))
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
    ///
    /// This is the reducers' single funnel (PLAN-SIGNALS §1.2): a **lazy signal** handed to a
    /// reducer (`mse`/`mean`/`sum`/`dot`/…) renders here at the **ambient resolution**
    /// (`engine::set_resolution`, default [`builtins::RESOLUTION_DEFAULT`]) — the resolution is a
    /// measurement knob, so it applies at the measurement. `signal::sample(sig, n)` remains the
    /// explicit override; a drawn noise realization pins its length at first materialization, so
    /// changing the resolution between uses of one realization errors instead of lying.
    fn expect_array(&mut self, name: &str, v: &Value, span: Span) -> Result<Rc<Vec<Value>>> {
        match v {
            Value::Array(xs) => Ok(xs.clone()),
            Value::Signal(s) => {
                let n = self.resolution;
                Ok(Rc::new(self.materialize_sig(&s.clone(), n, span)?))
            }
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
        let r = match name {
            "set_max_samples" => self.lib_set_max_samples(args, span),
            "set_max_opts" => self.lib_set_max_opts(args, span),
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
        let tau = self.noise_sigma(t, span)?; // reuse the number-extractor, then range-check
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
    fn materialize_noise(&mut self, spec: NoiseSpec, n: usize) -> Vec<Value> {
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
    fn draw_rotation(&mut self, d: usize) -> Result<Value> {
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

    /// Draw an `empirical(xs)` recipe (the iid bootstrap, PLAN-FINANCE F2): one fresh
    /// `unif_int(0, n-1)` index source, gathered over the constant data elements — exactly the
    /// manual idiom `i ~ unif_int(0, Len(xs)-1); xs[i]`. Each `~` (and each leaf of a shaped
    /// `~[n]` draw) builds a fresh index, so repeated draws resample independently. Gather is
    /// interpreter-only (`kernel::walk_cost`) — accepted for bootstrap workloads.
    fn draw_empirical(&mut self, data: DataId) -> Value {
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
    fn draw_block_bootstrap(&mut self, data: DataId, b: usize) -> Value {
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
                Some(n) => levels.push(n),
                None => {
                    return Err(NoiseError::runtime(
                        "quantize centroids must be constant numbers, not random variables"
                            .to_string(),
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
        let t = builtins::call(
            "transpose",
            std::slice::from_ref(m),
            &self.graph,
            self.max_samples,
            self.max_opts,
            span,
            self.check_mode,
        )?;
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
    fn complex_atan2(&mut self, y: Value, x: Value, span: Span) -> Result<Value> {
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

/// If `expr` is a direct `input::<base>(…)` call, return `(base, args)` — the hook the `Expr::Bind`
/// arm uses to infer the input's name from the binding LHS (PLAN-INPUTS §2). `None` otherwise.
fn as_input_call(expr: &Expr) -> Option<(&str, &CallArgs)> {
    if let Expr::Call(name, args) = expr {
        let (module, base) = split_path(name);
        if module == Some("input") {
            return Some((base, args));
        }
    }
    None
}

/// Bind an [`InputValue`] as an engine [`Value`] point mass.
fn input_value_to_value(v: InputValue) -> Value {
    match v {
        InputValue::Num(n) => Value::Num(n),
        InputValue::Bool(b) => Value::Bool(b),
    }
}

/// The always-on variable-introspection builtins (see [`crate::introspect`]). Routed before module
/// resolution because they need `&mut` graph (sampling roots) and the variable scope (`explain`).
#[inline]
fn is_introspection(name: &str) -> bool {
    matches!(
        name,
        "describe" | "hist" | "samples" | "corr" | "scatter" | "explain"
    )
}

/// A short label for an introspected operand, taken from its *source* expression (the evaluated
/// `Value` has no name). An identifier is its own name; a conditioned value reads `event | given`;
/// a call shows `name(…)`; anything else is a generic placeholder.
fn label_of(s: &Spanned) -> String {
    match &s.expr {
        Expr::Ident(name) => name.clone(),
        Expr::Cond { event, given } => format!("{} | {}", label_of(event), label_of(given)),
        Expr::Binary(op, l, r) => format!("{} {} {}", label_of(l), binop_symbol(*op), label_of(r)),
        Expr::Index(arr, idx) => format!("{}[{}]", label_of(arr), label_of(idx)),
        Expr::Call(name, _) => format!("{name}(…)"),
        Expr::Number(n) => crate::value::format_num(*n),
        _ => "value".to_string(),
    }
}

/// The source symbol for a binary operator — for labelling an introspection by its expression
/// (e.g. `D > 3`). Display-only; not a parser/printer.
fn binop_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Pow => "^",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

/// A row of numbers, as a Noise array value.
fn row_of(xs: Vec<f64>) -> Value {
    Value::Array(Rc::new(xs.into_iter().map(Value::Num).collect()))
}

/// A matrix of numbers (rows of equal length), as a Noise array-of-arrays value.
fn matrix_of(rows: impl IntoIterator<Item = Vec<f64>>) -> Value {
    Value::Array(Rc::new(rows.into_iter().map(row_of).collect()))
}

/// Resolve a count argument (the `k` in `samples(x, k)`) to a non-negative integer.
fn introspect_count(v: &Value, span: Span) -> Result<usize> {
    let n = match v {
        Value::Num(n) | Value::Est { val: n, .. } => *n,
        other => {
            return Err(NoiseError::runtime(
                format!("sample count must be a number, got {}", other.type_name()),
                span,
            ))
        }
    };
    if n < 0.0 || n.fract() != 0.0 || !n.is_finite() {
        return Err(NoiseError::runtime(
            format!("sample count must be a non-negative integer, got {n}"),
            span,
        ));
    }
    Ok(n as usize)
}

/// The teaching error for plotting a complex signal/array (PLAN-SIGNALS §4): a complex wave has
/// no single trace — the caller must pick a real view first.
fn complex_has_no_trace(span: Span) -> NoiseError {
    NoiseError::runtime(
        "a complex signal has no single trace to plot — take `math::re(z)`, `math::im(z)`, or \
         `math::abs(z)` first"
            .to_string(),
        span,
    )
}

/// A condition (or query) that yielded no usable draws — `describe(X | C)` / `corr`/`explain` where
/// the condition never held in `n` samples. Mirrors `builtins`' conditional-undefined message.
fn condition_never(n: usize, span: Span) -> NoiseError {
    NoiseError::runtime(
        format!(
            "the condition after `|` never occurred in {n} samples, so there is nothing to \
             summarize — use a more likely condition or raise the sample count"
        ),
        span,
    )
}

/// The set of nodes reachable upstream from `root` (its transitive dependency cone, including
/// `root`). Backs `explain`: a named variable can only drive the target if the target depends on it.
fn ancestors(graph: &RvGraph, root: RvId) -> HashSet<RvId> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match graph.node(id) {
            RvNode::Unary(_, a) => stack.push(*a),
            RvNode::Binary(_, a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Select { cond, a, b } => {
                stack.push(*cond);
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Gather { elems, index } => {
                stack.extend(elems.iter().copied());
                stack.push(*index);
            }
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
        }
    }
    seen
}

/// The built-in modules. `builtin` is always active; the others need a `use`. `plot` (charts) and
/// `stats` (the same numbers, unrendered) are **always qualified** — they are reachable as
/// `plot::hist(X)` / `stats::histogram(X)` without a `use`, and never unqualified, because their
/// short verbs (`fan`, `corr`, `moments`) would otherwise shadow far too much.
const MODULES: [&str; 8] = [
    "rand", "math", "vec", "signal", "engine", "builtin", "plot", "stats",
];

/// The `stats::` functions — the raw-data twin of each `plot::` chart. Not in [`module_of`]: they
/// resolve through the early `stats::` interception in `eval`, so this list exists only to point a
/// bare `fan(path)` at `stats::fan(path)`.
const STATS_FNS: [&str; 5] = ["histogram", "quantiles", "moments", "fan", "corr"];

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
        | "rotation" | "permutation" | "empirical" | "block_bootstrap" => "rand",
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
        | "norm" | "transpose" | "adjoint" | "normalize" | "has_duplicates"
        | "count_duplicates" | "mse" | "ones" | "zeros" | "iota" | "outer" | "quantize" => "vec",
        // prefix scans + the product reducer (PLAN-FINANCE F3): paths as fixed-horizon arrays.
        "prod" | "cumsum" | "cumprod" | "cummax" | "cummin" => "vec",
        // signal generation (DSP waveforms) + colored noise + materialization
        "sine"
        | "cosine"
        | "sample"
        | "noise_white"
        | "noise_white_complex"
        | "noise_brown"
        | "noise_pink"
        | "noise_ou" => "signal",
        // run-time knobs: tune the evaluator itself (e.g. the Monte Carlo budget and the signal
        // resolution). Imperative settings, not value-producing builtins.
        "set_max_samples" | "set_max_opts" | "set_resolution" => "engine",
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
/// on; using it in an expression is an error that points at `~`. An undrawn **noise generator**
/// (`noise_white(1)`) is equally random and subject to the same rule (PLAN-SIGNALS §1.1) — its
/// message additionally offers the length-pinning `~[n]` form.
#[inline]
fn forbid_undrawn(v: &Value, span: Span) -> Result<()> {
    match v {
        Value::Recipe(r) => Err(NoiseError::runtime(
            format!(
                "`{r}` is an undrawn distribution, not a value — draw it first with `~` \
                 (e.g. `X ~ {r}`) and use `X`"
            ),
            span,
        )),
        Value::Noise(spec) => Err(NoiseError::runtime(
            format!(
                "`{spec}` is an undrawn distribution, not a value — draw it first with `~` \
                 (e.g. `static ~ signal::{spec}`), or pin a length with `~[n]`"
            ),
            span,
        )),
        _ => Ok(()),
    }
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

/// Coerce a value to a signal-tree operand: a signal hands over its tree, a constant scalar
/// promotes to a `Konst` leaf. Backs the deferred `atan2` (and any future binary signal node).
fn sig_operand(v: &Value, span: Span) -> Result<Rc<SigExpr>> {
    match v {
        Value::Signal(s) => Ok(s.clone()),
        Value::Num(n) => Ok(Rc::new(SigExpr::Konst(*n))),
        Value::Est { val, .. } => Ok(Rc::new(SigExpr::Konst(*val))),
        other => Err(NoiseError::runtime(
            format!(
                "cannot combine a signal with {} — `signal::sample(sig, n)` it to an array first",
                other.type_name()
            ),
            span,
        )),
    }
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
        UnOp::Exp => "exp",
        UnOp::Ln => "log",
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
        UnOp::Exp => x.exp(),
        UnOp::Ln => x.ln(),
        UnOp::Not => unreachable!("Not is a boolean op, not a numeric ufunc"),
    }
}

/// Spanned arity error shared by the library methods.
fn arity_err(name: &str, want: usize, got: usize, span: Span) -> NoiseError {
    NoiseError::runtime(
        format!("{name} expects {want} argument(s), got {got}"),
        span,
    )
}

/// Spanned vector-length mismatch shared by `dot` and elementwise broadcast.
fn length_mismatch(name: &str, a: usize, b: usize, span: Span) -> NoiseError {
    NoiseError::runtime(
        format!("{name} needs equal-length vectors, got {a} and {b}"),
        span,
    )
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
