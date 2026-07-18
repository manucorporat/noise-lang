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
use crate::dist::{RvGraph, RvId, RvKind, RvNode};
use crate::error::{NoiseError, Result, Span};
use crate::input::{InputSpec, InputValue, ResolvedInput};
use crate::parser::parse;
use crate::sampler::{self, Moments};
use crate::signal::{RealizationId, SigExpr};
use crate::value::Value;

// The evaluator is split along its section seams (finding F1). Each submodule is an
// `impl Engine` block; the free helper fns and module tables stay in this root, where they
// remain visible to every submodule as ancestor items.
mod draw_lift;
mod eval_arms;
mod introspect_dispatch;
mod library;
mod ops;

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

/// Build-time **expression**-recursion budget for [`Engine::eval`]. `MAX_CALL_DEPTH` only guards
/// *user-function* calls; `eval` itself recurses down the left spine of a flat operator chain, so a
/// 10 000-term `1+1+1+…` (which parses fine — the parser's own guard is far higher) would otherwise
/// abort the process in eval. This bounds `eval`'s own recursion. `simplify::rewrite` and
/// `bytecode::compile` walk that *same* left spine when a chain becomes a `Dist` — those are handled
/// separately by the iterative worklists in `simplify`/`bytecode`/`kernel` (finding A4). Sized well
/// above any realistic expression yet below what a small (1–2 MiB) stack can hold.
const MAX_EVAL_DEPTH: usize = 2048;

/// Maximum number of elements a `a..b` range may materialize. A range builds a real array (one
/// `Value` per element), so an unbounded range OOMs or — for `a >= 2^53`, where `+= 1.0` can't
/// advance — loops forever. ~1M elements is far beyond any teaching loop yet cheap to reject.
const RANGE_MAX: usize = 1 << 20;

/// Maximum number of leaves a shaped draw `~[n, m, …]` may build (the product of its dimensions).
/// Each leaf is an independent draw (a fresh source node **and** a `Value`), so `~[1e15]` would
/// `Vec::with_capacity(10^15)` and abort. ~1M leaves bounds construction while leaving every
/// realistic shaped draw (`~[n]`, `~[d, d]`) untouched. Modeled on `complex_pow`'s 4096 cap.
const MAX_DRAW_ELEMS: usize = 1 << 20;

/// The tree-walking evaluator and the crate's central entry point: it lexes/parses a source string,
/// walks the AST (see [`Engine::eval`]), lowers every `~`-drawn random variable into an append-only
/// sample-DAG ([`Engine::graph`]), and forces `P` / `E` / `var` / `describe` queries through the
/// batched sampler.
///
/// **State persists across [`run`](Engine::run) calls.** An `Engine` is not a per-program scratchpad
/// — its variable bindings, user functions, sample-DAG, drawn-noise realizations, input manifest,
/// and buffered output stream all accumulate and survive between runs. This is deliberate: the
/// browser playground's introspection sidecar runs a program, then issues follow-up
/// `describe`/`corr` queries against the *retained* scope of that run (the same graph and roots), so
/// the results are consistent with what the program computed. Call [`Engine::new`] for a fresh,
/// empty engine; drain buffered output with [`Engine::take_output`]. Program-tunable settings
/// (`engine::set_precision` / `set_resolution`) are stored here too and likewise
/// persist for the engine's lifetime.
/// State of an in-progress shaped draw `~[n, …] recipe` — what turns `leaves` independent
/// [`RvNode::Src`] nodes into ONE [`RvNode::ArrDraw`] block plus `leaves` cheap
/// [`RvNode::ArrElem`] reads (PLAN-WEBGPU G½).
///
/// The redirection happens at the single point where a recipe's *base source* is pushed
/// ([`Engine::push_src`]), rather than in `draw_shaped` — which is why it works for every recipe and
/// not just the five obvious ones. A recipe is a little cone with one or more base sources under
/// some deterministic transform (`normal_int` is a `Normal` under a `Round`; `bernoulli(p)` is a
/// `Uniform` under a `Lt`; the hierarchical `*Dyn` family is a standard source under an affine map).
/// All of them push their bases through `push_src`, in a fixed order, so:
///
///   * on **leaf 0**, each base source push allocates the next `ArrDraw` block (recording its
///     recipe), and
///   * on **leaves 1..n**, the same push finds that block already there and just reads element `k`.
///
/// `pos` resets per leaf, so base *j* of every leaf lands in block *j*. The blocks are `n` wide, so
/// element `k` of block `j` gets draw ordinal `base_j + k` — exactly the stream the `n` separate
/// `Src` nodes drew from before this existed.
///
/// The array-valued recipes (`rotation`, `permutation`) never touch `push_src`: they push their own
/// whole-array source node and are already one node per leaf, which was never the problem.
struct ShapedDraw {
    /// Leaves in the whole shaped draw — the width of every block (`~[3,4]` is one 12-wide block,
    /// not three 4-wide ones, so a matrix draw is a single loop on the GPU too).
    leaves: u32,
    /// Which leaf is being built (`0..leaves`) — the element index into every block.
    k: u32,
    /// One `ArrDraw` per base-source position in the recipe's cone, filled on leaf 0.
    blocks: Vec<RvId>,
    /// Base-source position within the current leaf; reset to 0 at each leaf.
    pos: usize,
}

pub struct Engine {
    vars: HashMap<String, Value>,
    /// User functions live in their own namespace (a call resolves here before builtins).
    funcs: HashMap<String, Rc<UserFn>>,
    /// Append-only sample-DAG arena (Phase 2). Built during `run`; read-only when sampling.
    graph: RvGraph,
    /// Active shaped draw, if we are inside one (`~[n] recipe`). See [`ShapedDraw`].
    shaped: Option<ShapedDraw>,
    /// Current user-function call depth (guarded by `MAX_CALL_DEPTH`).
    call_depth: usize,
    /// Current [`Engine::eval`] expression-recursion depth (guarded by `MAX_EVAL_DEPTH`). Distinct
    /// from `call_depth`: this tracks the tree-walk itself (a deep flat `1+1+…` chain), not calls.
    eval_depth: usize,
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
    /// Document-wide **precision target** `(rel, abs)` (PLAN-PRECISION): an untargeted
    /// `P`/`E`/`Var` keeps drawing until `se ≤ max(abs, rel·|est|)`, bounded by `max_time`.
    /// `None` = the auto default (Phase 2: `se ≤ 5e-3·max(|est|, sd)` —
    /// [`crate::sampler::DrawBudget::Auto`]). Declared by the `engine::set_precision` pragma; a host
    /// override ([`Engine::set_precision`]) pins it (a pragma no longer changes the effective
    /// value — "pragmas declare, `run()` overrides").
    precision: Option<(f64, f64)>,
    /// The run's wall-clock ceiling (PLAN-PRECISION): past it, the in-flight query soft-stops
    /// (honest partial estimate + warning) and the remaining forcings are skipped. Runtime-only —
    /// deliberately **no pragma** (a `max_time` in program text would make a program's digits
    /// machine-dependent while looking like part of the question). `None` = no deadline.
    max_time: Option<std::time::Duration>,
    /// Which settings a host override has **pinned** (PLAN-PRECISION "pragmas declare, `run()`
    /// overrides"): a pragma assignment to a pinned setting is evaluated but doesn't change the
    /// effective value (a `NOISE_PROFILE=1` note records the shadowing).
    pinned_precision: bool,
    pinned_resolution: bool,
    /// The run's **warnings** (PLAN-PRECISION Track C): capped queries ("max_time stopped it short
    /// of the target"), soft-stopped runs. In a `RefCell` so the immutable `QueryCtx` can push
    /// from inside a builtin dispatch. Drained into `DocResult.warnings` per run.
    warnings: std::cell::RefCell<Vec<builtins::Warning>>,
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
    /// never met an explicit length — the time-axis measurement knob.
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
    /// The run's **runtime input values**, one f32 per manifest slot (`value as f32`; bool → 0/1),
    /// parallel to `input_manifest` (PLAN-UNIFORM-INPUTS). Populated as `input::` declarations
    /// resolve, and *installed* thread-locally around each forcing region so the free forcing
    /// functions (`sampler`/`reduce`) can hand `RvNode::Input` its slot value at run time without an
    /// engine parameter. Owned here (like the compile cache) so it drops with the engine.
    inputs: crate::input_rt::Inputs,
    /// Per-engine run-time counters (finding B8). Owned here — not a thread-local global — so two
    /// engines on one thread (the playground sidecar pattern) keep independent stats and reading
    /// them doesn't couple to whichever thread last forced. Installed as the thread's active
    /// recorder around each forcing region (see [`crate::stats`]).
    stats: crate::stats::Counters,
    /// Per-engine compiled-program cache (PLAN-PERF-2 §4): identical forcings (same simplified
    /// cone, same gate decision) share ONE compile, so an introspection pass re-forcing a root, a
    /// REPL, or a playground re-run on this persistent engine stop recompiling. Owned here (dropped
    /// with the engine) and installed as the
    /// thread's active cache around each forcing region, exactly like `stats` (see
    /// [`crate::compile_cache`]). Purely an optimization: results are bit-identical with or
    /// without a hit.
    compile_cache: crate::compile_cache::Cache,
    /// Per-engine readable **phase-timing accumulator** (the profiler's document path). Owned here
    /// like `stats`/`compile_cache`; installed as the thread's active accumulator around a run *only
    /// when* [`profile_enabled`](Engine::set_profiling) is set, so ordinary runs pay nothing. Each
    /// forcing folds its per-phase wall times (`compile`/`reduce`/`sample`/`gpu.*`) and notes in;
    /// [`run_to_document`](Engine::run_to_document) snapshots it into `DocResult.profile` for a host
    /// (the playground) to render. See [`crate::profile`].
    profile: crate::profile::Profile,
    /// Whether this engine captures timings into `profile` for the document. Off by default — the
    /// native `NOISE_PROFILE=1` stderr path is independent of this. A host opts in with
    /// [`set_profiling`](Engine::set_profiling) (the playground passes `profile: true` in run opts).
    profile_enabled: bool,
    /// This engine's cancellation token (PLAN-PREGPU Track A), installed as the thread's active
    /// token for the duration of every run so the reducer's per-chunk check can see it. A host
    /// clones it ([`Engine::cancel_token`]) and trips it from anywhere — another thread natively,
    /// or an `AbortSignal` listener in the browser — and the in-flight forcing aborts with
    /// `ErrorKind::Cancelled`.
    ///
    /// **Not** reset per run: a host must be able to grab the token *before* starting a run (that
    /// is the whole point — you cancel a run that is already going), so resetting it at the top of
    /// `run` would orphan the clone the host is holding. A cancelled engine therefore stays
    /// cancelled until [`Engine::reset_cancel`] — which is the honest default, since a cancelled
    /// run leaves the scope half-updated and the documented advice is to rebuild from a fresh
    /// engine anyway.
    cancel: crate::exec::CancelToken,
}

/// One item in a program's output stream — the unit `take_output` returns, in source order. `Print`
/// pushes a [`Text`](Output::Text) line; `plot::*` pushes a [`Plot`](Output::Plot). Keeping them in
/// one ordered vector is what lets text and charts interleave exactly as the program emitted them.
#[derive(Debug, Clone)]
#[non_exhaustive] // new output kinds (charts, controls, …) land across plan cycles; keep hosts wildcard-safe (E2)
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
            shaped: None,
            vars: HashMap::new(),
            funcs: HashMap::new(),
            graph: RvGraph::default(),
            call_depth: 0,
            eval_depth: 0,
            used: HashSet::new(),
            outputs: Vec::new(),
            current_stmt_span: Span::new(0, 0),
            dropped: 0,
            first_dropped_span: None,
            precision: None,
            max_time: None,
            pinned_precision: false,
            pinned_resolution: false,
            warnings: std::cell::RefCell::new(Vec::new()),
            check_mode: false,
            realizations: HashMap::new(),
            next_realization: 0,
            resolution: builtins::RESOLUTION_DEFAULT,
            datasets: Vec::new(),
            input_overrides: Vec::new(),
            input_manifest: Vec::new(),
            inputs: crate::input_rt::new_inputs(),
            stats: crate::stats::new_counters(),
            compile_cache: crate::compile_cache::new_cache(),
            profile: crate::profile::new_profile(),
            profile_enabled: false,
            cancel: crate::exec::CancelToken::new(),
        }
    }

    /// Opt this engine into **phase profiling** for the document (PLAN-DROP-JIT D0 profiler, made
    /// readable). When on, [`run_to_document`](Engine::run_to_document) times each forcing's phases
    /// (`compile`/`reduce`/`sample`/`gpu.*`) and returns them in `DocResult.profile` for a host to
    /// render. Off by default and cheap when off (no clock reads). Independent of the native
    /// `NOISE_PROFILE=1` stderr path.
    pub fn set_profiling(&mut self, on: bool) {
        self.profile_enabled = on;
    }

    /// Bundle the current engine knobs into a [`builtins::QueryCtx`] for a builtin/query dispatch at
    /// `span` (finding F6). One place threads `graph`/`precision`/`check_mode`, so the
    /// dispatch call sites don't repeat the knob list.
    pub(crate) fn query_ctx(&self, span: Span) -> builtins::QueryCtx<'_> {
        builtins::QueryCtx {
            graph: &self.graph,
            precision: self.precision,
            check: self.check_mode,
            span,
            warnings: &self.warnings,
        }
    }

    /// Host override: the **precision target** `(rel, abs)` — keep drawing until
    /// `se ≤ max(abs, rel·|est|)` (PLAN-PRECISION). `None` turns targeting off. Pins the setting
    /// over any `engine::set_precision` pragma in the program.
    pub fn set_precision(&mut self, target: Option<(f64, f64)>) {
        self.precision = target;
        self.pinned_precision = true;
    }

    /// Host override: the run's wall-clock ceiling (**`max_time`**, PLAN-PRECISION). Past the
    /// deadline the in-flight query soft-stops — partial results with honest se plus a warning —
    /// and the remaining forcings are skipped. `None` = no deadline. Runtime-only by design (no
    /// pragma twin: a deadline in program text would make a program's digits machine-dependent).
    pub fn set_max_time(&mut self, limit: Option<std::time::Duration>) {
        self.max_time = limit;
    }

    /// Host override: the ambient **sampling resolution** (the length reducers render a lazy
    /// signal at when it never met an explicit length). Pins the setting over the program's
    /// `engine::set_resolution` pragma.
    pub fn set_resolution(&mut self, n: usize) {
        self.resolution = n.max(1);
        self.pinned_resolution = true;
    }

    /// Ask the running program to **soft-stop** (PLAN-PRECISION Track H): the in-flight forcing
    /// folds the chunks that completed and returns an honest partial estimate, the remaining
    /// forcings are skipped, and the run returns a complete document with a "stopped early"
    /// warning. The counterpart of [`Engine::cancel`] (hard abort, everything discarded). Callable
    /// from any thread via a clone of [`Engine::cancel_token`].
    pub fn stop(&self) {
        self.cancel.stop();
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
        self.stats.get()
    }

    /// A clone of this engine's [`CancelToken`](crate::exec::CancelToken) — the handle a host uses
    /// to **stop a run that is already going** (PLAN-PREGPU Track A).
    ///
    /// Grab it *before* starting the run (that is the only order that makes sense), then trip it
    /// from wherever the stop signal comes from: another thread natively (a CLI `Ctrl-C` handler, a
    /// watchdog, a request timeout), or — once Track A's async spine lands — an `AbortSignal`
    /// listener in the browser. The in-flight run aborts with `ErrorKind::Cancelled`: the reducer
    /// notices within one 16,384-sample chunk, so even a 10⁷-draw `P(...)` stops in milliseconds
    /// rather than running to completion.
    ///
    /// ```
    /// # use noise_core::Engine;
    /// let mut eng = Engine::new();
    /// let token = eng.cancel_token();
    /// std::thread::spawn(move || token.cancel()); // e.g. a Ctrl-C handler
    /// # let _ = eng.run("1 + 1");
    /// ```
    ///
    /// **A cancelled engine is stale.** Bindings that completed before the abort persist, so the
    /// scope is half-updated; rebuild from a fresh [`Engine`] rather than trusting it. (The
    /// playground's introspection sidecar, which relies on scope surviving across runs, must do
    /// exactly this after a cancel.)
    pub fn cancel_token(&self) -> crate::exec::CancelToken {
        self.cancel.clone()
    }

    /// Trip this engine's token directly — the same thing [`cancel_token`](Engine::cancel_token)
    /// enables, for a host that already holds the `&Engine` (a nested/embedding case).
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Give this engine a *fresh*, un-cancelled token, orphaning any clone a host still holds.
    /// Only meaningful when deliberately reusing an engine after a cancel — see the staleness
    /// caveat on [`cancel_token`](Engine::cancel_token).
    pub fn reset_cancel(&mut self) {
        self.cancel = crate::exec::CancelToken::new();
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
    ///
    /// # Examples
    ///
    /// ```
    /// use noise_core::{Engine, Value};
    ///
    /// let mut engine = Engine::new();
    /// // Roll a fair die and estimate P(roll == 4) by Monte Carlo. `use rand;` brings the
    /// // distribution constructors into scope; `P` is an always-on builtin.
    /// let value = engine.run("use rand; D ~ unif_int(1, 6); P(D == 4)").unwrap();
    /// match value {
    ///     // `P` returns an estimate carrying a standard error; the point value is ~1/6.
    ///     Value::Est { val, .. } => assert!((val - 1.0 / 6.0).abs() < 0.01),
    ///     other => panic!("expected a probability estimate, got {other:?}"),
    /// }
    /// ```
    pub fn run(&mut self, src: &str) -> Result<Value> {
        // Fresh run-time counters for this program (the playground reads them after `run`), and
        // install this engine's counters as the thread's recorder — and its compile cache as the
        // thread's active cache — for the whole run (finding B8 / PLAN-PERF-2 §4).
        let _rec = crate::stats::install(&self.stats);
        let _cache = crate::compile_cache::install(&self.compile_cache);
        let _cancel = crate::exec::install(&self.cancel);
        let _inputs = crate::input_rt::install(&self.inputs);
        // The soft-stop flag and the deadline are per-run (PLAN-PRECISION): a fresh run starts
        // unstopped, and `max_time` counts from here.
        self.cancel.reset_stop();
        crate::exec::reset_host_stop();
        self.warnings.borrow_mut().clear();
        let _deadline =
            crate::exec::install_deadline(self.max_time.map(|d| web_time::Instant::now() + d));
        self.stats.set(crate::stats::RunStats::default());
        self.dropped = 0;
        self.first_dropped_span = None;
        self.input_manifest.clear();
        self.inputs.borrow_mut().clear();
        let program = parse(src)?;
        let mut last = Value::Unit;
        for stmt in &program.stmts {
            // Statement boundary: the coarse cancellation tier (the fine one is per reducer chunk,
            // inside `reduce::run_reduction`). Catches a token tripped between forcings — e.g.
            // during a long deterministic loop — without waiting for the next sampling pass.
            if self.cancel.is_cancelled() {
                return Err(NoiseError::cancelled());
            }
            last = self.eval_top(stmt)?;
            // Soft-stop boundary (Track H): a stopped/deadlined run keeps the values computed so
            // far (the in-flight query already returned its honest partial estimate) and skips the
            // remaining statements, returning a complete result plus a warning.
            if let Some(cause) = crate::exec::stop_cause() {
                self.warn_stopped_early(cause, stmt.span);
                break;
            }
        }
        self.check_input_overrides()?;
        // A program whose result is a bare tunable input (`sides = input::real(…); sides`) evaluates
        // to a symbolic value; realize it to its current number for the host now, while the input
        // values are still installed (PLAN-UNIFORM-INPUTS). Intermediate `Sym`s stay symbolic — only
        // the returned value is concretized.
        Ok(self.realize_result(last))
    }

    /// Record the run-level "stopped early" warning (Track H): which rung stopped it, at which
    /// statement. Complements (never replaces) a capped query's own warning: that one says what the
    /// in-flight estimate got, this one says the rest of the program was skipped.
    fn warn_stopped_early(&self, cause: crate::exec::StopCause, span: Span) {
        let what = match cause {
            crate::exec::StopCause::User => "stop()",
            crate::exec::StopCause::Time => "max_time",
        };
        self.warnings.borrow_mut().push(builtins::Warning {
            message: format!(
                "run stopped early by {what} — statements after this one were skipped; values \
                 computed so far are reported with their honest standard errors"
            ),
            span,
        });
    }

    /// The warnings the most recent run raised (PLAN-PRECISION Track C): capped queries, an early
    /// stop. Message text only; `run_to_document` also renders them with line numbers into
    /// `DocResult.warnings`.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings
            .borrow()
            .iter()
            .map(|w| w.message.clone())
            .collect()
    }

    /// Concretize a top-level `Value::Sym` to its current `Value::Num` (folded against the installed
    /// input values); every other value passes through. Applied to a run's final result so a host
    /// sees a number, not a symbolic handle whose value is only defined while inputs are installed.
    fn realize_result(&self, v: Value) -> Value {
        match v {
            Value::Sym(s) => Value::Num(self.force_sym(&s)),
            other => other,
        }
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
        let _rec = crate::stats::install(&self.stats);
        let _cache = crate::compile_cache::install(&self.compile_cache);
        let _cancel = crate::exec::install(&self.cancel);
        let _inputs = crate::input_rt::install(&self.inputs);
        // Per-run soft-stop + deadline (PLAN-PRECISION; see `run`).
        self.cancel.reset_stop();
        crate::exec::reset_host_stop();
        self.warnings.borrow_mut().clear();
        let _deadline =
            crate::exec::install_deadline(self.max_time.map(|d| web_time::Instant::now() + d));
        // Profiling is opt-in (`set_profiling`): only then do we reset + install the accumulator,
        // which is what turns per-phase timing on for this run. Off → the guard is `None` and every
        // `profile::span` stays inert.
        if self.profile_enabled {
            *self.profile.borrow_mut() = crate::profile::Timings::default();
        }
        let _profile = self
            .profile_enabled
            .then(|| crate::profile::install(&self.profile));
        self.stats.set(crate::stats::RunStats::default());
        self.dropped = 0;
        self.first_dropped_span = None;
        self.input_manifest.clear();
        self.inputs.borrow_mut().clear();

        // Frontmatter first — a malformed fence still yields a shaped document (no meta, spanned error).
        let fm = match crate::frontmatter::parse(src) {
            Ok(fm) => fm.map(|(fm, _)| fm),
            Err(e) => return Document::error_only(None, e, self.stats(), src),
        };
        let program = match parse(src) {
            Ok(p) => p,
            Err(e) => return Document::error_only(fm, e, self.stats(), src),
        };

        // Pure segmentation + comment attachment (no evaluation).
        let segs = crate::doc::segment(src, &program.stmts);
        let comments = match crate::doc::comment_layer(src, &program.stmts) {
            Ok(c) => c,
            // A trivia re-lex can only fail the same way `parse` already succeeded, but stay total.
            Err(e) => return Document::error_only(fm, e, self.stats(), src),
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
            // Soft-stop boundary (Track H): keep everything computed so far, skip the rest, and
            // return a COMPLETE document (values + honest se + the "stopped early" warning).
            if let Some(cause) = crate::exec::stop_cause() {
                self.warn_stopped_early(cause, stmt.span);
                break;
            }
        }
        // A stray input override is a post-run error (the manifest exists only now); it never
        // discards the blocks that ran, and defers to a real runtime error if one occurred first.
        if error.is_none() {
            if let Err(e) = self.check_input_overrides() {
                error = Some(e);
            }
        }
        // Realize a bare tunable-input result to its number while inputs are still installed (see
        // `realize_result`).
        let last = self.realize_result(last);
        let emissions = self.take_output();
        let inputs = std::mem::take(&mut self.input_manifest);
        let truncated = self.first_dropped_span.map(|span| (self.dropped, span));
        // Snapshot the accumulated phase timings (only when profiling was on; `None` otherwise, so
        // `DocResult.profile` is absent for ordinary runs). Drop the guard first isn't needed — the
        // cell is an `Rc`, we just read it.
        let profile = self.profile_enabled.then(|| self.profile.borrow().clone());
        // Render the run's warnings with their 1-based source lines (PLAN-PRECISION Track C) —
        // the CLI prints these to stderr, the playground shows them like its other diagnostics.
        let warnings = self
            .warnings
            .borrow()
            .iter()
            .map(|w| {
                let (line, _col) = w.span.line_col(src);
                format!("line {line}: {}", w.message)
            })
            .collect();
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
            profile,
            warnings,
        )
    }

    /// Convenience alias of [`Engine::run`]: the last statement's value is the RV.
    /// Tests do `let rv = eng.run_rv("X ~ unif(-1,1); X ^ 2")?;`.
    #[must_use = "run_rv returns the program's value/error; discarding it ignores both (finding F10)"]
    pub fn run_rv(&mut self, src: &str) -> Result<Value> {
        self.run(src)
    }

    /// Validate a program without running its Monte Carlo: parse it, evaluate every statement, and
    /// build the sample-DAG — surfacing syntax, scope, type, and shape errors — but skip the actual
    /// sampling in `P`/`E`/`Var`/`Q` (see [`Engine::check_mode`]). Returns the last value (whose
    /// estimator results are placeholders) so callers can just check for `Ok`. Fast regardless of
    /// the configured sample budget.
    #[must_use = "check returns the validation result; discarding it ignores any error (finding F10)"]
    pub fn check(&mut self, src: &str) -> Result<Value> {
        self.check_mode = true;
        let result = self.run(src);
        self.check_mode = false;
        result
    }

    /// The central recursion: evaluate one AST node to a [`Value`], dispatching on its expression
    /// kind (literals, operators, `~`/`=` bindings, calls, `if`/blocks/`for`, arrays, comprehensions,
    /// …). Every sub-expression flows back through here, so this is the hot path the whole evaluator
    /// is built around; larger arms are split into `eval_*` helpers to keep this frame small.
    ///
    /// It also guards `eval`'s own recursion depth (see [`MAX_EVAL_DEPTH`]): a deep flat chain like
    /// `1+1+1+…` recurses down its left spine one frame per term, and without this it would abort the
    /// process. On over-limit it returns a spanned error rather than crashing.
    fn eval(&mut self, node: &Spanned) -> Result<Value> {
        self.eval_depth += 1;
        if self.eval_depth > MAX_EVAL_DEPTH {
            self.eval_depth -= 1;
            return Err(NoiseError::runtime(
                format!(
                    "expression nests too deeply to evaluate (over {MAX_EVAL_DEPTH} levels) — a very \
                     long operator chain or deeply nested expression; split it into smaller bindings"
                ),
                node.span,
            ));
        }
        let r = self.eval_inner(node);
        self.eval_depth -= 1;
        r
    }

    fn eval_inner(&mut self, node: &Spanned) -> Result<Value> {
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
                    // A symbolic condition (`if slider > 1 { … }`) is deterministic given the
                    // inputs — a 0/1 indicator tree — so it short-circuits like a plain bool.
                    // Structural: the branch is baked, so the program rebuilds when the slider
                    // crosses the boundary (see `Engine::select`'s matching arm).
                    Value::Sym(ref s) => {
                        if self.force_sym(s) != 0.0 {
                            self.eval(then_b)
                        } else {
                            match else_b {
                                Some(eb) => self.eval(eb),
                                None => Ok(Value::Unit),
                            }
                        }
                    }
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
        // Account this forcing into the engine's own counters (finding B8), through its cache.
        let _rec = crate::stats::install(&self.stats);
        let _cache = crate::compile_cache::install(&self.compile_cache);
        let _cancel = crate::exec::install(&self.cancel);
        let _inputs = crate::input_rt::install(&self.inputs);
        sampler::sample_n(&self.graph, id, n, seed)
    }

    /// Rust sampling API (for tests): empirical mean + population variance of the RV `v`.
    pub fn moments(&self, v: &Value, n: usize, seed: u64) -> Result<Moments> {
        let id = self.expect_dist(v)?;
        let _rec = crate::stats::install(&self.stats);
        let _cache = crate::compile_cache::install(&self.compile_cache);
        let _cancel = crate::exec::install(&self.cancel);
        let _inputs = crate::input_rt::install(&self.inputs);
        sampler::moments(&self.graph, id, n, seed)
    }
}

/// Whether a value is a random variable (cheap pre-check for the lifting decision).
#[inline]
fn is_dist(v: &Value) -> bool {
    matches!(v, Value::Dist(_))
}

thread_local! {
    /// Whether `for` loops are captured as rolled [`crate::dist::RvNode::Scan`] nodes (G4c). On by
    /// default; a test flips it off to compare a rolled program against the unrolled baseline and
    /// prove the two produce bit-identical results.
    static LOOP_CAPTURE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Whether the G4c loop-capture is active on this thread (see [`LOOP_CAPTURE`]).
pub(crate) fn loop_capture_enabled() -> bool {
    LOOP_CAPTURE.with(std::cell::Cell::get)
}

/// Enable/disable loop capture on this thread; returns the previous value. Test-only knob so a rolled
/// run can be checked against the unrolled one.
pub fn set_loop_capture(on: bool) -> bool {
    LOOP_CAPTURE.with(|c| c.replace(on))
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
    INTROSPECT_FNS.contains(&name)
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
        // A tunable input as a sample count is a structural use — fold to its current value.
        Value::Sym(s) => sym_current(s),
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
            RvNode::ArrIndex { arr, index } => {
                stack.push(*arr);
                stack.push(*index);
            }
            RvNode::ArrElem { arr, .. } => stack.push(*arr),
            // A Scan depends on its carried inits and on whatever its body reads (loop-invariant
            // nodes, reached through `nexts`). A `Placeholder` is a leaf. `ScanOut` reads a Scan.
            RvNode::Scan { body } => {
                stack.extend(body.inits.iter().copied());
                stack.extend(body.nexts.iter().copied());
            }
            RvNode::ScanOut { scan, .. } => stack.push(*scan),
            RvNode::Placeholder { .. } => {}
            RvNode::Src(_)
            | RvNode::ConstNum(_)
            | RvNode::ConstBool(_)
            | RvNode::Input { .. }
            | RvNode::Permutation { .. }
            | RvNode::Rotation { .. }
            | RvNode::ArrDraw { .. } => {}
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

/// The `stats::` functions — the raw-data twin of each `plot::` chart. Not in [`BUILTINS`]: they
/// resolve through the early `stats::` interception in `eval`, so this list exists only to point a
/// bare `fan(path)` at `stats::fan(path)`. Kept in sync with [`Engine::stats_call`]'s dispatch by
/// the registry coverage test (finding F2).
const STATS_FNS: [&str; 5] = ["histogram", "quantiles", "moments", "fan", "corr"];

/// The always-on **introspection** builtins — reachable unqualified without a `use` (they precede
/// module resolution in `eval`). Single source for [`is_introspection`] and the coverage test
/// (finding F2). `plot::`-only verbs (`line`/`heatmap`/`value`/`show`/`dist`/`fan`) are *not* here —
/// they exist only under `plot::` (see [`PLOT_FNS`]).
const INTROSPECT_FNS: [&str; 6] = ["describe", "hist", "samples", "corr", "scatter", "explain"];

/// The `plot::` chart verbs. A superset of [`INTROSPECT_FNS`] (each introspection is also plottable)
/// plus the chart-only intents (`line`/`heatmap`/`value`/`show`/`dist`/`fan`). Single source for
/// [`Engine::plot_call`]'s dispatch/hint and the coverage test (finding F2).
const PLOT_FNS: [&str; 13] = [
    "histogram",
    "hist",
    "line",
    "heatmap",
    "value",
    "show",
    "dist",
    "describe",
    "scatter",
    "corr",
    "explain",
    "samples",
    "fan",
];

/// The builtin/constant registry — the single enumerable source of truth for **name scoping**
/// (finding F2): each `(name, module)` pair says which module owns a name. `module_of` is a lookup
/// into this table, and the registry coverage test walks it to prove every registered name actually
/// dispatches (and appears in the editor grammar), so the ≥7 formerly-disjoint name tables can't
/// silently drift. The *implementation* of a name (`lib_call` vs `builtins::call`) is orthogonal;
/// this governs only access.
///
/// Module builtins are **lowercase** (`sum`, `mse`, `normal`). The always-on core (`P`, `Q`, `E`,
/// `Var`, `Print`, `Len`) is the lone exception — it is **capital-only**. The math **constants**
/// `pi`/`e` are lowercase — note `E` (capital) is the expectation builtin while `e` (lowercase) is
/// Euler's number, so these two are intentionally distinct and never aliased.
const BUILTINS: &[(&str, &str)] = &[
    // --- rand: distribution constructors, incl. `rotation`/`permutation` (recipes drawn with `~`).
    ("unif", "rand"),
    ("unif_int", "rand"),
    ("bernoulli", "rand"),
    ("normal", "rand"),
    ("normal_int", "rand"),
    ("normal_complex", "rand"),
    ("exponential", "rand"),
    ("exponential_int", "rand"),
    ("poisson", "rand"),
    ("geometric", "rand"),
    ("categorical", "rand"),
    ("rotation", "rand"),
    ("permutation", "rand"),
    ("empirical", "rand"),
    ("block_bootstrap", "rand"),
    // --- math: constants (pi/e real, i/j the imaginary unit), then real + complex-aware ufuncs.
    ("pi", "math"),
    ("e", "math"),
    ("i", "math"),
    ("j", "math"),
    ("sqrt", "math"),
    ("round", "math"),
    ("log", "math"),
    ("log10", "math"),
    ("sin", "math"),
    ("cos", "math"),
    ("atan", "math"),
    ("sign", "math"),
    ("gcd", "math"),
    ("modpow", "math"),
    ("exp", "math"),
    ("abs", "math"),
    ("arg", "math"),
    ("conj", "math"),
    ("re", "math"),
    ("im", "math"),
    ("floor", "math"),
    ("ceil", "math"),
    // --- vec: collections / linear algebra (vector add/sub/matvec are the `+`/`-`/`@` operators).
    ("sum", "vec"),
    ("count", "vec"),
    ("any", "vec"),
    ("all", "vec"),
    ("max", "vec"),
    ("min", "vec"),
    ("mean", "vec"),
    ("dot", "vec"),
    ("vdot", "vec"),
    ("normsq", "vec"),
    ("norm", "vec"),
    ("transpose", "vec"),
    ("adjoint", "vec"),
    ("normalize", "vec"),
    ("has_duplicates", "vec"),
    ("count_duplicates", "vec"),
    ("mse", "vec"),
    ("ones", "vec"),
    ("zeros", "vec"),
    ("iota", "vec"),
    ("outer", "vec"),
    ("quantize", "vec"),
    ("onehot", "vec"),
    // prefix scans + the product reducer (PLAN-FINANCE F3): paths as fixed-horizon arrays.
    ("prod", "vec"),
    ("cumsum", "vec"),
    ("cumprod", "vec"),
    ("cummax", "vec"),
    ("cummin", "vec"),
    // --- signal: DSP waveforms + colored noise generators + materialization.
    ("sine", "signal"),
    ("cosine", "signal"),
    ("sample", "signal"),
    ("noise_white", "signal"),
    ("noise_white_complex", "signal"),
    ("noise_brown", "signal"),
    ("noise_pink", "signal"),
    ("noise_ou", "signal"),
    // --- engine: program-declared settings (PLAN-PRECISION: pragmas declare, `run()` overrides).
    //     The budget pragmas (`set_max_samples`/`set_max_ops`/`set_max_opts`) were REMOVED by
    //     PLAN-PRECISION — calling one is a spanned error naming its replacement (see
    //     `REMOVED_PRAGMAS`).
    ("set_precision", "engine"),
    ("set_resolution", "engine"),
    // --- builtin: always-on core (capital-only). No `range`/`push` (arrays are fixed-size).
    ("P", "builtin"),
    ("Q", "builtin"),
    ("E", "builtin"),
    ("Var", "builtin"),
    ("Print", "builtin"),
    ("Len", "builtin"),
];

/// Whether `m` names a known module.
#[inline]
fn is_module(m: &str) -> bool {
    MODULES.contains(&m)
}

/// The module a builtin/constant belongs to — a lookup into the [`BUILTINS`] registry (finding F2).
fn module_of(name: &str) -> Option<&'static str> {
    BUILTINS
        .iter()
        .find_map(|&(n, m)| if n == name { Some(m) } else { None })
}

/// The names registered under module `m` — the "did you mean?" candidate set for an unknown
/// `m::base` call.
fn module_fns(m: &str) -> impl Iterator<Item = &'static str> + '_ {
    BUILTINS
        .iter()
        .filter_map(move |&(n, module)| (module == m).then_some(n))
}

/// The closest name in `candidates` to `misspelled`, if one is near enough to be worth
/// suggesting. The budget scales with the typo's length (~one edit per three characters, at
/// least one), so a genuinely unrelated name yields `None` rather than a misleading hint.
fn nearest_name<'a>(
    misspelled: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> Option<&'a str> {
    let budget = (misspelled.chars().count() / 3).max(1);
    candidates
        .map(|c| (edit_distance(misspelled, c), c))
        .filter(|&(d, _)| d <= budget)
        .min_by_key(|&(d, _)| d)
        .map(|(_, c)| c)
}

/// Levenshtein edit distance (insert/delete/substitute), for the "did you mean?" suggester.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Pragmas deleted by PLAN-PRECISION, each with the error naming its replacement. Calling one —
/// qualified or bare — is a spanned error, so an old program fails with a migration hint rather
/// than "unknown function".
const REMOVED_PRAGMAS: &[(&str, &str)] = &[
    (
        "set_max_samples",
        "engine::set_max_samples was removed — untargeted queries now draw to a default \
         precision; ask for digits with `engine::set_precision(rel)` or pass a per-call count \
         `P(e, n)`",
    ),
    (
        "set_max_ops",
        "engine::set_max_ops was removed — the op budget is gone; use `engine::set_precision(rel)` \
         to ask for digits, and the runtime `max_time` guard (--max-time) to bound wall-clock",
    ),
    (
        "set_max_opts",
        "engine::set_max_opts was removed — the op budget is gone; use `engine::set_precision(rel)` \
         to ask for digits, and the runtime `max_time` guard (--max-time) to bound wall-clock",
    ),
];

/// The removal notice for a deleted pragma name, if `name` is one (see [`REMOVED_PRAGMAS`]).
fn removed_pragma(name: &str) -> Option<&'static str> {
    REMOVED_PRAGMAS
        .iter()
        .find_map(|&(n, msg)| if n == name { Some(msg) } else { None })
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
        Value::Recipe(r) => Err(NoiseError::not_drawn(
            format!(
                "`{r}` is an undrawn distribution, not a value — draw it first with `~` \
                 (e.g. `X ~ {r}`) and use `X`"
            ),
            span,
        )),
        Value::Noise(spec) => Err(NoiseError::not_drawn(
            format!(
                "`{spec}` is an undrawn distribution, not a value — draw it first with `~` \
                 (e.g. `static ~ signal::{spec}`), or pin a length with `~[n]`"
            ),
            span,
        )),
        _ => Ok(()),
    }
}

/// Reject the `continue` control sentinel in a **data position** (finding F8). `continue` is a loop
/// control statement — it short-circuits the enclosing `{ block }` / comprehension body — not a
/// value: it may not be bound (`x = continue`), stored in an array (`[1, continue, 3]`), or passed
/// as a call argument (`f(continue)`). Caught at the data-entry point with an accurate span, so the
/// error points at the misuse rather than surfacing later as a baffling "arithmetic on continue".
#[inline]
fn forbid_continue(v: &Value, span: Span) -> Result<()> {
    if matches!(v, Value::Continue) {
        return Err(NoiseError::runtime(
            "`continue` is a loop control statement, not a value — it can only appear as a \
             statement inside a `for`/comprehension body, not be bound, stored, or passed as an \
             argument"
                .to_string(),
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
        // A symbolic input meeting a signal folds to its current value here (P0 fallback: the value
        // bakes into the signal `Konst`, so a signal-param input still recompiles on change —
        // signal-uniform lowering is PLAN-UNIFORM-INPUTS P2). Uses the installed input values.
        Value::Sym(s) => Some(sym_current(s)),
        _ => None,
    }
}

/// Fold a symbolic input against the thread-installed runtime input values (PLAN-UNIFORM-INPUTS).
/// The free-function counterpart of `Engine::force_sym`, for helpers with no `&self`.
fn sym_current(s: &crate::sym::SymExpr) -> f64 {
    s.force_scalar(&crate::input_rt::current())
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
        // A symbolic input folds to its current value as a signal `Konst` (P0 fallback — see
        // `scalar_const`).
        Value::Sym(s) => Ok(Rc::new(SigExpr::Konst(sym_current(s)))),
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
        UnOp::Sqrt => "sqrt",
    }
}

/// Deterministic scalar evaluation of a transcendental unary op (the `Num` fast path; the RV path
/// uses the identical kernel in `bytecode::apply_un`).
pub(crate) fn apply_unop_f64(op: UnOp, x: f64) -> f64 {
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
        UnOp::Sqrt => x.sqrt(),
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

use crate::num::floored_mod;

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
            // The shared scalar kernel (finding F4) — `op` is one of the six arithmetic ops here,
            // so this is bit-identical to the old hand-written match.
            (Value::Num(a), Value::Num(b)) => Ok(Value::Num(crate::num::fold_binop(op, a, b))),
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

#[cfg(test)]
mod registry_coverage {
    //! Finding F2: the builtin namespace used to be defined in ≥7 disjoint tables (`module_of`,
    //! `lib_call`, `builtins::call`, `is_introspection`, `STATS_FNS`, `plot_call`, `stats_call`)
    //! plus the editor grammar, which had already drifted. These cross-check tests make the tables
    //! consistent and fail loudly if any future change registers a name in one place but not the
    //! others.
    use super::*;
    use crate::Engine;

    /// Run a snippet on a fresh engine and return the error message (empty string on success).
    fn err_of(src: &str) -> String {
        match Engine::new().run(src) {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        }
    }

    /// A name "dispatches" if resolving/calling it does NOT fall through to the generic
    /// "unknown function" / "module has no function" arms — an arity or type error still proves the
    /// name was recognized and routed to an implementation.
    fn dispatched(msg: &str, name: &str) -> bool {
        !msg.contains(&format!("unknown function '{name}'"))
            && !msg.contains(&format!("has no function '{name}'"))
            && !msg.contains(&format!("unknown plot 'plot::{name}'"))
            && !msg.contains(&format!("unknown 'stats::{name}'"))
    }

    const CONSTANTS: [&str; 4] = ["pi", "e", "i", "j"];

    /// How to exercise each `math`/`vec`/`rand` name with a **symbolic input** in its numeric slot:
    /// `(name, program, tolerance)`, where every `V` in the program is substituted twice — once
    /// with the slider `x` and once with the literal `2.5` — and the two runs must agree.
    ///
    /// Its reason to exist: `input::real` evaluates to a [`Value::Sym`](crate::value::Value::Sym),
    /// a *fourth* scalar shape beside `Num`/`Est`/`Dist`, and for a long time each new builtin
    /// arrived with `Num`/`Est`/`Dist`/`Signal`/`Array` arms and no `Sym` one. That is invisible
    /// until someone puts a slider in front of the function, so it kept shipping: every `math::`
    /// ufunc, `norm`/`normsq`/`normalize`/`mse`, `max`/`min`/`cummax`/`cummin`/`quantize`,
    /// `count`/`any`/`all`, and `poisson`/`geometric`/`normal_complex` all rejected a slider.
    /// [`sym_support_is_complete`] makes an untabulated name a test failure, so the next builtin
    /// cannot repeat it.
    ///
    /// A **deterministic** probe must match to `1e-12` (the `Sym` defers symbolically, so it is the
    /// same arithmetic). A **Monte-Carlo** probe gets a relative tolerance: a lowered `Sym` is an
    /// `RvNode::Input` uniform where the constant is a baked literal, so the two graphs legitimately
    /// consume the RNG stream differently — the point is that both estimate the same quantity.
    const SYM_PROBES: &[(&str, &str, f64)] = &[
        // --- math: the real ufuncs defer as `SymExpr::Unary`; the complex-aware ones ride the
        // real-promotion path in `complex_parts`. `abs`/`arg` take a negative so they do work.
        ("sqrt", "y = sqrt(V)", 1e-12),
        ("sin", "y = sin(V)", 1e-12),
        ("cos", "y = cos(V)", 1e-12),
        ("atan", "y = atan(V)", 1e-12),
        ("sign", "y = sign(0 - V)", 1e-12),
        ("exp", "y = exp(V)", 1e-12),
        ("log", "y = log(V)", 1e-12),
        ("log10", "y = log10(V)", 1e-12),
        ("abs", "y = abs(0 - V)", 1e-12),
        ("floor", "y = floor(V)", 1e-12),
        ("ceil", "y = ceil(V)", 1e-12),
        ("arg", "y = arg(0 - V)", 1e-12),
        ("conj", "y = conj(V)", 1e-12),
        ("re", "y = re(V)", 1e-12),
        ("im", "y = im(V)", 1e-12),
        ("round", "y = round(V, 1)", 1e-12),
        ("gcd", "y = gcd(floor(V), 6)", 1e-12),
        ("modpow", "y = modpow(floor(V), 3, 5)", 1e-12),
        // --- vec: reducers, linear algebra, scans. The bool-returning ones go through a lifted
        // `if` so both runs yield a number (a `Sym` predicate is a 0/1 indicator, not a `Bool`).
        ("sum", "y = sum([V, 1])", 1e-12),
        ("prod", "y = prod([V, 2])", 1e-12),
        ("mean", "y = mean([V, 1])", 1e-12),
        ("max", "y = max([V, 1])", 1e-12),
        ("min", "y = min([V, 1])", 1e-12),
        ("count", "y = count([V > 1, false])", 1e-12),
        ("any", "y = if any([V > 1]) { 1 } else { 0 }", 1e-12),
        ("all", "y = if all([V > 1]) { 1 } else { 0 }", 1e-12),
        ("dot", "y = dot([V, 1], [1, 2])", 1e-12),
        ("vdot", "y = vdot([V, 1], [1, 2])", 1e-12),
        ("norm", "y = norm([V, 1])", 1e-12),
        ("normsq", "y = normsq([V, 1])", 1e-12),
        ("normalize", "y = normalize([V, 1])[0]", 1e-12),
        ("mse", "y = mse([V, 1], [1, 2])", 1e-12),
        ("transpose", "y = transpose([[V, 1], [2, 3]])[1][0]", 1e-12),
        ("adjoint", "y = adjoint([[V, 1], [2, 3]])[1][0]", 1e-12),
        ("outer", "y = outer([V, 1], [1, 2])[0][1]", 1e-12),
        ("onehot", "y = sum(onehot(1, 3) * V)", 1e-12),
        ("quantize", "y = quantize([V], [0, 1, 3])[0]", 1e-12),
        ("cumsum", "y = cumsum([V, 1])[1]", 1e-12),
        ("cumprod", "y = cumprod([V, 2])[1]", 1e-12),
        ("cummax", "y = cummax([V, 1])[1]", 1e-12),
        ("cummin", "y = cummin([V, 1])[1]", 1e-12),
        (
            "has_duplicates",
            "y = if has_duplicates([V, 2.5]) { 1 } else { 0 }",
            1e-12,
        ),
        ("count_duplicates", "y = count_duplicates([V, 2.5])", 1e-12),
        // Size/shape constructors are STRUCTURAL — the `Sym` folds to its current value, which is
        // correct (a different size is a different program) but must still not error.
        ("ones", "y = sum(ones(floor(V)))", 1e-12),
        ("zeros", "y = sum(zeros(floor(V)) + V)", 1e-12),
        ("iota", "y = sum(iota(floor(V)))", 1e-12),
        // --- rand: a slider in a distribution parameter. Monte-Carlo, so relative tolerance.
        ("unif", "X ~ unif(0, V)\ny = E(X)", 0.02),
        ("unif_int", "X ~ unif_int(1, floor(V))\ny = E(X)", 0.02),
        ("bernoulli", "X ~ bernoulli(V / 10)\ny = P(X)", 0.05),
        ("normal", "X ~ normal(0, V)\ny = Var(X)", 0.02),
        ("normal_int", "X ~ normal_int(V, 1)\ny = E(X)", 0.02),
        ("normal_complex", "Z ~ normal_complex(V)\ny = E(abs(Z))", 0.02),
        ("exponential", "X ~ exponential(V)\ny = E(X)", 0.02),
        ("exponential_int", "X ~ exponential_int(V)\ny = E(X)", 0.05),
        ("poisson", "X ~ poisson(V)\ny = E(X)", 0.02),
        ("geometric", "X ~ geometric(V / 10)\ny = E(X)", 0.05),
        ("categorical", "X ~ categorical([V, 1])\ny = E(X)", 0.05),
        ("empirical", "X ~ empirical([1, V])\ny = E(X)", 0.02),
        (
            "block_bootstrap",
            "X ~ block_bootstrap([1, 2, 3, V], 2)\ny = E(sum(X))",
            0.02,
        ),
        ("rotation", "R ~ rotation(2)\ny = E(norm(R @ [V, 0]))", 0.02),
        (
            "permutation",
            "p ~ permutation(floor(V))\ny = E(sum(p) + V)",
            0.02,
        ),
    ];

    /// Names in `math`/`vec`/`rand` that [`SYM_PROBES`] deliberately does not cover, each with the
    /// reason. Anything else must be tabulated — see [`sym_support_is_complete`].
    const SYM_EXEMPT: &[(&str, &str)] = &[
        ("pi", "a constant, not a function — it takes no argument"),
        ("e", "a constant, not a function"),
        ("i", "a constant (the imaginary unit)"),
        ("j", "a constant (alias of `i`)"),
    ];

    /// Build the two programs a probe compares: the slider one and the constant one.
    fn sym_probe_pair(body: &str) -> (String, String) {
        let head = "use math\nuse vec\nuse rand\n\
                    x = input::real(min: 0.5, max: 9, step: 0.5, default: 2.5)\n";
        // The constant program keeps the (unused) input declaration so both runs see the same
        // input manifest — only the *use* of the value differs, which is what is under test.
        (
            format!("{head}{}\ny", subst_v(body, "x")),
            format!("{head}{}\ny", subst_v(body, "2.5")),
        )
    }

    /// Replace the standalone `V` placeholder — a bare `V`, not the one inside `Var` — with `with`.
    fn subst_v(body: &str, with: &str) -> String {
        let bytes = body.as_bytes();
        let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let mut out = String::with_capacity(body.len());
        for (i, &c) in bytes.iter().enumerate() {
            let prev_is_ident = i > 0 && ident(bytes[i - 1]);
            let alone =
                c == b'V' && !prev_is_ident && !bytes.get(i + 1).is_some_and(|&n| ident(n));
            if alone {
                out.push_str(with);
            } else {
                out.push(c as char);
            }
        }
        out
    }

    /// Run a probe program and return its final numeric value.
    fn sym_probe_value(src: &str, label: &str) -> f64 {
        match Engine::new().run(src) {
            Ok(Value::Num(n)) => n,
            Ok(Value::Est { val, .. }) => val,
            Ok(other) => panic!("{label}: expected a number, got {other:?}\n--- program:\n{src}"),
            Err(e) => panic!("{label}: {e}\n--- program:\n{src}"),
        }
    }

    /// A slider (`input::real` → `Value::Sym`) must be accepted wherever a number is, and give the
    /// same answer as the constant it currently holds.
    #[test]
    fn every_builtin_accepts_a_symbolic_input() {
        for &(name, body, tol) in SYM_PROBES {
            let (sym_src, const_src) = sym_probe_pair(body);
            let got = sym_probe_value(&sym_src, &format!("{name} (with a slider)"));
            let want = sym_probe_value(&const_src, &format!("{name} (with a constant)"));
            let ok = if tol < 1e-9 {
                (got - want).abs() <= tol
            } else {
                // Monte-Carlo probe: relative agreement (guard the want == 0 case).
                (got - want).abs() <= tol * want.abs().max(1.0)
            };
            assert!(
                ok,
                "`{name}` disagrees with a slider in its numeric slot: \
                 got {got} with `input::real`, {want} with the same value as a constant\n\
                 --- program:\n{sym_src}"
            );
        }
    }

    /// The completeness half: every registered `math`/`vec`/`rand` name is either probed by
    /// [`SYM_PROBES`] or explicitly exempt with a reason. Adding a builtin without teaching it
    /// `Value::Sym` — the bug this pair of tests exists to stop — fails right here.
    #[test]
    fn sym_support_is_complete() {
        let mut untabulated = Vec::new();
        for &(name, module) in BUILTINS {
            if !matches!(module, "math" | "vec" | "rand") {
                continue;
            }
            let probed = SYM_PROBES.iter().any(|&(n, _, _)| n == name);
            let exempt = SYM_EXEMPT.iter().any(|&(n, _)| n == name);
            if !probed && !exempt {
                untabulated.push(format!("{module}::{name}"));
            }
        }
        assert!(
            untabulated.is_empty(),
            "these builtins have no `input::real` (Value::Sym) coverage: {untabulated:?}\n\
             Add a probe to SYM_PROBES (or an entry with a reason to SYM_EXEMPT) — and make it pass, \
             which usually means adding the missing `Value::Sym` arm to the function."
        );
    }

    #[test]
    fn every_registered_name_dispatches() {
        for &(name, module) in BUILTINS {
            assert!(is_module(module), "{name} claims unknown module {module}");
            let src = if CONSTANTS.contains(&name) {
                // Constants are values, not calls — reference them through their module path.
                format!("{module}::{name}")
            } else if module == "builtin" {
                // The always-on core is capital-only and unqualified.
                format!("{name}()")
            } else {
                format!("{module}::{name}()")
            };
            let msg = err_of(&src);
            assert!(
                dispatched(&msg, name),
                "registry name `{module}::{name}` does not dispatch — got: {msg}"
            );
        }
    }

    #[test]
    fn introspection_plot_stats_names_dispatch() {
        // Introspection builtins are always-on and unqualified.
        for &name in &INTROSPECT_FNS {
            let msg = err_of(&format!("{name}()"));
            assert!(dispatched(&msg, name), "introspection `{name}`: {msg}");
        }
        // `plot::` and `stats::` verbs are always qualified.
        for &name in &PLOT_FNS {
            let msg = err_of(&format!("plot::{name}()"));
            assert!(dispatched(&msg, name), "plot::{name}: {msg}");
        }
        for &name in &STATS_FNS {
            let msg = err_of(&format!("stats::{name}()"));
            assert!(dispatched(&msg, name), "stats::{name}: {msg}");
        }
    }

    /// Every user-writable builtin/constant name (the scoped registry) must be highlighted by the
    /// editor's TextMate grammar, so adding a builtin without teaching the grammar fails here. Reads
    /// the canonical grammar at test time; if it isn't present (e.g. a packaged tarball), skips.
    #[test]
    fn registry_names_are_in_the_textmate_grammar() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../editors/vscode-noise/syntaxes/noise.tmLanguage.json"
        );
        let Ok(grammar) = std::fs::read_to_string(path) else {
            eprintln!("grammar not found at {path}; skipping coverage check");
            return;
        };
        // The grammar lists names as `\b(a|b|c)\b` alternations; a simple `|name|`/`(name|`/`|name)`
        // membership check is enough (all names are plain identifiers).
        let has = |n: &str| {
            grammar.contains(&format!("|{n}|"))
                || grammar.contains(&format!("({n}|"))
                || grammar.contains(&format!("|{n})"))
                || grammar.contains(&format!("({n})"))
        };
        let mut missing = Vec::new();
        for &(name, _) in BUILTINS {
            if !has(name) {
                missing.push(name);
            }
        }
        assert!(
            missing.is_empty(),
            "these registered builtins are not highlighted by the TextMate grammar \
             (editors/vscode-noise/syntaxes/noise.tmLanguage.json): {missing:?}"
        );
    }

    #[test]
    fn unknown_module_call_suggests_the_nearest_name() {
        // A near-miss typo points at the intended function...
        let msg = err_of("use vec\nvec::onehott(2, 4)");
        assert!(
            msg.contains("has no function 'onehott'") && msg.contains("did you mean 'vec::onehot'"),
            "expected a did-you-mean suggestion, got: {msg}"
        );
        // ...while an unrelated name gets the bare error, not a misleading guess.
        let msg = err_of("use vec\nvec::zzzzzz(2, 4)");
        assert!(
            msg.contains("has no function 'zzzzzz'") && !msg.contains("did you mean"),
            "expected no suggestion for an unrelated name, got: {msg}"
        );
    }

    #[test]
    fn nearest_name_stays_within_its_edit_budget() {
        let vec_fns = || module_fns("vec");
        assert_eq!(nearest_name("summ", vec_fns()), Some("sum")); // 1 insertion
        assert_eq!(nearest_name("transpoze", vec_fns()), Some("transpose")); // 1 substitution
        assert_eq!(nearest_name("xyz", vec_fns()), None); // nothing close
    }
}
