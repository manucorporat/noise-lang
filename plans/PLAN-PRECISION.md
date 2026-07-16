# PLAN-PRECISION — stop at the answer, not at a sample count

**Date:** 2026-07-15 · **Status: FULLY LANDED (2026-07-16)** — Phase 1, Phase 2 (precision
default-on), Track F (on-GPU fold), and all of Track H (CLI Ctrl-C rungs + browser stop cell +
playground `maxTime`/warnings UI). Grounded by a seam map (file:line below; line numbers predate
the implementation).

> **Phase 2 + F + H notes (what landed vs. the draft, second pass):**
> - **Phase 2 (default-on):** untargeted `P`/`E`/`Var` now use `DrawBudget::Auto` — the target is
>   `se ≤ 5e-3 · max(|est|, sd)` with `sd = se·√m`. The **sd floor** is the deviation from the
>   draft's bare `rel` default: a pure relative target never terminates on a ≈0-mean quantity, and
>   an `abs` floor isn't scale-free. The sd-floored rule solves to `m ≥ rel⁻² = 40k` *effective*
>   draws (one 65k pilot for an easy query, ~25× less than the old 1M; honest extension for
>   conditionals/rare events). Deliberately NOT expressible via `set_precision` (a declared target
>   is a pure demand; the default is a bounded-effort heuristic). `P_DEFAULT_N`, the `samples`
>   runtime setting (CLI/wasm/npm), and `Engine::set_max_samples` are all deleted; `Q` keeps its
>   own fixed `Q_DEFAULT_N = 1M`; plots keep their visual budgets. Corpus total ~4.25 s (≈ Phase-1
>   baseline; many small examples 2-4× faster, e.g. birthday 17→4 ms).
> - **Track F (on-GPU fold):** `wgsl_emit::emit_reduce` — each workgroup folds 4096 lanes
>   (64 threads × 64 sequential lanes + fixed-order tree reduce) and writes one `(Σx, Σx², count)`
>   triple; readback drops ~1300×. Routed in `gpu::try_reduce` for `Reducer::moments_mode()`
>   reducers at **n ≥ 1M lanes** (below that, column mode keeps its calibrated economics and a
>   reduce dispatch would fill only n/64 threads). Gate: column-gate acceptances stay, plus a
>   thin-cone amortization floor `n·ops ≥ 1e9` (cold-pipeline-safe; `prefer_runtime` uses `1e7` —
>   warm-pipeline crossover measured at ~1M lanes). Workgroup slices nest in chunks, so staged ==
>   single stays bit-identical *on the GPU's own fold* (unit test); the fold itself is f32 — tier 2
>   vs the CPU's f64 fold (~1e-6 relative; LANG.md notes the backend/mode-straddling edge).
>   Measured: π to 5 digits (2.7e9 draws) 1.46 s vs ~7 s CPU; the wasm bridge encodes reduce mode
>   as `cols == 0` (`dispatchShape` mirrored in `gpu-host.ts`) — a browser run swept 34.4B draws
>   (8 epochs) in a 10 s budget.
> - **Track H (complete):** CLI two-rung Ctrl-C (`ctrlc` dep; first = `CancelToken::stop()` +
>   partial doc + exit 0, second = exit 130). Browser: `exec::HOST_STOP` (a process-global cell
>   read at the per-chunk stop cadence; `noise-wasm` exports `stop_cell_ptr()`), the threaded
>   worker announces `{type:'stop-cell', sab, ptr}` once, and `pool.softStop(id)` is one
>   `Atomics.store` into it. `run()`/`runWithIntrospection()` now return a **`RunHandle`**
>   (`Promise` + `stop(): boolean` — `false` = unsupported/settled; single-threaded + Node fall
>   back to abort). Playground: Run button doubles as **Stop** mid-run, `maxTime: 10s` default,
>   and an amber warnings banner (`.pg-warnings`, in the `is:global` style block — scoped styles
>   can't reach runtime-created nodes). Capped-query warnings now report the **folded count**
>   (`out.count`), not the requested sweep — a soft-stopped GPU stage folds only part of its range.

> **Implementation notes (what landed vs. the draft):**
> - Tracks A, B, C, D (deadline-aware sizing off measured stage throughput), E (epoch reseeding),
>   G, and the Rust core of H (soft-stop flag + `max_time` deadline + partial folds + honest
>   `acc.count()`-based se + statement skip + warnings) are in. `run_reduction_range` takes a
>   **carry accumulator** (not just a range): resuming must continue the left fold chunk-by-chunk,
>   or f64 addition order would break the staged-⁠==⁠-single bit-identity.
> - The flagship `pi.noise` uses a `1e-4` per-call target (not the draft's `1e-5`) so the corpus
>   smoke test stays fast in debug builds; `rare_event.noise` is the new relative-precision example.
> - CLI default `max_time` = 60 s (`--max-time 0` = off). npm `RunOpts` gained
>   `precision`/`maxTime` (ms)/`samples`/`resolution`; `DocResult.warnings` carries capped-run notes.
> - **The "zero corpus usage" claim was wrong in one way that mattered:** no example *called* the
>   budget pragmas, but the **default** `max_ops = 1e9` clamp was silently load-bearing for the
>   three heaviest cones — `noise_colors` (67k ops/draw: 1M draws → ~15k under the clamp),
>   `turboquant`, `prisoners`. Deleting the clamp made their untargeted queries draw the full 1M
>   (noise_colors: 2s → 60s). Fixed the plan-conformant way: those examples now declare what they
>   need (`engine::set_precision(5e-3/2e-3)`; noise_colors uses per-call counts since its
>   ratio-of-means has a ≈ 0 numerator) — and all three ended up *faster* than master, with
>   `prisoners` printing `31.2%` where master's clamp showed `31 ± 1%`.
> - **Still open (H, browser half):** the SAB stop cell + `stop()` on the JS run handle, the
>   playground's default `maxTime` + warning UI, and the CLI double-Ctrl-C rungs. Track F (on-GPU
>   fold) remains its own follow-up, as planned.
**Depends on** counter keying (PLAN-PREGPU Track C) and the monoid reducers (`reduce.rs`) — those two
landed properties are what make adaptive sampling *exact* rather than approximate. Interacts with the
GPU gate (`gpu.rs`) and the compile cache; changes neither's contract.

## The thesis

The engine's two budget knobs are **input-side**: they pick a draw count *before* the run.

- `engine::set_max_samples(N)` — despite the name, not a max: the **default n** when a query carries
  no explicit count (`P_DEFAULT_N = 1_000_000`, `builtins.rs:30`; stored at `eval.rs:145`).
- `engine::set_max_ops(N)` (canonical name; `set_max_opts` is the back-compat alias, `eval.rs:1160`) —
  a per-query op ceiling that **silently clamps** n so `n × cone_ops ≤ budget`
  (`clamp_to_op_budget`, `builtins.rs:355`; default 1e9, `builtins.rs:37`).

But the question a user actually has is **output-side**: *"give me π to 4 digits."* Nobody knows —
or should have to know — that this means n ≈ 6e9 for `pi.noise`'s thin cone. The evidence that the
current knobs answer no real question: **zero of the 31 corpus examples call either one.** Every
example just eats the fixed 1M default, which is simultaneously wasteful (p ≈ 0.5 → se 5e-4, more
digits than the prose needs) and hopeless (a rare event at p = 1e-5 gets se ≈ 3e-6 — a 30% relative
error — with no path to more except hand-computing n).

Meanwhile the engine already *measures* precision on every query: `P`/`E`/`Var` return
`Value::Est { val, se }` (`builtins.rs:412`, `:479–485`) and the display auto-rounds to the digits the
se justifies. The missing piece is small and well-supported by the architecture:
**a driver that keeps drawing until `se ≤ target`, bounded by a runtime deadline** — and the same
move lets us **delete the op-budget machinery outright** (`clamp_to_op_budget` and both its pragmas)
instead of teaching it new tricks: its only real job was "never hang out of the box", and a default
`max_time` + soft-stop does that job while answering a question users actually have.

Two landed properties make this exact, not heuristic:

1. **Counter keying** — the draw at lane `i` is a pure function of `(seed, i, source)`. A reduction
   over lanes `0..n₁` extended by lanes `n₁..n₂` is **bit-identical** to a single run over `0..n₂`.
   The GPU dispatch already takes a `lane0` (`gpu.rs:254`); the CPU runner already positions at any
   lane (`backend.rs:199`). Extension is not a new capability — it's the existing chunk mechanism
   with a nonzero starting chunk.
2. **Monoid reducers** — `P`/`E`/`Var` all fold through `MomentsReducer`/`CondMomentsReducer`
   (Σx/Σx², `reduce.rs:373`/`:441`), whose accumulators combine associatively in chunk order
   (`combine_in_order`, `reduce.rs:91`). A stage's accumulator carries forward; nothing is re-drawn.

So adaptive precision costs: a resumable entry point into `reduce`, a pilot-extrapolate-extend loop
in `sampler`, one engine knob, and a reporting path when the deadline stops us short.

## The model: demand and supply, separated

| setting | role | semantics |
|---|---|---|
| `precision = (rel, abs)` — **new** | **demand**: what you want | keep drawing until `se ≤ max(abs, rel·\|est\|)`. Unset/0 = off (fixed-n behavior). |
| `max_time` — **new** | **supply**: wall-clock ceiling | soft-stop the query when the deadline passes and report what the time bought (honest se + warning). **Explicitly non-deterministic** — see "determinism" below. Hosts ship a sane default (playground ~2 s; CLI lenient, `--max-time 0` = off). |
| `samples` — runtime-only remnant | back-compat | the default n when no target is set (today's `P_DEFAULT_N = 1M`, deleted in Phase 2). An explicit per-call count `P(e, n)` always means exactly n — it disables adaptivity for that call. |
| `max_ops` / `max_opts` — **REMOVED** | — | `clamp_to_op_budget` (`builtins.rs:355`), the `max_opts` engine field, the `QueryCtx` member, and both pragmas are deleted in this plan. The op budget was a proxy for time that silently traded precision; `max_time` + soft-stop is the honest version of the same guard. |

## Where the knobs live: pragmas declare, `run()` overrides

The residence rule has two tests, and each surviving knob passes both or lives host-side:

1. **Deterministic?** A pragma in program text must reproduce identical digits on any machine and
   backend. `precision` passes (the stop rule reads only drawn values); `max_time` fails — a
   `max_time` pragma would make a program's digits machine-dependent while *looking* like part of
   the question. So `max_time` is runtime-only.
2. **Is it the question, or the run?** A program states what it *needs* (`precision` — π to five
   digits is the program's point); how much machine time a given run may spend belongs to whoever
   runs it (`max_time`).

| setting | pragma (program-wide default) | runtime (`run()` / CLI) |
|---|---|---|
| `precision` — **new** | `engine::set_precision(rel[, abs])` — different programs need different digits | `RunOpts.precision` / `--precision 1e-4` |
| `max_time` | none (test 1) | `RunOpts.maxTime` / `--max-time 2s` |
| `resolution` | `engine::set_resolution(N)` — existing, untouched | `RunOpts.resolution` / `--resolution 512` |
| `samples` | **pragma removed** (see below) | `RunOpts.samples` / `--samples 3e3` — runtime default-n for benches/goldens until Phase 2 deletes the fixed-n default |

**Removed outright, this plan:** `engine::set_max_ops` / `set_max_opts` / `set_max_samples`, the
`max_opts` engine field + `QueryCtx.max_opts`, `clamp_to_op_budget` (`builtins.rs:355`), and
`MAX_OPS_DEFAULT` (`builtins.rs:37`). The op budget was complexity with no constituency (zero corpus
usage; its silent clamp is a bug report waiting to happen), and the samples pragma is subsumed by
per-call counts + the runtime setting. The ~16 test call sites (`tests/probability.rs` mostly, one
each in `cancel`/`complex`/`signals`) migrate to per-call counts or the Engine's Rust setter, which
stays for hosts. npm is 0.1.x and no example uses any of them — deletion is cheap now and never
again. Calling a removed pragma is a spanned error naming the replacement.

> **A pragma is the program's declared default; a passed setting overrides anything the program
> sets.** Effective value: runtime override → program pragma → engine default — a `RunOpts`/CLI
> value wins unconditionally over any `engine::` call, wherever it appears in the document. The one
> deliberate carve-out: per-call arguments (`P(e, n)`, `P(e, 1e-4)`) are expression semantics, not
> settings — they govern their own call, the way an explicit count does today.

- **Self-describing programs**: `pi.noise` carries `engine::set_precision(1e-5)` and
  `noise run pi.noise` does what the author meant — the runner's `max_time` (default or flag)
  decides whether the machine gets there today, and the warning reports the gap if not.
- **Hosts stay in charge of their own runs**: the playground pins a snappy `maxTime` regardless of
  what a pasted program declares (its stop button + Track H soft-stop are the real guard); a
  benchmark harness pins `samples`; CI pins everything.
- **Mechanics of "override wins"**: overrides arrive before evaluation (they ride `optsJson` →
  Engine setters, like input overrides); each overridden setting is marked *pinned*, and a pragma
  assignment to a pinned setting is evaluated but doesn't change the effective value (a
  `NOISE_PROFILE=1` note records the shadowing). Mid-document re-assignment keeps today's
  statement-order semantics for the un-pinned case.

## The end state: two drivers, staged in two phases

**Phase 1 (this plan)** lands the adaptive machinery *and* the deletions — `max_ops` and the
`set_max_samples` pragma go now, not later — while keeping the default behavior fixed-n (the
internal `P_DEFAULT_N = 1M` constant survives as the no-target default, so untargeted corpus
outputs are unchanged). **Phase 2** then flips the default:

| | Phase 1 (this plan) | Phase 2 (flip the default) |
|---|---|---|
| default behavior | fixed n = 1M (`P_DEFAULT_N`), untargeted outputs bit-for-bit today's | **precision default-on** (e.g. `rel = 5e-3`, ≈ 3 digits); `P_DEFAULT_N` and the `samples` runtime setting deleted |
| runaway guard | **`max_time` set at runtime** (hosts ship defaults: playground ~2 s, CLI lenient + `--max-time 0` = off); `max_ops` machinery deleted | same |
| pragmas | `set_precision` (new), `set_resolution` (untouched); the three budget pragmas removed | same |
| drivers | `precision` (pragma / per-call / runtime) + `max_time` (runtime) | same — now with nothing else left |

The contract: *a query runs until it's precise enough or you've waited long enough, whichever comes
first — and either way the se is honest.* Both drivers connect to a real use case; everything
deleted was a proxy for one of them.

**The cost, stated plainly: the guard is no longer deterministic.** A run that **hits its target**
is fully deterministic (the stop rule reads only drawn values — same seed, same n, same digits
everywhere); a run that **hits the deadline** is machine-dependent, visibly (the warning + wider se
say so). That is the accepted trade — the se travels with every estimate, so a slower machine
prints *fewer digits*, never wrong ones. Tests and benches opt into exactness with per-call counts
or the runtime `samples` setting; corpus goldens pin `--max-time 0`. Phase 2 ships as its own
change: it rewrites every example's digits and is trivial to stage after Phase 1 proves the
machinery.

The `max(abs, rel·|est|)` stopping rule is the standard numerical-integration contract (QUADPACK,
GSL) and handles both regimes: `precision = 1e-4` reads as "4 significant digits" — and because
`pi = 4 · P(…)` scales se and value together, relative precision on `P` **is** relative precision on
π. The `abs` term rescues `E` of a quantity whose mean is ≈ 0.

**Determinism, precisely.** Reproducibility comes from *what stops the run*:

- Stopped by **the target** (or an explicit count): fully deterministic — the stop rule reads only
  drawn values, and draws are a pure function of `(seed, lane)`. Same program, same digits, every
  machine and backend. This is the common case: an op costs ~0.3 ns on M4-multicore, so any sane
  deadline leaves most queries finishing on target.
- Stopped by **`max_time`** (or a user's soft-stop): honest but machine-dependent — a faster box
  prints more digits. That is the right contract for the guard role ("spend at most 2 s on this
  slider drag"), and it is safe because the se travels with the estimate: fewer samples never mean
  a *wrong* answer, only a wider one. It rides the soft-stop machinery (Track H), so hitting the
  deadline reports the numbers so far instead of discarding them — the same path a user abort
  takes, and the warning says which it was.

## Per-call targets: the second argument of `P`/`E`/`Var` becomes an error

Today `P(event, n)` / `E(x, n)` / `Var(x, n)` take an explicit **count**, validated `n ≥ 1`
(`builtins.rs:384`, `:453`). Repurpose the numeric range the validation already rejects:

- **`0 < x < 1` → a per-call precision target**: relative se, i.e. stop when `se ≤ x·|est|`.
  `P(hit, 1e-4)` reads as "this probability to 4 significant digits". Overrides the document-wide
  setting (pragma or host override) for this call; still bounded by `max_time`.
- **`x ≥ 1` → an explicit count**, exactly as today (and it disables adaptivity for the call).

The two ranges are disjoint by construction — a count below 1 is an error today (free syntax space),
and a *relative* target ≥ 1 ("100%+ error, please") is meaningless. Relative (not absolute) is the
right per-call meaning because it is scale-free: it works identically for `P` (unit interval) and
`E` (any magnitude), so the one rule covers both builtins — absolute targets remain expressible via
the document-wide `precision = (rel, abs)` setting (pragma or override). **Corpus impact: zero** —
no example passes an explicit count
(the only two-arg query in the corpus is `Q(x, q)`, whose second slot is the quantile level, not a
count — `Q` is untouched). Longer-term the count form can be deprecated in favor of
`set_max_samples`, but nothing forces that now.

`Print`/display need no change: the target only decides *when to stop*; the printed digits still come
from the achieved se, so an impossible ask degrades gracefully into fewer digits plus the Track C
warning.

## The algorithm (per query, `P`/`E`/`Var`)

```
n₀ = PILOT_N (65_536 = 4 reducer chunks)          # cheap, se-of-se ~ 0.3%
acc = reduce(lanes 0..n₀)
loop:
    est, se = acc.estimate()
    target = max(abs, rel·|est|)
    if se ≤ target: done
    n_need = n_done · (se/target)²  · 1.1          # CLT: se ∝ 1/√n; 10% margin
    n_next = min(n_need, GROWTH_CAP·n_done, time_cap, LANE_CAP)     # chunk-aligned
    if n_next == n_done: done, capped=true         # deadline reached → report
    acc ⊕= reduce(lanes n_done..n_next)            # extension, not re-run
```

`time_cap` is the draws the remaining deadline is predicted to afford, from **measured** stage
throughput (draws/sec of completed stages, tracked per backend — see Track D); with no `max_time`
set it is ∞ and the loop is governed by the target alone. The deadline is also enforced *inside* a
stage (per-chunk / per-dispatch soft-stop, Track H), so a bad prediction overshoots by one check
granularity, not a stage.

- **Chunk-aligned stages** (`CHUNK_SAMPLES = 16·1024`, `reduce.rs:62` mirror at `gpu.rs:62`) keep the
  fold boundaries identical to a single run → the final `(est, se)` is **bit-identical to a
  non-adaptive run at the same final n**. That is the correctness invariant every test hangs off.
- **`GROWTH_CAP` (×64 per stage)** bounds the damage of a noisy pilot se (a lucky pilot under-
  estimates variance → extrapolation overshoots). Two or three stages reach any realistic target.
- **Rare events**: a pilot with `p̂ = 0` has se = rule-of-three floor `0.5/n` (`builtins.rs:412`),
  which extrapolates *linearly* (`n ≥ 0.5/target`), not quadratically — the loop handles it because
  each stage re-derives `n_need` from the current floor-or-binomial se, whichever is larger.
- **Conditional queries** (`P(A|B)`, `E(x|C)`) come along free: `CondMomentsReducer` tracks the
  in-condition count `m` and its se already uses `m` — a stage that loses lanes to rejection simply
  shows a bigger se and the loop draws more. This is a real win: today a tight conditioning silently
  gets a fraction of the nominal budget.
- **Stopping bias**: a data-dependent stop technically biases the estimate; with a 64k pilot and
  ≤ ~4 geometric stages the effect is O(1/n) against an se of O(1/√n) — negligible, and standard
  practice. Note it in LANG.md, don't fight it.
- **Determinism**: the stop rule reads only drawn values, and draws are a pure function of
  `(seed, lane)` — so the final n, and therefore the printed digits, are reproducible.

**Out of scope (documented, not broken):** `Q` (quantiles) collects the full draw vector
(`CollectReducer`, `reduce.rs:485`) — O(n) memory kills adaptive growth, and its se needs a density
estimate; it keeps fixed-n and already returns a plain `Num`, claiming no precision
(`builtins.rs` Q doc). Plots/introspection stay at their fixed visual budgets (`introspect.rs:30`)
— correctly so: a plot's collapse target is *visual* resolution, which is a constant, not a
question-dependent n, so precision targeting buys nothing there. Both can join later (P² estimator
for Q) without touching this plan's surface. Plots do meet this plan on the supply side: they are
forcings, so `max_time`/soft-stop (Track H) apply — a soft-stopped plot **renders what it
collected** (a sparser scatter / noisier hist is a valid smaller-sample plot, flagged by the same
"stopped early" warning), rather than being dropped.

## Tracks

### A — resumable reduction (the mechanical enabler)

`run_reduction` (`reduce.rs:109`) hardcodes lanes `0..n`. Add the range form and re-express the
existing one over it:

- `run_reduction_range<R>(graph, root, lanes: Range<u64>, seed, r) -> Result<R::Acc>` — start chunk
  = `lanes.start / CHUNK_SAMPLES` (callers guarantee alignment), same work-stealing drivers
  (`run_parallel`, `reduce.rs:203`/`:266`), same per-chunk cancellation.
- The GPU driver (`gpu::try_reduce`, `gpu.rs:136`) takes the same range — its dispatch already
  addresses `lane0..lane0+n` (`gpu.rs:254`, `:477`); only the driver loop's starting offset changes.
- The caller folds stage accumulators with the reducer's existing `combine` in stage order —
  same shape as `combine_in_order`.
- **Compile cache** (`compile_cache.rs`): stages of one query hit the same key (same simplified cone;
  the wasm gate bucket, `backend.rs:109`, keys on the *decision*, and every post-pilot stage is far
  above `MIN_DRAWS_WASM`). The GPU pipeline cache is content-addressed by shader text (`gpu.rs:186`)
  — stage 2 reuses stage 1's pipeline if the gate accepted both. No recompiles per stage.

### B — the adaptive driver

`sampler::moments_to_precision(graph, root, rel, abs, deadline, seed) -> (Moments, n_used, capped)`
next to `moments` (`sampler.rs:104`), implementing the loop above. `P`/`E`/`Var` (`builtins.rs:364`,
`:433`) call it when a target is set and no explicit per-call n was given. **In the same change,
delete the op budget**: `clamp_to_op_budget` (`builtins.rs:355`) and every call site (`:407`, `:478`,
and the `moment`/`quantile` twins), the `max_opts` field on the engine (`eval.rs:151`) and
`QueryCtx` (`builtins.rs:77`), `MAX_OPS_DEFAULT` (`builtins.rs:37`), and the two `lib_set_max_op*`
builtins — queries simply draw what the target/deadline says. `Est` construction is unchanged — se
comes out of the same accumulator it always did.

### C — the surface (pragma + runtime, and the deletions)

- Engine fields `precision: Option<(f64, f64)>` and `max_time: Option<Duration>` next to
  `max_samples` (`eval.rs:144–151`), threaded through `QueryCtx` (`builtins.rs:74`).
  Validation: `rel ≥ 0`, `abs ≥ 0`, not both 0.
- **Pragma**: `engine::set_precision` joins the `BUILTINS` table (`eval.rs:1160–1165`) and the
  playground keyword list (`noise-lang.ts:24`), with a `lib_set_precision` in `eval/library.rs`.
  (`max_time` gets no pragma — see "Where the knobs live".)
- **Deletions**: `set_max_samples` / `set_max_ops` / `set_max_opts` leave `BUILTINS` and the keyword
  list; calling one is a spanned error naming the replacement (`engine::set_max_samples was removed
  — pass a per-call count P(e, n), or --samples / RunOpts.samples`). The ~16 test call sites migrate
  to per-call counts; the Rust `Engine` setters stay for hosts (benches keep working via the
  setter). The op-budget machinery goes entirely (Track B).
- **Runtime half**: `RunOpts` gains `precision` / `samples` / `maxTime` / `resolution`
  (`index.ts:261` reserved the slot), serialized through `optsToJson` → the wasm boundary → Engine
  setters before evaluation, each marking its setting *pinned* (pragma writes to a pinned setting
  don't change the effective value). The CLI gains matching flags (`main.rs:57`), including the
  CLI's default `max_time` (lenient; `--max-time 0` = off).
- `check` mode (validate-only, `builtins.rs:404`) bypasses the loop exactly as it bypasses sampling.
- **Reporting when capped**: `DocResult` (`doc.rs:74`) grows a `warnings: Vec<String>` the CLI prints
  to stderr and the playground surfaces like its other diagnostics. Message names what bound the
  run: `P at line 16: target ±1e-5 needs ~2.4e9 draws (~35 s at measured throughput); max_time=2s
  stopped it at n=1.4e8 → ±2.1e-4. Raise --max-time to reach the target.` Also a `NOISE_PROFILE=1`
  note with per-stage n / se / backend (`profile.rs`), since the stage structure is exactly what
  that tool is for.

### D — a backend-aware driver (wasm + GPU are first-class, not afterthoughts)

The adaptive loop and the deadline predictor sit *above* the backend seam, but they must be written
knowing what's below it:

- **Per-stage gate, true n**: each stage calls `run_reduction_range` with the **stage's** lane
  count, so the GPU gate (`profitable`, `gpu.rs:136`) decides per stage honestly: the 64k pilot
  fails `MIN_WORK_GPU` (3e6, `gpu.rs:115`) on most cones → CPU, cheap; a fat-cone growth stage
  passes → GPU, and the pipeline is cached for the *next* stage. Thin cones (< 45 ops/draw,
  `gpu.rs:111`) stay on multicore CPU at any n until Track F changes the readback economics.
- **Throughput is per-backend state**: the `time_cap` predictor learns draws/sec from completed
  stages — but a CPU pilot's throughput says nothing about the GPU stage that follows (10×+ apart,
  plus a possible one-off pipeline compile of seconds). The predictor keys its estimate on the
  backend the gate will pick for the *next* stage (ask the gate before sizing), treats the first
  stage on a new backend as a calibration stage (size it modestly), and excludes one-off compile
  time from the steady-state rate. Getting this wrong doesn't break anything — the in-stage
  deadline check (Track H) catches overshoot — but getting it right is the difference between
  landing on the deadline and wasting it.
- **wasm, single- and multi-worker**: the rayon-pool driver takes the same chunk ranges
  (`reduce.rs:266`); the wasm codegen gate (`MIN_DRAWS_WASM = 10k`, `kernel.rs:342`) means the 64k
  pilot already clears the emitted-kernel threshold, and the gate *bucket* (`backend.rs:109`) keys
  the compile cache so pilot and growth stages share one artifact. Deadline checks use `web-time`
  (already a dependency). Single-threaded wasm is the slow rung the ladder was built for: same
  code, honest fewer digits per deadline.
- **Browser GPU (G3 bridge)**: growth stages dispatch through the SAB+Atomics main-thread bridge —
  the deadline/stop check happens between dispatches in the wasm-side loop (it must not park on
  `Atomics.wait` past the deadline; the wait gets a timeout equal to the remaining budget).
- **Verify, per backend**: staged == single-run bit-identity (interp / wasm kernel / native GPU /
  browser GPU), no per-stage recompiles (the probe at `backend.rs:124` counts them), and a
  deliberately mis-predicted stage still returns within one check granularity of the deadline.

### E — beyond 2³² draws (the "billions" cliff)

The lane index is `u32` end-to-end (`Runner::position`, `backend.rs:199`; GPU params `gpu.rs:256`;
the explicit panic at `gpu.rs:477`). π to 5 digits needs ~6e9 draws — over the cliff. Fix by
**epoch reseeding**: lanes `[e·2³², (e+1)·2³²)` draw under `seed_e = squares(seed, e)` (derived via
the existing `rng::Key` machinery), each epoch a normal ≤ 2³²-lane reduction, accumulators folded in
epoch order. Statistically independent streams, deterministic, zero changes below the boundary
(today the boundary panics, so no existing stream changes). Chunk counts stay u64 in the driver.

### F — follow-up: on-GPU fold (the thin-cone/huge-n unlock)

Not needed for correctness — flagged because precision targets change the GPU calculus. Today every
GPU lane is read back (4 B/lane, `gpu.rs:41`): 6e9 draws = 24 GB of readback, which is *why* the gate
rightly refuses thin cones like π's (~10 ops/draw) at any n. Emitting the Σx/Σx² fold **in the
shader** (workgroup-shared reduction, read back per-workgroup partials — ~256× less traffic) removes
the readback term entirely, making exactly the precision-plan workload — thin cone, enormous n —
GPU-profitable. Scope: a reduce-mode WGSL epilogue in `wgsl_emit.rs`, f32-partial-sum care (Kahan or
f32×2), a gate recalibration for reduce-mode (the `MIN_CONE_OPS = 45` floor is a readback-era
number), and NaN-skip semantics matching `CondMomentsReducer`. Its own mini-plan when this one lands.

### G — examples + docs

- `examples/pi.noise` becomes the flagship and is fully self-describing:
  `pi = 4 * P(X^2 + Y^2 < 1, 1e-5)` (the per-call target — the digits *are* this example's point).
  Nothing else: whether a given run gets there is the runner's `max_time`, and the warning reports
  the gap when it doesn't. Prose explains "ask for digits, not draws" (per the examples style:
  named steps, teenager-friendly comments).
- A rare-event example (insurance tail or similar) showing relative precision doing what fixed-n
  can't.
- LANG.md: the per-call target, the settings' roles and the pragma-declares/`run()`-overrides
  precedence, the stopping rule, the stopping-bias note, `Q`'s exclusion.
- Playground: surface `n_used` (the stats channel `stats::record`, `reduce.rs:146`, already carries
  draw counts) so the "it drew 2.4e9 samples" moment is visible.

### H — soft-stop: deadlines and aborts that keep their numbers

Today cancellation is all-or-nothing by design: a tripped `CancelToken` makes `run_reduction`
return `Err(cancelled)` (`reduce.rs:107` — "that partial answer must never escape"), and in the
browser abort doesn't even ask: `pool.ts` **terminates the worker** (a run never yields its thread,
so there is nothing to politely signal — `index.ts:275`) and the engine scope dies with it. Right
guarantee for teardown; wrong UX for "I've waited long enough, show me what you've got."

The statistics say partial results are legitimate: with counter keying, every chunk is an iid block
of draws, and **which** chunks completed depends on timing, never on the drawn values — so a fold
over any subset of completed chunks is an unbiased estimate with an honest se. And the accumulator
already knows its true size: `MomentAcc.count` (`reduce.rs:337`, exposed at `:345`). The one bug-in-
waiting is that `P`/`E`/`Var` currently compute se from the **requested** n (`let nf = n as f64`,
`builtins.rs:411`, `:480`) — switch them to `acc.count()`, which is also what makes the conditional
reducer's effective-m story uniform.

**Two-level stop.** `exec.rs` grows a second flag beside `cancel`:

- `stop()` — *soft*: workers stop claiming chunks (same per-chunk check cadence, one more relaxed
  load), the driver folds the chunks that completed and returns `Ok(acc)` with its true count plus a
  `stopped: bool` (or `Capped::{Time, Stop}`) marker. The adaptive loop (Track B) treats it exactly
  like budget exhaustion: finish the query with the partial estimate, mark it. Statement evaluation
  then **skips the remaining forcings** (each subsequent query sees the tripped flag before drawing)
  and the run returns a complete `DocResult` — values computed so far, honest se everywhere, plus a
  "stopped early at line N" warning. Bit-reproducibility is explicitly *not* promised for a stopped
  run (the completed-chunk set is timing-dependent); that's the ladder's bottom rung and it's
  labeled.
- `cancel()` — *hard*: unchanged (`Err(cancelled)`, everything discarded). Teardown, page
  navigation, engine drop.

**The `max_time` setting rides the same path.** The deadline is checked where cancellation
already is — once per 16,384-sample chunk (an `Instant::now()` per chunk is noise; `web-time` is
already a dependency for wasm) and between GPU dispatches (1M-lane granularity, ~ms — `gpu.rs:475`).
Deadline passed → trip the soft flag with `Capped::Time`. The adaptive loop additionally sizes
growth stages by measured throughput (ops/sec from completed stages) so a stage roughly lands on the
deadline instead of overshooting it 64×.

**The browser is the hard part, and G3 already solved its shape.** A blocked worker can't receive a
`postMessage`, but wasm-threads memory is a SharedArrayBuffer and the engine already round-trips
main-thread state through SAB+Atomics for the GPU bridge (`gpu-protocol.ts`/`gpu-host.ts`). Give the
run a **stop cell**: one `Int32Array` slot the JS host sets from the main thread and the engine's
soft-stop check reads (the `CancelToken` check site, fed from a SAB-backed flag on wasm). JS surface:

- `run(src, { signal })` — unchanged: hard abort, worker terminated, promise rejects.
- **new** `stop()` on the run handle (or a second `stopSignal`): sets the stop cell; the promise
  **resolves** with the partial `DocResult` (warnings included). The worker survives, and so does
  its engine scope — which the playground's introspection sidecar needs anyway (`run_with_introspection`
  rests on scope persisting).
- CLI: first Ctrl-C = soft stop (print what we have), second = hard kill. Same two rungs.

## Validation

1. **Bit-identity** (the invariant): for targets that stop at n*, the precision-targeted result ==
   the fixed-`n*` result, exactly — per backend (interp, wasm kernel, GPU under `NOISE_FORCE_GPU`),
   including a stage boundary mid-run and an epoch boundary (Track E).
   (Soft-stopped/deadline runs are the explicit exception — Track H.)
2. **It stops early**: `P(coin)` with `rel=1e-2` uses ~10⁴ draws, not 10⁶.
3. **It goes far**: π at `rel=1e-4` reaches 4 digits; profile shows staged n and per-stage backend.
4. **Cap honored + reported**: tiny `max_time` + tight target → warning names the setting and the
   measured shortfall, se honest.
5. **No regression when off**: corpus timings and outputs unchanged with no target set (default path
   untouched); compile-probe counts show zero extra compiles across stages.
6. **Conditionals**: a 1%-acceptance conditional query reaches the same target as an unconditioned
   one (with proportionally more draws).
7. **Per-call arg split**: `P(e, 3)` draws exactly 3; `P(e, 0.5)` is a 50%-relative target; `P(e, 0)`
   and negatives are spanned errors. `E` of a large-magnitude quantity with a count ≥ 1 behaves as
   today.
8. **Soft-stop**: `stop()` mid-run resolves with a partial `DocResult` whose `Est`s carry
   `acc.count()`-based se and a "stopped early" warning; completed statements keep their values;
   hard abort still rejects with nothing. `max_time` on a deliberately huge ask returns within
   deadline + one chunk/dispatch granularity, on CPU and GPU paths.
9. **Pragma/override precedence**: a program pragma applies when no override is passed; the same
   setting passed via `RunOpts`/CLI pins it (the pragma no longer changes the effective value); a
   runtime `precision` looser than the program's produces the looser (cheaper) run, and a tight
   target under a short `max_time` produces the capped warning.
10. **Removed pragmas**: `engine::set_max_ops` / `set_max_opts` / `set_max_samples` are spanned
   errors naming their replacements; the migrated tests (per-call counts / Engine setters) keep
   their goldens.

## Open questions

- **Phase 2 timing**: default-on precision + `max_time`-as-guard is *decided* (see "The end state")
  but deliberately unscheduled — it rewrites every example's digits and test goldens, so it ships
  only after Phase 1 has proven the adaptive machinery on the corpus. The exact default `rel`
  (5e-3?) and each host's default `max_time` are picked then.
- **`Var` targets**: se ≈ var·√(2/n) (`builtins.rs:482`) extrapolates the same way — include it, but
  its se is Gaussian-asymptotic; note the caveat.
- **Seeds**: `P_DEFAULT_SEED = 0` is baked in (`builtins.rs:58`). If run settings are becoming a real
  surface (this plan), a `seed` setting is the cheapest companion addition and `RunOpts`' doc comment
  already promises it ("sample budget/seed later"). In or out of scope here — decide at review.
