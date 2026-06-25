# PLAN.md — implementation roadmap for Noise

The build plan for taking Noise from "deterministic calculator" to the fast,
browser-runnable probabilistic language described in `GOAL.md`. For the current state of
the code see `AGENT.md`.

## Decisions locked (2026-06-25)

1. **Clean re-architecture.** Keep the repo, the `README`/`GOAL.md` vision, and the
   existing tests as a *reference corpus*, but rebuild the engine and value model from
   scratch on a modern toolchain. The current 2015-edition crate, 2017-era `lalrpop`/
   `regex`, and committed-generated-parser workflow are retired.
2. **Bytecode + batched / SIMD sampling** is the execution model for the simulation
   engine. RV expressions compile to flat bytecode; samples are evaluated **N at a time,
   column-oriented**, so the hot loop is cache-friendly and vectorizes (incl. WASM SIMD).
   No per-sample AST walking in the hot path.
3. **Hand-written Pratt (precedence-climbing) parser.** Full control over the operator
   set, fast compiles, no grammar-regeneration step, easy for agents to extend. `lalrpop`
   is dropped.
4. **WASM via `wasm-bindgen` / `wasm-pack`** targeting `wasm32-unknown-unknown`. The old
   Emscripten `www/` artifacts are replaced.
5. **The core model (2026-06-25, official — see `LANG.md` "The core model").** Two ideas:
   (a) *everything is a distribution* — a `number` is a point mass; `number`/`bool`/`dist`/
   `estimate` unify into one distribution notion (constants stay constant-folded for speed);
   (b) *`~` is a stochastic node, `=` is a deterministic node* — the Stan/BUGS/PyMC split.
   `name ~ dist` draws a fresh random variable (`~` is the only thing that draws; its RHS must
   be a distribution); `name = expr` is a transform or a plain value/recipe (no new
   randomness). A name is one fixed node (no re-draw on use → no coloring, no gotcha);
   independence comes from *separate* `~` declarations. **You cannot do arithmetic on an
   undrawn distribution** — `~` it first. Functions: `f() = …` pure, `f() ~ dist` stochastic.
   This supersedes both the earlier "`~` binds a distribution / same name = same draw" framing
   *and* an interim "`~` = lazy rule" idea (rejected — lazy re-draw colors variables invisibly
   and doesn't match math notation, where `X ~ Dist` is one fixed random variable).

## Core-model rework (next major implementation step)

`LANG.md` now specifies the core model as the target; the code does **not** match yet (it's
flagged with **Status:** notes there). The work, in dependency order:

1. **Distributions as first-class recipes + `~`-only drawing.** ✅ **Done (2026-06-25).**
   `unif`/`unif_int`/`bernoulli` return `Value::Recipe` (an undrawn distribution; see
   `dist::Recipe`), not a graph node. `name ~ dist` is the only thing that draws —
   `Engine::draw` instantiates fresh source node(s); `=` binds the recipe/derived-RV/constant
   verbatim. `forbid_recipe` enforces "no arithmetic on an undrawn distribution" (a recipe in an
   operand/condition position is a spanned error pointing at `~`). A drawn name stays one fixed
   node, so there is no re-draw and no coloring. Verified: `Die = unif_int(1,6); a ~ Die; b ~
   Die; P(a==b) ≈ 1/6`, all 15 examples unchanged, 45 tests green, clippy clean.
2. **User-defined functions.** ✅ **Done (2026-06-25).** `f(params) = body` (deterministic) and
   `f(params) ~ dist` (stochastic — each call draws via `draw_if_recipe`). The parser
   disambiguates a definition from a call by looking past the matching `)` for a `=`/`~`
   (`matching_paren_after`). A call binds args to params in a *fresh frame* (params-only scope —
   pure in arguments, no outer-variable capture/closures yet; user defs shadow builtins), then
   restores the caller frame; a `MAX_CALL_DEPTH` guard turns runaway recursion into a clean
   error. Verified: `roll() ~ unif_int(1,6); P(roll()+roll()==7) ≈ 1/6`, `max(a,b)=…` lifts over
   RVs, 57 tests green, clippy clean. See `examples/functions.noise`.
3. **Continuous distributions + moments + math builtins.** The smallest, most independently
   useful primitives — and the first hard requirements surfaced by the TurboQuant case study
   (see `TURBOQUANT.md`). In dependency order:
   - **`normal(μ, σ)`** distribution: a new `Source` variant + a Gaussian RNG column fill
     (Box–Muller; one extra `next_f64` per lane). Foundational — every matrix/projection in
     TurboQuant is Gaussian, and this replaces the 12-uniform CLT fake in
     `examples/clt_normal.noise`. (`beta(α,β)` is a later, optional sibling.)
   - **`E(expr)` / `var(expr)`**: language-level expectation/variance of a *numeric* RV,
     returning an `Est` with a Monte Carlo standard error — surfacing the existing
     `sampler::moments` the same way `P` surfaces the indicator mean. Today only `P` (a bool
     mean) is exposed; every TurboQuant claim is an `E[...]` / `Var[...]`.
   - **`sqrt`, `pi`**: trivial math builtins for scaling constants (`√(2/(πd))` etc.).
   - (Conceptual) collapse `number`/`bool`/`dist`/`estimate` toward one distribution notion,
     keeping the constant-fold fast path so point-mass arithmetic never samples; make
     `mean`/`variance`/`P`/`E` total over all of it.
4. **Collections (Go-style) + reduction, built on the above.** Arrays + `iid(dist, n)` (and
   `iid(dist, n, m)` for matrices) + `range` + `len` + indexing + an explicit `for x in xs { … }`
   loop that unrolls into the graph at build time (each `~` inside a loop body is a fresh node →
   independence). **A build-time `sum`/reduce** (a `for`-accumulator lowering to an unrolled
   `Add` chain — no new VM node) is the keystone that makes cross-coordinate reductions
   (`⟨y,x⟩ = Σ yᵢxᵢ`, `‖r‖² = Σ rᵢ²`) expressible. Then the reducers (`sum`, `count`, `any`,
   `max`, `has_duplicate`, …) **and a linear-algebra prelude** (`dot`, `norm`, `matvec`,
   `transpose`, `scale`, `map`, `sign`, `normalize`) are written **in Noise** — keeping the core
   minimal (Go philosophy: small core, library in-language).
5. **TurboQuant flagship example + adversarial test.** Once 3–4 land, write
   `examples/turboquant.noise` reproducing the headline bias: `E[⟨y,x̃⟩]/⟨y,x⟩ ≈ 0.637` for the
   MSE quantizer vs `≈ 1.0` for the two-stage `prod` quantizer (see `TURBOQUANT.md` §3/§5). This
   is an end-to-end test of `normal` + vectors + reduce + `E` + the prelude in one program.

Guardrail unchanged: loops/recursion unroll at **build time**, so they must run a deterministic
number of times. A *random* number of iterations is still the dynamics fork (Phase 3.5).
TurboQuant itself is **feed-forward** (no recurrence), so it needs none of Phase 3.5 — only the
vectors/reduce above. Reaching the paper's real `d = 1536` (vs. a `d = 32–256` proof) would need
a **native vector-column representation** (a register holding `[BATCH × d]` instead of one column
per scalar node); that is a Phase-4 VM upgrade, not required to demonstrate the claim.

## Performance thesis (why this is fast)

The hot operation in Monte Carlo is "evaluate this expression over millions of draws."
Interpreting the AST once per draw is pointer-chasing death. Instead:

1. Lower the RV expression into a **sample DAG**: `~` nodes are distribution sources, `=`
   values are constants, arithmetic/comparison are pure ops.
2. Compile the DAG to **flat bytecode** over a register file of **columns** — each
   register is a contiguous `[f64; BATCH]` (or a SIMD-lane buffer).
3. Run the whole batch through each instruction before moving to the next. One pass over
   bytecode evaluates `BATCH` samples. `P(C)` = fraction of `true` lanes, accumulated
   across batches until a target sample count / convergence.

This layout is the same one that makes WASM SIMD (`v128`, 2×f64 lanes) and autovectorizing
pay off, so it serves "fast" and "browser" with one design.

## Target crate layout

A small Cargo workspace separates the portable core (compiles to WASM, no OS deps) from
the host shells:

```
crates/
  noise-core/     # lexer, Pratt parser, AST, lowering, bytecode, batch sampler, builtins.
                  # #![no_std]-friendly where practical; no OS/threads in the hot path.
  noise-cli/      # REPL + file runner binary (native). Depends on noise-core.
  noise-wasm/     # wasm-bindgen bindings exposing run/eval/sample/plot-data to JS.
www/              # browser playground (editor + run + plot rendering) on the wasm build.
LANG.md           # the language spec — single source of truth for parser & engine.
```

Within `noise-core`, keep stable module seams so agents can fan out without colliding:
`lexer`, `parser`, `ast`, `value`, `lower` (AST→DAG), `bytecode`, `vm` (batch sampler),
`rng`, `dist` (distributions), `builtins`. Each distribution/builtin is an independent
unit behind a trait.

## Core data model

```rust
// Deterministic values
enum Value { Num(f64), Bool(bool), Text(String), Dist(DistId), Fn(FnId), Unit }

// A random variable is a node in the sample DAG, lowered to bytecode.
// At runtime a "register" holds a column of samples:
struct Column { lanes: Box<[f64]> }   // len == BATCH; bool stored as 0.0/1.0

// Distributions implement a trait so each is an independent, testable unit:
trait Distribution {
    fn sample_into(&self, rng: &mut Rng, out: &mut [f64]);  // fill a column
}
```

- `~` binds an identifier to a `Dist`/sample-DAG node (lazy — nothing is sampled until a
  query forces it).
- `=` binds an ordinary `Value`.
- Arithmetic/comparison **lift** over RVs: any op with ≥1 RV operand produces an RV node.
- `P(expr)` forces sampling of the boolean RV and returns the empirical probability.

## RNG

Fast, seedable, WASM-safe PRNG in the hot loop (e.g. xoshiro256++ or PCG) — **not** OS
entropy per draw. Seed once (from JS `crypto`/CLI flag/time) for reproducibility. Sampling
fills whole columns at a time.

## Phased roadmap

Each phase has a **definition of done** so progress is unambiguous and so the agent fleet
(Phase 6) has clear acceptance gates.

### Phase 0 — Foundation (solo, first; this is the contract)
- Create the workspace; modern edition; drop `lalrpop`/`regex` build path.
- Write **`LANG.md`**: tokens, grammar, operator precedence/associativity, the `~` vs `=`
  semantics, type/coercion rules, and the builtin signatures (`unif`, `P`, `plot`,
  `explain`, `===`).
- Stand up the **golden-test harness**: a corpus of `input → expected` cases runnable via
  `cargo test`, seeded RNG for deterministic probabilistic assertions (assert within
  tolerance). Port the 4 existing tests as the first cases.
- Wire the `wasm-bindgen` build and a minimal `noise-cli` REPL so programs are runnable
  end to end on day one (even if the language is tiny).
- **Done when:** `cargo test`, `cargo run -p noise-cli`, and a `wasm-pack build` all work
  on an empty-but-real language (numbers + arithmetic), and `LANG.md` is reviewed.

### Phase 1 — Deterministic core, clean
- Hand-written lexer + Pratt parser for the full deterministic surface: float & negative
  literals, `+ - * / **`, comparisons (`> < == != >= <=`), `!`, parentheses, `{ }` blocks,
  `if/else`, `;`-separated statements, identifiers, strings.
- Tree-walking evaluator over `Value` with **real error handling** (no `panic!`/`unwrap`;
  typed errors with spans). Remove the debug `println!`.
- **Done when:** every non-probabilistic example in `README`/`GOAL.md` parses and
  evaluates correctly, with golden tests and error-case tests.

### Phase 2 — Random-variable runtime (the heart)
- `Distribution` trait + `unif`; the `~` binding; AST→sample-DAG lowering; bytecode;
  the **batched/columnar VM**; the PRNG.
- Operators lift over RVs; deterministic constants fold.
- **Done when:** `X ~ unif(-1,1)` then arithmetic over `X` produces correct empirical
  moments (mean/variance within tolerance) under a seeded RNG, benchmarked.

### Phase 3 — Probability & builtins (re-prioritized after the design review)

An adversarial design review (3 independent reviewers, 2026-06-25) reshaped this phase. The
guiding correction: **capability before cosmetics, and correctness before either.** `plot`,
`explain`, and `===` are ergonomics on a language that can't yet express several of its own
headline examples — they are demoted. The items that make examples *correct* and unlock
real modeling are promoted. In priority order:

1. **`P(expr)` with honest statistics.** Batched sampling that reports an error bar and sample
   count (e.g. `0.167 ± 0.004 (N=1e6)`), accepts an explicit `N`, and has a default
   convergence stop. Returns a probability in `[0,1]` — so the π example is `4 * P(C)`, not
   `P(C)`. A bare stochastic float presented as fact is the one thing a probabilistic language
   must not ship.
2. **One shared sampling pass per run.** All queries evaluate against the same draws of the
   shared RV graph, so `P(A)`, `P(B)`, `P(A && B)` are mutually consistent. (Not per-query
   re-seeding.) Decide this *before* building `P`; retrofitting is painful.
3. **Discrete distributions: `unif_int(a,b)` (+ `bernoulli(p)`).** Without a discrete
   distribution the dice example is `P=0` (continuous `==` is measure-zero). Cheap: one
   `Source` variant each. Emit a warning on `==`/`!=` over a *continuous* RV.
4. **Boolean `&&` / `||` lifted over 0/1 indicator columns.** Needed for compound events and
   conditions (`X > 0 && Y < 1`). Trivial in the columnar VM (elementwise min/max). Add the
   tokens to the lexer + the `infix_op` table.
5. **Conditioning via `observe` / `P(A | B)` (rejection sampling).** ~30 lines on the existing
   batch loop: draw a batch, drop lanes where `B` is false, report `A` over survivors. This is
   the single change that most closes the gap from "prior calculator" to "can answer
   conditional questions." (Importance weighting, for tiny `P(B)`, is a later refinement.)
6. **User-defined functions where each *call* draws fresh.** `roll() ~ unif_int(1,6)` then
   `roll() + roll()` = two independent dice — the correct, general way to express N iid copies
   (the binding-shares / call-refreshes rule). Pairs with a `count`/`all` combinator so
   "10 in a row" is a *model*, not `P(X)**10` arithmetic.
7. **Lift `if`/branching over bool-RVs.** Today `if <dist>` is a runtime error, which blocks
   all data-dependent sampling and `max`/`min` over RVs. Needed for any conditional dynamics.

**Demoted (do after 1–7):** `plot(...)` (histogram data for JS), `explain(...)` (must report
"estimated p ± e from N draws", never an exact fraction it didn't compute), and `===` — which
the review flagged as a smell (a third `=`-like token that reads as strict-equality and
duplicates the name); prefer a comment or a label argument on `explain`/`plot`.

- **Done when:** the corrected π example returns `4*P(C) ≈ 3.14` *with an error bar*, the
  discrete dice returns `P(Dice==4) = 1/6 ± tol`, and `P(A==4 && B==4) ≈ 1/36` for two
  independent dice — in CLI and browser.

### Phase 3.5 — The dynamics fork (decision required, not yet scheduled)

The review's sharpest structural finding: **Noise today is a static, i.i.d., scalar Monte
Carlo engine, and the columnar VM is built for exactly that.** Dynamic stochastic systems —
random walks, Markov chains, and the headline **M/M/1 queue** (Lindley's recursion
`W_{n+1} = max(0, W_n + S_n − A_{n+1})`) — need *sequential per-lane state*: step `t+1`
depends on step `t`. The current engine evaluates one DAG across `BATCH` **independent** lanes
that never communicate, so it structurally cannot run a recurrence. No item on Phases 0–5 adds
this. "Queueing simulations are trivial" is therefore **false as specified** (the docs have
been corrected to say so).

Making dynamics real is an **architecture fork**, not a builtin. It needs, in dependency
order: (a) iteration/recursion that carries state across an index, (b) sequences/arrays of RVs
(sample *trajectories*, not just scalar columns), (c) dependent/lifted branching (item 7
above), and (d) likely a *second execution mode* — a per-lane stepper that samples whole paths
— since the column-per-scalar-node VM is the wrong shape for recurrences.

**This is a genuine strategic choice for the maintainer**, and it should be made before Phase 4
spends effort SIMD-tuning a VM that may not be the right machine for half the stated mission:

- **Track A — Own the Monte Carlo identity.** Keep Noise a best-in-class static RV-algebra /
  forward-MC calculator. Ship Phase 3, polish, browser. Drop dynamics from the pitch. Lower
  risk; the engine and the identity match.
- **Track B — Commit to dynamic systems.** Design the sequential execution mode and the
  trajectory type, making the queueing claim true. Higher effort and a real second engine, but
  it's the only path to the original ambition.

### Phase 4 — Speed pass
- WASM SIMD (`v128`) path for the columnar ops; batch-size tuning; criterion benchmarks
  with regression gates; optional parallel chains via web workers / native threads.
- **Done when:** documented throughput (samples/sec) targets met natively and in-browser,
  with a benchmark suite guarding against regressions.

**Progress (2026-06-25).** The execution model is being generalized from "interpret the bytecode"
to "compile the graph" behind a swappable backend seam, driven by a criterion harness.

- **Bench harness.** ✅ `crates/noise-core/benches/sampling.rs` reports **samples/sec** for seven
  representative graphs (tiny/RNG-bound, transcendental-bound, arithmetic-dense, and a mixed
  normal+arithmetic case for the JIT crossover). Run with `cargo bench -p noise-core`. Single-
  thread interpreter baseline: `dice_sum` 119 M/s, `exp_tail` 78, `normal_sum` 39, `poly_deep` 36,
  `poly_wide` 34, `pi_indicator` 30, `normal_poly` 28.
- **B0 — backend seam.** ✅ `backend.rs`: `Backend`/`Sampler` traits + `InterpBackend` wrapping the
  columnar VM. `sampler::for_each_batch` drives any `Sampler` (it owns its RNG via `reseed`), so a
  JIT slots in without touching call sites. Zero throughput change.
- **B1 — Cranelift native JIT.** ✅ `jit.rs` (feature `jit`, native only). One **fused per-lane
  kernel**: xoshiro inlined into CLIF, sources drawn + whole DAG computed in registers, one `f64`
  stored per lane (no intermediate columns). Covers `unif`/`unif_int`, `+ - * /`, integer-const
  `**`, comparisons, `&& ||`, unary `- !`, lifted `if`. A `supported()` pre-pass makes any other
  graph **fall back to the interpreter**, so `--features jit` never changes a result. Measured vs
  baseline: `pi_indicator` ×4.4 (30→134 M/s), `poly_deep` ×3.4 (36→123), `poly_wide` ×2.1 (34→71),
  `dice_sum` +12% (RNG-bound), `normal_sum` flat (falls back — Box–Muller is B2). 145 tests green
  with the JIT as the default forcing path.

- **B2 — full op coverage + profitability gate.** ✅ The JIT now emits every node except `Poisson`
  (Knuth's variable-length per-lane loop, still interpreter-only): `normal`/`exp`/`geometric`
  sources, `sin`/`cos`/`atan`/`sign`/`round` ufuncs, and non-constant `**` — via native `sqrt`/
  `floor` instructions plus six `extern "C"` math shims (`ln`/`sin`/`cos`/`atan`/`round`/`pow`)
  registered as JIT symbols. **Key finding:** per-lane transcendentals are non-inlined libcalls and
  (for `normal`) do ~2× the work of the interpreter's pair-sharing Box–Muller column fill, so a
  transcendental-bound graph is *slower* JITted (`normal_sum` 39→22 M/s). Fix: `jit_profitable`
  routes to the JIT only when fused-node count > transcendental-libcall weight — calibrated so
  `normal_sum`/`exp_tail` stay interpreted (no regression) while `normal_poly` (one normal feeding
  a deep polynomial) fuses for ~1.4× (28→38 M/s). The capability is retained for the future
  WASM-emitter backend and for mixed-graph fusion. 147 tests green.

- **Multicore — deterministic parallel reduction.** ✅ `reduce.rs`. Sampling is expressed as a
  **monoid fold**: a `Reducer` trait (commutative monoid over sample columns — `identity`/`absorb`/
  `merge`) with `MomentsReducer` (Welford + Chan's parallel merge) powering `P`/`E`/`Var`. The
  driver splits `N` into **fixed, deterministically-seeded chunks** (`splitmix(seed, chunk_index)`)
  and merges per-chunk accumulators in **chunk-index order**, so the result is **bit-for-bit
  identical for any thread count** (proven by a test across 1/2/3/5/8 threads). Native uses
  `std::thread::scope` work-stealing (zero-dep; gated `#[cfg(not(wasm32))]`); wasm and small `N`
  run the same chunks sequentially. The `Reducer` monoid is executor-agnostic, so a rayon backend
  could drop in unchanged.
- **Compile-once-share-program.** ✅ The backend seam was split into an immutable, `Send + Sync`
  `Program` (the bytecode / JIT kernel, `Arc`-shared) and a cheap per-worker `Runner` (scratch +
  RNG). The reducer now compiles **once** and fans out runners by reference, instead of
  recompiling on every thread — which was the multicore ceiling (esp. for the JIT, ~1–2 ms compile
  × every worker). This **~doubled** N=1e6 multicore throughput and lifted parallel efficiency from
  ~25% to ~50–56% on 14 cores. Final stacked numbers (vs the original interpreter single-thread
  baseline): `pi_indicator` 30 → **896 M/s (≈30×)**, `poly_deep` 36 → **900 M/s (≈25×)**,
  `poly_wide` 34 → **542 M/s (≈16×)**, `dice_sum` 119 → **1.05 Gelem/s (≈9×)**; transcendental
  cases run interpreter+parallel: `exp_tail` 78 → **572 M/s**, `normal_sum` 39 → **252 M/s**.
  - *Note on `dice_sum`:* it scales **7.9× on 14 cores** (best efficiency) — it's RNG-bound, which
    parallelizes perfectly. Its modest *total* (≈9×) is only because the JIT layer barely helps a
    2-node graph (+12%, nothing to fuse); the earlier apparent "not scaling" was per-thread compile
    overhead dominating its tiny per-core slice, which this seam split removed.

Browser note: **B1/B2 (Cranelift) cannot run in the browser** — a WASM sandbox can't emit/execute
native code. What's portable is the `RvGraph` IR, not the backend. The browser's equivalent is a
future **WASM-emitter backend** (emit `wasm` bytes → JS host `instantiate`s) behind the same
`Backend`/`Program`/`Runner` seam; the interpreter remains the wasm32 default until then. Note the
B2 finding will recur there (a WASM kernel's `ln`/`cos` are also calls), so the profitability gate
is reusable. Remaining Phase-4 options: **B3** (SIMD the per-lane kernel via CLIF vector types) and
the **WASM-emitter** backend (the browser story).

### Phase 5 — Browser playground
- Real web UI on the `noise-wasm` build: code editor, run button, distribution/plot
  rendering, shareable examples. Replaces the legacy Emscripten `www/`.
- **Done when:** the π and dice programs are runnable and visualized entirely in the
  browser.

### Phase 6 — Agent fleet to completion
- With `LANG.md`, the test harness, and the module seams frozen, fan out a `Workflow` of
  agents — one unit of work per distribution, builtin, operator, or language feature —
  each gated by golden tests so parallel work can't silently regress. Use adversarial
  verification on probabilistic results (statistical assertions are easy to fool).
- **Prereq:** Phases 0–2 must be stable first; the fleet needs a frozen contract and a
  trustworthy test gate to fan out against.

## Immediate next step

Begin **Phase 0**: scaffold the workspace, draft `LANG.md`, and port the existing 4 tests
into the new golden-test harness. Everything downstream depends on that contract existing.
