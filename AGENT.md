# AGENT.md — working notes for the Noise codebase

Guidance for an AI agent (or any new contributor) working in this repo. Records
**verified facts about the code as it exists today**. For the language's intent see
`GOAL.md`; for the exact syntax/semantics see `LANG.md`; for the roadmap and locked
architecture decisions see `PLAN.md`. Facts below confirmed by running `cargo test` /
`cargo run` on Rust 1.89.

## What this is

**Noise** is a probabilistic, expression-based language (see `GOAL.md`): variables are
meant to hold *probability distributions*, so mathematicians can write Monte Carlo /
queueing simulations idiomatically.

The repo was **re-architected from scratch** (PLAN.md decisions: clean rewrite,
hand-written Pratt parser, bytecode + batched/SIMD sampler, `wasm-bindgen`). The current
state is **Phases 0–3 plus core-model-rework Steps 1–4 complete** (see `PLAN.md`):
- the deterministic calculator, the random-variable runtime (bytecode + batched columnar
  sampler), and the Phase-3 probability surface (`P`, `unif_int`, `bernoulli`, `&&`/`||`,
  lifted `if`, strings, `print`/`round`);
- **Step 1 — recipes + `~`-only drawing:** `unif`/`unif_int`/`bernoulli`/`normal` return an
  undrawn `Value::Recipe`; `~` is the only thing that draws; `=` binds the recipe as-is.
- **Step 2 — user functions:** `f(a)=…` (pure) and `f()~dist` (draws per call).
- **Step 3 — continuous + moments + math:** `normal(μ,σ)`, language-level `E`/`var`, `sqrt`,
  the `pi`/`e` constants.
- **Step 4 — collections:** `Value::Array`, array literals `[a,b,c]` + indexing `xs[i]`/`M[i][j]`,
  `for x in xs {…}` (build-time unroll), `true`/`false` literals, the shaped draw `~[n]`/`~[n, m]`
  (independent batches — subsumes the old `iid`/`iidmat`), the matrix-product `@` operator, and the
  collections/linear-algebra library (`range`, `push`, `len`, `sum`, `count`, `any`, `all`, `max`,
  `min`, `mean`, `dot`, `normsq`, `norm`, `scale`, `vadd`, `vsub`, `vsign`, `matvec`, `normalize`,
  `has_duplicates`). The birthday problem for 23 and a generalized CLT are now one expression.
- **Modules (Rust-style scoping):** builtins are namespaced into `rand` (distributions + `rotation`),
  `math` (`pi`/`e`/`sqrt`/`round`/`log`/`log10`), `vec` (collections/linear-algebra, incl. `mse`),
  and `builtin` (`P`/`E`/`var`/`print`/`range`/`push`/`len`). `builtin` is active by default; the
  rest are **strict** — a bare name errors until `use <module>;` (or a `mod::name` path).
  `module_of()` in `eval.rs` is the single source of truth for membership; `Engine.used` tracks
  active modules.
- **Step 5 — TurboQuant capstone:** `transpose`/`ones`/`zeros`/`iota`, plus `rotation(d)` (random
  orthonormal matrix, Gram–Schmidt of a Gaussian seed lowered into the RV graph) and
  `quantize(v, centroids)` (nearest-centroid / Lloyd–Max snap). `examples/turboquant.noise` now
  reproduces the paper's **actual two-stage algorithm** (arXiv:2504.19874) by Monte Carlo:
  Algorithm 1's MSE-optimal quantizer (rotate → snap → rotate back) matches the D_mse table
  (`0.36/0.117/0.03/0.009` for `b=1..4`) and is biased by `2/π ≈ 0.637` for inner products;
  Algorithm 2 (`TurboQuant_prod` = `(b−1)`-bit MSE stage + 1-bit QJL on the residual) is unbiased
  and matches the D_prod table (`~1.57/0.56/0.18/0.047` over `d`). The faithful distortion win needs
  a true orthonormal rotation (a Gaussian projection leaves the residual too large), which is why
  `rotation` is a built-in rather than the earlier Gaussian shortcut. It is a **recipe** drawn with
  `~` (`Pi ~ rotation(d)`) like any distribution — a structured multivariate draw whose
  Gram–Schmidt source nodes are built in `Engine::draw_rotation` (`eval.rs`).
- **Signal modeling + telecom/DSP examples:** `math::sin`/`cos`/`atan` are **ufuncs** (scalar /
  lifted over RVs / elementwise over arrays — `UnOp::Sin`/`Cos`/`Atan` in the VM), and **arithmetic
  broadcasts over arrays** (`binop_broadcast` in `eval.rs`: array⊕array length-matched, array⊕scalar,
  nesting for `[I,Q]` pairs). The **`signal` module** adds **lazy waveform generators**:
  `signal::sine(f)`/`cosine(f)` return a `Value::Signal(Rc<SignalSpec>)` (a `signal.rs` generator
  that defers scalar/trig ops, O(1) memory) which materializes against a sized array (adopting its
  length, via `binop_signal`) or `signal::sample(sig, n)`. `examples/am_vs_fm.noise` is a full
  `mse(demodulate(modulate(msg) + static), msg)` pipeline (FM beats AM, emergent); `examples/
  nyquist.noise` shows aliasing below `2·f`. Also `math::log`/`log10` (dB), `vec::mse` (signal
  compare), and `Value::Num` Display now trims float dust (`format_num` in `value.rs`).
- **Complex numbers (PLAN-COMPLEX, done):** `Value::Complex { re: Box<Value>, im: Box<Value> }` — a
  complex scalar whose two channels are *real* `Value`s (`Num`/`Est`/`Dist`), so complex arithmetic
  reuses the whole real lifting machinery (`binop_complex` decomposes `* / ^` into real ops) and
  the sample-DAG/VM stay strictly `f64` — **no complex value-channel was added to the VM**. The type
  emerges from `math::i`/`j` (the unit `0+1i`, a constant like `pi`/`e`) or `rand::normal_complex`
  (a CSCG drawn with `~`, yielding `Complex{Dist,Dist}` in `Engine::draw`). Complex-aware `math::`
  ufuncs (`exp` Euler, `abs`/`arg` magnitude/phase, `conj`/`re`/`im`, principal `sqrt`) live in
  `math_ufunc` (dispatched from `lib_call`). `vec` made complex-correct (§7): `normsq`/`norm`/`mse`
  magnitude-based (real out), `dot` bilinear + new `vdot` (Hermitian)/`adjoint`/`outer`; `max`/`min`
  error. **`rand::exp` was renamed `rand::exponential`** (and `exp_int`→`exponential_int`) to free
  `exp` for the math function. Examples: `am_vs_fm_complex.noise` (the radio win), `shor_period.noise`
  (faithful quantum period-finding — complex inverse-QFT interference comb + `rand::categorical`).
- **General surface (PLAN-COMPLEX §8):** the `%` operator (`BinOp::Mod`, floored, real-only, all
  backends), `math::floor`/`ceil` (`UnOp::Floor`/`Ceil`, native cranelift/wasm instructions), and
  **comprehensions** `[for x in xs { body }]` — the `for` loop wrapped in `[ ]` to collect body
  values (`Expr::Comprehension`, build-time unroll that closes over the outer frame —
  `eval_comprehension`). **`continue`** (`Expr::Continue` → `Value::Continue`, a control sentinel
  that short-circuits a `{ block }`) skips a loop iteration / omits a comprehension element — that's
  the filter mechanism (deterministic only). Plus `vec::outer` and `rand::categorical`, and the
  deterministic integer builtins **`math::gcd`** / **`math::modpow`** (square-and-multiply in `i128`,
  exact past `2^53`; `builtins.rs`). `examples/shor_period.noise` is `shor(N)` → the factors of N.

**Next up:** optional perf fast-follow — the fused `Reduce` VM instruction (`PLAN-COLLECTIONS.md`
§3.5) to collapse the `O(d²)` matvec DAG and reach larger `d`. The dynamics fork (sequential/
stateful processes, e.g. M/M/1) remains the separate Phase-3.5 execution mode.

## Workspace layout

A Cargo workspace (`/Cargo.toml`, virtual manifest, `resolver = "2"`, edition 2021):

```
crates/
  noise-core/   # the language: lexer → Pratt parser → AST → tree-walking evaluator.
                # No OS/threads — compiles cleanly toward wasm32. THIS IS THE LIVE CODE.
  noise-cli/    # `noise` binary: file runner + REPL. Depends on noise-core.
legacy/         # the PRE-REWRITE crate, parked & excluded from the workspace (reference
                # only; old lalrpop 0.13/regex 0.2, committed-generated parser). Do not
                # build on it; it exists to preserve the original tests/behavior.
www/            # stale Emscripten asm.js/wasm playground — to be replaced in Phase 5.
GOAL.md LANG.md PLAN.md README.md
```

### noise-core modules
| File | Role |
|------|------|
| `error.rs` | `Span`, `NoiseError` (`ErrorKind`: UnexpectedChar/UnterminatedString/Parse/Runtime), `Result`. Every failure is typed + spanned. **No panics in the pipeline.** |
| `lexer.rs` | Hand-written lexer → `Vec<Token>` ending in `Eof`. Token set is a superset of what Phase 0 evaluates (comparisons, `^`, `!`, `if/else`, strings all tokenize). |
| `ast.rs` | `Expr` (Number/**Bool**/Str/Ident/Unary/Binary/Bind/**FnDef**/Call/Block/If/**Array**/**Index**/**For**/**Use**), `BinOp`, `UnOp`, `BindKind` (Assign=`=`, Sample=`~`), `Spanned`, `Program`, **`split_path`** (splits `mod::name`; qualified names ride inside `Ident`/`Call` name strings). |
| `parser.rs` | Pratt / precedence-climbing parser; `infix_op` precedence table. Disambiguates `f(x)=…`/`f()~…` function defs from calls via `matching_paren_after`. In-module `mod tests`. |
| `value.rs` | `Value`: Num/Bool/Str/Unit/**Recipe(Recipe)** (undrawn distribution)/**Dist(RvId)**/**Est{val,se}** (Monte Carlo estimate; displays rounded to its standard error)/**Array(Rc<Vec<Value>>)** (fixed-length, build-time)/**Signal(Rc<SignalSpec>)** (lazy waveform generator). `format_num` trims float dust from `Num` Display. |
| `signal.rs` | `SignalSpec` (lazy waveform: `wave` + `freq` + a deferred `SigOp` pipeline of scalar/`Unary` ops), `Wave{Sine,Cosine}`, `SigOp`. `sample(n)` materializes. |
| `rng.rs` | Hand-rolled xoshiro256++ PRNG, SplitMix64-seeded. No OS entropy/time/threads — WASM-clean, deterministic. `fill_uniform`/`fill_uniform_int`/**`fill_normal`** (Box–Muller) fill a whole column. |
| `dist.rs` | `RvId` handle; `Distribution` trait + `Uniform`; `RvKind{Num,Bool}`; `Source{Uniform,UniformInt,**Normal**}`; **`Recipe{Uniform,UniformInt,Bernoulli,Normal}`** (undrawn dists); `RvNode{Src,ConstNum,ConstBool,Unary,Binary,Select}`; append-only `RvGraph` (structural sharing). |
| `bytecode.rs` | `compile` (DAG→flat bytecode, CSE via `HashMap<RvId,Reg>`) + the columnar VM `run_batch`; `Inst` (incl. `Normal`, `Select`, And/Or); `BATCH = 1024`. In-module `mod tests`. |
| `sampler.rs` | Forcing path: `for_each_batch`/`sample_n`/`moments` (streaming Welford). Compiles once, allocs the column file once, seeds once. In-module `mod tests`. |
| `builtins.rs` | `pub fn call(name, args, graph: &RvGraph, span)` — **pure/scalar** dispatch: `unif`,`unif_int`,`bernoulli`,`normal`,`sqrt`,`round`,`log`,`log10`,`P`,`E`,`var`,`print`, plus the **pure collection** builtins `range`/`push`/`len`/`iota`/`ones`/`zeros`/`transpose` (no graph access). Takes `&RvGraph` (immutable) — cannot build/draw nodes. |
| `eval.rs` | `Engine { vars, funcs: HashMap<String,Rc<UserFn>>, graph, call_depth }`. `eval`; operator lifting (`lift_binary`/`lift_unary`/`operand_to_rv`); the extracted `binop`/`select`/`indicator` fold helpers; the shaped-draw `eval_sample`/`draw_shaped` and the matrix-product `eval_matmul`/`matmul`; the **collections library** (`lib_call` → `lib_sum`/`lib_dot`/… `Engine` methods, which build/draw so they need `&mut self`); the **module system** (`module_of`/`is_module`/`MODULES`, `Engine.used` set, `resolve_call`/`eval_ident` strict-scoping gates, `Expr::Use` arm); `draw(&mut, Recipe)` (only place sources are created); `call_user_fn` (fresh params-only frame, `MAX_CALL_DEPTH` guard; the new eval arms `eval_array`/`eval_index`/`eval_for` are split out to keep `eval`'s stack frame small for the recursion budget); free helpers `forbid_recipe`, `math_const`. Rust sampling API `sample`/`moments`/`run_rv`. |
| `lib.rs` | Re-exports (incl. `RvId`, `Moments`), `run(src)` helper, the golden-test corpus in `#[cfg(test)] mod tests`. |

## Build & run

```sh
cargo test                       # 191 tests in noise-core, all pass
cargo clippy --all-targets       # must stay clean (watch approx_constant: avoid literals near π)
cargo run -p noise-cli           # REPL
cargo run -p noise-cli -- f.noise  # run a file (prints last statement's value)
```

Modern toolchain; no future-incompat warnings (unlike `legacy/`). `Cargo.lock` is
git-ignored.

## What works today (tested)

### Distributions, recipes, and the `~`/`=` split (core-model Steps 1–3)

- **Distribution constructors return recipes:** `unif(a,b)`, `unif_int(a,b)`, `bernoulli(p)`,
  `normal(μ,σ)` evaluate to an undrawn `Value::Recipe` — *not* a number or a graph node.
- **`~` is the only thing that draws:** `X ~ unif(0,1)` instantiates a fresh sample-DAG node
  (`Engine::draw`). `=` binds the recipe verbatim, so `Die = unif_int(1,6); a ~ Die; b ~ Die`
  gives two *independent* draws. Arithmetic on an undrawn recipe (`unif(0,1)+3`) is a spanned
  error (`forbid_recipe`) pointing at `~`.
- **User functions:** `f(a,b) = …` (pure, lifts over RVs) and `f() ~ dist` (draws per call, so
  `roll()+roll()` is two independent dice). Params-only scope (no closures); user defs shadow
  builtins; a `MAX_CALL_DEPTH` guard turns runaway recursion into an error.
- **Operator lifting:** any op with a `Dist` operand yields a `Dist`; constants fold; comparisons
  give 0/1 indicator columns; `if cond {a} else {b}` over a bool-RV lifts to a per-lane `Select`.
  Structural sharing/CSE: a `Dist` reuses its `RvId` (`X - X == 0` exactly).

### Probability & moments surface (Phase 3 + Step 3)

- **`P(event[, n])`** — Monte Carlo probability of a bool-RV, returned as an `Est` (value + standard
  error) that **displays rounded to its justified precision** and **propagates error through
  arithmetic** (`4*P(C)` → `3.141`, not `3.1412`). Default `n=1e6`, fixed seed (reproducible).
- **`E(x[, n])` / `var(x[, n])`** — expectation/variance of a *numeric* RV (the companions to `P`
  for non-events); `E` of a bool equals `P`.
- **`sqrt(x)`, `pi`, `e`** math helpers (`pi`/`e` are constants resolved even inside function
  bodies). `round`, `print`, string `+` concatenation, `&&`/`||`.
- **Sampling** is lazy (compile RV cone → bytecode → columnar batches under seeded xoshiro256++);
  `P`/`E`/`var` force it, and the Rust API `Engine::sample`/`moments` is still available for tests.

### Deterministic core (Phases 0–1)

- Numbers: integer **and** float literals → `f64`.
- Arithmetic `+ - * / ^` with correct precedence; `^` right-associative.
- Prefix `-` (negate) and `!` (logical not). `-2 ^ 2 == -4` (unary minus looser than `^`).
- Comparisons `== != < > <= >=` → `bool`; equality is same-type only.
- Parentheses, `{ }` blocks (value = last statement; **no new scope** — bindings leak out).
- `if cond { .. } else { .. }`; `else` optional; condition must be `bool`.
- `=` and `~` bindings (right-assoc). Variables in a flat environment.
- `;`-separated statements; trailing `;` optional (improvement over the legacy grammar).
- Typed, spanned errors for undefined vars, type mismatches, and parse failures.

## What is NOT built yet (the gap — see PLAN.md / PLAN-COLLECTIONS.md)

- **TurboQuant capstone (Step 5, next).** Needs `transpose` and `examples/turboquant.noise` — the
  d-dim bias proof. The collections core it builds on (Step 4) is done.
- **No first-class functions / lambdas** (`map(v, f)`), **no `===` description operator**, **no
  `plot`/`explain`**. Element-wise array ops are explicit loops/builtins (no higher-order funcs).
- **No random indexing / random-length loops**, and no native columnar vector representation: each
  array element is its own scalar DAG node, so a `d×d` matvec builds `O(d²)` nodes (fine for
  `d ≤ ~64`; the fused `Reduce`/columnar upgrade is the deferred perf path — PLAN-COLLECTIONS §3.5).
- **Distributions:** `unif`, `unif_int`, `bernoulli`, `normal`. No `beta` yet (`Source`/`Recipe`
  are the extension seam).
- **One sampling pass per query (TODO).** Each `P`/`E` samples its own cone with the default seed;
  queries don't yet share one batch of draws (so cross-query consistency like `P(A&&B) ≤ P(A)`
  isn't guaranteed). The dynamics fork (sequential/stateful processes) is also unbuilt (Phase 3.5).
- **No modern WASM build** / browser playground (Phase 5). The core is WASM-clean (no deps, no
  OS/time/threads) but `wasm32-unknown-unknown` was not exercised in this environment.

## How to extend (practical guidance)

- **`LANG.md` is the contract.** Any syntax/semantics change updates `LANG.md` *and* the
  golden tests in `lib.rs` in the same change. Don't let them drift.
- **Keep the agent skill in sync.** `.claude/skills/noise-lang/SKILL.md` is the *authoring*
  guide agents load to write Noise (and doubles as human-readable docs — linkable from the
  site). When you add or change a language feature — a new builtin or module, a new
  distribution, syntax, or a changed default — update it in the **same change** as `LANG.md`
  and the golden tests. Stale idioms there silently mislead every agent that writes Noise.
- **Grammar/operator changes** are now plain Rust edits in `lexer.rs` + `parser.rs` (the
  `infix_op` table) — no generated-parser regeneration step anymore.
- **Keep errors typed.** Return `NoiseError` with a real `Span`; never `panic!`/`unwrap`
  on user input.
- **New distributions:** add a `Recipe` + `Source` variant (and `Inst` if it samples) — `normal`
  is the worked example (recipe → `Engine::draw` → `Source::Normal` → `Inst::Normal` →
  `Rng::fill_normal`). `RvNode`/`Source`/`Recipe` stay `Clone`/`Copy`/`PartialEq`/alloc-free.
- **Builtin seam:** *pure/scalar* builtins go in `builtins::call` (`&RvGraph`, no node-building).
  Anything that **builds graph nodes or draws** (the Step-4 reducers; the `~[shape]` and `@` arms)
  must be an `Engine` method dispatched from `eval`/`Expr::Call` (needs `&mut self`), reusing
  `lift_binary`. New non-trivial `eval` arms get their own `eval_*` helper (e.g. `eval_sample`,
  `eval_matmul`) so the big `eval` match's stack frame stays small for the recursion budget.
- **New builtin → also register its module:** add the name to `module_of()` in `eval.rs` (pick
  `rand`/`math`/`vec`/`builtin`) or it's unreachable under strict scoping. Membership (scoping) is
  independent of where the implementation lives (`builtins::call` vs `lib_call`).
- **Register allocation:** `bytecode::compile` uses one register per distinct node (no
  liveness reuse). CSE (sharing) is a correctness requirement and IS done; slot reuse is
  deferred to Phase 4.
- **Keep `noise-core` free of OS/threads** so the wasm32 path stays clean.

## Status snapshot (2026-06-25)

- Deterministic core + RV runtime + Phase-3 probability surface + core-model rework Steps 1–4
  (recipes/`~`-drawing, user functions, `normal`/`E`/`var`/`sqrt`/`pi`, **collections**: arrays,
  indexing, `for`, `true`/`false`, the `~[shape]` draw, the `@` product, the `sum`/`dot`/`has_duplicates`/… library) **plus a
  Rust-style module system** (`rand`/`math`/`vec` strict, `builtin` default; `use mod;` or
  `mod::name` paths) complete and green (**130 tests**, clippy clean). 21 runnable examples in
  `examples/` (each checks an analytic value), all carrying their `use` lines.
- **Uncommitted:** Steps 1–4 and the new tests/examples are in the working tree but not yet
  committed — a fresh session sees them in the tree, not in `git log`.
- **Next up: Step 5 — `transpose` + `examples/turboquant.noise`** (the d-dim bias capstone;
  `TURBOQUANT.md`). Optional perf fast-follow: the fused `Reduce` VM instruction
  (`PLAN-COLLECTIONS.md` §3.5).
