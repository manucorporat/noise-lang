# PLAN-WEBGPU — the GPU as a fourth lowering of the RvGraph

**Date:** 2026-07-13 · **Status: G0–G2 + G1b + G4a/G4b/G4c LANDED (2026-07-15). The GPU works;
turboquant and prisoners both run on it.**
Native corpus **3935.6 ms → 873.0 ms (4.5×)** with `--features jit,gpu`; **prisoners 304×**
(851→2.8 ms), **turboquant 19×** (1294→68 ms), `secretary` 12.2×, `barrier_option` 4.3×, and no
regressions. Every node now lowers except Poisson (declined) and draws inside a rolled loop (fall
back to unrolling). Remaining: **G3** (browser host — bridge DECIDED as (b) `Atomics.wait`, full spec
below; needs a live isolated playground + Chrome-WebGPU session to build and verify), and an optional
**G4d** (Poisson, joint kernels, draws-in-loop).
**Depends on PLAN-PREGPU** (complete), which moved every cross-backend decision out of this plan —
the GPU lands as just another backend under one shared contract.

## G0 results (measured — `tools/gpu-spike`, full write-up in `tools/gpu-spike/RESULTS.md`)

The spike answered its four questions on an M4 Pro, and **three of this plan's assumptions did not
survive**. The corrections are folded into the sections below; the headlines:

| G0 asked | answer |
|---|---|
| Does `squares64` survive WGSL (no `u64`)? | **Yes, bit for bit** — 4096/4096 lanes match `noise_core::rng`. Emulated on `vec2<u32>`; `rotl(x,32)` is a free half-swap, and a *wrapping* 64×64 multiply needs only one wide partial product. **C0's certification carries onto the GPU unchanged.** |
| What does the certified hash cost? | **~1.5×** on the most RNG-dense shape — not the feared "dominates". (A cheap u32 hash times identically to *no hash*: it is fully hidden behind the transcendentals. squares64 is 1.5× on top of free.) **Decision: keep squares64.** The plan's two escape hatches — a local GPU generator, or re-litigating the pcg family — are **closed**. |
| Do the giant shaders compile? | Yes, but **compile is the binding constraint, not throughput** — and it tracks *emitted instructions*, not graph nodes, because every RNG source inlines ~150 ops of emulated hash (~6.5 ms of compile each). `turboquant`-scale: **1.9 s cold, per forcing.** |
| What does a dispatch cost? | **~1.2 ms** flat (buffers pre-allocated), until the work dominates. |

**The finding that reshapes G1** (§"Where it plugs in"): the emitter must **emit array draws as
loops, not unroll them**. Same graph, same draws, 100M normal draws, against the *multicore JIT*:

| | cold compile | dispatch | end-to-end |
|---|---|---|---|
| CPU (Cranelift JIT, multicore) | — | 96 ms | 1.0× |
| GPU, unrolled (what `wasm_emit`/`jit` would do) | 572 ms | 3.4 ms (28×) | **6.0× SLOWER** |
| GPU, **looped** | **30 ms** | 5.0 ms (19×) | **2.7× faster** |

**And a correction with teeth:** the GPU **fuses multiply-add** (measured: 4095/4096 lanes match a
fused result, 3/4096 match mul-then-add). WGSL permits it; there is no portable off switch. So
bit-identical *lane arithmetic* with the CPU backends is **impossible**, and the risk table's
mitigation — "emit the shared `approx.rs` f32 polynomials instead of the built-ins, trading some of
the hardware-transcendental win for bitwise portability" — **does not work**: the polynomial gets
contracted too. Measured, it delivers neither bitwise parity nor better accuracy (both configs land
at 7.15e-7 max deviation on `normal`) and costs 1.4×. **Decision: use the WGSL built-ins.**

The contract is therefore **two-tier**, and this is what G1's conformance suite asserts:

| tier | what | claim |
|---|---|---|
| **1** | the draws (integer hash → 24-bit uniforms) | **bit-identical** on every backend, GPU included |
| **2** | everything computed from them in f32 | **ULP-close** — ≤ 1e-6 absolute, asserted per device |

Tier 1 holds precisely because it is integer arithmetic: there is nothing to contract. That is also
the tier the RNG certification lives in, which is why it is the one that had to hold.

Dependency status (PREGPU is otherwise complete):

| PREGPU gives us | status | needed by |
|---|---|---|
| f32 lanes in all modes | ✅ LANDED (Track B) | G1 — the GPU's numeric contract, unbent |
| **squares64** counter RNG in all modes (not pcg4d-3r — C0's criterion 8 killed that family) | ✅ LANDED (Track C) | G1 — identical draw streams, bit for bit |
| cancellation (native `CancelToken`, browser `AbortSignal`) | ✅ LANDED (Track A) | G3 — abort in-flight dispatches |
| **async evaluator** (`run_async`, `sampler::*_async`) | ⏸ DEFERRED — *this plan is now its only consumer, and may not need it either* | **G3 only, and only under bridge option (a)** |

**G0/G1/G2 need no async at all** — native `wgpu` polls synchronously, and that is the harness for
all three. Async is a **G3** (browser host) requirement, which is exactly why PREGPU stopped short
of building it: it would have been a speculative 400–800-line refactor of the evaluator spine with
no consumer. Build it when G3 does.

## The thesis

A Noise query is already shader-shaped: one pure per-lane kernel (`RvGraph` cone), run N
independent times, folded by a commutative monoid (`Reducer`). That is *exactly* the WebGPU
compute model — lanes are invocations, the fold is a reduction. We already have three
lowerings of the same IR (bytecode interpreter, Cranelift JIT, WASM emitter); WGSL is a
fourth, and structurally the easiest one:

- **No RNG state chain — and G0 settled the RNG question.** PLAN-PREGPU Track C moved every
  backend to a counter-based keyed generator, and the emitter spells the identical hash in WGSL
  for **bit-identical** draws — *verified*, 4096/4096 lanes (G0 finding 1). The fear was that
  Squares' wide multiplies, which WGSL must emulate (no `u64`, no `mulhi`), would cost ~70–90 ALU
  ops per uniform against ~10 for a pcg-style hash and swamp a normal-dense kernel. **Measured, the
  tax is ~1.5×** — real but nowhere near dominant, and the emulation is cheaper than feared because
  `rotl(x,32)` is a free half-swap and a *wrapping* 64×64 multiply needs only one wide partial
  product. **We keep squares64.** No state upload, no streams; each source node's offset is a
  compile-time constant, so every uniform in the kernel is an independent hash.
- **No transcendental inlining.** WGSL has native `log`/`exp`/`sin`/`cos`/`atan`/`pow`, and G0
  says use them: the `approx.rs` polynomial apparatus (built because `normal` costs ln+sincos per
  draw on CPU) is not merely unnecessary here, it is a **1.4× pure loss** — it cannot buy bitwise
  parity (the GPU contracts multiply-adds regardless) and it is no more accurate. Box–Muller is four
  built-ins. The ops the CPU cost model charges as expensive libcalls are the GPU's *cheapest* —
  which is precisely what makes the transcendental-heavy demos (`am_vs_fm`, `barrier_option`) the
  biggest winners.
- **But loops, not unrolling.** The one place the GPU is *not* like the other three backends. The
  CPU emitters flatten a cone into a scalar statement chain because their targets offer nothing
  better; WGSL has real loops, and G0 measured a **19× compile-time difference** between the two
  (§G0 results). An array draw folded by a vector op — which is what `barrier_option`,
  `turboquant` and `am_vs_fm` all are — is a loop by construction, and must be emitted as one.
- **The gate logic already exists.** `kernel::profitable(graph, root, …, draws, min_draws)`
  is the same shape: fixed cost (pipeline compile + dispatch/readback latency) amortized
  by per-draw savings. GPU just gates on `draws × cone_ops` (total work) instead of
  `draws` alone, with its own measured constant.

```
noise-core (Rust)                          browser (www playground)
  simplify → RvGraph cone                    nz_gpu_* shim (inline_js, mirrors nz_kernel_*)
    └─ wgsl_emit.rs: cone → WGSL text (NEW)    ├─ owns adapter/device, content-addressed
         └─ reduce driver dispatches            │  pipeline cache (like the kernel LRU)
            chunks to the GPU seam (NEW)        └─ dispatch + mapAsync + readback
  native: wgpu behind `gpu` feature —          engine runs in a Worker; Atomics.wait
  same emitter, blocking poll (tests/CLI)      bridges sync eval ↔ async GPU (NEW)
```

## The 32-bit question (resolved upstream — PLAN-PREGPU Track B)

WGSL has no `f64` and no timeline for one. Originally this plan carried a GPU-only "f32
mode" with a weakened cross-backend contract; **PLAN-PREGPU instead moves every backend to
f32 lanes / f64 aggregation first**, so by the time this plan runs there is no numeric
fork — the GPU computes the same f32 lanes every CPU backend computes, over bit-identical
pcg4d-3r draws. What remains GPU-specific:

- **Aggregation placement.** Reducers stay f64 and stay off the GPU in phase 1: read raw
  f32 samples back, widen, feed the existing `Reducer::absorb` (unchanged fold). Phase 2
  (optional): per-workgroup partial `{count, Σx, Σx²}` over ≤4096 lanes in f32 (safe at
  that size), CPU folds partials in f64 — shrinks a 1M-sample readback from 4 MB to
  ~50 KB.
- **`unif_int` needs no special case** — the ≤2²⁴ cap is already the language rule in all
  modes (PLAN-PREGPU B); the WGSL lowering draws in u32 like everyone else.
- **The NaN conditioning sentinel survives** — f32 has NaN; `select(C, q, NaN)` lowers as-is.
- **Residual determinism gap, GPU only — and G0 moved its boundary.** The plan assumed GPU lanes
  would be "bit-identical to CPU for add/mul/select graphs and tight-ULP elsewhere". **That is
  wrong**: the GPU *fuses multiply-add*, so even `a*b + c` — the most ordinary node pair in the
  IR — rounds once where the CPU rounds twice. There is no portable way to forbid it. The bitwise
  tier is therefore the **draws** (integer arithmetic, nothing to contract) and the ULP tier is
  **all f32 lane arithmetic**, measured at ≤1e-6 absolute. Still deterministic per `(seed, device)`
  (same chunk decomposition, chunk-ordered folds, dispatch-size independent), and still far
  stronger than the KS-only tier the original plan accepted — but the line sits elsewhere than
  this plan first drew it.
- **Non-goal:** double-single (two-f32) f64 emulation. ~10× cost to fix a problem the
  standard-error argument says we don't have. Revisit only if a real program disproves that.

## Where it plugs in (and why not the `Runner` seam)

`Runner::next_batch(len) -> &[f64]` is a synchronous pull of 1024 lanes. Both halves are
wrong for GPU: a dispatch wants ≥256k lanes to be worth the ~1–2 ms fixed latency, and
WebGPU readback (`mapAsync`) cannot be synchronous on the JS main thread. So the GPU
backend hooks **one level up, in `reduce::run_reduction`**: if the gate accepts, the
driver hands whole chunk *ranges* to the GPU seam (one dispatch covers many chunks —
lanes are stateless, so chunk boundaries are just arithmetic on the lane index), receives
per-chunk columns/partials, and absorbs them in chunk order. `Reducer` doesn't change;
`Program`/`Runner` don't change; the interpreter/JIT/wasm paths are untouched, and any
graph or device failure falls back exactly like `wasm_host` does (correct-slow, never throw).

**The async bridge (browser).** Evaluation is synchronous and queries force mid-eval, so
`Engine::run` cannot await. Two ways out, and the choice must be **re-made here at G3** — one of the
two arguments that originally settled it is now void:

- **(a) Async evaluator** (PREGPU Track A's A1/A2, unbuilt): the async set, one boxed recursion
  point in `eval`, sync wrappers via `exec::block_on`, and a `sampler::*_async` seam this backend
  routes into. Cost: a 400–800-line mechanical refactor of the eval spine.
- **(b) `Atomics.wait`**: the engine worker blocks while a device-owning agent runs the async
  dispatch. Needs no evaluator changes at all.

**Both of the original arguments for (a) have to be re-checked, and neither survives intact.**

1. ~~"async-first also buys cancellation and progress for free"~~ — **spent.** Cancellation shipped
   in PREGPU Track A *without* any async (native token; browser = terminate the worker). It can no
   longer be counted on async's side of the ledger.
2. ~~"(b) needs cross-origin isolation, which we can't have"~~ — **false, and it was my error.**
   `netlify.toml` sets `Cross-Origin-Opener-Policy: same-origin` +
   `Cross-Origin-Embedder-Policy: require-corp` **site-wide**, so the playground *is* cross-origin
   isolated and `SharedArrayBuffer`/`Atomics.wait` are available on it today. What `pool.ts` refuses
   is *requiring* isolation of everyone who installs `@noiselang/core` — so it feature-detects
   `crossOriginIsolated` and ships two builds (`wasm/` single-threaded, `wasm-mt/` threaded). **The
   package already tiers capability on isolation.** A GPU path that needs isolation would be a third
   item in an existing tier, not a new bridge: isolated pages get it, everyone else falls back to
   today's wasm path — which is exactly the fallback G3 needs anyway.

So the honest trade at G3 is a straight one, with no free lunch on either side:

| | (a) async evaluator | (b) `Atomics.wait` |
|---|---|---|
| core surgery | **400–800-line eval-spine refactor** | **none** |
| GPU on non-isolated hosts | yes | no (falls back to wasm — same tier as threads today) |
| agents | 1 | 2 (the engine worker blocks, so a *second* agent must own the device and run the async dispatch — a blocked worker cannot run its own `mapAsync`) |
| permanent hazards | none | SAB protocol + a deadlock surface |
| also enables | progress reporting; a GPU backend that can simply be `await`ed | — |

**DECIDED at G3 (2026-07-15): (b) `Atomics.wait`, with the device on the MAIN THREAD.** The trade
came down clean once the whole native backend existed:
- **(a)'s core cost is no longer worth its benefit.** G4c just landed a large change to the eval
  spine; a second 400–800-line refactor of the *same* code to make it async — purely to reach a
  browser dispatch — is risk with no native payoff (native stays blocking-poll either way). Its one
  remaining unique win is GPU for embedders who never set COOP/COEP, which is a nice-to-have, not the
  flagship.
- **(b) needs zero evaluator changes**, and the flagship playground is *already* cross-origin
  isolated (`netlify.toml`, site-wide) and *already* loads the `wasm-mt/` build with SAB. So
  `Atomics.wait` is available exactly where the GPU is wanted; everyone else falls back to today's
  wasm path — the same isolation tier the package already ships for threads (`pool.ts`), not a new one.
- **Device on the main thread, not a second worker.** The engine worker blocks on a SAB flag, so it
  cannot run its own `mapAsync`; the main thread is never blocked, so it owns the device and runs the
  async dispatch. This also answers the terminate-cancel cost above — an aborted worker no longer
  throws away the device/pipeline cache, because it never held them; the replacement worker re-attaches
  to the main thread's device with no async re-acquire. One agent owns the device for the page's life.

**The protocol.** The engine worker's wasm hits `nz_gpu_dispatch` mid-forcing (sync, as `nz_kernel_*`
is). The shim writes the shader + params into a `SharedArrayBuffer`, `postMessage`s the main thread,
then `Atomics.wait`s on a done flag. The main thread's handler runs the async WebGPU dispatch
(content-addressed pipeline cache, same LRU/liveness as `nz_kernel_*`), writes the result columns back
into the SAB, and `Atomics.notify`s. The worker wakes, reads the columns, and folds them with the
ordinary reducer — so the *fold* stays in wasm and the answer is bit-identical to native. Local dev
needs a COOP/COEP header shim on `astro dev` (production has them); non-isolated pages never take this
path (feature-detect `crossOriginIsolated` + `navigator.gpu`, else the wasm kernel).

**A cost of the terminate-based cancel, to price at G3.** Browser cancellation kills the worker. If
the worker owns the GPU device and the pipeline cache, an abort throws both away, and the
replacement worker must re-acquire a device (async, and not cheap). Options to weigh then: keep
device ownership on the main thread (worker holds only a proxy), or accept the re-acquire cost on
the abort path (it is a user-initiated stop, so tens of ms is likely fine). Flagged now so it is a
decision, not a surprise.

## Phases

- **G0 — spike. ✅ DONE (2026-07-14).** `tools/gpu-spike`; results above and in
  `tools/gpu-spike/RESULTS.md`. Verdict **GO**; three plan assumptions corrected.
- **G½ — `ArrDraw`: keep the array in the IR. ✅ LANDED (2026-07-14).**

  G0 says the emitter must loop. But it *can't*: `~[n] normal(0,1)` currently builds **n independent
  scalar `Src` nodes** (`eval::draw_lift::draw_shaped` calls `draw_if_recipe` n times), and the array
  exists only as a `Value::Array` at evaluation time. By the time any emitter sees the cone, the
  structure is gone. Recovering it in `wgsl_emit` would mean loop *re-rolling* — proving n scalar
  cones isomorphic — which is a real analysis pass, fragile (a miss silently falls back to the slow
  path), and benefits nobody else.

  So keep it instead. **One new node**, in the shape of the two array-valued sources the IR already
  has (`Permutation { n }`, `Rotation { d }`, both `RvKind::Arr`):

  ```
  RvNode::ArrDraw { n: u32, src: Source }     // RvKind::Arr(n) — n iid draws from one recipe
  RvNode::ArrElem { arr: RvId, k: u32 }       // static element read (NOT the random-index Gather)
  ```

  `~[n] recipe` pushes one `ArrDraw` plus n `ArrElem`s; every downstream vector op keeps building the
  scalar node chain it builds today. The cone goes from *n sources* to **one**, which is the whole
  ballgame — G0 finding 4 says compile tracks `nodes + ~150 × sources`.

  **Each backend picks its own lowering, so nothing regresses:**
    * *interpreter* — pattern-match `ArrElem(ArrDraw, k)` back into a direct scalar fill at
      lowering. Byte-for-byte what it does now; zero change in the hot loop.
    * *jit / wasm* — same: keep unrolling. Their compile costs are 3 orders of magnitude below
      Metal's, so they have nothing to gain and a working fast path to lose.
    * *wgsl* — emit the draw loop into a per-thread `array<f32, n>`. G0 finding 6 measured that this
      does **not** spill: identical dispatch to a fully fused loop at n = 52/100/256, with compile
      down 7–29×.

  **What shipped.** `RvNode::ArrDraw { n, src }` + `RvNode::ArrElem { arr, k }` in `dist.rs`;
  `kernel::source_ordinals` as the one shared ordinal assignment all four backends call; the
  redirection point is `Engine::push_src` (`eval/draw_lift.rs`) — *not* `draw_shaped`, which is why
  every derived recipe shapes for free: `bernoulli` is a `Uniform` under a `<`, `normal_int` a
  `Normal` under a `round`, the hierarchical `*Dyn` family a standard source under an affine map, and
  all of them push their base sources through that one function. `~[3,4]` is a single 12-wide block,
  so a matrix draw is one loop too (turboquant draws three `~[20,20]`: **1200 sources → 3**).
  Measured: `~[52] normal` now puts **one** source node in the graph, not 52
  (`kernel::shaped_tests`), and the example corpus is **3934.7 ms vs a 3938 ms baseline** — flat, as
  intended. Tests: `tests/shaped_draws.rs` (leaves iid; two blocks independent; an element read twice
  is one draw; derived + hierarchical recipes; survives the codegen gate) and the structural counts
  in `kernel::shaped_tests`.

  ❌ **A claim I made here was wrong, and the flat corpus is what caught it.** I wrote that this
  would also fix `turboquant` (22.2 s) and `prisoners` (12.9 s) on the CPU, on the theory that their
  17.6k / 45k-node cones exceed `MAX_CODEGEN_NODES` and drop them to the interpreter. They don't.
  Those two fall to the interpreter because **`rotation` and `permutation` are interpreter-only**
  (`kernel::walk_cost` returns `false` for them outright), and no amount of source collapsing changes
  that — their "17,586 ops/draw" is the *cost-model charge* for Rotation's `2d³`, not a node count.
  So G½ buys the CPU **nothing**, by construction: an `ArrElem` lowers to precisely the fill its
  `Src` lowered to. It is a GPU-enabling change and CPU-neutral, and the corpus staying flat is the
  evidence for both halves of that sentence. (Getting those two demos onto a code generator is a
  real prize, but it is **G4's** — WGSL has array indexing and divergent loops, so it can lower
  `Gather`/`Permutation`/`Rotation` that no CPU backend will.)

  ⚠️ **It breaks the seed, and that has release timing attached.** Source ordinals *were* the `RvId`
  itself (`Inst::Normal { src: id.0, .. }`). An `ArrDraw` is one node needing `n` ordinals for its
  elements, so ordinals moved to a dedicated sequential assignment — which renumbers every source and
  changes every draw stream. Two consequences:
    * This must ship **in the numerics-v2 release cut**, so the new RNG, the f32 lanes and this fold
      into one seed break rather than three. It is now the third unpublished break; there must not be
      a fourth.
    * It is an improvement on its own: keying draws on `RvId` made the draw stream a function of node
      *numbering*, so any new rewrite in `simplify` (a fold, a CSE) silently changed results. Dense
      ordinals depend only on which sources survive, in id order.

- **G1 — emitter + conformance (native). ✅ LANDED (2026-07-14).** `src/wgsl_emit.rs`: post-order
  walk of the simplified cone, memoized `let vN` per node, one column per root, the shared
  **squares64** sources spelled in WGSL, **WGSL built-ins** for the transcendentals, `ArrDraw` as a
  draw loop. `wgpu` is a **native dev-dependency** — the conformance harness *runs* the shaders it
  generates against the interpreter oracle over the shared cross-backend corpus
  (`conformance::{CONST_CASES, RNG_CASES}`), because a shader that merely parses proves nothing.
  Both tiers hold, on device:
    * **Tier 1 — the draws are bit-identical** to the interpreter (4096/4096 lanes, scalar and
      shaped). This is what carries C0's certification onto the GPU.
    * **Tier 2 — lane arithmetic is ULP-close**, ≤1e-6 absolute, over the whole corpus.

  Three things the emitter must NOT borrow from WGSL's same-named built-ins, all found by running the
  corpus rather than by reading the spec:
    * **`round` is ties-AWAY** in this engine, WGSL's is ties-to-even. **`%` is FLOORED** here
      (`-1 mod 3 == 2`), WGSL's is truncated. Both would disagree by a *whole unit* — invisible to a
      ULP bound. Lowered explicitly.
    * **NaN must be screened by BITS.** The vendors enable fast-math by default, which assumes NaNs
      don't exist: on Metal `x == x` folds to `true`. `log(X) == log(X)` is precisely how the language
      asks "is X in the domain", and the fold silently turned a 0.6 into a 1.0. Comparisons now screen
      operands with an integer `nz_isnan` bit test, which has no float identity to exploit.

  ⚠️ **G1 shipped with one scope cut, a correctness cut: explicit `sin`/`cos` were declined.** The
  engine's contract past `approx::TRIG_MAX_F32` is "compute in f64, round to f32" — the 2-term
  Cody–Waite reduction falls apart there, so all three CPU backends hand off to the f64 library
  (finding C3). **WGSL has no f64**, so that fallback cannot be reproduced, and WGSL only *guarantees*
  its `sin`/`cos` on `[-π, π]` anyway: measured, `sin(1e12 · X)` returns **0** on Metal against the
  interpreter's 0.0056. Not a rounding gap — a wrong answer. **Closed by G1b, below.**

- **G1b — exact trig range reduction. ✅ LANDED (2026-07-15).** `nz_sin`/`nz_cos` in the prelude:
  an integer **Payne–Hanek** reduction, then the engine's own f32 kernels and quadrant select. No
  f64, and exact for *every* f32 argument rather than for those under a threshold. Conformant on
  device from 1e3 to 1e30, sin and cos, against the interpreter oracle
  (`payne_hanek_holds_where_the_builtin_gives_up`), and the corpus's `sin_huge`/`cos_huge`/
  `sin_neg_huge` cases — which the harness had been *silently skipping*, since it skips declined
  cones — now actually run.

  **Why `x % 2π` is not the fix**, since it is the first thing anyone reaches for: the quotient
  `x/(π/2)` at `x = 1e12` needs 40 mantissa bits and f32 has 24. The quotient is therefore rounded
  *before* the subtraction, and `x − k·(π/2)` then cancels two nearly-equal large numbers whose
  difference is dominated by that rounding. The reduced argument comes out wrong by ~`x·2⁻²⁴`
  radians — it carries no information at all. Payne–Hanek sidesteps this by never forming the
  quotient in floating point: an f32 is `mant · 2^e`, with `mant` a 24-bit *integer*, so `x · 2/π`
  is an exact integer product against the bits of `2/π` — take a 96-bit window of them (chosen by
  the exponent), keep the result mod 4, and the quadrant and fraction fall out exactly. It is
  reachable here only because the wide-multiply machinery it needs was **already in the prelude**,
  built for `squares64` — WGSL has no `u64` either.

  🔬 **And the finding, which cost the most time: fast-math reassociation silently destroys
  Cody–Waite.** The natural shape is a cheap 2-term reduction below `TRIG_MAX_F32` and the exact one
  above. Written that way, it was off by **1e-5 at x = 1000** — a hundred times its budget, and in
  the *easy* range. Cody–Waite works by evaluating `(x − k·HI) − k·LO` **in that order**, with `HI`
  carrying only 8 mantissa bits so that `k·HI` is exact. The compiler is permitted to reassociate,
  and Metal does — collapsing it to `x − k·(HI + LO)`, which rounds `HI + LO` back to a single f32
  `π/2` and discards precisely the bits the split exists to protect. The residual error is
  `k · ulp(π/2)`, which is what was measured. So the fast path was **deleted**: every argument takes
  the exact integer path, and there is one code path instead of two. This is the same lesson as
  `nz_isnan` (`x == x` folding to `true`), arriving from a different direction — *a float identity
  the optimizer is allowed to "simplify" cannot carry a contract; state it in integers, where there
  is nothing to exploit.*

  Cost: **nothing.** Corpus 3023 ms vs 2999 ms — inside run-to-run noise; the driver dead-strips the
  reduction from the kernels that don't call it. And it corrects a claim made here earlier: this does
  **not** "hand `am_vs_fm` to the GPU". `am_vs_fm` already lowers and already wins 2.0× — its sine
  lives in the *signal generator*, not as a graph `Sin` node. The only example with an explicit `sin`
  is `buffon` (1.3 ms), which the gate declines on cost anyway. **G1b buys no speed at all; it closes
  the backend's one semantic hole**, which is the whole reason to do it.
- **G2 — reduce-driver integration + gate. ✅ LANDED (2026-07-14).** `src/gpu.rs`, behind
  `--features gpu` (native, off by default). Hooks into `reduce::run_reduction`, *not* `Runner`: a
  dispatch wants ≥256k lanes to be worth its ~1.2 ms floor where a `Runner` pulls 1024, so the GPU
  takes the **whole forcing or none of it** — dispatching 1M-lane ranges, folding on the reducer's own
  16,384-sample chunk boundaries in order, and handing back the accumulator. `Program`/`Runner`/
  `Reducer` are untouched. Counter keying is what makes it legal: a chunk is just a lane range.
  Process-wide device + content-addressed pipeline cache (keyed on the shader text). Every failure
  path — no adapter, an unsupported cone, a rejected shader — **declines to the CPU**, so the GPU can
  only ever change speed.

  **The gate's discriminator is the cone size per draw, and that was a surprise.** I expected total
  work; the corpus separates on `ops/draw` with a completely empty band:

  | | ops/draw | GPU vs multicore JIT |
  |---|---|---|
  | `secretary` | 124 | **12.2×** |
  | `barrier_option` | 401 | **4.3×** |
  | `birthday` | 784 | **2.1×** |
  | `am_vs_fm` | 845 | **2.0×** |
  | `noise_colors` | 1,020 | **1.15×** |
  | `st_petersburg` | 58 | 0.99× |
  | `beta_bernoulli` | 37 | 0.99× |
  | `kelly` | 6 | 0.99× |

  Which is the plan's own thesis reached from the other end: a fat cone is a lane's worth of
  independent ALU work — what a GPU is *for* — and it amortizes dispatch + compile over the *cone*
  rather than over the draw count. A thin cone is RNG-and-memory, where a warmed multicore JIT is hard
  to beat and the pipeline compile can never be earned back. `MIN_CONE_OPS = 100` sits inside the
  empty band; `MAX_WGSL_INSTRS = 8000` caps the compile.

  **Corpus: 3935.6 ms → 2999.2 ms (1.31×), no regressions.** `turboquant` and `prisoners` — still the
  two heaviest — remain on the CPU because `rotation`/`permutation` are interpreter-only. They are
  **G4's** prize, and they are now most of the remaining headroom.

  A second-order find, and a load-bearing one: the noise/signal generators (`library.rs`) build an
  `n`-sample realization from `n` white draws and have their *own* draw path, so `ArrDraw` did not
  reach them. `noise_colors`' cone was **256 sources → 39,680 emitted instructions**; the gate
  correctly declined it, so the demo silently never reached the GPU at all. Routing them through
  source *blocks* (white one, brown one, OU two, pink a pair per octave) took it to **1,177
  instructions**, and it — plus `am_vs_fm`, which shares the path — now lower. That is the second time
  the source count, not the node count, turned out to be what mattered.
- **G3 — browser host. ⏸ SPEC'D, NOT STARTED — bridge DECIDED (b, above). Do this in a session with
  a live cross-origin-isolated playground + a Chrome that has WebGPU**, because every piece (the SAB
  protocol, `mapAsync` timing, the main-thread device) is only verifiable in a browser — unlike the
  native backend, which native Rust tests prove. `run_async`/PREGPU A1–A2 are **NOT needed** (that was
  bridge (a); we chose (b)).

  **The shared-memory protocol (engine worker ⇄ main thread).** One `SharedArrayBuffer`, laid out as a
  header of `Int32Array` control words + a byte region:
    - `[0] REQ` — worker sets 1 to ask for a dispatch, main clears to 0 when done.
    - `[1] STATUS` — main writes 0 = ok, -1 = decline/fail before notifying.
    - `[2] SHADER_LEN`, `[3] N`, `[4] K0`, `[5] K1`, `[6] LANE0`, `[7] OUT_LEN`.
    - byte region: the WGSL text (utf-8, `SHADER_LEN` bytes), then the `OUT_LEN × 4` result bytes.

    Worker (in the `nz_gpu_dispatch` shim, called synchronously from wasm mid-forcing): write the
    shader + params into the SAB, `postMessage({type:'gpu'})` to the main thread, then
    `Atomics.wait(ctrl, DONE, 1)`. Main thread's handler: read the request, run the **async** WebGPU
    dispatch (content-addressed pipeline cache keyed on the shader text — same LRU/liveness as
    `nz_kernel_*`), copy the result column into the SAB, set `STATUS`, `Atomics.store(ctrl, DONE, 0)`,
    `Atomics.notify`. Worker wakes, reads the column, folds it with the ordinary reducer. **The fold
    stays in wasm, so the answer is bit-identical to native.** `postMessage`-then-`wait` is safe: the
    message is queued to the main thread's event loop before the worker blocks.

  **Rust side (the verifiable half — mirrors native `gpu::try_reduce`).** A `#[cfg(target_arch =
  "wasm32")]` `try_reduce` sharing the portable pieces (`simplify` → `wgsl_emit::emit` → `cost` →
  `profitable`/`emitted_instrs`) and swapping the dispatch: an `extern "C"` `nz_gpu_available() -> i32`
  (feature-detect: `crossOriginIsolated && navigator.gpu`), `nz_gpu_prepare(wgsl) -> i32` (compile +
  cache the pipeline via the bridge, so a shader the driver rejects declines *before* the fold loop —
  the same "decline, never fail" contract), and `nz_gpu_dispatch(wgsl, out, n, k0, k1, lane0) -> i32`
  filling `out`. Gate it under the SAME `gpu` feature but split the module: native keeps wgpu, wasm
  gets the shim. The reduce hook (`reduce.rs:134`) already fires for `feature = "gpu"`; it just needs
  to be reachable on wasm32.

  **JS side.** `pool.ts`/`worker.ts`: on the isolated (`wasm-mt`) path, the main thread creates the
  SAB, acquires a WebGPU device once (owns it for the page's life), and installs the `{type:'gpu'}`
  message handler; the worker gets the SAB at init. `astro.config.mjs` needs a tiny dev-only Vite
  plugin setting `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
  require-corp` (production already sets them in `netlify.toml`). Feature-detect + silent fallback to
  today's wasm kernel on any non-isolated / no-WebGPU / declined path.

  **Build & test.** `build-wasm.sh` gains a `wasm-mt` variant with `--features wasm-threads,gpu`
  (nightly, already the MT toolchain). Verify in Chrome: a GPU-eligible program (`pi`,
  `barrier_option`, `prisoners`) runs on the GPU and matches the wasm-kernel answer; a declined one
  (Poisson) falls back; abort (terminate the worker) leaves the main-thread device intact so the next
  run re-attaches with no re-acquire.

  **Cancellation** already exists (PREGPU Track A, terminate the worker); with the device on the main
  thread the abort no longer discards it.

  **Research note (2026-07-15) — why the bridge, not "just call WebGPU".** Confirmed the difficulty
  is *not* WebGPU and *not* wgpu-on-wasm (both are easy and fully supported on `wasm32` in every major
  browser, ~85% global). It is the impedance mismatch between Noise's **synchronous, run-to-completion
  forcing** (shared with native, never yields) and WebGPU being **async-only, with no synchronous
  readback**:
    - `mapSync`-on-workers, the one thing that would let sync wasm read a buffer back inline, was
      proposed in **2021 and is still open / unshipped / unowned** (gpuweb/gpuweb#2217). There is no
      synchronous GPU readback path, now or imminently.
    - `Atomics.wait` is banned on the main thread and, in a worker, freezes *that worker's* event
      loop — so a worker that owns the device and blocks on its own `mapAsync` **self-deadlocks** (a
      documented wgpu-on-wasm failure, gfx-rs/wgpu#5279). This is exactly why bridge (b) puts the
      device on a **different agent** (the main thread, event loop live) and only the engine worker
      blocks: a shared `GPUDevice` across agents is explicitly allowed (gpuweb "Multi Explainer").
    - The async-engine alternative (bridge (a) / PREGPU `run_async`) is "normal WebGPU" but needs an
      async fork of the whole sync forcing/reduce stack on wasm; rejected for that reason — bridge (b)
      reuses the already-present cross-origin isolation (`netlify.toml`) + blockable `wasm-mt` worker.
    - **Net:** scope is not inflated, but it is *irreducibly browser-only to verify* (SAB protocol,
      `mapAsync` timing, main-thread device, feature-detect + fallback) — the Rust half mirrors the
      native-tested `gpu::try_reduce`. A future session with live Chrome + WebGPU (browser automation
      available) can build and verify it end-to-end.

  **DECISION (2026-07-15) — keep the engine synchronous; do NOT make forcing async.** Considered
  making the forcing path `async` so it could `await` the GPU the "normal WebGPU" way (bridge (a)).
  Rejected, because in this codebase async is *more* invasive than the sync bridge, not less:
    - Forcing is sync top-to-bottom (`run_to_document → eval → P()/E() → reduce::run_reduction →
      dispatch`) and **shared with native**. A forcing sits anywhere in an expression and its result
      flows back up through the sync tree-walking evaluator, so `async` is viral upward — it infects
      the *entire* evaluator (`eval`/`ops`/`eval_arms`), not just the GPU leaf. That means either
      async-everywhere (native pays for nothing; the sync correctness-reference is lost) or a
      sync/async fork of the evaluator (two copies to keep bit-identical). Both worse than (b).
    - The one clean-ish async variant is **Asyncify** (`wasm-opt --asyncify`): keep Rust sync-looking,
      instrument the module so the dispatch call suspends to the JS event loop. But (1) this is
      `wasm32-unknown-unknown` + wasm-bindgen, not Emscripten, so Asyncify is a bolt-on that
      instruments the call path to the suspending import — i.e. the **Monte-Carlo hot loop**, taxing
      the very throughput the GPU is meant to raise; (2) GPU runs in the `wasm-mt` rayon build, and
      **Asyncify + shared-memory threads is notoriously fragile**; (3) whole-module size/speed cost.
    - Bridge (b) instead **quarantines** the async: one `#[cfg(wasm32)] gpu::try_reduce` shim (mirrors
      the native-tested one; evaluator + native path untouched, lanes bit-identical) and ~100 lines of
      **JS** host where async is native and easy. That is where "WebGPU is easy on the web" actually
      cashes out — on the JS side of the wall. The only thing that would justify async-native forcing
      is a *second* motivation (streaming partial results, cooperative cancellation) — but
      cancellation already works by terminating the worker, so nothing else pulls that way.

  **Status: BUILDING (started 2026-07-15).** Bridge (A) — JS host on the main thread — chosen and
  built: `gpu.rs` split into a backend seam (`available`/`prepare`/`dispatch`), native keeps `wgpu`,
  wasm calls `nz_gpu_*` inline_js shims → `globalThis.__noiseGpu` (installed by `worker.ts`) → SAB +
  `Atomics.wait` → main-thread `gpu-host.ts` (device + pipeline cache + async dispatch). Dev COOP/COEP
  via an astro Vite plugin. Verified live in Chrome (browser automation), and the verification
  immediately earned its keep:

  **FINDING (2026-07-15) — Tint rejects const NaN/Inf; naga accepts it.** The first live run declined
  every GPU forcing: `createComputePipelineAsync` rejected the shader with *"value nan cannot be
  represented as 'f32'"*. WGSL forbids a **const-expression** that evaluates to NaN or ±Inf, and Tint
  (Chrome/Dawn) enforces it — but `naga` (native wgpu) does **not**, so every native GPU test compiled
  the same shader fine and never caught it. This is the concrete proof of why G3 is browser-only
  verifiable. Fix (shared emitter, `wgsl_emit.rs`): route non-finite bit patterns through a runtime
  zero — `bitcast<f32>(BITS | (P.n & 0u))` — so the expression is non-const (Tint accepts) while the
  runtime NaN/Inf is identical on both backends. Touched `NAN_BITS` uses (sin/cos/arr-read inf-screen)
  and the general `f32c` constant emitter. Native GPU conformance tests still bit-identical after.
  Per user (2026-07-15): browser-vs-CPU numeric differences (NaN encoding, sin/cos ULP, f32 precision)
  are acceptable **as long as they're documented** — this one is a compile fix, not a precision change,
  and the two-tier contract already covers the rest.

  **G3 — LANDED + verified live in Chrome (2026-07-15).** End-to-end proof across every code path:
  `secretary`/`birthday` (single-root `cols=1`), `barrier_option`'s `plot::fan` (**joint `cols=52`**,
  4M f32 read back), `turboquant` (rotation, 163 KB shaders) — all compile, dispatch through the SAB
  bridge, and return correct results (secretary 38–39% ≈ 1/e). No CPU fallback on accepted forcings
  (profile shows `gpu.dispatch`/`gpu.fold`, no `reduce`); the gate still correctly declines thin cones
  (`kelly` → cpu, "gate: DECLINE — cone too thin"). Verified `crossOriginIsolated` via the dev Vite
  plugin, and that a WebGPU device is acquired on the main thread.

  **Diagnostics (this session).** `result.profile` (Rust: phase timings + gate notes) is now joined by
  `result.diagnostics` (JS: `crossOriginIsolated`, `threaded`, `workers`, and per-run GPU-host facts —
  `dispatches`/`lanes`/`shaderFailures`/`lastShaderError`). The playground shows a compact headline
  (`time · samples · backend`) with the full breakdown — throughput, phases, execution, engine notes —
  in a hover card. Two gotchas cost real time and are worth remembering: **(1)** Web Workers don't
  hot-reload — after editing `worker.ts`/`gpu-host.ts` you must restart `astro dev` or you get stale
  worker code (looked like an intermittent GPU deadlock). **(2)** Astro scoped `<style>` does NOT reach
  JS-created (innerHTML) nodes — runtime-built chips/tooltip must go in `<style is:global>`.
- **G4a — `Permutation` / `Gather` / `ArrIndex`, bit-identical. ✅ LANDED (2026-07-15).** All three
  are integer draws (Fisher–Yates swaps, Lemire bounded draws, clamped index reads), so they lower
  **bit-identically** to the interpreter — verified on device. `kernel::cell_stream_ordinals` is the
  new shared id-order assignment of `CellStream` ordinals (replacing a DFS-emit-order counter, the
  last traversal-dependent numbering — same brittleness `source_ordinals` removed; folds into the
  numerics-v2 cut). `wgsl_emit` gains `cell_bits48`/`bounded48` in the prelude and a Fisher–Yates
  fill; `ArrIndex`/`Gather` share one clamped, NaN-screened read matching `Inst::ArrIndex`.

  **Corpus-neutral, and the finding is why.** `prisoners` lowers now but the gate correctly declines
  it: its 100×50 cycle-following unrolls to ~15,000 *data-dependent* `ArrIndex` reads — one giant
  dependent basic block, on which the Metal compiler goes **super-linear**: 2.2 s cold, vs 127 ms for
  a 12k-instruction shader with ordinary parallelism. (A forced run showed "51 ms" only because the
  timed pass hit the process-wide pipeline cache the untimed warm-up filled.) Moving `prisoners` needs
  its cycle loop **re-rolled** into WGSL control flow — the loops are gone by the time the emitter
  sees the DAG, the same problem `ArrDraw` solved for shaped draws — which is **G4c**, below.

- **G4b — `Rotation` (turboquant). ✅ LANDED (2026-07-15). 1294 ms → 68 ms (19×).** Unlike
  `prisoners`, a rotation's work is a *bounded loop* (d² normals + O(d³) Gram–Schmidt), so it emits a
  **small** shader that compiles cheap — the gate takes turboquant naturally and the GPU crushes it.
  **Corpus 3022 ms → 1812 ms**, i.e. **3935 ms → 1812 ms = 2.17×** over the original CPU baseline.

  ⚠️ **Rotation is the one draw that is distribution-identical, not lane-identical — and that is a
  measured fact, not a concession.** The interpreter's MGS used to run in f64 scratch (~1e-7
  orthonormality); WGSL has no f64. The CPU rotation is now **also pure f32** (matching the GPU op for
  op, keeping the lane type f32 everywhere rather than forking precision per backend; the f64
  reference is in git history if ever wanted as an opt-in mode). Even so, **the two f32 rotations
  diverge lane-for-lane by ~2.6e-2 on the worst lane**: MGS occasionally lands on a near-singular
  Gaussian matrix where it is ill-conditioned, and there it turns the `ln`/`sin`/`cos` ULP gap
  (built-ins vs `approx`) and FMA into a finite rotation of the output basis. So rotation *cannot* be
  lane-for-lane on any two independent float implementations. What survives is what matters:
  turboquant's b=1 distortion is **0.347221 on both**, identical to six digits, because any valid Haar
  rotation gives the same distributional answer. f32 also loosens orthonormality to ~2e-5 typical with
  a heavy tail to ~1e-3 on the near-singular lanes — harmless for a Monte Carlo expectation, and the
  interpreter test now checks the RMS residual plus a generous cap rather than a tail-sensitive max.

- **G4c — re-roll `prisoners`. ✅ LANDED (2026-07-15). 851 ms → 2.8 ms (304×).** A `for` loop is
  captured as a single `RvNode::Scan` at eval time instead of unrolling, so the cycle-following
  becomes one WGSL loop rather than ~15k dependent reads — turning a 2.2 s pathological shader into a
  handful of statements the gate takes naturally. **Corpus 1812 ms → 873 ms, i.e. 3935 ms → 873 ms =
  4.5× over the original CPU baseline.**

  The mechanism that made it tractable: a loop-carried variable's placeholder is just an ordinary
  `Value::Dist` RV node, so binding every carried var (and, for a `0..n` iterator, the index) to a
  fresh `RvNode::Placeholder` and evaluating the body **once** reads the recurrence back directly.
  Carried slots are found by an AST walk of the body's assignments; nested loops fall out of
  recursion. Capture is best-effort with a clean fallback to unrolling for anything outside v1 — a
  draw inside the loop, a non-scalar accumulator, the index over a non-`0..n` iterator, or an
  emission (a `Print` is a per-iteration side effect one pass can't reproduce).

  Two lowerings from the one node: `simplify::unroll_scans` expands it to the flat DAG for the CPU
  **before** simplify, so the interpreter's answer and draw stream are byte-for-byte unchanged
  (proved by `tests/loops_g4c.rs`: capture-on == capture-off bit-for-bit, over a pointer-chase, an
  index-using accumulator, and the full doubly-nested prisoners); `wgsl_emit::emit_scan` rolls it
  into a `for` loop for the GPU, hoisting loop-invariant sources (the permutation is drawn once) and
  binding placeholders to the loop `var`s. The one rule throughout: an invariant is built/drawn once
  and shared; an index/carried-dependent node is rebuilt per iteration.

  Still open (a smaller G4d if wanted): Knuth `Poisson` as a legal divergent loop, GPU-side moments
  reduction, joint (multi-root) kernels for the introspection drivers, and draws *inside* a rolled
  loop (v1 falls back to unrolling those).

## Would it be worth it? The numbers

Shapes and warmed native times (M-series, `--features jit`, multicore) from
`example_times` / `example_shapes`, 2026-07-13:

| demo | native today | forcings × draws | ops/draw | why it's slow | GPU estimate |
|---|---|---|---|---|---|
| `turboquant` | 22.2 s | 10 × 10k | 17,586 | cone huge + 10k draws < JIT gate → interpreter | 0.5–2 s (**10–40×**) — compute is ~ms; cost is 10 big pipeline compiles (G0 risk) |
| `prisoners` | 12.9 s | 1 × 14k | 44,797 | `Gather` + 45k-node cone → interpreter-only everywhere | G4 + big-shader risk: **5–30×** if it compiles; the honest fix is also algorithmic (Fisher–Yates lowering) |
| `barrier_option` | 8.7 s | 5 × 175k | ~401 | 228M normal draws (path model) | **20–80×** — many lanes, moderate cone, hardware transcendentals: the ideal GPU shape |
| `am_vs_fm` | 3.5 s | 5 × 40k | 3,463 | sin-heavy signal kernel | **20–50×** — hardware `sin` |
| `bootstrap`, `beta_bernoulli`, `st_petersburg`, `secretary` | 0.2–0.5 s | ~1M draws | 30–60 | volume | 2–10× where gated in; borderline |
| `pi`, `dice`, `buffon`, … | ≤ 10 ms | ~1M | < 10 | not slow | gate declines — dispatch latency would *lose* |

**G0 reality-check on that table.** The per-draw throughput estimates hold — the spike measured a
barrier-shaped kernel at 108–169 M samples/s and beat the multicore JIT **19× on dispatch** for an
identical 100M-normal-draw kernel. What the table under-weighted is the **compile** column, and for
the two big-cone demos it is decisive:

* `turboquant` (17.6k ops/draw, ten forcings) would pay **~1.9 s of cold compile per forcing** if
  unrolled — ~19 s, which erases the entire win against today's 22.2 s. Its speedup is therefore
  **entirely contingent on loop-form emission** (G1), not on throughput. Same for `prisoners` at
  45k (8.9 s cold, unrolled).
* `barrier_option` and `am_vs_fm` (400 / 3.5k ops per draw) compile in single-digit ms and are
  unaffected. They are the safe wins, and they were always the ones the plan bet on.

Corpus total: ~52 s → an estimated **3–6 s**, with the four heavy demos supplying nearly
all of the win — but now with the caveat that two of the four are gated on G1 emitting loops rather
than on any GPU property. And these are the *native JIT multicore* baselines — **the playground is
the real target**: the browser has no Cranelift, threads are opt-in, and the emitted-wasm
path is worth 2–7× over the wasm interpreter. The same four demos in the browser today sit
at "go get a coffee"; WebGPU puts them at interactive latency, on the one surface (the
public playground) where perceived speed matters most. That asymmetry is the verdict:

- **Browser: yes.** This is where the 50–100×-class wins live, and the demo surface is
  the product.
- **Native CLI: nice-to-have** behind the `gpu` feature. Multicore JIT is already fine
  for everything except `turboquant`/`prisoners`; native `wgpu` earns its keep mainly as
  the conformance harness.

**Cost:** G1–G3 ≈ 2–4 focused weeks (emitter ~1–1.5k lines mirroring `wasm_emit`, RNG +
conformance ~0.5k, host shim + worker bridge is the gnarly part). G0 is 1–2 days and
answers the two questions that could sink it before any of that is spent.

## Risks

| risk | assessment |
|---|---|
| **Compile cost** (the real one — G0 promoted it from "an unknown" to "the binding constraint") | **Measured**: it tracks *emitted instructions*, not nodes — every RNG source inlines ~150 ops of emulated hash (~6.5 ms each), and arithmetic is superlinear (5k stmts → 325 ms; 17.6k → 1.9 s; 45k → 8.9 s, all cold). Mitigations, in order of leverage: **(a) loop-form emission for array draws** (19×, measured — G1), (b) an emitted-instruction cap + interpreter fallback, (c) the pipeline cache (Metal caches on disk; a repeat visitor pays ~1/3). Naga is ~25% of it and is never cached |
| Giant shaders: register spills to private memory | Not observed at 45k statements — they compile and run. Subsumed by the compile-cost row above |
| f32 changes published demo numbers | Moved upstream: PLAN-PREGPU B re-baselined the corpus once, before any GPU existed — this plan inherits already-f32 numbers |
| Cross-device reproducibility (vendor `÷`/`sqrt`/transcendental precision) | Accepted and documented: deterministic per `(seed, device)`, ≤1e-6 abs vs CPU. **The escape hatch this row used to name does not exist** — "emit the shared `approx.rs` f32 polynomials for bitwise portability" was falsified in G0: the GPU contracts multiply-adds, so the polynomial is no more bit-faithful than the built-in (and no more accurate), at 1.4× the cost. Bitwise parity stops at the draws, and that is where the certification lives |
| WebGPU availability (Safari shipped 2025, Firefox partial) | Feature-detect; gate declines → today's wasm path. Nothing regresses, ever |
| Device loss / driver reset mid-run | Same story as `nz_kernel_*` eviction (finding C5): status-return, reseed, degrade to CPU for the rest of the run |
| Async bridge deadlocks (Atomics.wait on a worker that owns nothing) | Device lives on a *different* agent than the engine worker by construction; timeout on the wait falls back to CPU |
| Readback bandwidth | Non-issue: 4 MB per 1M f32 samples, and G4's partial-sum reduction shrinks it 100× for `P`/`E`/`Var` |
