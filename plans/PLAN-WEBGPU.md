# PLAN-WEBGPU — the GPU as a fourth lowering of the RvGraph

**Date:** 2026-07-13 (G0 landed 2026-07-14) · **Status: G0 DONE — verdict GO.** G1 next.
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

**Undecided, deliberately.** (b) is a real contender now — it needs *zero* changes to the
evaluator, and the flagship deployment already has the isolation it wants. (a) buys a cleaner
long-term architecture and GPU for embedders who never set COOP/COEP. Decide at G3 with G0's cost
data in hand, not here. (Native: `wgpu` + blocking poll, trivially sync — which is also our test
harness either way, under both options.)

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

  ⚠️ **G1's one scope cut, and it is a correctness cut: explicit `sin`/`cos` are declined.** The
  engine's contract past `approx::TRIG_MAX_F32` is "compute in f64, round to f32" — the 2-term
  Cody–Waite reduction falls apart there, so all three CPU backends hand off to the f64 library
  (finding C3). **WGSL has no f64**, so that fallback cannot be reproduced, and WGSL only *guarantees*
  its `sin`/`cos` on `[-π, π]` anyway: measured, `sin(1e12 · X)` returns **0** on Metal against the
  interpreter's 0.0056. Not a rounding gap — a wrong answer. The cone is declined (falling back to a
  correct backend, exactly as `wasm_host` does) and the decline is *tested*, so it cannot rot into a
  silent divergence. **This does not touch the normal draw**: Box–Muller's trig lives inside the
  prelude's `src_normal` with `θ ∈ [0, 2π)`, so `barrier_option` and every Gaussian model still lower.
  It does, for now, cost us `am_vs_fm`.

- **G1b — exact trig range reduction (new).** Close the cut above with an integer **Payne–Hanek**
  reduction in the shader (a fixed-point multiply of the f32 mantissa against a multi-word `2/π`),
  which needs no f64 and is exact for every f32 argument. That restores one numeric contract across
  all four backends and hands `am_vs_fm` — the trig-heavy demo, and one of the four the plan's win is
  built on — to the GPU.
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
- **G3 — browser host.** `nz_gpu_*` inline-JS shim (device ownership, content-addressed
  pipeline cache with the same LRU/liveness story as `nz_kernel_*`), the async engine's
  `run_async` path (**this is where PREGPU A1–A2 gets built — G3 is their only consumer**),
  playground wiring, feature-detect + silent fallback to today's wasm path. Cancellation already
  exists (PREGPU Track A): the native `CancelToken` gives "check before each dispatch submission and
  abandon in-flight `mapAsync` readbacks", and the browser's `AbortSignal` already stops a run by
  terminating the worker — see the device-ownership cost noted in the async-bridge section above.
- **G4 — exceed the CPU codegen.** `Gather` is plain array indexing in WGSL — this
  unlocks `prisoners`, `empirical`, `block_bootstrap`, permutation programs that no CPU
  codegen path will ever take. Knuth `Poisson` is a legal (divergent) loop bounded by
  `POISSON_KNUTH_MAX`. GPU-side moments reduction. Joint (multi-root) kernels for the
  introspection drivers.

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
