# PLAN-WEBGPU — the GPU as a fourth lowering of the RvGraph

**Date:** 2026-07-13 (deps updated 2026-07-14) · **Status:** proposal (nothing started). G0 spike
gates everything else. **Depends on PLAN-PREGPU**, which moves every cross-backend decision out of
this plan — the GPU then lands as just another backend under one shared contract.

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

- **No RNG state chain — but one RNG decision deferred to G0.** PLAN-PREGPU Track C
  moves every backend to a counter-based keyed generator before this plan starts, and
  the emitter spells the identical hash in WGSL for **bit-identical** draws. The C0
  certification runs (2026-07-14, tools/rng-cert) forced a trade this plan must own:
  every GPU-cheap u32-native hash tested (pcg4d, pcg4d-3r, pcg4d-3rf) shows real
  sequential structure by 256 GB of PractRand, while the certified survivors (Squares,
  Philox) are built on wide multiplies WGSL must emulate (no `u64`, no `mulhi`) at
  **~70–90 ALU ops per uniform vs ~10** for the pcg family — enough to dominate a
  normal-dense kernel whose transcendentals are hardware builtins. **G0 therefore also
  measures a Squares-in-WGSL kernel on a real shape.** If RNG ALU is material there, the
  owner chooses explicitly: accept the compressed (still large) GPU win; give the GPU a
  local generator under statistical-only conformance (this plan's original stance); or
  re-freeze C0's PractRand depth with a measured bound and re-admit a cheap hash. No
  state upload, no streams either way; each source node's offset is a compile-time
  constant, so every uniform in the kernel is an independent hash.
- **No transcendental inlining.** WGSL has native `log`/`exp`/`sin`/`cos`/`atan`/`pow`.
  The entire `approx.rs` polynomial apparatus (built because `normal` costs ln+sincos per
  draw on CPU) is unnecessary: Box–Muller is four built-ins. The ops the CPU cost model
  charges as expensive libcalls are the GPU's *cheapest* — which is precisely what makes
  the transcendental-heavy demos (`am_vs_fm`, `barrier_option`) the biggest winners.
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
- **Residual determinism gap, GPU only.** CPU backends (and the draws everywhere) are
  bit-identical per seed. On GPU, WGSL guarantees correctly-rounded `+ − ×` but allows
  ≤2.5 ULP on `÷`/`sqrt` and vendor-specified transcendentals — so GPU *lanes* are
  bit-identical to CPU for add/mul/select graphs and tight-ULP elsewhere, deterministic
  per `(seed, device)` always (same chunk decomposition, chunk-ordered folds, dispatch-
  size independent). Conformance asserts bitwise where the spec allows it, ULP/statistical
  bounds where it doesn't — far stronger than the KS-only tier the original plan accepted.
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

- **G0 — spike (1–2 days, kills or scales the plan).** Hand-write WGSL for two kernels in
  a scratch page: `pi` (trivial) and a turboquant-scale one (~17k generated statements).
  Measure: pipeline-compile time vs statement count (Tint/Naga/Metal), samples/s,
  dispatch+readback latency, whether a 40k-statement shader compiles at all — **and the
  emulated-Squares RNG share on a normal-dense kernel (the deferred RNG decision above:
  it only binds if this number is material)**. These are the only real unknowns;
  everything else in this plan is known-shape work.
- **G1 — emitter + conformance (native).** `wgsl_emit.rs`: post-order walk of the
  simplified cone, memoized `let vN: f32` per node, the shared pcg4d-3r sources spelled in
  WGSL, scope = the CPU-codegen subset (no `Poisson`, no `Gather`). `wgpu` as a
  dev-dependency; parity tests vs the interpreter — bitwise for the draws and the
  add/mul/select subset, ULP/statistical elsewhere.
- **G2 — reduce-driver integration + gate (native, `gpu` feature).** Chunk-range
  dispatch, chunk-ordered fold, `MIN_WORK_GPU` measured the way `MIN_DRAWS_WASM` was
  (bench the corpus, find where the fixed costs pay back), node-count cap from G0 data.
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

Corpus total: ~52 s → an estimated **3–6 s**, with the four heavy demos supplying nearly
all of the win. And these are the *native JIT multicore* baselines — **the playground is
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
| Giant shaders (17k–45k statements): compile time, register spills to private memory | **The** unknown. G0 measures it; mitigation = per-backend node cap (an analog of `MAX_CODEGEN_NODES`, likely lower) + interpreter fallback |
| f32 changes published demo numbers | Moved upstream: PLAN-PREGPU B re-baselines the corpus once, before any GPU exists — this plan inherits already-f32 numbers |
| Cross-device reproducibility (vendor `÷`/`sqrt`/transcendental precision) | Accepted, documented: deterministic per `(seed, device)`, tight-ULP vs CPU. If exact cross-vendor parity ever matters, emit the shared `approx.rs` f32 polynomials (PLAN-PREGPU B3) instead of WGSL built-ins — trading some of the hardware-transcendental win for bitwise portability |
| WebGPU availability (Safari shipped 2025, Firefox partial) | Feature-detect; gate declines → today's wasm path. Nothing regresses, ever |
| Device loss / driver reset mid-run | Same story as `nz_kernel_*` eviction (finding C5): status-return, reseed, degrade to CPU for the rest of the run |
| Async bridge deadlocks (Atomics.wait on a worker that owns nothing) | Device lives on a *different* agent than the engine worker by construction; timeout on the wait falls back to CPU |
| Readback bandwidth | Non-issue: 4 MB per 1M f32 samples, and G4's partial-sum reduction shrinks it 100× for `P`/`E`/`Var` |
