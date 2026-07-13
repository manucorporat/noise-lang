# PLAN-WEBGPU — the GPU as a fourth lowering of the RvGraph

**Date:** 2026-07-13 · **Status:** proposal (nothing started). G0 spike gates everything
else. **Depends on PLAN-PREGPU** (async engine, f32 lanes in all modes, the pcg4d
counter RNG in all modes), which moves every cross-backend decision out of this plan —
the GPU then lands as just another backend under one shared contract.

## The thesis

A Noise query is already shader-shaped: one pure per-lane kernel (`RvGraph` cone), run N
independent times, folded by a commutative monoid (`Reducer`). That is *exactly* the WebGPU
compute model — lanes are invocations, the fold is a reduction. We already have three
lowerings of the same IR (bytecode interpreter, Cranelift JIT, WASM emitter); WGSL is a
fourth, and structurally the easiest one:

- **No RNG state chain — and no RNG *decision* left.** PLAN-PREGPU Track C moves every
  backend to the counter-based **pcg4d**, keyed
  `(key_lo, key_hi, global_lane, source_offset)`, before this plan starts. The WGSL
  emitter just spells the identical hash in WGSL — pcg4d is pure u32 ops precisely so it
  can be, ~5 ALU ops per uniform — so the GPU's draws are **bit-identical** to the CPU
  backends', not merely equidistributed. No state upload, no streams, no
  latency-vs-throughput policy; each source node's offset is a compile-time constant, so
  every uniform in the kernel is an independent hash. (xoshiro couldn't have come along
  anyway: WGSL has no `u64`.)
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
pcg4d draws. What remains GPU-specific:

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
`Engine::run` cannot await — **resolved upstream: PLAN-PREGPU Track A makes the evaluator
async** (the async set, the one boxed recursion point in `eval`, sync wrappers via
`exec::block_on`, and the `sampler::*_async` seam this backend routes into). The
originally-sketched alternative — engine in a dedicated Web Worker blocking on
`Atomics.wait` while the device-owning agent runs the async dispatch — still works and
needs no evaluator changes, but it adds a second agent, a SAB protocol, and a deadlock
surface permanently; async-first also buys cancellation and progress for free. (Native:
`wgpu` + blocking poll, trivially sync — which is also our test harness either way.)

## Phases

- **G0 — spike (1–2 days, kills or scales the plan).** Hand-write WGSL for two kernels in
  a scratch page: `pi` (trivial) and a turboquant-scale one (~17k generated statements).
  Measure: pipeline-compile time vs statement count (Tint/Naga/Metal), samples/s,
  dispatch+readback latency, and whether a 40k-statement shader compiles at all. These
  are the only real unknowns; everything else in this plan is known-shape work.
- **G1 — emitter + conformance (native).** `wgsl_emit.rs`: post-order walk of the
  simplified cone, memoized `let vN: f32` per node, the shared pcg4d sources spelled in
  WGSL, scope = the CPU-codegen subset (no `Poisson`, no `Gather`). `wgpu` as a
  dev-dependency; parity tests vs the interpreter — bitwise for the draws and the
  add/mul/select subset, ULP/statistical elsewhere.
- **G2 — reduce-driver integration + gate (native, `gpu` feature).** Chunk-range
  dispatch, chunk-ordered fold, `MIN_WORK_GPU` measured the way `MIN_DRAWS_WASM` was
  (bench the corpus, find where the fixed costs pay back), node-count cap from G0 data.
- **G3 — browser host.** `nz_gpu_*` inline-JS shim (device ownership, content-addressed
  pipeline cache with the same LRU/liveness story as `nz_kernel_*`), the async engine's
  `run_async` path (prerequisite: PLAN-PREGPU A1–A2), playground wiring, feature-detect +
  silent fallback to today's wasm path.
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
