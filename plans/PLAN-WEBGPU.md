# PLAN-WEBGPU — the GPU as a fourth lowering of the RvGraph

**Date:** 2026-07-13 · **Status:** proposal (nothing started). G0 spike gates everything else.

## The thesis

A Noise query is already shader-shaped: one pure per-lane kernel (`RvGraph` cone), run N
independent times, folded by a commutative monoid (`Reducer`). That is *exactly* the WebGPU
compute model — lanes are invocations, the fold is a reduction. We already have three
lowerings of the same IR (bytecode interpreter, Cranelift JIT, WASM emitter); WGSL is a
fourth, and structurally the easiest one:

- **No RNG state chain.** The whole multi-stream apparatus in `kernel.rs` exists because
  xoshiro is a serial dependency the OoO core must overlap. On GPU we switch to a
  **counter-based RNG**: each lane hashes `(chunk_seed, lane, source_offset)` statelessly
  — the same trick `chunk_seed()` already plays at chunk granularity (it *is* SplitMix64
  as a counter hash), pushed down to lane level. Each source node gets a compile-time
  constant offset, so every uniform in the kernel is an independent hash. No state
  upload, no readback, no streams, no latency-vs-throughput policy — and no choice,
  either: WGSL has no `u64`, so xoshiro256++ can't even be expressed on this target.
  Generator: **pcg4d** (Jarzynski–Olano 2020; pure u32 ops, 4 uniforms per hash — one
  Box–Muller pair per call), validated by G1's statistical battery, with Philox4x32-10
  (BigCrush-certified, but needs a 16-bit-split emulated mulhi in WGSL) as the fallback
  if pcg4d shows bias.
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

## What "32-bit mode" actually costs (the f64 question)

WGSL has no `f64` and no timeline for one. So the GPU backend computes lanes in `f32`:

- **Per-sample noise is fine.** Monte-Carlo standard error is `O(1/√N)`; f32 rounding is
  ~1e-7 relative. For any N a demo runs, sampling noise dwarfs rounding by orders of
  magnitude. `P`/`E` estimates move *within their own confidence interval*.
- **Accumulation is not fine — keep it off the GPU or stage it.** Summing 1e6 f32s
  naively loses digits. Phase 1: read raw f32 samples back, widen to f64, feed the
  existing `Reducer::absorb` (unchanged fold). Phase 2 (optional): per-workgroup partial
  `{count, Σx, Σx²}` over ≤4096 lanes in f32 (safe at that size), CPU folds partials in
  f64 — shrinks a 1M-sample readback from 4 MB to ~50 KB.
- **`unif_int` ranges above 2²⁴** don't fit an f32 mantissa. Draw and clamp in `u32`
  (WGSL integer ops), convert at the end; ranges beyond 2³² decline to CPU.
- **The NaN conditioning sentinel survives** — f32 has NaN; `select(C, q, NaN)` lowers as-is.
- **Determinism contract changes tier.** Today: bit-identical for `(seed, n)` across
  thread counts. GPU: *deterministic per `(seed, device)`*, statistically equivalent to
  CPU, but not bit-equal to it (f32 + different RNG) nor across GPU vendors (WGSL
  transcendental precision is implementation-defined). We reuse the exact
  `chunk_seed(seed, chunk)` decomposition — lane's RNG key is
  `(chunk_seed, lane_in_chunk)` — and fold partials in chunk order, so within one device
  the answer never depends on dispatch size. Document the tier; conformance tests assert
  statistical parity (KS tests + moment tolerances, mirroring `jit`'s parity suite), not bits.
- **Non-goal:** double-single (two-f32) f64 emulation. ~10× cost to fix a problem the
  standard-error argument says we don't have. Revisit only if a real program disproves that.
- **Non-goal: switching the CPU backends to the counter RNG.** Measured (scratch bench,
  M-series, single thread): SplitMix64-as-counter 229 M u64/s vs xoshiro256++×4-stream
  234 M u64/s — a wash, because independent counters expose the same ILP the 4-stream
  interleave was built to expose. Unification wouldn't buy cross-backend bit-parity
  anyway (f32 breaks it regardless), and it would invalidate every seeded output (the
  `rng.rs` known-answer test, committed docs) plus force retuning `MIN_DRAWS_*`.
  Worthwhile *separable* follow-up: a counter RNG on CPU would let us delete the whole
  multi-stream layer (`STREAMS`, `seed_state`, `choose_streams`, `latency_bound`, the
  stream-strided emitter layout) — try it after WebGPU ships, keep only if
  neutral-or-better on the corpus.

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
`Engine::run` cannot await — this must change before G3. **Chosen (revised 2026-07-13):
make the evaluator async — see PLAN-ASYNC.md** for the full migration (the async set, the
one boxed recursion point in `eval`, sync wrappers via `exec::block_on`, and the
`sampler::*_async` seam this backend routes into). The originally-sketched alternative —
engine in a dedicated Web Worker blocking on `Atomics.wait` while the device-owning agent
runs the async dispatch — still works and needs no evaluator changes, but it adds a second
agent, a SAB protocol, and a deadlock surface permanently; async-first also buys
cancellation and progress for free. (Native: `wgpu` + blocking poll, trivially sync —
which is also our test harness either way.)

## Phases

- **G0 — spike (1–2 days, kills or scales the plan).** Hand-write WGSL for two kernels in
  a scratch page: `pi` (trivial) and a turboquant-scale one (~17k generated statements).
  Measure: pipeline-compile time vs statement count (Tint/Naga/Metal), samples/s,
  dispatch+readback latency, and whether a 40k-statement shader compiles at all. These
  are the only real unknowns; everything else in this plan is known-shape work.
- **G1 — emitter + conformance (native).** `wgsl_emit.rs`: post-order walk of the
  simplified cone, memoized `let vN: f32` per node, Philox/pcg4d sources, scope = the
  CPU-codegen subset (no `Poisson`, no `Gather`). `wgpu` as a dev-dependency; statistical
  parity tests vs the interpreter for every `Source` and op.
- **G2 — reduce-driver integration + gate (native, `gpu` feature).** Chunk-range
  dispatch, chunk-ordered fold, `MIN_WORK_GPU` measured the way `MIN_DRAWS_WASM` was
  (bench the corpus, find where the fixed costs pay back), node-count cap from G0 data.
- **G3 — browser host.** `nz_gpu_*` inline-JS shim (device ownership, content-addressed
  pipeline cache with the same LRU/liveness story as `nz_kernel_*`), the async engine's
  `run_async` path (prerequisite: PLAN-ASYNC A1–A2), playground wiring, feature-detect +
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
| f32 changes published demo numbers | Estimates move within their own MC confidence interval; conformance suite quantifies per-example drift before shipping |
| Cross-device reproducibility (vendor transcendental precision) | Accepted, documented tier: deterministic per `(seed, device)`. If it ever matters, emit our own `approx.rs` polynomials instead of built-ins |
| WebGPU availability (Safari shipped 2025, Firefox partial) | Feature-detect; gate declines → today's wasm path. Nothing regresses, ever |
| Device loss / driver reset mid-run | Same story as `nz_kernel_*` eviction (finding C5): status-return, reseed, degrade to CPU for the rest of the run |
| Async bridge deadlocks (Atomics.wait on a worker that owns nothing) | Device lives on a *different* agent than the engine worker by construction; timeout on the wait falls back to CPU |
| Readback bandwidth | Non-issue: 4 MB per 1M f32 samples, and G4's partial-sum reduction shrinks it 100× for `P`/`E`/`Var` |
