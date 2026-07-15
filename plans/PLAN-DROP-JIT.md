# PLAN-DROP-JIT — retire Cranelift, make the GPU the performance backend everywhere

**Date:** 2026-07-15 · **Status: DRAFT — measurements done, work not started.**
**Depends on PLAN-WEBGPU** (G0–G4c landed; the native GPU backend works and wins 4.17× on the
corpus). **Feeds** PLAN-WEBGPU G3 (browser host): every hour not spent maintaining a
native-only backend is an hour spent on the one that reaches both targets.

## The thesis

The JIT was built when it was the only escape from the interpreter. It no longer is. Measured on
the full example corpus (M4 Pro, `run_to_document`, fresh engine per rep, median of 5 —
harness: `crates/noise-core/examples/bench_table.rs`):

| config | corpus total | vs interp |
|---|---|---|
| interpreter | 3854 ms | 1.00× |
| `--features jit` | 3274 ms | 1.18× |
| `--features gpu` | 923 ms | **4.17×** |
| `--features jit,gpu` | 843 ms | 4.57× |

Three facts make the decision:

1. **The JIT's marginal value once the GPU exists is 80 ms — 9% — and it is native-only.** It
   never reaches the browser (the browser's codegen backend is `wasm_emit`, a separate emitter
   that stays). Every hour of JIT maintenance buys speed on exactly one target.
2. **The shipped CLI never had it.** `noise-cli` depends on `noise-core` with **no features** —
   the released binary is interpreter-only today. Nobody loses what nobody ships. Meanwhile,
   shipping `gpu` as the CLI default takes real users **3854 → 923 ms (4.2×)** — dropping the
   JIT and shipping the GPU is a *speedup* for every user we actually have.
3. **On the programs that matter, the JIT does nothing.** `turboquant` 0.99×, `prisoners` 1.00×,
   `am_vs_fm` 1.00×, `noise_colors` 1.01× — the heavy programs are transcendental-, rotation-,
   and loop-bound, which CPU codegen cannot touch and the GPU eats.

### The full table (per-example, ms, median of 5)

| example | interp | jit | gpu | jit+gpu | jit/interp | gpu/interp |
|---|--:|--:|--:|--:|--:|--:|
| noise_colors | 309.9 | 307.6 | 220.9 | 219.6 | 1.01× | 1.40× |
| am_vs_fm | 418.8 | 418.9 | 210.6 | 210.8 | 1.00× | 1.99× |
| barrier_option | 1181.9 | 786.9 | 181.2 | 180.6 | 1.50× | 6.52× |
| beta_bernoulli | 60.8 | 21.1 | 61.4 | 21.2 | **2.88×** | 0.99× |
| turboquant | 1473.8 | 1482.4 | 55.5 | 56.8 | 0.99× | **26.5×** |
| st_petersburg | 47.0 | 32.0 | 32.4 | 25.6 | 1.47× | 1.45× |
| bootstrap | 26.7 | 23.0 | 25.6 | 23.4 | 1.16× | 1.04× |
| clt_normal | 13.4 | 11.8 | 13.7 | 12.1 | 1.13× | 0.97× |
| qjl_scalar | 12.3 | 8.5 | 12.8 | 8.0 | 1.45× | 0.96× |
| kelly | 12.1 | 11.6 | 12.3 | 11.9 | 1.04× | 0.98× |
| birthday | 125.5 | 23.9 | 11.2 | 4.6 | 5.25× | 11.2× |
| dithering | 10.4 | 5.9 | 10.5 | 6.0 | 1.76× | 0.99× |
| conditional_bayes | 9.7 | 7.5 | 10.3 | 7.6 | 1.29× | 0.94× |
| exactly_two_heads | 9.3 | 3.5 | 9.1 | 3.6 | 2.65× | 1.03× |
| prisoners | 59.0 | 58.9 | 2.8 | 2.7 | 1.00× | **21.3×** |
| secretary | 32.3 | 24.6 | 1.8 | 1.7 | 1.31× | 18.3× |
| *(15 more, all ≤ 7 ms)* | | | | | ~1.2× | ~1.0× |
| **TOTAL** | **3854** | **3274** | **923** | **843** | **1.18×** | **4.17×** |

The 80 ms the JIT still buys over gpu-only is concentrated in the **thin-cone, many-draw**
programs — `beta_bernoulli` (+40), `st_petersburg` (+7), `birthday` (+6.6), `exactly_two_heads`
(+5.5), `qjl_scalar` (+4.8), `dithering` (+4.5). Rejection sampling and count-style programs:
few ops per draw, lots of draws, exactly where per-instruction interpreter overhead dominates
and where the GPU's dispatch floor makes it (mostly, but see D4a) decline. Those programs are
the recovery target of D3 and D4a below.

## What we get

| | today | after |
|---|---|---|
| shipped CLI | interpreter, 3854 ms | **gpu, 923 ms → target ≤ 700 ms** |
| browser | wasm_emit (+ G3 WebGPU, spec'd) | same — and G3 gets the freed effort |
| native dev build | 3 CPU backends + GPU | 2 backends (interp floor + GPU) |
| code | `jit.rs` 1,834 lines, 4 cranelift crates, ~20 `cfg` sites across `backend.rs`/`compile_cache.rs`/`lib.rs`, a CI matrix leg | deleted |
| lowerings of the IR to keep conformant | 4 (interp, jit, wasm, wgsl) | 3 |
| build | cranelift in every `--features jit` build | gone (wgpu enters the *CLI* build only) |

The conformance corpus (`conformance.rs`) stays — `wasm_emit` and the GPU tests consume the same
tables. `kernel.rs` (`supported`, cost model) stays — the gate and `wasm_emit` use it.

## What we lose, honestly

1. **The no-GPU native floor becomes the interpreter.** Headless CI, docker, a lost device —
   anywhere `request_adapter` returns `None` — falls to 3854 ms where a jit build had 3274 ms
   (1.18×). This is the real cost of the decision, and it is what D3 exists to claw back: every
   interpreter win pays on *all* targets (the wasm small-forcing path runs the interpreter too),
   which is more than the JIT could ever say.
2. **9% on a GPU machine**, concentrated in the six thin-cone programs above. D4a recovers about
   half of it by recalibration alone (measured below).
3. **`--features jit` as a comparison baseline for future backend work.** Acceptable: git keeps
   it, and the two-tier conformance contract is what actually guards correctness.

Deliberately *not* a loss: web performance (jit never built on wasm32), determinism (the GPU is
already under the two-tier contract — tier-1 draws bit-identical, tier-2 f32 ULP-close — and
falls back rather than erring), or the threaded reducer (it is generic over `Program`; multicore
interpretation stays).

## D0 — profile before optimizing (the missing data)

The corpus after gpu-only is **923 ms, and 66% of it is three programs**: `noise_colors` (221),
`am_vs_fm` (211), `barrier_option` (181). We do **not** know where inside those the time goes —
eval/graph-build, lowering, pipeline compile, dispatch, readback, CPU fold, or the plot/introspect
passes (which take the sampler path, not `run_reduction`, and today never touch the GPU). Nothing
in D3/D4 beyond the already-measured items should start before this exists.

- [ ] **Per-forcing phase timers** behind `NOISE_PROFILE=1`, printed per forcing: eval (graph
  build), unroll+simplify+lower, gate decision **and the reason** (which of the three `profitable`
  terms failed), pipeline cache hit/miss + compile ms, dispatch ms, readback ms, fold ms, and the
  plot/introspect sampler passes. `stats.rs` already installs a per-engine recorder; extend it
  rather than adding a second channel.
- [ ] **samply on the top three** (`noise_colors`, `am_vs_fm`, `barrier_option`) in the gpu-only
  build; attach flamegraph findings here. Hypothesis to kill or confirm: the residue is
  plot/introspection sampling on the CPU + forcings the gate declines, not GPU inefficiency.
- [ ] **Gate confusion matrix**: run the corpus gated vs `NOISE_FORCE_GPU=1`, per-forcing diff.
  The gate was calibrated against the *multicore JIT* (`gpu.rs` table); its floor is about to
  get 1.18× slower, so its declines are now suspect (first evidence in D4a).
- [ ] **Interpreter microscope**: samply `beta_bernoulli` + `exactly_two_heads` interp-only.
  Hypothesis for D3a: per-instruction column passes are memory-bound (a 40-register program
  walks 160 KB of columns per instruction sweep, blowing L1).
- [ ] **Cold-start truth**: the bench's process-wide pipeline cache + warm-up rep hides cold
  pipeline compile (G0: ~6.5 ms per RNG source, super-linear on statement volume). Measure a
  cold `noise <file>` run per example; this decides D4e's priority.

## D1 — remove the JIT (mechanical)

- [ ] Delete `src/jit.rs`; drop the 4 `cranelift*` deps and the `jit` feature from
  `noise-core/Cargo.toml`.
- [ ] Collapse the `#[cfg(feature = "jit")]` arms: `backend.rs` (6 sites — `compile_root` /
  `compile_roots` pick interp directly off-wasm), `compile_cache.rs` (3 — the cache key's
  backend/gate bucket loses a variant), `lib.rs` (1).
- [ ] Migrate the JIT-only conformance assertions (`jit.rs` tests consumed `conformance.rs`
  tables plus its own moment probes) — anything not already covered by `wasm_emit`/GPU suites
  moves, nothing is silently dropped.
- [ ] CI: replace the `test (workspace + jit)` leg with a `--features gpu` leg. macOS runners
  have Metal; for ubuntu either install lavapipe/llvmpipe (slow but exercises the real path +
  fallback) or rely on the graceful `request_adapter → None → interp` decline, which is itself
  the thing to test.
- [ ] Sweep docs/plans/bench comments that say "and again with `--features jit`"
  (`benches/examples.rs`, `sampling.rs`, module docs in `reduce.rs`/`backend.rs`, CONTRIBUTE.md).
- [ ] Keep `benches/` and `bench_table.rs` running in interp and gpu configs — the two that
  remain meaningful.

## D2 — ship the GPU in the CLI

- [ ] `noise-cli`: enable `noise-core/gpu` (native targets). `noise-core`'s default features
  stay empty — the crate must remain wasm-clean and dependency-light for `noise-wasm`.
- [ ] Measure and record: CLI build time and binary size, cranelift-build vs wgpu-build (wgpu is
  not small either; the honest ledger needs both numbers). Startup adapter probe cost (device
  acquisition is lazy in `gpu.rs::device()` — confirm first-forcing latency is acceptable, or
  warm it on a background thread at engine construction).
- [ ] Release note: results remain under the two-tier contract — draws bit-identical, f32
  arithmetic ULP-close; a user diffing output between a GPU and a no-GPU machine can see
  last-ulp differences in tier-2 stats. This is already true of `--features gpu`; it becomes
  default-visible now, so say it out loud.

## D3 — make the fallback floor faster (interpreter)

Everything here pays *triple*: the no-GPU native floor, the gate-declined small forcings on GPU
machines, and the wasm small-forcing path in the browser. Ordered by expected value; D0's
interpreter profile gates the order.

- [ ] **D3a — L1-tile the batch.** The prime suspect for the JIT's whole 1.2–2.9× on thin cones.
  Columns are `[f32; 1024]` = 4 KB per register; each instruction is a full pass over its
  columns, so a 40-register program streams a 160 KB working set per instruction sweep —
  nothing stays in L1. Tile the 1024-lane batch into sub-blocks (128–256 lanes) and run the
  *whole instruction list* per tile: the working set drops to 20–40 KB and every operand read
  after the first hits L1. Values are unchanged by construction — counter-keyed RNG makes a
  draw a pure function of `(seed, lane, source)` and instructions are elementwise per lane, so
  loop order is free. Risk: shaped ops (`Permutation`, `Rotation`, `ArrDraw`) are lane-major
  arrays sized to BATCH — they either keep full-batch passes or tile with them; handle by
  splitting the instruction list at shaped boundaries if needed.
- [ ] **D3b — register liveness reuse** (the deferral noted in `bytecode.rs` since Phase 4).
  Fewer live columns = smaller working set; multiplies D3a rather than competing with it.
- [ ] **D3c — SIMD audit.** `tests/simd_probe.rs` exists; verify the hot arms (compare, select,
  bernoulli-style draw+threshold, the inlined transcendental polys) actually autovectorize on
  aarch64 + x86-64, fix the ones that spill. No hand-SIMD unless the probe proves a specific
  arm doesn't vectorize.
- [ ] **D3d — superinstructions, only with D0 evidence.** Fuse the patterns the thin-cone
  corpus actually runs (draw→affine, compare→accumulate). This is the "small JIT" slippery
  slope — do it only for arms the profile shows dominating after D3a/D3b.

Success gauge: `beta_bernoulli` interp from 61 ms toward the JIT's 21 ms; corpus interp total
from 3854 toward ~3300 (i.e., recover the JIT's whole 1.18× *for every target at once*).

## D4 — GPU headroom

- [ ] **D4a — recalibrate the gate against the real (interpreter) floor.** Measured already,
  gpu-only build, gated vs forced:

  | example | gated | forced | verdict vs interp floor |
  |---|--:|--:|---|
  | beta_bernoulli | 69.8 | **49.3** | gate now wrong — GPU wins |
  | exactly_two_heads | 14.8 | **9.5** | gate now wrong — GPU wins |
  | pi | 4.8 | **3.8** | gate now wrong — GPU wins |
  | st_petersburg | 33.7 | 91.2 | gate right — keep CPU |
  | dithering | 10.5 | 14.3 | gate right |
  | birthday | 5.6 | 8.5 | gate right (mixed forcings) |
  | bootstrap | 26.8 | 33.9 | gate right |

  `ops/draw` alone no longer separates the classes (`beta_bernoulli` at 37 wins, `bootstrap` at
  30 loses, `st_petersburg` at 58 loses *badly*) — the missing terms are total draws, readback
  volume, and how much of the program is *many small forcings* vs one big one. Re-derive the
  discriminator from the D0 confusion matrix; expected recovery ~25–30 ms of the JIT's 80.
- [ ] **D4b — joint kernels on the GPU** (the G4d item, now with a target). `noise_colors`
  forces 10 queries over one graph and is the single biggest corpus item (221 ms). The CPU
  already compiles joint programs (`compile_roots`); emit ONE shader computing all roots into a
  strided out-buffer, one dispatch, one readback, one pipeline compile instead of ten.
- [ ] **D4c — on-GPU reduction.** Today every dispatch reads back 4 MB per 1M lanes and folds on
  the CPU. For moment/count reducers, fold in workgroup shared memory and read back per-chunk
  partials (16,384-sample chunks are the determinism unit — map workgroups onto chunk
  boundaries and the chunk-ordered fold is preserved bit-for-bit). Readback drops ~256×;
  quantile/collect reducers keep the raw path.
- [ ] **D4d — overlap dispatch and fold.** `gpu.rs` runs dispatch → map → fold sequentially per
  1M-lane range; double-buffer so the CPU folds chunk N while the GPU runs N+1. (Mostly
  subsumed by D4c for moments; still pays for collect/quantile.)
- [ ] **D4e — pipeline cache to disk** (`wgpu::PipelineCache`, Metal/Vulkan support varies).
  The bench hides cold compile; a cold CLI run pays G0's ~6.5 ms/source, super-linear on big
  bodies. Priority decided by D0's cold-start measurement.
- [ ] **D4f — coverage:** Poisson as a divergent WGSL loop, and draws *inside* a rolled loop
  (today: falls back to unrolling — fine on CPU, but it's what keeps some loop programs off the
  GPU). Both widen which forcings escape the CPU floor at all.

## Order and success criteria

**D0 → D1 → D2** land together as "the switch" (D1/D2 are mechanical; D0 is a prerequisite for
everything after and independent of the switch). **D4a** is the first optimization (cheapest,
data in hand). Then **D3a** and **D4b** in either order — they attack the two biggest remaining
pools (the thin-cone floor and `noise_colors`). The rest as D0's profiles justify.

| milestone | corpus (gpu-only build) | gate |
|---|--:|---|
| today | 923 ms | — |
| + D4a gate recalibration | ~890 ms | no regressions vs gated baseline, per-example |
| + D4b joint kernels | ~750 ms | `noise_colors` ≥ 2× its 221 ms |
| + D3a interp tiling | ~700 ms | `beta_bernoulli` ≤ 40 ms without the GPU |
| stretch (D4c/D4e/D4f) | ≤ 600 ms | cold-run corpus within 1.3× of warm |

And the one that matters most: **the shipped CLI goes from 3854 ms to whatever that row says** —
dropping a backend is the fastest release we will ever cut.

## Risks

| risk | answer |
|---|---|
| Headless/CI native regresses 1.18× | D3 exists for exactly this; fallback is *correct* today (decline → interp, tested); llvmpipe optionally exercises the GPU path in CI |
| wgpu build weight in the CLI | measured in D2 against the cranelift weight it replaces; noise-core default build stays dependency-light either way |
| tier-2 ULP visibility for default users | already the contract; release-note it (D2) |
| gate recalibration regresses a program | per-example no-regression gate in D4a; `NOISE_FORCE_GPU` harness makes the confusion matrix reproducible |
| we want a CPU codegen backend back someday | git history keeps `jit.rs` + its tests whole; the conformance corpus it would need to re-pass never left |
