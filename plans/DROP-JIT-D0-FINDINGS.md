# PLAN-DROP-JIT D0 — profiling findings

**Date:** 2026-07-15 · M4 Pro, `--features gpu`, `bench_table` (fresh engine/rep, median of 5),
`NOISE_PROFILE=1` per-forcing phase timers (see `crate::profile`).

## Baseline reconfirmed (matches the plan within noise)

| config | corpus total | plan said |
|---|--:|--:|
| interpreter | 3868 ms | 3854 |
| `--features jit` | 3268 ms | 3274 |
| `--features gpu` | 937 ms | 923 |
| `--features jit,gpu` | 851 ms | 843 |

The thesis holds: gpu-only is 937 ms; the JIT's marginal value over gpu-only is ~86 ms (9%);
dropping JIT + shipping GPU is a 4.1× speedup for CLI users (3868 → 937).

## The headline: the residue is **CPU joint/plot passes**, not GPU inefficiency

The 66%-of-corpus top three break down completely differently than "GPU is slow":

### am_vs_fm (210 ms) — **95% is two `plot::line` CPU passes**
```
run_reduction n=40000 ops/draw=821   →  ~3 ms  (GPU, warm: pipeline HIT, readback 2.6ms)
run_reduction n=40000 ops/draw=1136  →  ~3 ms  (GPU, warm)
joint         n=40000 ops/draw=63    →  0.06 ms (plot::line(msg) — constant, clamped)
joint         n=40000 ops/draw=567   →  78 ms   (plot::line(recovered_am) — CPU interp!)
joint         n=40000 ops/draw=882   →  123 ms  (plot::line(recovered_fm) — CPU interp!)
```
The `E(...)` forcings run on the GPU and are ~free warm. The **two signal plots are 200 ms of CPU
interpreter** — `for_each_joint_batch` (64 signal points × 40000 draws) never touches the GPU.

### barrier_option (180 ms) — **one `plot::fan` CPU pass is 53%**
```
4× run_reduction (E/P) n=200000    →  ~3 ms each warm (GPU)
joint n=76923 ops/draw=315         →  96 ms  (plot::fan(path) — 52 weekly points, CPU interp!)
```

### noise_colors (232 ms) — **GPU dispatch-floor + 2 CPU-declined big cones**
```
8× run_reduction n=3000            →  6–25 ms each — readback/round-trip latency bound
                                      (dispatch 0.03 ms, readback 6–25 ms: 3000 lanes when the
                                       GPU wants ≥256k — the dispatch floor, paid 8×)
2× run_reduction n=436             →  ~51 ms each on CPU
                                      (gate DECLINE — cone too big, 12206 instrs > 8000)
```

## Per-phase truths (from the profiler)

1. **`gpu.readback` dominates every GPU forcing**, not compute. It is the blocking `poll(Wait)` —
   i.e. GPU round-trip *latency*, ~2.6 ms floor even for a warm 200k-lane dispatch, 6–25 ms for a
   cold/tiny 3000-lane one. `gpu.dispatch` (the compute submit) is 0.02–0.2 ms. The GPU is
   latency-bound, never throughput-bound, on this corpus.
2. **Pipeline compile is 1.5–5.7 ms per distinct shader, always a cold MISS on first sight.** The
   process-wide cache makes reps 2..N HIT (0.02 ms), so the bench's warm-up rep hides it — but a
   **cold CLI run pays it once per distinct query** (noise_colors = 10 distinct shaders ≈ 30–50 ms
   of pipeline compile alone). This is D4e's evidence.
3. **The joint/plot path (`for_each_joint_batch`) is CPU-interpreter-only and is the single biggest
   pool.** am_vs_fm 200 ms, barrier_option 96 ms — ~300 M lane-ops/s on the columnar interpreter,
   exactly the memory-bound signature D3a predicts (a 567–882-node cone streams 2–3.5 MB of columns
   per instruction sweep, blowing L1).
4. **Gate confusion**: `beta_bernoulli` is gated onto the GPU (72.97 ms) and *loses* to the interp
   floor (64 ms) — the D4a mis-gate, confirmed live.

## What this means for D3/D4 ordering (revised)

- **D3a (L1-tile the interpreter) is now the highest-value optimization**, not just the fallback
  floor: it directly attacks the dominant cost (the CPU joint/plot passes in am_vs_fm +
  barrier_option ≈ 300 ms of the 937), *and* pays on the wasm browser path and the no-GPU floor.
- **D4b (joint GPU kernels)** should be reframed: the biggest joint passes are `plot::line`/
  `plot::fan` (`for_each_joint_batch`), which are CPU-only today. Routing *those* through a GPU
  joint driver (or D3a speeding them) is the prize — bigger than noise_colors' 10 separate `E`/`P`
  queries the plan originally scoped.
- **D4a (gate recal)** stands: cheap, `beta_bernoulli`/`exactly_two_heads`/`pi` mis-gate.
- **D4c (on-GPU reduction / avoid readback)** matters less than expected: readback is *latency*, not
  data volume (2.6 ms floor at 200k lanes). The win there is fewer round-trips (D4b joint = one
  dispatch for many roots), not smaller readbacks.

## D2 — shipped-CLI measurements (the switch, measured)

`noise-cli` now enables `noise-core/gpu` (M4 Pro, release):

| metric | value | note |
|---|--:|---|
| clean build time | 12.85 s | noise-cli + wgpu 26 (incremental after wgpu compiles once) |
| binary size | 6.0 M (4.9 M stripped) | the honest cost of wgpu vs the old interpreter-only CLI (~2–3 M). Cranelift would have added comparable weight had the JIT ever shipped — it never did. |
| adapter probe | ~ms (Metal) | `dice` cold-runs in 0.02 s, so `gpu::device()` acquisition is cheap; no background warm needed |
| cold-start (light) | dice 0.02 s, prisoners 0.08 s | fine |
| cold-start (heavy) | turboquant **2.88 s** cold vs 56 ms warm; noise_colors 0.86 s | the cold **pipeline-compile** tax (10 distinct heavy shaders), NOT adapter init — a one-time per-process cost, and the concrete case for D4e (disk pipeline cache) |
| correctness | dice P=16.7%, hist mean 3.50 | runs correctly on the GPU end-to-end |

**Ledger:** dropping the JIT + shipping the GPU takes the *shipped* CLI from interpreter-only
(3868 ms corpus) to 937 ms warm — a 4.1× speedup for every real user — at a cost of ~3.5 M binary
and a cold pipeline-compile tax on heavy programs (D4e).

## D4a — gate recalibration (landed, −27 ms)

`MIN_CONE_OPS` 100 → 45, re-derived from the confusion matrix (bootstrap tops at 41 ops/draw and
loses on GPU; beta_bernoulli starts at 47 and wins). Corpus 936.8 → 909.7 ms. beta_bernoulli −24.3,
noise_colors −5.8, st_petersburg −2.4, barrier_option −1.2; no real regression. See the D4a commit.

## D3a — L1-tiling: **attempted, measured a regression, reverted**

Implemented `run_batch` tiling (TILE=256, bit-identical — the wasm-vs-interp conformance stayed
byte-exact) and measured it: **+28.7 ms on the gpu-build corpus** (am_vs_fm +6.6, noise_colors +7.6,
st_petersburg +7.1, barrier_option +3.4). Reverted.

**Why it regressed — the plan's D3a/D3b ordering is backwards for this codebase.** With
one-register-per-node (no liveness reuse — `bytecode::compile` allocates a fresh register per graph
node, D3b deferred since Phase 4), the dominant CPU cones are the *wide* plot/joint ones: am_vs_fm's
`plot::line` cone is ~882 registers. Its full-batch working set is 882 × 4 KB = **3.5 MB**; a 256-lane
tile is still 882 × 1 KB = **882 KB**, which *does not fit L1* either (128 KB on M4). So tiling buys
no L1 locality on exactly the cones that dominate — it only adds the 4× instruction-walk overhead of
re-scanning the instruction list per tile. Tiling *does* help narrow cones (beta_bernoulli ~47 regs →
47 KB tile fits L1), but those are now on the GPU (D4a) or small in the total.

**D3b is a prerequisite for D3a, not a multiplier.** Register-liveness reuse would shrink the *live*
set from n_regs (hundreds, mostly dead) to ~10–50, making a 256-lane tile's working set ~50 KB —
comfortably L1-resident — at which point tiling pays. The correct sequence is **D3b then D3a**, a
larger change than D3a alone. Left as documented follow-up; the tiling patch is in this session's
scratchpad + git reflog if D3b lands.

## D4b / noise_colors −119 ms — gated behind cold-compile safety (D4e)

The confusion matrix's biggest single gap is noise_colors (forced 113.9 vs gated 232.7, −119): the
gate declines its two ~12,206-instr cones via `MAX_WGSL_INSTRS=8000`, and forcing them onto the GPU
wins 2× **warm**. But those two shaders cost ~1 s each to *cold*-compile (G0's super-linear pipeline
curve), so raising the cap would take a cold `noise noise_colors.noise` from 0.86 s (D2) to ~2.9 s —
a real regression for the shipped CLI, which runs cold. So this win is **gated behind D4e** (disk
pipeline cache) or a true D4b joint kernel (one shader for all roots, compiled once). Not a safe
standalone change now. The other D4b prize — routing the CPU-only `for_each_joint_batch` plot passes
(am_vs_fm 200 ms, barrier 96 ms) through a GPU joint driver — is a substantial new code path
(multi-column dispatch + per-column fold under the two-tier contract); scoped, not yet built.

## D0 deliverable status
- [x] Per-forcing phase timers behind `NOISE_PROFILE=1` — `crate::profile`, wired into
  `reduce::run_reduction`, `gpu::try_reduce` (simplify/emit/gate+reason/pipeline hit-miss/dispatch/
  readback/fold), and `sampler::{for_each_batch, for_each_joint_batch}`.
- [x] Top-three phase breakdown (above); hypothesis **confirmed** — residue is CPU joint/plot passes
  + GPU dispatch-floor latency, not GPU compute.
- [x] Gate confusion (beta_bernoulli mis-gate) confirmed; full matrix regenerated for D4a.
- [x] Cold-start truth: pipeline compile 1.5–5.7 ms/shader, cold-once per distinct query (D4e input).
