# PLAN-PREGPU — one numeric + RNG contract across every backend, before the GPU exists

**Date:** 2026-07-13 · **Status:** proposal (nothing started). Absorbs the former
PLAN-ASYNC (now Track A). PLAN-WEBGPU builds on all three tracks.

## The decision

PLAN-WEBGPU originally treated the GPU as a *different numeric tier*: f32 lanes and a
counter RNG on the GPU, f64 lanes and xoshiro everywhere else, with a weakened
"statistically equivalent" contract bridging the two. This plan flips that: **move every
backend to the GPU's model first** — then the GPU lands as just another backend, not a
semantic fork.

Three tracks, one contract:

- **A — async engine.** `Engine::run`'s internals become suspendable so a forcing can
  await an async backend (and the playground gains cancellation + a responsive tab).
- **B — 32-bit lanes, 64-bit aggregation, in all modes.** Random-variable *lanes* (the
  sample columns every backend fills) become `f32`; everything deterministic — scalar
  `Value::Num` math, reducer accumulation (Σx/Σx² already `f64`), final estimates — stays
  `f64`. `unif_int` and friends get much smaller (but honest) limits everywhere.
- **C — counter-based RNG (pcg4d) in all modes.** One generator — pcg4d, keyed
  `(seed, lane, source)` — on the interpreter, the JIT, the wasm emitter, and later the
  GPU, bit-identically.

What this buys:

- **Draw-stream parity.** A uniform draw is `pcg4d(key) → u32 → f32` — pure u32
  arithmetic + one exactly-specified conversion, native in both Rust and WGSL — so the
  *same seed produces the same draws, bit for bit, on every backend including the future
  GPU*. The determinism story survives the GPU intact instead of degrading to "per
  backend".
- **Near-bitwise conformance.** With identical draws and f32 lanes everywhere,
  cross-backend tests can assert bit-equality for add/mul/select graphs and tight-ULP
  equality elsewhere (WGSL guarantees correctly-rounded `+ − ×`; `÷`/`sqrt` are ≤2.5 ULP,
  transcendentals vendor-specified — the only remaining daylight).
- **One published breaking change.** B and C each invalidate every seeded output (C
  changes the draws, B their precision). They land in sequence (C, then B — see
  Sequencing) but the release cuts after both, so users see one seed-break and the corpus
  re-baselines once. An intervening release would mean a second break — a choice to make
  explicitly, not stumble into.
- **CPU upside, not a tax.** f32 doubles SIMD width (NEON: 4 lanes → 8) and halves memory
  traffic in every column; f32 polynomial approximations need fewer terms than the current
  f64 ones in `approx.rs`; and the counter RNG deletes the entire multi-stream apparatus —
  measured on this machine (scratch bench, M-series): pcg4d 188 M u64-equiv/s vs
  xoshiro256++×4-streams 194 M u64/s single-threaded, i.e. the ILP the 4-stream interleave
  was built to expose comes free with independent counters. The RNG swap itself is
  expected ~neutral; the **checkable f32 prediction** is the big-cone interpreter demos:
  the columnar VM's register file is `cone × 1024 × 8 B` — ~144 MB for `turboquant`
  (17.6k nodes), ~367 MB for `prisoners` (44.8k) — and they run at 50–80 M op-lanes/s
  today (locality-bound, not arithmetic-bound; measured 2026-07-13). Halving the footprint
  should show up directly in `example_times` for exactly those two demos; if it doesn't,
  the locality theory is wrong and B4 should say so.

## Track A — async engine

Scope and design are unchanged from the former PLAN-ASYNC (full call-graph walk,
2026-07-13). Summary; the detail that matters is the inventory and the traps.

**Shape:** `eval` is the evaluator's one recursion cycle → the one `Box::pin` point
(`fn eval(&mut self, node) -> Pin<Box<dyn Future<Output = Result<Value>> + '_>>`); every
function that transitively calls `self.eval` or forces sampling becomes `async fn`; the
forcing leaves await async seams in `sampler.rs` (`moments_async`, `sample_n_async`,
`cond_moments_async`, `cond_sample_n_async` — where the GPU later routes). Public sync
names (`run`, `run_to_document`, `check`, …) remain as one-line wrappers over a new
`exec::block_on` (~60 lines, std-only: native park/unpark executor; wasm32 single-poll
that panics with a pointer at the async API if something suspends), so all ~90 existing
call sites and every test pass unchanged. `noise-wasm` keeps its sync exports and gains a
Promise-returning `run_async` (`wasm-bindgen-futures`).

**The async set** (nothing outside it changes):

| file | functions that become `async` |
|---|---|
| `eval.rs` | `run`→`run_async`, `run_with_inputs`, `run_to_document`, `check`, `eval_top`, `eval` (boxing point), `eval_inner` |
| `eval/eval_arms.rs` | `render_template`, `eval_array`, `eval_range`, `range_bound`, `eval_index`, `eval_for`, `eval_comprehension`, `eval_bind`, `eval_call`, `dispatch_call`, `input_call`, `eval_input_num/str/value` |
| `eval/draw_lift.rs` | `call_user_fn`, `eval_sample`, `lift_if` |
| `eval/ops.rs` | `eval_unary_expr` |
| `eval/library.rs` | `eval_matmul` only |
| `eval/introspect_dispatch.rs` | `eval_cond`, `query_cond` |
| `builtins.rs` | `call`, `prob`, `moment`, `quantile`, `prob_cond`, `moment_cond`, `quantile_cond` |

**Pre-found traps:** `lib_adjoint` calls `builtins::call("transpose", …)` — a pure
builtin; call a `pub(crate) builtins::transpose` directly or all 30+ `lib_call` arms go
async for nothing. `dispatch_call`'s `#[inline(never)]` rationale goes stale (drop it; the
box at `eval` bounds future sizes). `MAX_EVAL_DEPTH` stays load-bearing — async does not
trampoline the poll stack. The `stats::install` guard held across `.await` becomes an
interleaving hazard once real suspension exists (note now, fix in PLAN-WEBGPU G3).
`input_call` and `render_template` are in the set even though they don't look like
evaluation. `QueryCtx` borrows across `.await` are sound (shared graph borrow only).

**Cancellation (AbortSignal).** The core stays web-agnostic: a cloneable
`exec::CancelToken` (`Arc<AtomicBool>`; `cancelled()` is a relaxed load), threaded into
the async entry points via run options — the sync wrappers pass a never-cancelled token,
so nothing changes for existing callers. Two check tiers:

- **statement boundaries** — A3's yield points between top-level statements;
- **per reducer chunk** — inside `run_reduction`'s chunk loop, one relaxed load per
  16,384-sample chunk (free), so aborting a single 10⁷-draw `prob(...)` takes
  milliseconds, not the rest of the forcing.

A tripped token surfaces as `Err(Error::Cancelled)` — a new non-program error (no
diagnostic, no partial document), distinguishable from both program errors and panics.

The wasm subtlety that shapes A3: on the single-threaded web build nothing can *set* the
flag while Rust runs — `abort()` is JS that only executes when the engine returns to the
event loop, and `'abort'` listeners fire from a task, so a microtask-only yield is not
enough. The cooperative yield must be a real event-loop hop (a `setTimeout(0)`/
`MessageChannel`-backed Promise via `wasm-bindgen-futures`), and it must happen *inside*
long forcings, not just between statements: the `sampler::*_async` twins own an async
chunk driver that walks `run_reduction`'s chunks (the per-chunk function stays sync) and
hops the event loop on a ~50 ms timer — per-chunk cost stays one branch, and the hop is
also what keeps the tab responsive. Native needs no yields: any thread can set the token
(CLI Ctrl-C, embedding hosts), and the per-chunk check inside the `std::thread::scope`
workers picks it up. The `wasm-threads` build is the same as single-threaded wasm from
the main thread's perspective (workers share the flag through linear memory once set,
but *setting* it still requires the driving thread to yield).

JS surface follows the platform convention (`fetch`-style): `noise-wasm` exports
`run_async(src, signal?: AbortSignal)` and `@noiselang/core` exposes `run(src, { signal })`.
Semantics mirror `fetch`: if `signal.aborted` is already true, reject immediately;
otherwise register an `'abort'` listener that sets the token (removed when the run
settles), and reject with `signal.reason` — the standard `AbortError` `DOMException` — so
callers' existing `err.name === "AbortError"` habits work unchanged.

One semantic to state, not discover: a cancelled run leaves the `Engine` scope partially
updated (bindings defined before the abort persist). Rule: treat a cancelled engine as
stale — the playground's introspection sidecar, which relies on scope persisting across
`run()`, must rebuild from a fresh `Engine` after a cancel rather than trust partial
state.

**Steps:** A1 spine (~400–800 mechanical lines; proof: an `exec` test drives a
pending-once future through `block_on`) · A2 `run_async` wasm export + playground `await`
· A3 cooperative event-loop yield (statement boundaries + the ~50 ms timer in the async
chunk driver) + `CancelToken` → `AbortSignal` wiring — real suspension *and* abort
exercised end-to-end before any GPU code exists.

## Track B — f32 lanes, f64 aggregation, everywhere

**The line it draws.** The language has two kinds of numbers and this makes the split
explicit: *deterministic values* (`Value::Num`, `gcd`/`modpow`, ranges, array literals,
signal time axes) stay `f64`; *random-variable lanes* (every column a backend fills — the
sample-DAG's per-draw values) become `f32`; *aggregation* (Σx/Σx² reducers, in-condition
counts, quantile interpolation, everything that becomes a reported estimate) stays `f64`.
Monte-Carlo standard error `O(1/√N)` dwarfs f32's ~1e-7 relative rounding at any feasible
N, so estimates move within their own confidence interval — the reducers being f64 is what
keeps *accumulation* from being the place f32 actually hurts.

**What changes, per file:**

- `backend.rs` — `Runner::next_batch → &[f32]`, `JointRunner::col → &[f32]`. The seam's
  one type change; everything else follows from it.
- `bytecode.rs` — column register file `Box<[f32]>`; instruction fills in f32.
- `rng.rs` — fill functions produce f32 columns (`(u32 >> 8) as f32 * 2^-24` uniforms;
  Box–Muller / inverse-CDF in f32). Merges with Track C's generator swap.
- `jit.rs` — f32 SSA (`f32.mul` equivalents, `movss`/`mulps` world); `approx.rs` gains
  f32 polynomials (fewer terms than the f64 ones — a speedup, and they must be *shared*
  with `wasm_emit` and later the WGSL emitter so all backends agree).
- `wasm_emit.rs` — `f64.*` → `f32.*`; imports (`Math.*`) compute in f64 and round — an
  accepted ≤1-ULP seam on the rare fallback paths, or truncate explicitly for parity.
- `reduce.rs` — `absorb(&[f32])` widening each element to f64 in the lane loop
  (vectorizable convert); accumulators unchanged.
- `sampler.rs` / `introspect.rs` — public draw vectors stay `Vec<f64>` (widen at the copy
  boundary) so quantiles/plots/stats upstream are untouched.

**The smaller limits (the user-visible part):**

- `unif_int(lo, hi)`: every value in the range must be exact in f32 → require
  `|lo|, |hi| ≤ 2²⁴` (16,777,216). Enforced at the constructor with a teaching error, in
  all modes. Corpus audit (2026-07-13): the largest range any example draws is **365**
  (`birthday`); nothing is near the cap. Deterministic integer builtins (`gcd`, `modpow`)
  are untouched — they never enter a lane.
- `poisson(lambda)`: draws above 2²⁴ stop being exact integers; cap or document (the
  normal-approximation regime is already an approximation — document).
- Finite range: f32 overflows at 3.4e38 where f64 ran to 1.8e308 — a lane holding
  `2^k`-style payouts or long products can now hit `inf`. Corpus audit: `st_petersburg`'s
  `2^(k+1)` needs k > 126 (probability 2⁻¹²⁶ — unobservable) and `barrier_option` works in
  log-space; no example is at risk, but this goes in LANG.md as a documented boundary.
- Subnormal tails: f32 flushes around 1e-38; irrelevant at MC precision but noted.

## Track C — pcg4d counter RNG, everywhere

**Generator (settled after weighing Squares, 2026-07-13): pcg4d** (Jarzynski–Olano,
*Hash Functions for GPU Rendering*, JCGT 2020) — pure u32 mul/add/xor/shift, 4×u32 in →
4×u32 out, native in both Rust and WGSL (no u64, no mulhi, nothing to emulate). One hash
yields four u32s → up to four f32 uniforms; `normal` takes its Box–Muller pair from a
single hash.

**Why pcg4d and not Squares/Philox.** Measured per f32 uniform (scratch bench, M-series,
single thread): **pcg4d ~750 M/s** · squares32 ~136 M/s · Philox ~118 M/s; on the GPU,
pcg4d is ~5 ALU ops per uniform vs ~70–90 for either alternative (both are built on wide
multiplies WGSL must emulate via 16-bit splits). The quality question is *certification
depth, not known flaws*: pcg4d was a top performer in its paper's TestU01-based
evaluation and is a de-facto standard in production shaders, but nobody has published for
it the dedicated BigCrush + large-volume PractRand campaign Widynski ran on Squares — and
its 4-word keying is ad-hoc where Squares' sequential-counter regime is exactly what was
certified. For Monte-Carlo estimation (means/variances/quantiles at 10⁶–10⁸ draws) that
evidence gap is expected to be immaterial — generators that fail batteries only after
terabytes are indistinguishable in MC results — but it is *checkable*, which is C0's job.
**Fallback: Squares** (`squares32`; it dominates Philox — 3× faster on CPU, similar
emulated GPU cost, equal certification). Exactly one generator ships; a swap happens only
on a C0 failure or a G0 measurement, always before the numerics-v2 release, frozen after
(a later change is a second seed-break).

**Keying.** Input words `(key_lo, key_hi, global_lane, source_offset)`: the key is
SplitMix64(seed) split into two u32s (computed once per run), `source_offset` is a
compile-time constant per RNG source node, and `global_lane` is the draw's absolute lane
index. `global_lane` being one u32 caps a single forcing at 2³² lanes — far above the op
budget's practical draw caps, but stated in LANG.md rather than discovered. This keying
is the part with the least prior certification, so it is what C0 stresses hardest.

**What it deletes** (the payoff beyond parity): `STREAMS`, `seed_state`,
`choose_streams`, `latency_bound` — the entire multi-stream policy in `kernel.rs` — plus
the stream-strided state layout and per-stream state load/store in both emitters, the
serial xoshiro state in the interpreter's `Rng`, **and `chunk_seed` itself** (an insight
from the Squares evaluation that carries over): with the global lane index in the key, a
reducer chunk is just a lane range — no per-chunk key derivation, and thread-count
invariance is a triviality instead of a theorem.

**What it breaks:** every seeded sequence. The `rng.rs` known-answer test gets re-pinned
to pcg4d vectors; committed example outputs re-baseline — and re-baseline *again* when
Track B lands, which is why the release cuts after B, not between (see Sequencing).

**C0 — the generator spike (run first; zero codebase dependencies).** pcg4d is ~15 lines
of Rust and the battery needs nothing from the engine; one day of harness + an unattended
weekend of compute settles the question before any integration exists. It certifies *our
exact usage*, where pcg4d's evidence is thinnest — the ad-hoc keying: cross-word
avalanche (flip any input bit, especially low `global_lane` bits → each output bit flips
with p = 0.50 ± 0.01), lane-*i*-vs-lane-*i+1* and source-*k*-vs-source-*k+1* correlations
over millions of pairs (the pairings that would bias joint queries), domain known-answers
at high N (`pi/4` CI, chi-square on `unif_int(1,6)`, Box–Muller normality through
kurtosis), and PractRand to ≥1 TB over bits serialized in kernel-consumption order — with
Squares running the same statistics as the in-harness certified reference. **Pass/fail
criteria fixed before running** (PractRand clean, avalanche in band, conformance
false-positive rate at the Bonferroni-corrected alpha) so the verdict can't be negotiated
after the fact. Fail → Squares replaces pcg4d and the same harness re-certifies the swap.

**Gate:** land only if the example corpus is neutral-or-better on `example_times`
(expectation: neutral-to-better — pcg4d matches the current 4-stream xoshiro on the
microbench (188 vs 194 M u64-equiv/s) before counting the deleted stream machinery and —
with B — wider SIMD; the corpus verdict, not the microbench, decides).

## Sequencing

Order (owner's call, 2026-07-13): **C → B → A**, releasing after B.

0. **C0 (generator spike)** — immediately: zero dependencies, certifies pcg4d over our
   exact keying (with Squares as the in-harness reference) before any integration exists.
1. **C (pcg4d + the simplification)** — the smallest self-contained cut: swap the
   generator, delete the stream machinery (`STREAMS`/`seed_state`/`choose_streams`/
   `latency_bound`/`chunk_seed`), lanes stay f64 for now — each f64 uniform takes two
   u32s from one hash (~376 M f64-uniforms/s, ~2× today's per-uniform rate). Validates
   the counter design (determinism, chunk-as-lane-range) before the wider f32 surgery.
2. **B (f32 lanes)** — riding the already-simplified RNG layer (the fills just drop to
   one u32 per uniform): B1 seam type change (`&[f32]`) → B2 interpreter + fills → B3
   JIT/wasm emitters + shared f32 polys → B4 corpus re-baseline + `example_times` gate.
   **Cut the numerics-v2 release here** — after B, not between C and B — so the two
   internal seed-breaks publish as one (one corpus re-baseline, one LANG.md update: f32
   lane semantics, the 2²⁴ integer-draw cap, the new RNG).
3. **A (async)** — last (or in parallel: it touches the eval spine while B/C touch the
   backends, overlapping only in `sampler.rs`/`builtins.rs`). It gates nothing until
   PLAN-WEBGPU G3; A3's playground payoff (cancellation, responsive tab) just arrives
   correspondingly later.
4. **Then PLAN-WEBGPU** G0–G4, now inheriting: identical draw streams, an already-async
   engine, and a numeric contract the GPU doesn't bend. (G0, being dependency-free, can
   run any time earlier for its compile-time answers.)

## Risks

| risk | assessment |
|---|---|
| f32 lanes change every published estimate | Within each estimate's own MC confidence interval; the one-time re-baseline is the cost, and it's shared with the RNG break |
| A real program needs integer draws > 2²⁴ or lane magnitudes > 3.4e38 | Corpus max today: 365 and ~2²³; teaching errors at the constructor make the boundary honest. If it ever genuinely bites, that program belongs on f64 CPU — a per-program escape hatch (`engine::set_lane_precision`?) is deliberately NOT in scope until someone real asks |
| pcg4d's certification is thinner than Squares'/Philox's (one paper's hash evaluation, ad-hoc keying) | No known flaws — the gap is evidence depth, and MC estimation is far less demanding than the batteries. C0 closes it empirically over our exact keyed usage, with fixed pass/fail criteria and Squares as the certified in-harness reference; a C0 failure swaps to Squares before the seed-break |
| f32 transcendental polys diverge across backends | One shared `approx.rs` f32 table consumed by JIT, wasm emitter, and later the WGSL emitter — divergence is a compile error, not a drift |
| Interpreter throughput regresses on the per-lane hash | Bench says counter ≈ 4-stream xoshiro before SIMD; corpus-neutral gate enforces it |
| Async borrow/stack surprises | Scoped by call-graph walk; `MAX_EVAL_DEPTH` test guards the poll stack (see Track A traps) |
