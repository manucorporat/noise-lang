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
- **C — counter-based RNG (Squares) in all modes.** One generator — squares64
  (Widynski; criterion-8 verdict, see Track C), a seeded construction-compliant key,
  sequential per-source counters — on the interpreter, the JIT, the wasm emitter, and
  later the GPU, bit-identically.

What this buys:

- **Draw-stream parity.** A uniform draw is `squares64(ctr, key) → 48 consumed bits →
  f64/f32` — pure integer arithmetic + one exactly-specified conversion — so the *same
  seed produces the same draws, bit for bit, on every backend including the future GPU*
  (WGSL pays u64-emulation ALU for it — measured at G0, see PLAN-WEBGPU). The
  determinism story survives the GPU intact instead of degrading to "per backend".
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
  f64 ones in `approx.rs`; and the counter RNG deleted the entire multi-stream apparatus —
  measured post-swap (M4 Pro, fill shape): squares64 at 540 M f64-uniforms/s vs the
  xoshiro-era ~194 M/s serial ceiling, and the example corpus 5% FASTER than the
  pre-Track-C baseline. The **checkable f32 prediction** is the big-cone interpreter
  demos:
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

## Track C — pcg4d-3r counter RNG, everywhere

**Generator (SWAPPED 2026-07-14, owner go): squares64** — Widynski's Squares, with a
per-seed key from his key construction (distinct non-zero hex digits per half, odd LSD —
`rng::Key::from_seed`; an arbitrary u64 is NOT a valid key, per C0 defect #3). Verdict
trail in tools/rng-cert/RESULTS.md: pcg4d-3r and pcg4d-3rf both failed PractRand at
256 GB (real low-consumed-bit sequential structure — pcg's LCG-fed counter mixing is
too shallow at 10¹¹-sample depth, and bijective per-word finalizers can't remove
cross-hash correlation); squares with a compliant key is **clean at 1 TB, zero
anomalies**. Consumption as built: one squares64 per draw counter yielding the 48
consumed bits `((w >> 40) << 24) | ((w >> 8) & 0xFFFFFF)` (never a u32's low byte —
the C0 contract); scalar counters `(source << 36) + lane` (sequential per source, the
certified regime); `normal` takes u1/u2 from the pair's even/odd counters, cos/sin by
lane parity; `CellStream` (Knuth poisson, Permutation, Rotation) uses the dedicated
region `(1 << 63) | (stream_ordinal << 49) | (lane << 17) | j` with per-program
ordinals assigned at lowering. **Gate CLOSED: the corpus runs 4130 ms vs the 4351 ms
pre-Track-C baseline (−5%)** — squares64's five 64-bit multiplies beat pcg's ~40 u32
ops on wide multiply pipes (fill shape: 540 M f64-uniforms/s). GPU cost (~50–90
emulated WGSL ALU ops/uniform) stays parked behind PLAN-WEBGPU G0's measurement.

**Why pcg4d-3r was tried first (historical — superseded by criterion 8 above).** C0's first battery
(tools/rng-cert, 2026-07-14) **disqualified pcg4d as published**: u32 carries only
propagate upward and its single `^= >>16` is the only downward path, giving it fully
deterministic input-bit → output-bit relations *inside the consumed region under
realistic keying* (e.g. key-bit 25 → mantissa bit 8, p = 0 or 1). The third round fixes
exactly that: pcg4d-3r is clean on every consumed bit (0/9696 cells outside ±0.01,
worst 0.0052 ≈ null) and passes all seven statistical criteria. Measured keyed-batch
throughput (M4 Pro, single thread): **pcg4d-3r 942 M u32/s** (2.4× the current 4-stream
xoshiro's ~388 M u32/s) · squares32 211 M/s (0.54× — a corpus-gate risk) · Philox ~118
M/s; on the GPU, pcg4d-3r is ~10 ALU ops per uniform vs ~70–90 for Squares/Philox (wide
multiplies emulated via 16-bit splits). The cost of the choice: pcg4d-3r is a custom
variant with **no published certification — the C0 harness carries the entire evidence
burden** (amended criterion 1, criteria 2–7, and PractRand ≥ 1 TB over the
consumed-bit stream, with squares32 running as the certified in-harness reference).
**Fallback: Squares** (`squares32` — clean over every reachable input bit; its only
avalanche failures are ctr bits 58–63, unreachable below 2⁵⁸ draws): a criterion-8
failure of pcg4d-3r swaps to Squares, accepting the ~2× CPU cost, always before the
numerics-v2 release, frozen after (a later change is a second seed-break).

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
to pcg4d-3r vectors; committed example outputs re-baseline — and re-baseline *again* when
Track B lands, which is why the release cuts after B, not between (see Sequencing).

**C0 — the generator spike (RAN 2026-07-14; harness + evidence in `tools/rng-cert/`).**
The battery — cross-word avalanche, lane-adjacent/source-adjacent correlations, domain
known-answers (`pi/4` CI, die chi-square, Box–Muller skew/kurtosis), PractRand over the
consumed-bit stream in kernel-consumption order, Squares as the certified in-harness
reference, pass/fail frozen in `tools/rng-cert/README.md` before running — produced a
verdict the plan didn't predict: **pcg4d as published failed** (deterministic
key-bit → consumed-mantissa-bit relations; carries only propagate upward and one `>>16`
xorshift is the only downward path), and the certified reference *also* failed the
literal criterion — only on unreachable input bits — flagging the criterion itself as
miscalibrated. Owner-ratified resolution (amendment in the README): criterion re-frozen
to consumed-bits × reachable-inputs, pcg4d disqualified, **pcg4d-3r adopted**, PractRand
≥ 1 TB still pending on both pcg4d-3r and Squares. A pcg4d-3r PractRand failure swaps to
Squares and the same harness re-certifies the swap.

**Gate:** land only if the example corpus is neutral-or-better on `example_times`
(expectation: better — pcg4d-3r measures 2.4× the current 4-stream xoshiro per u32 in
keyed-batch shape (942 vs ~388 M u32/s) before counting the deleted stream machinery and
— with B — wider SIMD; the corpus verdict, not the microbench, decides).

**Consumption contract (fixed by C0, holds in every backend and later WGSL):**

- One hash per `(lane, source)` cell; only bits 8..31 of a word are consumable.
- f32 uniform (after B): word 0, `(w0 >> 8) as f32 · 2⁻²⁴`.
- Interim f64 uniform (C, before B): words 0+1, `((w0>>8) << 24 | (w1>>8)) as f64 · 2⁻⁴⁸`.
- `normal`: `u1` from words 0+1 (offset +0.5 to dodge 0), `u2` from words 2+3, **cos
  branch only** — one normal per lane per hash, so lane-range chunking never straddles a
  Box–Muller pair (the sin twin is discarded; transcendentals dominate the cost anyway).
- `unif_int`: Lemire multiply-high on the 48 consumed bits (bias ≤ count/2⁴⁸; B's 2²⁴ cap
  makes it ≤ 2⁻²⁴).
- Fills needing more than one hash per lane (Knuth `poisson`, `Permutation`'s
  Fisher–Yates, `Rotation`'s Gaussian seed) chain via `rng::CellStream`: the base cell's
  words 2+3 become a chain key and iteration `j ≥ 1` hashes `(chain_key, j, source)` —
  full-u32 iteration space, no consumed-word reuse as key material, no cross-source
  aliasing (which any `source + f(j)` scheme would risk).
- `source_offset` = the source's `RvId` index in the simplified graph (stable across
  backends and joint compiles — which is exactly what `corr`'s shared-draw semantics
  need).

**Implementation status (2026-07-14 — steps 1–5 LANDED, gate open):**

1. ✅ `rng.rs`: `pcg4d_3r`, `Key::from_seed`, keyed fills, `CellStream` (iteration
   chain), KATs pinned to exact bit patterns.
2. ✅ Interpreter cut: `Runner::position(seed, lane)` replaces `reseed`; `Src`
   instructions carry `source_offset = RvId`; `chunk_seed` deleted (a chunk is a lane
   range).
3. ✅ `jit.rs`: inline pcg4d-3r (i32 ops), stateless `kernel(out, n, k0, k1, lane0)`
   ABI, stream machinery gone. **Conformance upgraded to bitwise**: interpreter and JIT
   batches are bit-identical across the whole RNG corpus (the lane-path `sin`/`cos`/`ln`
   now use the shared `approx` polynomials on every backend — `approx::ln_guarded` is
   the new full-domain twin).
4. ✅ `wasm_emit.rs`/`wasm_host.rs`: same swap; kernel ABI
   `kernel(out, n, key_lo, key_hi, lane0)`; `nz_kernel_seed` deleted (stateless kernel —
   the reused-instance leftover-state hazard is structurally gone); `unif_int` is exact
   split-multiply Lemire for `count < 2³⁹`; wasm conformance also bitwise vs interp.
5. ✅ Old `Rng` + `STREAMS`/`seed_state`/`choose_streams`/`latency_bound` deleted.
   Workspace green (440+ tests), clippy clean, wasm32 builds.

**Consumption amendment (perf-driven, same C0 words):** `normal` consumes the
Box–Muller pair over the *lane pair* `(2i, 2i+1)` — one hash (of the even lane), even
lane takes cos, odd takes sin (`rng::normal_pair`). Fill ranges always start on even
lanes (batch/chunk boundaries are multiples of 1024). Per-lane lowerings (JIT/wasm)
compute both branches and select by parity — bit-identical, and the shared trig
reduction computes both kernels anyway. `CellStream::next_normal` pair-caches likewise.

**Gate status (example_times, M4 Pro, `--features jit`): OPEN — 5177 ms vs 4351 ms
baseline (+19%).** Recovered by pairing: prisoners 961→982 (+2%), noise_colors 336→350
(+4%), am_vs_fm 405→435 (+7%). Still hot: turboquant 1428→2003 (+40%, interpreter,
Rotation/uniform-heavy), barrier_option 973→1131 (+16%, JIT normal-bound), clt_normal
~2× (JIT). Known levers, in order:
1. **Pair-unrolled JIT/wasm loop**: emit two lanes per iteration sharing the normal
   pair's hash+ln+trig (the parity-select interim recomputes them per lane) — should
   recover most of barrier/am_vs_fm/clt_normal.
2. **Interpreter fill vectorization**: check LLVM vectorizes the 3-round hash across
   lanes in `fill_uniform`/`fill_normal` (the rng-cert keyed-batch bench reached
   942 M u32/s; if the in-crate fills don't match that shape, restructure).
3. Track B's f32 halves per-uniform hash work again (one word pair → one word).
The gate is judged at C-landing after these — and after the generator verdict
(criterion 8) settles, since a swap re-prices everything.

## Sequencing

Order (owner's call, 2026-07-13): **C → B → A**, releasing after B.

0. **C0 (generator spike)** — DONE 2026-07-14 (fast battery + verdict: pcg4d-3r;
   PractRand deep runs pending — they must be clean before the numerics-v2 release).
1. **C (pcg4d-3r + the simplification)** — the smallest self-contained cut: swap the
   generator, delete the stream machinery (`STREAMS`/`seed_state`/`choose_streams`/
   `latency_bound`/`chunk_seed`), lanes stay f64 for now — each f64 uniform takes the
   consumed 24 bits of two words from one hash (2⁻⁴⁸ granularity; the low byte of a word
   is never consumed, per C0). Validates the counter design (determinism,
   chunk-as-lane-range) before the wider f32 surgery.
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
| pcg4d-3r is a custom variant with no published certification | Deliberate, owner-ratified trade after C0 disqualified published pcg4d: the C0 harness carries the full evidence burden (amended avalanche criterion clean, 7/7 stats, PractRand ≥ 1 TB pending), with certified Squares running as in-harness reference; a PractRand failure swaps to Squares before the seed-break |
| f32 transcendental polys diverge across backends | One shared `approx.rs` f32 table consumed by JIT, wasm emitter, and later the WGSL emitter — divergence is a compile error, not a drift |
| Interpreter throughput regresses on the per-lane hash | Bench says counter ≈ 4-stream xoshiro before SIMD; corpus-neutral gate enforces it |
| Async borrow/stack surprises | Scoped by call-graph walk; `MAX_EVAL_DEPTH` test guards the poll stack (see Track A traps) |
