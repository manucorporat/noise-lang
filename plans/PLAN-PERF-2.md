# PLAN-PERF-2.md — second performance pass: what shipped, what's left

Follow-on to the work recorded in `PERF.md` (the first pass: columnar interpreter, Cranelift JIT,
WASM emitter, multi-stream RNG, power-sum reduction). Conducted 2026-07-13 against `master`
(07e8a37).

Every number below marked **[measured]** was produced by actually running the thing on this machine
(Apple M4 Pro, 14 cores; Node 22 / Chrome for the browser figures), not inferred. Claims that were
*assumed* and turned out false are called out explicitly, because two of them were load-bearing.

---

## Executive summary

The first pass optimized the sampling hot loop and measured it with hand-written micro-kernels. That
was the wrong instrument, and it hid three things:

1. **The benchmarks didn't measure real programs.** `benches/sampling.rs` times one RV expression:
   compile once, draw a million samples. Real programs force *many* queries with *few* draws each
   (`noise_colors`: 14 forcings × 3,000 draws; `turboquant`: 10 × 10,000). In that regime the JIT
   never amortizes its compile — it was a **4.5× pessimization on `noise_colors`** [measured] — and
   no micro-benchmark could see it.
2. **The JIT was silently returning wrong numbers.** `E[cos X]` came back `0.495` instead of `0.607`,
   and `E[S]` came back `NaN`, non-deterministically, under CPU load. A use-after-free
   (`JITModule::free_memory()` unmapping executable pages that were then recycled under still-running
   kernels). On `master`, invisible to the test suite.
3. **The browser — the target almost everyone uses — was never benchmarked at all.** When it finally
   was, the conclusion inverted: the WASM emitter is worth **2–7×** on real programs, while the
   native JIT only wins on trivial ones. And the browser was leaving a **6–8× threading win** on the
   floor.

The second pass fixed those. What remains is concentrated in one structural gap (introspection and
plots bypass the backend seam entirely — **since closed, see item 1**), two feature gaps in codegen
(`Gather`, `Poisson`), and a language-level graph-blowup problem that makes `turboquant` the
slowest program we have.

---

## Part 1 — What this pass landed

**Status: implemented and green in the working tree, NOT committed.** 415 JIT tests, 403 interpreter
tests, clippy clean, TS typechecks. Whoever picks this up should commit it in coherent slices first;
suggested split is at the end.

### P0 — `master` did not build [verified]
Commit 07e8a37 bumped `workspace.package.version` to 0.2.0 but left
`workspace.dependencies.noise-core` pinned at `0.1.2`. Nothing depending on `noise-core` (the CLI,
`noise-wasm`) resolved. Fixed in `Cargo.toml`. **Worth its own commit** — it's unrelated to the perf
work and it blocks everyone.

### P0 — JIT use-after-free [verified]
`JitProgramInner::drop` called `JITModule::free_memory()`, unmapping that module's executable pages.
In a process that compiles and drops many kernels (a REPL, a server, a parallel test binary) those
pages were recycled by the next module's mmap **while other threads were still executing kernels**.
Symptom: wrong values and NaNs, only under CPU load, only on the JIT.

Rust's lifetimes never caught it and never could: they prove no runner of the *freed* module
survives, which is true and beside the point. `free_memory`'s safety contract is about the whole
process's JIT code, not one module's borrows.

* Repro: `cargo test --release -p noise-core --features jit --test signals` under CPU load —
  **25/25 failures before, 0/12 after** [measured].
* Fix: don't free; the module lives as long as the program.
* **Known cost, deliberately accepted:** this reverses the old "finding C4". A long-lived process now
  retains a few KB per *distinct* compiled kernel. A bounded leak beats silently wrong answers, but
  **the real fix is to stop churning modules — see item 4 in Part 2 (compile cache).**
* **No unit-level regression test exists.** One was written; it passed even with the bug reinstated.
  Narrower repros don't trigger it either — which is exactly what a use-after-free looks like
  (whether the stale page has been handed out again is an allocator accident). An unreliable test
  would imply coverage that isn't there, so the working repro is documented in the code instead.
  **If someone can build a deterministic repro, that is genuinely valuable.**

### P1 — Benchmarks that measure real programs
* `crates/noise-core/benches/examples.rs` — every `examples/*.noise`, end-to-end through
  `run_to_document` (the same path `noise <file>` uses): parse + eval + codegen + every forcing.
* `packages/core/bench/examples.mjs` — the same corpus in V8, through the shipped wasm build. **This
  is the one that matters**; nothing measured the browser before.
* `tests/bench_corpus.rs::example_shapes` — a diagnostic printing forcings and draws-per-forcing per
  program. This table is what explained every regression; look at it first when something is slow.

### P1 — Codegen amortization gate
`profitable()` decided emit-vs-interpret from the graph's *shape* alone; the draw count was never
passed in, so it happily compiled a 20k-node kernel to take 3,000 draws.

Now `compile_root(graph, root, draws)` threads the draw count through the `Backend` seam, and codegen
is declined below a minimum. **One constant per backend, not a cost curve** —
`jit::tests::bench_amortization` measures the true break-even per cone size (13k–75k draws over a
6→1,536-node sweep) and a fitted `f(nodes)` was tried, but it decided every program in `examples/`
exactly as a single threshold does. The curve was complexity with nothing to show for it.

* `MIN_DRAWS_JIT = 100_000`, `MIN_DRAWS_WASM = 10_000`. The 10× gap is real and measured:
  `WebAssembly.instantiate` on a ~1 KB kernel is far cheaper than a Cranelift compile. Reusing the
  native constant in the browser cost real wins (`am_vs_fm` 0.94× gated vs 1.20× emitted) [measured].
* Result: `noise_colors` 0.22× → 0.99×, `turboquant` 0.79× → 0.99× [measured].

### P1 — The browser inverts the backend story [measured]
| | interp | emitted | |
|---|---:|---:|---|
| `birthday` | 1524 ms | 199 ms | **7.6×** |
| `beta_bernoulli` | 778 ms | 114 ms | **6.8×** |
| `pi` | 45 ms | 7 ms | **6.3×** |
| corpus total | 18,186 ms | 14,006 ms | **1.30×** |

The *native* interpreter is LLVM-compiled and already fast, so Cranelift only beats it on trivial
graphs. The *WASM* interpreter is a bytecode VM inside wasm — slow — so the emitted kernel wins big.
**The WASM emitter is the backend that earns its keep; the native JIT is the marginal one.**

### P1 — Nothing runs on the main thread
`packages/core/src/{pool,worker}.ts`. The public API was already `async`, so it's a drop-in. A
persistent worker pool; the engine never touches the main thread. Verified: a 20M-draw `pi` run,
main thread ticked 67× during it (0 would mean a frozen tab) [measured].

### P1 — WASM threads: 6.2–8.3× in the browser [measured]
`reduce.rs` now has two executors behind one monoid: `std::thread::scope` natively, and rayon over a
Web Worker pool on wasm32 under the `wasm-threads` feature. **Wasm cannot spawn a thread itself** —
the threads proposal explicitly leaves thread creation to the embedder — so the pool is bootstrapped
from JS by `wasm-bindgen-rayon`.

* `pi` 6.23×, `normal_poly` 8.34×, **answers bit-identical** to the single-threaded build (the
  reducer merges chunks in index order, so thread count changes wall clock and nothing else).
* Two builds ship: `wasm/` (stable, universal) and `wasm-mt/` (nightly, `-Z build-std`).
  `pool.ts`/`worker.ts` feature-detect `crossOriginIsolated` and pick. **This is required**, not
  belt-and-braces: SharedArrayBuffer needs COOP/COEP, and a library cannot impose those headers on
  the app that installs it.
* `wasm_host.rs` had to change: a kernel handle is **not portable across workers** (each worker is a
  separate JS agent with its own `nz_kernel_*` registry — linear memory is shared, JS globals are
  not). Programs now carry the kernel *bytes* and instantiate on the thread that drives them.
  Without this, every worker would have silently fallen back to the interpreter, throwing away the
  emitter win exactly when threads were added.
* `packages/core/scripts/build-wasm.sh` documents all six link flags and what each one's absence
  breaks — each was hit in turn, and the failure modes are all unhelpful (a missing `--shared-memory`
  fails at *runtime* with `DataCloneError: #<Memory> could not be cloned`).

### Negative result — no SIMD emitter [measured]
Hand-written `f64x2` runs at **0.90–0.97×** of the existing 4-stream scalar kernel in V8, while
issuing strictly *fewer* instructions (4 vector xoshiro steps vs 8 scalar, for the same 4 samples).
This independently reproduces the NEON finding already in `PERF.md` (pi 0.87× native), for the same
reason: multi-stream scalar already harvests the ILP, on integer ports that overlap with the FP work
for free, whereas a vector kernel makes RNG and math contend for the same pipes.

A vector emitter would cost several hundred lines across `wasm_emit.rs` plus a vectorized `approx.rs`
and a second conformance oracle — to make the dominant graph class *slower*. Probe kept at
`tests/simd_probe_wasm.{rs,wat}` so this isn't re-litigated from assumption. (Its native sibling
`tests/simd_probe.rs` already existed and reaches the same verdict for NEON.)

**The one class not probed** is the transcendental / FP-throughput-bound one (`normal_poly`), where
the NEON probe *did* win 1.15×. That is the only place SIMD could pay, and 1.15× on one graph class
is a poor trade for vectorizing `ln`/`sin`/`cos`. Recommend leaving it.

### Also rejected — f32 mode
Scalar `f32` and `f64` arithmetic run at **the same speed** on x86-64 and ARM64 (same ports, same
latency), and `wasm_emit` is fully scalar, so there are no lanes for f32 to double. The real f32
gains would be indirect (two uniforms per xoshiro `u64`; shorter polynomials) — ~1.2–1.6× — against a
steep cost: 24-bit uniforms, a second `approx.rs` (it's binary64 *bit-surgery*, not a cast), a second
test oracle, ~800–1000 LOC, and **truncated tails** (the normal clips at ~5.5σ instead of ~8.5σ, so
tail-probability / VaR queries get *wrong* answers, not noisy ones). Revisit only if a lane-count
multiplier ever exists to hang it on.

---

## Part 2 — What's left, ranked

### 1. ~~Introspection and plots bypass the backend seam entirely~~ (DONE — third pass, 2026-07-13)
`sampler::for_each_joint_batch` called `bytecode::compile_roots` **directly**, never the backend
seam. So every multi-root query — `plot::scatter` (`sample_pairs`), `plot::fan` (`grid_draws`),
`plot::corr` (`corr_matrix`), `grid_moments`, `describe` — ran on the **interpreter,
single-threaded**.

**What was built.** The seam now has a multi-root twin at every level: `simplify_roots` (one shared
builder, so cross-root source sharing — the thing that makes a paired statistic correct — survives
the rewrite), `profitable_roots`/`choose_streams_roots`/`cone_size_roots` (one gate decision over
the *union* cone), `Backend::compile_joint` → `JointProgram`/`JointRunner` (default = the
multi-root bytecode interpreter, so any backend without a joint lowering is automatically correct),
a multi-output Cranelift kernel and a multi-column wasm kernel (both: one shared memo per stream —
joint draws by construction — each root stored to its own BATCH-strided column), and
`nz_kernel_run_cols` in the JS host. Pinned by exact-pairing tests (`b == 2a` lane-for-lane on
whichever backend the build selects) at both the sampler and the wasmi level.

**Measured first, as instructed — and the measurement re-ranked the work.** Timing
`for_each_joint_batch` per example (native interpreter): 5% of corpus total, but concentrated —
`nyquist` 100%, `dithering` 73%, `shor_period` 54%, `am_vs_fm` 47%. Probing the gate then showed
**most joint time wasn't sampleable work at all**: the dominant case is a *deterministic* vector
(`plot::line(signal::sample(...))`, a curve of already-forced `P()` results) reaching the joint
driver as k constant roots and being re-evaluated `n × k` times — 200k draws × 60 elements in
`birthday`, for values already known. Codegen correctly declines those (nothing fusible). The fix
is in the driver: **a union cone with zero RNG sources is clamped to one batch** (all lanes are
identical; quantiles/moments/correlations of a constant are that constant).

**Results** (V8, `bench/examples.mjs`, before → after): `nyquist` **143×** (114 ms → 0.8 ms — the
clamp), `dithering` 2.2×, `pi` 2.1×, `coin_streak` 2.6×, `irwin_hall` 1.9×, `am_vs_fm` 1.24×,
`kelly`/`birthday`/`st_petersburg` ~1.13×, corpus total **1.09×** (67.8 s → 62.5 s). Native
(Criterion, `--features jit`): `dithering` −26%, `kelly` −6.6% (significant); `am_vs_fm` unchanged
natively because its 40k draws/forcing sit under `MIN_DRAWS_JIT` — exactly the regime split the
two thresholds encode, and its browser win (1.24×) is the emitter earning its keep again.

**Also fixed here:** `benches/examples.rs` had no `[[bench]] harness = false` entry, so it ran
under libtest and **Criterion never executed** — the bench "passed" while measuring nothing. Any
number anyone believed came from it came from somewhere else. It measures now.

### 2. `Gather` and `Poisson` hard-reject codegen (P1)
`kernel::walk_cost` returns `false` for both, so the whole cone falls to the interpreter.

* `prisoners` — **the slowest example in the browser (3.6 s)** [measured] — is ~5,000 `Gather` nodes
  from the random-index `boxes[box]`. It gets *nothing*: no emitter, no threads.
* `bootstrap` too (`rand::empirical` and `rand::block_bootstrap` both lower to `Gather`).
* A gather is a bounded, clamped indexed load over a known table. Both Cranelift and wasm can emit it
  (a select chain for small tables, an indirect load for large). Both emitters currently have
  `unreachable!("profitable() excludes Gather")`.
* `Poisson` (a variable-length per-lane loop) is a genuinely poor codegen fit and **no example uses
  it** — leave it interpreted; the reject is currently dead weight w.r.t. the corpus.

### 3. Eval-time graph blowup (P1 — a language problem, not a backend one)
`turboquant` is **6.5 s in the browser** [measured] and the sampler is not the bottleneck.
`rand::rotation(d)` expands to O(d³) nodes and `permutation(n)` to O(n²) — **before a single draw**.
Its cone is ~17,500 ops *per sample* (1.75B ops total at 100k draws) and it exceeds
`MAX_CODEGEN_NODES`, so codegen declines it anyway. `prisoners` builds ~500k `RvId`s at eval time.

`@`, `matvec`, `rotation`, `transpose` carry no loop — they expand element-by-element into scalar
graph nodes. That is *ideal* at small `d` (everything in registers) and quadratically wrong as `d`
grows. **Structured nodes** (a `MatMul` / `Sort` / `Rank` node, or a loop node) instead of unrolling
would collapse this. Sizeable; touches `eval/library.rs`, `dist.rs`, and every backend.

### 4. No compile cache (P1 — and it bounds the JIT leak)
Every forcing recompiles its cone from scratch: `noise_colors` compiles 14 kernels, `kelly` 13. A
per-`Engine` cache keyed by the simplified graph would cut that, **and it is the right fix for the
memory the JIT now leaks** (see P0 above): stop churning modules rather than reinstate the unsafe
free. A REPL re-running the same program would then compile once, not once per run.

### 5. `sqrt` is charged as a libcall (P2 — cheapest item here)
`sqrt` lowers to `Binary(Pow, x, 0.5)`, and non-integer `pow` counts as a **libcall** in the cost
model — so it argues *against* codegen — even though both Cranelift and wasm have a native
`f64.sqrt` instruction. Add `UnOp::Sqrt` and most of `turboquant`'s and `am_vs_fm`'s libcalls become
single fused instructions. Same story for `UnOp::Exp` (the JIT lowers it to a `pow(e, x)` call).

### 6. Quantiles don't parallelize (P2)
`Q(...)` goes through `sample_n` → `for_each_batch`, which is sequential — it never touches
`run_reduction`, so it gets none of the threading. `P`/`E`/`Var` do. (A quantile is an order
statistic, so it isn't a monoid — but the *sampling* still parallelizes; only the final sort must be
centralized.)

### 7. Recursive emitters → the `MAX_CODEGEN_NODES` cliff (P2)
`MAX_CODEGEN_NODES = 20_000` exists **only** because `jit::emit_node` and `wasm_emit::emit_node` are
recursive; the cost walkers (`walk_cost`, `latency_bound`, `cone_size`) are already iterative. The
cap is what declines `turboquant`. Making the emitters iterative removes the cliff *and* a thin
stack-safety margin (a 6,144-node chain overflowed a 2 MiB test-thread stack; real use is guarded by
eval's 2048-level recursion budget on an 8 MiB main stack, so this is latent, not live).

### 8. Decision: does `noise-cli` ship the JIT? (P2)
`noise-cli` has **no `jit` feature at all** — every CLI user is on the interpreter today. Now that
the use-after-free is fixed it is safe to enable. But it only wins on trivial programs (4.4× on
`beta_bernoulli`, break-even or worse on complex ones), so this is a real call, not an oversight.
Given the browser is the primary target, "leave it off and keep the dependency surface small" is a
defensible answer.

---

## Measurement gaps to close first

Two cheap things that would re-rank the list above:

1. **The 1.30× WASM corpus figure predates threads.** The full `examples.mjs` corpus has never been
   run against the *threaded* build. The real end-to-end browser number is unknown. **Still open.**
   (The single-threaded corpus number is now 62.5 s after item 1 — that's the fresh baseline to
   compare the threaded build against.)
2. ~~The joint/introspection path has never been timed.~~ **Closed** — see item 1: 5% of corpus
   total on the native interpreter (concentrated: `nyquist` 100%, `dithering` 73%, `am_vs_fm` 47%),
   and its dominant cost was deterministic vectors being re-sampled, not missing codegen. Both
   fixed. With the joint path off the table, the corpus is now overwhelmingly items 2 and 3:
   `turboquant` (26.5 s in V8) + `prisoners` (15.8 s) are **two-thirds of the corpus total** by
   themselves.

Also worth fixing: `packages/core/bench/examples.mjs` takes >10 min over the full corpus (5 reps ×
31 programs). Trim reps or subset it, or nobody will run it. (A quick per-example wall-time
diagnostic now exists natively: `bench_corpus.rs::example_times`.)

---

## Suggested commit split for the working tree

1. `fix: workspace noise-core dep pinned to 0.1.2 while the crate is 0.2.0` (unblocks all builds).
2. `fix(jit): don't free module memory on drop — it was a use-after-free` (+ the honest note about
   the leak it trades for).
3. `bench: measure real programs, not micro-kernels` (`benches/examples.rs`,
   `packages/core/bench/examples.mjs`, `example_shapes`).
4. `perf(codegen): decline codegen below a per-backend draw threshold` (the amortization gate).
5. `feat(wasm): run the engine off the main thread in a worker pool`.
6. `feat(wasm): multi-threaded reduction over a Web Worker pool` (incl. `wasm_host` per-thread
   instantiation, `build-wasm.sh`, COOP/COEP headers, nightly in CI).
7. `test: wasm SIMD probe — f64x2 loses to multi-stream scalar` (negative result).

`PERF.md` needs updating alongside: its "backend choice only ever changes speed, never results" claim
was false until the JIT fix, and its browser section predates threads.
