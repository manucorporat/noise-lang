# Why Noise is fast

Noise is a Monte-Carlo language: almost every interesting program ends in "evaluate this
expression over a few million random draws" (`P(...)`, `E(...)`, `Var(...)`, `Q(...)`, a histogram).
The whole performance story is about making that one loop cheap — on native hardware *and* in the
browser — without ever sacrificing correctness or determinism.

This document explains what we built, why it's fast, and how it stacks up against hand-written
native Rust. All numbers were measured on an **Apple M4 Pro (10 performance + 4 efficiency cores),
rustc 1.96, release builds, Node 22 / V8 for the WASM figures**. They're a single machine and a
single run — read them as *ratios and orders of magnitude*, not benchmark-suite gospel. Reproduce
commands are at the end.

---

## TL;DR

- **The simplest possible program is already maximally optimized.** A one-line `P(...)` is
  automatically constant-folded, CSE'd, fused into a single machine-code kernel, given a latency-hidden
  multi-stream RNG, reduced with a vectorized power-sum pass, and **fanned out across every core** — no
  flags, no annotations, no rewrite. You'd have to hand-write a fused, SIMD-aware, multi-threaded
  Monte-Carlo kernel to match it, at which point you've reimplemented the backend.
- That one-liner sustains **~5.8 billion samples/sec** on a 14-core M4 Pro (`pi` Monte Carlo,
  generate + reduce, all cores) — and it scales **~9.6×** from one core to all of them.
- The hot loop never walks an AST. Random variables compile to a **graph IR** lowered several ways: a
  **columnar batch interpreter** (portable default + correctness oracle and the fallback floor), a
  **native WebGPU backend** (dispatches the fused kernel across GPU lanes — the native performance
  path), and a **WASM emitter** (the same fused kernel, for the browser).
- **vs native Rust:** per core, the generated kernel is within **~1.15× of hand-written,
  LLVM-compiled Rust** (~87% of "as fast as you could write it by hand"). But because Noise
  parallelizes the one-liner automatically and the typical hand-written kernel is single-threaded, the
  Noise program *beats* hand-written single-threaded Rust — and trounces the naive `rand`-crate loop a
  developer actually writes.
- **In the browser**, the emitted WASM kernel runs the *same* fused loop at **~0.5–0.75× of native
  codegen speed** in V8 — hundreds of millions of samples/sec from a sandbox, client-side.
- A profitability gate guarantees codegen is only used where it wins; otherwise the interpreter runs.
  **The backend choice only ever changes speed, never results**, and results are bit-identical across
  thread counts.

## The point: you write trivial code and get an expert kernel

The thing to internalize is *what you didn't have to do*. Write `X ~ unif(-1,1); Y ~ unif(-1,1); 4 *
P(X^2 + Y^2 < 1)` and the engine, with zero tuning, gives you:

| You'd otherwise hand-write… | Noise does it automatically |
|---|---|
| a fused kernel keeping intermediates in registers | graph → one machine-code/WASM loop |
| an inlined, fast, seedable PRNG | xoshiro256++ inlined into the kernel |
| latency hiding for the RNG dependency chain | 4 interleaved independent streams |
| a vectorized moment accumulator | 8-lane power-sum reduction |
| a work-stealing thread pool + a deterministic merge | automatic, bit-identical across core counts |
| a SIMD-friendly data layout | columnar batches |

The honest framing of "slower than Rust" is narrow: it means **one Noise-generated core is ~13%
behind one core of hand-tuned, already-fused LLVM Rust**. But nobody writes that Rust for a one-off
estimate — they write a single-threaded loop with the `rand` crate. The Noise one-liner runs fused
*and* on all 14 cores, so in practice it is several times faster than the Rust a person actually
writes, and competitive with the Rust an expert spends an afternoon on.

---

## The problem

Interpreting an AST once per draw is pointer-chasing death: for `X^2 + Y^2 < 1` you'd re-walk the
tree, re-dispatch on every node, and re-box intermediate values a million times. The entire design
exists to avoid that.

## The architecture: one IR, several backends

`~` and the distribution constructors build an **`RvGraph`** — an append-only DAG of nodes (`Src`,
`ConstNum`, `Unary`, `Binary`, `Select`) with structural sharing (so `X + X` is *one* draw of `X`,
not two). That graph is the single portable representation. Everything downstream is a *lowering* of
it — not a re-interpretation, code generation:

| Backend | Target | Where it runs | Role |
|---|---|---|---|
| Columnar interpreter | flat bytecode over column registers | everywhere | default, correctness oracle, and fallback floor |
| WebGPU (`gpu` feature) | WGSL compute shader | native (via wgpu) | fused kernel across GPU lanes — native performance path |
| WASM emitter (`wasm_emit`) | WebAssembly bytes | browser (via JS host) | fused kernel, portable |

A shared module (`kernel.rs`) holds the one copy of *what the graph means* — the cost model, the
multi-stream policy, the RNG seeding layout — so the code generators are thin "how to spell it on
this target" layers (`f64.mul` vs WGSL's `*`; the same inlined RNG step either way) and can never
drift apart.

The default forcing path is `compile_root`: simplify the graph once, then pick the lowering for the
target (`wasm32` → WASM host; otherwise → interpreter). On native, the WebGPU backend hooks a level up
in `reduce` and takes the forcing when its cost gate accepts; every codegen path falls back to the
interpreter for anything it can't profitably emit.

> **History:** the native performance backend used to be a **Cranelift JIT** (the `jit` feature),
> which fused each expression into one native kernel. It has since been retired in favour of the
> WebGPU backend; the JIT benchmark numbers quoted later in this document are kept as *historical*
> measurements of that retired backend, and the "vs hand-written Rust" comparison it enabled.

---

## Why it's fast — the techniques, each with its measured win

### 1. Columnar batch evaluation (the interpreter baseline)

The graph compiles to flat bytecode over a register file of **columns** (`BATCH = 1024` lanes each).
The whole batch runs through one instruction before moving to the next, so the inner loop is a tight,
cache-friendly, auto-vectorizing pass over contiguous `f64`s — one `match` dispatch per *1024* draws
instead of per draw. This is already fast and is the floor everything else builds on.

### 2. Graph simplification (free, once per compile)

Before any backend, `simplify` constant-folds, applies finite-safe identities (`x+0`, `x*1`, `x^0`,
double `-`/`!`, …), and hash-conses common subexpressions (CSE) into a smaller DAG. Fewer nodes →
fewer hot-loop ops and (for the interpreter) fewer materialized columns. Node-count reduction scales
with redundancy: ~0% for `dice`, ~10% for a clean polynomial (a repeated `X*X`), up to ~67% for an
identity-laden expression. Risky identities (`x*0`, `x/x`) are deliberately *excluded* — they're
wrong for a non-finite lane a user could construct.

### 3. Kernel fusion (the codegen win)

The interpreter materializes every intermediate column to memory. The code generators don't: they
emit **one loop that draws its sources, computes the entire expression keeping intermediates in
registers, and stores only the root**. On arithmetic-dense graphs that eliminates almost all the
memory traffic — which is exactly where fusion pulls away (see `poly_deep` below).

### 4. Inlined PRNG *and transcendentals*

xoshiro256++ / SplitMix64 is **inlined directly into the generated code** — no per-lane call back
into Rust. The kernel's `next_u64` is a handful of shifts/xors/rotates with zero call overhead, on
native and in WASM alike (WASM even has a native `i64.rotl`).

The same logic applies to `ln`/`sin`/`cos` — the hot path of `normal` (Box–Muller is `ln`+`cos`),
`exp`/`geometric` (`ln`), and the signal trig. Those were `extern "C"` calls to libm; the code
generators emit them as **straight-line polynomial approximations** (`crate::approx`: an atanh series for
`ln`, fdlibm minimax kernels with Cody–Waite range reduction for `sin`/`cos`, all bit-for-bit
documented and ~1e-9 vs libm — far tighter than Monte-Carlo noise, so the draws stay
distribution-identical to the interpreter oracle). Dropping the call roughly **doubles** a
transcendental-bound kernel:

| `normal_poly` kernel (single thread, M samples/s) | s1 | s2 | s4 |
|---|---:|---:|---:|
| libm call | 53 | 52 | 50 |
| inlined polynomial | **105** | 106 | 107 |

The call *was* the bottleneck. The **WASM emitter inlines the same polynomials** (`crate::approx` is
the shared reference; the wasm kernel spells them in `f64.*`/`i64.*` with `i64.reinterpret_f64` for
the bit-surgery and `f64.nearest` for the range reduction), so the win reaches the browser too — and
there it's worth even more, since the call it replaces was a per-draw crossing of the JS boundary, not
just a libcall.

### 5. Multi-stream RNG (latency hiding — the non-obvious one)

xoshiro is a *serial dependency chain*: each `next_u64` waits on the previous one mutating the state,
and because the state threads through every loop iteration, the **whole loop is one chain**. On
RNG-bound graphs that latency — not the arithmetic — is the ceiling. So the kernel runs **4
independent xoshiro streams**, emitting 4 samples per iteration; the out-of-order core overlaps the
independent chains. This is the *scalar* form of SIMD, and it's the trick that actually won where a
hand-rolled `f64x2` NEON kernel lost.

Single-thread fused kernel, M samples/sec, by stream count:

| case | 1 stream | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| pi_indicator | 517 | 690 | **720** | 554 |
| dice_sum | 506 | 777 | **891** | 645 |
| poly_deep | 853 | 907 | **921** | 706 |
| normal_poly | 105 | 106 | **107** | 103 |

Multi-stream buys ~1.4× (pi) to ~1.75× (dice) on RNG-bound graphs; `poly_deep` is already
arithmetic-bound (one draw, deep math) so it barely moves. `normal_poly` is **flat** — and this is the
subtle part: even with `ln`/`cos` now inlined (no calls to serialize the lanes), the long polynomial
makes it *arithmetic-throughput*-bound, not RNG-*latency*-bound. The execution units are already
saturated, so extra streams just multiply the work with nothing to overlap. The stream count is
therefore **graph-aware**: multi-stream only for latency-bound graphs (pure `uniform`/`uniform_int` +
arithmetic); everything with a transcendental draw, ufunc, or call stays single-stream. 4 is the
sweet spot before register pressure bites (note the regression at 8).

### 6. Power-sum reduction (the hidden bottleneck)

Computing moments over the samples was secretly the slowest stage — a streaming Welford update has a
per-element divide that ran *slower than generation*. We rewrote it as raw power sums (count, Σx,
Σx²) across 8 unrolled lane accumulators, which removes the divide and auto-vectorizes: **~9.5× on
the reduction step**, turning it from the ceiling into a rounding error. Without this, none of the
codegen wins above would be visible end-to-end.

### 7. Profitability gate (never lose)

A graph still dominated by a *real* per-draw call is faster on the interpreter's vectorized column
fill than as a fused kernel. So a cost model (`fusible > libcalls`) decides emit-vs-interpret per
graph; `Poisson` (a variable-length per-lane loop) always stays interpreted. The result: codegen is
used only where it wins, and correctness is never at stake.

The gate is **one cost function, parameterized by whether the backend inlines transcendentals** —
both backends now do (technique 4), so both pass `inline_trans = true` and `normal`/`exp`/trig graphs
are fusible on each. `atan`/`round`/non-integer `pow` remain real calls on both and still count
against codegen. (The parameter earns its keep as the honest seam: if a backend ever *couldn't*
inline a transcendental, it would pass `false` and the gate would correctly leave those graphs to the
interpreter rather than emit a per-draw call.)

### 8. Deterministic multicore (free, automatic)

Sampling fans out across cores with a work-stealing chunk loop, and the per-chunk accumulators merge
as an **exactly-associative monoid in chunk-index order** — so results are **bit-identical regardless
of thread count** (and reproducible across runs from a seed). Speed scales with cores; the answer
doesn't change. This is the part that makes a one-liner beat hand-written code: you get the whole
machine for free, deterministically, without writing a single line of concurrency.

End-to-end `moments(pi)` (generate + reduce), 1 thread vs all 14 cores, M samples/sec:

| backend | 1 thread | 14 threads | scaling |
|---|---:|---:|---:|
| interpreter | 151 | **1662** | 11.0× |
| JIT (retired, historical) | 605 | **5824** | 9.6× |

(Measured at a high sample count so per-call thread-spawn is negligible. At the default `P()` budget
of 1e6 samples the work is too small to amortize spawning 14 threads, so scaling there is modest — the
parallel win shows up on the high-precision queries that actually need it.)

---

## End-to-end: interpreter vs the retired JIT (native, historical)

Full `P()`-style workload: 1,000,000 samples + moments reduction, multicore, median M samples/sec.
These figures were measured against the now-retired Cranelift JIT; they still illustrate *which graph
shapes* codegen wins on (the same shapes the WebGPU backend now targets), which is why they're kept.

| case | interpreter | JIT (retired) | speedup | notes |
|---|---:|---:|---:|---|
| pi_indicator | 167 | **1226** | 7.3× | RNG + arithmetic, fully fusible |
| poly_deep | 184 | **1061** | 5.8× | deep math, fusion kills column traffic |
| poly_wide | 188 | **483** | 2.6× | many draws, register-pressure-bound |
| normal_poly | 144 | **303** | 2.1× | normal feeding a deep chain — fusion beats the `ln`/`cos` |
| dice_sum | 1404 | 1494 | 1.06× | already RNG-bound; interpreter's column fill is excellent |
| normal_sum | 245 | 245 | — | transcendental-bound → gate keeps the interpreter |
| exp_tail | 656 | 656 | — | one `ln` + compare → gate keeps the interpreter |

The big wins are on exactly the graphs where fusion removes memory traffic; where the interpreter is
already optimal (RNG-bound `dice_sum`) or codegen wouldn't help (transcendental `normal_sum`,
`exp_tail`), the gate transparently leaves you on the interpreter, so there's no regression.

(This table isolates the *fusion* win at the 1e6 default budget, where multicore is spawn-limited —
so these are roughly the all-cores-but-overhead-bound figures. On the high-precision queries that run
many more samples, the multicore scaling above kicks in on top, which is how `pi` reaches ~5.8 G/s.)

---

## The other end of the spectrum: a 20,000-node fused kernel (TurboQuant)

`pi` is a tiny kernel. The opposite extreme is `examples/turboquant.noise`, which reproduces a
linear-algebra quantization paper by Monte Carlo: **every sample builds a fresh random orthonormal
rotation of a d=20 vector** (modified Gram–Schmidt over d² = 400 Gaussian draws), quantizes the
rotated coordinates, and rotates back.

This is where Noise's matrix story matters. `@`, `matvec`, `rotation`, `transpose` carry **no GEMM
loop** — they expand element-by-element into the same scalar `RvGraph`. So the entire d×d linear
algebra of one sample collapses into a single straight-line fused kernel:

| estimator | fused kernel | interpreter | JIT (retired) | speedup | estimate (paper) |
|---|---:|---:|---:|---:|---|
| D_mse  b=4 (rotate · quantize · rotate back) | 20,391 nodes | 0.12 | **0.63** | 5.3× | 0.0087 (0.009) |
| D_prod b=4 (MSE stage + 1-bit QJL residual) | 21,983 nodes | 0.12 | **0.31** | 2.6× | 0.0023 (0.0024) |

(The codegen column above is the retired JIT; the WebGPU backend now carries this workload on native.)

Two things to read here. First, the estimates land on the paper's table — the kernel is correct, not
just fast. Second, **a ~20,000-node kernel still fuses, compiles, and runs across all cores with zero
special-casing**: CSE collapses the shared Gram–Schmidt sub-expressions, the profitability gate emits
it, and the multicore reducer parallelizes it — the same pipeline as the two-node `pi` kernel.

The flip side is the honest caveat on the matmul question: because there is no loop, the kernel size
grows with the matrix (here O(d²)–O(d³) nodes for d=20). At small d this is *ideal* — everything lives
in registers, no cache traffic, no loop overhead, and you sample across cores for free. It is **not** a
cache-blocked GEMM and would not be the right tool for large dense matrix multiply, where data reuse
and tiling — an inner loop the straight-line kernel deliberately doesn't have — are the whole game.
Noise optimizes the *Monte-Carlo* axis (millions of independent samples of a fixed expression), not the
*linear-algebra* axis (reuse within one large matrix product).

Run it: `cargo test -p noise-core --features gpu --release -- --ignored --nocapture bench_turboquant`

---

## How the codegen compared to hand-written native Rust (historical, retired JIT)

This section measured the **retired Cranelift JIT**, kept because it establishes the codegen-quality
ceiling the project reached on native CPU. The honest ceiling: race the Cranelift kernel against a
hand-written, LLVM-compiled Rust loop computing the *identical* graph (same inlined xoshiro, same
arithmetic, both filling a column so the comparison is pure codegen quality):

```
fused poly_deep kernel, single thread, M samples/sec:
  cranelift (Noise JIT)  910      llvm (hand-written Rust)  1047      llvm/jit  1.15×
```

**Cranelift produces code within ~15% of hand-written Rust** — and you got there by typing
`X*X*X*X*X + 2*X*X*X*X - ...` instead of writing a SIMD-aware Monte-Carlo kernel by hand. The ~1.15×
gap is structural: Cranelift is a lightweight, fast-compiling, WASM-clean backend, not LLVM with its
heavyweight optimizer. For a teaching language that compiles kernels at runtime, that's the right
trade — which is why we did **not** pursue an LLVM JIT.

But "within 15% of Rust" understates the practical picture, because that race is **one core vs one
core**, against Rust that is *already fused and SIMD-aware*. Two things flip it in Noise's favour for
real code:

- **The comparison Rust isn't the Rust people write.** The hand-written baseline above is an expert
  kernel: manually inlined xoshiro, hand-unrolled polynomial, column-filling for cache behaviour. The
  Rust a developer reaches for is a `for` loop with the `rand` crate that allocates and boxes — far
  slower. Noise emits the expert version from one line.
- **Noise uses every core; the hand-written kernel above is single-threaded.** Folding in the measured
  9.6× multicore scaling, a one-line Noise `P()` runs at ~0.87 × 9.6 ≈ **~8× a hand-written
  single-threaded Rust kernel** — and to beat *that* you'd have to add a deterministic work-stealing
  reducer to your Rust too. At that point you've rebuilt Noise's backend by hand.

So: per-core, Noise is a hair behind hand-tuned Rust. Per *program you'd actually write*, the trivial
Noise version wins — that's the whole point of pushing the optimization into the language.

---

## WASM vs non-WASM (the browser story)

A WASM sandbox can't emit or run native machine code or reach the GPU, so the native performance
backend is native-only. The browser's equivalent is the **WASM emitter**: it walks the same graph and emits a WebAssembly module
(`kernel(out, n, state)` over its own linear memory, same multi-stream interleave, the same inlined
xoshiro/Box–Muller draws spelled in `f64.*`/`i64.*` — including the inlined `ln`/`sin`/`cos`
polynomials, so a `normal` kernel makes **no host calls at all**). A tiny JS host
`WebAssembly.instantiate`s it and drives it; only `atan`/`round`/`pow` are still imported from the
host (`Math.*`). Emitted kernels stay small — a single-stream `normal` kernel is ~940 bytes, well
under the 4 KB main-thread sync-compile limit.

The emitted kernel, timed in V8 (the browser's engine), single thread:

| case | WASM in V8 | native codegen (retired JIT) | WASM / native |
|---|---:|---:|---:|
| dice_sum | 587 | 890 | 0.66× |
| pi_indicator | 531 | 718 | 0.74× |
| poly_deep | 446 | 919 | 0.49× |

So the browser runs the *same fused, multi-stream kernel* at roughly **half to three-quarters of
the (then-JIT) native codegen speed** — hundreds of millions of samples/sec, entirely client-side,
from ~1.3–1.6 KB of generated WASM per program. The gap vs native was the cost of the WASM sandbox +
V8's JIT vs Cranelift's; it's remarkably small for what you get (no install, no server, runs in a tab).

Two practical notes:
- These WASM figures are **single-thread** (the host runs one instance on the main thread). Native
  end-to-end numbers above are multicore. A Web Worker pool would close much of that gap; it's the
  obvious next browser optimization.
- The host is **content-addressed** (instances are cached by kernel bytes with an LRU cap), so
  re-running the same program is free of recompilation, and it falls back to the interpreter if a
  module fails to instantiate (e.g. the main-thread 4 KB sync-compile limit — real kernels are well
  under it).

Either way, the same `Backend`/`Program`/`Runner` seam is used on every target, so the browser path
gets simplification, fusion, multi-stream RNG, the profitability gate, and the power-sum reduction
for free — the browser is not a second-class citizen.

---

## What we deliberately did *not* do (and why)

Honest negative results — each was implemented and/or measured, then rejected:

- **Vectorized SIMD codegen (`f64x2` / NEON).** Built it in Cranelift, measured it, *removed* it —
  then re-probed the question from scratch with a hand-written, correctly-instruction-selected NEON
  kernel (`crates/noise-core/tests/simd_probe.rs`) to be sure the removal wasn't just a
  backend-lowering artifact. The probe's finding is more nuanced than "scalar always wins": on Apple
  Silicon (M4 Pro), hand-written NEON *does* win at the extremes — **1.27× on pure RNG** (`rng_only`)
  and **1.15× on the FP-throughput-bound class** (`poly_thru`, the `normal_poly`-shaped regime) — but
  it **loses ~13–16%** on the RNG-bound and mixed graphs Noise actually runs (`pi` 0.87×, `dice`
  0.84×) and is flat on deep arithmetic (`poly_deep` 0.99×). The loss in the middle is a
  *port-contention* signature — the vector RNG and the vector FP work compete for the same four NEON
  pipes, while the scalar kernel runs its integer RNG and its FP on disjoint port sets that the
  out-of-order core overlaps for free — not "SIMD is slow". So on the graphs that dominate real
  programs, multi-stream *scalar* RNG (technique 5) wins, and the vector path stays removed. A vector
  path *would* buy up to ~1.15× on the transcendental/FP-bound class, but Cranelift can't currently
  emit the instruction selection that earns it and the prize is bounded to that one graph class — so
  vectorizing `approx::{ln, cos}` is a live but low-priority follow-up, not dormant headroom being
  left on the table everywhere.
- **LLVM JIT.** Would close the ~1.15× codegen gap but is a heavy, slow-to-compile, non-WASM-clean
  dependency. Wrong trade for a teaching language that JITs at runtime and must also ship to the
  browser.
- **Lemire integer sampling on the critical path.** Replaced the float round-trip in `unif_int` with
  Lemire's multiply-high. It's cleaner, but measured *perf-neutral* — the kernel is xoshiro-latency-
  bound, so the float ops were already hidden by the out-of-order core. (The WASM emitter uses the
  float method anyway, since WASM lacks a 64×64→128 high-multiply; it's identical in distribution.)

---

## Reproduce

```sh
# End-to-end (sample + reduce, multicore) — samples/sec per case:
cargo bench -p noise-core --bench sampling             # interpreter (the floor)
cargo bench -p noise-core --features gpu --bench sampling  # with the WebGPU backend enabled

# Multicore scaling (1 thread vs all cores, generate + reduce) — the "uses every core for free" number:
cargo test -p noise-core --release -- --ignored --nocapture --test-threads=1 bench_parallel_scaling

# Reduction-pass A/B (power-sums vs streaming Welford):
cargo test -p noise-core --release -- --ignored --nocapture --test-threads=1 bench_reduce_absorb

# Large fused-kernel stress test (TurboQuant: a ~20k-node per-sample rotation kernel):
cargo test -p noise-core --features gpu --release -- --ignored --nocapture bench_turboquant

# WASM: dump an emitted kernel, then time it in V8 (see the dump_kernel test in wasm_emit.rs):
NOISE_KERNEL_OUT=/tmp/k.wasm NOISE_KERNEL_SRC='use rand; X ~ unif(0,1); X*X*X*X*X - 5*X + 6' \
  cargo test -p noise-core --release -- --ignored dump_kernel
# (drive /tmp/k.wasm through a JS host: instantiate, seed state at byte 0, call kernel(4096, N, 0),
#  read N f64s at byte 4096 — exactly what crates/noise-core/src/wasm_host.rs does in the browser.)
```

The single-thread fused-kernel micro-benches (`bench_streams`, `bench_cranelift_vs_llvm`) lived in the
retired JIT backend and no longer exist; their results are preserved above as historical context.

Numbers will vary with machine, thermal state, and load — the *ratios* (codegen vs interpreter, WASM
vs native) are the durable story.
