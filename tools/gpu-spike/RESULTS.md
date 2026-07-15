# G0 — the WebGPU spike (PLAN-WEBGPU)

**Date:** 2026-07-14 · **Device:** Apple M4 Pro (Metal, integrated) · **Harness:** `tools/gpu-spike`
(`cargo run --release`) · **Verdict: GO**, with one design constraint that reshapes G1.

G0's job was to kill the plan or scale it, by answering the questions no amount of reading could:
does the certified RNG survive a language with no `u64`, do the giant shaders compile, what does a
dispatch cost, and — underneath all of it — are the GPU's draws actually *our* draws.

Everything below is measured, and the harness re-measures it on demand. Where a number moved under
scrutiny, the write-up says so; three of them did, and two of my own early numbers were wrong.

---

## The five findings

### 1. The certified RNG survives WGSL, bit for bit

WGSL has no `u64`, and `squares64` is five 64-bit wrapping multiplies. Emulated on `vec2<u32>`, it
reproduces `noise_core::rng` **exactly** — 4096/4096 lanes, for both the pair-shared draws
(`unif`/`normal`/`exp`) and the per-lane `unif_int` draw.

Two structural gifts made this far cheaper than the plan feared:

* `rotate_left(x, 32)` on a u64 is a **half-swap** — `x.yx` in WGSL. Free.
* A 64×64 *wrapping* multiply only needs the low 64 bits, so three of the four partial products
  collapse. One wide 32×32→64 multiply survives per `mul64`.
* And the C0 consumption contract ("the top 24 bits of each u32 half") means `draw48`'s two 24-bit
  halves are just `w.x >> 8` and `w.y >> 8`. The 48-bit value is never even assembled.

**Consequence:** C0's 1 TB PractRand certification carries onto the GPU *unchanged* — same counters,
same 48 bits, same order. The plan's deferred RNG decision ("accept the cost, or give the GPU a local
generator under statistical-only conformance, or re-litigate the cheap-hash family") is **settled by
the first option**, and the other two are off the table. See finding 3 for what it costs.

### 2. The GPU fuses multiply-add — so bitwise *lane* parity is impossible

The decisive experiment: compute `a*b + c` on the GPU and compare against **both** candidate CPU
answers.

```
a*b + c matches the CPU's mul-then-add in    3/4096 lanes
a*b + c matches a FUSED multiply-add   in 4095/4096 lanes
```

Metal contracts `a*b + c` into a single-rounding FMA. WGSL explicitly permits this, and there is no
portable way to disable it. So no amount of care with constants or Horner order can make GPU lane
arithmetic bit-identical to the CPU backends — the divergence isn't in *our* code, it's in the
instruction selection.

This lands the cross-backend contract in **two tiers**, and they are the ones G1's conformance tests
must assert:

| tier | what | claim |
|---|---|---|
| **1** | the draws (integer hash → 24-bit uniforms) | **bit-identical**, everywhere, including GPU |
| **2** | everything computed *from* them in f32 | **ULP-close**: ≤ 1e-6 absolute, measured |

Tier 1 survives because it is *integer* arithmetic — there is nothing to contract. That is the tier
the RNG certification lives in, which is why it is the one that matters.

**This falsifies a mitigation the plan was relying on.** The risk table said that if cross-vendor
parity ever mattered, we could "emit the shared `approx.rs` f32 polynomials instead of WGSL built-ins,
trading some of the hardware-transcendental win for bitwise portability." That trade does not exist:
the polynomial is contracted too. Measured, the polynomial gives us *neither* bitwise parity *nor*
better accuracy (both land at 7.15e-7 max deviation on `normal`) — and it costs 1.4× (finding 3).

> The 573-ulp figure in the raw output is ULP-as-a-metric misbehaving on near-zero values, not a real
> error: `normal`'s max *absolute* deviation is 7.15e-7 under both configs. Absolute error is the
> honest metric here, and it is the one to assert.

### 3. What the certified hash and the shared polynomials actually cost

Barrier-option shape (100 normals per lane, 1M lanes), configs timed **round-robin** so the GPU's
thermal drift can't be charged to whichever ran last:

| config | samples/s | vs shippable |
|---|---|---|
| squares64 / poly | 106–129 M/s | 1.00× |
| **squares64 / built-in** | **160–201 M/s** | 1.5× |
| cheap-u32 / poly | 196–206 M/s | 1.6–1.9× |
| cheap-u32 / built-in | 245–294 M/s | 2.3× |
| no hash at all / built-in | 288–309 M/s | 2.3–2.9× |

Read it as a decomposition:

* **A cheap u32 hash is *free*** — it times identically to no hash at all. Fully hidden behind the
  transcendentals.
* **squares64 costs ~1.5×** over that, on the most RNG-dense shape we have. Real, but nowhere near the
  "dominates the kernel" fear (the plan estimated ~70–90 ALU ops/uniform vs ~10 and worried it would
  swamp everything).
* **The polynomial costs ~1.4×** and, per finding 2, buys nothing.

**Decisions:** keep `squares64` (a 1.5× tax for a certified, bit-identical draw stream is a good
trade, and it keeps one RNG across four backends). Use the **WGSL built-ins** for `log`/`sin`/`cos`
(the polynomial is a pure loss on this backend).

> Caveat on the built-ins, for G1 to own: WGSL only *guarantees* ~2^-11 absolute error on `sin`/`cos`.
> Apple's are ~1 ulp, and the measured deviation is 7e-7, but a hostile vendor is within spec to be
> much worse. The conformance suite should assert the ULP bound on every device it runs on rather
> than trusting the spec floor.

### 4. Compile cost is the binding constraint — not throughput

This is the finding that reshapes G1, and it arrived by accident: `sum_normals(100)` is **201
statements** but compiled as slowly as a **17,602-statement** chain. Compile cost doesn't track graph
nodes — it tracks *emitted instructions*, and every RNG source inlines ~150 ops of emulated
`squares64`.

Confirmed directly (cold compiles, salted to defeat Metal's on-disk shader cache):

| sources | stmts | squares64 | cheap-u32 | ratio |
|---|---|---|---|---|
| 1 | 3 | 29 ms | 17 ms | 1.7× |
| 10 | 21 | 69 ms | 26 ms | 2.7× |
| 50 | 101 | 273 ms | 67 ms | 4.1× |
| 100 | 201 | 568 ms | 118 ms | 4.8× |
| 200 | 401 | 1311 ms | 225 ms | 5.8× |

~6.5 ms of compile per RNG source. And on pure arithmetic it is superlinear:

| stmts | naga | Metal | **cold** | warm (cached) |
|---|---|---|---|---|
| 102 | 1.6 ms | 87 ms | 88 ms | 0.8 ms |
| 1,002 | 3.4 ms | 73 ms | 76 ms | 5 ms |
| 5,002 | 35 ms | 290 ms | **325 ms** | 64 ms |
| 17,602 (`turboquant`) | 348 ms | 1,544 ms | **1,892 ms** | 717 ms |
| 45,002 (`prisoners`) | 2,109 ms | 6,800 ms | **8,908 ms** | 4,167 ms |

So the giant shaders *do* compile — but at `turboquant` scale that is ~1.9 s **per forcing**, and
`turboquant` forces ten times. Naga (our own parse/validate) is a quarter of it and is never cached.

> Two measurement traps here, both of which I fell into first. **(a)** Metal caches compiled shaders
> **on disk**, across processes — so re-running the spike measured cache hits, and the first compile
> table was ~2.6× too fast. Every compile measurement now injects a unique salt. **(b)** Allocating
> buffers inside the timed region put most of a "1.2 ms dispatch floor" into `create_buffer`.

### 5. The fix, measured: emit array draws as **loops**

`zs ~[100] normal(0,1)` is 100 sources with *consecutive ids and one body* — that is a loop, not 100
inlined hashes. The same graph, the same draws, emitted two ways:

**Head to head, same kernel, same draws, 1M draws × 100 normals = 100M normal draws:**

| | cold compile | dispatch | end-to-end |
|---|---|---|---|
| CPU (Cranelift JIT, multicore — the then-native backend, since retired) | — | 96 ms | 1.0× |
| GPU, **unrolled** | 572 ms | 3.4 ms (28×) | **6.0× SLOWER** |
| GPU, **looped** | **30 ms** | 5.0 ms (19×) | **2.7× faster** |

Both agree with the CPU to **1.14e-5** on a value of magnitude ~10 (f32 rounding over a 100-term sum
— a wrong draw stream would be O(1) off, not O(1e-5)).

The loop pays ~26% at dispatch (loop overhead, and the compiler can no longer schedule across
iterations) and wins **19× at compile**. On a single cold forcing that is the difference between
losing to the CPU and beating it. With a warm pipeline it is ~19× either way.

**This is the G1 design constraint:** a WGSL emitter must NOT be a straight transcription of
`wasm_emit` (or the since-retired `jit`). Those flatten the cone into a scalar statement chain because
their targets have no cheaper option; WGSL has real loops, and the demos that motivate this plan (`barrier_option`,
`turboquant`, `am_vs_fm`) are all array-draw-and-fold shapes, which are loops by construction. The
emitter has to preserve that structure rather than unroll it.

### 6. The array does not spill — so the IR change stays small

Finding 5 says the emitter must produce loops. There are two ways to get there, and they differ by an
order of magnitude in how much of the engine they touch:

* **Materialized array** — draw all `n` elements into a per-thread `array<f32, n>` in one loop, then
  consume them with the ordinary (unrolled) arithmetic. Needs *one* new IR node: an array-valued
  source, exactly like the `Rotation`/`Permutation` nodes that already exist.
* **Fused loop** — the draw fuses into the consuming loop; no array is ever materialized. Needs
  map/scan/reduce *region* nodes in the IR (a body subgraph with a loop variable) — a much larger
  change.

The materialized form is only viable if a per-thread array of `n` floats stays in registers. A spill
to thread-local memory would trade the compile win for a throughput collapse. Measured (1M lanes):

| n | | cold compile | dispatch | samples/s |
|---|---|---|---|---|
| 52 (`barrier`) | unrolled | 284 ms | 3.44 ms | 304 M/s |
| | **array (loop draw)** | **41 ms** | **3.43 ms** | **306 M/s** |
| | fused loop | 30 ms | 3.45 ms | 304 M/s |
| 100 | unrolled | 566 ms | 3.44 ms | 305 M/s |
| | **array** | **43 ms** | 4.96 ms | 211 M/s |
| | fused loop | 30 ms | 4.96 ms | 211 M/s |
| 256 | unrolled | 1862 ms | 9.54 ms | 110 M/s |
| | **array** | **65 ms** | **9.51 ms** | **110 M/s** |
| | fused loop | 30 ms | 9.36 ms | 112 M/s |

**It does not spill.** The materialized array matches the fully fused loop at dispatch *at every
size* — the Metal compiler keeps it in registers and re-fuses the consuming arithmetic back into the
draw loop itself. Compile collapses 7–29×.

So the region-node surgery is unnecessary: **one array-valued source node buys essentially the whole
win.** (One caveat worth a knob: at `n = 100` both loop forms are 1.44× slower at dispatch than the
unrolled one, which exposes more instruction-level parallelism across independent hashes. A partial
unroll — 4 draws per loop iteration — should recover the ILP while keeping the compile small.)

---

## Secondary numbers

**Dispatch + readback floor** (buffers pre-allocated, as a real backend would):

| lanes | round trip |
|---|---|
| 1 | 1.2–1.6 ms |
| 65,536 | 1.2 ms |
| 1,048,576 | 1.8 ms |

A ~1.2 ms fixed cost, flat until the work itself dominates. The gate's floor. (G2 can hide most of it
by pipelining — submit chunk *k+1* while folding chunk *k* — since lanes are stateless and chunks are
just ranges.)

**Demo-shaped throughput** (squares64 + built-ins):

| shape | stmts | dispatch | samples/s |
|---|---|---|---|
| barrier × 175k lanes | 403 | 1.6 ms | 108 M/s |
| barrier × 1M lanes | 403 | 6.2 ms | 169 M/s |
| signal × 1M lanes | 103 | 1.9 ms | 540 M/s |

---

## What this means for G1–G4

1. **Keep `squares64`.** Bit-identical draws, certification intact, ~1.5× on the most RNG-dense shape.
   The plan's RNG escape hatches are closed — no local GPU generator, no re-litigating pcg.
2. **Use the WGSL built-in transcendentals.** Delete the "emit the polynomials for bitwise
   portability" mitigation from the risk table — finding 2 proves it doesn't work.
3. **Conformance is two-tier**: bitwise on the draws, ULP-bounded (≤1e-6 abs) on lane arithmetic.
   Assert both, per device.
4. **The emitter must emit loops for array draws.** This is the difference between a 6× loss and a
   2.7× win on a cold forcing, and it is not a straight port of the existing emitters.
5. **A node/instruction cap is mandatory**, and it should be denominated in *emitted instructions*
   (≈ nodes + ~150 × sources), not graph nodes. A `MAX_WGSL_NODES` of ~5k emitted instructions keeps
   cold compiles under ~325 ms.
6. **The gate needs both terms**: `draws × cone_ops` must clear the ~1.2 ms dispatch floor *and* the
   cold compile, which is now the dominant fixed cost.
