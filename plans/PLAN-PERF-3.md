# PLAN-PERF-3.md — third performance pass: closing the Track C gate

Follow-on to `PLAN-PERF-2.md`. Conducted 2026-07-14, in the middle of PLAN-PREGPU Track C
(counter-keyed pcg-family RNG in every backend, bit-identical draws). Track C's landing
condition is **corpus-neutral-or-better `example_times`**, and the first cut is not there
yet — this plan is the ranked path to closing that gap, plus the perf follow-ons the new
RNG architecture unlocks.

Numbers marked **[measured]** were run on this machine (Apple M4 Pro, 14 cores,
`--features jit`, release).

---

## Where the gate stands [measured]

`example_times`: **3938 ms vs 4351 ms pre-Track-C baseline (−9.5%) — gate CLOSED, well
better than baseline** after Track B's f32 lanes (trajectory: first cut +30% → lane pairing
+19% → items 1–2 +3.8% → squares64 −5.1% → **f32 lanes −9.5%**). Post-B fill ceilings
[measured, M4 Pro]: uniform **1039 M/s** (was 540 — the pair-shared draw's full 2×),
uniform_int 490 M/s, exp 260 M/s, normal **90 M/s** (was 86 — Box–Muller is
transcendental-*latency* bound, so halving the hashing barely moves it).

Browser (V8, 4M draws, emitted f32 kernel): π 16 ms · normal-heavy 25 ms · dice 23 ms ·
arithmetic 9 ms [measured].

| example | before | first cut | + pairing | + items 1–2 | note |
|---|---|---|---|---|---|
| turboquant | 1428 | 2185 | 2003 | 1565 (+10%) | interpreter; item 1 (bulk normals) |
| barrier_option | 973 | 1141 | 1131 | **890 (−8%)** | JIT; item 2 (pair-unroll) |
| prisoners | 961 | 1129 | 982 | 972 (+1%) | recovered by pairing |
| am_vs_fm | 405 | 493 | 435 | 434 (+7%) | interpreted trig ufuncs → item 4? |
| noise_colors | 336 | 432 | 350 | 355 (+6%) | ~noise |
| clt_normal | 12 | 23 | 22 | 23 (~2×) | small absolute; below JIT gate? |

Keyed-fill ceilings, single thread [measured, `bench_keyed_fills`] — the pcg4d-3r/f64 column is
the original PERF-3 baseline, the last one is what ships after Squares + Track B:

| fill | pcg4d-3r, f64 | squares64, **f32 (shipped)** | note |
|---|---|---|---|
| `fill_uniform` | 554 | **1039** | 1.92× — one hash per lane pair, the designed halving |
| `fill_uniform_int(1,6)` | 529 | 490 | unchanged by design (still one 48-bit Lemire draw per lane) |
| `fill_exp` | 169 | **260** | one hash + two `ln` per pair |
| `fill_normal` (lane-paired) | 90 | 90 | **flat** — Box–Muller is `ln`+trig *latency* bound; the hash was never its bottleneck |
| `CellStream` normals (d²=256, Rotation's shape) | 38 → 96 (item 1) | 96 | stays f64 internally (Rotation's MGS scratch) |

The honest read of that table: the RNG halving is real and lands squarely on uniform-heavy
graphs, while normal-heavy graphs are gated by the transcendental chain — which is exactly what
item 4 (vectorized `approx`) targets, and why it is now the top open lever.

---

## Ranked work

### 1. ~~`CellStream` bulk fills~~ (LANDED 2026-07-14 — 38 → 96 M normals/s [measured]; turboquant 2003 → 1565 ms)

`CellStream::next_normal` is a serial, branchy state machine (Option pending-branch +
pair bookkeeping per call): 38 M normals/s where the independent-lane `fill_normal` does
90. Fix: two-phase bulk consumption — `fill_u48s` (tight hash loop, one iteration per
hash, independent iterations → vectorizes like the 236 M hashes/s bench) into a reused
scratch, then a normals pass over independent pairs (ln/sincos chains overlap across
pairs). Consumption order stays exactly the scalar stream's, so draws are bit-identical.
Wire into `Inst::Rotation` (d² normals per lane) first; `Inst::Permutation`'s
`next_bounded` sequence is the same shape if it ever shows up in a profile.

### 2. ~~Pair-unrolled JIT + wasm kernel loops~~ (LANDED 2026-07-14 — JIT: barrier_option 1131 → 890 ms, now *below* baseline; wasm emitter ported the same two-lane loop the same day, conformance upgraded to full-batch bitwise vs the interpreter. Cones ≤ `PAIR_UNROLL_MAX_NODES` = 2048 unroll, larger keep parity-select. Browser wall-clock win still unmeasured — re-run the PERF-2 Node/Chrome bench when convenient)

The interim parity-select `emit_normal` computes the full hash + `ln` + both trig
kernels **per lane** — everything the pair shares is recomputed for the odd lane, so
JIT normal draws cost ~2× the xoshiro version (hash is ~40 i32 ops vs 12). Fix: emit two
lanes per loop iteration (the deleted multi-stream loop's shape, two memos per
iteration), with `Normal` nodes computing the pair once — cos to the even lane's memo,
sin to the odd's via a side map. `n` is always `BATCH` (even) from every runner.
Expected to recover most of the ~200 ms the three JIT examples lost.

### 3. ~~Uniform lane-pairing (2 uniforms per hash)~~ (LANDED 2026-07-14, *as part of Track B*)

Exactly as predicted, Track B rewrote this consumption contract rather than bolting pairing
onto the f64 one — which is why the plan said not to pre-optimize it. Shipped shape: one
squares64 (48 consumable bits) feeds a whole lane PAIR for `unif`/`normal`/`exp`/`geometric`
— even lane takes the low 24 bits, odd the high 24. `fill_uniform` **540 → 1039 M/s (1.92×)**,
the clean halving. `unif_int` deliberately opted out (24-bit Lemire would put the bias at
`count/2²⁴`), so it keeps one 48-bit draw per lane.

The lesson worth keeping: the win landed on `unif` and **not** on `normal` (1.04×). Box–Muller
was never hash-bound — it is `ln` + two-branch trig latency — so halving its hashing bought
almost nothing. Profile the bottleneck, not the op count.

### 4. Vectorized `approx` transcendentals in the interpreter lane path (P2 — now the top open lever)

`fill_normal`'s 90 M/s is transcendental-bound (scalar `approx::ln`/`sin`/`cos` per
pair), and `apply_un`'s Sin/Cos/Ln columns now also run the shared polys per-lane —
`am_vs_fm`'s residual +7% likely lives there (approx-vs-libm and no vectorization). The
polynomials are branch-light (selects, not branches) and block-vectorizable (4–8 lanes
of Horner in NEON). Measure `apply_un` trig columns vs libm first; if approx is slower
scalar-for-scalar, block-vectorize the column loops (the bit-parity contract only pins
the *math*, not the loop shape). B's f32 polys (fewer terms) shift this again — don't
gold-plate f64.

### 5. ~~Generator-swap re-pricing~~ (EXECUTED 2026-07-14 — squares64 swapped in, corpus −5.1% vs baseline)

pcg4d-3r **and** pcg4d-3rf failed PractRand at 256 GB (real low-consumed-bit sequential
structure); **Squares with a construction-compliant key finished 1 TB with zero
anomalies** (2026-07-14). Additional measured shape since the table below: `squares64`
serving two f32 uniforms (or one Box–Muller pair) per call — **1036 M f32 draws/s**,
0.80× pcg — which also ~halves the emulated WGSL cost per uniform (~50 ops). **CPU cost re-measured in the actual fill shapes** (independent
per-lane loops — the earlier "0.54×, 4.7 ns/word" figure came from serial-accumulator
benches; M4's multiply pipes overlap independent middle-square chains fine) [measured]:

| shape | pcg4d-3r | squares |
|---|---|---|
| f64 uniform (interim) | 478 M/s (1 hash) | 506 M/s (2× squares32) · **652 M/s (1× squares64, top 48 bits)** |
| f32 uniform (post-B) | 1290 M/s (w0 of 1 hash) | 876 M/s (1× squares32) — 0.68× |

So the swap is ~**neutral on CPU today** (use squares64 for the interim f64 uniform, one
call per draw) and ~1.3× on post-B f32 uniforms — while `normal` stays
transcendental-dominated either way (hash share ≲ 20% of a Box–Muller pair). The real
architectural cost is the future **GPU**: ~70–90 emulated ALU ops per uniform vs pcg's
~10 — but WEBGPU G0 measures whether RNG ALU even matters in real kernels before that
choice binds. Consumption schedule per the certified regime: sequential counters, base
`source << 36`, 1 (f32) / 1×squares64 (f64) per uniform, 2–4 per normal pair — counter
budget 2³⁶ per source covers the 2³² lane cap with room. Swap seam:
`rng::cell`/`CellStream` + the two `emit_cell`s + KATs; items 1–2 are hash-count
optimizations and carry over unchanged. Re-run the gate under the final generator.

### 6. Carried from PLAN-PERF-2 (unchanged priority)

- Quantiles parallelize via `sample_n_par` (landed third pass) — the remaining P2 there
  is memory, not CPU.
- Recursive emitters → `MAX_CODEGEN_NODES` cliff (P2).
- Does `noise-cli` ship the JIT? (P2) — note the JIT compile-vs-interpret gate now keys
  on the same cost model but kernels got slightly bigger (inline hash per source); worth
  re-measuring `BREAK_EVEN_*` once items 1–2 settle the kernel shape.

---

## Non-goals here

- f32 lanes: that's PLAN-PREGPU Track B, sequenced after C lands its gate.
- GPU: PLAN-WEBGPU, after PREGPU.
- Re-litigating the RNG: quality/certification lives in tools/rng-cert + PREGPU Track C;
  this plan only prices whatever generator survives criterion 8.
