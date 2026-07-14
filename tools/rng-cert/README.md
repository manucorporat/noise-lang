# rng-cert — PLAN-PREGPU C0, the generator spike

Certifies **pcg4d** (Jarzynski–Olano, JCGT 2020) over the engine's *exact planned keying*
— `pcg4d(key_lo, key_hi, global_lane, source_offset)` with `key = SplitMix64(seed)` split
into two u32s and `source_offset` a small per-source constant — before Track C integrates
it. **Squares** (`squares32`, Widynski) runs the identical battery as the in-harness
certified reference; it is also the contingency generator if pcg4d fails.

Zero dependencies on noise-core (workspace-excluded). Deterministic: fixed seeds, single
thread, same output every run.

## Pass/fail criteria — FROZEN 2026-07-14, before the first run

Per PLAN-PREGPU Track C0, the verdict cannot be negotiated after the fact. A generator
**fails C0** if any of the following trips; pcg4d failing ⇒ Squares replaces it (and must
itself pass this battery).

Statistical tests use a global two-sided α = 10⁻³ Bonferroni-corrected over the m = 7
z/p-value tests below ⇒ per-test α = 1.43·10⁻⁴, i.e. **|z| ≥ 3.81 fails** (harness gates
at 3.81; chi-square gates on tail probability < 7.1·10⁻⁵ in either tail).

1. **Avalanche band** (not a z-test): flip any 1 of the 128 input bits — keys random,
   lanes/sources small and realistic — and every output bit must flip with
   p = 0.50 ± 0.01, at N = 2¹⁷ trials per input bit (σ ≈ 0.0014, so the band is ≈7σ wide;
   under the null the worst of the 128×128 cells sits near ±0.006). Any cell outside the
   band fails.
2. **Adjacent-lane correlation**: Pearson r of `u(lane, s0)` vs `u(lane+1, s0)` over
   n = 2²³ pairs; |z| = |r|·√n.
3. **Batch-stride correlation**: same at lane distance 1024 (the interpreter's batch).
4. **Adjacent-source correlation**: `u(lane, s0)` vs `u(lane, s1)` — the pairing joint
   queries depend on.
5. **π/4 known-answer**: P(u² + v² < 1) with u from source 0, v from source 1, same lane
   (the cross-source joint pairing), n = 2²⁸ pairs, f32 pipeline as the engine will run it
   (`(word >> 8) as f32 · 2⁻²⁴`).
6. **Die chi-square**: `floor(u·6)` over n = 10⁸ draws, df = 5, two-sided on the tail
   probability.
7. **Box–Muller normality**: skewness z and excess-kurtosis z of 2²⁷ normals computed the
   engine's way (pair per hash, f32 `ln`/`sqrt`/`cos`/`sin`, f64 accumulation). Both
   moments must pass (counted as two of the m tests; π/4, die, and the three correlations
   are the other five).
8. **PractRand ≥ 1 TB** over words serialized in kernel-consumption order
   (`rng-cert stream …`): any explicit `FAIL` verdict fails. `unusual` / `suspicious`
   grades are recorded, and a `very suspicious` at some length must clear itself by 4× that
   length or it counts as a fail.

## Amendment — 2026-07-14, owner-ratified after the first battery run

The first run (see RESULTS.md) showed **no candidate passes criterion 1 as originally
frozen** — including squares32, the certified reference — because the full 128×N grid
counts (a) output bits 0..8, which the engine's `w >> 8` conversion discards, and (b)
input bits no reachable run can flip (Squares ctr bits 58–63 require > 2⁵⁸ draws; the
engine caps a forcing at 2³² lanes). The reference failing is the calibration signal, so
criterion 1 is re-frozen as: **every *consumed* output bit (8..32 of each word) must be
in the 0.50 ± 0.01 band for every input bit reachable in engine keying** (all 64 key
bits, lane bits 0..28, source bits 0..9) — the `usage-restricted` line `avmap` prints.
Consequence recorded in the same decision: no fill may ever consume a word's low byte
(the interim f64 uniform takes 24+24 bits from two words, not 32+21).

Under the amended criterion, **pcg4d as published still fails** (deterministic cells at
e.g. key-bit 25 → consumed bit 8) and was **disqualified**. The owner ratified
**pcg4d-3r** — pcg4d with one appended xorshift + product round — as the Track C
generator (0/9696 restricted cells outside, 7/7 stats, 2.4× current CPU RNG throughput,
~10 WGSL ops/uniform), with the full evidence burden on this harness: criteria 2–7 plus
criterion 8 (PractRand ≥ 1 TB) apply to it unchanged, with squares32 as the running
reference.

## Usage

```sh
cargo run --release -- all            # fast battery (1–7) for pcg4d and squares32
cargo run --release -- all pcg4d      # one generator
cargo run --release -- stream pcg4d | RNG_test stdin32 -tlmax 1TB   # criterion 8
cargo run --release -- stream squares | RNG_test stdin32 -tlmax 1TB
```

`stream` emits u32 LE words in kernel-consumption order: batches of 1024 lanes, 4 sources
per batch, word 0 of each hash (what f32 uniforms will consume). `stream <gen> words4`
emits all four output words per hash instead.
