# C0 results log

## 2026-07-14 — fast battery (criteria 1–7), M4 Pro, single thread

Criteria as frozen in README.md. Full battery: `cargo run --release -- all <gen>`;
avalanche structure: `avmap` / `avrow`; throughput: `bench`.

### Verdicts against the frozen criteria

| generator | avalanche (crit. 1, full 128-bit grid) | stats (crit. 2–7) |
|---|---|---|
| **pcg4d** (as published) | **FAIL — disqualifying.** 2565/16384 cells outside ±0.01, including *fully deterministic* cells (p = 0 or 1) in every input-word × output-word block, and inside the consumed region under realistic inputs (worst: key-bit 25 → consumed out-bit 8, p deterministic; 410/9696 usage-restricted cells outside). Cause: u32 mul/add carries only propagate upward; the single `^= >>16` is the only downward path, so low output bits have bounded input support. | all 7 pass |
| **squares32** (reference) | FAIL on the full grid — but *only* ctr bits 58–63 (worst 0.252 at bit 63; bits 32–57 all clean, worst 0.005). Unreachable in engine usage: the plan caps a forcing at 2³² lanes, keying keeps ctr < 2⁴⁰. Domain-restricted: clean. | all 7 pass |
| **pcg4d-xs** (+1 xorshift) | FAIL — no deterministic cells, but worst 0.167 in consumed bits. | not run (already out) |
| **pcg4d-f** (+fmix32/word) | FAIL — words 1–3 fully clean (incl. discarded bits); word 0 still weak (worst 0.126, lane bits → w0). w0 is the least-mixed word: its final-round update uses round-1 inputs. | not run |
| **pcg4d-3r** (+xorshift + 3rd product round, +12 ops) | **Consumed bits fully clean**: 0/9696 usage-restricted cells outside (worst 0.0052 ≈ null). 17/16384 full-grid cells outside, **all in the discarded low byte** (`w & 0xFF`; worst 0.054, source-bit 31 → w2 bit 0). | **all 7 pass** (worst z = 0.86) |

### Throughput (keyed-batch shape: keys fixed, lane sequential — vectorizes)

| generator | M hashes/s | M usable u32 words/s | vs today's xoshiro×4 (~388 M u32/s) | est. WGSL ALU ops / uniform |
|---|---|---|---|---|
| pcg4d | 391 | 1563 | 4.0× | ~7 |
| pcg4d-f | 325 | 1302 (976 if w0 dropped) | 3.4× | ~12 |
| **pcg4d-3r** | 236 | 942 | 2.4× | ~10 |
| squares32 | 211 | 211 | **0.54×** | ~70–90 (u64 mul emulation) |

### Reading

- **pcg4d as published fails C0** on the frozen avalanche criterion, in the consumed
  region, with deterministic input→output bit relations. Not rescuable by domain
  arguments.
- **No candidate passes criterion 1 literally as frozen** — including the certified
  reference — because the criterion counts discarded output bits (0..8, dropped by the
  `w >> 8` conversion) and input bits outside any reachable domain (Squares ctr bits
  58–63 ⇒ > 2⁵⁸ draws). The reference failing is the calibration signal the plan built in.
  Amending the criterion (restrict to consumed out-bits × reachable in-bits) is a
  re-freeze and needs an explicit owner decision, recorded here when made.
- **pcg4d-3r** (one extra xorshift + product round appended to pcg4d) passes everything
  the amended criterion would ask, at 2.4× today's CPU throughput and ~10 WGSL ops per
  uniform. It is a *custom variant* — no published certification; C0's battery +
  PractRand would carry the entire evidence burden.
- **squares32** carries Widynski's published BigCrush/PractRand certification and is
  clean over the reachable domain, but at 0.54× today's CPU throughput (risk to the
  corpus-neutral gate on RNG-bound JIT examples) and ~8× pcg4d-3r's GPU cost.
- If a word's low byte is ever consumed (e.g. the Track C interim f64 fill), the contract
  must stay "only bits 8..31 of each word are consumable": an interim f64 uniform should
  take 24+24 bits from two words (2⁻⁴⁸ granularity), not 32+21.

### Verdict — 2026-07-14, owner-ratified

- **Criterion 1 re-frozen** to consumed output bits × reachable input bits (see README
  Amendment). Consequence: fills may never consume a word's low byte; the Track C
  interim f64 uniform is 24+24 bits from two words.
- **pcg4d (as published): disqualified** — deterministic consumed-bit relations even
  under the amended criterion.
- **pcg4d-3r adopted** as the Track C generator, full certification burden on this
  harness. Squares stays the contingency: a criterion-8 failure of pcg4d-3r swaps to
  Squares and re-runs the battery, accepting the CPU cost.

## 2026-07-14 — criterion 8 (PractRand)

- **Negative control, by accident:** the first Squares reference run streamed word 0 of
  the 4-word cell mapping (`ctr = src<<36 | lane<<2 | word`), i.e. counters at stride 4.
  For middle-square, stride-4 counters ≡ sequential counters with key `4·key` — an
  *even* key, violating Widynski's key invariant. **PractRand failed it at 16 GB**
  (Gap-16 p = 4e-175, DC6/FPF/TMFn FAILs on [Low1/32]) while the 7-stat battery had
  passed the same subsampled stream. Two lessons, both recorded as constraints:
  the deep run catches what moment/correlation stats can't, and **Squares-as-fallback
  must consume counters sequentially** (per-distribution schedules: uniform `ctr =
  base+lane`, normal `ctr = base+2·lane(+1)` — consumed sets stay sequential; no
  power-of-2 strides, ever). The re-run streams all four words in order (sequential
  counters, the certified regime).

- **Harness defect #2 — lane wrap (voided the first round of deep verdicts).** The
  stream's `lane0 = batch · 1024` wrapped u32 at 2³² lanes = **206 GB** in words4-consumed
  format, repeating the stream verbatim from there. Every generator therefore "failed"
  the 256 GB evaluation block with main-stream BCFN — *including certified Squares on
  sequential counters*, which is what exposed the artifact (the reference earning its
  keep a second time). All ≥ 206 GB verdicts from the first round are void; genuine
  pre-wrap data: squares clean ≤ 137 GB; pcg4d-3r clean ≤ 64 GB with `[Low1/32]` FPF
  *very suspicious* (not FAIL) in the 128 GB block — under the frozen 4×-clearance rule
  that adjudicates at 512 GB in the corrected re-run. Fixed by advancing to a fresh
  source block when a block's 2³²-lane space is exhausted (matching real usage: no
  `(lane, source)` cell ever repeats).
- Candidates added while re-running:
  - **pcg4d-3rf** (3r + fmix32 per word, +20 ops): first candidate with a *fully clean*
    avalanche grid (0/16384 cells outside, discarded bits included); 856 M u32/s (2.2×
    today), ~15 WGSL ops/uniform.
  - **pcg4d-f w1–w3** (2 rounds + fmix32, word 0 discarded): clean avalanche on the
    three consumed words; ~940 M effective u32/s, ~16 WGSL ops/uniform (not in the
    re-run round — dominated by 3rf's strictly stronger mixing at similar cost).

## 2026-07-14 — corrected-stream verdicts (round 2) and harness defect #3

- **pcg4d-3r: FAILS criterion 8, legitimately.** Corrected stream: clean at 128 GB,
  hard `[Low1/32]` FPF/TMFn/DC6 fails by 256 GB. (Both rounds share the identical
  first-206 GB prefix, which is why the statistics matched — the wrap only voided the
  *conclusion*, not this one.) The `[Low1/32]` derived stream samples the lowest
  consumed bits (word bits 8/16/24 at the packing's 32-bit strides): real sequential
  structure, invisible to avalanche.
- **pcg4d-3rf: FAILS identically at 256 GB** — a per-word fmix32 is a bijection; it
  relocates cross-hash sequential correlation but cannot remove it. Together these
  read as: the pcg family's LCG-fed counter mixing is structurally too shallow at the
  10¹¹-sample scale, regardless of finalizer.
- **Harness defect #3 — invalid Squares reference key.** The harness key
  (`0xc58efd154ce32f6d`) was not from Widynski's key construction (repeated nibble in
  the upper half); the round-2 Squares failure at 256 GB is attributed to it. Re-running
  with a construction-compliant key (`0xf7c3b1a9e6d5c8b3`).

### Pending

- `squares` (good key) 1 TB run in flight. It doubles as the harness exoneration: a
  certified generator through the exact packing/stream. Clean → **Squares is the
  generator, per the ratified fallback protocol** (both custom candidates are out);
  another 256 GB failure → the packing/harness is indicted instead and everything
  re-runs.
- The Track C integration (landed on pcg4d-3r) keeps the generator behind `rng::cell` /
  the emitters' `emit_cell`; a swap to Squares touches those functions plus the
  consumption schedule (one u32 per counter, sequential per source) and re-pins KATs.
