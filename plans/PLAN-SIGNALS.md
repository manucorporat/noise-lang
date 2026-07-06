# Signals as drawn values — plan for review

Make the signal pipeline honest and expressive enough that `am_vs_fm` can be written *entirely in
signal land* — modulate, add static, demodulate — with **no lengths in the math at all**: the
sampling resolution becomes an engine knob (`engine::set_resolution`), the time-axis twin of the
existing Monte-Carlo budget. Two defects drive this, one semantic and one aesthetic:

1. **`noise_white` breaches the language's core invariant.** The load-bearing rule is "you cannot
   operate on an undrawn distribution — draw it first with `~`" (`eval.rs:2966`). `normal(0, 1)`
   in arithmetic is a runtime error; yet `noise_white(σ)` — equally random — flows into arithmetic
   bare, and an `=`-bound generator re-draws at **every mention**. Measured today:

   ```
   static = signal::noise_white(1);
   a = sample(static, 4);  b = sample(static, 4);
   Var(a[0] - b[0])   # → 2, not 0 — two mentions of one name are independent draws
   ```

   That is exactly the "hidden re-draw" the landing page promises never happens (`X − X` is 0,
   never "two samples"). LANG.md documents the behaviour, but documented inconsistency is still
   inconsistency.

2. **The resolution pollutes the whole program.** Today `sample(…, 64)` must happen *first*, so
   every downstream line — modulators, noise, demodulators — works on arrays and inherits an
   arbitrary `64` that is no part of the question being asked. The chain should stay lazy; the
   `64` belongs at the *measurement*, next to `mse`.

> **Status: IMPLEMENTED** (2026-07-06). All five build-order steps landed: the `SigExpr` tree,
> `~`-only noise drawing + the realization cache, complex signals + `noise_white_complex`,
> `engine::set_resolution` + reducer materialization, and the §5 migration (`am_vs_fm.noise` is
> the §2 program; `am_vs_fm_complex.noise` deleted; LANG.md / SKILL.md / `AmFmDemo.astro` updated).

> **Read first:** `AGENT.md`, `LANG.md` (signals section — this plan rewrites it),
> `PLAN-COMPLEX.md` (complex machinery this builds on; §5 of that plan is explicitly superseded
> here). `PLAN-COLLECTIONS.md` is the sibling plan for arrays.

---

## 1. Design decisions (settled in discussion)

1. **Noise generators are undrawn distributions — `~` is the only way in.** `noise_white(σ)` /
   `noise_brown` / `noise_pink` / `noise_ou` become subject to the same rule as `normal(0, 1)`:

   ```
   static ~ signal::noise_white(sigma);     # ONE realization; length still lazy
   w ~[n] signal::noise_white(1);           # realization pinned to n and materialized eagerly
   noise_white(1) + msg                     # error: undrawn distribution — draw it first with `~`
   sample(noise_white(1), n)                # error: same (was the sanctioned drawer; no longer)
   ```

   A `~`-bound noise is a **drawn realization**: every mention is the same noise, `static − static`
   is exactly 0, and re-materializing it (two `sample` calls, two carriers) reuses the same RV
   nodes. The plain-`~` form stays length-lazy: the realization pins its length at first
   materialization and a later mention meeting a *different* length is an error naming the pinned
   one (white noise has no refinable continuous path, so silent re-render at another resolution
   would be a lie). The `~[n]` form pins up front and materializes to an ordinary array of RVs —
   it *is* the old `sample(noise_*(…), n)`, now spelled as a draw.

   We considered allowing lazy *composition* of undrawn generators (`noise_white(σ) +
   i*noise_white(σ)`, drawn at the `~`). Rejected: `g = noise_white(σ); g − g` re-opens the exact
   sharing ambiguity this plan exists to close. Strict rule, no exceptions.

2. **Reducers materialize at the ambient resolution.** The language already solved this problem
   once, on the other axis: `E` does not take `40000` inline — the Monte-Carlo budget is set once,
   ambiently, via `engine::set_max_samples(n)`. The sampling resolution is the same kind of number
   (a measurement knob, not part of the question), so it gets the symmetric treatment:

   ```
   engine::set_resolution(64);    # the time-axis twin of set_max_samples — set once
   am_err = E(mse(rec_am, msg));  # reducers render lazy signals at the ambient resolution
   ```

   `mse`, `mean`, `sum`, `dot`, `neighbor_corr`, … meeting a lazy signal materialize it at the
   ambient resolution (engine state next to `max_samples`; **default 256**, so toy programs never
   mention it). `signal::sample(sig, n)` remains the explicit override. All reducers funnel
   through `expect_array` (`eval.rs`), so this is one hook, not twenty. Drawn noise realizations
   pin their length at first materialization as ever — changing the resolution *between* uses of
   the same realization is caught by the length-clash error (§4).

   One honest asterisk, for LANG.md: `E`'s reported digits reflect *measured* Monte-Carlo error;
   resolution bias (aliasing, under-resolved nonlinearities) is **not** in those error bars. That
   asterisk already existed with an inline `sample(…, 64)` — the knob just makes it worth stating.

   Rejected alternatives: erroring out ("sample first") keeps a wart in every program for no
   safety gain over the pinning rules; adaptive refinement (double `n` until the estimate
   stabilizes) is deferred (§7) — with per-sample noise semantics (decision 4) it is only honest
   for per-sample metrics like `mse`, and silently wrong for integrative ones.

3. **Signals go complex — by composition, not by a new type.** `Value::Complex { re, im }` already
   boxes two arbitrary real `Value`s (PLAN-COMPLEX §2); a complex signal is simply
   `Complex { re: Signal, im: Signal }`. No new value kind, no complex channel in the sample-DAG or
   VM. What it *does* require is generalizing `SignalSpec` from a linear op-pipeline to an
   expression **tree** (§3), because the complex decompositions in `binop_complex` produce
   signal×signal arithmetic (`abs(z) = sqrt(re² + im²)`, complex `*`), which today is a hard error
   (`eval.rs:1580`).

4. **Noise strength stays per-sample; PSD is rejected.** `noise_white(σ)` keeps meaning "each
   sample is `normal(0, σ)`" — it matches `normal(0, σ)`, matches what the discrete Monte-Carlo
   engine actually does, and the language has no time base to hang a power-spectral-density on
   (`sine(3)` is 3 cycles per *normalized window*). The resolution-dependence this implies is
   exactly why decision 2 keeps the resolution a *set* knob (never inferred): the `n` is explicit
   engine state, not something the engine guesses. If adaptive-resolution queries ever land (§7),
   revisit PSD then — not before.

5. **`signal::noise_white_complex(σ)` — the lazy CSCG.** PLAN-COMPLEX §5 deferred this "until a
   lazy complex-signal example needs one"; `am_vs_fm` now does, and decision 1 bans building it by
   composition. Circularly-symmetric complex white noise: independent `normal(0, σ/√2)` per
   quadrature per sample ⇒ **`E|z|² = σ²`**, matching `rand::normal_complex`'s total-power
   convention exactly. Drawn with `~` like the rest; materializes as
   `Complex { re: <n RVs>, im: <n RVs> }` lanes. (Complex brown/pink/ou: not until an example
   needs them.)

6. **What defers, post-draw.** The chain the target program needs, beyond today's scalar
   arithmetic and `sin`/`cos`/`atan`/`sign`/`neg` ufuncs:

   - `math::exp` on a real signal (new `SigOp`); complex `exp` then falls out of the existing
     channel decomposition `e^re·(cos im + i·sin im)`.
   - signal × signal `+ − * / ^` (the tree, §3) — needed by complex `*`, `abs`, and plain
     two-tone arithmetic (`sine(3) + sine(7)`), a long-standing gap.
   - `math::abs` / `math::arg` on complex signals: `sqrt` (i.e. `^0.5`) and a binary `atan2`
     node over signals.
   - `signal::sample` of a complex signal → array of complex (samples each channel; the realization
     cache keeps the quadratures consistent).

---

## 2. The payoff: `am_vs_fm` becomes the textbook

```
use math;    # i, exp, abs, arg
use vec;     # mse
use signal;  # sine, noise_white_complex

engine::set_max_samples(40000);   # Monte-Carlo budget    (amplitude axis)
engine::set_resolution(64);       # sampling resolution   (time axis)

am_modulate(m)        = 1 + m;             # message in the LENGTH
fm_modulate(m, dev)   = exp(i * dev * m);  # message in the ANGLE
am_demodulate(z)      = abs(z) - 1;
fm_demodulate(z, dev) = arg(z) / dev;

dev   = 3;
sigma = 0.4;

msg    = 0.3 * signal::sine(3);                 # lazy — a waveform, not yet samples
static ~ signal::noise_white_complex(sigma);    # ONE drawn realization, also lazy

rec_am = am_demodulate(am_modulate(msg)      + static);        # the whole chain stays
rec_fm = fm_demodulate(fm_modulate(msg, dev) + static, dev);   # in signal land

am_err = E(mse(rec_am, msg));   # mse renders both signals at the ambient resolution
fm_err = E(mse(rec_fm, msg));
```

No lengths in the math at all: the two measurement knobs sit together at the top, one per axis.
Note what the semantics buy: both `mse` calls materialize `static`, and the realization cache
guarantees they see the **same** noise — the "perfectly fair fight" is now a property of `~`, not
of careful program ordering.

---

## 3. The signal expression tree

`SignalSpec` (base wave + `Vec<SigOp>` pipeline, `signal.rs`) becomes a small expression tree:

```rust
enum SigExpr {
    Wave { wave: Wave, freq: f64 },            // sine / cosine leaf
    Konst(f64),                                // scalar promoted into signal land
    Noise { id: RealizationId, spec: NoiseSpec },  // a DRAWN noise realization (leaf)
    Unary(SigUnOp, Rc<SigExpr>),               // neg/sin/cos/atan/sign/round/floor/ceil/not + exp
    Binop(BinOp, Rc<SigExpr>, Rc<SigExpr>),    // + − * / ^ % and comparisons, and atan2
}
```

- `Rc` keeps the lazy builder cheap, as today; `Value::Signal(Rc<SigExpr>)` replaces
  `Value::Signal(Rc<SignalSpec>)`.
- `materialize(expr, n, engine)` walks the tree per sample index. Deterministic subtrees produce
  `f64`s exactly as `SignalSpec::sample` does now; a `Noise` leaf consults the **realization
  cache** and yields RV-node `Value`s, at which point the walk switches from float folding to the
  ordinary `binop` lifting (the same split `materialize_noise_like` navigates today). The colored
  kinds (`Brown`/`Ou`/`Pink`) keep their existing constructions in `Engine::materialize_noise`.
- **Realization cache**: `Engine` gains `realizations: HashMap<RealizationId, Realized>` where
  `Realized` pins `(len, channels: Vec<Vec<Value>>)` at first materialization; a later request at
  a different `len` is a runtime error quoting both lengths. Lives next to the scope map so the
  playground's `run_with_introspection` sidecar (which relies on Engine scope persisting across
  `run()`) keeps working unchanged.
- The `~` arm in `eval.rs` (the draw path at `eval.rs:709`) learns: RHS `Value::Noise(spec)` →
  allocate a `RealizationId`, bind `Value::Signal(SigExpr::Noise { .. })` (or, for
  `noise_white_complex`, `Complex` of two such). The `~[n]` draw-shape arm materializes eagerly
  and binds the resulting array.
- **Backends are untouched.** All of this is front-end: materialization emits the same `normal`
  RV source nodes as today, so the interpreter, Cranelift JIT, and WASM emitter see an unchanged
  compiled graph. No cost-model or `inline_trans` interaction.

---

## 4. Errors (the teaching surface)

- `noise_white(1) + msg` / `sample(noise_white(1), n)` →
  ``"`noise_white(1)` is an undrawn distribution, not a value — draw it first with `~` (e.g. `static ~ signal::noise_white(1)`), or pin a length with `~[n]`"``
  — same shape as the `normal(0, 1)` message, because it is the same rule.
- re-materialization length clash (explicit `sample` at one length, or `set_resolution` changed
  between two uses of the same realization) →
  ``"this drawn noise was realized at length 64 and cannot be re-rendered at 128 — noise has no finer version of itself; keep one resolution across its uses, or pin it with `~[64]`"``
- `plot::line` of a complex signal/array → error suggesting `math::re/im/abs` (a complex wave has
  no single trace).

Tests to flip: `lib.rs:1351-1364` (noise draw semantics — rewrite around `~`), `lib.rs:2317`
(`noise_white(1) + math::i` still errors, but the message becomes the undrawn-distribution one).
New tests: same-realization sharing across two `sample` calls (`Var(a[0]-b[0]) == 0`), length-clash
error, signal×signal arithmetic, reducers rendering at the ambient resolution (default and after
`set_resolution`), complex signal chain end-to-end (`E(mse(...))` of the §2 program lands within
tolerance of the current `am_vs_fm` numbers, ratio ≈ `dev²`).

---

## 5. Migration (breaking on purpose)

| Site | Change |
| --- | --- |
| `examples/am_vs_fm.noise` | Becomes the §2 program — the flagship of the feature. |
| `examples/am_vs_fm_complex.noise` | Delete: it converges with the above (the `[I,Q]` on-ramp story moves to a comment). |
| `examples/noise_colors.noise` | `signal::sample(signal::noise_white(1), n)` → `w ~[n] signal::noise_white(1)` etc. — reads *better* (each color is a named, drawn thing). |
| `examples/nyquist.noise` | Untouched (deterministic signals only). |
| `packages/www/src/components/AmFmDemo.astro` | Code panel + step narration follow the §2 program; the "same static on both carriers" caption becomes *true* (today the two carriers get independent draws). Scrolly step boundaries need re-cutting since the code shape changes. |
| `LANG.md` §Signals | Rewrite: signals as lazy waveform expressions; `~` draws a noise realization; reducers render at the ambient resolution (`engine::set_resolution`, default 256) with the resolution-bias asterisk; complex signals; the per-sample σ convention stated explicitly. |
| `.claude/skills/noise-lang/SKILL.md` | Update the signal-module guidance to the `~` idiom. |
| `PLAN-COMPLEX.md` §5 | Add a superseded-by-this-plan note. |

---

## 6. Build order

1. `SigExpr` tree + signal×signal arithmetic + `exp` deferral (pure refactor of the deterministic
   path; `nyquist.noise` is the regression test, plus new two-tone tests).
2. Undrawn-noise enforcement + `~` / `~[n]` draw arms + realization cache (the semantic fix; flip
   the `lib.rs` noise tests; migrate `noise_colors.noise`).
3. Complex signals over the tree (`Complex{re: Signal, im: Signal}` through `binop_complex`,
   `abs`/`arg`/`exp`, complex `sample`) + `noise_white_complex`.
4. `engine::set_resolution` + reducer materialization at the ambient resolution + error messages.
5. Rewrite `am_vs_fm.noise`, delete `am_vs_fm_complex.noise`, update `AmFmDemo.astro`, LANG.md,
   the skill.

Steps 1–2 are independently shippable and fix the API defect on their own; 3 is where the demo
payoff lands.

---

## 7. Deferred (explicitly out of scope)

- **Adaptive-resolution queries** — `E(mse(sig_a, sig_b))` with no `sample`, the engine doubling
  `n` until the estimate stabilizes within its reported digits (the time-axis twin of the existing
  sample-budget machinery). Wants the PSD noise convention to be honest for integrative metrics;
  revisit as one package if a motivating example appears.
- **PSD noise convention** — see decision 4.
- **Complex colored noise** (`noise_brown/pink/ou` complex variants) — no example needs them.
- **Deferred reducers** (lazy `mse` as a "number defined as a limit") — subsumed by the first
  bullet.
