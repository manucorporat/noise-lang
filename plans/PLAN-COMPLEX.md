# Complex numbers in Noise — plan for review

Add **complex numbers** as a first-class scalar type. The immediate payoff is the `am_vs_fm`
example, which already does complex arithmetic *by hand* (`[I, Q]` phasors). The longer game is
that the same machinery — Euler `exp`, complex `@`, `abs`/`arg` — is exactly what a faithful
small-scale **quantum** simulation (period-finding / Shor) needs. One feature, two payoffs.

> **Status: IMPLEMENTED.** Both tracks shipped. Complex is a first-class scalar
> (`Value::Complex`, operators, `math::i`/`j`, the `math::` ufuncs, `rand::normal_complex`, the
> `vec` consistency pass) with `examples/am_vs_fm_complex.noise`; the general surface (`%`,
> `math::floor`/`ceil`, comprehensions, `vec::outer`, `rand::categorical`) and the quantum capstone
> `examples/shor_period.noise` are in. One deviation from the sketch below: the exponential
> *distribution* was renamed **`rand::exponential`** (from `exp`) so `math::exp` is the function.

> **Read first:** `AGENT.md` (current code), `LANG.md` (the language contract — update it as you add
> surface), then this doc. `PLAN-COLLECTIONS.md` is the sibling plan for arrays/loops.

---

## 1. Design decisions (already settled in discussion)

1. **No `complex::` namespace.** Complex lives in `math::`, polymorphic by input type. The
   operators (`+ - * /`, `@`) *cannot* be namespaced anyway — `z1 * z2` has to be complex-multiply
   — so the functions follow the operators into `math::` for consistency. This also matches numpy /
   `std::complex` and Noise's own lifting philosophy (ops map uniformly, types lift).

2. **Branch by input type.** Real in → real semantics (so `math::sqrt(-1.0)` stays `NaN`, IEEE
   unchanged); complex in → complex semantics (`math::sqrt(-1 + 0*i) == i`). Same ufunc, two arms.

3. **The type emerges; no lexer change.** There is no complex literal. A complex value is built
   from the constant `math::i` plus the existing operators: `2 + 3*math::i`. Nothing in
   `lexer.rs`/`parser.rs` changes.

4. **The distribution is `rand::normal_complex`** — suffix flavor, matching the existing `_int`
   modifier (`unif_int`, `normal_int`, `exp_int`). Keeps the `rand` module a regular grid:

   ```
   unif     unif_int     (unif_complex)
   normal   normal_int   normal_complex
   exp      exp_int
   ```

5. **`signal::noise_white` is untouched.** It stays real (needed by `noise_colors`). Complex random
   generation is a `rand` concern, drawn with `~[n]` like every other distribution — not a
   signal-module special case. No `noise_white_complex` until a lazy *complex-signal* example needs
   one.

---

## 2. The runtime type

Add a `complex` form to the `Value` enum (`value.rs`), alongside `Num/Bool/Str/Unit/Array/Dist/Est`:

```rust
Complex { re: f64, im: f64 }
```

Update `type_name()` and `Display` (`Display` e.g. `"2 + 3i"`, `"-1i"`, `"0"` for `0+0i`). Arrays of
complex are just `Array` holding `Complex` values (broadcasting handles the rest).

**Folding.** Pure-constant complex subexpressions fold eagerly, same as reals. When a `dist` operand
is involved the op lifts to a `dist` (a complex-valued random variable — see §5).

---

## 3. Operators (lift by type)

| Op | Rule |
|---|---|
| `+ - * /` | complex when **either** operand is complex (real promotes to `re+0i`). Scalar `*` of two complexes = complex multiply; arrays stay **elementwise**; `@` is complex matmul (multiply+add over ℂ) |
| `**` | complex base with **integer** exponent = repeated multiply (enough for QFT/quantum); general complex `**` can be deferred |
| `== !=` | exact compare of (re, im) |
| `< > <= >=` | **type error** — no total order on ℂ (consistent with "ordering needs two numbers") |
| prefix `-` | negate (re, im); `!` stays bool-only |

`@` already exists for real matmul; extend its kernel to complex element ops. This is the single
most load-bearing operator for the quantum use case (`QFT @ state`).

---

## 4. `math::` additions

All are **ufuncs**: scalar, lifted over RVs, mapped over arrays (same machinery as today's
`sin`/`cos`/`atan`). Routing lives at `eval.rs:1684` (the `"sqrt" | "round" | … => "math"` arm) —
add the new names there. Real `exp` currently does **not** exist as a function (only `rand::exp` the
*distribution*, `builtins.rs:93`); we add it.

| Name | Real branch | Complex branch | New? |
|---|---|---|---|
| `i` (alias `j`) | — | the unit `0+1i` (a *constant*, like `pi`/`e`) | new |
| `exp` | `e^x` | Euler: `e^{re}·(cos im + i·sin im)` | **new (reals too)** |
| `abs` | `\|x\|` | magnitude `√(re²+im²)` | **new (reals too)** |
| `sqrt` | IEEE (`sqrt(-1.0)=NaN`) | principal root | extend (`builtins.rs:153`) |
| `arg` | `0` or `π` | `atan2(im, re)` | new |
| `conj` | `x` | `re − i·im` | new |
| `re` / `im` | `x` / `0` | `re` / `im` | new |

`sin`/`cos` over complex are **not** needed by AM/FM or the QFT — defer them.

`math::j` alias: the AM/FM example is electrical-engineering, where `j` is the imaginary unit (`i`
clashes with current/index). Provide both names for the same constant.

---

## 5. `rand::normal_complex`

> **Superseded in part by `PLAN-SIGNALS.md` (implemented):** the "no `noise_white_complex` until a
> lazy complex-signal example needs one" deferral below is over — `signal::noise_white_complex(σ)`
> now exists as an undrawn generator drawn with `~`/`~[n]` (per-quadrature `normal(0, σ/√2)`,
> `E|z|² = σ²`, matching this section's total-power convention), and `am_vs_fm.noise` is written
> entirely in signal land. `rand::normal_complex` itself is unchanged.

Circularly-symmetric complex Gaussian (CSCG) — the textbook model for radio static, thermal noise,
and Rayleigh fading. Mean 0; `re` and `im` each `~ N(0, σ/√2)`, independent ⇒ **`E|z|² = σ²`** and
`|z|` is Rayleigh(scale σ/√2). Drawn with `~` / `~[n]`.

Convention note: this puts *total* power at σ², so `normal_complex(σ)` matches `normal(0, σ)` on
power. (The old `am_vs_fm` added a real `noise_white(σ)` to *each* quadrature → total `2σ²`. The
AM/FM **ratio** is invariant to this scaling — both modulations face the same static — so the
lesson is unaffected; only the absolute MSE shifts by √2.)

Implementation: a new `Recipe`/distribution that samples two independent normals and packs a
`Complex`. The drawn `Dist` carries complex values (the RV graph gains a complex value channel, or
stores re/im as a paired node — pick whichever is least invasive in `dist.rs`).

**Queries.** `E` of a complex RV → a complex estimate (`E re + i·E im`); `Var` of complex =
`E|z − Ez|²` (a real). `P` still requires a bool. Both `E`/`Var` complex extensions can be
**deferred** — early on, query `math::re(z)` / `math::im(z)` separately.

---

## 6. The payoff: `am_vs_fm` rewritten

Keep the original (explicit `[I,Q]`) as the beginner on-ramp; add this complex variant to show the
payoff. The code becomes the textbook:

```noise
use math;   # i, exp, abs, arg
use rand;   # normal_complex
use vec;    # mse
use signal; # sine, sample

engine::set_max_samples(40000);

# A carrier is a spinning arrow (phasor) — a complex number z.
# AM writes the message into |z| (length); FM writes it into arg(z) (angle).
am_modulate(m)      = 1 + m;                      # message in the LENGTH:  (1+m) + 0i
fm_modulate(m, dev) = math::exp(math::i*dev*m);   # message in the ANGLE:   e^{i·dev·m}

am_demodulate(z)      = math::abs(z) - 1;         # arrow LENGTH, minus the carrier
fm_demodulate(z, dev) = math::arg(z) / dev;       # arrow ANGLE, undo the deviation

dev = 3; sigma = 0.3;
msg = signal::sample(0.3*signal::sine(3), 64);

# Static = circularly-symmetric complex Gaussian — radio noise by its real name. ONE object gives
# both quadratures, identical in every direction, so AM and FM face the same fight by construction.
static ~[Len(msg)] rand::normal_complex(sigma);

am_err = E(vec::mse(am_demodulate(am_modulate(msg)      + static),      msg));
fm_err = E(vec::mse(fm_demodulate(fm_modulate(msg, dev) + static, dev), msg));

Print("FM is", round(am_err/fm_err, 1), "x cleaner  (small-signal ideal: dev^2 =", dev**2, ")");
```

This also kills the original's prose caveat about "an independent set for EACH of the I and Q
quadratures" — circular symmetry is now a property of the *type*, not an assertion about how two
real noise lanes materialize.

**Unlocks for free** (future examples): Rayleigh/Rician **fading** (`h ~ rand::normal_complex(1)`, a
complex RV times the signal), the **FM threshold / SNR cliff**, and **QAM constellations** — all
awkward-to-impossible with hand-rolled `[I,Q]`.

---

## 7. Consistency: complex is a first-class scalar (the whole language, not just Shor)

If `complex` is a scalar type, then **every** numeric-polymorphic API owes it a defined behavior —
no silent partial support. Each operation falls into one of three buckets:

- **Lift component-wise** — the op is linear / structural, so it just works over ℂ:
  `vec::sum`, `vec::mean`, `vec::normalize`, `vec::transpose`, and the constructors
  (`ones`/`zeros`/`iota` stay real; `0*math::i` promotes). `E` of a complex RV lifts to a complex
  estimate. (Vector `+`/`-` and matvec/matmul are **not** `vec` functions — they are the `+`/`-`/`@`
  operators, already lifted in §3. There is no `vadd`/`vsub`/`matvec`.)
- **Magnitude-based → returns a real** — the op is defined through `|z|²`, so a complex input
  yields a **real** output: `vec::normsq` (`Σ|zᵢ|²`), `vec::norm`, `vec::mse` (`Σ|aᵢ−bᵢ|²/n`),
  `Var` (`E|z−Ez|²`). These are exactly what measurement probabilities and signal error need.
- **Deliberate type error — no meaning on ℂ** (consistent with operators rejecting `<` on complex):
  `vec::max`, `vec::min`, `vec::quantize`, the `Q` quantile query, and ordering comparisons. Error
  with a clear message, don't silently compare `|z|`.

**One real decision — `vec::dot`.** Two inequivalent things share the name:
- the **bilinear** product `Σ aᵢbᵢ` (what `@` / matmul does — *no* conjugation), and
- the **Hermitian** inner product `⟨a,b⟩ = Σ conj(aᵢ)·bᵢ` (what quantum/signal inner products mean).

Recommendation, matching numpy (`dot` vs `vdot`): **`vec::dot` stays bilinear** (consistent with
`@`), and we add **`vec::vdot`** (conjugating, Hermitian) plus **`vec::adjoint`** (conjugate
transpose, `conj`∘`transpose`) for the quantum/linear-algebra path. `math::sign(z)` similarly
generalizes to `z/|z|` (phase) or errors — defer; not load-bearing.

Test the contract directly: a program that runs `sum`/`mean`/`normsq`/`norm`/`mse` over a complex
array and asserts the real-vs-complex return types, plus one that asserts `max(complex_array)` is a
type error.

---

## 8. General language additions: `%`, `floor`/`ceil`, `map`

These earn their place on their own merits (clock/modular arithmetic, binning, angle-wrapping,
array transforms) — Shor's classical oracle just happens to exercise all three. They are **real-only**
(complex `%`/`floor` → type error; Gaussian-integer modulo is out of scope).

- **`%` operator** — modulo, at precedence level 7 (with `* / @`). Define as **floored** modulo:
  `a % b = a − b*floor(a/b)`, so the result has the sign of `b` and `x % n ∈ [0, n)` for `n > 0`
  (what modular arithmetic wants). IEEE edge cases follow `floor` (`x % 0 = NaN`, no panic).
- **`math::floor` / `math::ceil`** — complete the rounding family next to the existing `round`
  (nearest). Ufuncs: scalar, lifted over RVs, mapped over arrays. Add to the `eval.rs:1684` routing.
- **`map` / comprehension** — the genuine expressiveness gap: today arrays are built only from
  literals, `a..b`, `~[n]`, and `vec` constructors; there is **no way to build an array by applying
  a formula with control flow** over a range (`iota`+broadcast only covers pure-ufunc maps).

  Because Noise has **no closures / no first-class functions**, a higher-order `map(xs, f)` is
  awkward (and `f` couldn't capture outer state like `a`/`N`). The idiomatic realization is a
  **comprehension expression** that reuses the existing `for`/`in` machinery (already reserved
  words, already build-time-unrolled, and — crucially — the body sees the **outer environment**,
  unlike a pure function body):

  ```noise
  fx = [ modexp(a, N, x) for x in 0..Q ];          # body closes over a, N — no closure type needed
  evens = [ 2*k for k in 0..n ];                   # general use
  ```

  Optional filter form `[ expr for x in xs if cond ]` if cheap. This is a small `parser.rs` /
  `ast.rs` addition (a new `Comprehension` expr that desugars to a leaking-`for` accumulator). It
  supersedes the "Shor needs `map`" gap and is the recommended surface. (A restricted
  `map(xs, namedfn)` could come later if first-class function values are ever added.)

  Build-time cost is unchanged from a hand-written `for`: the comprehension unrolls to one node per
  element (so an oracle that itself loops is still O(Q²) for `Q` elements — flag in the demo).

---

## 9. Downstream: faithful small-scale quantum (Shor period-finding)

The complex type is the single biggest missing piece for *expressing* a quantum algorithm. Mapping
the obstacles from earlier discussion:

| Obstacle | After this plan |
|---|---|
| reals only → **no interference** | ✅ fixed — complex amplitudes cancel; this was *the* blocker |
| 2ⁿ state in build-time arrays | ⚠️ unchanged — exponential, fine for small N |
| no sequential state | ✅ non-issue — amplitude evolution is *deterministic*; a `for` fold over gates is legal |
| measure ∝ \|ψ\|² | 🔶 expressible via `count(cumsum<U)`+gather; wants a clean `categorical` |

### The corrected period-finding core

Using the **principle of deferred measurement** (don't measure the work register — trace it out),
the core is pure complex linear algebra:

```noise
# f(x) = a^x mod N is the classical oracle; fx = [f(0) … f(Q-1)], Q = 2^t
Psi  = onehot(fx, W) / Q**0.5;     # Q×W state, one-hot per row at column f(x)
Ahat = QFT(Q) @ Psi;               # complex matmul — inverse-QFT the counting index
probs = rownormsq(Ahat);           # P(y) = Σ_w |Ahat[y][w]|²  (trace out work register)
y ~ categorical(probs);            # measure → y ≈ k·Q/r ; continued fractions → period r
```

(An earlier sketch used `work_weights`/`keep_where` to *measure the work register first* — those
were artifacts of a worse formulation and are deleted by deferred measurement.)

### Remaining gaps — all are general features (§7–8), none Shor-specific

Everything the quantum core needs is a general language addition covered above, plus two more
linear-algebra/distribution primitives that are equally general:

1. **Complex type + `math::` ufuncs** — §2–4 (`exp`, `abs`, complex `@`).
2. **`%`, `math::floor`, comprehensions** — §8, for the classical oracle `modexp(a, N, x)` and
   `fx = [modexp(a, N, x) for x in 0..Q]`.
3. **`vec::outer` / `(Q,1)·(1,W)` broadcast** — builds `QFT(Q)` (outer product of `iota(Q)` through
   complex `exp`) and `onehot(fx, W)`. A general linear-algebra op; without it these matrices need
   O(Q²) build-time unroll. The matmul itself is one vectorized `@`.
4. **`rand::categorical(weights)`** — sample an index ∝ weights; the honest measurement op (and a
   generally useful distribution). Expressible *today* as `count(cumsum(w) < U)` + the interpreter's
   random gather, but a named primitive is cleaner.

None of these touch the type system again; each stands on its own outside Shor.

**Unmovable caveat:** this is a 2ⁿ *simulation* — gorgeous for N=15 (the inverse-QFT peak comb is a
killer scrollytelling demo) but never *efficient*. "Efficient Shor" would require BQP ⊆ BPP.

---

## 10. Milestones

**Track A — complex, end to end (delivers the AM/FM win):**

1. **Complex core** — `Value::Complex`, operator lifting (`+ - * / ** @`, no order), `math::i`/`j`.
   Test: `2 + 3*math::i`, complex `@`.
2. **`math::` ufuncs** — `exp`, `abs`, `sqrt`(extend), `arg`, `conj`, `re`, `im`, branch-by-type.
   Test: `math::exp(math::i*pi) ≈ -1` (Euler).
3. **`vec` consistency pass (§7)** — make every numeric API complex-correct: lift
   (`sum`/`mean`/`normalize`/`transpose`; the `+`/`-`/`@` operators are §1), magnitude→real
   (`normsq`/`norm`/`mse`), or error (`max`/`min`/`quantize`/`Q`). Add `vdot`/`adjoint`; keep `dot`
   bilinear. Test: the type-contract program from §7.
4. **`rand::normal_complex`** + `~[n]` draws; `math::re`/`im` queries (complex `E`/`Var` optional).
5. **Ship `am_vs_fm` complex variant** (keep the original). Acceptance test for 1–4.

**Track B — general expressiveness (independent; §8 + quantum follow-on):**

6. **`%` operator + `math::floor`/`ceil`** — real-only; floored-modulo semantics. General-purpose.
7. **Comprehensions** `[expr for x in xs (if cond)]` — `ast.rs`/`parser.rs`, desugar to leaking-`for`.
8. **`vec::outer` + `rand::categorical`** — general linear-algebra/distribution primitives.
9. **Quantum demo** — small-N period-finding (§9), reusing 1–4 and 6–8. The 2ⁿ-simulation payoff.

Track A is self-contained and ships the radio win. Track B is general language work that any example
can use; the quantum demo (9) is its capstone, not its justification.
