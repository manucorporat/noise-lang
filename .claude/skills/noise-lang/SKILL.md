---
name: noise-lang
description: Write correct, idiomatic Noise — an expression-based probabilistic language where every value is a probability distribution and you compute probabilities / expectations by Monte Carlo (run via the `noise` CLI). Use when authoring or editing .noise programs, or when modeling a probability / Monte-Carlo question in Noise.
---

# Writing Noise

Noise is a small, expression-based **probabilistic** language: every value is a probability
distribution, and you compute probabilities and expectations by Monte Carlo. A program is a
sequence of `;`-separated statements and its result is the value of the **last** statement. This
guide is self-contained — everything you need to write correct, idiomatic Noise is below.

## Run it

```sh
cargo run -p noise-cli -- file.noise   # run a program; prints the LAST statement's value
cargo run -p noise-cli                 # REPL (one line at a time, persistent env)
```

(If a `noise` binary is installed, `noise file.noise` / `noise` work the same way.)

## The mental model

Four load-bearing rules. Internalize these and the rest follows.

1. **Everything is a distribution.** A number is just the degenerate case — a *point mass*
   (`mean(5)=5`, `variance(5)=0`, `P(5>3)=1`). Operators map distributions to distributions
   uniformly, so `P`, `E`, `Var`, `Q` apply to anything.

2. **`~` draws; `=` transforms.** `name ~ dist` is a **stochastic node** — it draws a fresh random
   variable, and `~` is the *only* thing that draws. `name = expr` is a **deterministic node** — a
   transform, a constant, or an undrawn *recipe*. **You cannot do arithmetic on an undrawn
   recipe:**
   ```noise
   unif(0, 1) + 3        # ERROR — unif(...) is a recipe, not a number
   X ~ unif(0, 1); X + 3 # correct — draw with ~, then transform with =
   ```
   `unif`, `unif_int`, `bernoulli`, `normal`, … all return an *undrawn recipe*. Bind a recipe with
   `=` to name it (`Die = unif_int(1, 6)`), then draw it with `~` (`a ~ Die`).

3. **One name = one fixed node.** Every mention of a name is the *same* draw, exactly like `X` in
   math. So `X + X` is `2X`, `X - X` is exactly `0`, and `P(X == 6 && X > 3)` uses one `X`. There
   is **no re-draw on reuse.**

4. **Independence is explicit.** Two independent draws come from two `~` declarations, or from the
   shaped draw `~[n] dist` — never from repeating a name.
   ```noise
   A ~ unif_int(1, 6); B ~ unif_int(1, 6)   # two independent dice
   dice ~[2] unif_int(1, 6)                  # same thing, as a length-2 array
   ```

Nothing is sampled until a query (`P`/`E`/`Var`/`Q`) forces it; everything upstream stays symbolic.

## Language reference

**Lexical.** Whitespace is insignificant (separates tokens). Comments run to end of line, started
by `#` or `//`. Numbers are `f64` integer or decimal literals — **no exponent syntax**, and a
leading `-` is the unary-minus operator, not part of the literal. Identifiers are
`[A-Za-z_][A-Za-z0-9_]*`, case-sensitive. Reserved words: `if else for in use true false`.
Strings are double-quoted with **no escape sequences** (`"like this"`); they are a label/utility
type for `Print` and messages — a string can never enter a random-variable expression.

**Operators** (precedence low → high; all left-associative except `^` and binding, which are
right-associative; prefix `-`/`!` bind tighter than everything below `^`, so `-2 ^ 2 == -4`):

| Level | Operators | Meaning |
|------:|-----------|---------|
| 1 | `=` `~` | binding (right-assoc) |
| 2 | `..` | half-open integer range `[a, b)` |
| 3 | `\|\|` | logical or |
| 4 | `&&` | logical and |
| 5 | `== != < > <= >=` | comparison |
| 6 | `+ -` | add / subtract (`+` also concatenates if either side is a string) |
| 7 | `* / @` | multiply / divide (elementwise/broadcast) · `@` = matrix product |
| 8 | `^` | power (right-assoc) |
| 9 | prefix `- ! ~` | negate · logical not · draw |
| 10 | postfix `[index]` | index (repeatable: `M[i][j]`) |
| 11 | call, `()` grouping, `{}` blocks | |

**Types & rules** (every value is conceptually a distribution; these are the runtime forms):
`number` (point mass), `bool` (point mass on {T,F}), `string`, `unit` `()`, `array` (fixed-length,
known at build time), `dist` (a drawn random variable / distribution handle), `estimate` (a number
carrying a standard error, produced by `P`/`E`/`Var`), and `signal` (a lazy waveform).

- Arithmetic `+ - * / ^` need numbers → number (division & `^` are IEEE-754: `1/0 == inf`, no
  panic). With a `dist` operand the op **lifts** to a `dist`; pure-constant subexprs fold eagerly.
- `+` **concatenates** when either side is a string (the other is stringified): `"x = " + 5`.
- Ordering `< > <= >=` need two numbers → bool. Equality `== !=` compares two values of the *same*
  primitive type → bool (mixed-type is an error).
- Prefix `-` needs a number; prefix `!` needs a bool.
- `if cond { a } else { b }` — `cond` must be a `bool` or a bool random variable (see "lifted if").
- **No implicit coercions.** Type mismatches are runtime errors with a source span.
- **Scope is flat:** blocks do **not** introduce a new scope — bindings made inside a block remain
  visible afterwards. (This is how `for`-loop accumulators work.)

## Modules & `use`

`builtin` is **always active**: `P`, `Q`, `E`, `Var`, `Print`, `Len` (capitalized). Everything
else is **strict** — a bare name errors until you `use` its module (or write the `mod::name`
path). Start each program with the `use` lines you need.

| Module    | `use`?   | Items |
|-----------|----------|-------|
| `builtin` | always   | `P`, `Q`, `E`, `Var`, `Print`, `Len` |
| `rand`    | `use rand;` | `unif`, `unif_int`, `bernoulli`, `normal`, `normal_int`, `normal_complex`, `exponential`, `exponential_int`, `poisson`, `geometric`, `categorical`, `rotation`, `permutation` |
| `math`    | `use math;` | `pi`, `e`, `i`/`j` (imaginary unit), `sqrt`, `exp`, `abs`, `arg`, `conj`, `re`, `im`, `floor`, `ceil`, `round`, `log` (natural), `log10`, `sin`, `cos`, `atan`, `sign`, `gcd`, `modpow` |
| `vec`     | `use vec;`  | `sum`, `count`, `any`, `all`, `max`, `min`, `mean`, `dot`, `vdot`, `normsq`, `norm`, `transpose`, `adjoint`, `normalize`, `outer`, `quantize`, `has_duplicates`, `count_duplicates`, `mse`, `ones`, `zeros`, `iota` |
| `signal`  | `use signal;` | `sine`, `cosine`, `sample`, `noise_white`, `noise_brown`, `noise_pink`, `noise_ou` |

```noise
use rand;            # unif, unif_int, …
math::sqrt(2)        # or reach one item by its full path, no `use` needed
```

A user definition shadows a module item of the same name. Module paths are single-level
(`mod::name`); a constant path resolves a value (`math::pi`), a function path must be called.

## The standard workflow

recipe (`=`) → draw (`~` / `~[n]`) → transform (`=`) → query (`P`/`E`/`Var`/`Q`) → `Print`.

```noise
use rand;   # unif_int
use vec;    # has_duplicates

n     = 23;
bday  = unif_int(1, 365);   # a recipe
days  ~[n] bday;            # n independent draws
match = has_duplicates(days);
Print("P(shared birthday among", n, ") =", P(match))
```

## Distributions & queries

**Distributions** (all `rand`, all return recipes; draw with `~`):

- `unif(a, b)` — continuous uniform on `[a, b)`. **Continuous → never use `==` on it.**
- `unif_int(a, b)` — discrete uniform on `a..=b` *inclusive*. Use this for dice/coins/counts.
- `bernoulli(p)` — `true` with probability `p` (a bool-RV).
- `normal(mu, sigma)`, `exponential(rate)` (`mean = 1/rate`), `poisson(lambda)`, `geometric(p)`
  (failures before first success). **Note:** the exponential *distribution* is `rand::exponential`;
  `exp` is the exponential *function* `math::exp`.
- `_int` family — `normal_int`, `exponential_int` round each draw to the nearest integer (so
  `==`/counts are meaningful). `unif_int` is already discrete.
- `normal_complex(sigma)` — a circularly-symmetric complex Gaussian (`E|z|² = sigma²`); a complex
  RV. `categorical(weights)` — sample an index ∝ weights (`y ~ rand::categorical(probs)`).
- `rotation(d)` — a fresh random `d×d` orthonormal matrix per sample (Haar rotation).
- **Random parameters (hierarchical models).** A parameter can itself be a random variable:
  `p ~ unif(0,1); k ~ bernoulli(p)`. Supported for `unif`/`unif_int`/`normal`/`normal_int`/
  `exponential`/`exponential_int`/`bernoulli` (not yet `poisson`/`geometric`/`normal_complex`). Two
  draws of the same parameterized recipe are independent **given** the parameter. Combine with the
  `|` bar for (rejection) Bayesian inference: `E(bias | count(flips) == 7)` is a posterior mean.

**Queries** (all `builtin`; default `n = 1e6` samples, fixed seed → reproducible):

- `P(event[, n])` — probability a bool-RV is true. Returns an **estimate** carrying its standard
  error: it **self-rounds to the digits the sample size justifies**, and the error **propagates
  through arithmetic** (`4 * P(C)` shows one fewer digit). Pass a bigger `n` to reveal more digits.
  `P` of a non-event (numeric) is an error.
- `E(x[, n])` / `Var(x[, n])` — expectation / variance of a numeric (or bool) quantity. `E` of a
  bool equals `P`.
- `Q(x, q[, n])` — quantile (inverse CDF): `Q(X, 0.5)` median, `Q(X, 0.95)` 95th pct, `Q(X, 0)` /
  `Q(X, 1)` min/max draw. Returns a plain number.
- **`event | given` — conditioning** (Bayes, scoped to one query, no `observe`/side effect):
  `P(A | C)` is "P(A) given C holds", `E(X | C)` / `Q(X | C, q)` likewise. The `|` binds looser than
  everything (below `||`). `given` must be an event; for `P`, the left side must be an event too.
  `X | C` is also a **first-class value**: bind it (`hi = D | D > 3`), query it later (`P(hi < 5)`),
  and operate on it (`2*(X|C)+1` is `(2X+1) | C`). You **cannot** combine two values conditioned on
  *different* events — condition once, at the end (`(X + Y) | C`).
- `Print(args…)` — space-separated, then newline; combine with string `+` and `round(x, d)`.
- `Len(xs)` — element count of an array (a build-time constant).

## Collections, arrays & idioms

**Arrays** are fixed-length and known at build time. Build them with literals (`[1, 2, 3]`, `[]`),
the range `a..b` (half-open: `0..n` is `0 … n-1`), the shaped draw `~[n] d`, or `vec`
constructors (`ones(n)`, `zeros(n)`, `iota(n)`). Index with `xs[i]` (chains: `M[i][j]`); the index
must be a **constant non-negative integer in range** — never a random variable. There is no
append/push.

**Arithmetic broadcasts** over arrays (NumPy-style, nesting for matrices):
`[1,2,3] + [10,20,30]` → `[11,22,33]`, `1 + [1,2,3]` → `[2,3,4]`, `[1,2,3] ^ 2` → `[1,4,9]`.
The `@` operator is the **matrix product** (`v @ w` dot, `M @ v` matvec, `M @ N` matmul); `*`
stays elementwise. `sin`/`cos`/`atan` are ufuncs (scalar, lifted over RVs, or mapped over arrays).

**Shaped draws & reducers.** `~[n] d` is `n` iid draws (an array); `~[n, m] d` a matrix. Fold with
a `vec` reducer — independence becomes a one-liner:

```noise
use rand; use vec;
dice  ~[2] unif_int(1, 6); Print("P(sum==7) =", P(sum(dice) == 7))      # two dice
flips ~[3] bernoulli(0.5); Print("P(all heads) =", P(all(flips)))       # 3-coin streak
flips ~[3] bernoulli(0.5); Print("P(exactly 2) =", P(count(flips)==2))  # count of true
parts ~[3] bernoulli(0.9); Print("uptime =", P(any(parts)))             # at-least-one
```

**Lifted `if` = per-lane select** (a value, not control flow; `else` required, both branches
evaluated and reuse the condition's per-lane draws). Gives `max`/`min`/`abs` over RVs for free:

```noise
A ~ unif_int(1, 6); B ~ unif_int(1, 6);
higher = if A > B { A } else { B };     # max of two dice
Print("P(higher==6) =", P(higher == 6))
```

**Ranges & `for` (build-time unroll).** `for x in xs { }` runs the body once per element; **bindings
leak** (blocks don't scope) — that's exactly how an accumulator persists. Each `~` inside the body
is a *distinct* node, so it's a clean way to make many independent draws:

```noise
use vec;
acc = 0; for x in 1..5 { acc = acc + x }; acc      # 1+2+3+4 = 10
```

**Comprehensions** `[for x in xs { body }]` build an array — it's the `for x in xs { body }` loop
wrapped in `[ ]` so each body value is collected. The body **closes over outer variables** (Noise
has no closures, so this is how you "map with captured state"). Use **`continue`** to skip an
element — that's how you *filter*:

```noise
a = 7; N = 15;
fx = [for x in 0..6 { (a ^ x) % N }];                  # body closes over a, N
evens = [for x in 0..10 { if x % 2 != 0 { continue }; x }];  # filter via continue
```

`continue` skips the rest of the loop body (in a `for` loop it drops that iteration's side effects;
in a comprehension it omits the element). The skip condition must be deterministic.

**`%` (floored modulo)** is `a − b·floor(a/b)`, so it takes the sign of `b` and `x % n ∈ [0, n)`
for `n > 0` (clock/modular arithmetic): `-1 % 3 == 2`. `math::floor` / `math::ceil` round (real-only).

**User functions.** `f(a) = expr` is pure (lifts over RVs); `f() ~ dist` draws fresh per call:

```noise
use rand;
max(a, b) = if a > b { a } else { b };   # pure
roll() ~ unif_int(1, 6);                 # fresh draw each call
P(roll() + roll() == 7)                  # two INDEPENDENT rolls
```
Functions are **pure in their parameters** — the body sees only its args (plus `pi`/`e`), no outer
variables, no closures. Calls unroll at build time, so recursion must terminate.

**Conditional probability** uses the `|` bar (Bayes, scoped to the query) — prefer it over the
hand-written ratio:

```noise
use rand;
D ~ unif_int(1, 6);
Print("P(D==6 | D>3) =", P(D == 6 | D > 3))               # = 1/3 (≡ P(A && C) / P(C))
hi = D | D > 3;                                           # a conditioned value — bind & reuse
Print("E(roll | >3)  =", E(hi), " median:", Q(hi, 0.5))   # 5, 5
```

**Signals (lazy waveforms).** `signal::sine(f)` / `cosine(f)` describe a waveform by frequency
(O(1) memory). Scalar/trig ops defer into the signal; it **materializes** to an array when it meets
a sized array (adopting its length) or via `signal::sample(sig, n)`. `noise_white(sigma)` is a lazy
*random* generator (materializes to fresh iid `normal(0, sigma)` draws). `sine(n, f)` is shorthand
for `sample(sine(f), n)`.

```noise
use signal;
msg = sample(0.3 * sine(3), 64);   # render the lazy tone at 64 points
```

**Complex numbers.** `complex` is a first-class scalar. There's no literal — it **emerges** from
`math::i` (alias `math::j`, the unit `0 + 1i`) plus the ordinary operators, or from
`rand::normal_complex(sigma)` (a circularly-symmetric complex-Gaussian RV). Real promotes to
`re + 0i`; a pure-real expression stays a number. `* / ^` are true complex ops; **ordering
(`< > <=`) and `%` are type errors** on ℂ. Functions branch by type: `math::exp` (Euler
`e^{iθ} = cos θ + i sin θ`), `math::abs`/`math::arg` (magnitude/phase, real out), `math::conj`,
`math::re`/`math::im`, `math::sqrt`. In `vec`: `normsq`/`norm`/`mse` are magnitude-based (real out),
`dot` is bilinear while `vdot` conjugates (Hermitian), plus `outer` and `adjoint`.

```noise
use math; use rand; use vec;
z = 2 + 3*math::i;                       # complex emerges from math::i
math::abs(z); math::arg(z);              # magnitude & phase (reals)
math::exp(math::i * math::pi)            # ≈ -1  (Euler's identity)
static ~[64] rand::normal_complex(1);    # 64 iid complex-Gaussian static samples
```

## Hazards — what NOT to do

- **`==` on a continuous RV is ~never true.** `unif(1,6) == 4` ≈ 0. Use a discrete distribution
  (`unif_int`, `bernoulli`, `*_int`) whenever you compare for equality or count.
- **Arithmetic on an undrawn recipe is an error.** Draw with `~` first (rule 2 above).
- **No closures.** Function bodies see only parameters and the `pi`/`e` constants.
- **Indices must be constant non-negative integers**, known at build time — never a random
  variable.
- **Arrays are fixed-length.** No `push`/append. Build with literals, `a..b`, `~[shape]`, or the
  `vec` constructors.
- **Blocks don't scope** — bindings made inside leak out (there is one flat environment).
- **No sequential / stateful processes.** A random walk, Markov chain, or M/M/1 queue
  (`W_{n+1} = max(0, W_n + S_n − A_{n+1})`) is *not* expressible — the engine samples independent
  lanes that can't carry state across a time index. A lifted `if` picks a value per lane; it can't
  thread state across steps.
- **Each query samples its own pass.** `P(A)`, `P(B)`, `P(A && B)` are estimated independently, so
  exact cross-query consistency (`P(A && B) ≤ P(A)`) is not guaranteed. (Inside *one* conditional
  query, event and condition share a pass — so `P(A | C)` is internally consistent.)
- **Conditioning is rejection-based.** `P(A | C)` keeps the lanes where `C` happened, so its error
  uses the in-condition count `m ≈ n·P(C)` — fine when `P(C)` isn't tiny. It is **not** posterior
  inference: conditioning on a continuous measurement (`X == 4.7`, probability ~0) or a rare event
  won't work; that's the separate inference track.

## Style conventions

- Open with a comment stating the question and, where one exists, the analytic answer to check
  against.
- Use **readable, named intermediate steps**, not nested one-liners (`days`, `match`, `total`,
  `higher` — one idea per binding).
- Put the needed `use` lines at the top.
- **End in `Print(...)`**, building the message with string `+` and `round(x, d)`.

## Pre-flight checklist

- [ ] Every random variable is introduced with `~` (no arithmetic on a bare recipe).
- [ ] Independence comes from `~[n]` or separate `~` — not from repeating a name.
- [ ] Equality / counting uses a **discrete** distribution (`unif_int`/`bernoulli`/`*_int`).
- [ ] `use` lines present for every non-`builtin` name; queries are capitalized (`P`/`E`/`Var`/`Q`/`Print`/`Len`).
- [ ] No sequential-state recurrence, no random index, no expectation of block scoping.
- [ ] Program ends in `Print(...)`; run it and sanity-check against the analytic value.
