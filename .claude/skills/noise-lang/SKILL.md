---
name: noise-lang
description: Write correct, idiomatic Noise — an expression-based probabilistic language where every value is a probability distribution and you compute probabilities / expectations by Monte Carlo. A .noise program compiles to a literate document — a short, live, optionally interactive article. Use when authoring or editing .noise programs, or when modeling a probability / Monte-Carlo question in Noise.
---

# Writing Noise

Noise is a small, expression-based **probabilistic** language: every value is a probability
distribution, and you compute probabilities and expectations by Monte Carlo. A program is a
sequence of statements, one per line — and when compiled, it is rendered as a **document**: a
short live article with prose, code cells, typeset math, plots, and interactive controls, in the
spirit of a really good Jupyter notebook. You are always writing two things at once: a correct
model, and a readable article. This guide is self-contained — the language first, then how to
make the compiled document read well.

## Run it

```sh
cargo run -p noise-cli -- file.noise   # run a program; renders the document (prose + results) to the terminal
cargo run -p noise-cli                 # REPL (one line at a time, persistent env)
```

(If a `noise` binary is installed, `noise file.noise` / `noise` work the same way. The web
playground renders the same document with full typography: abstract, margin notes, LaTeX,
sliders, charts.)

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
   unif(0, 1) + 3        // ERROR — unif(...) is a recipe, not a number
   X ~ unif(0, 1); X + 3 // correct — draw with ~, then transform with =
   ```
   `unif`, `unif_int`, `bernoulli`, `normal`, … all return an *undrawn recipe*. Bind a recipe with
   `=` to name it (`Die = unif_int(1, 6)`), then draw it with `~` (`a ~ Die`).

3. **One name = one fixed node.** Every mention of a name is the *same* draw, exactly like `X` in
   math. So `X + X` is `2X`, `X - X` is exactly `0`, and `P(X == 6 && X > 3)` uses one `X`. There
   is **no re-draw on reuse.**

4. **Independence is explicit.** Two independent draws come from two `~` declarations, or from the
   shaped draw `~[n] dist` — never from repeating a name.
   ```noise
   A ~ unif_int(1, 6); B ~ unif_int(1, 6)   // two independent dice
   dice ~[2] unif_int(1, 6)                  // same thing, as a length-2 array
   ```

Nothing is sampled until a query (`P`/`E`/`Var`/`Q`) forces it; everything upstream stays symbolic.

## The literate layer — a program compiles to an article

A run produces one flat, ordered document: frontmatter meta, then blocks (code cells, prose
notes, plots, input controls), plus a margin-comment layer. The pieces:

**Frontmatter** — an optional `---`-fenced YAML block at the very top with `title`, `abstract`,
and `tags`. It renders as the article's title, abstract paragraph, and keywords line. **The
abstract IS the article's introduction** — hook, setup, and the headline claim.

```noise
---
title: "Estimate π"
abstract: >
  Scatter darts at random across a square and count how many land inside a circle
  drawn in it. That fraction alone is enough to recover π.
tags: [basics, monte carlo]
---
```

**Code cells** — consecutive top-level statements with no blank line between them render as one
code block. A **blank line splits cells**; a template statement also splits. Comments never
split a cell. Group statements into cells deliberately: one cell = one step of the story.

**Templates — prose and math blocks.** A triple-backtick fence with a syntax tag, written as a
*statement*, emits a rendered block at that point in the document:

    ```md
    **The verdict.** Rejecting the first ${r} of ${n} applicants wins **${p_win * 100}%** of the time.
    ```
    ```latex
    P(\text{win}) = \frac{r}{n}\sum_{k=r}^{n-1}\frac{1}{k} \approx ${p_win}
    ```

- ` ```md ` renders as body prose (full Markdown: **bold**, *italics*, inline code).
- ` ```latex ` renders as display math (KaTeX/LaTeX source).
- `${expr}` holes interpolate any Noise expression, evaluated in the current scope. An
  **estimate** value (from `P`/`E`/`Var`) prints self-rounded to its justified digits.
- A single-backtick template `` `…` `` is the inline/untagged form; in expression position a
  template is just a string value.

**Comments become margin notes.** A `//` comment run attaches to the code cell below it and
renders as a *side note* in the margin — physically small and off to the side. That placement is
the design constraint: margin notes are for asides, never for the narrative.

**Inputs — interactive controls.** `input::real(…)` / `input::int(…)` / `input::bool(…)` declare
a host-tunable parameter inline, with named args `min:`, `max:`, `step:`, `default:`, and
optional `label:`/`name:`. The call evaluates to its current value (a plain number/bool), and
the document renders a slider/checkbox at that point. Changing it re-runs the program — every
`${…}` hole downstream updates. Use one wherever the reader should ask "what if I change this?"

```noise
r = input::int(min: 1, max: 19, step: 1, default: 7)
```

**Plots** — `plot::histogram/line/scatter/heatmap/corr/fan/…` push chart cards into the document
at their statement's position.

## Language reference

**Statements.** A statement normally ends at the end of its line — do NOT write trailing
semicolons. `;` exists for exactly two cases: separating several statements on one line
(`p = 1; q = N`, a one-line block body `{ acc = 0; for … }`), and guarding the one ambiguity —
a line that *starts* with `[` continues the previous expression as an *index*, so a statement
before a line like `[for p in QFT @ Psi { … }]` or `[p, q]` must end in `;`. Multi-line
expressions (arrays, function bodies, chained `+`/`*` continuations) work without any marker.

**Lexical.** Whitespace is insignificant (separates tokens). Comments are `//` (to end of line)
or `/* … */` (block, non-nesting) — `#` is **not** a comment marker (only a `#!` shebang on
line 1 is allowed). Numbers are `f64` integer or decimal literals — **no exponent syntax**, and a
leading `-` is the unary-minus operator, not part of the literal. Identifiers are
`[A-Za-z_][A-Za-z0-9_]*`, case-sensitive. Reserved words: `if else for in use true false`.
Strings are double-quoted with **no escape sequences** (`"like this"`); they are a label/utility
type — a string can never enter a random-variable expression.

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

`builtin` is **always active**: `P`, `Q`, `E`, `Var`, `Len` (capitalized). Everything
else is **strict** — a bare name errors until you `use` its module (or write the `mod::name`
path). Start each program with the `use` lines you need.

| Module    | `use`?   | Items |
|-----------|----------|-------|
| `builtin` | always   | `P`, `Q`, `E`, `Var`, `Len` |
| `rand`    | `use rand;` | `unif`, `unif_int`, `bernoulli`, `normal`, `normal_int`, `normal_complex`, `exponential`, `exponential_int`, `poisson`, `geometric`, `categorical`, `empirical`, `block_bootstrap`, `rotation`, `permutation` |
| `math`    | `use math;` | `pi`, `e`, `i`/`j` (imaginary unit), `sqrt`, `exp`, `abs`, `arg`, `conj`, `re`, `im`, `floor`, `ceil`, `round`, `log` (natural), `log10`, `sin`, `cos`, `atan`, `sign`, `gcd`, `modpow` — `exp`/`log`/`log10` lift over RVs like `sin`/`cos` |
| `vec`     | `use vec;`  | `sum`, `prod`, `count`, `any`, `all`, `max`, `min`, `mean`, `cumsum`, `cumprod`, `cummax`, `cummin`, `dot`, `vdot`, `normsq`, `norm`, `transpose`, `adjoint`, `normalize`, `outer`, `quantize`, `onehot`, `has_duplicates`, `count_duplicates`, `mse`, `ones`, `zeros`, `iota` |
| `signal`  | `use signal;` | `sine`, `cosine`, `sample`, `noise_white`, `noise_white_complex`, `noise_brown`, `noise_pink`, `noise_ou` |
| `plot`    | path-only | `histogram`, `line`, `scatter`, `heatmap`, `corr`, `fan` (quantile-band cone of a path), `explain`, `value` — write the path (`plot::fan(...)`); charts are pushed into the document at their statement's position |
| `stats`   | path-only | `histogram(x[, bins])` → `[[midpoints],[counts]]`, `quantiles(x, [q…])`, `moments(x)` → `[n, mean, sd, min, max]`, `fan(path)` → 6×cols (`q05,q25,q50,q75,q95,mean`), `corr(a, b)` → number / `corr(v)` → n×n matrix. The numbers behind the `plot::` charts — same computation, so `stats::quantiles(x, [0.05])[0]` *is* the `q05` the card prints. Forces sampling; takes `x | cond`. |
| `input`   | path-only | `real`, `int`, `bool` — inline tunable parameters (see the literate layer above) |
| `engine`  | `use engine;` (or path) | `set_max_samples`, `set_max_opts`, `set_resolution` |

```noise
use rand            // unif, unif_int, …
math::sqrt(2)        // or reach one item by its full path, no `use` needed
```

A user definition shadows a module item of the same name. Module paths are single-level
(`mod::name`); a constant path resolves a value (`math::pi`), a function path must be called.

## The standard workflow

recipe (`=`) → draw (`~` / `~[n]`) → transform (`=`) → query (`P`/`E`/`Var`/`Q`) → present
through ```md / ```latex templates (see "Writing the article"). In the REPL the last statement's
value prints by itself.

    use rand   // unif_int
    use vec    // has_duplicates

    class_size = 23
    birthday   = unif_int(1, 365)        // a recipe
    birthdays  ~[class_size] birthday    // class_size independent draws
    shared     = has_duplicates(birthdays)
    p_shared   = P(shared)

    ```md
    A class of ${class_size} shares a birthday **${p_shared * 100}%** of the time.
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
- `empirical(xs)` — the **iid bootstrap**: each draw is a uniformly random element of the
  constant numeric array `xs` (resample history instead of assuming a distribution). A true
  recipe: draw with `~`; `~[n]` gives `n` iid resamples. `block_bootstrap(xs, b)` — the
  **moving-block bootstrap** for autocorrelated series: one draw is an *array* of `Len(xs)`
  values glued from random contiguous length-`b` blocks of `xs`, keeping streaks intact. Both
  need a flat, non-empty, constant numeric array (`1 <= b <= Len(xs)`) and run interpreter-only
  (they gather).
- **`math::exp` / `math::log` / `log10` lift over RVs** — `Z ~ normal(mu, sigma); exp(Z)` is a
  lognormal, `E(log(1 + f*R))` is Kelly log-growth. A per-lane bad value follows IEEE
  (`log` of `x < 0` → NaN, of `0` → −inf); a *deterministic* `log(0)` is still a friendly
  build-time error.
- **Random parameters (hierarchical models).** A parameter can itself be a random variable:
  `p ~ unif(0,1); k ~ bernoulli(p)`. Supported for `unif`/`unif_int`/`normal`/`normal_int`/
  `exponential`/`exponential_int`/`bernoulli` (not yet `poisson`/`geometric`/`normal_complex`). Two
  draws of the same parameterized recipe are independent **given** the parameter. Combine with the
  `|` bar for (rejection) Bayesian inference: `E(bias | count(flips) == 7)` is a posterior mean.

**Queries** (all `builtin`; default `n = 1e6` samples, fixed seed → reproducible):

- `P(event[, n])` — probability a bool-RV is true. Returns an **estimate** carrying its standard
  error: it **self-rounds to the digits the sample size justifies**, and the error **propagates
  through arithmetic** (`4 * P(C)` shows one fewer digit). Pass a bigger `n` to reveal more digits.
  A fractional second argument (`P(hit, 1e-4)`) asks for **precision** instead of a sample count —
  the engine keeps sampling until the answer is that good. `P` of a non-event (numeric) is an error.
- `E(x[, n])` / `Var(x[, n])` — expectation / variance of a numeric (or bool) quantity. `E` of a
  bool equals `P`.
- `Q(x, q[, n])` — quantile (inverse CDF): `Q(X, 0.5)` median, `Q(X, 0.95)` 95th pct, `Q(X, 0)` /
  `Q(X, 1)` min/max draw. Returns an estimate (self-rounding, like `P`/`E`/`Var`) whose standard
  error comes from the order-statistic band — density-free.
- **`event | given` — conditioning** (Bayes, scoped to one query, no `observe`/side effect):
  `P(A | C)` is "P(A) given C holds", `E(X | C)` / `Q(X | C, q)` likewise. The `|` binds looser than
  everything (below `||`). `given` must be an event; for `P`, the left side must be an event too.
  `X | C` is also a **first-class value**: bind it (`hi = D | D > 3`), query it later (`P(hi < 5)`),
  and operate on it (`2*(X|C)+1` is `(2X+1) | C`). You **cannot** combine two values conditioned on
  *different* events — condition once, at the end (`(X + Y) | C`).
- `Len(xs)` — element count of an array (a build-time constant).

## Collections, arrays & idioms

**Arrays** are fixed-length and known at build time. Build them with literals (`[1, 2, 3]`, `[]`),
the range `a..b` (half-open: `0..n` is `0 … n-1`), the shaped draw `~[n] d`, or `vec`
constructors (`ones(n)`, `zeros(n)`, `iota(n)`). Index with `xs[i]` (chains: `M[i][j]`); the index
is normally a **constant non-negative integer in range** — a *random* numeric index lifts to a
per-lane **gather** (each lane picks its own element; interpreter-only, not codegen-eligible). There is no
append/push. A deterministic **array** of indices **slices**: `xs[0..r]` takes the first `r`
elements (so `max(quality[0..r])` is "the best of the first r"), `xs[[2, 0, 0]]` reorders/repeats,
`xs[perm]` applies a deterministic permutation — sugar for `[for i in inds { xs[i] }]`; random
indices inside the array are an error.

**Arithmetic broadcasts** over arrays (NumPy-style, nesting for matrices):
`[1,2,3] + [10,20,30]` → `[11,22,33]`, `1 + [1,2,3]` → `[2,3,4]`, `[1,2,3] ^ 2` → `[1,4,9]`.
The `@` operator is the **matrix product** (`v @ w` dot, `M @ v` matvec, `M @ N` matmul); `*`
stays elementwise. `sin`/`cos`/`atan`/`exp`/`log`/`log10` are ufuncs (scalar, lifted over RVs, or
mapped over arrays).

**Shaped draws & reducers.** `~[n] d` is `n` iid draws (an array); `~[n, m] d` a matrix. Fold with
a `vec` reducer — independence becomes a one-liner:

```noise
use rand; use vec
dice  ~[2] unif_int(1, 6);  p_seven    = P(sum(dice) == 7)       // two dice
flips ~[3] bernoulli(0.5);  p_streak   = P(all(flips))           // 3-coin streak
p_two_heads = P(count(flips) == 2)                               // count of true
parts ~[3] bernoulli(0.9);  p_uptime   = P(any(parts))           // at-least-one
```

**Paths & finance idioms (scans).** The scans `cumsum`/`cumprod`/`cummax`/`cummin` (running
folds: element `t` is the fold of `xs[0..=t]`) turn a shaped draw into a whole **fixed-horizon
path** — no loop needed:

```noise
use rand; use vec; use math
rets ~[252] normal(0.0004, 0.01)          // a year of iid daily returns
walk = cumsum(rets)                        // random walk = cumsum(increments)
path = cumprod(1 + rets)                   // compounding wealth path
// exact GBM, no discretization bias:  path = s0 * exp(cumsum(logrets))
hit      = any(path < 0.9)                 // barrier: did it EVER dip 10%?
asian    = mean(path)                      // Asian option averages the path
lookback = max(path)                       // lookback takes its peak
drawdown = min(path / cummax(path)) - 1    // worst peak-to-trough
final = path[251]
var95 = Q(final, 0.05)                     // VaR — a numeric estimate...
es95  = E(final | final < var95)           // ...that feeds straight into ES/CVaR
plot::fan(path)                             // the cone: q05/25/50/75/95 bands over the index
bands = stats::fan(path)                   // ...and the same bands as a 6×252 matrix
```

`vec::prod` is the product reducer (`prod([]) == 1`). Scans work on any process whose **length
is known up front**; a random-length process is not expressible (see Hazards).

**Lifted `if` = per-lane select** (a value, not control flow; `else` required, both branches
evaluated and reuse the condition's per-lane draws). Gives `max`/`min`/`abs` over RVs for free:

```noise
A ~ unif_int(1, 6); B ~ unif_int(1, 6)
higher = if A > B { A } else { B }     // max of two dice
p_six  = P(higher == 6)
```

**Ranges & `for` (build-time unroll).** `for x in xs { }` runs the body once per element; **bindings
leak** (blocks don't scope) — that's exactly how an accumulator persists. Each `~` inside the body
is a *distinct* node, so it's a clean way to make many independent draws:

```noise
use vec
acc = 0; for x in 1..5 { acc = acc + x }; acc      // 1+2+3+4 = 10
```

**Comprehensions** `[for x in xs { body }]` build an array — it's the `for x in xs { body }` loop
wrapped in `[ ]` so each body value is collected. The body **closes over outer variables** (Noise
has no closures, so this is how you "map with captured state"). Use **`continue`** to skip an
element — that's how you *filter*:

```noise
a = 7; N = 15
fx = [for x in 0..6 { (a ^ x) % N }]                  // body closes over a, N
evens = [for x in 0..10 { if x % 2 != 0 { continue }; x }]  // filter via continue
```

`continue` skips the rest of the loop body (in a `for` loop it drops that iteration's side effects;
in a comprehension it omits the element). The skip condition must be deterministic.

**`%` (floored modulo)** is `a − b·floor(a/b)`, so it takes the sign of `b` and `x % n ∈ [0, n)`
for `n > 0` (clock/modular arithmetic): `-1 % 3 == 2`. `math::floor` / `math::ceil` round (real-only).

**User functions.** `f(a) = expr` is pure (lifts over RVs); `f() ~ dist` draws fresh per call:

```noise
use rand
max(a, b) = if a > b { a } else { b }   // pure
roll() ~ unif_int(1, 6)                 // fresh draw each call
P(roll() + roll() == 7)                  // two INDEPENDENT rolls
```
Functions are **pure in their parameters** — the body sees only its args (plus `pi`/`e`), no outer
variables, no closures. Calls unroll at build time, so recursion must terminate.

**Conditional probability** uses the `|` bar (Bayes, scoped to the query) — prefer it over the
hand-written ratio:

```noise
use rand
D ~ unif_int(1, 6)
p_six_given_high = P(D == 6 | D > 3)   // = 1/3 (≡ P(A && C) / P(C))
high = D | D > 3                       // a conditioned value — bind & reuse
mean_high = E(high); median_high = Q(high, 0.5)   // 5, 5
```

**Signals (lazy waveforms).** `signal::sine(f)` / `cosine(f)` describe a waveform by frequency
(O(1) memory). Arithmetic (scalar AND signal×signal, e.g. `sine(3) + sine(7)`), trig ufuncs, and
`math::exp` defer into the signal; it **materializes** to an array when it meets a sized array
(adopting its length), via `signal::sample(sig, n)`, or when a reducer (`mse`/`mean`/`sum`/…) or
`plot::line` renders it at the **ambient resolution** (`engine::set_resolution(N)`, default 256).
`sine(n, f)` is shorthand for `sample(sine(f), n)`.

**Noise generators are undrawn distributions** — `noise_white(sigma)` (`noise_brown`/`noise_pink`/
`noise_ou`/`noise_white_complex`) obey the same rule as `normal(0, 1)`: **draw with `~` first**.
`static ~ noise_white(s)` is ONE realization (every mention is the same noise; `static - static`
is exactly 0); it pins its length at first materialization — re-rendering at another length is an
error. `w ~[n] noise_white(s)` pins to `n` up front (an ordinary array of RVs). `sample(noise, n)`
and `noise + x` are errors on the undrawn generator. `noise_white_complex(sigma)` draws complex
static with `E|z|² = sigma²` — combine with `math::exp(i*θ)`/`abs`/`arg` for a fully lazy
modulate → demodulate chain.

```noise
use signal
engine::set_resolution(64)        // the one resolution knob — set once, next to the budget
msg = 0.3 * sine(3)               // a waveform (no length anywhere in the math)
static ~ noise_white_complex(0.4) // ONE drawn realization of complex static
err = E(vec::mse(math::abs(1 + msg + static) - 1, msg))   // reducers render at the knob
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
use math; use rand; use vec
z = 2 + 3*math::i                       // complex emerges from math::i
math::abs(z); math::arg(z)              // magnitude & phase (reals)
math::exp(math::i * math::pi)            // ≈ -1  (Euler's identity)
static ~[64] rand::normal_complex(1)    // 64 iid complex-Gaussian static samples
```

## Hazards — what NOT to do

- **`==` on a continuous RV is ~never true.** `unif(1,6) == 4` ≈ 0. Use a discrete distribution
  (`unif_int`, `bernoulli`, `*_int`) whenever you compare for equality or count.
- **Arithmetic on an undrawn recipe is an error.** Draw with `~` first (rule 2 above).
- **No closures.** Function bodies see only parameters and the `pi`/`e` constants.
- **Indices are constant non-negative integers** known at build time; a *random* index works but
  lifts to a per-lane gather that runs **interpreter-only** (slow path — fine for bootstrap-style
  lookups, not for hot inner math).
- **Arrays are fixed-length.** No `push`/append. Build with literals, `a..b`, `~[shape]`, or the
  `vec` constructors.
- **Blocks don't scope** — bindings made inside leak out (there is one flat environment).
- **No random-length / early-stopping processes.** A process that runs *until* something happens
  (expected time to ruin, an unbounded first-passage time, a queue run until empty) is *not*
  expressible — the engine samples independent lanes of a fixed-size graph. **Fixed-horizon**
  paths ARE expressible, as one-liners via the scans: `path = cumprod(1 + rets)` with
  `rets ~[252] normal(mu, sigma)` (see "Paths & finance idioms"). A lifted `if` picks a value per
  lane; it can't decide *when to stop*.
- **Each query samples its own pass.** `P(A)`, `P(B)`, `P(A && B)` are estimated independently, so
  exact cross-query consistency (`P(A && B) ≤ P(A)`) is not guaranteed. (Inside *one* conditional
  query, event and condition share a pass — so `P(A | C)` is internally consistent.)
- **Conditioning is rejection-based.** `P(A | C)` keeps the lanes where `C` happened, so its error
  uses the in-condition count `m ≈ n·P(C)` — fine when `P(C)` isn't tiny. It is **not** posterior
  inference: conditioning on a continuous measurement (`X == 4.7`, probability ~0) or a rare event
  won't work; that's the separate inference track.

## Writing the article

A finished `.noise` example must read, compiled, like a short live article — a really good
notebook. The recipe (converged on `examples/pi.noise` and `examples/secretary.noise`; read those
two as the canonical models):

**Structure**

- **The abstract is the introduction.** Hook, setup, headline claim — in the frontmatter. Never
  open the body with an ```md block that restates it.
- **One cell = one step of the story.** Split cells with blank lines deliberately. Give each code
  cell a SHORT ```md lead-in (1–3 sentences) that advances the story at the idea/math/game level.
  Bare code with nothing between cells is as wrong as walls of prose.
- **Open each md lead-in with a bold step name** — `**The pool.**`, `**Observation phase.**`,
  `**The verdict.**` — so the article skeleton is scannable. Italics for key terms.
- **Structure the code for the narrative.** Unwrap helper functions so each phase of the model is
  its own cell in reading order (pool → cutoff → observe → hire → verdict), rather than one
  `trial()` blob. No orphan cells (`n = 20;` alone between prose) — group constants with the cell
  that uses them, or inline them.
- **No brace/semicolon ugliness.** Never stack closers on one line (`} } }`) — flatten nested
  `if`s with `&&` or a `continue` guard (`if a >= N || p > 1 { continue }`). Never end a block
  with a bare `[...]` line that forces the `;`-before-`[` guard on the statement above — bind
  the array to a name (`factors = [p, q]`) and return the name.
- **End with a verdict.** A closing ```md that weaves the computed result into a sentence, plus —
  when a law/closed form exists — a ```latex display of it.

**Prose discipline**

- md prose must ADD something: the idea, the math, the rules of the game. Never narrate what code
  lines do — a comment or sentence that paraphrases code is an excuse for unclear code
  (clean-code rule); rename variables or restructure instead.
- **Margin comments (`//`) are rare asides**: at most 1–2 per file, each saying something the code
  *can't* (e.g. "1e-4 asks for precision, not a draw count", or a quip beside a plot). If it's
  narrative, it belongs in an ```md block; if it's obvious, delete it.
- **Sentence-like variable names** make the code read as prose: `win = hired_candidate ==
  best_candidate`, `quality`, `bar` — not `c`, `pk`, `tmp`.

**Live numbers**

- **Never hardcode a result in prose or comments — the whole point is that it's computed.** Every
  number a reader sees must come from a `${…}` hole so it tracks sliders and parameter edits.
  Symbolic constants (π/4, 1/e, r = n/e) are fine as *symbols*.
- **Templates hold references, not expressions.** Compute in code (`p_win = P(win);`), then
  interpolate `${p_win * 100}` — trivial arithmetic in a hole is fine, big expressions are not.
- **Queries self-round; deterministic arithmetic doesn't.** `P`/`E`/`Var`/`Q` all return
  estimates that print the digits their sample justifies. A *deterministic* computation
  (`mean(rets)`, a folded constant) prints full float junk — wrap those holes in `round(x, d)`.
- **Don't re-derive the analytic answer numerically in a hole** (no `${sum([for k in r..n
  {1/k}]) * …}`) — that's not the point of Noise. State the law symbolically in latex and let the
  simulation supply the number. The canonical latex shape is **symbolic law ≈ ${simulated}**:
  `\pi \approx 4 \cdot P(X^2+Y^2<1) = ${pi}`, or
  `P(\text{win}) = \frac{r}{n}\sum_{k=r}^{n-1}\frac{1}{k} \approx ${p_win}`. Cheap *locations* of
  a law are fine to compute (`r = \frac{n}{e} \approx ${math::round(n / math::e, 0)}`).
- **Make it interactive where a parameter invites play**: an `input::` slider on the quantity the
  reader will wonder about ("what if I observe more first?"), with the verdict recomputing live.

**A compiled-article skeleton** (abridged from `examples/secretary.noise`):

    ---
    title: "The secretary problem"
    abstract: >
      You interview applicants one at a time … about 37% of the time.
    tags: [classic, optimal stopping]
    ---

    use vec

    ```md
    **The pool.** Twenty applicants walk in, one at a time, each with a hidden
    quality score. Nothing short of hiring the *best of all of them* counts.
    ```
    total_candidates = 20
    quality ~[total_candidates] rand::unif(0, 1)
    best_candidate = max(quality)

    ```md
    **The cutoff.** The whole strategy hinges on one number … Try dragging **r**.
    ```
    r = input::int(min: 1, max: 19, step: 1, default: 7)

    … observation & hiring cells, each with a bold md lead-in …
    win   = hired_candidate == best_candidate
    p_win = P(win)

    ```md
    **The verdict.** Rejecting the first ${r} of ${total_candidates} lands the very
    best applicant **${p_win * 100}%** of the time…
    ```
    ```latex
    P(\text{win}) = \frac{r}{n}\sum_{k=r}^{n-1}\frac{1}{k} \approx ${p_win},
    \qquad \text{maximized at } r = \frac{n}{e} \approx ${math::round(total_candidates / math::e, 0)}
    ```

## Pre-flight checklist

- [ ] Every random variable is introduced with `~` (no arithmetic on a bare recipe).
- [ ] Independence comes from `~[n]` or separate `~` — not from repeating a name.
- [ ] Equality / counting uses a **discrete** distribution (`unif_int`/`bernoulli`/`*_int`).
- [ ] `use` lines present for every non-`builtin` name; queries are capitalized (`P`/`E`/`Var`/`Q`/`Len`).
- [ ] No random-length / early-stopping recurrence (fixed-horizon paths via scans are fine), no
      expectation of block scoping.
- [ ] Run it and sanity-check the result against the analytic value.

For an example/demo (the article form), additionally:

- [ ] Frontmatter with title + abstract; the abstract is the intro and is never restated.
- [ ] Each code cell has a short bold-led ```md lead-in that adds idea/math, not code narration.
- [ ] At most 1–2 margin `//` comments, each saying something the code can't.
- [ ] No number appears in prose that isn't a `${…}` hole (symbols like 1/e are fine); templates
      reference precomputed variables; no closed-form re-derivations in holes.
- [ ] Ends with a verdict ```md (and a ```latex `law ≈ ${simulated}` when a law exists).
- [ ] Rendered check: cells alternate prose/code cleanly, no orphan blocks, sliders where play is
      natural.
