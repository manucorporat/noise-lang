# LANG.md — the Noise language specification

The single source of truth for Noise's syntax and semantics. **The core model below is the
official target**; `crates/noise-core` is being brought into line with it. Where the
implementation currently lags, a **Status:** note says so — when the code and this document
disagree on the *model*, the document wins (the code is the thing being fixed). For everything
else, the golden tests enforce the spec: fix the bug or fix the spec, never let them drift.
Sections marked *(Phase N)* are not implemented yet (see plans/PLAN.md).

## The core model (official semantics)

The conceptual foundation everything else hangs on. Two orthogonal ideas: *what a value is*
(§1) and *what a name means* (§2).

### 1. Everything is a distribution

Every value is a probability distribution. A **number** `N` is the degenerate case — a *point
mass* (Dirac delta) with all its weight at `N`: `mean = N`, `variance = 0`, `P(== N) = 1`. A
**bool** is a point mass on `{true, false}` (and `bernoulli(p)` is its non-degenerate sibling).
`unif`, `unif_int`, and anything derived from them (`X + Y`, `X > 0`, `if c {a} else {b}`) are
distributions too. There is **no "number vs. random variable" dichotomy** — operators map
distributions to distributions, uniformly.

- Point masses (constants) are **constant-folded** as a pure optimization — `2 + 3` computes
  `5` directly, no sampling — but conceptually it is "combine two point masses → a point mass."
  The fold is invisible; the concept stays uniform.
- Therefore `mean`, `variance`, and `P` apply to *anything*: `mean(5) = 5`, `variance(5) = 0`,
  `P(5 > 3) = 1`.
- Internally a distribution is a node in the sample-DAG: a point mass is a constant column
  (every lane = `N`), a real distribution is a column of draws. Same kind of object — the
  value-level split (`number` vs `dist`) is an implementation detail being removed.
- **Strings** (and `unit`) are the exception: a string is a label/utility type for `Print` and
  messages, not part of the distribution family.

> **Status:** the implementation still has distinct `number` / `bool` / `dist` / `estimate`
> value types rather than one distribution type. Behaviour already matches the model for the
> common cases (constants fold; RVs lift), but the type-level unification is not done.

### 2. Binding: `~` is a stochastic node, `=` is a deterministic node

The distinction is **whether the binding introduces new randomness** — the standard
graphical-model split (Stan / BUGS / PyMC), and the one mathematicians already write:

- `name ~ dist` — a **stochastic node**: draw a fresh random variable from a distribution.
  `X ~ unif_int(1, 6)` makes `X` *one* random variable, distributed uniformly on 1..6. The
  right-hand side must be a **distribution**. `~` is the *only* thing that draws.
- `name = expr` — a **deterministic node**: a transformation that adds *no* new randomness.
  `Y = X + 3` (a derived RV), `n = 5` (a constant), `Die = unif_int(1, 6)` (binding a
  distribution *value* — a recipe — without drawing from it).

A name is always **one fixed node** — there is no "re-draw on use." Every mention of a name is
the same node, exactly like `X` in mathematics. So `X + X` is `2X`, `X - X` is `0`, and
`P(X == 6 && X > 3)` uses one `X`. **No invisible "color," no re-draw gotcha.**

Load-bearing rule: **you cannot do arithmetic on an undrawn distribution.** `unif(0,1) + 3` is
an error ("draw it with `~` first") — `unif(0,1)` is a recipe, not a number. Write
`X ~ unif(0,1); Y = X + 3`. Draws happen at `~`; transforms happen at `=`.

### 3. Independence is explicit

Two mentions of a name are the *same* draw; independence comes from *separate `~`
declarations*, never from repeating a name — exactly as on paper (`X₁,…,Xₙ iid ∼ Dist`):

```
X ~ unif_int(1, 6)
Y ~ unif_int(1, 6)
X + Y                 # two independent dice -> a real 2d6 distribution

Die = unif_int(1, 6)  # a distribution (recipe), bound with `=`
a ~ Die               # draw one
b ~ Die               # draw another, independent
```

For N independent draws, give `~` a **shape**: `~[n] dist` draws an `n`-vector, `~[n, m] dist` a
matrix — each leaf a fresh `~`, so the whole batch is iid. (`~` is still the only thing that draws;
the shape just says how many.)

```
Bday = unif_int(1, 365)
days ~[23] Bday            # 23 independent draws, one fresh node each
P(has_duplicates(days))
```

A bare `~` is a scalar (rank 0); `~[1] dist` is a length-1 *array* — same draw, different shape,
exactly like NumPy `size=()` vs `size=(1,)`. The prefix `~` is an ordinary expression, so it works
inline too: `sum(~[3] unif(0, 1))`.

### 4. Functions

- `f(args) = expr` — a **deterministic** function: `max(a, b) = if a > b { a } else { b }`.
- `f(args) ~ dist` — a **stochastic** function: each *call* is a fresh draw. `roll() ~
  unif_int(1, 6)` makes `roll() + roll()` two independent rolls (the call parentheses make the
  fresh draw visible). This is the README's `X(y) ~ { … }`.

A call evaluates its arguments, binds them to the parameters in a fresh frame, then evaluates
the body. Functions are **pure in their parameters**: the body sees only its arguments and may
call other functions, but it does *not* capture outer variables (no closures yet) — `n = 10;
f(x) = x + n; f(1)` is an error. User definitions shadow a builtin of the same name. Calls
unroll at build time, so they must terminate (a deep-recursion guard turns runaway recursion
into an error rather than a crash).

### 5. Sampling

Nothing is sampled until a query forces it. `P(event[, n])` realizes the event's distribution
over `n` draws and returns the probability — itself a point mass we know only approximately,
carried as an *estimate* with a standard error (see "Probability"). Everything upstream stays
symbolic (a distribution) until then.

> **Status (§2–§4): implemented.** `~` and `=` genuinely differ: `unif`/`unif_int`/`bernoulli`
> return an undrawn **recipe**, `~` is the *only* thing that draws (it instantiates a fresh
> sample-DAG node), and `=` binds the recipe as-is — so `Die = unif_int(1,6); a ~ Die; b ~ Die`
> gives two *independent* dice. Arithmetic on an undrawn distribution (`unif(0,1) + 3`, or `Die +
> 1`) is an error that points you at `~`. A bound *draw* is one fixed node (`X - X == 0`);
> independence comes from distinct `~`. User-defined functions (§4) work: `f(a)=…` is pure,
> `f()~dist` draws on each call. Still to build: the full value-type unification (see §1 Status).

## Lexical structure

- **Whitespace** (spaces, tabs) separates tokens and is otherwise insignificant. A **newline**
  also separates tokens but, *between two complete statements*, doubles as a statement separator
  (see grammar) — so `;` is optional and only needed to put several statements on one line. A
  newline *inside* an unfinished expression is insignificant, so an operator can lead a
  continuation line (`total = a\n  + b`).
- **Comments** run to end of line, started by `#` or `//`.
- **Numbers**: `[0-9]+` or `[0-9]*.[0-9]+` / `[0-9]+.[0-9]*`, parsed as `f64`. No exponent
  syntax yet. A leading `-` is *not* part of the literal — it is the unary minus operator.
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`. Case-sensitive. `if`, `else`, `for`, `in`, `true`,
  and `false` are reserved.
- **Booleans**: `true` and `false` are literals — point masses on `{true, false}` (i.e.
  `bernoulli(1)` and `bernoulli(0)`; see core model §1).
- **Strings**: double-quoted, no escape sequences yet (`"like this"`). A newline or EOF
  before the closing quote is an error.
- **Templates**: backtick-fenced multi-line text with `${expr}` interpolation (see "Templates"
  below). A single backtick `` `…` `` delimits a plain body; a triple fence ` ```tag … ``` ` carries
  a syntax tag (e.g. ` ```md `). Unterminated fences are an error.

### Tokens

```
+  -  *  /  ^            arithmetic  (`*` is elementwise / broadcast)
@                         matrix product  (dot / mat·vec / matmul by shape)
== != < > <= >=           comparison
&& ||                     logical and / or
|                         conditioning bar (`P(A | C)`); only inside a query
!                         logical not (prefix)
=                         assignment bind
~  ~[n]  ~[n, m]          sample / distribution bind (optional draw shape)
( ) { }                   grouping, blocks
[ ]                       array literal / indexing
..                        half-open integer range (0..n)
`…`  ```tag…```           template (interpolated text; `${expr}` holes)
::                        module path separator (rand::unif)
,  ;                      argument sep, statement separator (or use a newline)
if else for in use        keywords
true false                boolean literals
```

## Frontmatter (literate files)

A `.noise` file may open with a `---`-fenced **frontmatter** block that turns it into a
self-describing document: a `title`, a paper-style `abstract`, keyword `tags`, and an `extra:` map
for host-specific metadata. Frontmatter is **purely descriptive** — tunable parameters are not
declared here; they are inline `input::…` calls in the program body (see **Inputs** below). The
fence is recognized **only at byte 0** — line 1 must be exactly `---`, and the block runs to the next
line that is exactly `---`. Anywhere else in a file, `---` keeps meaning three unary minuses, so no
ordinary program is affected. The engine treats the whole block as trivia (it never becomes a
token), so error spans still point at the original source.

```
---
title: "Roll a die"
abstract: >
  A die is discrete, so it uses unif_int — integers 1..6.
tags: [basics, discrete]
extra:
  category: "Basics"          # host-specific metadata, engine passes it through
---
dice_sides = input::int(min: 1, max: 100, step: 1, default: 6);
Dice ~ rand::unif_int(1, dice_sides);
Print("P(rolling a 4) =", P(Dice == 4))
```

- **Syntax**: the block is **YAML** (a `{ … }` JSON block also works, since JSON is valid YAML).
- **Recognized keys**: only `title`, `abstract`, `tags`, and `extra` are accepted at the top level —
  any other key is a validation error. Host-specific metadata (a `blurb`, a `category`, a `seed`, …)
  goes under `extra:`, a free-form map the engine passes through untouched.
- **Validation**: `noise validate file.noise` reports frontmatter errors alongside the usual checks.
  A retired `knobs:` block is a clear error pointing at `input::` (see below).

## Inputs

An **input** is a host-tunable parameter declared **inline in the program body**, where it is used,
with an `input::<type>(…)` call. It replaces the old frontmatter `knobs:` map: instead of an
out-of-band YAML schema, a tunable value is an ordinary namespaced call, and a literate host renders
a control (a slider, a checkbox) at the point of declaration rather than a bar of every input at the
top of the page.

```
dice_sides    = input::int(min: 1, max: 100, step: 1, default: 6);
target_number = input::int(min: 1, max: 100, step: 1, default: 4);

Dice ~ rand::unif_int(1, dice_sides);
p = P(Dice == target_number);
```

- **Value**: `input::…(…)` evaluates to the input's **current value** — a deterministic point mass
  (its default, or a host override) — so downstream code reads it like any number. A program may
  shadow the name with a normal rebind.
- **Types**: `input::real` (a continuous slider), `input::int` (snaps to whole numbers), and
  `input::bool` (a checkbox). Numeric inputs take optional `min`, `max`, and `step`; every input
  needs a `default`. An optional `label` names it in a UI.
- **Named arguments**: an input's spec is passed with **named arguments** (`min: 1, max: 10`) — a
  general call feature (see below), not `input::`-specific.
- **Name**: an input needs a name. When `input::…` is the **direct right-hand side of a bind**
  (`dice_sides = input::int(…)`), the name is inferred from the left-hand identifier; otherwise pass
  it explicitly with `name: "dice_sides"`.
- **Overrides**: a host may retune an input (the CLI's `noise file.noise --input dice_sides=20`, the
  playground's inline sliders). The engine type-checks the override, clamps it to `[min, max]`, and
  snaps it to `step` — one implementation, so every host behaves identically. An override naming an
  input the program never declares is an error. Headless (no override), an input resolves to its
  default, so programs run unchanged with no UI.
- **Discovery**: inputs are found **by running the program** — the engine returns a manifest of every
  input it evaluated. (An input inside a branch that didn't run won't appear until that branch runs,
  the same as a `plot::` inside a branch.)

## Named arguments

A call's arguments are **either all positional or all named — never mixed**: `f(x, y)` or
`f(a: x, b: y)`, but not `f(x, b: y)`. Named arguments (`name: value` pairs) bind to parameters by
name, in any order, and work for **any** user-defined function as well as `input::`:

```
sub(a, b) = a - b;
sub(b: 2, a: 10)          # 8 — named, any order
sub(10, 2)                # 8 — positional, in parameter order
```

Every parameter must be filled exactly once; an unknown name, a missing parameter, or a duplicate
name is an error.

## Templates

A **template** is backtick-fenced multi-line text with `${expr}` interpolation — the
`Print`-without-`Print`. Two fence weights, identical body semantics:

```
`
P(rolling a ${target_number}) = ${p}
`
```
```md
## Result
The probability is **${p}**.
```

- **Single backtick** `` `…` `` — a plain body; it cannot contain a backtick.
- **Triple fence** ` ```tag … ``` ` — carries a syntax tag (e.g. `md` for markdown, or `latex` which a
  host typesets with a math renderer like KaTeX — one display equation per non-blank line) so a host can
  render the note richly vs as preformatted text; the body may contain single backticks. A bare ` ``` `
  (no tag) is just the plain template. The engine is tag-agnostic — it carries the tag through on the
  note and leaves rendering to the host.
- **Body**: raw text with `${expr}` holes. The shared leading indentation is stripped and the blank
  opening/closing lines next to the fences are removed, so a template indented inside code still
  renders flush-left. A hole renders via its value's **display form** (an `Est` self-rounds to its
  standard error, exactly like `Print`); holes are **deterministic-only** (an undrawn recipe is an
  error — draw it with `~` first). A hole ends at its *matching* `}` (so `${ if c { a } else { b }}`
  works); errors inside a hole point at the real source location. No `${{`-style escape in v1.
- **Emission**: at **root statement position** a template *emits* (like `Print`) and yields unit.
  Anywhere else — inside a function, as an argument, in an expression — it is just a `string` value
  (usable with `+`, as a `Print` argument, etc.). To emit from inside a function, pass a template to
  `Print`.

## Grammar (informal)

```
program   := stmt*                         # statements separated by ';' or a newline
stmt      := expr ( ';' | NEWLINE )?       # separator optional before a terminator; ';' is
                                           #   only required for two statements on one line
expr      := bind | opexpr
bind      := IDENT ('=' | '~') expr        # right-associative
opexpr    := precedence-climbing over the operator table below
postfix   := primary ('[' expr ']')*       # indexing, repeatable (M[i][j]); binds like a call
primary   := NUMBER | STRING | path | 'true' | 'false'
           | path '(' args? ')'            # call
           | '(' expr ')'
           | '[' (expr (',' expr)*)? ']'   # array literal
           | block
           | if
           | for
           | 'use' IDENT                   # bring a module into unqualified scope
path      := IDENT ('::' IDENT)*           # bare name or module path (rand::unif)
block     := '{' stmt* '}'
if        := 'if' opexpr block ('else' (if | block))?
for       := 'for' IDENT 'in' opexpr block
args      := expr (',' expr)*
```

Everything is an expression: `bind`, `block`, and `if` all produce values.

## Operator precedence

Lowest to highest. All binary operators are left-associative **except** `^` and
binding, which are right-associative. Prefix `-`/`!` bind tighter than everything below
`^` (so `-2 ^ 2 == -(2 ^ 2) == -4`, matching common math convention).

| Level | Operators            | Assoc  |
|-------|----------------------|--------|
| 1     | `=` `~` (binding)    | right  |
| 2     | `\|` (conditioning)  | left   |
| 3     | `..` (range)         | left   |
| 4     | `\|\|`               | left   |
| 5     | `&&`                 | left   |
| 6     | `== != < > <= >=`    | left   |
| 7     | `+ -`                | left   |
| 8     | `* / % @`            | left   |
| 9     | `^`                 | right  |
| 10    | prefix `- ! ~`       | —      |
| 11    | postfix `[index]`    | —      |
| 12    | call, grouping       | —      |

## Values and types

Conceptually every value is a distribution (see **The core model §1**); the types below are the
*current implementation's* representation of that idea, which still splits the distribution
family into separate runtime types.

Runtime values: `number` (`f64`, a point mass), `bool` (a point mass on `{T,F}` — i.e. a
Bernoulli; see core model §1), `string`, `unit` (`()`), `array` (a fixed-length, build-time
sequence of values — see "Collections"), `signal` (a **lazy waveform expression** — see
"Signals"), and **`complex`** (a complex scalar — see "Complex numbers"). An **`estimate`** is a
`number` carrying a standard error (produced by `P`); it
behaves like a number in arithmetic (propagating the error) and displays rounded to its
justified precision. **`dist`** — a non-degenerate random variable / distribution handle — is
produced by `unif`/`unif_int`/`bernoulli`, bound by a binding, and propagated by operator
lifting (see "Random variables"). A `dist` Displays as `<dist #n>` and never samples on Display;
sampling is forced by `P` (or the Rust API `Engine::sample` / `Engine::moments`). Per the core
model, `number`/`bool`/`dist`/`estimate` are all the *same kind of thing* (a distribution); the
split is the implementation detail being removed.

Type rules (current):
- Arithmetic `+ - * / ^` require both operands to be `number` → `number`.
  Division and `^` follow IEEE-754 (`1/0 == inf`, no panic).
- **`%` (floored modulo)** requires two `number`s → `number`, defined as `a − b·floor(a/b)`, so
  the result takes the sign of `b` and `x % n ∈ [0, n)` for `n > 0` (clock/modular arithmetic).
  `x % 0` is `NaN` (no panic). Real-only — `%` on a `complex` is an error.
- If **either** operand of `+ - * / ^` is `complex` (or both), the op is the corresponding
  complex operation (the real operand promotes to `re + 0i`); see "Complex numbers".
- **`+` also concatenates** when *either* operand is a `string`: the other operand is
  stringified via its display form, giving a `string` (e.g. `"x = " + 5` → `"x = 5"`). This
  is deterministic-only — a string can never enter a random-variable expression.
- Ordering `< > <= >=` require two `number`s → `bool`.
- Equality `== !=` compares two values of the *same* primitive type (`number`, `bool`,
  or `string`) → `bool`. Mixed-type comparison is an error.
- Prefix `-` requires `number`; prefix `!` requires `bool`.
- `if` requires a `bool` **or** a bool random variable (`dist<bool>`) condition. A
  deterministic `bool` takes one branch (with no `else` and a false condition it yields
  `unit`); a random-variable condition lifts to a per-lane select — see "Random variables".

There are **no implicit coercions**. Type mismatches are runtime errors with a source span.

## Binding semantics: `=` vs `~`

The authoritative rules are **The core model §2–§4**: `~` is a **stochastic node** (draws a
fresh random variable from a distribution; `~` is the only thing that draws), `=` is a
**deterministic node** (a transform or a plain value/recipe — no new randomness). A name is one
fixed node, reused identically; independence comes from *separate* `~` declarations, never from
repeating a name. You **cannot do arithmetic on an undrawn distribution** — `~` it first. A
`bind` expression evaluates to the bound value, so `y = (x = 3)` sets both to `3`.

> **Status:** the recipe/draw split is **implemented**. `unif`/`unif_int`/`bernoulli` return an
> undrawn *recipe*; `~` is the only thing that draws (instantiating a fresh node); `=` binds the
> recipe as-is; arithmetic on an undrawn recipe is a spanned error. The model's behaviour holds:
> - a drawn name is one fixed node: `X - X` is exactly `0`, `X + X` is `2·X`;
> - independence comes from distinct `~` draws:
>   ```
>   A ~ unif_int(1, 6)
>   B ~ unif_int(1, 6)
>   A + B            # two independent dice — a real 2d6 distribution
>
>   Die = unif_int(1, 6)   # a recipe, bound with `=`; not yet drawn
>   a ~ Die; b ~ Die       # two independent draws of the same recipe
>   ```
> Still to build: user-defined functions (§4) and the full value-type unification (§1).

## Random variables (Phase 2)

- **`unif(a, b)`** is a distribution constructor. `a` and `b` are numbers **or numeric random
  variables** (a *random* parameter makes a **hierarchical** distribution — see "Hierarchical
  distributions" below; a `bool` RV or an undrawn recipe is a spanned error). It returns an undrawn
  **recipe** for the uniform on `[a, b)`; `X ~ unif(a, b)` draws a random variable from it (see
  "Binding semantics" above).
- **Operator lifting:** any arithmetic/comparison/unary operator with at least one `dist`
  operand yields a `dist`; deterministic operands fold in as constants. Purely deterministic
  subexpressions still evaluate eagerly to plain values (and still surface spanned type
  errors). Type rules mirror the deterministic evaluator, enforced on the RV's kind:
  arithmetic needs two numeric RVs; ordering needs two numeric RVs and yields a bool-RV;
  `== !=` need matching kinds and yield a bool-RV; prefix `-` needs a numeric RV, prefix `!`
  needs a bool-RV. Violations (e.g. `X + "a"`, `!X` on a numeric RV) are spanned runtime
  errors.
- **Bool-RVs as 0/1:** comparisons over RVs produce indicator columns (`1.0`/`0.0`), so the
  empirical mean of `X < 0.5` is the probability — pre-wiring Phase 3's `P()`.
- **Execution model:** an RV expression lowers to a sample-DAG, compiles to flat bytecode
  over column registers (with CSE so shared sub-RVs compile once), and is sampled
  column-at-a-time over `BATCH`-sized batches under a seedable xoshiro256++ PRNG. Sampling
  is lazy — nothing is drawn until the Rust API forces it.

### Live Phase-3 builtins and operators

- **`unif_int(a, b)`** — discrete uniform over integers `a..=b` (inclusive). Use this, not
  `unif`, for dice/coins/counts; `==` on it is meaningful.
- **`bernoulli(p)`** — a bool-RV that is `true` with probability `p` (≡ `unif(0,1) < p`).
- **`normal(mu, sigma)`** — the Gaussian `N(mu, sigma^2)` (continuous; `sigma >= 0`, `sigma = 0`
  is a point mass at `mu`). Like the others it returns an undrawn **recipe**; `Z ~ normal(0, 1)`
  draws it. Sampled via Box–Muller. (Being continuous, `==` on it is almost surely false — see
  hazards.)
- **`exponential(rate)`** — the exponential `Exp(rate)` (continuous; `rate > 0`, `mean = 1/rate`).
  Inverse-CDF sampled. (Renamed from `exp`, which is now the exponential **function** `math::exp`.)
- **`normal_complex(sigma)`** — a circularly-symmetric complex Gaussian (CSCG): `re`/`im` each
  `~ N(0, sigma/√2)`, independent, so `E|z|² = sigma²`. Draws a `complex` RV (`z ~ …` or `~[n] …`).
  The textbook model for radio static / thermal noise / Rayleigh fading.
- **`categorical(weights)`** — sample an index `0..len(weights)` with probability proportional to
  the (constant, non-negative) `weights`. The honest "measure a discrete distribution" primitive
  (`y ~ rand::categorical(probs)`). Built by inverse-CDF from one `unif(0,total)` draw.
- **`empirical(xs)`** — the **iid bootstrap**: a recipe whose every draw is a uniformly random
  element of the constant numeric array `xs` (resample history instead of assuming a
  distribution). A true recipe: draw with `~`, `~[n]` gives `n` iid resamples, one name is one
  draw. Internally a `unif_int` index plus a per-lane **gather**, so — like all gathers — it runs
  on the interpreter only (no JIT).
- **`block_bootstrap(xs, b)`** — the **moving-block bootstrap** for autocorrelated series: one
  draw yields an *array* of `Len(xs)` values assembled from random contiguous blocks of `xs` of
  length `b` (non-wrapping start indices in `[0, n − b]`; the last block truncates to fit), so
  within-block autocorrelation survives where iid resampling would shuffle it away. Same recipe
  semantics and interpreter-only gather note as `empirical`. Both require a flat, non-empty,
  constant numeric array; `b` must be an integer with `1 <= b <= Len(xs)`.
- **`poisson(lambda)`** — Poisson counts (discrete, `lambda > 0`, `mean = variance = lambda`),
  support `0, 1, 2, …`. Sampled via Knuth's algorithm.
- **`geometric(p)`** — the number of failures before the first success (discrete, `0 < p <= 1`,
  support `0, 1, 2, …`, `mean = (1-p)/p`). `==`/counts are meaningful.
- **The `_int` family** — `normal_int(mu, sigma)` and `exponential_int(rate)` draw the matching
  continuous distribution and round each draw to the nearest integer, so `==`/counts are meaningful
  (the discrete `unif_int` is already its own constructor).
- **`E(x)`** / **`E(x, n)`** and **`Var(x)`** / **`Var(x, n)`** — the Monte Carlo **expectation**
  and **variance** of a *numeric* quantity (a number or a numeric/bool RV), the companions to
  `P` for non-events. Both return an `estimate`: `E` carries the standard error of the mean
  (`sqrt(Var/n)`), `Var` an asymptotic `var·sqrt(2/n)`; a deterministic value is exact
  (`E(5) = 5 ± 0`). `E` of a bool-RV equals `P`. Default `n = 1e6` (set per-run with
  `engine::set_max_samples`), fixed seed.
- **`Q(x, q)`** / **`Q(x, q, n)`** — the **quantile** (inverse CDF) of a *numeric* quantity at
  level `q ∈ [0, 1]`: `Q(X, 0.5)` is the median, `Q(X, 0.95)` the 95th percentile, and
  `Q(X, 0)`/`Q(X, 1)` the min/max draw. The companion to `E`/`Var` for tail/spread questions.
  Estimated by Monte Carlo — draw `n` samples (default `1e6`, or `engine::set_max_samples`; fixed
  seed), sort, and linearly
  interpolate between the bracketing order statistics. Returns a plain `number` (unlike `P`/`E`,
  it does *not* auto-round to a confidence precision: a sample quantile's error depends on the
  density there). A deterministic value is its own quantile at every level.
- **`sqrt(x)`** (`x >= 0`) and the constants **`pi`**, **`e`** — math helpers for scaling
  factors. `pi`/`e` are bare identifiers resolved as constants, so they also work *inside*
  function bodies (which otherwise see only their parameters).
- **`math::exp(x)` / `math::log(x)` / `math::log10(x)` lift over random variables** — like
  `sin`/`cos`, a scalar computes directly and an RV lifts to a per-lane graph node, so
  `Z ~ normal(mu, sigma); exp(Z)` (a lognormal) and Kelly's `E(log(1 + f*R))` just work. One deliberate
  asymmetry: a *deterministic* `log(0)`/`log(-x)` keeps the friendly build-time "needs x > 0"
  error, while a per-lane RV value follows IEEE (`x < 0` → `NaN`, `x == 0` → `-inf`) — a rare bad
  lane shouldn't kill a million-sample run. Complex `exp` (Euler `e^{iθ}`) is unchanged.
- **`math::gcd(a, b)`** and **`math::modpow(base, exp, mod)`** — deterministic integer number
  theory (the modular-arithmetic core). `gcd` is Euclid's algorithm on `|a|`, `|b|` (`gcd(0,0)=0`);
  `modpow` is `base^exp mod` by square-and-multiply, **exact** even when `base**exp` would overflow
  `f64`'s `2^53` (the explicit, predictable form of the `(base ^ exp) % mod` idiom). Both require
  whole-number arguments with `|x| <= 2^53` (`modpow` needs `exp >= 0`, `mod > 0`) and are
  deterministic-only — a random-variable argument is an error (no per-lane integer loop in the VM).
- **`P(event)`** / **`P(event, n)`** — the probability that a bool-RV (or deterministic bool)
  is true, returned as a plain `number` in `[0,1]` so it composes in arithmetic (`4 * P(C)`).
  `P` of a numeric (non-event) value is an error. Estimated by Monte Carlo over `n` samples
  (default `1e6`, or set per-run with `engine::set_max_samples`) under a fixed seed, so a run is
  reproducible.
  - **`P` returns an *estimate*** — a number that carries its standard error
    `se = sqrt(p(1-p)/n)` (a *deterministic* event is exact, `se = 0`). The value keeps full
    precision; it **displays rounded to the digits the error justifies** (`floor(-log10(se))`
    decimals). More samples → smaller error → more digits (`P(D==4, 1000)` → `0.2`;
    `P(D==4, 1e8)` → `0.1666`), with no manual rounding.
  - **The error propagates through arithmetic** (first order): `+ - * /` on estimates combine
    their errors, so a *derived* result self-rounds too. `4 * P(C)` has 4× the error of `P(C)`,
    so it correctly shows one fewer digit (`3.141`, not a spurious `3.1412`); a ratio
    `P(A&&B)/P(B)` likewise prints to its own justified precision. (The `round(x, d)` builtin is
    still available if you want a fixed number of places.)
- **`Print(args...)`** — prints its arguments space-separated (each via its display form),
  then a newline. Returns `unit` (the CLI does not echo a trailing `unit`, so a program can
  end in `Print(...)`). Use it with string concatenation to emit messages:
  `Print("P(win) =", round(p, 4))`.
- **`round(x, digits)`** — round `x` to `digits` decimal places (handy for tidy messages).
- **`&&` / `||`** — logical and/or; lift over bool-RVs as elementwise ops on 0/1 columns.
- **`if` over a random variable** — when the condition is a `dist<bool>`, `if c { a } else { b }`
  lifts to a per-lane **select**: each sample takes `a`'s draw where `c` is true and `b`'s where
  false. It is a *value*, not control flow, so:
  - an **`else` branch is required** (every sample needs a value), and the two branches must
    have the same kind;
  - **both branches are evaluated** (to build the select) — avoid side-effecting bindings inside
    an RV-`if`;
  - branches reuse the *same per-lane draws* as the condition (sharing flows through), so
    `if D == 6 { D } else { 0 }` yields `6` exactly on the lanes where `D` rolled a 6.
  - This gives `max`/`min`/`abs` over RVs for free, e.g. `if A > B { A } else { B }`.
  It is **not** sequential branching — the engine still samples independent lanes. A
  **fixed-horizon** path is nonetheless idiomatic via the scans (`vec::cumsum`/`cumprod`/… — see
  "Collections"); what remains out of scope is a *random-length* / early-stopping process
  (that's the dynamics fork in `plans/PLAN.md`).

### Conditioning: `event | given`

The **`|` bar** conditions a query on an event — Bayes' rule written the way mathematicians write
it. `P(A | C)` is "the probability of `A` **given** `C` holds", restricted to the worlds where `C`
is true. It is **scoped to the one query** — no global state, no side effect, no `observe`-style
mutation: a `|` changes *that* `P`/`E`/`Var`/`Q` and nothing else.

```
D ~ unif_int(1, 6)
P(D == 6 | D > 3)        # ≈ 0.3333 — given the roll beat 3, how often a 6? (= (1/6)/(1/2))
E(D | D > 3)             # 5 — the mean roll among {4, 5, 6}
Q(D | D > 3, 0.5)        # 5 — the conditional median
```

- The bar binds **looser than every other operator** (below `||`), so the whole event is on the
  left and the whole condition on the right: `P(A && B | C || D)` is `P((A && B) | (C || D))`.
- **`given` must be an event** (a `bool`/`dist<bool>`); for `P`, the left side must be an event too
  (`E`/`Var`/`Q` take any numeric quantity). A conditional whose condition **never occurs** in the
  sample (e.g. `P(D == 6 | D > 100)`) is a `0/0` — a spanned error, not a silent `NaN`.

**A conditioned value is first-class.** `event | given` evaluates to a *conditioned value* you can
**bind** and **query later** — and **operate on**, where an operation pushes into the quantity and
carries the condition along (`2*(X|C)+1` is `(2X+1) | C`; a comparison `(X|C) < 3` is the
conditioned event `(X<3) | C`). Like a `Recipe`, it is *consumed* by `P`/`E`/`Var`/`Q`, not used in
plain arithmetic past that:

```
high = D | D > 3         # a conditioned value (bound, not yet queried)
E(high)                  # 5
P(high < 5)              # ≈ 0.3333 — P(D == 4 | D > 3)
```

The **one rule** that keeps conditioning consistent: you **cannot combine two values conditioned on
different events** (`(X | C) + (Y | D)` is a spanned error) — condition once, at the end: `(X + Y) |
C`. Under the hood a conditioned value keeps its `quantity` and `condition` as separate sample-DAG
nodes; a query fuses them into one root, `select(condition, quantity, NaN)` (the quantity where the
condition holds, a `NaN` sentinel elsewhere), and samples it in a **single joint pass** that skips
the `NaN` lanes. Drawing event and condition together (shared upstream draws) is what makes
`P(A && C) ≤ P(A)` hold within a conditional and what makes the conditional **quantile** correct —
two separate passes would mis-pair the lanes.

> **Status: implemented** for `P`/`E`/`Var`/`Q`, including bound conditioned values and operations
> on them. The estimate uses the *in-condition* sample size `m ≈ n·P(C)` for its standard error, so
> a rarer condition self-reports a looser estimate. Conditioning is **rejection-based** (it keeps the
> lanes where `C` happened): excellent when `P(C)` is not tiny, but it does **not** do importance
> weighting or posterior/MCMC inference — observing a *continuous* measurement (`X == 4.7`) or a
> rare/high-dimensional event is the separate inference track (see `GOAL.md`).

### Hierarchical distributions (a random parameter)

A distribution's **parameter may itself be a random variable** — the building block of hierarchical
/ Bayesian models. `p ~ unif(0, 1); k ~ bernoulli(p)` draws a coin whose *bias is itself random*:
per Monte Carlo lane, `p` takes that lane's draw and `k` is a Bernoulli at that `p`.

```
p ~ unif(0, 1)
k ~ bernoulli(p)
P(k)                     # ≈ 0.5 — the parameter is integrated out: P(k) = E[p]

mu ~ normal(0, 1)
X  ~ normal(mu, 1)
Var(X)                   # ≈ 2 — Var[mu] + E[σ²] = 1 + 1 (variance adds up the hierarchy)
```

- Supported for **`unif`, `unif_int`, `normal`, `normal_int`, `exponential`, `exponential_int`,
  and `bernoulli`** (the location–scale / threshold families). `poisson`, `geometric`, and
  `normal_complex` with a random parameter are not supported yet (a spanned error names the fix).
- A parameter must be a number or a **numeric** RV; a `bool` RV or an **undrawn recipe**
  (`bernoulli(unif(0,1))`) is an error — draw the parameter first (`m ~ unif(0,1); …(m)`).
- A random parameter is **not range-checked** at construction (it's random): the draw transform
  handles it per lane (e.g. `bernoulli(p)` is `U < p`, so a lane with `p > 1` is simply always true).
- **Independence:** every `~` draw of a parameterized recipe reuses the *same* parameter node but a
  *fresh* base draw — so `a ~ bernoulli(p); b ~ bernoulli(p)` are independent **given `p`** (and
  positively correlated marginally, `Cov = Var[p]`), exactly the conditional-independence a
  hierarchical model means.

Under the hood, a random-parameter draw lowers to a **standard base draw plus a deterministic
transform** (`unif(a,b)` → `a + (b−a)·U`; `normal(μ,σ)` → `μ + σ·Z`; `bernoulli(p)` → `U < p`;
`exponential(r)` → `E/r`), so the sample-DAG, VM, and RNG are unchanged — it's ordinary graph nodes
that simplify/CSE/JIT like any other. An all-constant call still uses the single efficient source
instruction (no transform), so nothing slows down.

**Hierarchical + conditioning = (rejection) Bayesian inference.** A random parameter gives you a
*prior*; the `|` bar conditions on *data*; together they read off a *posterior*:

```
bias  ~ unif(0, 1)                       # prior over a coin's bias
flips ~[10] bernoulli(bias)              # 10 flips at that bias
E(bias | count(flips) == 7)              # ≈ 0.667 — posterior mean after 7 heads (= 8/12)
```

> **Status: implemented** (rejection-based). This makes priors and posteriors *expressible and
> queryable*, but inference is still by **rejection** (keep the lanes matching the data) — great for
> a handful of discrete observations, but it does not scale to many continuous observations or rare
> data. Importance weighting / MCMC (so the posterior survives lots of data) is the next inference
> step; see `GOAL.md`.

### Hazards and still-planned semantics

- **Equality on a *continuous* RV is almost surely false.** `unif(a,b)` is continuous, so
  `X == c` (and any `==` between continuous RVs) has probability ~0 — e.g. `unif(1,6) == 4`
  is essentially never true. Use a discrete distribution (`unif_int`, `bernoulli`). A
  **warning** on `==`/`!=` over a continuous RV is still TODO.
- **Multiple queries do not yet share one sampling pass (TODO).** Each `P()` call currently
  samples its own cone with the default seed. The intended semantics is that all queries in
  a run share *one* batch of draws so `P(A)`, `P(B)`, `P(A && B)` are mutually consistent
  (`P(A && B) ≤ P(A)`). Not yet implemented. (*Within* a single conditional query the event and
  condition **do** share one pass — see "Conditioning" — so `P(A | C) ∈ [0, 1]` and the conditional
  quantile are internally consistent; the open item is consistency *across separate* queries.)
- **`P()` self-rounds to confidence precision** (see above) — a lightweight, honest form of
  error reporting: the number of shown digits reflects the standard error. An explicit
  `estimate ± stderr` value and an auto-`N` convergence stop are still planned.

## Modules

Builtins are organized into **modules** and accessed Rust-style. A name is reached either by a
qualified **path** (`rand::unif(0, 1)`, `math::pi`) — which always works — or unqualified after a
`use` brings its module into scope:

```
use rand;            # now `unif`, `unif_int`, `normal`, … are available unqualified
X ~ unif(0, 1);
math::sqrt(2)        # or reach a single item by its full path, no `use` needed
```

The modules:

| Module    | Items | Default? |
|-----------|-------|----------|
| `builtin` | `P`, `Q`, `E`, `Var`, `Print`, `Len` (capital-only) | **always active** (no `use`) |
| `rand`    | `unif`, `unif_int`, `bernoulli`, `normal`, `normal_int`, `normal_complex`, `exponential`, `exponential_int`, `poisson`, `geometric`, `categorical`, `empirical`, `block_bootstrap`, `rotation`, `permutation` (batched sampling is the `~[shape]` operator, not a builtin) | needs `use rand;` |
| `math`    | `pi`, `e`, `i`/`j` (imaginary unit), `sqrt`, `exp`, `abs`, `arg`, `conj`, `re`, `im`, `floor`, `ceil`, `round`, `log` (natural), `log10`, `sin`, `cos`, `atan`, `sign`, `gcd`, `modpow` | needs `use math;` |
| `vec`     | `sum`, `prod`, `count`, `any`, `all`, `max`, `min`, `mean`, `cumsum`, `cumprod`, `cummax`, `cummin`, `dot`, `vdot`, `normsq`, `norm`, `transpose`, `adjoint`, `normalize`, `outer`, `quantize`, `has_duplicates`, `count_duplicates`, `mse`, `ones`, `zeros`, `iota` (vector `+`/`-` and `@` cover add/sub/matvec) | needs `use vec;` |
| `signal`  | `sine`, `cosine` (lazy waveforms), `noise_white`, `noise_white_complex`, `noise_brown`, `noise_pink`, `noise_ou` (undrawn noise generators — drawn with `~`), `sample` | needs `use signal;` |
| `plot`    | `histogram`, `line`, `scatter`, `heatmap`, `corr`, `fan`, `explain`, `value` (charts, pushed to the output stream like `Print`) | path-only (`plot::fan(...)`) |
| `stats`   | `histogram`, `quantiles`, `moments`, `fan`, `corr` — the same computations as `plot::`, returned as numbers | path-only (`stats::fan(...)`) |
| `engine`  | `set_max_samples`, `set_max_opts`, `set_resolution` (run-time evaluator knobs) | needs `use engine;` |

**The `stats` module** is the raw-data twin of `plot`: every chart's numbers, without the chart.
`stats::histogram(x)` returns the bins `plot::hist(x)` draws; `stats::fan(path)` returns the bands
`plot::fan(path)` shades. Not an approximation of them — literally the same computation, at the same
budget and seed, so a picture is always auditable. Like `plot::`, it is always written as a path.

| Call | Returns |
|---|---|
| `stats::histogram(x)`, `stats::histogram(x, bins)` | a 2×bins matrix `[[midpoints], [counts]]` (default 30 bins; an event always gets the two buckets `[0, 1]`) |
| `stats::quantiles(x, [q…])` | one value per level, in the order asked |
| `stats::moments(x)` | `[n, mean, sd, min, max]` — `describe(x)`'s header line, as data |
| `stats::fan(path)` | a 6×cols matrix: rows `q05, q25, q50, q75, q95, mean` |
| `stats::corr(a, b)` | their correlation, one number |
| `stats::corr(v)` | the element×element `n×n` matrix (diagonal 1) |

They **force sampling** (that's why they are `stats::` and never `math::`, which never samples) and
accept a conditioned variable: `stats::moments(x | x > 0)` summarizes only the in-condition lanes,
and its `n` reports how many survived. Their budget is the *introspection* budget (200 000 draws),
not `P`'s — a chart's numbers, not a probability's last digit. So `Q(x, 0.5)` and
`stats::quantiles(x, [0.5])` can differ in the final places: `Q` is the estimator, `stats::quantiles`
is what the picture is made of.

```noise
use rand; use vec;
X ~ normal(100, 15);
h = stats::histogram(X, 6);
m = stats::moments(X);
Print("counts:  ", h[1]);      # [248, 11058, 76769, 92348, 18937, 640]
Print("mean, sd:", m[1], m[2]);  # 100.00483911223 15.00450459098
Print("VaR95:   ", stats::quantiles(X, [0.05])[0]);  # 75.269722471943
```

**`engine::set_max_samples(N)`** sets the default Monte Carlo budget — the sample count `P`/`E`/
`Var`/`Q` use when called *without* an explicit count — to `N` (an integer `>= 1`) for the rest of
the run. It's the one-place alternative to threading `n` through every query when you want to trade
accuracy for speed (or buy more digits): `engine::set_max_samples(20000);` then a bare `P(C)` draws
20 000 samples instead of the `1e6` default. An explicit per-call count (`P(C, n)`) still overrides
it. Returns `unit` — it's a setting, not a value.

**`engine::set_max_opts(N)`** caps the *operations* each `P`/`E`/`Var`/`Q` query may spend (`N` an
integer `>= 1`), bounding complexity by the model's size rather than by a fixed draw count. A query
over a cone of `C` distinct nodes costs `draws × C` per-lane operations, so it auto-clamps its draws
to `N / C` (never below 1) — a heavier model simply draws *fewer* samples (a looser estimate)
instead of doing unbounded work. Unlike `set_max_samples` it never errors and never changes a result
exactly: it makes each query's worst-case cost **deterministic in the model size**. The query draws
the smaller of the two budgets, so `set_max_samples` still bounds light cones and `set_max_opts`
bounds heavy ones. Defaults to a built-in ceiling of `1e9` ops per query — high enough that ordinary
small-cone queries (even at millions of draws) are never clamped, so only very large models feel it.
Returns `unit` — it's a setting, not a value.

**`engine::set_resolution(N)`** sets the ambient **sampling resolution** — the length at which
reducers (`mse`, `mean`, `sum`, `dot`, …) render a lazy signal that never met an explicit length
(default `256`). It is the time-axis twin of `set_max_samples`: one measurement knob per axis, both
set once at the top instead of threaded through the math. `signal::sample(sig, n)` remains the
explicit per-site override. Returns `unit` — it's a setting, not a value. See "Signals".

Rules:
- **`builtin` is active by default**; `rand`/`math`/`vec`/`signal`/`engine` are **strict** — a
  bare name in one of them is an error until you `use` the module (or write the path). The error
  tells you the fix:
  `'unif' is in module 'rand' — add `use rand;` or write `rand::unif``.
- **`use module;`** activates a module for the rest of the program (in the REPL, the rest of the
  session). `use builtin;` is a harmless no-op. An unknown module is an error.
- **A user definition shadows a module item** — `sum(xs) = …` wins over `vec::sum` for bare
  `sum(...)`. Module membership only governs *name access*; which implementation runs is separate.
- Paths are single-level for now (`module::name`); the qualifier on a constant resolves a value
  (`math::pi`), and on a function must be called (`math::sqrt(x)`).

> **Status: implemented (internal modules).** Module membership is fixed (no user-defined modules
> or re-exports yet). The split is purely a scoping/namespacing layer over the existing builtins.
> *Note:* the core-model and random-variable sections above show distribution calls bare
> (`unif(0, 1)`) for brevity; a real program prefixes them with `use rand;` (and `use math;` /
> `use vec;`) or writes the `rand::`/`math::`/`vec::` path.

## Collections (Step 4)

The library below lives in the `vec` module (reducers/linear algebra); `Len` is in the always-on
`builtin`, the half-open range is the `a..b` operator (below), and a **batch of independent draws**
is the `~[shape]` operator (a shaped form of `~`, see §2). Add `use vec;` / `use rand;` (or paths).

Arrays are **fixed-length** sequences whose length is known when the graph is built (everything
but a `~` draw is deterministic at build time). Elements are arbitrary values — `number`, `bool`,
`dist`, or nested `array` (so a **matrix** is an array of arrays). A vector of random variables is
just an array of `dist`.

- **Literals & indexing.** `[a, b, c]` builds an array; `[]` is empty. `xs[i]` indexes it, and
  indexing chains: `M[i][j]`. The index `i` is normally a **constant non-negative integer in
  range** (out-of-bounds or a non-integer is a spanned error); a *random* numeric index lifts to
  a per-lane **gather** — each Monte Carlo lane picks its own element (`boxes[box]` in
  `prisoners.noise`, and the machinery under `rand::empirical`). Gathers run on the interpreter
  only (no JIT).
- **Arithmetic broadcasts over arrays** (NumPy-style): `[1,2,3] + [10,20,30] = [11,22,33]`,
  `1 + [1,2,3] = [2,3,4]`, `[2,4,6] / 2 = [1,2,3]`, `[1,2,3] ^ 2 = [1,4,9]`. It nests, so an
  array-of-arrays (a matrix, or an `[I, Q]` signal pair) broadcasts recursively. Lengths must match.
- **`sin`/`cos`/`atan` are ufuncs** (in `math`): a scalar computes directly, a random variable
  lifts to a graph node (sampled per lane — `E[cos(X)] = e^{-σ²/2}`), and an **array maps
  elementwise**. So `cos(phase_vector)` builds a waveform and `atan(noisy_Q / noisy_I)` demodulates
  one with the same function. (`sqrt` over an RV is `^ 0.5`.)
- **`for x in xs { body }`** is a **build-time unroll**: it evaluates `xs` to a concrete array,
  then runs `body` once per element with `x` bound in the *current* frame. Bindings leak (blocks
  don't scope — see below), which is exactly how an accumulator persists:
  `acc = 0; for x in xs { acc = acc + x }; acc`. The loop evaluates to `unit`. Because the body is
  re-run per element, each `~` inside is a **distinct node** — `n` independent draws for free.
- **Comprehensions** `[for x in xs { body }]` build an array by collecting `body` over `xs` —
  literally the `for x in xs { body }` loop wrapped in `[ ]` so each body value is kept instead of
  discarded. The body sees the **outer environment** (it closes over surrounding variables, so no
  closures or higher-order `map` are needed): `fx = [for x in 0..Q { (a ^ x) % N }]`.
- **`continue`** skips the rest of the enclosing loop body. In a `for` loop it discards that
  iteration's side effects; in a comprehension it **omits** that element — which is how a
  comprehension *filters*: `evens = [for x in 0..10 { if x % 2 != 0 { continue }; x }]`. The skip
  condition must be **deterministic** (a `continue` guarded by a random variable is an error — the
  array length is fixed at build time and can't vary per Monte Carlo lane). Mechanically, `continue`
  evaluates to a control sentinel that short-circuits the current `{ block }` and is consumed by the
  loop; using it as a data value (`1 + continue`) is a type error.

### The collections / linear-algebra library

These ship as builtins, but each is *also expressible in Noise* (the equivalence is enforced by
tests). Reducers fold with the ordinary operators, so a constant and a random element are handled
identically (e.g. `sum` over `dist` elements lifts to an Add-chain RV).

- **Ranges:** `a..b` is the half-open integer range `[a, b)` as an array (Rust-style): `0..n` has
  `n` elements `0 … n-1`, `a >= b` is empty. The bounds are full expressions (`i + 1 .. Len(xs)`)
  and must be deterministic integers. This replaces the old `range` builtin — `for i in 0..n { … }`.
- **Length:** `Len(xs)` — the element count (a build-time constant; capital-only `builtin`).
- **Construction:** `~[n] d` (an array of `n` independent draws of recipe `d`), `~[n, m] d` (an
  `n`×`m` matrix of independent draws) — the shaped draw operator (§2); and `rotation(d)`, a
  **recipe** for a fresh random `d`×`d` **orthonormal** matrix (a Haar rotation, built by
  Gram–Schmidt of a Gaussian seed). Like any distribution it is drawn with `~` — `Pi ~ rotation(d)`
  gives a fresh rotation per sample, and `~[k] rotation(d)` an array of `k` independent ones. Arrays
  are fixed-size — there is no append/`push`; build them with literals, `a..b`, `~[shape]`, or the
  `vec` constructors below.
- **Reducers:** `sum`, `prod` (product; `prod([]) == 1`), `mean`, `count` (number of true
  elements), `any` (`||`), `all` (`&&`), `max`, `min`.
- **Scans (running folds):** `cumsum`, `cumprod`, `cummax`, `cummin` — array → same-length array
  whose element `t` is the fold of `xs[0..=t]` (element 0 is `xs[0]`; empty → empty; a matrix
  broadcasts like `sum`: `cumsum([[1,2],[3,4]]) == [[1,2],[4,6]]`). Scans make **fixed-horizon
  paths** idiomatic: a random walk is `cumsum(increments)`, a compounding path is
  `cumprod(1 + rets)` (or, exactly, `s0 * math::exp(cumsum(logrets))`), a barrier is
  `any(path < b)`, and a worst drawdown is `min(path / cummax(path)) - 1`.
- **`plot::fan(path)`** (in the `plot` module) — the cone chart for an array of random values (a
  path): all indices are sampled **jointly in one pass**, and the per-index
  `q05/q25/q50/q75/q95` quantile bands become a cone: two translucent bands with the median line
  on top (the CLI prints the same numbers as a text card). A deterministic array gives the
  degenerate fan (every band equals the values); a scalar or a matrix is a friendly error; a
  path caps at 1024 elements.
- **Linear algebra:** the `@` operator is the **matrix product** — `v @ w` (dot), `M @ v` (mat·vec),
  `M @ N` (matmul), dispatched by shape; `*` stays elementwise/broadcast, and vector `+`/`-` are
  elementwise add/sub. Also `dot`, `normsq`,
  `norm`, `scale(v, c)`, `vsign` (elementwise ±1), `transpose(M)`,
  `normalize`, the constructors `ones(n)`/`zeros(n)`/`iota(n)`,
  `mse(a, b)` (mean squared error between two equal-length signals), `quantize(v, centroids)` (snap
  each coordinate of `v` to its nearest value in a constant codebook — the optimal scalar/Lloyd–Max
  quantizer), plus `has_duplicates(xs)` (true iff some pair is equal — the birthday predicate) and
  `count_duplicates(xs)` (how many pairs `i<j` are equal — the number of birthday collisions, of which
  `has_duplicates` is just the `> 0` case). `dot`/`mse` and vector `+`/`-` require equal-length vectors; `transpose` a rectangular matrix (array of
  equal-length rows).

```
use rand; use vec;
days ~[23] unif_int(1, 365);         # 23 independent birthdays
P(has_duplicates(days))               # ≈ 0.507 — the birthday paradox in one expression

clt = sum(~[12] unif(0, 1)) - 6;     # sum of 12 uniforms, centered → ~ N(0, 1)
P(clt > 1)                           # ≈ 0.159, agreeing with normal(0, 1)
```

> **Status: implemented.** A user definition shadows a library name (your `sum` wins). Deferred:
> first-class functions / `map` (element-wise ops are explicit loops/builtins for now),
> random-length loops / early stopping, and a native columnar vector representation (so a `d×d`
> mat·vec builds `O(d²)` nodes, and a scan unrolls to `O(steps)` fold nodes — fine for
> `d ≤ ~64`). See `plans/PLAN-COLLECTIONS.md` §1.

### Signals (lazy waveform expressions)

A **signal** is a waveform described by *frequency*, not yet by samples — a lazy expression.
`signal::sine(f)` / `cosine(f)` produce `f` cycles over an as-yet-unknown window and cost O(1)
memory. Operations **defer** into the signal's expression tree — scalar arithmetic
(`1 + 0.3*sine(3)`), **signal×signal arithmetic** (`sine(3) + sine(7)` is a two-tone chord), the
`sin`/`cos`/`atan`/`sign`/`floor`/`ceil` ufuncs, prefix `-`, and `math::exp` — so a whole
processing chain stays symbolic. A signal **materializes** to a concrete array when:

- it meets a sized array (broadcast), adopting that array's length — e.g. `0.3 * sine(3) + xs`
  samples the tone to `Len(xs)`;
- you call `signal::sample(sig, n)` to take `n` samples explicitly (Nyquist's knob); or
- a **reducer** (`mse`, `mean`, `sum`, `dot`, …) or `plot::line` meets it, rendering it at the
  **ambient resolution** — `engine::set_resolution(N)`, default `256`. The resolution is a
  measurement knob (the time-axis twin of the Monte-Carlo budget), so it lives next to the
  measurement, not inside the math.

One honest asterisk: `E`'s reported digits reflect *measured* Monte-Carlo error; resolution bias
(aliasing, under-resolved nonlinearities) is **not** in those error bars. If the answer depends on
the resolution, raise `set_resolution` until it stops moving.

**Noise generators are undrawn distributions.** `signal::noise_white(sigma)` (and
`noise_brown`/`noise_pink`/`noise_ou(sigma, tau)`/`noise_white_complex`) are subject to the same
core rule as `normal(0, 1)`: you cannot operate on them — **`~` is the only way in**.

```
static ~ signal::noise_white(sigma);   # ONE realization; length still lazy
w ~[n] signal::noise_white(1);         # realization pinned to n, materialized eagerly (an array of RVs)
noise_white(1) + msg                   # error: undrawn distribution — draw it first with `~`
sample(noise_white(1), n)              # error: same (drawing is `~`'s job, not sample's)
```

A `~`-bound noise is a **drawn realization**: every mention is the *same* noise — `static - static`
is exactly `0`, and re-materializing it (two `sample` calls, two carriers) reuses the same RV
nodes, so "both sides see the same static" is a property of `~`, not of program ordering. Each
sample is `normal(0, sigma)` (the strength is **per-sample**, matching `normal`; there is no
power-spectral-density convention). The plain-`~` form stays length-lazy and **pins its length at
first materialization**; a later mention at a *different* length is an error naming the pinned one
(white noise has no finer version of itself, so a silent re-render would be a lie). The `~[n]` form
pins up front. An `=`-bound generator stays a recipe: each `~` of it draws an independent
realization.

**`signal::noise_white_complex(sigma)`** is the complex (circularly-symmetric) white noise —
radio static. Drawing it yields a **complex signal** whose two channels are independent
`normal(0, sigma/√2)` lanes per sample, so `E|z|² = sigma²` (the same total-power convention as
`rand::normal_complex`).

**Complex signals** need no new type: a real signal meeting `math::i` (or a drawn complex noise)
gives `complex` with lazy channels. `math::exp(i*θ)` (Euler), `abs`, `arg`, and complex `+ - * /`
all defer; `signal::sample` of a complex signal yields an array of complex. `plot::line` of a
complex signal is an error — take `re`/`im`/`abs` first (a complex wave has no single trace).

```
engine::set_resolution(64);            # the one resolution choice, next to the measurement
msg    = 0.3 * signal::sine(3);        # lazy — a waveform, not yet samples
static ~ signal::noise_white_complex(0.4);
rx     = math::exp(math::i * 3 * msg) + static;   # FM-modulate, add static — still lazy
err    = E(vec::mse(math::arg(rx) / 3, msg));     # mse renders both at the ambient resolution
```

The two-argument `sine(n, f)` is an eager shorthand for `sample(sine(f), n)`. This mirrors the rest
of the language: a signal is symbolic until a sized context (or a measurement) forces it, just as a
random variable is symbolic until `E`/`P` forces it.

### Complex numbers

`complex` is a first-class scalar type. There is **no complex literal**: a complex value emerges
from the constant **`math::i`** (alias **`math::j`**, the imaginary unit `0 + 1i`) plus the ordinary
operators — `2 + 3*math::i` — or from a complex distribution (`rand::normal_complex`). A pure-real
expression stays a `number`; `math::i` is the only seed of complexity, and a real operand promotes
to `re + 0i` whenever it meets a complex one.

- **Operators.** `+ - * /` are complex when either operand is complex; `*`/`/` are true complex
  multiply/divide. `^` with a **constant integer** exponent is repeated multiply (`z ^ 3`). `==`/
  `!=` compare both channels. **Ordering `< > <= >=` is a type error** — ℂ has no total order
  (compare `math::abs(z)` if you mean magnitude). `%` is real-only. Arrays of complex broadcast
  elementwise like any other array; `@` is complex matmul.
- **`math::` functions** (all branch by input type — real in → real semantics, complex in → complex):
  - **`exp(z)`** — `e^z`. Real `e^x`; complex Euler `e^a·(cos b + i·sin b)`. (This is the renamed
    exponential *function*; the *distribution* is now `rand::exponential`.)
  - **`abs(z)`** — magnitude `√(re²+im²)` (real out); for a real `x` it is `|x|`.
  - **`arg(z)`** — phase `atan2(im, re)` (real out).
  - **`conj(z)`** — `re − i·im`. **`re(z)` / `im(z)`** — the channels (real out).
  - **`sqrt(z)`** — real branch is IEEE (`sqrt(-1.0)` is `NaN`); complex branch is the principal
    root (`sqrt(-1 + 0*i) == i`).
- **Distribution.** `rand::normal_complex(sigma)` draws a circularly-symmetric complex Gaussian
  (`E|z|² = sigma²`) — radio static / Rayleigh fading. Drawn with `~` / `~[n]` like any distribution.
- **`vec` over complex** (PLAN-COMPLEX §7): linear ops lift component-wise (`sum`, `mean`,
  `normalize`, `transpose`); magnitude ops return a **real** (`normsq = Σ|zᵢ|²`, `norm`, `mse`);
  ordering ops are a type error (`max`, `min`, `quantize`). `dot` stays **bilinear** (`Σ aᵢbᵢ`, no
  conjugation, matching `@`); **`vdot`** is the **Hermitian** inner product `Σ conj(aᵢ)·bᵢ`, and
  **`adjoint`** is the conjugate transpose. **`outer(a, b)[i][j] = aᵢ·bⱼ`** builds a matrix.

```
z = 2 + 3*math::i;                    # complex emerges from math::i
math::abs(z)                          # 3.6055… (a real)
math::exp(math::i * math::pi)         # ≈ -1  (Euler's identity)
static ~[64] rand::normal_complex(1)  # 64 iid complex-Gaussian noise samples
```

## Scope

There is currently **one flat environment**. Blocks do **not** introduce a new scope —
bindings made inside a block remain visible afterwards (`a = { b = 6; c = (b+1)/2 }; b + c`
is valid and equals `9.5`). This mirrors the legacy semantics and will be revisited when
user-defined functions land *(Phase 3)*.

## Programs and results

A program is a sequence of statements; its value is the value of the **last** statement
(or `unit` if empty). The CLI prints that value; the REPL evaluates one line at a time
against a persistent environment.

### The document contract

A run produces exactly one **`Document`** (PLAN-LITERATE §D5) — a self-describing structure every
host (CLI, playground, `@noiselang/core`) renders:

- `meta` — the frontmatter (title, abstract, tags, extra).
- `blocks` — one flat, ordered array in emission order, each tagged `code` (a verbatim source
  group), `note` (emitted text — a template or a `Print`, with an optional `syntax` tag), `plot`
  (a `plot::*`/`describe` chart), or `input` (an inline `input::…` control). A note/plot/input
  carries the `stmt_span` of the statement that produced it, so a host can group outputs under their
  code or highlight the producing line. `result.inputs` additionally lists every input as a manifest.
- `comments` — the annotation *layer*: each comment a `(self_span, code_span?)` pair. An attached
  comment names the code it annotates (one line or a whole group); a detached run (blank line
  between it and the code) has no `code_span` and reads as free-standing prose.
- `result` — `{ value?, error?, stats, truncated? }`. `value` is `{ kind, text }` (absent when the
  program ends in `unit`); `error` is spanned and still returns all blocks that ran before it;
  `truncated` is set when a run exceeds the emission cap (it keeps running, but stops recording).

Every view is a pure filter over `blocks`: *only plots*, *hide code*, *only text*, or the full
literate render — no re-run, one contract.

## Not yet implemented (see plans/PLAN.md / GOAL.md)

- *(Phase 3)* the remaining builtins — `P`, `plot`, `explain` — the `===` description
  operator, and user-defined functions (`f(x) ~ { ... }`). `unif` is live; other call names
  still error with "builtins arrive in Phase 3". There is no language-surface sampling
  builtin yet — sampling is a Rust API.
- A true point-mass distribution for `X ~ <number>` (currently binds a plain `number`).
- String escapes, number exponent syntax, and block-level scoping.
