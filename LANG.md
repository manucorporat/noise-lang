# LANG.md — the Noise language specification

The single source of truth for Noise's syntax and semantics. **The core model below is the
official target**; `crates/noise-core` is being brought into line with it. Where the
implementation currently lags, a **Status:** note says so — when the code and this document
disagree on the *model*, the document wins (the code is the thing being fixed). For everything
else, the golden tests enforce the spec: fix the bug or fix the spec, never let them drift.
Sections marked *(Phase N)* are not implemented yet (see PLAN.md).

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
P(has_duplicate(days))
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

- **Whitespace** (spaces, tabs, newlines) separates tokens and is otherwise insignificant.
- **Comments** run to end of line, started by `#` or `//`.
- **Numbers**: `[0-9]+` or `[0-9]*.[0-9]+` / `[0-9]+.[0-9]*`, parsed as `f64`. No exponent
  syntax yet. A leading `-` is *not* part of the literal — it is the unary minus operator.
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`. Case-sensitive. `if`, `else`, `for`, `in`, `true`,
  and `false` are reserved.
- **Booleans**: `true` and `false` are literals — point masses on `{true, false}` (i.e.
  `bernoulli(1)` and `bernoulli(0)`; see core model §1).
- **Strings**: double-quoted, no escape sequences yet (`"like this"`). A newline or EOF
  before the closing quote is an error.

### Tokens

```
+  -  *  /  **            arithmetic  (`*` is elementwise / broadcast)
@                         matrix product  (dot / matvec / matmul by shape)
== != < > <= >=           comparison
&& ||                     logical and / or
!                         logical not (prefix)
=                         assignment bind
~  ~[n]  ~[n, m]          sample / distribution bind (optional draw shape)
( ) { }                   grouping, blocks
[ ]                       array literal / indexing
..                        half-open integer range (0..n)
::                        module path separator (rand::unif)
,  ;                      argument sep, statement terminator
if else for in use        keywords
true false                boolean literals
```

## Grammar (informal)

```
program   := stmt*                         # statements, ';'-separated
stmt      := expr ( ';' )?                 # trailing ';' optional; ';' may also separate
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

Lowest to highest. All binary operators are left-associative **except** `**` and
binding, which are right-associative. Prefix `-`/`!` bind tighter than everything below
`**` (so `-2 ** 2 == -(2 ** 2) == -4`, matching common math convention).

| Level | Operators            | Assoc  |
|-------|----------------------|--------|
| 1     | `=` `~` (binding)    | right  |
| 2     | `..` (range)         | left   |
| 3     | `\|\|`               | left   |
| 4     | `&&`                 | left   |
| 5     | `== != < > <= >=`    | left   |
| 6     | `+ -`                | left   |
| 7     | `* / @`              | left   |
| 8     | `**`                 | right  |
| 9     | prefix `- ! ~`       | —      |
| 10    | postfix `[index]`    | —      |
| 11    | call, grouping       | —      |

## Values and types

Conceptually every value is a distribution (see **The core model §1**); the types below are the
*current implementation's* representation of that idea, which still splits the distribution
family into separate runtime types.

Runtime values: `number` (`f64`, a point mass), `bool` (a point mass on `{T,F}` — i.e. a
Bernoulli; see core model §1), `string`, `unit` (`()`), `array` (a fixed-length, build-time
sequence of values — see "Collections"), and `signal` (a **lazy waveform generator** — see
"Signals"). An **`estimate`** is a `number` carrying a standard error (produced by `P`); it
behaves like a number in arithmetic (propagating the error) and displays rounded to its
justified precision. **`dist`** — a non-degenerate random variable / distribution handle — is
produced by `unif`/`unif_int`/`bernoulli`, bound by a binding, and propagated by operator
lifting (see "Random variables"). A `dist` Displays as `<dist #n>` and never samples on Display;
sampling is forced by `P` (or the Rust API `Engine::sample` / `Engine::moments`). Per the core
model, `number`/`bool`/`dist`/`estimate` are all the *same kind of thing* (a distribution); the
split is the implementation detail being removed.

Type rules (current):
- Arithmetic `+ - * / **` require both operands to be `number` → `number`.
  Division and `**` follow IEEE-754 (`1/0 == inf`, no panic).
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

- **`unif(a, b)`** is a distribution constructor. `a` and `b` must be deterministic `number`s
  (a `dist` argument is a spanned error). It returns an undrawn **recipe** for the uniform on
  `[a, b)`; `X ~ unif(a, b)` draws a random variable from it (see "Binding semantics" above).
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
- **`exp(rate)`** — the exponential `Exp(rate)` (continuous; `rate > 0`, `mean = 1/rate`).
  Inverse-CDF sampled.
- **`poisson(lambda)`** — Poisson counts (discrete, `lambda > 0`, `mean = variance = lambda`),
  support `0, 1, 2, …`. Sampled via Knuth's algorithm.
- **`geometric(p)`** — the number of failures before the first success (discrete, `0 < p <= 1`,
  support `0, 1, 2, …`, `mean = (1-p)/p`). `==`/counts are meaningful.
- **The `_int` family** — `normal_int(mu, sigma)` and `exp_int(rate)` draw the matching continuous
  distribution and round each draw to the nearest integer, so `==`/counts are meaningful (the
  discrete `unif_int` is already its own constructor).
- **`E(x)`** / **`E(x, n)`** and **`Var(x)`** / **`Var(x, n)`** — the Monte Carlo **expectation**
  and **variance** of a *numeric* quantity (a number or a numeric/bool RV), the companions to
  `P` for non-events. Both return an `estimate`: `E` carries the standard error of the mean
  (`sqrt(Var/n)`), `Var` an asymptotic `var·sqrt(2/n)`; a deterministic value is exact
  (`E(5) = 5 ± 0`). `E` of a bool-RV equals `P`. Default `n = 1e6` (set per-run with
  `engine::set_max_loops`), fixed seed.
- **`Q(x, q)`** / **`Q(x, q, n)`** — the **quantile** (inverse CDF) of a *numeric* quantity at
  level `q ∈ [0, 1]`: `Q(X, 0.5)` is the median, `Q(X, 0.95)` the 95th percentile, and
  `Q(X, 0)`/`Q(X, 1)` the min/max draw. The companion to `E`/`Var` for tail/spread questions.
  Estimated by Monte Carlo — draw `n` samples (default `1e6`, or `engine::set_max_loops`; fixed
  seed), sort, and linearly
  interpolate between the bracketing order statistics. Returns a plain `number` (unlike `P`/`E`,
  it does *not* auto-round to a confidence precision: a sample quantile's error depends on the
  density there). A deterministic value is its own quantile at every level.
- **`sqrt(x)`** (`x >= 0`) and the constants **`pi`**, **`e`** — math helpers for scaling
  factors. `pi`/`e` are bare identifiers resolved as constants, so they also work *inside*
  function bodies (which otherwise see only their parameters).
- **`P(event)`** / **`P(event, n)`** — the probability that a bool-RV (or deterministic bool)
  is true, returned as a plain `number` in `[0,1]` so it composes in arithmetic (`4 * P(C)`).
  `P` of a numeric (non-event) value is an error. Estimated by Monte Carlo over `n` samples
  (default `1e6`, or set per-run with `engine::set_max_loops`) under a fixed seed, so a run is
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
  It is **not** sequential branching — the engine still samples independent lanes; a lifted `if`
  cannot carry state across a time step (that's the dynamics fork in `PLAN.md`).

### Hazards and still-planned semantics

- **Equality on a *continuous* RV is almost surely false.** `unif(a,b)` is continuous, so
  `X == c` (and any `==` between continuous RVs) has probability ~0 — e.g. `unif(1,6) == 4`
  is essentially never true. Use a discrete distribution (`unif_int`, `bernoulli`). A
  **warning** on `==`/`!=` over a continuous RV is still TODO.
- **Multiple queries do not yet share one sampling pass (TODO).** Each `P()` call currently
  samples its own cone with the default seed. The intended semantics is that all queries in
  a run share *one* batch of draws so `P(A)`, `P(B)`, `P(A && B)` are mutually consistent
  (`P(A && B) ≤ P(A)`). Not yet implemented.
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
| `rand`    | `unif`, `unif_int`, `bernoulli`, `normal`, `normal_int`, `exp`, `exp_int`, `poisson`, `geometric`, `rotation` (batched sampling is the `~[shape]` operator, not a builtin) | needs `use rand;` |
| `math`    | `pi`, `e`, `sqrt`, `round`, `log` (natural), `log10`, `sin`, `cos`, `atan`, `sign` | needs `use math;` |
| `vec`     | `sum`, `count`, `any`, `all`, `max`, `min`, `mean`, `dot`, `normsq`, `norm`, `vadd`, `vsub`, `matvec`, `transpose`, `normalize`, `quantize`, `has_duplicate`, `mse`, `ones`, `zeros`, `iota` | needs `use vec;` |
| `signal`  | `sine`, `cosine` (lazy waveforms), `noise_white` (lazy white noise), `sample` | needs `use signal;` |
| `engine`  | `set_max_loops` (run-time evaluator knobs) | needs `use engine;` |

**`engine::set_max_loops(N)`** sets the default Monte Carlo budget — the sample count `P`/`E`/
`Var`/`Q` use when called *without* an explicit count — to `N` (an integer `>= 1`) for the rest of
the run. It's the one-place alternative to threading `n` through every query when you want to trade
accuracy for speed (or buy more digits): `engine::set_max_loops(20000);` then a bare `P(C)` draws
20 000 samples instead of the `1e6` default. An explicit per-call count (`P(C, n)`) still overrides
it. Returns `unit` — it's a setting, not a value.

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
  indexing chains: `M[i][j]`. The index `i` must evaluate to a **constant non-negative integer in
  range** — never a random variable (a random gather is the dynamics fork). Out-of-bounds, a
  non-integer, or a `dist` index is a spanned error.
- **Arithmetic broadcasts over arrays** (NumPy-style): `[1,2,3] + [10,20,30] = [11,22,33]`,
  `1 + [1,2,3] = [2,3,4]`, `[2,4,6] / 2 = [1,2,3]`, `[1,2,3] ** 2 = [1,4,9]`. It nests, so an
  array-of-arrays (a matrix, or an `[I, Q]` signal pair) broadcasts recursively. Lengths must match.
- **`sin`/`cos`/`atan` are ufuncs** (in `math`): a scalar computes directly, a random variable
  lifts to a graph node (sampled per lane — `E[cos(X)] = e^{-σ²/2}`), and an **array maps
  elementwise**. So `cos(phase_vector)` builds a waveform and `atan(noisy_Q / noisy_I)` demodulates
  one with the same function. (`sqrt` over an RV is `** 0.5`.)
- **`for x in xs { body }`** is a **build-time unroll**: it evaluates `xs` to a concrete array,
  then runs `body` once per element with `x` bound in the *current* frame. Bindings leak (blocks
  don't scope — see below), which is exactly how an accumulator persists:
  `acc = 0; for x in xs { acc = acc + x }; acc`. The loop evaluates to `unit`. Because the body is
  re-run per element, each `~` inside is a **distinct node** — `n` independent draws for free.

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
- **Reducers:** `sum`, `mean`, `count` (number of true elements), `any` (`||`), `all` (`&&`),
  `max`, `min`.
- **Linear algebra:** the `@` operator is the **matrix product** — `v @ w` (dot), `M @ v` (matvec),
  `M @ N` (matmul), dispatched by shape; `*` stays elementwise/broadcast. Also `dot`, `normsq`,
  `norm`, `scale(v, c)`, `vadd`, `vsub`, `vsign` (elementwise ±1), `matvec(M, v)`, `transpose(M)`,
  `normalize`, the constructors `ones(n)`/`zeros(n)`/`iota(n)`,
  `mse(a, b)` (mean squared error between two equal-length signals), `quantize(v, centroids)` (snap
  each coordinate of `v` to its nearest value in a constant codebook — the optimal scalar/Lloyd–Max
  quantizer), plus `has_duplicate(xs)` (true iff some pair is equal — the birthday predicate).
  `dot`/`vadd`/`vsub`/`mse` require equal-length vectors; `transpose` a rectangular matrix (array of
  equal-length rows).

```
use rand; use vec;
days ~[23] unif_int(1, 365);         # 23 independent birthdays
P(has_duplicate(days))               # ≈ 0.507 — the birthday paradox in one expression

clt = sum(~[12] unif(0, 1)) - 6;     # sum of 12 uniforms, centered → ~ N(0, 1)
P(clt > 1)                           # ≈ 0.159, agreeing with normal(0, 1)
```

> **Status: implemented.** A user definition shadows a library name (your `sum` wins). Deferred:
> first-class functions / `map` (element-wise ops are explicit loops/builtins for now), random
> indexing / random-length loops, and a native columnar vector representation (so a `d×d` matvec
> builds `O(d²)` nodes — fine for `d ≤ ~64`). See `PLAN-COLLECTIONS.md` §1.

### Signals (lazy waveforms)

A **signal** is a waveform described by *frequency*, not yet by samples — a lazy generator.
`signal::sine(f)` / `cosine(f)` produce `f` cycles over an as-yet-unknown window and cost O(1)
memory. Scalar arithmetic and the `sin`/`cos`/`atan`/`sign` ufuncs **defer** into the signal (it
stays a generator), and it **materializes** to a concrete array only when:

- it meets a sized array (broadcast), adopting that array's length — e.g. `0.3 * sine(3) + xs`
  samples the tone to `Len(xs)`; or
- you call `signal::sample(sig, n)` to take `n` samples explicitly.

**`signal::noise_white(sigma)`** is a lazy generator too, but **random**: zero-mean white noise
with no length yet. Unlike the deterministic waveforms it materializes into `n` fresh
iid `normal(0, sigma)` **random variables** (so `E`/`P`/`Var` average over realizations), one
independent draw per leaf-vector lane — so adding it to an `[I, Q]` carrier gives I and Q *distinct*
noise. It pins its length the same way: meeting a sized array, or `sample(noise_white(s), n)`.

```
msg = sample(0.3 * sine(3), 64);     # render the lazy tone at 64 points (the one resolution choice)
rx  = msg + noise_white(s);          # noise adopts length 64 here, as 64 fresh iid normal RVs
sample(cosine(7), 10)                # explicit: 10 samples of a 7-cycle wave (Nyquist's knob)
```

The two-argument `sine(n, f)` is an eager shorthand for `sample(sine(f), n)`. This mirrors the rest
of the language: a signal is symbolic until a sized context (or `sample`) forces it, just as a
random variable is symbolic until `E`/`P` forces it.

## Scope

There is currently **one flat environment**. Blocks do **not** introduce a new scope —
bindings made inside a block remain visible afterwards (`a = { b = 6; c = (b+1)/2 }; b + c`
is valid and equals `9.5`). This mirrors the legacy semantics and will be revisited when
user-defined functions land *(Phase 3)*.

## Programs and results

A program is a sequence of statements; its value is the value of the **last** statement
(or `unit` if empty). The CLI prints that value; the REPL evaluates one line at a time
against a persistent environment.

## Not yet implemented (see PLAN.md / GOAL.md)

- *(Phase 3)* the remaining builtins — `P`, `plot`, `explain` — the `===` description
  operator, and user-defined functions (`f(x) ~ { ... }`). `unif` is live; other call names
  still error with "builtins arrive in Phase 3". There is no language-surface sampling
  builtin yet — sampling is a Rust API.
- A true point-mass distribution for `X ~ <number>` (currently binds a plain `number`).
- String escapes, number exponent syntax, and block-level scoping.
