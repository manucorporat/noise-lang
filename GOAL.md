# Noise — Project Goal

## The vision

**Noise** is an expression-based, *probabilistic* programming language. Its defining
idea: a variable does not hold a single exact value — it holds a **probability
distribution** (a random variable). The language abstracts away the mechanics of
simulation so that mathematicians can write idiomatic code to reason about random
variables directly.

The intended sweet spot is making **Monte Carlo simulations** and **queueing-theory
simulations** trivial to express.

> Tagline from the README: *"Noiselang is an expression based, probabilistic language
> where variables do not take exact values but a probability distribution."*

## What the finished language should feel like

These examples come from `README.md`. **The probabilistic features below do not work yet** —
they describe the destination, not the current state. The deterministic surface they rest on
(arithmetic, comparisons, `if/else`, blocks, `=`/`~` bindings) *is* implemented today. See
`AGENT.md` for the precise current state and `PLAN.md` for the build order.

### Random-variable assignment with `~`
```
X ~ unif(-1, 1)      # X is a uniform random variable, not a number
Dice ~ unif(1, 6)
```
`~` binds an identifier to a *distribution*. `=` binds an ordinary (deterministic) value.
This `~` vs `=` distinction is the heart of the language.

**Sharing rule (the load-bearing semantic):** a name bound with `~` is *one fixed random
variable* — every mention reuses the **same** draw. So `X - X == 0` exactly, and `X + X == 2·X`.
Independent draws come only from *separate* `~` bindings (and, later, from function calls that
re-draw on each call). This means `Dice + Dice` is **not** two dice — it is one die doubled. To
roll two dice you write two bindings (`A ~ unif_int(1,6); B ~ unif_int(1,6)`). One sentence:
**same name ⇒ same draw; new binding (or call) ⇒ new draw.**

### Everything is an expression
```
X + Y

d = {a=2 b=2 c=a+b} * 10

e = if d > a {
  d
} else {
  a
}
```
Blocks, `if/else`, and assignments all evaluate to values. **This works today.**

### Operators
Arithmetic `+ - * / **`, comparison `> < == != >= <=`, and negation `!`. **All work today**
(over deterministic numbers/bools).

### Functions
```
X(y) ~ {
  x = !y;
};

max(x, y) ~ if x > y { x } else { y }
```
User-defined functions, callable; `~`-defined functions can be stochastic. (Call *syntax*
`f(args)` parses today; calling anything is an error until Phase 3.)

### Built-in probabilistic vocabulary
- `unif(a, b)` — uniform distribution (and presumably other distributions later)
- `P(event)` — probability of an event/condition (returns a number in [0,1])
- `plot(X)` — visualize a distribution
- `explain(...)` — human-readable explanation of a probability
- `===` — attach a description/label to an event, e.g. `C === "fall inside circle"`

None of these builtins exist yet.

### Worked example — estimate π (Monte Carlo)
A point uniform in the 2×2 square is inside the unit circle with probability `π/4`, so π is
`4 · P(C)` — `P(C)` itself is ≈ 0.785, not 3.14.
```
X ~ unif(-1, 1)
Y ~ unif(-1, 1)

C = X**2 + Y**2 < 1     # fell inside the circle

4 * P(C)                // ≈ 3.14
```

### Worked example — dice
A die is **discrete**: it needs `unif_int` (discrete uniform). With *continuous* `unif(1,6)`,
`P(Dice == 4)` is `0` — a continuous draw almost surely never equals 4 exactly.
```
Dice ~ unif_int(1, 6)   # integers 1..=6

P(Dice == 4)            // ≈ 1/6

# "4 ten times in a row" is a MODEL of ten independent rolls — not P(X)**10 done by hand.
# It needs fresh draws (function calls) + boolean `&&`, both planned for Phase 3.
```

## The core gap

The interpreter today is a **deterministic numeric/boolean calculator** (rebuilt from
scratch in `crates/noise-core`; the old crate is parked in `legacy/`). The deterministic
language surface is done and tested; the **probabilistic layer — the reason the language
exists — is unbuilt**:

- `~` is parsed (`BindKind::Sample`) but evaluates identically to `=` — no distribution is
  created.
- There is no distribution type, no sampling engine, no `unif(...)`, no `P(...)`, no
  `plot`/`explain`, no `===` description operator.
- `Call(name, args)` parses, but evaluating any call returns "builtins arrive in Phase 3".

Closing that gap is the goal. Milestones in dependency order (see `PLAN.md` for the full
phased roadmap and definitions of done):

1. **Deterministic core** *(done)* — floats & negatives, `**`, comparisons, `if/else`,
   blocks, strings, typed errors with spans, a runnable CLI/REPL.
2. **Random-variable runtime** — a distribution / sample representation behind a new
   `Value::Dist`, `~` binding distributions, arithmetic/comparison lifting over random
   variables, lowered to bytecode and sampled in batched/columnar form.
3. **Sampling / Monte-Carlo engine** — evaluate `P(condition)` by sampling so the π and
   dice examples produce correct numbers under a seeded RNG.
4. **Builtins & ergonomics** — `unif`, `plot`, `explain`, the `===` description operator,
   user-defined functions.
5. **Browser playground** — a `wasm-bindgen` build plus a real web UI, replacing the legacy
   Emscripten `www/` artifacts.

See `AGENT.md` for the precise current state of each of these.
