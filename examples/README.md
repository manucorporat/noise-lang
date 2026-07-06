# Noise examples

A gallery of runnable Noise programs that stress-test the language. Each is a real Monte
Carlo experiment whose answer has a known closed form, so the printed output can be checked.

Run one with:

```sh
$ cargo run -p noise-cli -- examples/monty_hall.noise
P(win by switching) = 0.6668
```

Each example ends in `Print(...)`, building a message with string concatenation and `round`.
All of these exercise the live language: `unif` / `unif_int` / `bernoulli`, `~` random-variable
bindings, operator lifting (`+ - * / ^`, comparisons, `&& || !`), lifted `if`, `P(event)`,
and `Print` / `round` / string `+`. Several also use **collections** — the shaped draw `~[n] d`
to sample `n` independent variables (`~[n, m] d` for a matrix) and reducers like `sum` / `count` /
`any` / `all` / `has_duplicates`. Names
are **module-scoped** (Rust-style): each example opens with `use rand;` / `use math;` / `use vec;`
to bring the distributions, math, and vector helpers into scope (`P` / `Print` / `Len` and the
`a..b` range are in the always-on `builtin` module). Sampled values use `P`'s fixed default budget
(N = 1e6, fixed seed), so they're reproducible. `P` returns an **estimate that carries its
standard error**: the printed digits are the ones the sample size justifies, and the error
*propagates through arithmetic* — pass a larger `N` (e.g. `P(C, 100000000)`) to reveal more,
and a derived value like `4 * P(C)` or a ratio rounds itself correctly.

| Example | What it shows | Analytic | Printed |
|---|---|---:|---:|
| `pi.noise` | Monte Carlo π via `4·P(inside circle)` | 3.14159 | 3.14 |
| `dice.noise` | discrete die; `P(Dice == 4)` (why `unif_int`, not `unif`) | 0.16667 | 0.166 |
| `dice_sum.noise` | two **independent** dice via `sum(iid(…, 2))`; `P(sum == 7)` | 0.16667 | 0.166 |
| `advantage.noise` | D&D advantage: keep the higher of 2d20 (`max` via `if`); `P(≥ 15)` | 0.51 | 0.51 |
| `max_of_dice.noise` | `max(A,B)` over RVs via a lifted `if`; `P(higher == 6)` | 0.30556 | 0.306 |
| `dice_bet.noise` | a payoff random variable via `if`; `P(profit > 0)` | 0.16667 | 0.167 |
| `insurance.noise` | insurance payout with a deductible (`if`); `P(insurer pays)` | 0.8 | 0.8 |
| `coin_streak.noise` | "3 heads in a row" as 3 independent coins (not `P(H)³` by hand) | 0.125 | 0.125 |
| `exactly_two_heads.noise` | a tiny Binomial built from boolean events | 0.375 | 0.375 |
| `monty_hall.noise` | Monty Hall reframed: switching wins iff first pick was wrong | 0.66667 | 0.667 |
| `birthday.noise` | birthday paradox for a group of 5 (one term per pair) | 0.0271 | 0.027 |
| `prisoners.noise` | **the 100 Prisoners Riddle** — cycle-following strategy; boxes are a `permutation(n)` and a random box index is a per-lane *gather* (`boxes[box]`) | 0.3118 | 0.31 |
| `reliability.noise` | 3-way parallel redundancy at 0.9 each; `P(any up)` | 0.999 | 0.999 |
| `conditional_bayes.noise` | conditional probability with the `\|` bar: `P(D==6 \| D>3)` (≡ the ratio) | 0.33333 | 0.334 |
| `beta_bernoulli.noise` | **Bayesian coin** — flat prior on the bias, a random parameter feeding `bernoulli`, then `\|` to read off the posterior mean / interval / predictive after 7-of-10 heads | 0.6667 | 0.667 |
| `irwin_hall.noise` | `P(sum of three U(0,1) > 2)` | 0.16667 | 0.167 |
| `clt_normal.noise` | a standard normal built from 12 uniforms (CLT); a tail prob | ~0.159 | 0.16 |
| `functions.noise` | user functions: `max(a,b)=…` (pure, lifts over RVs) + `roll()~…` (draws per call) | 0.30556 / 0.16667 | 0.306 / 0.166 |
| `qjl_scalar.noise` | QJL unbiasedness in 1-D (TurboQuant building block): `normal`, `E`/`Var`, `sqrt`, `pi` | 1.0 / 0.5708 | 1 / 0.572 |
| `turboquant.noise` | **the d-dim capstone** — the MSE 1-bit quantizer is inner-product biased by 2/π *and* carries ~3× the squared error; the QJL rescaling is unbiased (`~[d,d]`, `@`, `transpose`, `sign`, `E`) | bias 2/π; err ≪ | 0.64× / 1.0×; err 0.11 vs 0.035 |
| `am_vs_fm.noise` | telecom, end to end: `mse(demodulate(modulate(msg) + static), msg)` — same static, but FM (message in the angle) recovers cleaner than AM (message in the amplitude). Uses lazy `sine` + `cos`/`sin`/`atan` ufuncs + array broadcasting | FM ≫ AM cleaner | AM 0.087, FM 0.014 (6× cleaner) |
| `am_vs_fm_complex.noise` | the **complex-number** retelling of `am_vs_fm`: the carrier is one complex phasor `z`, AM writes the message into `\|z\|` and FM into `arg(z)`, and static is a single `rand::normal_complex` draw (circular symmetry is now a property of the *type*). Uses `math::i`/`exp`/`abs`/`arg` | FM ≫ AM cleaner | AM 0.044, FM 0.005 (8× cleaner) |
| `shor_period.noise` | **Shor's factoring algorithm** end to end as `shor(N)`: a complex inverse-QFT (`math::exp` + `vec::outer` + complex `@`) makes the inputs interfere into a comb whose spike count is the period `r`, then `gcd(a^(r/2)±1, N)` yields the factors. Uses the `math::gcd`/`math::modpow` builtins, comprehensions, and `for`-loop control flow | `shor(15)` | `[3, 5]` |
| `nyquist.noise` | the Nyquist–Shannon theorem by counterexample: a 7-cycle wave sampled below `2·7` aliases into a 3-cycle one (identical samples); above, they separate. Lazy `signal` + `sample(sig, n)` | 0 vs > 0 | 0 (aliased) / 1 (resolved) |

## What these deliberately show about the design

- **Collections make independence a one-liner.** `birthday`, `dice_sum`, `coin_streak`,
  `exactly_two_heads`, `irwin_hall`, and `reliability` use the shaped draw `~[n] d` to sample `n`
  *independent* variables at once, then a reducer (`sum`/`count`/`any`/`all`/`has_duplicates`).
  `birthday` scales to 23 people — 253 pairwise comparisons — that the old hand-unrolled form
  couldn't express.
- **The sharing rule still holds underneath.** `~[n] d` produces `n` *distinct* draws (independence);
  reusing one name twice would instead reuse one draw (`Dice + Dice` is `2·Dice`). See "Random
  variables and sharing" in `../LANG.md`.
- **Modeling, not hand-arithmetic.** `coin_streak` and `exactly_two_heads` *model* the events with
  independent random variables and boolean logic, rather than multiplying probabilities by hand.
- **Conditioning with the `|` bar.** `conditional_bayes` uses `P(D == 6 | D > 3)` — Bayes scoped to
  the query, no `observe`/side effect — and binds a conditioned value (`D | D > 3`) to reuse across
  `E`/`Q`. It's exactly `P(A && C) / P(C)`, just less ceremony. (Rejection-based, so best when the
  condition isn't rare; continuous/rare-event conditioning is the separate inference track.)
- **Hierarchical models & Bayesian inference.** `beta_bernoulli` puts a *random parameter* into a
  distribution (`bias ~ unif(0,1); flips ~[10] bernoulli(bias)`) — a prior — then conditions on the
  data (`| count(flips) == 7`) to read off a posterior mean, credible interval, and predictive. The
  inference is rejection-based (keep the lanes matching the data): great for a few discrete
  observations, not yet for lots of continuous data (that needs importance/MCMC weighting).
- **`if` over a random variable is a value, not control flow.** `dice_bet`, `insurance`,
  `advantage`, and `max_of_dice` use `if cond { a } else { b }` where `cond` is a random event —
  it builds a new random variable by selecting per sample (and gives `max`/`min`/`abs` for free).
- **Feed-forward d-dimensional experiments reproduce real research.** `turboquant` draws a fresh
  `d×d` Gaussian projection per sample, runs matrix–vector products and reductions, and recovers a
  published bias (`2/π`) and its fix — empirical validation of an arXiv paper in ~20 readable lines.

## Where the language hits its ceiling (honest limits)

These are *not* expressible today, by design — they need features beyond the static
random-variable algebra (see the "dynamics fork" in `../plans/PLAN.md`):

- **Sequential / stateful processes.** A random walk, a Markov chain, or an **M/M/1 queue**
  (`W_{n+1} = max(0, W_n + S_n − A_{n+1})`) needs per-step state — the columnar engine samples
  independent lanes that can't carry state across a time index.
- **Sequential / stateful control flow.** You *can* now pick a **value** from a random outcome —
  `if D == 6 { 10 } else { -2 }` lifts to a per-lane select (see `dice_bet.noise`,
  `max_of_dice.noise`). What's still missing is *sequential* branching that carries state across
  a time step (the recurrences above), which is a different execution mode.
