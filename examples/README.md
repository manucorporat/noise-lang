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
| `qjl_scalar.noise` | QJL unbiasedness at d=2 (TurboQuant building block) — sign bits of Gaussian projections answer inner products against an unseen query: `~[n,2]`, `@`, `transpose`, `sign`, `E` | 1.0; err ≤ π/(2·n_bits)·‖x‖² | 1; err 0.66 vs bound 0.785 |
| `turboquant.noise` | **the d-dim capstone, as an interactive article** — the MSE 1-bit quantizer is reconstruction-optimal (D_mse ≈ paper's 0.36) yet inner-product biased by exactly 2/π; TurboQuant's QJL sketch is unbiased on the same bit budget. Sliders for `d`/θ/`m`, bias histograms, estimate-vs-truth scatters, and an error-vs-bits `plot::line(xs, ys)`; rotation invariance models Πx as a random direction and the sketch as correlated Gaussian projections (no d×d matrix) | bias 2/π; err ≤ π/(2m) | 0.494 vs 2/π·0.766 = 0.488; err 0.049 ≤ 0.0785 |
| `am_vs_fm.noise` | telecom, end to end: `mse(demodulate(modulate(msg) + static), msg)` — same static, but FM (message in the angle) recovers cleaner than AM (message in the amplitude). Uses lazy `sine` + `cos`/`sin`/`atan` ufuncs + array broadcasting | FM ≫ AM cleaner | AM 0.087, FM 0.014 (6× cleaner) |
| `am_vs_fm_complex.noise` | the **complex-number** retelling of `am_vs_fm`: the carrier is one complex phasor `z`, AM writes the message into `\|z\|` and FM into `arg(z)`, and static is a single `rand::normal_complex` draw (circular symmetry is now a property of the *type*). Uses `math::i`/`exp`/`abs`/`arg` | FM ≫ AM cleaner | AM 0.044, FM 0.005 (8× cleaner) |
| `shor_period.noise` | **Shor's factoring algorithm** end to end: a complex inverse-QFT (`math::exp` + `vec::outer` + complex `@`) over a register sized past `N²` makes the inputs interfere into a comb of spikes spaced `Q/r`, then a **continued-fraction** expansion of the measured spike recovers the period `r` — repaired by multiplying up until `a^r ≡ 1` when the convergent lands on a divisor — and `gcd(a^(r/2)±1, N)` yields the factors. Uses the `math::gcd`/`math::modpow`/`math::floor` builtins, comprehensions, and `for`-loop control flow | `N = 15` | `15 = 3 × 5` |
| `nyquist.noise` | the Nyquist–Shannon theorem by counterexample: a 7-cycle wave sampled below `2·7` aliases into a 3-cycle one (identical samples); above, they separate. Lazy `signal` + `sample(sig, n)` | 0 vs > 0 | 0 (aliased) / 1 (resolved) |
| `kelly.noise` | **the Kelly criterion** — sweep the stake fraction `f` and tabulate `E(math::log(growth))` per round (`log` of an RV); the curve peaks at the Kelly fraction and goes negative past ~0.4 (overbetting a winning game loses) | f\* = 2p−1 = 0.2; E[log g] ≈ 0.0201 | peak 0.02 at f = 0.2 |
| `bootstrap.noise` | **the bootstrap** — `rand::empirical(rets)` makes tomorrow a random draw from 24 pasted days of history (no Gaussian fitted, crash included): 1-day VaR and `P(another −4% day)`; then `block_bootstrap(rets, 5)` keeps the panic *week* glued together, so weekly VaR comes out honestly scarier than iid resampling | 1/24 ≈ 0.0417 | 0.042; week VaR −0.051 (iid) vs −0.068 (block) |
| `barrier_option.noise` | **a knock-out (down-and-out) call** — a year of *exact* GBM in one line (`s0 * exp(cumsum(logrets))`), barrier = `any(path < 80)`, worst drawdown = `min(path / cummax(path)) − 1`, and a `plot::fan` cone of the paths; the vanilla leg must land on Black-Scholes | 10.4506 | 10.5 / KO 10.4 |

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
- **Fixed-horizon paths are one-liners.** `barrier_option` builds a whole simulated year as
  `s0 * exp(cumsum(logrets))` — a scan turns 52 draws into a price path, `any`/`cummax` read the
  barrier and the drawdown off it, and `plot::fan` draws the cone. `kelly` needs `math::log` of a
  random growth factor; `bootstrap` swaps the Gaussian for history itself
  (`rand::empirical` / `rand::block_bootstrap`).

## Where the language hits its ceiling (honest limits)

The ceiling has moved: **fixed-horizon** processes are now in scope. What remains out, by
design, needs features beyond the static random-variable algebra (see the "dynamics fork" in
`../plans/PLAN.md`):

- **Fixed-horizon paths are expressible — idiomatically.** A random walk is
  `cumsum(increments)`, a compounding price path is `cumprod(1 + rets)` or, exactly,
  `s0 * exp(cumsum(logrets))`; a barrier is `any(path < b)`, a lookback is `max(path)`, a
  drawdown is `min(path / cummax(path)) − 1` (see `barrier_option.noise`). The scans
  (`vec::cumsum`/`cumprod`/`cummax`/`cummin`) cover any process whose *length is known up front*.
- **Random-length / early-stopping processes are the real ceiling.** "Expected *time* to ruin",
  an unbounded first-passage time, or a queue run *until* it empties needs a per-lane stepper
  that stops at a data-dependent step — the columnar engine samples independent lanes of a
  fixed-size graph. Within a fixed horizon, a lifted `if` still covers absorbing states
  (freeze-at-absorption: `bank = if ruined { bank } else { bank + pnl }`), but it picks a
  *value* per lane; it cannot decide *when to stop*.
