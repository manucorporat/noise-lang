# COMPARISON.md — Noise vs NumPy, Stan, and the PPL world

Where Noise sits, who it's for, and what it gives up. The thesis in one line:

> **NumPy makes you write the simulation. Stan makes you declare a model and wait for a sampler.
> Noise lets you write the probability *as math* and reads the answer back by Monte Carlo — and it
> does it in a browser tab.**

Noise is a **static random-variable algebra with forward Monte Carlo** (plus rejection-based
conditioning and hierarchical priors). It is deliberately small. That smallness is the product: it
makes the language a place to *learn and explore* probability, not a production inference engine.

---

## The landscape at a glance

| | **Noise** | **NumPy** | **Stan** | **PyMC** |
|---|---|---|---|---|
| What it is | RV algebra + Monte Carlo | array numerics | declarative Bayesian modeling | Bayesian modeling in Python |
| Core abstraction | every value is a distribution | n-d arrays | `data`/`parameters`/`model` blocks | `with Model()` context |
| How you get an answer | `P` / `E` / `Var` / `Q` force sampling | you write the loop | compile → NUTS/HMC → summarize | `pm.sample()` → arviz |
| Inference | forward MC + **rejection** conditioning | none (you build it) | **HMC/NUTS** (scales to continuous data) | HMC/NUTS, VI, SMC |
| Independence model | explicit: one `~` = one draw, names reuse it | implicit: re-call the sampler | one `~` per declared node | one distribution per RV |
| Setup to first answer | zero — type in a browser | `import numpy` | install toolchain, compile a model | install stack, build a context |
| Speed | millions of draws/s (columnar VM + JIT) | very fast (vectorized C) | fast, but compile + warmup | slower, Python-bound |
| Visualization | **built into the language** (`plot::`, inline) | matplotlib (separate) | bayesplot / external | arviz (separate) |
| Best at | learning, quick estimates, uncertainty propagation, demos | general array compute | real posterior inference at scale | the same, friendlier |
| Runs in the browser | **yes** (WASM, no install) | no | no | no |

The honest summary: **Stan and PyMC beat Noise at the thing they're built for** — fitting a
posterior to lots of continuous data. **NumPy beats Noise at raw array crunching.** Noise beats both
at the distance between *"I have a probability question"* and *"I see the answer and the shape of the
distribution."*

---

## Side by side

Each task below is the *same* problem in Noise and in the mainstream tool. Watch the line count, but
more importantly watch how much of each program is *the idea* versus *the plumbing*.

### 1. Estimate π by Monte Carlo

**Noise** — the probability *is* the program:
```
X ~ rand::unif(-1, 1)
Y ~ rand::unif(-1, 1)
4 * P(X^2 + Y^2 < 1)        # ≈ 3.14159
```

**NumPy** — you allocate the draws, pick `n`, and reduce by hand:
```python
import numpy as np
rng = np.random.default_rng(0)
n = 1_000_000
x = rng.uniform(-1, 1, n)
y = rng.uniform(-1, 1, n)
pi = 4 * np.mean(x**2 + y**2 < 1)
```

Same idea, but in NumPy *you* manage the sample count, the arrays, and the reduction. In Noise `P`
is the only thing that draws, and it self-reports its precision: `4 * P(C)` prints `3.141` (one fewer
digit than `P(C)`) because the error propagated through the `4 *`. Nobody told it to round.

### 2. A fair die — and why `Dice + Dice` is a trap

**Noise** — names are mathematical variables, so independence is *explicit*:
```
A ~ unif_int(1, 6)
B ~ unif_int(1, 6)
P(A == 4 && B == 4)     # ≈ 1/36

D ~ unif_int(1, 6)
P(D + D == 12)          # ≈ 1/6  —  D + D is 2·D, ONE die, not two!
```

`D + D` is `2·D` because `D` is one fixed draw every mention reuses — exactly like `x` in algebra.
This is *the* rule that catches beginners, and Noise makes it a visible law: independence comes only
from a second `~`. NumPy has the same sharing (`d + d` is `2*d` on one array) but never *names* the
rule — you just have to remember to call `rng.integers` twice. Noise turns a silent footgun into a
teachable invariant.

### 3. The birthday paradox

**Noise** — the shaped draw `~[n]` gives 23 iid birthdays; the predicate is one builtin:
```
use rand; use vec;
days ~[23] unif_int(1, 365)
P(has_duplicates(days))     # ≈ 0.507
```

**NumPy** — vectorizing "any collision among 23" across many trials is real work:
```python
trials = 100_000
days = rng.integers(1, 366, size=(trials, 23))
days.sort(axis=1)
collision = (np.diff(days, axis=1) == 0).any(axis=1)
p = collision.mean()        # ≈ 0.507
```

The NumPy version is faster per draw, but the *probability question* is buried under a 2-D array, a
sort, a diff, and two axis arguments. The Noise version is the sentence "what's the chance two of 23
birthdays collide?" written down.

### 4. Learning a coin's bias — a real Bayesian update

This is the case people reach for Stan/PyMC for. Noise expresses the *model* just as compactly; it
just infers differently (see the caveat after).

**Noise** — prior, likelihood, and posterior in three lines:
```
bias  ~ unif(0, 1)                 # flat prior over the bias
flips ~[10] bernoulli(bias)        # 10 flips at that (random) bias
E(bias | count(flips) == 7)        # ≈ 0.667 — posterior mean after 7 heads (= 8/12)
```

**Stan** — the model is declarative too, but it lives in blocks, compiles, and needs a host program:
```stan
data { int<lower=0> n; int<lower=0> heads; }
parameters { real<lower=0,upper=1> bias; }
model {
  bias  ~ uniform(0, 1);
  heads ~ binomial(n, bias);
}
```
```python
# ...plus a Python/R driver to pass data={'n':10,'heads':7}, compile, sample, and summarize.
```

**PyMC** — friendlier, but still a context, an `observed=`, a sampler, and an arviz summary:
```python
import pymc as pm, arviz as az
with pm.Model():
    bias = pm.Uniform("bias", 0, 1)
    pm.Binomial("heads", n=10, p=bias, observed=7)
    idata = pm.sample()
az.summary(idata)["mean"]["bias"]   # ≈ 0.667
```

All three encode the *same* graphical model — Noise borrows the `~` from Stan/BUGS on purpose. The
difference is everything *around* the model: Noise has no `data` block, no compile step, no sampler
to configure, no `observed=` keyword, no separate summary call. You condition with the `|` bar the
way you'd write it on paper, and you can ask the *predictive* in the same breath:
```
next ~ bernoulli(bias)
P(next | count(flips) == 7)    # ≈ 0.667 — the next flip, given the data
```

### 5. Propagating uncertainty through a formula (risk)

**Noise** — a lifted `if` makes the payout a random variable; `P`/`E`/`Q` read off any summary:
```
use rand;
loss  ~ unif(0, 1000)
claim = if loss > 200 { loss - 200 } else { 0 }   # deductible, per-lane select
Print("P(insurer pays) =", P(claim > 0))          # 0.8
Print("expected payout =", E(claim))
Print("95th-pct payout =", Q(claim, 0.95))
```

This is Noise's sweet spot — *uncertainty propagation*. The `if` isn't control flow; it's a
per-sample select that turns a piecewise formula into a distribution you can then poke at with any
query. In NumPy you'd `np.where`; in Stan this isn't even the kind of question the tool is shaped
for.

### 6. Seeing the distribution, not just a number

Every tool can *compute* a posterior mean. The learning question is "what does the distribution
*look like*?" — and that's where Noise removes the most friction, because plotting is part of the
language rather than a second library you import and configure.

**Noise** — `plot::` is a builtin, interleaved with your output in source order:
```
bias  ~ unif(0, 1)
flips ~[10] bernoulli(bias)

plot::histogram(bias)                  # the flat prior
plot::histogram(bias | count(flips) == 7)   # the peaked posterior — before/after, side by side
```
That's the whole program. Each `plot::` emits a chart *spec*, so the prior→posterior story is
*visible* the moment you run it: real charts in the browser playground, a one-line summary card in
the CLI. The same surface covers `plot::scatter`, `plot::value` (a point estimate with its error
bar), `plot::line`, and `plot::heatmap` (matrices / correlation), and the chart kind is inferred
from what you hand it.

**PyMC** — the model is compact, but seeing it means a separate library, a sampler, and plot calls:
```python
import pymc as pm, arviz as az, matplotlib.pyplot as plt
with pm.Model():
    bias = pm.Uniform("bias", 0, 1)
    pm.Binomial("heads", n=10, p=bias, observed=7)
    idata = pm.sample()
az.plot_posterior(idata, var_names=["bias"])
plt.show()
```

**NumPy** — you draw, then drive matplotlib by hand:
```python
import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)
bias = rng.uniform(0, 1, 200_000)
heads = rng.binomial(10, bias)
plt.hist(bias[heads == 7], bins=40, density=True)   # rejection-condition on 7 heads, then plot
plt.show()
```

Both mainstream versions need `import matplotlib`, a figure, bins, and a `show()` — visualization is
*bolted on*. In Noise, `describe(X)` / `hist(X)` / `corr(A, B)` / `explain(Y)` and the `plot::`
family are part of the same surface as `P`/`E`/`Var`, so *looking at* a variable is as native as
*querying* it — and you can even inspect a variable **without editing the program** from the
playground's sidecar. For a learner, that closes the loop between "I wrote a model" and "I can see
what it does" with nothing in between.

---

## What Noise does that the others make hard

- **Conditioning reads like Bayes on paper.** `P(D == 6 | D > 3)` — the `|` bar binds looser than
  everything, scopes to the one query, and a never-occurring condition is a *spanned error*, not a
  silent `NaN`. A conditioned value is first-class: `high = D | D > 3; E(high); Q(high, 0.5)`.
- **Estimates carry their error and self-round.** `P` returns an estimate with a standard error and
  prints only the digits that error justifies — `P(D==4, 1000)` → `0.2`, `P(D==4, 1e8)` → `0.1666`.
  The error propagates through arithmetic, so derived numbers round honestly too. Students *see*
  Monte Carlo precision instead of being handed false digits.
- **Introspection is built in.** `describe(X)`, `hist(X)`, `corr(A, B)`, `explain(Y)` (rank the
  upstream drivers of a variable's variance) — and a `plot::` surface (`plot::histogram`,
  `scatter`, `value`, `heatmap`) so a program can *show* the distribution, not just collapse it to a
  scalar. You can even inspect a variable **without editing the code**, from the playground sidecar.
- **It runs in a browser tab.** The whole engine compiles to WASM. No install, no toolchain, no
  compile-and-wait — paste a program into the playground and it samples millions of lanes live.
- **It's genuinely fast for what it is.** Programs lower to a sample-DAG, compile to columnar
  bytecode over register columns (with common-subexpression sharing), and JIT with inlined
  transcendentals — millions of draws per second, deterministic under a fixed seed.

---

## Where Noise loses (read this before betting on it)

Being honest is the point — these are *deliberate* scope boundaries, not bugs:

- **Inference doesn't scale to lots of continuous data.** Conditioning is **rejection-based**: it
  keeps the Monte Carlo lanes where the condition happened. Wonderful for a handful of discrete
  observations (7 of 10 heads); useless for "fit these 10,000 continuous measurements." That's
  importance weighting / **HMC-NUTS** territory — exactly what **Stan** and **PyMC** are *for*. If
  your problem is a real posterior over many parameters with real data, use them.
- **No dynamic / stateful systems.** A lifted `if` samples independent lanes; it cannot carry state
  across a time step. Markov chains, queues, random walks, and SDEs need sequential sampling —
  a planned track, not a current capability. (NumPy + a loop, or a dedicated simulator, today.)
- **Not a general array language.** Arrays are fixed-length, build-time, and a `d×d` mat·vec builds
  `O(d²)` graph nodes (fine to ~64). For heavy linear algebra, FFTs at scale, or tensor workloads,
  reach for **NumPy** / **JAX**.
- **`==` on a continuous RV is almost surely false**, there's no exponent literal yet, and the value
  types aren't fully unified. Small language, sharp edges — see `LANG.md` for the live status notes.

The neighbors: **WebPPL** and **Turing.jl** / **Pyro** are the closest *universal* PPLs — more
expressive (arbitrary stochastic control flow, real inference algorithms) but heavier and less
immediate. Noise trades that generality for a tiny, transparent, browser-native core.

---

## Positioning: Noise is a *learning* language first

Most probability tools are built to ship an answer to an expert. Noise is built to *teach the
model*. The design choices all point the same way:

- **The notation is the math.** `X ~ unif(0,1)`, `P(A | C)`, `E(D | D > 3)` — a student who can read
  a probability textbook can read a Noise program, and vice-versa. There's no API to learn between
  the idea and the code.
- **The `~` / `=` split teaches the graphical-model distinction** (stochastic node vs deterministic
  transform) that Stan/BUGS encode — but for *every* value, with no surrounding ceremony to obscure
  it.
- **Referential transparency makes the #1 beginner mistake un-makeable-by-accident.** "Two mentions
  of a name are the same draw" is a law you can see, so `Dice + Dice` ≠ two dice becomes a lesson
  instead of a silent bug.
- **You see the distribution, not just a number.** `describe`/`hist`/`plot::` turn an abstract RV
  into a picture; `explain` shows *which inputs drive the output*. Probability stops being a single
  scalar you have to trust.
- **Errors are honest and self-rounding**, so the very first thing a learner internalizes is that a
  Monte Carlo answer is an *estimate with a precision* — not an exact truth.
- **Zero friction to play.** It's in the browser. A class can open the playground, change `n` in the
  birthday example from 5 to 23 to 50, and *watch the paradox appear* — no install, no notebook
  server, no `pip`.

**Use Noise to understand a probability problem and to estimate or propagate uncertainty quickly.
Graduate to Stan/PyMC when you need to fit a real posterior to real data, and to NumPy/JAX when you
need raw array horsepower.** Noise is the whiteboard you can run.
