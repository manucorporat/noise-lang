# PLAN-FINANCE — Noise as a quant prototyping tool

**Goal.** Not to tick off three tweet examples, but to make Noise one of the best tools for
quants to *prototype ideas*: idea → distribution → decision in under 20 lines, faster than
opening a notebook. The tweet (exotic options / sizing & ruin / correlation collapse) is
evidence of pull; this plan treats those as probes into the full quant workflow:

```
get data in → model the randomness → simulate → query risk → visualize → iterate
```

Noise's core bet already matches how quants think: every variable IS a distribution,
`P`/`E`/`Q` are first-class, `|` is scenario conditioning. Nothing else in the space
(Python+numpy, Excel, @RISK, Julia) gives you `P(ruin | fed_cuts)` as literal syntax.

---

## Status (2026-07-10)

- **F1 SHIPPED.** `math::exp`/`math::log`/`log10` lift over RVs (new `UnOp::Exp`/`Ln`,
  const-folded in the simplifier; interpreter = `f64::exp`/`ln`). One deviation from the plan:
  on JIT/wasm, **`ln` is the inlined poly** (with the full IEEE domain guard: `x < 0` → NaN,
  `0` → −inf) but **`exp` is a `pow(e, x)` libcall, not an inlined poly** — so `Ln` counts as
  inlined/fusible in the cost model and `Exp` as a libcall. Deterministic scalar
  `log(0)`/`log(-x)` keeps the friendly build-time "needs x > 0" error; only per-lane RV values
  get IEEE semantics. Complex `e^{iθ}` lowering unchanged. `examples/kelly.noise` shipped.
  The `Q`/display-precision audit (G5's paper cuts) remains OPEN.
- **F2 SHIPPED (the distribution half).** `rand::empirical(xs)` and
  `rand::block_bootstrap(xs, b)` — both true Recipes (draw with `~`, iid under `~[n]`, one
  name = one draw); `block_bootstrap` draws an array of `Len(xs)` values from non-wrapping
  contiguous blocks (starts in `[0, n−b]`, last block truncates). Interpreter-only via gather
  (open question 5 stands). **Data loading (`data::csv` / `--data` / playground paste) still
  OPEN**, as are the `std`/`var` reducers. `examples/bootstrap.noise` shipped.
- **F3 SHIPPED (v1).** `vec::cumsum`/`cumprod`/`cummax`/`cummin` + `vec::prod`, implemented as
  **eval-level DAG builders** (unrolled fold nodes) — so the columnar-scan perf win and the
  single-core-cliff investigation remain OPEN. `plot::fan(path)` v1 samples the path jointly in
  one pass and renders q95/q75/q50/q25/q05 as **stacked sparklines** on one shared scale (CLI)
  plus a structured payload (playground, serde tag `"fan"`) — no spaghetti sample paths yet, and
  the www frontend does not render the fan payload yet. `examples/barrier_option.noise` shipped
  (vanilla leg anchored to Black-Scholes 10.4506).
- **Bug discovered along the way (pre-existing, to fix):** `~[n] rand::categorical(w)` shares
  ONE node across the leaves instead of making `n` iid draws. `empirical`/`block_bootstrap`
  deliberately don't inherit it — they are Recipes, so every leaf is a fresh draw.

## 1. Research: what works TODAY (all validated by running programs, 2026-07-10)

The docs oversell one limitation. "No sequential/stateful processes" is only true for
*random-length* processes: **fixed-horizon recurrences work fine** via the for-loop
accumulator idiom (bindings leak out of blocks by design; `st_petersburg.noise` and
`prisoners.noise` already rely on this). That makes most of finance expressible, because
quants almost always simulate a fixed horizon (252 days, 200 trades, 12 months).

| Probe | Result | Wall time (n=1e6) |
|---|---|---|
| Barrier (down-and-out) call, 52-step Euler GBM | vanilla leg **10.5 vs Black-Scholes 10.4506** ✓; KO leg cheaper; `P(knocked)=0.193` | 0.34s (10+ cores) |
| Risk of ruin, 200 asymmetric trades w/ fees, absorption | `P(ruin)=0.263`; `E(bank \| !ruined)` and `Q(bank, .05)` are one-liners | 2.4s |
| Correlation collapse, 3-asset one-factor, ρ 0.3→0.95 | VaR95 −3.5%→−4.7%; **ES/CVaR = `E(X \| X < Q(X,.05))`** works | 0.4s |
| Bootstrap from pasted historical returns | **works today** — random-index gather `data[i]`, `i ~ unif_int(0,n-1)` | 0.02s |
| `sqrt(RV)`, `RV^0.5` | works → Student-t constructible by hand (z/√(χ²/k)) | — |
| 252 steps × 5 correlated assets, running-min drawdown | correct, but **fell to a single core** | **5.3s** |

Validated idioms worth documenting (they are non-obvious but load-bearing):

- **Path**: `s = s0; for t in 0..252 { z ~ normal(0,1); s = s*(1 + mu*dt + vol*z) }`
- **Barrier / ever-crossed**: `knocked = knocked || (s < barrier)` (boolean accumulator)
- **Running max/min (drawdown)**: `worst = if s < worst { s } else { worst }`
- **Absorption (ruin freezes the account)**: `bank = if ruined { bank } else { bank + pnl }`
- **Correlated assets**: shared factor `sqrt(rho)*m + sqrt(1-rho)*e`, or hand `L @ z`
- **VaR/ES**: `var = Q(port, 0.05); es = E(port | port < var)` (Q returns a plain number, so it feeds back into a condition — this is accidental but great)
- **Empirical/bootstrap dist**: literal array + random gather

## 2. Research: the gaps, ranked by pain

**G1 — `exp`/`log` of a random value is unsupported.** `math::exp(RV)` → "the VM has no
exp node"; `math::log(RV)` → type error. This blocks the three most idiomatic finance
constructions: log-space GBM (`logS += drift + vol*z; S = exp(logS)`), lognormal via
`exp(normal)`, and **Kelly log-growth `E(log(1+f*R))`** — the sizing use case's canonical
form. Today you must fall back to Euler discretization (biased) and can't do Kelly at all.
Cheapest fix, biggest unlock: the JIT already inlines ln/exp polynomials internally for the
normal/exponential sources (see inlined-transcendentals work) — they just aren't exposed as
user-facing VM nodes.

**G2 — No way to load data. At all.** No CSV, no file I/O, no playground paste/fetch. Every
input is a literal or a distribution. Quants live on historical series; today they'd have to
paste arrays by hand (which *does* work, see bootstrap probe, but caps at what you'll
tolerate pasting).

**G3 — Paths are hand-rolled and hit a perf cliff.** No `cumsum`/`cumprod`/`prod`/scan; a
252-step path is ~250+ unrolled scalar DAG nodes per asset, and the 252×5 probe dropped to
one core (vs 10+ for the small barrier graph) → 5.3s. Usable, but 10–100× headroom exists
per PERF.md numbers. Also: no random horizon / early stopping — "time to ruin" unbounded is
inexpressible (the PLAN.md §3.5 "dynamics fork").

**G4 — Thin distribution zoo for finance.** Missing: `student_t` (fat tails — THE quant
distribution), `lognormal`, `gamma`/`chi2`, `beta`, `mvnormal(mu, cov)`, copulas, stable.
Constructible by hand where sqrt/sums suffice (t, chi2), impossible where exp is needed
(lognormal, G1). Jump-diffusion is *almost* expressible today (poisson + normal + Euler).

**G5 — Risk vocabulary & precision.** VaR/ES work but are two-step incantations. No
drawdown/Sharpe helpers. Two output paper cuts observed: (a) estimate self-rounding shows
`10.5` vs `10.4` for vanilla-vs-KO — option traders compare in basis points, and the
interesting number (the difference) drowns in display rounding; (b) `Q` prints raw
`0.72186769052` while `E`/`P` self-round — inconsistent. No variance reduction (antithetic,
control variates, QMC) to buy digits.

**G6 — Visualization stops at marginals.** `hist`/`describe`/`corr`/`scatter` are good for
single distributions. `plot::line(path_array)` renders only the **mean path** — no fan chart
(quantile bands over time), no sample-path spaghetti, no drawdown curve. For quants the
*shape of the cone* is the product.

**G7 — Multi-query cost.** Each `P`/`E`/`Q` re-samples the whole graph; a typical finance
script ends with 4–6 queries → 4–6× cost on an already node-heavy graph.

## 3. Plan: what to build, phased

### Phase F1 — Unlock (days): `exp`/`ln` VM nodes + output precision
- Add real `Exp`/`Ln` nodes to the VM + interpreter + JIT (reuse the inlined poly kernels;
  gate by `inline_trans` like the existing transcendentals). Complex `e^{iθ}` lowering stays.
- This alone unlocks: log-space GBM (exact, no Euler bias), `lognormal = exp(normal)`,
  Kelly `E(log(1+f*R))`, log-returns from price ratios, entropy/log-scoring.
- Make `Q` return/print consistently with `E`/`P`; add a way to ask for more digits
  (`round(x, d)` exists — maybe `E(x, n)` guidance is enough, but audit the display story
  for "compare two prices 30bp apart").
- Ship `examples/kelly.noise` the same day — it's the canonical demo this unlocks
  (sweep f with a comprehension, show growth peaks at f* = 2p−1... the full curve).

### Phase F2 — Data in (the "pandas?" answer): arrays from files + empirical distributions
Answer to the question: **no dataframes.** Noise should not grow pandas. A quant's workflow
is: munge in Python/wherever → export a column of numbers → *model in Noise*. What Noise
needs is exactly three things:
1. **`data::csv("returns.csv")` / `data::col(file, name)`** → a literal array at build time
   (CLI: read relative to the .noise file; playground: a paste-data box and/or URL fetch —
   the WASM host passes it in as a named array).
2. **`rand::empirical(xs)`** — sugar for the gather-bootstrap (iid resampling), and
   **`rand::block_bootstrap(xs, block_len)`** for autocorrelated series. This is the killer
   feature: *fit-free prototyping* — resample history instead of assuming a distribution,
   which is precisely what suspicious quants want ("I don't believe your Gaussian").
   Caveat to fix or accept: gather is interpreter-only today.
3. `vec` stats over data arrays already mostly exist (`mean`; add `std`/`var` reducers).
Optional: `noise run file.noise --data rets=spy.csv` binds arrays without touching source.

### Phase F3 — Paths as a first-class fixed-horizon object
Do NOT take the full dynamics fork yet. A **fixed-horizon path** covers ~90% of quant
prototyping and keeps the static-DAG engine:
- `vec::cumsum` / `vec::cumprod` / `vec::cummax` / `vec::cummin` / `vec::prod` over shaped
  draws: `path = cumprod(1 + steps)` where `steps ~[252] normal(mu*dt, vol*sqrt(dt))` —
  collapses today's for-loop into one line AND gives the engine a columnar structure it can
  execute as O(steps) columns instead of an unrolled scalar DAG (fixes the G3 perf cliff
  without a new execution mode).
- Drawdown becomes `min(path / cummax(path)) - 1`. Barrier becomes `any(path < b)`.
  Asian option becomes `mean(path)`. Lookback becomes `max(path)`.
- **`plot::fan(path)`** — quantile bands (5/25/50/75/95) over the index + a few sample
  spaghetti paths. This is the visualization quants screenshot into Slack. Also
  `plot::line` over an array of RVs should at least label that it shows means.
- Investigate the single-core fallback on big graphs (measured: 252×5 probe at 99% CPU)
  — likely a JIT profitability gate or the parallel reduction bailing; may be a quick win
  independent of the columnar work.

### Phase F4 — Correlation machinery
- `rand::mvnormal(mu, cov)` (Cholesky inside, error on non-PSD) and/or `vec::cholesky(M)`
  so power users compose. Factor models stay the teaching idiom; mvnormal is for "here's my
  measured covariance matrix" (which arrives via F2's `data::csv`).
- `rand::student_t(nu)` and multivariate t (fat tails + tail dependence ≈ poor man's
  copula). Full copula machinery only if demand shows up.
- `corr(vec)` heatmap already exists as diagnostic — advertise it in finance examples.

### Phase F5 — The dynamics fork (only after F1–F4 traction)
Random horizon / stopped processes / first-passage *time* as a value (`T = first t: bank<0`,
possibly unbounded). This is PLAN.md §3.5, a real architecture fork (per-lane stepper
execution mode). The freeze-at-absorption idiom covers fixed horizons meanwhile. Revisit
when users ask "expected time to ruin" instead of "P(ruin within N)".

### Cross-cutting: the finance example gallery + website demo
- Promote the validated probes into `examples/finance/` (house style: teenager-friendly
  top comment, named steps, analytic check where one exists):
  `barrier_option.noise` (checks against Black-Scholes ✓), `asian_option.noise`,
  `kelly.noise` (F1), `risk_of_ruin.noise`, `portfolio_var.noise` (correlation collapse,
  ES), `bootstrap_var.noise` (F2), `jump_diffusion.noise`, `vol_targeting.noise`.
- Website: a scroll demo "price an exotic option in 15 lines — no stochastic calculus"
  (noise-demos skill); the fan chart (F3) is the visual anchor.
- A `QUANT.md` (or docs page) that teaches the five idioms from §1 — they exist today and
  nobody can discover them from LANG.md.

## 4. Sequencing rationale

F1 before everything: it's days of work and Kelly/GBM/lognormal are table stakes — every
quant's first three programs hit G1 within minutes. F2 next because "can I use MY data" is
the first question every practitioner asks (and the tweet's author will ask it); empirical +
bootstrap is also the feature that differentiates Noise from "a worse numpy". F3 is the
biggest engineering item but converts the language from "can express paths" to "paths are
pleasant and fast"; it also deliberately absorbs demand that would otherwise force the F5
fork prematurely. F4 rides on F2 (cov matrices come from data). F5 stays parked.

## 5. Open questions

1. `data::csv` at build time vs a host-binding design (`--data` flag / playground box) —
   the second keeps the language pure (no I/O in-language) and works in WASM; leaning host-binding.
2. Should shaped-draw paths (`cumprod`) become the *only* blessed idiom, with the for-loop
   documented as the general fallback? (Probably yes: columnar perf lives there.)
3. Precision story for pricing: bigger default n for `E` on request vs antithetic variates
   vs printing the standard error explicitly. Traders need to see ±.
4. Does `empirical` need weights (for importance-tilted scenarios)?
5. Gather is interpreter-only — is that acceptable for bootstrap-heavy workloads, or does
   F2 force a JIT gather?

## Appendix — validated probe programs

All run on 2026-07-10 against the installed `noise` binary; kept verbatim so they can be
promoted to examples later.

### barrier_option.noise — vanilla matches Black-Scholes 10.4506
```noise
use rand; use math;
s0 = 100; k = 100; barrier = 80;
r = 0.05; sigma = 0.2; t = 1; steps = 52;
# NOTE: math::exp of a random value is unsupported (G1), so log-space GBM is impossible.
# Euler in price space instead:
dt = t / steps;
vol = sigma * sqrt(dt);
s = s0; knocked = false;
for i in 0..steps {
  z ~ normal(0, 1);
  s = s * (1 + r * dt + vol * z);
  knocked = knocked || (s < barrier);
};
vanilla_payoff = if s > k { s - k } else { 0 };
ko_payoff = if knocked { 0 } else { vanilla_payoff };
discount = exp(-r * t);
Print("vanilla call  =", discount * E(vanilla_payoff), " (BS closed form: 10.4506)");
Print("knock-out call =", discount * E(ko_payoff));
Print("P(knocked out) =", P(knocked))
# => vanilla 10.5, KO 10.4, P(knocked) 0.193   [0.34s]
```

### risk_of_ruin.noise — sizing with friction and absorption
```noise
use rand;
start = 10; n_trades = 200;
p_win = 0.38; win_r = 2; loss_r = 1; fee = 0.02;
bank = start; ruined = false;
for i in 0..n_trades {
  win ~ bernoulli(p_win);
  pnl = if win { win_r } else { -loss_r };
  bank = if ruined { bank } else { bank + pnl - fee };   # freeze at absorption
  ruined = ruined || (bank <= 0);
};
Print("P(ruin within " + n_trades + " trades) =", P(ruined));
Print("E(final bankroll | survived) =", E(bank | !ruined));
Print("median final =", Q(bank, 0.5), " 5th pct =", Q(bank, 0.05))
# => P(ruin) 0.263, E(bank|survived) 40.3   [2.4s]
```

### portfolio_var.noise — correlation collapse, VaR + ES
```noise
use rand; use math; use vec;
port(rho) = {
  m ~ rand::normal(0, 1);              # shared market factor
  e ~[3] rand::normal(0, 1);           # idiosyncratic
  z = math::sqrt(rho) * m + math::sqrt(1 - rho) * e;
  rets = [0.02, 0.03, 0.05] * z;       # elementwise vol scaling
  [0.5, 0.3, 0.2] @ rets               # portfolio return
};
calm = port(0.3); crisis = port(0.95);
Print("calm   VaR95 =", round(Q(calm, 0.05), 4), "  VaR99 =", round(Q(calm, 0.01), 4));
Print("crisis VaR95 =", round(Q(crisis, 0.05), 4), "  VaR99 =", round(Q(crisis, 0.01), 4));
var95 = Q(crisis, 0.05);
Print("crisis ES95 (CVaR) =", E(crisis | crisis < var95))
# => calm VaR95 -3.5% / crisis -4.7%, ES95 -5.9%   [0.4s]
```

### bootstrap_probe.noise — empirical distribution via gather (works today)
```noise
use rand; use vec;
data = [0.012, -0.008, 0.003, -0.021, 0.007, 0.015, -0.004, 0.009, -0.013, 0.006,
        0.002, -0.017, 0.011, 0.004, -0.002, 0.008, -0.009, 0.019, -0.006, 0.001];
i ~ unif_int(0, Len(data) - 1);
ret = data[i];                          # random gather = iid bootstrap
Print("E(bootstrap ret) =", E(ret), " (sample mean:", mean(data), ")")
```

### stress_5asset_252.noise — the perf-cliff witness (5.3s, single core)
```noise
use rand; use math; use vec;
steps = 252;
w = [0.3, 0.25, 0.2, 0.15, 0.1];
vols = [0.010, 0.014, 0.018, 0.022, 0.030];
rho = 0.4; a = math::sqrt(rho); b = math::sqrt(1 - rho);
port = 1; worst = 1;
for t in 0..steps {
  m ~ normal(0, 1);
  e ~[5] normal(0, 1);
  rets = vols * (a * m + b * e);
  port = port * (1 + w @ rets);
  worst = if port < worst { port } else { worst };
};
Print("E(terminal) =", E(port));
Print("P(max drawdown > 20%) =", P(worst < 0.8));
Print("VaR95 terminal =", Q(port, 0.05))
# => P(dd>20%) 0.25, VaR95 0.722   [5.3s, 99% CPU — no parallelism]
```

### Failing probes (the G1 witnesses)
```noise
# kelly_probe.noise → runtime error: expected a number, got dist
growth = if win { 1 + f } else { 1 - f };
E(log(growth))                          # log of an RV: unsupported

# original barrier attempt → runtime error: math::exp of a random real value
# is not supported (the VM has no exp node)
s_t = exp(log_s)
```
