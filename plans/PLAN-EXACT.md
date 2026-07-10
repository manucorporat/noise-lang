# PLAN-EXACT — closed-form and deterministic-numeric operators

An **opportunistic exact layer** over the sample-DAG: where a query can be answered in closed form
or by deterministic numerics, answer it that way; otherwise fall through to Monte Carlo, which
stays the oracle and the universal fallback.

Prior art: Ruckdeschel & Kohl, *General Purpose Convolution Algorithm in S4-Classes by means of
FFT*, J. Statist. Softw. 59(4), 1–25 (2014) — the R package `distr`. arXiv:1006.0764.

---

## 0. Why Noise can do this more soundly than `distr`

`distr` overloads `+` on *distribution objects* and (§2.5) "identif[ies] distributions with
corresponding (independent) random variables". So `X + X` convolves two independent copies:
for `X ~ N(0,1)` it returns `N(0, √2)`, not `2X ~ N(0,2)`. The library has no way to know the two
operands are the same draw — a distribution object carries no identity.

Noise's IR does. From `crates/noise-core/src/dist.rs:5`:

> Structural sharing is *required* for correctness: a variable bound to a `Dist` reuses its single
> `RvId`, so `X + X` references one draw of X.

and `simplify.rs:10`:

> Source nodes are copied 1:1, never deduplicated — each `~` draw is a distinct random variable.

So the `RvGraph` already distinguishes *shared randomness* from *independent randomness*. That
distinction is exactly the side condition every convolution identity needs. **Noise can carry an
exact layer soundly; `distr` can only assume its precondition.** That is the interesting part of
this project, and it is the first thing to build.

## 1. The soundness gate: source sets (`exact/srcs.rs`)

For each `RvId`, the set of `Src` nodes in its cone, as a bitset.

`RvGraph` is an append-only arena (`dist.rs:190`) so children always have lower ids than parents —
one forward sweep over `0..len` computes every source set. O(N · S/64).

```rust
pub struct SourceSets { bits: Vec<u64>, words: usize }   // row per RvId
impl SourceSets {
    pub fn of(graph: &RvGraph) -> Self;
    pub fn independent(&self, a: RvId, b: RvId) -> bool;  // srcs(a) ∩ srcs(b) = ∅
}
```

`independent(a, b)` is the precondition for every convolution rule below. It is also useful on its
own, *before any of the rest of this lands*:

- `explain(x)` can name which draws actually drive `x`.
- The `x + x` vs. `~x + ~x` distinction — the single most confusing thing in the language — becomes
  something the tooling can point at.

Conservative and cheap. Ship it first, standalone.

## 2. The representation: `distr`'s four slots, in Rust

`distr` makes a distribution an object with four *constitutive* slots — `r` (sampler), `d`
(density/pmf), `p` (cdf), `q` (quantile) — precisely so that the result of a convolution is again a
first-class distribution (§2.3). Mirror that:

```rust
pub enum Dist {
    /// Closed-form parametric family. Exact. No grid.
    Family(Family),
    /// Lebesgue-decomposed numeric: atoms + an absolutely-continuous lattice part.
    /// Mirrors distr's UnivarLebDecDistribution (§2.4) — needed because e.g. X·Y with an
    /// atom at 0 in Y produces a Dirac atom in the result.
    Lebesgue(LebDec),
}

pub struct LebDec {
    atoms: Vec<(f64, f64)>,   // (location, mass)  ← the discrete part
    ac: Option<Lattice>,      // the a.c. part
    ac_mass: f64,
}
pub struct Lattice { a: f64, h: f64, p: Vec<f64> }   // support a + j·h, j < m, m = 2^q

impl Dist {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn quantile(&self, u: f64) -> f64;
    fn sample_into(&self, rng: &mut Rng, out: &mut [f64]);   // ← see §8
}
```

The driver is a bottom-up abstract interpretation over the DAG:

```rust
pub fn exact(graph: &RvGraph, srcs: &SourceSets, root: RvId) -> Option<Dist>
```

`None` means "no exact route" — the caller samples, exactly as today. Every rule below is a match
arm; every binary arm consults `srcs.independent(a, b)` first.

## 3. Stage 1 — symbolic families (`exact/family.rs`)

Pure parameter arithmetic. No numerics, no grids, no error. This is `distr`'s "dispatch to the
exact method where one exists" (§2.4) — except our dispatch is gated on the independence oracle.

Coverage over Noise's existing `Source` set (`dist.rs:55`):

| rule | precondition |
|---|---|
| `a·X + b` where `X ~ Normal` | — (affine, no independence needed) |
| `Normal + Normal` | independent |
| `Poisson(λ₁) + Poisson(λ₂)` | independent |
| `Bernoulli(p)` sum → `Binomial(n,p)`; `Binomial + Binomial` same `p` | independent |
| `Geometric(p)` sum → `NegBinomial` | independent |
| `Exp(r)` n-fold sum → `Gamma(n, r)`; `Gamma + Gamma` same rate | independent |
| `Normal(0,1)²` → `ChiSq(1)`; `ChiSq + ChiSq` | independent |
| `Uniform(0,1)` → `Beta(1,1)` | — |
| any `ConstNum` | — (already folded by `simplify.rs`) |
| `Select { cond, a, b }` → mixture `p·a + (1−p)·b` | `cond` independent of `a`, `b` |

That `Select` row is a freebie worth noticing: a lifted `if` is a mixture, and `LebDec` is already a
mixture representation.

**The headline case.** `~[n]` produces `n` independent sources, so `sum(~[n] X)` is an n-fold
convolution — `distr`'s `convpow`, and by Proposition 3.3 of the paper it costs *one* FFT (or one
parameter update) regardless of `n`. Today `sum(~[10000] exponential(1))` builds 10 000 source nodes
and samples each across every lane. It is `Gamma(10000, 1)`. Exactly. Instantly.

## 4. Stage 2 — moment propagation (`exact/moments.rs`)

The path you actually shipped in Python. Propagate `(mean, var)` analytically:

- exact through `+`, `−`, scalar `·`
- exact through independent products: `E[XY] = E[X]E[Y]`, `Var[XY]` by the product formula
- gives `E`/`Var` with **zero samples and no standard error** — `builtins.rs:353` already has the
  shape for this (a deterministic value returns `E(5) = 5 ± 0`)
- gives a normal-approximation fallback for `P`/`Q` over long independent sums

Cheap, broad, and honest as long as the CLT approximation is *labelled* as one (see §7).

## 5. Stage 3 — the FFT fallback (`exact/lattice.rs`, `exact/fft.rs`)

The paper's Algorithm 3.4, verbatim, for sums with no closed form:

1. **Truncate** each operand at its ε-quantiles → `[A, B]`. Needs a `q` slot on each operand; the
   recursion bottoms out at `Family`, which has analytic quantiles. (Unbounded support is the
   normal case, so this step is load-bearing.)
2. **Discretize** onto a shared lattice, `h = (B−A)/m`, `m = 2^q`, `p_j = F([A+jh, A+(j+1)h])`.
3. **Zero-pad** to `2m`. This is what kills the circular-convolution aliasing — not optional.
4. **FFT**, pointwise multiply, inverse. Hand-rolled radix-2 (~120 LOC): `m` is always a power of
   two, and this keeps `rustfft` out of the wasm bundle.
5. **Back-transform** with the `h/2` continuity correction (paper: "improves the accuracy").
6. **Smooth** by linear interpolation → continuous `d` and `p`.
7. **Renormalise** so the result integrates to 1.

Specialisations, all from §2.4 / §3.3:

- discrete ⊛ discrete on a common lattice, finite support → **numerically exact** (paper Tables 1–2:
  total-variation distance ~1e-15 for Binomial/Poisson `convpow`). Steps 2–5 only.
- a.c. ⊛ discrete → direct shift-and-add, no FFT.
- `LebDec` ⊛ `LebDec` → componentwise over the decomposition.

## 6. Stage 4 — the rest of the arithmetic (`distr` §2.5)

Bootstrapped from convolution, and this is where the bugs will live:

- affine transforms: exact on all four slots
- `X − Y = X + (−Y)`
- `X · Y` with positive support: `exp(log X + log Y)`. General support: split each operand's support
  into `(−∞,0)`, `{0}`, `(0,∞)`, treat as a three-way mixture → up to four convolutions plus a Dirac
  atom at 0.
- `X / Y`, `X^Y`: same trick.

Do this **last**. It has the worst effort-to-payoff ratio of anything here, and the support-splitting
is fiddly enough that it needs Stage 6 in place before it is worth attempting.

## 7. Two tiers, and never conflate them

| tier | what | error |
|---|---|---|
| **exact** | family arithmetic (§3); finite-support lattice convolution | f64 round-off, ~1e-15 |
| **deterministic** | FFT on continuous distributions (§5) | ~1e-6 at `q=12`, ~1e-10 at `q=18` |
| **approximate** | CLT / normal approximation (§4) | model error, unbounded |
| **monte carlo** | today | O(1/√N) ≈ 1e-3 at N=1e6 |

The middle two are *deterministic* — no run-to-run variance — and beat MC on accuracy, but they are
not exact, and the FFT path must not print the word "exact". Paper Tables 4 and 6 are the honest
numbers; `q` and `ε` are knobs, not guarantees.

Surface this through the existing `explain` op (`eval.rs:1207`, `introspect.rs:210`):

```
explain(x)
  exact · Normal(μ = -1, σ = 2.2360679…) · via Normal+Normal
explain(y)
  monte carlo · a and b share draw #3 — no independent factorization
explain(z)
  deterministic · FFT lattice, q=14, ε=1e-8 · |truncated mass| < 1e-8
```

**Refuse rather than lie.** If ε-truncation clips more mass than tolerance (heavy tails — Cauchy,
lognormal), do not return a `Dist`. Return `None` and sample. Exponential tilting (paper §3.3) is
the real fix and is out of scope for v1.

## 8. `Source::Tabulated` — making the layer pay off even when it fails

Once `exact()` succeeds on a *subgraph*, `simplify.rs` can replace that whole subgraph with a single
`Src(Source::Tabulated(Rc<Dist>))`, sampled by inverse-CDF from a precomputed table. Anything
downstream that *can't* be solved exactly — a `Gather`, a matmul, a conditioned query — then samples
from a one-node graph instead of a ten-thousand-node one.

This is the fusion `simplify.rs` is already shaped for: it builds a fresh graph post-order, and a
tabulated source is just another `Src`.

**Caveat, and it is a real one.** `simplify.rs:12` preserves the relative order of surviving sources
precisely so RNG consumption is unchanged. A tabulated source consumes a different number of uniforms
per lane than the subgraph it replaced, so seeded outputs change. Either gate this behind a flag or
accept "same distribution, different seed" as the contract. Decide before writing code.

## 9. Where this must not go

Name the boundary explicitly, in the docs, so nobody expects magic:

- **Conditioning** (`Value::Cond`, `value.rs:70`; `sampler::cond_moments`, `sampler.rs:69`).
  Conditioning destroys the independence factorization. Exception: if the whole cone is discrete with
  finite support, exact enumeration + reweighting works. Otherwise → MC.
- **`Gather`** (`dist.rs:185`) — the node that already forces the interpreter. Stays MC.
- **`Rotation`, `Permutation`, `@`, complex** — multivariate. `prisoners.noise` and
  `turboquant.noise` are Monte Carlo forever, and that is fine; it is what MC is *for*.
- **Naive characteristic-function inversion.** The paper (§4.2) points out `Uniform^n` has cf
  `(sin(t/2)/t)^n`, which inverts terribly. Discretize the cdf. Don't get clever.
- **Long heterogeneous chains.** The paper reports reliability "up to 40 (non-)iid summands". Our
  `~[n]` sums route through `convpow` (one FFT), so this only bites on chains of *different*
  distributions.

## 10. Validation harness (`exact/tests.rs`) — build this second, not last

Directly from paper §5. Total-variation and Kolmogorov distance between the exact path and:

1. known closed forms (`Binom(k,p)` convpow → `Binom(nk,p)`; `Pois(λ)` → `Pois(nλ)`;
   `N(μ,σ²)` → `N(nμ, nσ²)`; `Exp(λ)` → `Γ(n,λ)`) — the paper's Tables 1–6, reproduced as
   assertions with the paper's own tolerances;
2. the MC engine, which remains the oracle — the same discipline `approx.rs:12` already applies
   ("The interpreter itself keeps using `libm` — it stays the exact oracle").

A non-central χ² check (paper §5.6, Table 7) makes a good single end-to-end test: three different
FFT decompositions of the same distribution, all agreeing with `pchisq` to 7 digits.

Without this harness none of the above is trustworthy, and the Stage 4 arithmetic is unshippable.

## 11. Module layout and hooks

```
crates/noise-core/src/exact/
  mod.rs       Dist, exact(&RvGraph, &SourceSets, RvId) -> Option<Dist>
  srcs.rs      source-set bitsets, independence oracle          §1
  family.rs    Family enum, closed-form rules, analytic pdf/cdf/quantile   §3
  moments.rs   (mean, var) propagation                          §4
  lattice.rs   LebDec, Lattice, discretize / smooth / renormalize          §5
  fft.rs       radix-2 FFT, convolve                            §5
```

Hooks, all additive:

- `backend::compile_root` (`backend.rs:39`) — compute + cache `SourceSets` and `exact()` once, next
  to the existing `NodeCost`.
- `builtins::{prob, moment, quantile}` (`builtins.rs:295,354,420`) — consult `exact()` before
  sampling. Each already has a "deterministic value is exact" fast path to slot into.
- `introspect::dist1` (`introspect.rs:239`) — add `Dist1::from_dist` beside `from_draws`, so
  `hist(x)` plots the true density instead of a sampled histogram.
- `simplify.rs` — the `Source::Tabulated` fusion of §8.

wasm: the FFT is host-side Rust, not emitted code, so `wasm_emit.rs` and `noise-wasm` need no
changes. Hand-rolling radix-2 keeps the bundle small.

## 12. Sequencing

Ordered by value per line of code, not by dependency depth:

1. **§1 source sets** + **§10 harness skeleton**. Unblocks everything. §1 alone improves `explain`.
2. **§3 families** + query dispatch in `builtins`. This is ~80% of the demo value.
   `normal(0,1) + normal(0,1)` answering exactly and instantly is the whole pitch.
3. **§4 moments/CLT.** Cheap, broad.
4. **§5 FFT + LebDec.** The real engineering. `sum(~[n] X)` via `convpow` is the headline.
5. **§8 `Source::Tabulated`.** Decide the seed-compatibility question first.
6. **§6 products/quotients.** Last. Most bugs, least payoff.

## 13. Open question — transparent or explicit?

Should `P(x > 3)` silently become exact when it can, or should the user write `exact P(x > 3)`?

**Recommendation: transparent, with the decision visible in `explain`,** plus a `--mc` escape hatch
for differential testing. Noise's proposition is "write the model, don't think about the method."
Making the user ask for exactness reintroduces exactly the thing the language removed. The paper
takes the same line (§1): "The user does not have to interfere with the dispatching mechanism."

The tier label in `explain` is what keeps that honest.

## 14. Payoffs, concretely

- **Tail quantiles.** `Q(x, 0.9999)` at `N=1e6` rests on ~100 order statistics. The lattice does not
  care. This is the strongest single argument, and it is why `distr` found users in actuarial risk.
- **Plots.** `hist(x)` becomes a true density curve — smooth, instant, no re-jitter on every
  keystroke in the playground.
- **Speed.** `sum(~[10000] exponential(1))`: 10 000 sources × N lanes → one parameter update.
- **No standard error.** `E`/`Var` return a number, not an `Est`.
- **Pedagogy.** `explain` telling you *why* your model did or didn't factor is a teaching tool no
  other probabilistic language has, and it falls straight out of §1.
