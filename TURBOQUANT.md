# Case study: can Noise prove TurboQuant's bias?

A feasibility analysis of using Noise to **empirically reproduce** the central claims of
*TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*
(Zandieh, Daliri, Hadian, Mirrokni — Google Research / DeepMind, arXiv:2504.19874v1).

The question the maintainer asked: *is Noise expressive enough to demonstrate that the
technique causes the error it claims, and would such a proof be easy to write?* Short answer:
**Noise is the right kind of tool, but not expressive enough today.** The gap is well-defined,
additive, and — importantly — does **not** require the "dynamics fork" (PLAN.md Phase 3.5).

> **Update (2026-06-25): BUILT.** Steps 4–5 closed every gap below. The proof now lives in
> [`examples/turboquant.noise`](examples/turboquant.noise) and runs today. It reproduces both the
> **bias** (`E[est]/true → 2/π ≈ 0.637` for the MSE quantizer vs `→ 1.0` for the QJL fix) **and**
> the **error reduction** (`E[(est−true)²]` is several times smaller for the unbiased quantizer,
> because the MSE quantizer's error is dominated by an irreducible bias floor). It uses the
> Gaussian-projection route (Lemma 4 / QJL identity), so no QR/Beta is needed. The §4 gap table is
> kept below as a record of what was built. Open perf item (not correctness): paper-scale `d` still
> needs the native vector-column representation of §6.

---

## 1. What the paper claims (the empirically-checkable parts)

TurboQuant is a vector quantizer `Q: R^d → {0,1}^{b·d}` (b bits per coordinate) with an inverse
`Q⁻¹`. Two distortions matter, both **expectations over the quantizer's randomness** for a
*fixed worst-case* input:

- **MSE:** `D_mse = E‖x − Q⁻¹(Q(x))‖²`
- **Inner product:** `D_prod = E[(⟨y,x⟩ − ⟨y, Q⁻¹(Q(x))⟩)²]`, plus *unbiasedness*
  `E⟨y, Q⁻¹(Q(x))⟩ = ⟨y,x⟩`.

The mechanism (Algorithm 1, MSE-optimal):
1. Random rotation `Π` (QR of a Gaussian matrix) → `Πx` is uniform on the sphere `S^{d−1}`.
2. **Lemma 1:** each coordinate of a uniform sphere point is Beta, `f_X(x) ∝ (1−x²)^{(d−3)/2}`
   on `[−1,1]`, → `N(0, 1/d)` in high d. Distinct coordinates are near-independent.
3. So quantize each coordinate with an optimal scalar (Lloyd–Max) quantizer: `2^b` centroids
   `c_i`. Quant: `idx_j = argmin_k |（Πx)_j − c_k|`. DeQuant: `x̃ = Πᵀ c[idx]`.

**The headline result the maintainer wants to reproduce — "the technique causes the error":**
MSE-optimal quantizers are **biased for inner products**. At `b=1` the optimal codebook is
`{±√(2/(πd))}`, so `Q_mse(x) = sign(Πx)` and, by the QJL identity (Lemma 4),

> `E⟨y, Q⁻¹_mse(Q_mse(x))⟩ = (2/π)·⟨y,x⟩`  — a multiplicative bias of **2/π ≈ 0.6366**.

The fix (Algorithm 2, `TurboQuant_prod`): run `Q_mse` at `b−1` bits, take the residual
`r = x − Q⁻¹_mse(Q_mse(x))`, apply a **1-bit QJL** on `r` (`sign(S·r)`, S Gaussian), and add it
back scaled by `‖r‖`. **Theorem 2:** the result is *unbiased* (`E⟨y,x̃⟩ = ⟨y,x⟩`) with
`D_prod ≤ √3 π² ‖y‖² /d · 4^{−b}`.

Supporting numeric claims a simulation could check:
- `D_mse ≈ 0.36, 0.117, 0.03, 0.009` for `b = 1,2,3,4` (unit-norm x).
- `D_prod ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d`.
- QJL (Lemma 4) is **exactly** unbiased for any d, with `Var ≤ (π/2d)‖y‖²`.

The cleanest single falsifiable experiment: **`E[⟨y,x̃⟩]/⟨y,x⟩` is ≈ 0.637 for the MSE
quantizer and ≈ 1.0 for the prod quantizer.** That is precisely "the MSE technique causes an
inner-product bias, and the two-stage fix removes it."

---

## 2. Verdict: right tool, wrong reach (today)

Noise is a Monte Carlo language whose native unit of meaning is *the expectation of a function
of random variables* (`P(event)` is already "the mean of an indicator"). TurboQuant's claims
are all expectations/variances over injected randomness for a fixed input — **a perfect fit for
Noise's identity.**

The key architectural finding: **TurboQuant is feed-forward (spatial), not sequential
(temporal).** A single Monte Carlo sample draws a Gaussian matrix and a vector, does
matrix–vector products and reductions over the `d` coordinates, and yields scalars (inner
products) whose distribution we study. There is **no recurrence** — step `t+1` never depends on
step `t`. So, unlike the M/M/1 queue (which genuinely needs PLAN.md's Phase-3.5 second
execution mode), TurboQuant fits the **existing columnar VM**: a length-`d` vector is just `d`
scalar RV nodes, and a dot product is a build-time-unrolled chain of `Add`/`Mul` nodes
collapsing `d` columns into one scalar column. **All of it is data-parallel feed-forward — the
engine we already have can run it.** What's missing is *surface and primitives*, not a new
machine.

But today it is **not expressive enough**: Noise has only scalar RVs (`unif`, `unif_int`,
`bernoulli`), no Gaussian, no vectors/arrays, no reductions, and no language-level expectation
of a *numeric* RV (only `P` over booleans is exposed).

---

## 3. What the proof would look like in Noise (target syntax)

This is the *headline bias* experiment, in the syntax Noise would have after the additions in
§4. ~15 lines, and it reads like the math — that is the "easy to write" bar.

```noise
# Fixed worst-case input x (unit norm) and query y. d small — the 2/π bias is ~d-independent.
d = 64;
x = normalize(ones(d));              # any fixed unit vector
y = normalize(iota(d));              # any fixed query vector

# Quantizer randomness: a d x d Gaussian projection, drawn fresh per Monte Carlo sample.
# (Stands in for the random rotation; the b=1 sign-quantizer bias is a projection identity.)
S ~ iid(normal(0, 1), d, d);

# b=1 MSE quantizer:  q = sign(S x);  dequant = sqrt(2/(pi d)) * Sᵀ q
sign(v) = map(v, fn t = if t > 0 { 1 } else { -1 });   # prelude helper
q    = sign(matvec(S, x));
xhat = scale(matvec(transpose(S), q), sqrt(2 / (pi * d)));

# The claim: the MSE quantizer is biased by 2/π for inner products.
ratio = dot(y, xhat) / dot(y, x);
print("E[est]/true =", E(ratio), "  vs  2/pi =", 2 / pi);   # -> ~0.637, the bias
```

The fix (`TurboQuant_prod`) adds a residual + 1-bit QJL and shows the ratio → 1.0:

```noise
r        = sub(x, xhat_b0);                 # residual from the (b-1)-bit MSE stage
qjl      = sign(matvec(S2, r));             # 1-bit QJL on the residual
xqjl     = scale(matvec(transpose(S2), qjl), sqrt(pi/2) / d * norm(r));
xhat_p   = add(xhat_b0, xqjl);
print("prod E[est]/true =", E(dot(y, xhat_p) / dot(y, x)));  # -> ~1.0, unbiased
```

`dot`, `norm`, `matvec`, `transpose`, `scale`, `add`/`sub`, `map`, `normalize`, `sign`, `iota`,
`ones` are all **prelude** functions written in Noise; only `normal`, vectors, `E`, `pi`, and
`sqrt` are primitives.

---

## 4. Gap analysis — what to build vs. what to write in Noise

| # | Capability | Why TurboQuant needs it | Kind | Effort |
|---|---|---|---|---|
| 1 | **`normal(μ,σ)` distribution** | every matrix/projection is Gaussian; the coordinate law is `N(0,1/d)` | **primitive** (new `Source` + Gaussian RNG, e.g. Box–Muller/ziggurat) | small |
| 2 | **Vectors/arrays** (fixed, build-time length) + literals, indexing, `len`, `range`, `iid(dist,n[,m])`, `for` | `x,y ∈ R^d`, the `d×d` matrices `S,Π` | **primitive** (this *is* PLAN.md Step 4) | medium |
| 3 | **Reduction** (`sum`/fold over an array within one sample) | `⟨y,x⟩ = Σ yᵢxᵢ`, `‖r‖² = Σ rᵢ²` — the cross-coordinate collapse | **primitive-ish**: a `for` accumulator that unrolls into `Add` nodes (no new VM node needed) | small–medium |
| 4 | **Expectation/variance of a numeric RV**: `E(expr)`, `var(expr)` → `Est` | every claim is `E[...]` / `Var[...]`; today only `P` (bool mean) is exposed | **primitive** (surface the existing Rust `moments`) | small |
| 5 | `sqrt`, `pi` | scaling constants `√(2/(πd))`, `√(π/2)/d` | **primitive** (trivial; `sqrt` ≈ `**0.5`) | trivial |
| 6 | Linear-algebra **prelude**: `dot`, `norm`, `matvec`, `transpose`, `scale`, `add/sub`, `map`, `sign`, `normalize`, `argmin`, `iota`, `ones` | the quantizer bodies | **in-Noise prelude** (Go-philosophy: small core, library in-language) once 1–3 exist | medium |
| 7 | `beta(α,β)` and/or QR rotation | *faithful* Algorithm 1 (per-coordinate Beta of Lemma 1; true rotation `Π`) | **primitive** (`beta`) + **prelude** (Gram–Schmidt) | optional |

**Not needed:** the dynamics fork / sequential execution mode (Phase 3.5). TurboQuant is
feed-forward.

---

## 5. Build order (smallest path to the headline proof)

The bias experiment in §3 needs **1, 2, 3, 4, 5, and the §6 prelude** — but *not* 7 (the
Gaussian-projection route to the `2/π` bias avoids QR, and you can skip the Beta check). So:

1. **`normal` + `sqrt`/`pi`** (primitives 1, 5) — independently useful; unblocks the CLT example
   too (replaces the 12-uniform fake in `examples/clt_normal.noise`).
2. **Collections** (primitive 2 = PLAN.md Step 4) — arrays, `iid`, `for`, indexing, `len`.
3. **Build-time `sum`/reduce** (primitive 3) — lower a `for`-accumulator into unrolled `Add`s.
4. **`E(expr)` / `var(expr)`** (primitive 4) — expose `sampler::moments` to the surface.
5. **Linear-algebra prelude** (§6) — `dot`, `norm`, `matvec`, `sign`, … written in Noise.
6. **Write `examples/turboquant.noise`** — reproduce `E[est]/true ≈ 0.637` (MSE) vs `≈ 1.0`
   (prod). This becomes a flagship example and an adversarial test of the whole stack.

This reuses and extends the already-planned Step 4 collections work rather than forking the
roadmap — TurboQuant is a forcing function that says "Step 4 must include a reduce primitive and
a numeric `E()`, and we need a `normal` distribution."

---

## 6. Caveats / honest limits

- **Scale.** At the paper's real `d = 1536`, the naive lowering (one `[f64; BATCH]` column per
  scalar node) makes a `d×d` matvec ~`d²` nodes × 8 KB ≈ gigabytes. A *proof* at `d = 32–256`
  is fine (the `2/π` bias converges fast and is essentially d-independent). Reaching paper scale
  needs a **native vector-column representation** (a register holding `[BATCH × d]`, or batching
  over vectors) — a Phase-4-style VM upgrade, not required to demonstrate the claim.
- **Rotation vs. Gaussian projection.** The faithful Algorithm 1 uses an orthonormal `Π` (QR).
  The headline bias has a Gaussian-only route (QJL identity, Lemma 4), so QR is optional. QR /
  the per-coordinate Beta check (Lemma 1) are the parts that would need `beta` and a Gram–Schmidt
  prelude.
- **What this proves.** A Monte Carlo reproduction is empirical evidence (an estimate with error
  bars), not the paper's analytic proof. That is exactly Noise's value proposition — and exactly
  what the paper's own §4.1 "empirical validation" does. Noise would let you write that
  validation in ~15 readable lines instead of a GPU experiment harness.
