# Step 4 — Collections, loops, and the in-Noise library (detailed plan for review)

This is the detailed sub-plan for PLAN.md's Step 4. It is the bridge from "scalar Monte Carlo"
to the general, expressive language: the **birthday problem for 23 people in one line**, a
**generalized CLT**, and the d-dimensional **TurboQuant** experiment (see `TURBOQUANT.md`).

> **Status: proposal — not yet built.** Review and adjust before implementation.

---

## Orientation (read this first if you're a fresh session)

**Read in this order:** `AGENT.md` (current state of the code), `LANG.md` (the language spec — the
contract; update it as you add surface), this doc, then `TURBOQUANT.md` (the *why* — the d-dim
experiment Steps 4–5 build toward). `PLAN.md` has the overall roadmap; this is its Step 4.

**Crate / module map** (`crates/noise-core/src/`):
- `lexer.rs` — `tokenize() -> Vec<Token>`; `TokKind` enum (add `[` `]`, `for`, `in`). Has a
  `#[cfg(test)] mod tests`.
- `parser.rs` — hand-written Pratt parser. `parse_primary` handles literals/idents/`(`-groups/
  blocks/`if`; an `Ident` immediately followed by `(` is parsed as a `Call` *inline* (and `f(x)=`
  / `f()~` as a `FnDef`, disambiguated by `matching_paren_after`). `infix_op` is the precedence
  table. **There is no general postfix layer yet** — you'll add one for `[index]` (see §3). Tests
  in-module.
- `ast.rs` — `Expr` enum (`Number`, `Str`, `Ident`, `Unary`, `Binary`, `Bind`, `FnDef`, `Call`,
  `Block`, `If`), `BinOp`, `UnOp`, `BindKind` (`Assign` = `=`, `Sample` = `~`), `Spanned`.
- `value.rs` — `Value` enum: `Num(f64)`, `Bool(bool)`, `Str(String)`, `Unit`, `Recipe(Recipe)`,
  `Dist(RvId)`, `Est { val, se }`. Add `Array(Rc<Vec<Value>>)`. `type_name()` + `Display` here.
- `dist.rs` — `RvId`, `RvGraph` (append-only arena; `push(node, kind) -> RvId`, `node`, `kind`,
  `len`), `RvKind {Num,Bool}`, `RvNode {Src, ConstNum, ConstBool, Unary, Binary, Select}`,
  `Source {Uniform, UniformInt, Normal}`, `Recipe {Uniform, UniformInt, Bernoulli, Normal}`.
- `eval.rs` — the `Engine { vars: HashMap<String,Value>, funcs: HashMap<String,Rc<UserFn>>, graph:
  RvGraph, call_depth }`. Key methods you'll reuse: `eval(&Spanned) -> Result<Value>`,
  `lift_binary(op, l, r, span)`, `lift_unary`, `operand_to_rv(v, span) -> (RvId, RvKind)`,
  `draw(&mut self, Recipe) -> Value` (the only place sources are created), `call_user_fn`. Free
  helpers: `forbid_recipe`, `math_const`, `eval_binary`/`eval_unary` (deterministic fold).
- `builtins.rs` — `pub fn call(name, arg_vals: &[Value], graph: &RvGraph, span) -> Result<Value>`.
  **Note: it takes `&RvGraph` (immutable) — it cannot build/draw nodes.** Scalar/pure builtins
  live here (`unif`, `unif_int`, `bernoulli`, `normal`, `sqrt`, `round`, `P`, `E`, `var`, `print`).
- `bytecode.rs` — `Inst` enum, `compile(graph, root) -> Program`, `run_batch`, `apply_bin`/
  `apply_un`. (Only touched by the optional §3.5 fused `Reduce`.)
- `sampler.rs` — `moments`, `sample_n`, `for_each_batch`. `rng.rs` — `Rng` (xoshiro256++) with
  `fill_uniform`/`fill_uniform_int`/`fill_normal`.

**How `Expr::Call` dispatches today** (eval.rs): evaluate args → if the name is in `self.funcs`,
`call_user_fn`; else `builtins::call(name, &args, &self.graph, span)`. **Routing for the new
library:** the reducers/constructors *build graph nodes and/or draw*, so they need `&mut self` —
intercept their names in the `Expr::Call` arm (a `match name { "sum"|"dot"|... => self.lib_*(...) }`
*before* the `builtins::call` fallback), implemented as `Engine` methods in `eval.rs`. Keep the
*pure* ones (`range`, `push`, `len` — they only move `Value`s, no graph) in `builtins.rs`.

**Reuse, don't duplicate, the fold logic.** To combine two element `Value`s with a `BinOp`, factor
the body of the existing `Expr::Binary` arm into a helper like
`Engine::binop(op, l, r, span) -> Result<Value>` (it already does `forbid_recipe` then
`lift_binary` if either side is a `Dist`, else `eval_binary`). Then `sum` folds with `Add`, `any`
with `Or`, `dot` with `Add` over per-index `Mul`, `max` via the lifted-`if`/`Select` path, etc. —
so RV and constant elements are handled identically and stay correct.

**Commands:** `cargo test -p noise-core` (unit tests; baseline 93 green), `cargo clippy
--all-targets` (must stay clean — watch `approx_constant`: avoid literals near π), run an example
with `cargo run -q -p noise-cli -- examples/<name>.noise`. Tests go in `lib.rs`'s
`#[cfg(test)] mod tests` (golden/surface) or the relevant module's `mod tests` (unit).

---

## 0. The library ships as builtins — but the language can still express it

**Decision (2026-06-25):** the collections/linear-algebra library (`iid`, `sum`, `count`, `any`,
`all`, `max`, `dot`, `norm`, `matvec`, `has_duplicate`, …) ships as **Rust builtins**, for
*simplicity* and to open a *performance* path — **not** as a `prelude.noise` string. Two reasons,
plus a property we deliberately keep:

- **Simplicity.** No prelude bootstrap (and no prelude parse-error surface); `for` / `push` /
  loop-rebinding come **off the critical path**. The examples need only array literals + indexing
  + `len` + the builtins.
- **Performance — with one honest caveat.** A builtin that just builds the same `a[0]*b[0] + …`
  Add-chain DAG is **no faster at sample time** than the Noise version (sampling cost = DAG size,
  which is identical) — it only saves one-time build-time interpretation. The *real* win is a
  **fused reduction VM instruction** (§3.5): `dot` at `d=64` today builds ~127 instructions and
  ~127 column registers (~1 MB of intermediates *per dot*; matvec → 64 MB+); a fused `Reduce`
  collapses that to **one** register and one vectorizable pass. Builtins are exactly what make
  that lowering easy to add. So: builtins now (simple, correct), fused `Reduce` as a fast-follow
  (the actual speed/memory win, and what unlocks larger `d`).
- **Property we keep: the library *is* expressible in Noise.** Because recipes are values (Step 1)
  and we have functions + value-blocks + rebinding `=` (Step 2), each of these *can* be written in
  Noise — e.g. `sum(xs) = { acc = 0; for x in xs { acc = acc + x }; acc }`. We keep arrays +
  `for` + `push` in the language as the expressiveness backstop, and we **prove the equivalence
  with a test** (a Noise-written `dot`/`sum` must match the builtin). So "small core, library
  in-language" stays *true*; we just ship the fast/simple path as builtins.

Implementation note: the reducers live in `eval` (Engine methods), not `builtins.rs`, so they
reuse the existing `lift_binary` / `operand_to_rv` node-building instead of re-implementing it.

---

## 1. Scope

**In scope (lands in Step 4):**
- A new `Value::Array` (fixed length, known at build time).
- Array literals `[a, b, c]`, empty `[]`; indexing `xs[i]` (and chained `M[i][j]`); `len(xs)`.
- The **library as builtins**: `iid`, `iidmat`, `range`, `sum`, `count`, `any`, `all`, `max`,
  `min`, `mean`, `dot`, `normsq`, `norm`, `scale`, `vadd`, `vsub`, `vsign`, `matvec`, `normalize`,
  `has_duplicate` (implemented in `eval`, reusing the lifting machinery).
- A `for x in xs { … }` loop + `push(xs, v)` that **unroll at build time** (each `~` in the body
  is a fresh node → independence) — kept as the expressiveness backstop, *not* load-bearing for
  the stdlib, and used by the equivalence tests.
- Migrating `birthday`, `clt_normal`, and several other examples to the new style.

**Explicitly deferred (NOT in Step 4):**
- **First-class functions / lambdas.** No `map(v, f)` with a function argument. Element-wise ops
  (`scale`, `vsign`, …) are written as explicit loops instead. (A `Value::Fn` is a later phase.)
- **Random-length loops / random indexing.** Loop bounds and indices must be build-time
  constants. A *random* count or a `xs[randomIndex]` gather is the dynamics fork (Phase 3.5).
- **Native vector-column representation.** Each array element is still its own scalar DAG node,
  so a `d×d` matvec builds `O(d²)` nodes. Fine for `d ≤ ~64`; the `[BATCH × d]` register upgrade
  for paper-scale `d` is a Phase-4 performance item (see `TURBOQUANT.md` §6).

---

## 2. Design decisions (please review each)

1. **Arrays are build-time, fixed-length.** `Value::Array(Rc<Vec<Value>>)`. Length is known when
   the graph is built (everything except a `~` draw is deterministic at build time). Elements are
   arbitrary `Value`s — `Num`, `Dist`, `Bool`, or nested `Array` (matrices = arrays of arrays).
   A vector of random variables is just an array of `Dist`.

2. **`for` unrolls at build time, body leaks scope.** `for x in xs { body }` evaluates `xs` to a
   concrete array, then for each element binds `x` and evaluates `body` in the *current* frame
   (block bindings already leak in Noise). That leak is what makes accumulators work
   (`acc = acc + x` persists across iterations). The loop evaluates to `unit`. Running the body
   `len(xs)` times is what unrolls the graph — each `~` inside is a distinct node, giving
   independence for free.

3. **Indices must resolve to a constant integer.** `xs[i]` requires `i` to evaluate to a
   non-random integer in range (it will, since loop variables from `range` are concrete numbers).
   A non-integer, out-of-bounds, or **random (`Dist`)** index is a spanned error
   ("array index must be a constant integer, not a random variable"). Random gather = dynamics
   fork.

4. **`iid` is a builtin** (loops `n` times calling `draw_recipe`). Because recipes are values
   (Step 1) it *could* be Noise — `iid(d, n) = { out=[]; for i in range(0,n) { x ~ d; out=push(out,x) }; out }`
   — and the equivalence test keeps that honest; but the shipped path is the builtin.

5. **`range(a, b)` is half-open** `[a, b)` (Go/Python convention): `range(0, n)` has `n` elements
   `0 … n-1`. Returns an array of numbers.

6. **`push` is functional** (returns a new array). `push([], v) → [v]`. `O(n²)` copies across a
   build loop — acceptable at the small `n` we unroll.

7. **No new VM node (yet).** A reducer like `sum` over RV elements lowers to an unrolled `Add`
   chain of existing `RvId`s — the columnar VM and CSE already handle it. `for` is purely an
   eval-time (graph-build-time) construct; it never reaches the bytecode. (A *fused* `Reduce`
   instruction is the fast-follow perf pass — §3.5.)

8. **Reducers compose with existing lifting.** `dup = dup || (xs[i] == xs[j])` lifts the moment
   an operand is a `Dist`; `count` sums `if x { 1 } else { 0 }` over bool-RVs; etc. No new
   operator semantics needed.

---

## 3. Surface changes (concrete)

| Layer | Addition |
|---|---|
| **lexer** | tokens `[` `]` (`LBracket`/`RBracket`); keywords `for`, `in` |
| **ast** | `Expr::Array(Vec<Spanned>)`, `Expr::Index(Box<Spanned>, Box<Spanned>)`, `Expr::For { var: String, iter: Box<Spanned>, body: Box<Spanned> }` |
| **parser** | array literal `[a, b, …]` in `parse_primary`; **add a postfix layer** between prefix and primary that loops `[expr]` after any primary (repeatable → `M[i][j]`; binds tighter than operators, like a call). Fold the existing inline call-arg handling into this postfix layer too, so `f(x)[i]` works. `for IDENT in EXPR BLOCK` parsed in `parse_primary` (like `if`). |
| **value** | `Value::Array(Rc<Vec<Value>>)`; `type_name` → `"array"`; `Display` → `[a, b, c]` (comma-joined elements) |
| **eval** | new arms: `Array` (eval each element), `Index` (eval array + index; error on non-array, non-integer, out-of-bounds, or `Dist` index), `For` (eval iter to a concrete `Array`, then for each element bind the loop var in the current frame and eval the body block — bindings leak, which is how accumulators work; loop returns `Unit`). The **node-building library lives here as `Engine` methods** (`iid`, `iidmat`, `sum`, `count`, `any`, `all`, `max`, `min`, `mean`, `dot`, `normsq`, `norm`, `scale`, `vadd`, `vsub`, `vsign`, `matvec`, `normalize`, `has_duplicate`) dispatched from the `Expr::Call` arm, folding via the extracted `Engine::binop` helper. Extract `draw` into a free `draw_recipe(&mut RvGraph, Recipe) -> Value` so `~` and `iid` share one draw path. |
| **builtins** | the **pure** new ones only: `range(a,b)` → `Array` of `Num`s, `push(xs,v)` → new `Array`, `len(xs)` → `Num`. (No graph access needed.) |
| **engine** | no prelude bootstrap — the library is builtins/methods |

No changes to `bytecode`, `sampler`, or `rng` for correctness — Step 4 is front-end + eval. The
optional **§3.5 fused `Reduce`** touches `bytecode`/`sampler` and is a separate perf pass.

---

## 3.5. Fast-follow perf: a fused `Reduce` instruction (optional, separate pass)

The builtins above build correct-but-wide DAGs (a `dot` is an `Add`-chain of `Mul`s). Sample-time
cost and memory scale with that node count, which is the `O(d²)` blow-up TURBOQUANT.md §6 flags.
The fix, addable later without any language change:

- New `Inst::Reduce { dst, op, srcs: Box<[Reg]> }` (and/or a fused `Dot`) that folds many source
  columns into **one** `dst` column in a single pass — one register instead of `n-1` intermediates,
  and a tight vectorizable loop.
- `sum` / `count` / `any` / `all` / `dot` lower to it directly (the Engine methods already know the
  full operand list at build time, so emitting one `Reduce` instead of a chain is local).
- Pure optimization: identical results, far less memory and instruction dispatch — the practical
  unlock for larger `d`. Sequence it *after* the builtins are correct and golden-tested.

---

## 4. Reference Noise implementations (for the equivalence tests, not the shipped path)

The library ships as builtins (§0), but each is *expressible in Noise* — these are the reference
forms used by the equivalence tests (`a Noise dot must equal the builtin dot`), and the proof that
the language stays expressive enough to grow its own library. Every line is valid Noise given the
§3 array core + `for`/`push`.

```noise
# --- construction (range / push / len are builtins; everything here is Noise) ---
iid(d, n)       = { out = []; for i in range(0, n) { x ~ d; out = push(out, x) }; out };
iidmat(d, n, m) = { out = []; for i in range(0, n) { out = push(out, iid(d, m)) }; out };
zeros(n)        = { out = []; for i in range(0, n) { out = push(out, 0) }; out };
ones(n)         = { out = []; for i in range(0, n) { out = push(out, 1) }; out };
iota(n)         = range(0, n);

# --- reducers ---
sum(xs)   = { acc = 0;     for x in xs { acc = acc + x };                acc };
any(xs)   = { acc = false; for x in xs { acc = acc || x };               acc };
all(xs)   = { acc = true;  for x in xs { acc = acc && x };               acc };
count(xs) = { acc = 0;     for x in xs { acc = acc + (if x { 1 } else { 0 }) }; acc };
max(xs)   = { m = xs[0];   for x in xs { m = if x > m { x } else { m } };    m };
min(xs)   = { m = xs[0];   for x in xs { m = if x < m { x } else { m } };    m };
mean(xs)  = sum(xs) / len(xs);

# --- linear algebra (no higher-order funcs; explicit loops) ---
dot(a, b)    = { acc = 0; for i in range(0, len(a)) { acc = acc + a[i] * b[i] }; acc };
normsq(a)    = dot(a, a);
norm(a)      = sqrt(normsq(a));
scale(v, c)  = { out = []; for x in v { out = push(out, c * x) }; out };
vadd(a, b)   = { out = []; for i in range(0, len(a)) { out = push(out, a[i] + b[i]) }; out };
vsub(a, b)   = { out = []; for i in range(0, len(a)) { out = push(out, a[i] - b[i]) }; out };
vsign(v)     = { out = []; for x in v { out = push(out, if x > 0 { 1 } else { -1 }) }; out };
normalize(v) = scale(v, 1 / norm(v));
matvec(M, v) = { out = []; for i in range(0, len(M)) { out = push(out, dot(M[i], v)) }; out };

# --- problem helpers ---
has_duplicate(xs) = {
  dup = false;
  for i in range(0, len(xs)) {
    for j in range(i + 1, len(xs)) {
      dup = dup || (xs[i] == xs[j])
    }
  };
  dup
};
```

`transpose`, `argmin`/nearest-centroid (for general b-bit TurboQuant), and `beta` come later;
they are not needed for the headline examples.

---

## 5. Worked examples in the sub-plan

### 5a. `birthday.noise` — the headline win

**Before (today, hand-unrolled, 5 people, capped at "23 is impractical"):**
```noise
b1 ~ unif_int(1,365); b2 ~ unif_int(1,365); b3 ~ unif_int(1,365);
b4 ~ unif_int(1,365); b5 ~ unif_int(1,365);
p = P( b1==b2 || b1==b3 || b1==b4 || b1==b5 || b2==b3 || ... );   # 10 terms; 23 → 253
```

**After (general N — the canonical motivating program from LANG.md §3):**
```noise
n    = 23;
days = iid(unif_int(1, 365), n);
print("P(shared birthday among", n, ") =", P(has_duplicate(days)))   # analytic ≈ 0.5073
```
This is *the* proof that collections matter: 23 people, one line, and `n` is a knob. We will
ship it at `n = 23` (the famous 0.5073) and keep a comment showing it scales.

### 5b. `clt_normal.noise` — generalized CLT, cross-checked against native `normal`

**Before (today, 12 uniforms hand-summed):**
```noise
u1 ~ unif(0,1); ... u12 ~ unif(0,1);
p = P(u1+u2+...+u12 - 6 > 1);
```

**After (CLT for arbitrary n, validated against the real Gaussian from Step 3):**
```noise
n   = 12;
clt = sum(iid(unif(0, 1), n)) - n / 2;     # mean 0, variance n/12 = 1 at n=12 → ~N(0,1)
print("CLT (sum of", n, "uniforms) P(Z>1) ~", P(clt > 1));
Z ~ normal(0, 1);
print("native normal           P(Z>1) =", P(Z > 1));   # ≈ 0.1587 — the two should agree
```
Now it *demonstrates* the CLT (sum of n i.i.d. uniforms → normal) **and** validates it against
`normal(0,1)`, turning a hard-coded curiosity into a general statement with a built-in check.

### 5c. Broad migration (the expressiveness payoff, many examples collapse to one line)

| Example | After (sketch) | Analytic |
|---|---|---|
| `dice_sum` | `P(sum(iid(unif_int(1,6), 2)) == 7)` | 1/6 |
| `coin_streak` | `P(all(iid(bernoulli(0.5), 3)))` | 0.125 |
| `exactly_two_heads` | `P(count(iid(bernoulli(0.5), 3)) == 2)` | 0.375 |
| `irwin_hall` | `P(sum(iid(unif(0,1), 3)) > 2)` | 1/6 |
| `reliability` | `P(any(iid(bernoulli(0.9), 3)))` | 0.999 |

We migrate these (keeping their analytic comments and golden values), proving the library on
already-verified problems before trusting it on TurboQuant.

### 5d. Capstone (Step 5, after this lands): `turboquant.noise`

```noise
d = 64;
x = normalize(ones(d));
y = normalize(iota(d));
S      = iidmat(normal(0, 1), d, d);          # fresh Gaussian matrix per sample
q      = vsign(matvec(S, x));                  # b=1 MSE quantizer: sign(Sx)
xhat   = scale(matvec(transpose(S), q), sqrt(2 / (pi * d)));
print("MSE quantizer  E[est]/true =", E(dot(y, xhat) / dot(y, x)));   # ≈ 0.637 = 2/π  (the bias)
```
Needs `transpose` (deferred prelude addition) — the only piece §4 leaves out. The 1-D version
(`examples/qjl_scalar.noise`) already runs today.

---

## 6. Implementation increments (small, reviewable, each green before the next)

1. **Array core.** Lexer `[` `]`; `Expr::Array`/`Index`; `Value::Array`; eval + Display; `len`.
   Tests: literals, nesting, indexing, bounds/integer/random-index errors. *(no loops yet)*
2. **Library builtins.** `iid`/`iidmat`/`range`/`sum`/`count`/`any`/`all`/`max`/`min`/`mean`/`dot`/
   `norm`/`scale`/`vadd`/`vsub`/`vsign`/`matvec`/`normalize`/`has_duplicate` as Engine methods
   (reusing `lift_binary`). Tests: each against its analytic value; `iid` independence
   (`P(a==b)` over two `iid` elements).
3. **`for` loop + `push`** (expressiveness backstop). Lexer `for`/`in`; `Expr::For`; build-time
   unroll with leaking body. Tests: accumulation, nesting, `~`-in-body independence,
   zero-iteration — **plus the equivalence test**: a Noise-written `dot`/`sum` matches the builtin.
4. **Migrate examples** 5a/5b/5c; update `examples/README.md`. Each must hit its existing golden
   value within tolerance.
5. **(perf, optional) fused `Reduce` instruction** (§3.5) — once the builtins are golden.
6. **(Step 5) `transpose` + `turboquant.noise`.** The d-dim bias capstone.

---

## 7. Tests to add (beyond per-increment)

- **Independence:** `days = iid(unif_int(1,6), 2); P(days[0] == days[1]) ≈ 1/6` (two draws, not
  shared) vs. a single draw reused `== 1`.
- **Build-time determinism:** a `for` over `range(0, n)` builds exactly `n`× the body's nodes
  (assert graph size), proving unroll, not runtime branching.
- **Library ⇄ Noise equivalence:** a Noise-written `dot`/`sum`/`has_duplicate` matches the
  builtin on the same inputs — proving the library *could* be in-language (the §0 property).
- **Errors:** index out of bounds, non-integer index, `Dist` index, `for` over a non-array,
  `len` of a non-array, ragged `dot`/`vadd` length mismatch.
- **Golden migrations:** the five §5c examples + birthday(23)=0.507 + CLT agreement with
  `normal`.
- **Adversarial:** `has_duplicate` correctness on a known small case; `dot`/`norm` on constant
  vectors with hand-computed answers.

---

## 8. Risks & open questions (for your review)

- **`O(n²)` node blow-up.** `has_duplicate(23)` = 253 comparison nodes (fine); `matvec` at
  `d=64` ≈ 4k+ nodes (fine); `d=1536` is not (deferred — §1). Confirm `d ≤ 64` is acceptable for
  the TurboQuant *proof*.
- **Loop-var leakage.** After `for i in …`, `i` keeps its last value (blocks leak). Harmless and
  consistent with current scope rules, but worth confirming you're OK with it vs. block-scoping
  the loop var.
- **`min`/`max`/`sum` shadowing.** These are builtins; a user def of the same name still wins
  (user defs already shadow builtins). Fine, but means `sum` isn't reserved.
- **No `map` / first-class functions.** Element-wise library ops are direct builtins; a user who
  wants their own element-wise transform writes a `for` loop (Go-style). General `map` + lambdas
  (which need closures) stays a deliberate later step. Confirm you're OK deferring it.
- **Fused `Reduce` is where the perf actually is** (§3.5), not in "builtin vs Noise." Confirm the
  sequencing: ship builtins first (correct, simple), add the fused instruction as a later pass.

---

## 9. Definition of done

- `birthday.noise` computes `P(shared birthday among 23) ≈ 0.507`, and `clt_normal.noise` shows
  the generalized CLT agreeing with native `normal`.
- The five §5c examples are migrated to one-liners and still hit their golden values.
- The library builtins (`iid`, `sum`, `count`, `any`, `all`, `max`, `min`, `dot`, `norm`,
  `matvec`, `normalize`, `has_duplicate`, …) are covered by tests, **and** the Noise ⇄ builtin
  equivalence test is green (proving the library stays expressible in-language).
- `cargo test`, `cargo clippy --all-targets`, and all examples are green.
- Unblocks Step 5 (`transpose` + `turboquant.noise`), the d-dim bias proof.
