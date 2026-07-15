# PLAN-UNIFORM-INPUTS — inputs as shader uniforms via symbolic scalar values

**Date:** 2026-07-15 · **Status: DRAFT.** Grounded by a full seam map (file:line below).
**Depends on** the landed PLAN-DROP-JIT cost model (`set_prefer_runtime`) — this plan is what makes
that cost model *pay* for input-driven interactivity.

## The thesis

An `input::real(…)` value bakes into the compiled kernel today. `input_value_to_value`
(`eval.rs:810`) returns a plain `Value::Num`, which lifts into the sample-DAG as an `RvNode::ConstNum`
(`draw_lift.rs:567` for operands, `draw_lift.rs:407` for source parameters), and every emitter spells
that constant as a *literal* — WGSL `bitcast<f32>(0x…u)` (`wgsl_emit.rs:532`), WASM `f32.const`
(`wasm_emit.rs:652`). So the shader text — and therefore the pipeline-cache key (`gpu.rs` caches by
shader text) and the compile-cache key (`compile_cache.rs:161` serializes the f64) — **changes when
the input changes**. Every slider drag recompiles.

On the GPU that recompile is *seconds* (turboquant cold-compiles ~2.9 s; noise_colors ~10 shaders).
For an interactive host — the playground slider, a JS caller sweeping inputs — that is the dominant
cost, and it is exactly the case `prefer_runtime` was built to reward but currently cannot, because
each input value is a *different* shader.

**Make value-inputs shader UNIFORMS.** Lower an input as a distinguished graph node whose value is
supplied at dispatch (a uniform read), not baked into the shader. Then the kernel is a pure function
of the program *structure*, stable across input *values* → the pipeline cache hits → **compile once,
dispatch per keystroke.** 15 of 31 corpus examples use inputs; the value ones (`barrier`,
`deductible`, `dither`, `my_stake`, `fm_swing`) are exactly the interactive sliders.

## The one distinction that runs through everything: structural vs value

| kind | examples | lowering | recompile on change? |
|---|---|---|---|
| **structural** | array sizes (`~[n]`), `set_max_samples`, loop bounds (`for i in 0..d`), control-flow `if` on a deterministic value | **force to a concrete scalar at build** — the graph *structure* depends on it | **yes, and correct** (a different `n` is a genuinely different program). Mostly `input::int`. |
| **value** | thresholds (`path < barrier`), source params (`normal(0, k)`), arithmetic operands, signal params | **lower to a uniform-reading `Input` node** — the structure is identical, only the value differs | **no** — the win. Mostly `input::real`. |

An input's kind is decided *by use*, not by declaration: the same input can size an array (structural)
in one place and scale a draw (value) in another. So an input must stay **symbolic** until the context
forces it one way or the other — which is exactly what `Value::Signal` already does for waveforms.

## The model: symbolic scalar values (the `Value::Signal` playbook, scalar)

`Value::Signal(Rc<SigExpr>)` (`signal.rs:91`) is the proven template: a lazy `Rc`-DAG that stays
symbolic through arithmetic (`ops.rs:248` `binop_signal`, `ops.rs:38` unary) and materializes **two
ways** — fold to `f64` when deterministic (`sample_f64`, `signal.rs:145`), or lower to graph nodes
when it has gone dynamic (`materialize_sig_memo`, `ops.rs:454`, `Rc`-identity memoized). The
"has it gone dynamic" predicate is `has_noise` (`signal.rs:119`).

Symbolic inputs are the same shape, one dimension simpler (a scalar, not a per-index window):

```
Value::Sym(Rc<SymExpr>)                       // new Value variant (Value is #[non_exhaustive])
enum SymExpr {
    Input(u32),                               // slot idx — the leaf that must NOT fold
    Const(f64),                               // a promoted scalar (the Konst analog)
    Unary(UnOp, Rc<SymExpr>),
    Binary(BinOp, Rc<SymExpr>, Rc<SymExpr>),  // + transcendentals as the corpus needs
}
```

- `input::real(…)` returns `Value::Sym(Rc::new(SymExpr::Input(idx)))` — `idx` assigned in declaration
  order (today's implicit `input_manifest` position, `eval.rs:180`; add an explicit `idx` to
  `ResolvedInput`, `input.rs:204`). It **also** records the resolved f64 (for structural folding).
- Arithmetic keeps it symbolic, mirroring `binop_signal`: `Sym ∘ Sym`/`Sym ∘ Num` → `Sym`. (`Sym ∘ RV`
  lowers the `Sym` to a node first, then uses the existing RV path.) The scalar-promotion seam is the
  analog of `sig_operand`/`scalar_const` (`eval.rs:1216`/`1184`).
- **Two materializations** (the `SigExpr` two-mode pattern is the whole design):
  - **`force_scalar(&values) -> f64`** — when structure needs a concrete value (array size, sample
    count, loop bound, `if`/index on a deterministic value). Folds the SymExpr with the *current*
    input values. The structure then legitimately depends on the value → structural input, recompiles.
  - **`lower(engine) -> RvId`** — when the Sym enters the RV graph as a *value* (an RV operand via
    `operand_to_rv`/`draw_lift.rs:564`; a source param via `arg_id`/`draw_lift.rs:405`; a signal
    `Konst`; a complex channel; an array element). Emits `RvNode::Input{idx}` leaves + `Binary`/`Unary`
    nodes for the arithmetic — **value never baked.**

## The `Input` graph node

Add `RvNode::Input { idx: u32 }` next to `ConstNum` (`dist.rs:269`), `RvKind::Num`. A numeric leaf
structurally like `ConstNum` but **keyed on the index, never the value.** Then:

- **`simplify.rs` must treat `Input` as opaque** — this is the load-bearing subtlety. `Input` is NOT
  const-foldable (folding it back to a `ConstNum` re-bakes the value). `Input * 2` stays
  `Binary(Mul, Input, Const(2))`. Audit the fold/CSE arms (`simplify.rs`, ~123 `RvNode::` refs);
  `Input` participates in CSE and structural rewrites but never in constant propagation.
- **The cache key keys on `idx`** (`compile_cache.rs:161` currently `push_f64` for `ConstNum`) — a new
  arm `push tag; push_u32(idx)`, no value. Two forcings differing only in input *values* now produce
  the **identical** key → the compiled artifact is identical → pipeline/compile cache **hits**.
- **Backends read the value at runtime, not from the node:**
  - **interp** (`bytecode.rs:524` emits `Inst::ConstNum`): add `Inst::Input { dst, idx }` that fills
    the column with `input_values[idx] as f32`; the `Runner` carries the input-values slice.
  - **WGSL** (`wgsl_emit.rs:532`): emit `P.inputs[idx]` where a const is baked today; same redirect in
    `draw_expr` (`wgsl_emit.rs:675`) for an input-backed source parameter.
  - **WASM** (`wasm_emit.rs:652`): read the input from a params region instead of `f32.const`.

## Runtime plumbing: input values → the kernel

The compiled program records `K` = number of input slots (max idx + 1, small — a handful of sliders).
At forcing time the driver reads the current values (from `input_manifest`/overrides) into `[f32; K]`
and hands them to each backend:

- **interp** — the `Runner` (`backend.rs`) holds the slice; `Inst::Input` reads it. (Structural values
  were already folded at build, so only *value* inputs reach here.)
- **GPU** — extend `struct Params` (`wgsl_emit.rs:722`: today `key, lane0, n`) with `inputs: array<f32,
  K_MAX>` and the `[u32;4]` pack + 16-byte buffer in `dispatch` (`gpu.rs:241,247`). **Uniform alignment:
  a `array<f32>` in a uniform block needs 16-byte stride** — pack four inputs per `vec4<f32>`, or use a
  small fixed `K_MAX` (e.g. 16) padded. Same struct reused by `try_joint` (`gpu.rs:415`).
- **WASM** — the kernel takes 5 i32 params today (`wasm_emit.rs:89`, `OUT/N/K0/K1/LANE0`); add a
  pointer to an input-values region the host writes before calling, threaded through `wasm_host.rs:107`
  (`inst.exports.kernel(…)`). (A memory region beats new params — `K` varies per program.)

## Phases

**P0 — the symbolic model + `Input` node, interpreter only (the foundation).** `Value::Sym` + `SymExpr`
+ arithmetic + `force_scalar`/`lower`; `RvNode::Input`; `simplify` opacity; `bytecode` `Inst::Input` +
the `Runner` input-values slice; the cache key on `idx`. **Gate:** forcing the same program at two
*different* input values compiles **once** (compile-cache hit, provable via `backend::probe`), and every
result is bit-identical to today's baked-const behavior (`Input` reads the same `val as f32`).

**P1 — both emitters + the dispatch plumbing (the win).** WGSL `P.inputs[idx]` + the `gpu.rs` params
buffer; WASM input-region + `wasm_host` threading. **Gate:** `NOISE_PROFILE=1` shows `gpu.pipeline:
cache HIT` across an input change (today it MISSES); a changed slider re-dispatches with **no**
pipeline compile. Measure turboquant/noise_colors: N runs at N input values pay compile **once**, not N
times.

**P2 — the symbolic paths: signals, complex, arrays.** Make an input flowing through `Value::Signal`
(a `Konst` carrying a `SymExpr`, or a new `SigExpr::Input` leaf that lowers via the noise-walk path,
`ops.rs:467`), `Value::Complex` channels, and `Value::Array` elements lower to `Input` nodes rather
than folding. **Gate:** `am_vs_fm`'s `fm_swing` (a signal param) and `noise_strength` become uniforms.

**P3 — the host API + polish.** A `set_input_value(idx, v)` fast path that re-dispatches without
re-eval where the graph is unchanged (the JS/playground slider loop); confirm the pipeline cache
survives across `run` calls on a held engine; the profiler surfaces the HIT. This is where the
interactive story becomes real for the browser.

## Determinism / the two-tier contract

Value inputs are f32 uniforms (the lane type), so the interpreter reads the *same* `val as f32` the
old `ConstNum` lane held → **bit-identical CPU results** to the baked-const behavior; the GPU stays
tier-2 as always. Structural inputs fold to the exact f64 at build (integer sizes/counts), unchanged.
The draw stream is unaffected: `Input` draws nothing (it is a deterministic leaf, like `ConstNum`), so
`kernel::source_ordinals` and the whole draw contract are untouched.

## Risks

| risk | answer |
|---|---|
| **`simplify` folds `Input` back to a constant** (re-baking) | The single most important invariant: `Input` is opaque to constant propagation. A focused audit of `simplify.rs` fold arms + a test that `input * 2` keeps an `Input` node. |
| The structural/value boundary is subtle | It is exactly `SigExpr`'s materialization boundary (fold-vs-lower), which already ships and is tested. Reuse the pattern; the contexts that force-to-scalar are enumerable (array size, sample count, loop bound, `if`/index). |
| `Value` variant ripple | `Value` is `#[non_exhaustive]` with wildcard arms (E2); adding `Sym` is mostly additive, and `Signal` proves the surface is manageable. |
| WGSL uniform alignment / variable `K` | Fixed `K_MAX` padded, or pack into `vec4<f32>`. `K` is tiny (sliders), so a 16-slot padded array is cheap and dodges dynamic-binding complexity. |
| Signal/complex interaction (P2) | Deferred to its own phase; P0/P1 already cover the direct value inputs (`barrier`, `deductible`, `dither`, `my_stake`) — the majority. |
| A structural input that *also* feeds a value use | Works by construction: `force_scalar` at the structural site (bakes the size), `lower` at the value site (uniform). The same input is both a compile-time size and a runtime uniform — no conflict. |

## Success criteria

- **Dragging a value slider does NOT recompile the kernel** — `gpu.pipeline: cache HIT` in the
  profiler across the change; the shipped-CLI cold tax (turboquant 2.9 s) is paid **once** across a
  whole input sweep, not per value.
- Every value input in the corpus (`barrier`, `deductible`, `dither`, `my_stake`, `fm_swing`,
  `noise_strength`) lowers to a uniform; structural `input::int`s still (correctly) recompile.
- Results bit-identical (CPU) / tier-2 (GPU) to today's baked-const behavior — the conformance corpus
  and a new "input value change is invisible to the answer" test both green.
- The interactive loop (`prefer_runtime` on + a held engine + slider changes) runs at dispatch speed,
  not compile speed — the whole point.
