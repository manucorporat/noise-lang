# PLAN-DROP-JIT — API / cost-model notes (for the JS host and the browser WebGPU path)

Two related concerns surfaced while landing the GPU: how the backend choice should behave for a host
that re-runs a program (the playground, a JS caller sweeping inputs) rather than running it once.

## 1. The configurable cost model — **implemented**

Every codegen gate weighs two independent things:

- **Runtime:** is the fused kernel faster *per lane*? (cone fatness — `MIN_CONE_OPS`, the joint
  per-root term.) A property of the graph.
- **Amortization:** does a *single* forcing draw enough to refund *compiling* the kernel? (`MIN_WORK_GPU`
  for the GPU, `MIN_DRAWS_WASM` for the wasm emitter.) A **bet about how the artifact is used.**

The amortization bet is only the host's to make. A one-shot `noise file.noise` must earn the compile
back in that run. An **interactive** host reuses the same pipeline across many runs, so the compile is
amortized and codegen wins whenever its *runtime* does — even for a short forcing.

`noise_core::set_prefer_runtime(true)` (or `NOISE_PREFER_RUNTIME=1`) flips the model to prefer runtime:
it **drops the amortization term** in both gates and keeps the runtime terms plus the cold-compile
*feasibility* cap (`MAX_WGSL_INSTRS` — a shader that takes seconds to compile blocks the first run no
matter how it's reused). Verified: a fat cone (102 ops/draw) at n=20k (work 2.0e6 < 3e6) goes from
`DECLINE — work too small` to `ACCEPT (prefer-runtime)`, and returns the correct answer on the GPU.

Wiring: `kernel::{set_prefer_runtime, prefer_runtime}` (process-wide, like the rest of the backend
policy), consulted by `gpu::profitable`/`gate_reason` (native GPU) and `kernel::profitable_roots`
(wasm emitter, the JS-facing gate today). Exposed publicly as `noise_core::set_prefer_runtime` so the
wasm bindings / JS host can set it, and the browser WebGPU backend (G3) mirrors it.

**How a JS host should use it:** set `prefer_runtime(true)` for an interactive/editor session (the
user will re-run as they tweak), leave it off (default) for a batch/one-shot evaluation.

## 2. Inputs change the kernel today — **the prerequisite for #1 to pay on input sweeps**

The headline case for #1 is "the user drags an `input::real(…)` slider and the GPU stays fast because
the compile was paid once." **That does not hold yet.** An `input::real(…)` evaluates to a
`Value::Num` (`eval_arms::input_value_to_value`), which enters the sample-DAG as a `ConstNum` (or a
source parameter — `normal(0, k)`'s sigma). The WGSL emitter **bakes those as literals**, so the shader
text — and therefore the pipeline-cache key (`gpu.rs` caches by shader text) — *changes when the input
changes*. Each distinct input value compiles its own shader: a slider drag recompiles every step.

So `prefer_runtime` amortizes compile across re-runs **of the same shader** (re-running the identical
program, an introspection pass forcing a root several times) — real wins — but **not** yet across an
input sweep, because each value is a different shader.

### The fix (future work): lower inputs as shader **uniforms**, not baked constants

Make an `input::real(…)` a distinguished graph node ("input parameter i") that the emitters lower to a
**uniform** read (`P.inputs[i]`) rather than a literal, and pass the current input values in the params
buffer at dispatch. Then:

- The shader is a pure function of the program *structure*, independent of input *values* → the pipeline
  cache hits across an input sweep → compile is paid **once**, and every slider step is a pure
  dispatch. This is exactly when `prefer_runtime` should be on, and it becomes a large interactive win
  (the noise_colors-class cones — expensive to compile, cheap to re-dispatch — finally pay off).
- The compile cache (`compile_cache`) key must stop including input constants for input nodes; the CPU
  interpreter reads the same uniform slot.

Scope: a new `RvNode::Input { idx }` (or a flag on `Src`/`ConstNum`), emitter arms in `wgsl_emit` +
`wasm_emit` + the interpreter, a params-buffer extension in `gpu.rs`, and eval threading the input
values to the runner rather than folding them into the graph. Sizeable but self-contained; it is the
single change that turns the GPU from "fast per run" into "fast per keystroke" for input-driven pages.
Not required for the D0–D4 switch (which is landed), so tracked here rather than blocking it.
