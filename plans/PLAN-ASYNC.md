# PLAN-ASYNC — an async `Engine::run`, the evaluator groundwork for WebGPU

**Date:** 2026-07-13 · **Status:** proposal (nothing started). Supersedes the
engine-in-a-worker + `Atomics.wait` bridge sketched in PLAN-WEBGPU's "async bridge"
paragraph — this is the cleaner shape, and it lands value before any GPU work exists.

## The decision

Make evaluation suspendable: the internals of `Engine::run` / `run_to_document` become
`async`, so a forcing (`P`/`E`/`Var`/`Q`) can `await` an asynchronous backend — the WebGPU
plan's dispatch + `mapAsync` readback — instead of blocking a thread the browser doesn't
let us block. Queries force *mid-expression* (`x = P(A) * 2`), so suspension must reach
through the evaluator; `async fn` is the compiler writing that state machine for us, and
the alternatives don't work: a hand-rolled effect/trampoline can't suspend mid-expression
without CPS-rewriting `eval`, and the `Atomics.wait` worker bridge works but adds a second
agent, a SharedArrayBuffer protocol, and a deadlock surface — permanently.

Beyond GPU, an async engine is worth having on its own in the playground: cancellation
(stop a runaway program), progress reporting between forcings, and awaiting a `setTimeout`
yield so a 20-second run stops freezing the tab.

**Breaking-change posture:** breaking the public API is acceptable (owner's call,
2026-07-13). The design below still keeps the existing sync names as thin `block_on`
wrappers — not for compatibility's sake, but because all ~90 existing call sites (CLI +
native tests; counted 2026-07-13) would otherwise each wrap themselves in `block_on`
anyway. Async capability is the point, not async spelling at every call site.

## Design

### The async set (scoped by call-graph walk, 2026-07-13)

Async infection propagates from two roots: (a) anything that calls `self.eval(…)`, and
(b) the forcing leaves. The full set — nothing outside it changes:

| file | functions that become `async` |
|---|---|
| `eval.rs` | `run`→`run_async`, `run_with_inputs`, `run_to_document`, `check`, `eval_top`, `eval` (the boxing point), `eval_inner` |
| `eval/eval_arms.rs` | `render_template`, `eval_array`, `eval_range`, `range_bound`, `eval_index`, `eval_for`, `eval_comprehension`, `eval_bind`, `eval_call`, `dispatch_call`, `input_call`, `eval_input_num/str/value` |
| `eval/draw_lift.rs` | `call_user_fn`, `eval_sample`, `lift_if` |
| `eval/ops.rs` | `eval_unary_expr` (only — `binop` etc. take evaluated `Value`s) |
| `eval/library.rs` | `eval_matmul` (evals its operands). **Not `lib_call`** — see the adjoint note below |
| `eval/introspect_dispatch.rs` | `eval_cond`, `query_cond` |
| `builtins.rs` | `call`, `prob`, `moment`, `quantile`, `prob_cond`, `moment_cond`, `quantile_cond` |
| `sampler.rs` | new async twins: `moments_async`, `sample_n_async`, `cond_moments_async`, `cond_sample_n_async` — **the GPU seam**; today they wrap the sync fns and complete on first poll |

Everything else stays sync: `eval_ident`, `gather`, `binop*`, `draw*`, `lift_unary/binary`,
`operand_to_rv`, all of `introspect.rs` and the `plot::`/`stats::` dispatch (their
joint-kernel GPU path is PLAN-WEBGPU G4 — flipping them later is a local change since
`introspect_dispatch` is already on the async spine), `reduce.rs`, every backend,
`doc.rs`'s assembly, `conformance.rs`, and the `jit`/`wasm_emit` parity tests (they call
the sync `sampler::` fns, which remain).

### The one recursion point boxes

`eval → eval_inner → {arms} → eval` is the only cycle, so `eval` is where the future gets
boxed (a plain `async fn` cycle would be an infinitely-sized type):

```rust
fn eval<'a>(&'a mut self, node: &'a Spanned)
    -> Pin<Box<dyn Future<Output = Result<Value>> + 'a>>
{
    Box::pin(async move { /* depth guard, then self.eval_inner(node).await */ })
}
```

Call sites change `self.eval(x)?` → `self.eval(x).await?`. Everything between two `eval`
levels (`eval_call → dispatch_call → builtins::call`) stays as plain `async fn`s whose
state machines embed into the parent — the box at `eval` cuts the size recursion, so
future sizes stay bounded. Not `Send` (Engine holds `Rc`), which is fine: `block_on` and
`wasm_bindgen_futures::spawn_local` don't need it.

Cost: one heap allocation per `eval` visit (graph *build* time, not the sampling hot loop
— builds are thousands of nodes, mallocs are ~20 ns; sampling throughput is untouched).

### `exec::block_on` — the sync bridge

New `exec.rs` (~60 lines, std-only, no dependency), exported as `noise_core::block_on`:

- **native:** park/unpark executor (`std::task::Wake` over `thread::current()`), truly
  blocks until ready — correct even for a genuinely-suspending future later.
- **wasm32:** poll once with `Waker::noop()`; `Pending` panics with a pointer at the async
  API. Today no CPU backend suspends, so the sync wasm exports keep working *unchanged*;
  the day a GPU forcing suspends, the browser host must be on `run_async` — enforced
  loudly, not by silent deadlock.

Public surface: `run`/`run_with_inputs`/`run_to_document`/`run_rv`/`check` keep their
names and sync signatures as one-line `block_on(self.…_async(…))` wrappers; the `_async`
twins are the real implementations. `Engine::moments`/`sample` (Rust test API) stay sync.
`noise-wasm` keeps its sync exports and gains a `run_async` export
(`wasm-bindgen-futures`, returns a JS `Promise`) for the playground to migrate to.

### Subtleties found while scoping (each cost real reading — don't rediscover them)

- **`lib_adjoint` is the one `library.rs` leak.** It calls `builtins::call("transpose", …)`
  — a *pure* builtin that never forces. Making `builtins::call` async would drag all of
  `lib_call` (30+ dispatch arms) into the async set for nothing. Fix: make `builtins::`
  `transpose` `pub(crate)` and call it directly; `lib_call` stays fully sync.
- **`dispatch_call`'s `#[inline(never)]`** exists to keep locals out of the recursive
  `eval_call` stack frame. As an async fn the attribute applies to the constructor, not
  `poll`, and the boxing at `eval` already bounds future sizes — drop the attribute and
  update its comment rather than carrying a stale rationale.
- **`MAX_EVAL_DEPTH` stays.** Polling nested ready futures recurses on the machine stack
  just like direct calls (async is not a trampoline); the boxed frames move *storage* to
  the heap but not the poll chain. The 2048 budget and its test remain load-bearing.
- **`stats::install` guard across `.await`.** `run_async` holds the thread's stats
  recorder for the whole run. Today nothing suspends; once GPU forcings do, two engines
  interleaving on the JS main thread could mis-attribute counters. Note it on the guard
  now; fix (re-install around suspension points, or task-local) belongs to PLAN-WEBGPU G3.
- **`QueryCtx` borrows across `.await`** (`builtins::call(base, &args, &self.query_ctx(span)).await`)
  are fine: shared borrow of the graph, no `&mut self` use during the await, futures never
  cross threads.
- **`Expr::Bind`'s `input_call` interception and the `Template` arm** both route through
  async fns (`input_call` evals its field exprs; `render_template` evals holes) — they're
  in the set above; easy to miss since they don't look like evaluation.

## Steps

- **A1 — the spine (one PR, ~400–800 mechanical lines).** `exec.rs`; the async set above;
  sync wrappers; `sampler::*_async` seam with its GPU-seam doc block. All 90 existing call
  sites and every test pass unchanged. Deliverable proof: an `exec` unit test drives a
  pending-once future (the shape of a GPU readback) through `block_on`.
- **A2 — wasm async export.** `noise-wasm::run_async` returning a Promise
  (`wasm-bindgen-futures`); playground switches its run path to `await`. The panic-catch
  wrapper (`catch_unwind`) can't span an `await` — accept Promise rejection as the error
  path for the async export and keep the catching sync export until the playground
  migrates.
- **A3 — the payoff before GPU.** A cooperative yield between top-level statements on
  wasm (`await` a resolved-next-microtask future every N statements) + a host cancellation
  flag checked at forcing boundaries. This is user-visible value (responsive tab, a Stop
  button) that exercises real suspension through the whole spine — retiring the risk that
  A1's awaits are wired wrong before any GPU code exists.
- **A4 — hand off to PLAN-WEBGPU G3.** The GPU backend routes inside `sampler::*_async`;
  no evaluator changes remain on that path.

## Risks

| risk | assessment |
|---|---|
| Borrow-checker surprises holding borrows across `.await` in 60+ fns | Scoping pass found none structural (sequential awaits, disjoint borrows); expect point fixes, not redesign |
| Poll-chain stack depth differs from sync recursion | Same order of magnitude; `MAX_EVAL_DEPTH` test catches regression — if it trips, lower the constant |
| Future-size bloat in `eval_inner` (largest arm dominates) | Boxed at `eval`, so bounded per level; check `cargo llvm-lines`/compile warnings if it balloons |
| Per-node `Box::pin` slows graph build | Build time is negligible vs sampling; verify with `example_times` before/after (turboquant builds ~17k nodes/forcing — the worst case) |
| wasm sync export panics once GPU lands | By design (loud, not deadlock); A2 must land before any suspending backend ships |
