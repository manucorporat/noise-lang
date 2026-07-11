# PLAN-TECH-DEBT.md — runtime tech-debt review & remediation plan

A full review of the Rust runtime (`crates/noise-core`, `noise-cli`, `noise-wasm`) covering
crash safety, correctness, backend parity, diagnostics quality, API surface, structure,
documentation, and test coverage. Conducted 2026-07-11 against `master` (4a497d4).
Every crash/misbehavior claim marked **[verified]** was reproduced by actually running a
program against the engine or CLI, not inferred from reading.

**Method:** all ~19.4k lines of runtime Rust were read; `cargo test` (349 pass native,
358 pass `--features jit`), `cargo clippy --all-targets` (**fails** — 3 errors),
`cargo fmt --check` (**fails**), and `cargo bench --bench sampling` (**fails at startup**)
were run to ground the findings.

## Executive summary

The codebase is in *unusually good shape internally* — typed spanned errors, invariants
written down at the right places (append-only graph, CSE-as-correctness, Recipe-vs-drawn,
determinism contract), a ~350-test golden corpus, and every codegen failure path falling
back to the interpreter. The debt is concentrated in five places:

1. **A crash/hang class reachable from a few characters of playground input.** The repo's
   own contract is "no panics in the pipeline" (`error.rs`, AGENT.md), but unbounded
   recursion (parser, eval, signal trees, graph walks) **aborts the process** — worse than
   a panic — and graph *construction* is uncapped while sampling is carefully budgeted.
2. **Silently wrong numbers** in the introspection surface (catastrophic cancellation:
   `sd` off by 17× at mean 1e8 **[verified]**), in conditioning (NaN sentinel conflation),
   in display rounding at large SE, and in large-λ Poisson.
3. **No CI whatsoever.** `.github/` does not exist; clippy, fmt, and the benches have
   already rotted, and the JIT feature is only tested when someone remembers the flag.
4. **Backend parity is enforced by hand-maintained near-copies** that have already
   diverged (jit vs wasm test corpora), with three documented-vs-actual divergences.
5. **The public API is the entire crate** (26 `pub mod`s, zero `#[non_exhaustive]`),
   and 98.6% of `lib.rs` is a test module hiding a 49-line API.

The plan below is phased so the guardrails land first (CI catches regressions while the
rest of the work proceeds), then the crash class, then correctness, parity, diagnostics,
and finally structure/API/doc cleanup.

---

## Findings catalog

Severity: **P0** = user-reachable crash/hang · **P1** = wrong/misleading results ·
**P2** = drift/maintenance hazard · **P3** = polish.

### A. Crash & hang class (P0)

| # | Where | Finding |
|---|-------|---------|
| A1 | `parser.rs:226-419` | `parse_bp`/`parse_prefix`/`parse_primary` have no depth guard. `"(".repeat(1000) + "1" + ")".repeat(1000)` **aborts the debug CLI** (SIGABRT, no error produced); `-`/`^`/`[` chains likewise. **[verified]** The wasm playground has a much smaller stack than the native 8 MB. Fix: `depth: usize` on `Parser`, error past ~500 — the exact pattern `MAX_CALL_DEPTH` (`eval.rs:37`) already uses. |
| A2 | `eval.rs:388` | `eval` recurses down the left spine of flat chains: a 10k-term `1+1+1+…` parses fine but **aborts in eval** (release). **[verified]** `MAX_CALL_DEPTH` only guards user-function calls. Fix: shared recursion budget in `eval` (and note `simplify::rewrite` + `bytecode::compile` walk the same spine). |
| A3 | `signal.rs:114-146` | `has_noise`/`eval_at` walk the `Rc` DAG *as a tree*: (a) 50k-stage pipeline aborts on depth; (b) `for k in 0..40 { s = s + s }` — a 4-line program — hangs forever (2^40 walks; `max_opts` never fires because the work is outside the VM). **[verified]** `fmt_expr` has the same blowup. Fix: memoize per-`Rc`-node or convert to an arena like `RvGraph`; cap depth in `binop_signal` (`eval.rs:2366`). |
| A4 | `bytecode.rs:88-176`, `kernel.rs:87-153,178,237` | `lower`/`walk_cost`/`latency_bound`/`cost`/`cone_size` recurse with unbounded depth; `cumsum(~[200000] noise_white(1))` builds a 200k-deep `Add` chain and overflows the stack. `ancestors` (`eval.rs:4094`) already does it right with a worklist. Fix: iterative worklists (or a depth cap in `profitable()` routing oversized graphs to the interpreter). |
| A5 | `eval.rs:648-654` | `a..b` iterates a `f64` counter with no size cap: `0..1e12` OOMs, and for `a ≥ 2^53` the `i += 1.0` is a no-op → **true infinite loop**. Fix: integer length computed up front + a `RANGE_MAX` cap (in the spirit of `CORR_MAX`). |
| A6 | `eval.rs:1206-1238,3047,3081` | Graph *construction* is uncapped while sampling is budgeted: `~[1e15] unif(0,1)` → `Vec::with_capacity(10^15)` abort; `rotation(500)` → O(d³)≈10⁸ nodes; `permutation(1e5)` → O(n²). `complex_pow`'s 4096 cap (`eval.rs:2495`) is the right idea applied once. Fix: node-count ceiling in `RvGraph::push` or shape-product cap in the structured draws, with a teaching error. |
| A7 | `eval.rs:3194` | `quantize` panics on a NaN centroid: `sort_by(partial_cmp().unwrap())`. `use vec; quantize([1,2], [0/0, 1])` **panics**. **[verified]** Fix: `f64::total_cmp` + reject non-finite centroids like `lib_categorical` (`eval.rs:3424`) does. |
| A8 | `rng.rs:86-100` | Knuth Poisson is O(λ) per draw with no λ ceiling — `poisson(1e12)` is an effective hang the op budget can't see (`kernel::cost` prices it at 1 op). **Also P1:** for λ ≳ 745 `(-λ).exp()` underflows and results are silently far below λ. Fix: λ cap or normal/PTRS approximation above a threshold; price Poisson as O(λ) in the cost model. |
| A9 | `crates/noise-wasm/src/lib.rs` | No panic hook, no `catch_unwind`: the doc says `run` "never throws", but any Rust panic (A7, the `unreachable!`s) becomes an opaque JS `RuntimeError: unreachable` and poisons the instance. Fix: panic hook routing the message into the existing `doc_error` Document shape. |

### B. Silently wrong results (P1)

| # | Where | Finding |
|---|-------|---------|
| B1 | `introspect.rs:104-105,158-162` | `Dist1::from_draws`/`Dist2::from_pairs` use the textbook `E[x²]−E[x]²` formula → catastrophic cancellation. `X ~ normal(1e8, 1); plot::hist(X)` reports **sd = 17.03** (true: 1.0). **[verified]** Every `describe`/`hist`/`scatter`/`corr`/`explain` card is affected; `sampler::moments` already has streaming Welford. Fix: two-pass (draws are already materialized) or reuse Welford; same in `Dist2`. |
| B2 | `eval.rs:1565-1576`, `reduce.rs:276-284`, `sampler.rs:78-84` | The NaN conditioning sentinel conflates "condition false" with "quantity is NaN": `E(math::log(X) \| X > -1)` with `X ~ unif(-1,1)` silently drops the NaN in-condition lanes → biased estimate with a *tighter* reported SE. Fix (min): document the hole at all three sites; (better): a dedicated condition column, as `sample_pairs` already does. |
| B3 | `value.rs:135-142` | `round_to_se` clamps digits at 0 — never rounds *left* of the decimal: `normal(3,1000)` displays `241`, false precision exactly when error is largest. **[verified]** Fix: negative digits or `mean ± se` display when `se ≥ 1`. |
| B4 | `eval.rs:1314-1323` vs `rng.rs:65-71` | `unif_int` has two algorithms: constant bounds → Lemire (validated); random bounds → `lo + floor((hi−lo+1)·U)` with no per-lane validation — an inverted per-lane range silently yields out-of-range values, and `fill_uniform_int`'s `.max(1.0)` silently converts inversion into a point mass. Fix: clamp width in the lowered graph; `debug_assert!(hi >= lo)` in the constructor path. |
| B5 | `bytecode.rs:243-252` | Gather of a NaN index silently reads element 0 (`NaN as usize == 0`). Fix: define the semantics (NaN → NaN, or documented first-element) and test it. |
| B6 | `eval.rs:2295-2328` | `fold_identity`'s `0*x → 0` justifies discarding inf/NaN propagation with a "measure-zero" claim that is false for discrete RVs (`X ~ unif_int(0,5); 0*(1/X)` folds to `0`; honest IEEE gives NaN with p=1/6). Fix: state the discrete trade-off in the comment or restrict the fold to provably-finite operands. |
| B7 | `dist.rs:220-225`, `bytecode.rs:102` | `RvId(len as u32)` / `len as Reg` truncate silently — combined with A6, aliases unrelated nodes. Fix: checked casts (build-time only, free). |
| B8 | `stats.rs:29-54` | Run counters are a **thread-local global**: two engines on one thread (the documented playground sidecar pattern, `eval.rs:79-80`) corrupt each other; cross-thread `stats()` reads zeros. Also the four joint-pass drivers (`sample_pairs`/`grid_moments`/`grid_draws`/`corr_matrix`, `sampler.rs:93-251`) duplicate the batch loop and **none records `RunStats`** — every `describe`/`hist`/`corr`/plot pass is invisible in the engine readout, contradicting `stats.rs:3`. Fix: move counters into `Engine`; extract one `for_each_joint_batch` and record there. |

### C. Backend parity & conformance (P1/P2)

| # | Where | Finding |
|---|-------|---------|
| C1 | `benches/sampling.rs:29,48` | **The benchmark suite is broken**: `X**2` no longer lexes and `exp(2)` was renamed `exponential` — `cargo bench` panics at startup. **[verified]** PERF.md's "Reproduce" commands fail. Fix: update programs; add a `#[test]` that `run_rv` succeeds for every `CASES` entry so plain `cargo test` validates the bench corpus. |
| C2 | `jit.rs:776-835` vs `wasm_emit.rs:712-773` | No shared cross-backend conformance suite: the two parity corpora are hand-maintained near-copies that have **already diverged** (wasm gained 2nd/4th-moment and wide-range-trig probes; jit never did), tolerances are loose (mean-only, 0.05), and nothing runs `--features jit` automatically. Fix: one shared `const CASES` consumed by both, plus a deterministic const-graph exact-equality suite per op (no RNG → exact comparison across interpreter/JIT/wasm). |
| C3 | `jit.rs:660`, `wasm_emit.rs:529`, `approx.rs:77-81` | User-reachable `sin(1e12*X)` routes into the 2-term Cody–Waite reduction that degrades for large \|x\| (~1e-6 at 1e10, nonsense at 1e15) while the interpreter uses libm — **the backend changes results**, violating the "backend only changes speed" contract (`backend.rs:33-34`, PERF.md). Fix: range-guard the emitted trig (`\|x\| < T ? poly : call` — the shim machinery exists) or treat unbounded-argument trig as a call in `walk_cost`. |
| C4 | `jit.rs:107-114` | **Every JIT-compiled kernel leaks its executable memory**: `JITModule` is dropped without `free_memory()`, and cranelift-jit's `Memory::drop` deliberately `mem::forget`s (verified in cranelift-jit 0.127.4 sources). Unbounded in a REPL/server. Fix: `impl Drop for JitProgramInner` calling `free_memory()` (sound — the existing `Arc`/Send/Sync argument already establishes no kernel pointer outlives it). |
| C5 | `wasm_host.rs:68,75,80-82,129` | The JS LRU evicts by insertion order with no liveness check and `nz_kernel_seed/run` dereference without an `undefined` check → eviction of a live handle throws instead of falling back. Also `WasmRunner` is the only runner created **without seeding** — skipping `reseed` yields all-zero xoshiro state or a *previous program's* state via the content-addressed shared instance. Fix: status-returning host calls with interpreter fallback + pin/release handles; seed in `Program::runner()` like `jit.rs:140`. |
| C6 | `kernel.rs:118,127,188` | Cost-model catch-alls (`_ => *fusible += 1`, denylist in `latency_bound`) silently misclassify **future** ops — the emitters are protected by exhaustive matches, the gate/stream policy is not. Fix: exhaustive matches over `Source`/`UnOp` at the policy layer; longer-term derive gate + latency + `supported` from one per-op `CostClass` table (three parallel classifications exist today: `kernel.rs:87-153` vs `178-203` vs `209-221`). |
| C7 | `backend.rs:44-49` | `#[cfg(feature = "jit")]` selects Cranelift with no `not(target_arch = "wasm32")` qualifier — feature unification onto a wasm32 build selects an impossible backend. Fix: `all(feature = "jit", not(target_arch = "wasm32"))`. |
| C8 | `wasm_emit.rs:624-629` | `&&`/`\|\|` lowered as `f64.min/max` vs interpreter/JIT's `(a≠0)∧(b≠0)` — equivalence rests on an asserted-nowhere "bools are exactly 0/1, never NaN" invariant. Fix: same `!=0` lowering (2 ops) or document + cover in the conformance suite. |
| C9 | `approx.rs:78,139-141`, `jit.rs:635-649` | Trig reference uses `round` (half-away) vs emitters' `nearest` (half-even) → the "op-for-op agreement" claim is false at ties; `ln` accuracy only tested to 1e-5 though Box–Muller feeds ~1.1e-16; subnormals pass the `x > 0` guard into exponent bit-surgery and get a wrong finite answer. Fix: `round_ties_even` in the reference, extend the `ln` test to uniform-reachable range, flush or document subnormals. Related P3: `Exp` lowered as `pow(e,x)` vs `x.exp()`; wasm float-method `unif_int` can yield `hi+1` where Lemire can't (`wasm_emit.rs:347-351`) — clamp with one `f64.min`. |
| C10 | `wasm_emit.rs:107-108` | `pub fn emit` panics (assert + `unreachable!`) when handed an ungated graph; its safe wrapper is `emit_for`. Fix: `pub(crate)` or return `Result`. |

### D. Diagnostics & error-type quality (P1/P2)

| # | Where | Finding |
|---|-------|---------|
| D1 | `error.rs:46-56`, `noise-cli/main.rs:226`, `doc.rs:694` | **No line/column anywhere**: errors render raw byte offsets (`(at 42..43)`) in the CLI and ship bare `{start,end}` in Document JSON — while a byte→line index already exists, private and unused for errors (`doc.rs:172-195 LineIndex`). Fix: promote `LineIndex`/`Span::line_col(src)`, render `file:line:col` + caret in the CLI, add `line`/`col` to the JSON error. Biggest UX win per line changed. |
| D2 | `error.rs:20-29` | Errors are stringly typed: no codes, no expected/found data, no structured "undefined name"; hosts must substring-match, `DocError` flattens even the kind away. Fix: enum-ify the frequent cases (`UndefinedName{name}`, `TypeMismatch`, `NotDrawn`, `ArityMismatch`) with a `code()`, keep `Runtime(String)` as catch-all during migration. |
| D3 | `parser.rs:834-843` | Lexer errors inside a template hole carry hole-local offsets — `tokenize(src)?` propagates before spans are rebased, breaking LANG.md:267's explicit promise. **[verified]** Fix: shift the error span by `base_offset` before `?`. |
| D4 | `lexer.rs:100,296-300` | Non-ASCII source → mojibake diagnostic (`π` reported as `'Ï'`) with a 1-byte span that is **not a char boundary** — hosts slicing `&src[span]` for a caret will panic. **[verified]** Fix: `chars().next()` + `len_utf8()` span at the error site. |
| D5 | `parser.rs:541-548,565-572` | `for 5 in xs` error span points at the token *after* the offender (bumped before capture); `parse_fn_def:194-202` does it right. **[verified]** Fix: bind the bumped token's span. |
| D6 | `input.rs:130-137` | `InputSpec::resolve` errors use `Span::new(0,0)` though the declaration span is known at the caller. Fix: thread `stmt_span` in. |
| D7 | `parser.rs:82,115,199,371,394,415,544,567` | Messages interpolate `{:?}` of `TokKind`: users see `expected Eq, found Ident("foo")` / `found Eof`. Fix: a `describe(&TokKind) -> &str` used by all sites. |
| D8 | `parser.rs:17-24`, `doc.rs:101-115` | No error recovery: one typo → zero-block Document, the playground loses the whole literate rendering. Fix (stretch): statement-level resync to next `;`/`}`, collect multiple errors — the doc model already tolerates partial results. |
| D9 | lexer | `1e6` isn't lexed (confusing `found Ident("e6")` error) and 300-digit literals silently become inf-adjacent f64s. Fix: scientific notation is one branch in the number lexer (`lexer.rs:123-145`); warn on overflow. |

### E. Public API surface & semver (P2)

| # | Where | Finding |
|---|-------|---------|
| E1 | `lib.rs:7-34` | All 26 modules are `pub mod` — `bytecode`, `rng`, `jit`, `kernel`, `simplify`, `wasm_emit` are all public API, making the curated `pub use` block meaningless for semver. Fix: `pub(crate)` internals (transition via `#[doc(hidden)]`); keep `error`, `value`, `eval` facade, `input`, `doc`, `introspect`, `frontmatter`, `stats`. |
| E2 | `value.rs:17-80`, `error.rs:19-29`, `doc.rs:38-43` | Zero `#[non_exhaustive]` in the repo; `Value` gains a variant nearly every plan cycle and both hosts match on it. Fix: `#[non_exhaustive]` on `Value`, `ErrorKind`, `Output`, `Block`, `Payload`, `InputKind`, and public-field structs. |
| E3 | `lib.rs:51-3532` | 98.6% of lib.rs is one `#[cfg(test)]` module (244 tests, all via public API); the real API is 49 lines. All 178 unwraps are test-only. Fix: move to `tests/` split by theme — this also *enforces* E1, since integration tests only see the exported surface. |
| E4 | `lib.rs:1-5,47-49` | Crate rustdoc says the runtime "lands in Phase 2"; `run(src)` silently drops emissions and errors on bare builtin names under strict scoping — the first thing every docs.rs reader tries. Fix: rewrite the header; document or fix `run`. Zero doc-tests exist crate-wide. |
| E5 | `eval.rs:39,388` | `Engine` (the main entry point) and `Engine::eval` (the central recursion) have no doc comments while every helper does; the load-bearing persists-across-runs lifecycle is only discoverable via a field comment. |

### F. Structure, duplication & idiom (P2)

| # | Where | Finding |
|---|-------|---------|
| F1 | `eval.rs` (4,468 lines) | Five modules in one file, with seams already named by section comments: builtin library (~1,200 lines), introspection/plot/stats dispatch (~600), input system, complex arithmetic, noise/signal materialization, module-scoping tables. Each talks to the core only through `binop`/`select`/`operand_to_rv`/`&mut RvGraph` — extraction is mechanical. |
| F2 | 7 string tables | The builtin namespace is defined in ≥7 disjoint places: `module_of` (`eval.rs:4146` — the declared "single source of truth"), `lib_call`, `builtins::call`, `is_introspection`, `STATS_FNS`, `plot_call`, `stats_call` — plus the TextMate grammar (and its vendored copy), LANG.md, and the skill. AGENT.md itself documents the two-place registration trap. Fix: one `const BUILTINS: &[(name, Module, Impl)]` registry; minimum: a `#[test]` that every `module_of` name dispatches and every dispatchable name has a module. |
| F3 | `builtins.rs:192-198,256` | **Dead `"sqrt"` arm with contradicting semantics**: `lib_call` intercepts sqrt first (NaN on negative); the unreachable builtins arm *errors* on negative — a dispatch reorder silently changes `sqrt(-1)`. `"Print"` arm likewise dead. Fix: delete both. |
| F4 | 4 copies of scalar `BinOp` | eval const-fold, bytecode VM, `simplify::binary`, `signal::scalar_binop` (+ jit/wasm legitimately separate) — `Mod` is hand-spelled in two. Fix: one `fold_binop(op, a, b)` for the interpretive paths. |
| F5 | `flint.rs:170-181` vs `value.rs:147`, `introspect.rs:367` vs `builtins::Q` | Acknowledged copy-paste pairs ("local to avoid a dep" — the dep is a `use` line in the same crate) that already behave differently. Fix: one home in `value.rs`/a small `num.rs`. |
| F6 | `builtins.rs:49-57,505,538,574` | 7-positional-arg `call` signature, widened per engine knob — classic parameter-object smell. Fix: `QueryCtx { graph, default_n, max_opts, check, span }`. |
| F7 | `dist.rs:26-41` | Vestigial `Distribution` trait: one impl, zero trait-dispatch callers, five distributions added since ignored it, module doc still advertises it as "the extension seam". Fix: delete or make real. |
| F8 | `value.rs:76-82`, `eval.rs:487` | `Value::Continue` control sentinel leaks into data positions: `x = continue; x + 1` errors with "arithmetic on continue and number" at the wrong place; arrays can contain it. **[verified]** Fix: reject in `eval_bind`/`eval_array`/call args, or restrict `continue` to statement position in the parser. |
| F9 | naming | `set_max_opts` means max *operations* (`MAX_OPS_DEFAULT` spells it right) — public language surface; decide now, alias `set_max_ops`. `noise_sigma` reused to parse `tau` (wrong error text); `lib_extreme(name,…)` re-dispatches on strings inside string-dispatched functions. |
| F10 | misc idiom | `scalar_const(x).is_some()` guard + `.unwrap()` re-eval (`eval.rs:2377,2381`) → `if let`; clone-pressure in fold loops (`binop` takes owned `Value` everywhere; a `binop_ref` would stop the pattern spreading); missing `#[must_use]` on `Engine::check`/`run_rv`/`Moments`/`RunStats`/`kernel::cost`; `signal.rs:140` release-path `unreachable!` guarding a cross-module caller contract. |

### G. Hosts: CLI & wasm (P2)

| # | Where | Finding |
|---|-------|---------|
| G1 | `noise-cli/main.rs:20-33` | Hand-rolled argv: any unknown flag becomes a file path (`noise --version` → "cannot read --version"); no `--version` at all; `-h` only in position 1. Fix: `lexopt`/`clap` or reject unknown `-` args + add `--version`. |
| G2 | `noise-cli/main.rs` | All failures exit 1 (usage errors included); `validate` silently ignores extra args and can't take `--input`; piped stdin drops into the REPL with `»` prompts in the output stream; REPL has no line editing/multi-line. Fix: exit 2 on usage, stdin detection, wire `--input` into validate. |
| G3 | `noise-cli/build.rs:29-43` | Build script **mutates the source directory** on every build (vendoring the VS Code extension) — breaks read-only/hermetic builds; sync failures demoted to warnings so a stale grammar ships silently. Fix: invert (xtask sync + CI equality check). |
| G4 | `noise-wasm/lib.rs:73-82,190-233` | Triple serialization per run (Document → `serde_json::Value` → `String` → UTF-16 → `JSON.parse`); `opts_json: Option<String>` forces a copy. Fix: `Serialize` directly / `serde_wasm_bindgen`; take `&str`. |
| G5 | `noise-wasm/lib.rs:141-156` | `Request::to_call` builds source by string-interpolating host-supplied names — not a security boundary today, but produces baffling errors and couples the sidecar protocol to the surface grammar. Fix: validate names against `engine.bindings()` first. |
| G6 | both hosts | **Zero tests** in noise-cli and noise-wasm: `parse_input_arg`, the JSON `doc_error` contract (with a hand-written fallback string at `lib.rs:65` that must stay in sync), `Request` parsing, opts clamping — all untested. `noise-wasm/Cargo.toml` lacks version on the path dep + publish metadata (inconsistent with siblings). |

### H. Workspace hygiene & CI (P0 for the project's trajectory)

| # | Where | Finding |
|---|-------|---------|
| H1 | `.github/` absent | **No CI.** AGENT.md mandates "clippy must stay clean" but `cargo clippy --all-targets` **fails today** (3 `approx_constant` errors in `flint.rs:636,644,645` test literals), `cargo fmt --check` fails, benches are broken (C1), and the JIT config is only tested by hand. The only automation is Netlify's site deploy. |
| H2 | `.gitignore:3` | `Cargo.lock` is ignored in a workspace shipping a binary — non-reproducible `cargo install`/CI, bad upstream releases land silently. Fix: commit it. |
| H3 | `noise-core/Cargo.toml:24` | `serde_yaml 0.9.34` is archived/unmaintained (RUSTSEC-2024-0320), used only for frontmatter. Fix: `serde_yml`/`serde-yaml-ng` or a minimal hand parser; add `cargo-deny`/`audit` to CI. |
| H4 | root `Cargo.toml` | No `[workspace.lints]`, no `[workspace.dependencies]` (wasm-bindgen and serde_json declared independently twice), edition 2021, no rustfmt/clippy config, zero `#![warn(missing_docs)]`. |

### I. Documentation drift (P2)

| # | Where | Finding |
|---|-------|---------|
| I1 | LANG.md:306-319 vs `parser.rs:622,631` | The precedence table gives `\|` and `..` different levels; the parser gives them identical binding powers — `a \| b .. c` parses as `(a\|b)..c` against the documented table. The informal grammar is missing comprehensions, named args, templates, `continue`, `~[shape]`. |
| I2 | AGENT.md:119-133,138,241 | Module table omits the entire document-model + backend layer (13 modules); test counts stale twice over (191 / 253 / actual 244+). No end-to-end picture of the document data flow (each module header is fine; the joined story isn't told anywhere). |
| I3 | `kernel.rs:12-14,68-73`, `jit.rs:317,377` | Cost-model docs claim the wasm emitter imports ln/sin/cos from the host — it inlines them (`wasm_emit.rs:184-189`); "six math shims" (three); `supported()` credited for a `profitable()` guard. Anyone tuning the gate reasons about the wrong browser behavior. |
| I4 | No "add an op" checklist | A new op needs coordinated edits in ≥6 places (ast, bytecode, jit, wasm_emit, kernel cost+latency, wasm_host imports + wasmi test linker), discoverable only by reading all of them. AGENT.md's worked example predates Phase 4. |
| I5 | PERF.md | "Reproduce" commands fail (C1); "multi-stream scalar dominates the vector path on every target" contradicts `simd_probe.rs`'s recorded NEON findings (1.27× / 1.15× on the FP-bound class); `**` in examples no longer lexes. Also worth one line each in LANG.md: `print(1e-13)` prints `0` (`format_num`), and `&&`/`\|\|` don't short-circuit side effects. |
| I6 | `sampler.rs`/`builtins.rs:413` | The NaN-sentinel contract is documented in three places, its known hole (B2) in none; `P` vs `Q` on the same event can disagree in the tail (different RNG stream consumption) — one sentence at `quantile` would save an afternoon of confusion. |

### J. Test gaps (cross-cutting)

- **No adversarial parser tests**: deep nesting (A1 — a 30-line test on a small-stack thread), template-hole offsets (D3), non-ASCII (D4), `1e6`, 300-digit literals. A `cargo-fuzz` target on `parser::parse` (pure `&str -> Result`) is the classic hand-written-Pratt payoff.
- **No "no program may panic" sweep** — `errors_dont_panic` (lib.rs:142) covers a fixed handful; A7 would have been caught.
- **No cross-backend conformance suite** (C2) and the jit corpus lacks the wasm corpus's moment/trig probes.
- **Untested mechanics**: Gather edge semantics (B5), `Est` display at large SE (B3), huge `..` bounds (A5), `RunStats` accounting (B8 — which is how the missing `record` calls went unnoticed), `kernel.rs` gate/stream policy (no in-file tests), deep-chain compilation (A4 — a `cumsum(~[100_000]…)` smoke test), conditional-quantity-NaN (B2), `cond_moments`/`sample_pairs`/`grid_*` lane mechanics under partial batches, `next_batch(len > cap)` on all three runners.
- **Frontmatter edges**: CR-only line endings fail with a *lexer* error **[verified]**; serde-YAML error locations aren't rebased to file coordinates **[verified]**; BOM before the fence silently disables frontmatter.
- **Zero CLI/wasm host tests** (G6), zero doc-tests, no test executing `examples/*.noise`.
- Error tests assert message substrings — consider `insta` snapshots so wording is reviewed, not fossilized.

---

## Remediation plan

Ordered so guardrails land first and each phase makes the next cheaper. Sizes:
S ≤ ½ day, M ≈ 1–2 days, L ≈ 3–5 days.

### Phase 0 — Guardrails (do first; ~2 days total)

Everything else in this plan will regress without this.

1. **CI workflow** (M): `cargo test --workspace`, `cargo test -p noise-core --features jit`,
   `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`,
   `cargo build --target wasm32-unknown-unknown -p noise-wasm`, `cargo audit`/`cargo-deny`.
2. **Make the gate pass** (S): fix the 3 `approx_constant` clippy errors (`flint.rs` tests —
   use a non-π literal), run `cargo fmt`, fix the bench programs (C1) + add the
   bench-corpus smoke test.
3. **Commit `Cargo.lock`** (S) (H2); hoist shared deps to `[workspace.dependencies]`,
   add `[workspace.lints]` (H4).
4. **Replace `serde_yaml`** (S) (H3).

*Acceptance: CI green on the branch; bench runs; `cargo audit` clean.*

### Phase 1 — Kill the crash/hang class (~1 week)

All P0s in section A. Each fix ships with the regression test from section J.

1. Parser depth guard (A1) + eval expression-depth guard (A2) sharing one budget (S/M).
2. Signal-tree memoization or arena + construction-depth cap (A3) (M).
3. Iterative worklists for `lower`/`walk_cost`/`latency_bound`/`cost`/`cone_size` (A4) (M).
4. `RANGE_MAX` + integer range iteration (A5) (S).
5. Graph-construction node budget (`RvGraph::push` ceiling + shape-product cap) (A6) (M);
   checked casts for `RvId`/`Reg` (B7) (S).
6. `quantize` NaN (A7) (S); Poisson λ cap/approximation + cost pricing (A8) (M).
7. wasm panic hook → `doc_error` Document (A9) (S).
8. Add the adversarial test module + "no program may panic" sweep + a `cargo-fuzz`
   target for `parse` (M).

*Acceptance: every reproduced abort in section A returns a spanned `NoiseError` instead;
fuzz target survives 10 min locally.*

### Phase 2 — Correct the numbers (~1 week)

1. Welford/two-pass in `Dist1`/`Dist2` (B1) (S) + a `normal(1e8,1)` regression test.
2. `round_to_se` at large SE (B3) (S).
3. `unif_int` dynamic-bounds clamp + debug_assert (B4) (S); Gather NaN semantics + test (B5) (S).
4. NaN-sentinel: document the hole everywhere it's described, then implement the
   dedicated condition column (B2) (M).
5. Per-engine `RunStats` + extract `for_each_joint_batch` (B8) (M).
6. `fold_identity` discrete-case honesty (B6) (S).

*Acceptance: introspection sd matches `sampler::moments` at mean 1e8; stats counters
correct with two interleaved engines; conditioning test with in-condition NaN pinned.*

### Phase 3 — Backend parity (~1 week, can overlap Phase 2)

1. Shared conformance corpus consumed by jit + wasm tests; const-graph exact-equality
   suite across interpreter/JIT/wasm (C2) (M).
2. Trig range guard in both emitters (C3) (M).
3. JIT `Drop` → `free_memory()` (C4) (S).
4. wasm host: seed on `runner()`, status-returning host calls with interpreter fallback
   (C5) (M).
5. Exhaustive matches in the kernel cost model; then the single `CostClass` table (C6) (M).
6. `backend.rs` cfg fix (C7) (S); `&&`/`||` lowering unification or documented invariant
   (C8) (S); approx reference `round_ties_even` + extended ln tests + subnormal policy
   (C9) (S); `emit` visibility (C10) (S).
7. Write the "add an op across backends" checklist in AGENT.md (I4) (S).

*Acceptance: one corpus, three backends, exact-equality suite green; `--features jit`
in CI (already from Phase 0) exercises it.*

### Phase 4 — Diagnostics UX (~1 week)

1. `Span::line_col` + CLI caret rendering + line/col in Document JSON (D1) (M).
2. Span fixes: template holes (D3), UTF-8-safe lexer errors (D4), for-loop spans (D5),
   input resolve spans (D6) — each is a few lines (S each).
3. `describe(TokKind)` — no more `{:?}` in user messages (D7) (S).
4. Structured `ErrorKind` cases with `code()` (D2) (M).
5. Scientific notation in the lexer + overflow warning (D9) (S).
6. Stretch: statement-level parser recovery + multi-error Documents (D8) (L).

*Acceptance: `π = 3` produces a correct single-char span; a template-hole error points
at the file location; CLI shows `file:line:col` with a caret; snapshot tests (insta)
for the top 20 messages.*

### Phase 5 — API surface & structure (~1.5 weeks)

Do *after* the behavior fixes so the churn doesn't conflict with them.

1. Move the lib.rs test module to `tests/` split by theme (E3) (M) — do this first;
   it enforces the next step.
2. Privatize internals, `#[non_exhaustive]`, curated re-exports (E1, E2) (M).
3. Split `eval.rs` along its own section seams into `eval/{lib,introspect,input,complex,noise}.rs`
   (F1) (L — mechanical, but do it in one PR with no behavior change).
4. Single `BUILTINS` registry replacing the 7 string tables, + the tmLanguage coverage
   test (F2) (M); delete dead builtins arms (F3) (S).
5. `fold_binop` shared scalar semantics (F4) (S); dedupe `fmt_n`/`quantile_sorted` (F5) (S).
6. `QueryCtx` parameter object (F6) (S); delete the `Distribution` trait (F7) (S);
   reject `Continue` in data positions (F8) (S); `set_max_ops` alias (F9) (S);
   `#[must_use]` + `if let` cleanups (F10) (S).
7. CLI: argv parsing, exit codes, stdin, `--version` (G1, G2) (M); build.rs inversion
   + CI sync check (G3) (M); wasm serialization path + request validation + host tests
   (G4, G5, G6) (M).

*Acceptance: `cargo doc` shows only the intended surface; `cargo semver-checks` adoptable;
eval.rs < 1,500 lines; one builtin registry with a coverage test; CLI/wasm test suites exist.*

### Phase 6 — Documentation refresh (~3 days, can trail each phase)

1. Regenerate LANG.md's precedence table from `infix_op` + fix the `|`/`..` discrepancy
   (decide: doc follows parser, or parser follows doc — this is a *language decision*,
   flag it before fixing) (I1).
2. AGENT.md: module table incl. document-model + backend layers, end-to-end document
   data-flow paragraph, test counts sourced from CI (I2).
3. Fix the three stale kernel/jit comments (I3); PERF.md repro commands + SIMD-probe
   refinement (I5); LANG.md notes for `format_num` tiny-number rounding and non-short-
   circuit `&&`/`||` (I5); NaN-sentinel + P-vs-Q stream notes (I6).
4. Crate rustdoc header rewrite + doc-tested examples for `Engine::run`/`run_to_document`
   (E4); doc comments for `Engine`/`Engine::eval` (E5).

*Acceptance: a new contributor can add an op or builtin using only AGENT.md; docs.rs
front page describes the real pipeline with runnable examples.*

---

## Sequencing notes & decision points

- **Total: roughly 6–7 engineer-weeks**, but Phases 2/3 and 4 are parallelizable, and
  Phase 0 + the top of Phase 1 (~3 days) removes the majority of the *user-visible* risk.
- **Decision needed (language surface):** `set_max_opts` naming (F9), `|` vs `..`
  precedence (I1), gather-NaN semantics (B5), and `sqrt(-1)` NaN-vs-error (F3) are
  user-visible choices — settle them before the fixes land, since each ships in error
  messages/results that examples and the skill depend on.
- **Deliberately out of scope:** parser error recovery beyond statement resync,
  Unicode identifiers (D4 only fixes the *diagnostic*), SIMD work (simd_probe's
  conclusion stands), and any language-feature work.

## Top 10 quick wins (each ≤ ½ day, immediately shippable)

1. Fix clippy errors + fmt + broken benches → CI green (H1/C1).
2. `quantize` NaN panic (A7).
3. `RANGE_MAX` for `a..b` (A5).
4. Welford in `Dist1`/`Dist2` (B1) — biggest wrong-number fix in the repo.
5. JIT executable-memory leak `Drop` (C4).
6. wasm panic hook (A9).
7. Template-hole + UTF-8 error spans (D3, D4).
8. Delete the dead contradicting `sqrt` arm (F3).
9. Commit `Cargo.lock`, replace `serde_yaml` (H2, H3).
10. wasm runner seeding (C5, the seed half).
