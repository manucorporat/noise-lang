//! Backend-agnostic kernel support shared by the code generators (PLAN.md Phase 4).
//!
//! The WASM emitter ([`crate::wasm_emit`], the browser codegen backend) and the WGSL emitter
//! ([`crate::wgsl_emit`], the native GPU backend) are *lowerings* of the same [`RvGraph`] IR: they
//! walk the identical graph the identical way over the same draw-ordinal contract — only the per-node
//! encoding differs (`f64.mul` vs a WGSL `*`, an inlined xoshiro step vs an emulated `squares64`).
//! Everything that is about *what the graph means* rather than *how to spell it on a target* lives
//! here, so there is exactly one copy:
//!
//!   * [`STREAMS`] / [`choose_streams`] — the multi-stream RNG policy.
//!   * [`seed_state`] — SplitMix64 expansion of a seed into the per-stream xoshiro state layout.
//!   * [`profitable`] / [`walk_cost`] — the cost model deciding whether the wasm emitter beats the
//!     interpreter (the "B2" gate). One cost function, parameterized by `inline_trans`: the WASM
//!     emitter inlines `ln`/`sin`/`cos` as polynomials (`wasm_emit::emit_ln`/`emit_trig`) and passes
//!     `true`; the `false` branch is a retained seam for a hypothetical backend that couldn't inline
//!     them. `exp` and the large-argument trig fallback remain real calls (a host import on wasm) —
//!     see PLAN.md "Browser note" and findings C3/C9. (This model also priced the retired native JIT,
//!     which shared it.)
//!   * [`const_int_exponent`] — the `x ^ k` small-integer-power test (repeated multiply vs a `pow`
//!     call), shared so the emitters agree on which exponents fuse.

// Backend-support helpers whose live set depends on the build config (wasm target, tests). This
// module was previously `pub`, which masked the same dead-code warnings.
#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ast::{BinOp, UnOp};
use crate::bytecode::BATCH;

/// **The configurable cost model** (PLAN-DROP-JIT, API note). Every codegen gate — the WASM emitter's
/// [`profitable_roots`] and the native GPU's `gpu::profitable` — balances two things: whether the
/// fused kernel is faster *per lane* (a runtime property) and whether a *single* forcing draws enough
/// to refund *compiling* it (an amortization property). The second is a bet about how the artifact is
/// used, and only the host knows the answer: a one-shot `noise file.noise` pays the compile once and
/// must earn it back that run; an **interactive** host (the playground re-running as a slider moves, a
/// JS caller sweeping inputs) reuses the same pipeline across many runs, so the compile is amortized
/// and the GPU is worth it whenever its *runtime* wins — even for a short forcing.
///
/// When this is set, the gates drop the amortization term and keep only the runtime terms (`ops/draw`
/// fatness, the joint per-root term). They do NOT drop the cold-compile *feasibility* cap
/// (`MAX_WGSL_INSTRS`): a shader that takes seconds to compile blocks the first run no matter how it
/// is reused, and — until inputs lower as shader *uniforms* rather than baked constants (see the API
/// note in `plans/`) — changing an input still recompiles, so a giant shader would pay that cost on
/// every slider drag. Off by default (one-shot). Set it from the host for an interactive session, or
/// with `NOISE_PREFER_RUNTIME=1`.
static PREFER_RUNTIME: AtomicBool = AtomicBool::new(false);

/// Set the cost model to prefer codegen runtime over one-shot compile amortization (see
/// [`PREFER_RUNTIME`]). Idempotent; process-wide, like the rest of the backend policy.
pub fn set_prefer_runtime(on: bool) {
    PREFER_RUNTIME.store(on, Ordering::Relaxed);
}

/// Whether the cost model prefers codegen runtime (see [`set_prefer_runtime`]). Also honors
/// `NOISE_PREFER_RUNTIME=1` (read once) so a host or a bench can flip it without code.
pub fn prefer_runtime() -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    PREFER_RUNTIME.load(Ordering::Relaxed)
        || *ENV.get_or_init(|| std::env::var("NOISE_PREFER_RUNTIME").is_ok_and(|v| v == "1"))
}
use crate::dist::{RvGraph, RvId, RvNode, Source};

/// Number of independent xoshiro streams a kernel interleaves (samples emitted per loop iteration).
/// xoshiro256++ is a serial dependency chain threaded through the whole loop, so on RNG-bound graphs
/// that latency — not the arithmetic — is the ceiling; running four independent states lets the
/// out-of-order core (native) or the host engine overlap the chains for ~2×. Four keeps register
/// pressure modest; past four the gain flattens (measured on the codegen backends). Must divide
/// [`BATCH`].
pub const STREAMS: usize = 4;

const _: () = assert!(
    BATCH.is_multiple_of(STREAMS),
    "BATCH must be a multiple of STREAMS"
);

/// Expand `seed` into `streams` independent xoshiro states. Each stream gets four consecutive
/// SplitMix64 outputs (mirroring `Rng::seed_from_u64`, so `streams == 1` seeds bit-identically to
/// the interpreter) at the strided positions `state[k * streams + j]` — the layout the kernels read
/// back. Distinct streams get well-separated substreams.
pub fn seed_state(seed: u64, streams: usize) -> Vec<u64> {
    let mut z = seed;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let mut state = vec![0u64; 4 * streams];
    for j in 0..streams {
        for k in 0..4 {
            state[k * streams + j] = next();
        }
    }
    state
}

/// Assign every source node in `graph` its **draw ordinal** — the `source` argument the `rng::fill_*`
/// functions key their counters on (`ctr = (source << 36) + lane`). Returned indexed by `RvId`, with
/// `NO_SOURCE` for nodes that draw nothing.
///
/// One rule, and it is the whole cross-backend draw contract: **walk node ids in order and hand out
/// ordinals sequentially.** A scalar [`RvNode::Src`] takes one; an [`RvNode::ArrDraw`] takes a
/// *contiguous block of `n`*, so its element `k` draws from `base + k` — the same stream `n`
/// separate `Src` nodes would have produced, which is what lets `ArrDraw` be a pure structural
/// change rather than a numeric one.
///
/// Every backend calls this on the same (simplified) graph and therefore agrees bit for bit. That
/// is the point: the ordinal must be a function of the *graph*, not of any backend's emission order,
/// or the interpreter and a codegen backend would draw different numbers from the same program.
///
/// **This replaced keying draws on the `RvId` itself** (`Inst::Normal { src: id.0 }`), which had to
/// go — one `ArrDraw` node cannot hand `n` distinct ordinals to its elements out of its single id.
/// The old scheme was also quietly brittle in a way this one isn't: it made the draw stream a
/// function of node *numbering*, so any new rewrite in `simplify` (a fold, a CSE) silently changed
/// results. Ordinals are now dense and depend only on which sources survive, in id order.
pub fn source_ordinals(graph: &RvGraph) -> Vec<u32> {
    let mut ord = vec![NO_SOURCE; graph.len()];
    let mut next: u32 = 0;
    for i in 0..graph.len() {
        let id = RvId(i as u32);
        let take = match graph.node(id) {
            RvNode::Src(_) => 1,
            RvNode::ArrDraw { n, .. } => *n,
            _ => 0,
        };
        if take > 0 {
            ord[i] = next;
            // The counter layout is `(source << 36) + lane`, so an ordinal past 2^28 would overflow
            // the u64 counter and alias another source's stream. Construction is capped far below
            // this (`eval::MAX_DRAW_LEAVES`), so this is insurance, not a live limit.
            next = next
                .checked_add(take)
                .filter(|&n| n < (1 << 28))
                .expect("source ordinals exceeded the 2^28 counter region");
        }
    }
    ord
}

/// Ordinal of a node that draws nothing (see [`source_ordinals`]).
pub const NO_SOURCE: u32 = u32::MAX;

/// Assign every **cell-stream** node its ordinal — the `stream` argument [`rng::CellStream::new`]
/// keys its counter region on. These are the draws that consume a *variable or per-lane-large*
/// number of u48s and so cannot pair-share a single hash: Knuth [`Poisson`](RvNode::Src), the
/// Fisher–Yates in [`Permutation`](RvNode::Permutation), the Gaussian seed of
/// [`Rotation`](RvNode::Rotation). Indexed by `RvId`; [`NO_STREAM`] for every other node.
///
/// Same contract, and for the same reason, as [`source_ordinals`]: **id order, sequential**, so the
/// assignment is a function of the graph and every backend agrees. It replaces a running counter
/// that `bytecode::lower` incremented in DFS *emission* order — which made the stream a function of
/// traversal, exactly the brittleness `source_ordinals` was created to remove (a `simplify` rewrite
/// could renumber it). A cell stream and a plain source draw from disjoint counter regions
/// (`CellStream`'s `base` sets bit 63), so the two ordinal spaces are independent — a `Poisson` node
/// carries one of each.
pub fn cell_stream_ordinals(graph: &RvGraph) -> Vec<u32> {
    let mut ord = vec![NO_STREAM; graph.len()];
    let mut next: u32 = 0;
    for i in 0..graph.len() {
        let id = RvId(i as u32);
        // A shaped Poisson (`~[n] poisson`) is an `ArrDraw` whose every element is an independent
        // Knuth draw, so it owns a *contiguous block of `n` streams* — the same block shape
        // `source_ordinals` uses. Its element `k` reads stream `base + k` (see [`elem_stream`]).
        // No other recipe an `ArrDraw` can hold needs a stream, and `Permutation`/`Rotation` are
        // array-valued sources of their own (never shaped), so each takes exactly one.
        let take = match graph.node(id) {
            RvNode::Src(Source::Poisson { .. })
            | RvNode::Permutation { .. }
            | RvNode::Rotation { .. } => 1,
            RvNode::ArrDraw {
                n,
                src: Source::Poisson { .. },
            } => *n,
            _ => 0,
        };
        if take > 0 {
            ord[i] = next;
            // `CellStream::new` debug-asserts `stream < 2^14`; keep the same ceiling here so an
            // overflow surfaces as this named error rather than a counter-region collision.
            next = next
                .checked_add(take)
                .filter(|&n| n < (1 << 14))
                .expect("cell-stream ordinals exceeded the 2^14 region");
        }
    }
    ord
}

/// Ordinal of a node with no cell stream (see [`cell_stream_ordinals`]).
pub const NO_STREAM: u32 = u32::MAX;

/// The cell-stream ordinal of an [`RvNode::ArrElem`] over a shaped-Poisson block: base plus index.
pub fn elem_stream(stream_ords: &[u32], arr: RvId, k: u32) -> u32 {
    let base = stream_ords[arr.0 as usize];
    debug_assert_ne!(
        base, NO_STREAM,
        "ArrElem's parent is not a cell-stream block"
    );
    base + k
}

/// The draw ordinal of an [`RvNode::ArrElem`]: its parent block's base, plus the static index.
pub fn elem_ordinal(ords: &[u32], arr: RvId, k: u32) -> u32 {
    let base = ords[arr.0 as usize];
    debug_assert_ne!(base, NO_SOURCE, "ArrElem's parent is not a source block");
    base + k
}

/// The `Source` recipe an [`RvNode::ArrElem`] draws from — i.e. its parent [`RvNode::ArrDraw`]'s.
pub fn elem_source(graph: &RvGraph, arr: RvId) -> Source {
    match graph.node(arr) {
        RvNode::ArrDraw { src, .. } => *src,
        other => unreachable!("ArrElem's parent must be an ArrDraw, got {other:?}"),
    }
}

/// If node `id` is a constant non-negative integer in `0..=64`, return it as an exponent count.
/// Such `x ^ k` lower to repeated multiply (no `pow` call) on every backend.
pub fn const_int_exponent(graph: &RvGraph, id: RvId) -> Option<u32> {
    match graph.node(id) {
        RvNode::ConstNum(x) if x.fract() == 0.0 && *x >= 0.0 && *x <= 64.0 => Some(*x as u32),
        _ => None,
    }
}

/// Largest non-const `Gather` table the backends lower as a compare/select chain
/// ([`GatherClass::SelectChain`]). The chain is `len` selects per lane, emitted inline with every
/// element's cone — cheap at bootstrap-block sizes, quadratic waste on a `permutation(5000)` deck,
/// where the interpreter's indexed load wins. 8 covers the small structured draws without letting
/// the chain dominate a kernel.
pub const GATHER_SELECT_MAX: usize = 8;

/// How the code generators lower a [`RvNode::Gather`] — the one shared classification the gate
/// ([`walk_cost`]), the sizing walks ([`cone_size_roots`]) and both emitters must agree on, or the
/// wasm local-slot pool would disagree with what actually gets emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherClass {
    /// Every element is a `ConstNum`: the table is materialized as an `f64` array at compile time
    /// (wasm: an active data segment) and the node is a **leaf** over
    /// its elems — round index, clamp, one indexed 8-byte load. This is the `rand::empirical` /
    /// `block_bootstrap` / literal-array shape, so a 10k-point table costs one load per lane and
    /// zero graph nodes.
    ConstTable,
    /// Small non-const table (≤ [`GATHER_SELECT_MAX`]): elems are emitted as ordinary nodes and
    /// the pick is a branch-free compare/select chain (no rounding needed — see the emitters).
    SelectChain,
}

/// Classify a `Gather` by its element list; `None` means the backends can't lower it (large
/// non-const table — data-dependent addressing over per-lane values stays interpreter-only).
pub fn gather_class(graph: &RvGraph, elems: &[RvId]) -> Option<GatherClass> {
    if elems
        .iter()
        .all(|&e| matches!(graph.node(e), RvNode::ConstNum(_)))
    {
        Some(GatherClass::ConstTable)
    } else if elems.len() <= GATHER_SELECT_MAX {
        Some(GatherClass::SelectChain)
    } else {
        None
    }
}

/// Whether codegen is expected to outperform the interpreter for this graph: the graph is supported
/// (no `Poisson` — its Knuth loop stays interpreter-only) **and** the count of fused nodes strictly
/// exceeds the transcendental-call weight. See [`walk_cost`] for the calibration.
///
/// `inline_trans` says whether *this backend* inlines `ln`/`sin`/`cos` as polynomials. The WASM
/// emitter does — via `wasm_emit::emit_ln`/`emit_trig` (over [`crate::approx`]) — so it passes `true`
/// and a `normal`/`exp`/trig graph counts as fusible. The `false` branch is a retained seam: a
/// backend that *couldn't* inline a transcendental would pass `false` and the gate would correctly
/// leave those graphs to the interpreter rather than emit a per-draw call. `atan`/`round`/`exp`/
/// non-integer `pow` (and the rare large-|x| trig fallback, finding C3) are real calls regardless.
pub fn profitable(
    graph: &RvGraph,
    root: RvId,
    inline_trans: bool,
    draws: usize,
    min_draws: usize,
) -> bool {
    profitable_roots(graph, &[root], inline_trans, draws, min_draws)
}

/// Multi-root [`profitable`]: gate the *union* of several roots' cones — the joint drivers compile
/// all roots into one shared kernel, so one gate decision covers the whole pass. Shared nodes are
/// weighed once (shared `seen`), matching what the joint kernel actually emits.
pub fn profitable_roots(
    graph: &RvGraph,
    roots: &[RvId],
    inline_trans: bool,
    draws: usize,
    min_draws: usize,
) -> bool {
    let mut seen = HashSet::new();
    let (mut fusible, mut libcalls) = (0u32, 0u32);
    for &root in roots {
        if !walk_cost(
            graph,
            root,
            &mut seen,
            &mut fusible,
            &mut libcalls,
            inline_trans,
        ) {
            return false; // unsupported (Poisson / large non-const Gather) → interpreter
        }
    }
    // Route very large cones to the interpreter. The code generator (`wasm_emit::emit_node`) emits
    // each node **recursively**, so a hundreds-of-thousands-deep graph (`cumsum(~[200000] …)`) would
    // overflow the emitter and abort (finding A4). The interpreter's lowering is now iterative
    // (stack-safe at any depth), and codegen of 10^4+ nodes rarely beats interpreting them, so this
    // only ever trades a little speed for safety. `seen` is the cone's distinct-node count (walk_cost
    // just filled it).
    if seen.len() > MAX_CODEGEN_NODES {
        return false;
    }
    if fusible <= libcalls {
        return false;
    }
    // Codegen has to *pay for itself*. Everything above asks whether the fused kernel is faster per
    // draw; this asks whether the query takes enough draws to refund *compiling* it — the
    // amortization term the configurable cost model drops for an interactive host that reuses the
    // pipeline across runs (see [`prefer_runtime`]).
    prefer_runtime() || draws >= min_draws
}

/// Minimum draws before the **WASM emitter** is worth emitting (see [`profitable`]'s `min_draws`).
///
/// Emitting is a fixed cost paid once per forcing; fusion is a saving earned per draw, so a query
/// that draws few samples never earns its emit back. That is not a corner case — real programs force
/// many small queries: `examples/noise_colors.noise` forces 14 at 3,000 draws each. The gate keeps
/// the emitter off those and on the ones that pay.
///
/// **One constant, not a cost curve.** A fitted `f(nodes)` break-even was tried (the retired native
/// JIT had a `bench_amortization` probe measuring it per cone size) and decided every program in
/// `examples/` exactly as this single threshold does — complexity with no behavior to show for it.
///
/// Measured with `packages/core/bench/examples.mjs` (V8, real programs): `WebAssembly.instantiate` on
/// a ~1 KB kernel is cheap, so the threshold sits an order of magnitude below where the old native
/// JIT's Cranelift compile paid off. `am_vs_fm` (40k draws/forcing) ran 1.20× emitted but 0.94×
/// gated at 100k, and `turboquant` (10k) 1.06× vs 0.90×; dropping to 10k keeps both emitting while
/// still declining `noise_colors` (3k draws/forcing), which is 0.31× — a 3× pessimization — when
/// always emitted.
pub const MIN_DRAWS_WASM: usize = 10_000;

/// Cone-node ceiling above which codegen is declined in favor of the interpreter (see
/// [`profitable`]). The recursive emitters would otherwise risk a stack overflow on a very deep
/// graph, and a graph this large compiles slowly for little benefit. Far above any ordinary model.
const MAX_CODEGEN_NODES: usize = 20_000;

/// Accumulate `(fusible, libcalls)` weights over the distinct nodes of `id`'s cone (each `RvId`
/// counted once, matching CSE). Returns `false` if the cone contains an unsupported node. See
/// [`profitable`] for the meaning of `inline_trans`.
pub fn walk_cost(
    graph: &RvGraph,
    id: RvId,
    seen: &mut HashSet<RvId>,
    fusible: &mut u32,
    libcalls: &mut u32,
    inline_trans: bool,
) -> bool {
    // Charge a transcendental as fusible when the backend inlines it, else as a call (`normal` is
    // ln+cos = 2 calls — the heaviest; `exp`/`geometric` are one `ln`).
    let charge = |weight: u32, fusible: &mut u32, libcalls: &mut u32| {
        if inline_trans {
            *fusible += 1;
        } else {
            *libcalls += weight;
        }
    };
    // Iterative worklist (not recursion): a 200k-deep `Add` chain would otherwise overflow the
    // stack (finding A4). Each distinct `RvId` is charged once (CSE); an unsupported node returns
    // `false` immediately (partial counts don't matter — `profitable` ignores them when unsupported).
    let mut stack = vec![id];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue; // shared node already counted
        }
        // TODO(CostClass): the three classifications in this module (`walk_cost` gate weights,
        // `latency_bound`, `supported`) are parallel per-op tables; a single `CostClass` table
        // deriving all three would remove the duplication. Until then every match here is
        // **exhaustive** (no `_` catch-all): a new `Source`/`UnOp`/`BinOp` variant must be
        // classified here explicitly or the crate fails to compile (finding C6) — the emitters'
        // exhaustive matches already force this on the codegen side, and this keeps the gate/stream
        // policy from silently misclassifying a future op as fusible.
        match graph.node(id) {
            RvNode::Src(Source::Poisson { .. }) => return false, // Knuth loop stays interpreter-only
            // The array-valued draws (permutation's Fisher–Yates, rotation's Gaussian-fill + MGS)
            // and their per-lane element read stay interpreter-only, like Poisson: per-lane loops
            // over an array register — not fusible scalar expressions.
            RvNode::Permutation { .. } | RvNode::Rotation { .. } | RvNode::ArrIndex { .. } => {
                return false
            }
            // A shaped draw emits nothing (it owns an ordinal block — see `source_ordinals`), so it
            // is free and a pure leaf. Its readers carry the whole cost.
            RvNode::ArrDraw { .. } => {}
            // An `ArrElem` IS a draw: it costs exactly what the equivalent scalar `Src` costs, and
            // it is unsupported for exactly the same recipe (Poisson's Knuth loop).
            RvNode::ArrElem { arr, .. } => {
                match elem_source(graph, *arr) {
                    Source::Poisson { .. } => return false,
                    Source::Normal { .. } => charge(2, fusible, libcalls),
                    Source::Exp { .. } | Source::Geometric { .. } => charge(1, fusible, libcalls),
                    Source::Uniform(_) | Source::UniformInt { .. } => *fusible += 1,
                }
                stack.push(*arr); // free, but it must land in `seen` so the cone count is honest
            }
            RvNode::Gather { elems, index } => match gather_class(graph, elems) {
                // Const-table gather is a LEAF over its elems: the emitters materialize the table
                // in memory and never emit the element nodes, so only the index cone is walked (a
                // 10k-point `empirical` table must not trip MAX_CODEGEN_NODES). Fixed fused cost:
                // round + clamp + convert + load.
                Some(GatherClass::ConstTable) => {
                    *fusible += 4;
                    stack.push(*index);
                }
                // Small non-const table: one select per element, elems emitted as ordinary nodes.
                Some(GatherClass::SelectChain) => {
                    *fusible += elems.len() as u32;
                    for &e in elems.iter() {
                        stack.push(e);
                    }
                    stack.push(*index);
                }
                None => return false, // large non-const table stays interpreter-only
            },
            RvNode::Src(Source::Normal { .. }) => charge(2, fusible, libcalls),
            RvNode::Src(Source::Exp { .. }) | RvNode::Src(Source::Geometric { .. }) => {
                charge(1, fusible, libcalls)
            }
            // uniform / uniform_int: cheap inline draw everywhere.
            RvNode::Src(Source::Uniform(_) | Source::UniformInt { .. }) => *fusible += 1,
            RvNode::ConstNum(_) | RvNode::ConstBool(_) => {} // neutral
            // A host input uniform (PLAN-UNIFORM-INPUTS P1): one aligned f32 load from the
            // first-page input region the host writes before each kernel call — as cheap as a
            // constant, so neutral. A slot past the region can't be addressed and declines (a
            // document would need >1024 inputs; unreachable in practice).
            RvNode::Input { idx } => {
                if *idx as usize >= crate::wasm_emit::INPUT_SLOTS {
                    return false;
                }
            }
            RvNode::Unary(op, a) => {
                match op {
                    // Inlined on both backends (trig/ln polynomials). `Sqrt` rides the same
                    // gate: an inline `sqrt` instruction natively, and on wasm a `Math.sqrt`
                    // import that V8 executes at near-instruction cost (the inline `f64.sqrt`
                    // form regressed V8/arm64 ~30% on large kernels, 2026-07-14) — so it is
                    // charged fusible wherever the backend inlines transcendentals, and as a
                    // call on a hypothetical backend that can't (`inline_trans = false`).
                    UnOp::Sin | UnOp::Cos | UnOp::Ln | UnOp::Sqrt => charge(1, fusible, libcalls),
                    UnOp::Atan | UnOp::Round | UnOp::Exp => *libcalls += 1, // still a call everywhere
                    // Cheap fused instructions on every backend (native/wasm floor/ceil/neg).
                    UnOp::Neg | UnOp::Not | UnOp::Sign | UnOp::Floor | UnOp::Ceil => *fusible += 1,
                }
                stack.push(*a);
            }
            RvNode::Binary(BinOp::Pow, a, b) => {
                if const_int_exponent(graph, *b).is_some() {
                    *fusible += 1; // repeated multiply, no call
                    stack.push(*a);
                } else {
                    *libcalls += 1; // pow call
                    stack.push(*a);
                    stack.push(*b);
                }
            }
            // Every non-`Pow` binary op is a single fused instruction on all backends. Named
            // exhaustively (no `_`) so a future `BinOp` must be classified here (finding C6).
            RvNode::Binary(
                BinOp::Add
                | BinOp::Sub
                | BinOp::Mul
                | BinOp::Div
                | BinOp::Mod
                | BinOp::Eq
                | BinOp::Ne
                | BinOp::Lt
                | BinOp::Gt
                | BinOp::Le
                | BinOp::Ge
                | BinOp::And
                | BinOp::Or,
                a,
                b,
            ) => {
                *fusible += 1;
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Select { cond, a, b } => {
                *fusible += 1;
                stack.push(*cond);
                stack.push(*a);
                stack.push(*b);
            }
            // A re-rollable loop (G4c): the interpreter unrolls it, so its codegen support is the
            // interpreter's, not the wasm codegen path — that declines (returns false) and lets the
            // interpreter take the whole cone. v1 Scans wrap a permutation read anyway, which is
            // already interpreter-only, so this changes no backend choice.
            RvNode::Scan { .. } | RvNode::ScanOut { .. } => return false,
            RvNode::Placeholder { .. } => {
                unreachable!("Placeholder appears only inside a ScanBody sub-graph")
            }
        }
    }
    true
}

/// Whether every node in the cone of `root` is something a code generator can emit — after "B2"
/// that is everything except `Poisson` and large non-const `Gather` (see [`gather_class`]). (The
/// backend selector uses [`profitable`], which rejects the same set; this stricter capability
/// check is retained for tests.)
#[cfg(test)]
pub fn supported(graph: &RvGraph, root: RvId) -> bool {
    match graph.node(root) {
        RvNode::Src(Source::Poisson { .. }) => false,
        RvNode::Permutation { .. } | RvNode::Rotation { .. } | RvNode::ArrIndex { .. } => false, // interpreter-only
        RvNode::ArrDraw { .. } => true, // emits nothing; only its ArrElem readers draw
        RvNode::ArrElem { arr, .. } => !matches!(elem_source(graph, *arr), Source::Poisson { .. }),
        RvNode::Gather { elems, index } => match gather_class(graph, elems) {
            Some(GatherClass::ConstTable) => supported(graph, *index),
            Some(GatherClass::SelectChain) => {
                supported(graph, *index) && elems.iter().all(|&e| supported(graph, e))
            }
            None => false,
        },
        RvNode::Src(_) => true,
        RvNode::ConstNum(_) | RvNode::ConstBool(_) | RvNode::Input { .. } => true,
        RvNode::Unary(_, a) => supported(graph, *a),
        RvNode::Binary(_, a, b) => supported(graph, *a) && supported(graph, *b),
        RvNode::Select { cond, a, b } => {
            supported(graph, *cond) && supported(graph, *a) && supported(graph, *b)
        }
        // Interpreter-only: the CPU unrolls a Scan, the wasm codegen path declines it (G4c).
        RvNode::Scan { .. } | RvNode::ScanOut { .. } | RvNode::Placeholder { .. } => false,
    }
}

/// Per-draw cost of a cone: its distinct-node count (`ops`) and how many of those are RNG sources
/// (`sources`). Both walk the cone counting each `RvId` once (matching CSE), so they reflect what a
/// single lane actually evaluates — the playground multiplies them by the draw count for its
/// "operations" / "random numbers" readout. Backend-independent: computed on the simplified graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[must_use = "NodeCost is a pure cost measurement; discarding it makes the cone walk dead work (finding F10)"]
pub struct NodeCost {
    /// Distinct nodes in the cone — one per-lane operation each.
    pub ops: u64,
    /// Distinct RNG source nodes — one random draw per lane each.
    pub sources: u64,
}

/// Compute the [`NodeCost`] of `root`'s cone (see [`NodeCost`]).
pub fn cost(graph: &RvGraph, root: RvId) -> NodeCost {
    cost_roots(graph, &[root])
}

/// Compute the [`NodeCost`] of the *union* of several roots' cones — a joint pass evaluates every
/// distinct node across all roots once per lane, so a shared `seen` set is the right accounting
/// (nodes feeding more than one root are counted once, matching the shared instruction stream the
/// joint drivers compile). Used to price the joint introspection passes (finding B8).
pub fn cost_roots(graph: &RvGraph, roots: &[RvId]) -> NodeCost {
    // Iterative worklist (not recursion) so a deep chain can't overflow the stack (finding A4).
    let mut seen = HashSet::new();
    let mut c = NodeCost::default();
    let mut stack: Vec<RvId> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        c.ops += 1;
        match graph.node(id) {
            // A `Poisson` draw is a Knuth loop of ~`lambda` iterations per lane, not one op — price
            // it realistically (finding A8) so the op budget (`max_opts`) can see it and clamp the
            // draw count, rather than under-charging it at a single op. Capped by the sampler's own
            // `POISSON_KNUTH_MAX` (above which it's the `O(1)` normal approximation).
            RvNode::Src(Source::Poisson { lambda }) => {
                c.sources += 1;
                // `lambda` is finite and > 0 by construction, so `clamp` is safe here.
                let extra = lambda.clamp(0.0, crate::rng::POISSON_KNUTH_MAX) as u64;
                c.ops += extra;
            }
            RvNode::Src(_) => c.sources += 1,
            // The shaped-draw pair. `ArrDraw` is free (it emits nothing — it owns an ordinal block);
            // each `ArrElem` reader is one draw, exactly as the scalar `Src` it replaced. Charging
            // the block's whole `n` here would over-count: an element nobody reads is never drawn.
            RvNode::ArrDraw { .. } => {}
            RvNode::ArrElem { arr, .. } => {
                c.sources += 1;
                stack.push(*arr);
            }
            // A whole-array draw: the Fisher–Yates consumes exactly `n-1` bounded RNG draws per
            // lane (what `sources` means — the playground's "random numbers" readout) and does
            // ~2n per-lane work (n identity writes + n-1 swaps); charge n extra ops on top of the
            // node's own 1 so the op budget (`max_opts`) sees the real cost.
            RvNode::Permutation { n } => {
                c.sources += (*n as u64).saturating_sub(1);
                c.ops += *n as u64;
            }
            // A whole-matrix Haar draw: `d²` normal draws per lane (`sources`) and ~`2d³` flops of
            // native Gaussian-fill + modified Gram–Schmidt (each of the ~d²/2 row projections is a
            // length-`d` dot plus a length-`d` axpy). The work didn't vanish when the graph-level
            // MGS collapsed to one node — it moved into the `Inst::Rotation` loop — so charge it
            // honestly here (consistent with Permutation's charge): under-charging would let
            // `clamp_to_op_budget` admit more draws than the real per-draw cost supports, and the
            // playground's ops readout would flatter rotation-heavy programs.
            RvNode::Rotation { d } => {
                let d = *d as u64;
                c.sources += d * d;
                c.ops += 2 * d * d * d;
            }
            RvNode::ConstNum(_) | RvNode::ConstBool(_) | RvNode::Input { .. } => {}
            RvNode::Unary(_, a) => stack.push(*a),
            RvNode::Binary(_, a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Select { cond, a, b } => {
                stack.push(*cond);
                stack.push(*a);
                stack.push(*b);
            }
            // Priced over elems + index regardless of `gather_class`: this readout reflects what a
            // single interpreter lane evaluates (the elem columns are materialized there even when
            // codegen treats a const table as a leaf), and it must stay backend-independent.
            RvNode::Gather { elems, index } => {
                for &e in elems.iter() {
                    stack.push(e);
                }
                stack.push(*index);
            }
            // One per-lane indexed read (the node's own 1 op) over the array + index cones.
            RvNode::ArrIndex { arr, index } => {
                stack.push(*arr);
                stack.push(*index);
            }
            // A re-rollable loop (G4c): the honest per-lane cost is the body run `trip` times, plus
            // the initial values — exactly what the unrolled form charged, so the op budget (and thus
            // the sample count) is unchanged whether the loop is rolled or not. The body is a separate
            // sub-graph, priced by a recursive walk over its `nexts`.
            RvNode::Scan { body } => {
                // The body cone lives in this same graph, reached via `nexts`; price it once (a
                // recursive walk, placeholders/index counting as leaves) and multiply by `trip`, the
                // honest unrolled cost that keeps the sample budget unchanged whether rolled or not.
                let bc = cost_roots(graph, &body.nexts);
                c.ops += bc.ops.saturating_mul(u64::from(body.trip));
                c.sources += bc.sources.saturating_mul(u64::from(body.trip));
                for &init in body.inits.iter() {
                    stack.push(init);
                }
            }
            RvNode::ScanOut { scan, .. } => stack.push(*scan),
            // A leaf inside a ScanBody (a carried value or the index); the recursive `cost_roots`
            // above walks these when pricing the body.
            RvNode::Placeholder { .. } => {}
        }
    }
    c
}

/// Number of distinct nodes in the cone of `root` (each `RvId` once) — the count of per-stream
/// value slots a stack-machine backend (WASM) must reserve, since it memoizes every node into a
/// local rather than an SSA value.
pub fn cone_size(graph: &RvGraph, root: RvId) -> usize {
    cone_size_roots(graph, &[root])
}

/// Distinct nodes in the *union* of several roots' cones — the value-slot count of a joint kernel
/// (shared nodes get one slot, matching the shared memo the joint emitters use).
pub fn cone_size_roots(graph: &RvGraph, roots: &[RvId]) -> usize {
    // Iterative worklist (not recursion) so a deep chain can't overflow the stack (finding A4).
    let mut seen = HashSet::new();
    let mut stack: Vec<RvId> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match graph.node(id) {
            RvNode::Src(_)
            | RvNode::ConstNum(_)
            | RvNode::ConstBool(_)
            | RvNode::Input { .. }
            | RvNode::Permutation { .. }
            | RvNode::Rotation { .. }
            | RvNode::ArrDraw { .. } => {}
            RvNode::ArrElem { arr, .. } => stack.push(*arr),
            RvNode::Unary(_, a) => stack.push(*a),
            RvNode::Binary(_, a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Select { cond, a, b } => {
                stack.push(*cond);
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Gather { elems, index } => {
                // Must mirror the emitters exactly — this count sizes the wasm local-slot pool. A
                // const-table gather is a leaf over its elems (they are never emitted as nodes and
                // get no value slot), so only the index cone counts.
                if gather_class(graph, elems) != Some(GatherClass::ConstTable) {
                    for &e in elems.iter() {
                        stack.push(e);
                    }
                }
                stack.push(*index);
            }
            // Interpreter-only (the gate rejects these cones before any emitter sizes them), but
            // the walk must stay total.
            RvNode::ArrIndex { arr, index } => {
                stack.push(*arr);
                stack.push(*index);
            }
            // A Scan is interpreter-only (walk_cost declines), so no wasm emitter sizes it; count it
            // as its own node over the initial-value cones. Placeholders live only in the body.
            RvNode::Scan { body } => {
                for &init in body.inits.iter() {
                    stack.push(init);
                }
            }
            RvNode::ScanOut { scan, .. } => stack.push(*scan),
            RvNode::Placeholder { .. } => {}
        }
    }
    seen.len()
}

#[cfg(test)]
mod shaped_tests {
    use super::*;
    use crate::dist::{RvKind, RvNode, Uniform};
    use crate::eval::Engine;

    /// Count the source-ish nodes in an engine's graph after running `src`.
    fn counts(src: &str) -> (usize, usize, usize) {
        let mut eng = Engine::new();
        eng.run(src).expect("program runs");
        let g = eng.graph();
        let (mut srcs, mut draws, mut elems) = (0, 0, 0);
        for i in 0..g.len() {
            match g.node(RvId(i as u32)) {
                RvNode::Src(_) => srcs += 1,
                RvNode::ArrDraw { .. } => draws += 1,
                RvNode::ArrElem { .. } => elems += 1,
                _ => {}
            }
        }
        (srcs, draws, elems)
    }

    /// The structural claim of PLAN-WEBGPU G½, stated as a number: a shaped draw of 52 normals must
    /// put **one** source node in the graph, not 52.
    ///
    /// This is the entire reason the node exists. Shader compile time tracks the *source* count —
    /// each `squares64` inlines ~150 ALU ops in WGSL, at ~6.5 ms of pipeline compile apiece — so 52
    /// sources is 332 ms of cold compile and one is 31 ms (`tools/gpu-spike/RESULTS.md`). If this
    /// test ever reads 52 again, the WGSL emitter has silently lost its loop and the GPU backend is
    /// slower than the CPU it was built to beat.
    #[test]
    fn a_shaped_draw_is_one_source_node_not_n() {
        let (srcs, draws, elems) = counts("use rand;\nzs ~[52] normal(0, 1);\nzs[0]");
        assert_eq!(draws, 1, "expected ONE ArrDraw block");
        assert_eq!(elems, 52, "expected 52 element reads");
        assert_eq!(
            srcs, 0,
            "a shaped draw must push no scalar Src nodes at all"
        );
    }

    /// A `~[d, d]` matrix draw is likewise ONE block of `d²` — not `d` blocks of `d`. (turboquant
    /// draws three of these at `d = 20`: 1200 sources collapsing to 3.)
    #[test]
    fn a_matrix_shaped_draw_is_a_single_block() {
        let (srcs, draws, elems) = counts("use rand;\nm ~[4, 5] normal(0, 1);\nm[0][0]");
        assert_eq!((srcs, draws, elems), (0, 1, 20));
    }

    /// A const-table gather is a LEAF over its elems ([`gather_class`]): a 10k-point `empirical`-shaped
    /// table is materialized as data, not 10k graph nodes, so the counted cone is just `{gather,
    /// index}` — it neither trips `MAX_CODEGEN_NODES` nor blocks the gate. Migrated from the retired
    /// JIT's `const_table_gather_is_profitable_and_matches`; now a backend-independent `kernel` check
    /// (the property is about the cost model, not any one emitter) plus an interpreter distribution
    /// check that the indexed load reproduces uniform-over-the-data.
    #[test]
    fn const_table_gather_is_a_leaf_and_profitable() {
        let mut g = RvGraph::default();
        let elems: Vec<RvId> = (0..10_000)
            .map(|i| g.push(RvNode::ConstNum(i as f64), RvKind::Num))
            .collect();
        let idx = g.push(
            RvNode::Src(Source::UniformInt {
                lo: 0.0,
                hi: 9999.0,
            }),
            RvKind::Num,
        );
        let root = g.push(
            RvNode::Gather {
                elems: elems.into_boxed_slice(),
                index: idx,
            },
            RvKind::Num,
        );
        // The counted cone is just {gather, index}: the 10k elems are the emitted table, not nodes.
        assert_eq!(cone_size(&g, root), 2);
        assert!(supported(&g, root));
        assert!(profitable(&g, root, true, usize::MAX, MIN_DRAWS_WASM));
        // The interpreter's indexed load reproduces uniform-over-the-data: E[table[i]] = 4999.5.
        let m = crate::sampler::moments(&g, root, 200_000, 42).unwrap();
        assert!((m.mean - 4999.5).abs() < 100.0, "mean={}", m.mean);
    }

    /// Poisson keeps the interpreter (Knuth's variable-length per-lane loop is not a fusible scalar
    /// expression): the gate must decline it, and the interpreter fallback must still draw correctly
    /// (mean ≈ lambda). Migrated from the retired JIT's `poisson_still_falls_back_to_interpreter`.
    #[test]
    fn poisson_is_unsupported_but_samples_correctly() {
        let mut g = RvGraph::default();
        let root = g.push(RvNode::Src(Source::Poisson { lambda: 3.0 }), RvKind::Num);
        assert!(!supported(&g, root), "Poisson must stay interpreter-only");
        let m = crate::sampler::moments(&g, root, 200_000, 7).unwrap();
        assert!((m.mean - 3.0).abs() < 0.05, "fallback mean = {}", m.mean);
    }

    /// Ordinals are handed out by walking node ids in order: a scalar source takes one, a shaped
    /// draw takes a contiguous block of `n`. That contiguity is what lets the WGSL emitter turn a
    /// block into `for (j) { draw(base + j) }`, and what makes element `k` draw exactly the stream
    /// the `k`-th independent `Src` used to draw.
    #[test]
    fn source_ordinals_hand_a_shaped_draw_a_contiguous_block() {
        let mut g = RvGraph::default();
        let a = g.push(
            RvNode::Src(Source::Normal {
                mu: 0.0,
                sigma: 1.0,
            }),
            RvKind::Num,
        );
        let blk = g.push(
            RvNode::ArrDraw {
                n: 4,
                src: Source::Uniform(Uniform { lo: 0.0, hi: 1.0 }),
            },
            RvKind::Arr(4),
        );
        let _e2 = g.push(RvNode::ArrElem { arr: blk, k: 2 }, RvKind::Num);
        let z = g.push(RvNode::ConstNum(1.0), RvKind::Num);
        let b = g.push(
            RvNode::Src(Source::Normal {
                mu: 0.0,
                sigma: 1.0,
            }),
            RvKind::Num,
        );

        let ords = source_ordinals(&g);
        assert_eq!(
            ords[a.0 as usize], 0,
            "the first scalar source takes ordinal 0"
        );
        assert_eq!(ords[blk.0 as usize], 1, "the block's base follows it");
        assert_eq!(
            elem_ordinal(&ords, blk, 2),
            3,
            "element k draws from base + k"
        );
        assert_eq!(ords[z.0 as usize], NO_SOURCE, "a constant draws nothing");
        assert_eq!(
            ords[b.0 as usize], 5,
            "the next scalar source resumes AFTER the whole block (1 + 4), so no two sources can \
             ever share a counter"
        );
    }
}
