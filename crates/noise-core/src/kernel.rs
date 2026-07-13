//! Backend-agnostic kernel support shared by the code generators (PLAN.md Phase 4).
//!
//! The native Cranelift JIT ([`crate::jit`]) and the WASM emitter ([`crate::wasm_emit`]) are two
//! *lowerings* of the same [`RvGraph`] IR: they walk the identical graph the identical way and emit
//! the identical fused, multi-stream kernel — only the per-node instruction encoding differs
//! (`mulsd` vs `f64.mul`, an inlined xoshiro step in either). Everything that is about *what the
//! graph means* rather than *how to spell it on a target* lives here, so there is exactly one copy:
//!
//!   * [`STREAMS`] / [`choose_streams`] — the multi-stream RNG policy.
//!   * [`seed_state`] — SplitMix64 expansion of a seed into the per-stream xoshiro state layout.
//!   * [`profitable`] / [`walk_cost`] — the cost model deciding whether codegen beats the
//!     interpreter (the "B2" gate). One cost function, parameterized by `inline_trans`: **both**
//!     backends now inline `ln`/`sin`/`cos` as polynomials and pass `true` — the native JIT via
//!     `jit::emit_ln`/`emit_trig`, the WASM emitter via `wasm_emit::emit_ln`/`emit_trig`. The
//!     `false` branch is a retained seam for a hypothetical backend that couldn't inline them.
//!     `exp` and the large-argument trig fallback remain real calls on both (a host import on wasm,
//!     an `nz_*` shim on the JIT) — see PLAN.md "Browser note" and findings C3/C9.
//!   * [`const_int_exponent`] — the `x ^ k` small-integer-power test (repeated multiply vs a `pow`
//!     call), shared so both backends agree on which exponents fuse.

// Backend-support helpers whose live set depends on the build config (`--features jit`, wasm
// target, tests). This module was previously `pub`, which masked the same dead-code warnings.
#![allow(dead_code)]

use std::collections::HashSet;

use crate::ast::{BinOp, UnOp};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId, RvNode, Source};

/// Number of independent xoshiro streams a kernel interleaves (samples emitted per loop iteration).
/// xoshiro256++ is a serial dependency chain threaded through the whole loop, so on RNG-bound graphs
/// that latency — not the arithmetic — is the ceiling; running four independent states lets the
/// out-of-order core (native) or the host engine overlap the chains for ~2×. Four keeps register
/// pressure modest; past four the gain flattens (see `jit::bench_streams`). Must divide [`BATCH`].
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

/// If node `id` is a constant non-negative integer in `0..=64`, return it as an exponent count.
/// Such `x ^ k` lower to repeated multiply (no `pow` call) on every backend.
pub fn const_int_exponent(graph: &RvGraph, id: RvId) -> Option<u32> {
    match graph.node(id) {
        RvNode::ConstNum(x) if x.fract() == 0.0 && *x >= 0.0 && *x <= 64.0 => Some(*x as u32),
        _ => None,
    }
}

/// Whether codegen is expected to outperform the interpreter for this graph: the graph is supported
/// (no `Poisson` — its Knuth loop stays interpreter-only) **and** the count of fused nodes strictly
/// exceeds the transcendental-call weight. See [`walk_cost`] for the calibration.
///
/// `inline_trans` says whether *this backend* inlines `ln`/`sin`/`cos` as polynomials. **Both**
/// backends do — the native JIT via [`crate::approx`] / `jit::emit_ln`, the WASM emitter via
/// `wasm_emit::emit_ln`/`emit_trig` — so both pass `true` and a `normal`/`exp`/trig graph counts as
/// fusible on each. The `false` branch is a retained seam: a backend that *couldn't* inline a
/// transcendental would pass `false` and the gate would correctly leave those graphs to the
/// interpreter rather than emit a per-draw call. `atan`/`round`/`exp`/non-integer `pow` (and the
/// rare large-|x| trig fallback, finding C3) are real calls on both backends regardless.
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
            return false; // unsupported (Poisson / Gather) → interpreter
        }
    }
    // Route very large cones to the interpreter. The code generators (`jit::emit_node`,
    // `wasm_emit::emit_node`) emit each node **recursively**, so a hundreds-of-thousands-deep graph
    // (`cumsum(~[200000] …)`) would overflow their emitters and abort (finding A4). The interpreter's
    // lowering is now iterative (stack-safe at any depth), and JIT-compiling 10^4+ nodes rarely beats
    // interpreting them, so this only ever trades a little speed for safety. `seen` is the cone's
    // distinct-node count (walk_cost just filled it).
    if seen.len() > MAX_CODEGEN_NODES {
        return false;
    }
    if fusible <= libcalls {
        return false;
    }
    // Codegen has to *pay for itself*. Everything above asks whether the fused kernel is faster per
    // draw; this asks whether the query takes enough draws to refund *compiling* it.
    draws >= min_draws
}

/// Minimum draws before the **Cranelift JIT** is worth compiling (see [`profitable`]'s `min_draws`).
///
/// Compiling is a fixed cost paid once per forcing; fusion is a saving earned per draw, so a query
/// that draws few samples never earns its compile back. That is not a corner case — real programs
/// force many small queries: `examples/noise_colors.noise` forces 14 at 3,000 draws each,
/// `examples/turboquant.noise` forces 10 at 10,000. Without this, both ran *slower* with the JIT on
/// (0.22× and 0.79× — on `noise_colors` the JIT was a 4.5× pessimization).
///
/// **One constant, not a cost curve.** `jit::tests::bench_amortization` measures the true break-even
/// per cone size (13k–75k draws over a 6→1,536-node sweep) and a fitted `f(nodes)` was tried, but it
/// decides every program in `examples/` exactly as this single threshold does — complexity with no
/// behavior to show for it. The corpus separates cleanly: losers draw ≤40k per forcing, winners
/// ≥175k. If a graph class ever lands in that gap, the bench is there to justify a curve then.
pub const MIN_DRAWS_JIT: usize = 100_000;

/// Minimum draws before the **WASM emitter** is worth emitting (see [`profitable`]'s `min_draws`).
///
/// An order of magnitude below [`MIN_DRAWS_JIT`], because the costs it amortizes are an order of
/// magnitude apart: `WebAssembly.instantiate` on a ~1 KB kernel is far cheaper than a Cranelift
/// compile. The two thresholds are the *same rule* with each backend's own measured constant — not
/// two heuristics.
///
/// Measured with `packages/core/bench/examples.mjs` (V8, real programs): reusing the native 100k here
/// left real wins on the table — `am_vs_fm` (40k draws/forcing) ran 1.20× emitted but 0.94× gated,
/// and `turboquant` (10k) 1.06× vs 0.90×. Dropping to 10k keeps both emitting while still declining
/// `noise_colors` (3k draws/forcing), which is 0.31× — a 3× pessimization — when always emitted.
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
            RvNode::Gather { .. } => return false, // data-dependent addressing stays interpreter-only
            RvNode::Src(Source::Normal { .. }) => charge(2, fusible, libcalls),
            RvNode::Src(Source::Exp { .. }) | RvNode::Src(Source::Geometric { .. }) => {
                charge(1, fusible, libcalls)
            }
            // uniform / uniform_int: cheap inline draw everywhere.
            RvNode::Src(Source::Uniform(_) | Source::UniformInt { .. }) => *fusible += 1,
            RvNode::ConstNum(_) | RvNode::ConstBool(_) => {} // neutral
            RvNode::Unary(op, a) => {
                match op {
                    UnOp::Sin | UnOp::Cos | UnOp::Ln => charge(1, fusible, libcalls), // inlined on both backends
                    UnOp::Atan | UnOp::Round | UnOp::Exp => *libcalls += 1, // still a call everywhere
                    // Cheap fused instructions on every backend (native/wasm floor/ceil/neg, etc.).
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
        }
    }
    true
}

/// Pick the RNG stream count for a graph. Multi-stream pays off only when the kernel is bound by the
/// **xoshiro latency chain** — pure inline arithmetic over `uniform`/`uniform_int` draws (`pi`,
/// `dice`) — because independent streams let the out-of-order core overlap the otherwise-serial
/// chains (~2×, see `jit::bench_streams`).
///
/// Transcendental draws/ufuncs are different: even inlined (`ln`/`sin`/`cos` polynomials), they are
/// *arithmetic-throughput*-bound, not latency-bound — the execution units are already saturated, so
/// adding streams just multiplies the work with nothing to overlap (measured flat `s1≈s4`, with
/// worse register pressure on large kernels). Those — and any remaining real call (`atan`/`round`/
/// non-integer `pow`) — stay single-stream. So the rule is "multi-stream iff latency-bound".
pub fn choose_streams(graph: &RvGraph, root: RvId) -> usize {
    choose_streams_roots(graph, &[root])
}

/// Multi-root [`choose_streams`]: one stream policy for a joint kernel — multi-stream only if
/// *every* root's cone is latency-bound (any transcendental anywhere makes the whole loop body
/// throughput-bound, exactly as it would in a single fused cone).
pub fn choose_streams_roots(graph: &RvGraph, roots: &[RvId]) -> usize {
    let mut seen = HashSet::new();
    if roots.iter().all(|&r| latency_bound(graph, r, &mut seen)) {
        STREAMS
    } else {
        1
    }
}

/// Whether `root`'s cone is purely the xoshiro-latency-bound regime: only `uniform`/`uniform_int`
/// sources and plain fusible arithmetic — no transcendental draw (`normal`/`exp`/`geometric`) or
/// ufunc (`sin`/`cos`/`atan`/`round`) and no non-integer `pow`. (`Poisson` returns `false` here too,
/// but the backend selector rejects it earlier via [`profitable`].)
fn latency_bound(graph: &RvGraph, id: RvId, seen: &mut HashSet<RvId>) -> bool {
    // Iterative worklist (not recursion) so a deep chain can't overflow the stack (finding A4).
    // Returns `false` on the first node that breaks the latency-bound predicate.
    let mut stack = vec![id];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue; // shared node already checked
        }
        // Exhaustive (no `_`) so a future op is force-classified here too (finding C6); see the
        // `TODO(CostClass)` in `walk_cost`.
        match graph.node(id) {
            RvNode::Src(Source::Uniform(_) | Source::UniformInt { .. }) => {}
            // normal / exp / geometric / poisson: transcendental draw → throughput-bound.
            RvNode::Src(
                Source::Normal { .. }
                | Source::Exp { .. }
                | Source::Geometric { .. }
                | Source::Poisson { .. },
            ) => return false,
            RvNode::Gather { .. } => return false, // interpreter-only (gated out before this)
            RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
            RvNode::Unary(op, a) => {
                match op {
                    // Transcendental/ufunc draws are arithmetic-throughput-bound, not latency-bound.
                    UnOp::Sin | UnOp::Cos | UnOp::Atan | UnOp::Round | UnOp::Exp | UnOp::Ln => {
                        return false
                    }
                    // Plain fused instructions keep the cone latency-bound.
                    UnOp::Neg | UnOp::Not | UnOp::Sign | UnOp::Floor | UnOp::Ceil => {}
                }
                stack.push(*a);
            }
            RvNode::Binary(BinOp::Pow, a, b) => {
                if const_int_exponent(graph, *b).is_none() {
                    return false;
                }
                stack.push(*a);
                stack.push(*b);
            }
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
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Select { cond, a, b } => {
                stack.push(*cond);
                stack.push(*a);
                stack.push(*b);
            }
        }
    }
    true
}

/// Whether every node in the cone of `root` is something a code generator can emit — after "B2"
/// that is everything except `Poisson`. (The backend selector uses [`profitable`], which also
/// rejects Poisson; this stricter capability check is retained for tests.)
#[cfg(test)]
pub fn supported(graph: &RvGraph, root: RvId) -> bool {
    match graph.node(root) {
        RvNode::Src(Source::Poisson { .. }) => false,
        RvNode::Gather { .. } => false,
        RvNode::Src(_) => true,
        RvNode::ConstNum(_) | RvNode::ConstBool(_) => true,
        RvNode::Unary(_, a) => supported(graph, *a),
        RvNode::Binary(_, a, b) => supported(graph, *a) && supported(graph, *b),
        RvNode::Select { cond, a, b } => {
            supported(graph, *cond) && supported(graph, *a) && supported(graph, *b)
        }
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
            RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
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
                for &e in elems.iter() {
                    stack.push(e);
                }
                stack.push(*index);
            }
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
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
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
                for &e in elems.iter() {
                    stack.push(e);
                }
                stack.push(*index);
            }
        }
    }
    seen.len()
}
