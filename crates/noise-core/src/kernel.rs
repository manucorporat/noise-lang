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
//!     interpreter (the "B2" gate). One cost function, parameterized by `inline_trans`: the native
//!     JIT inlines `ln`/`sin`/`cos` (passes `true`), the WASM emitter still imports them from the
//!     host (passes `false`) — see PLAN.md "Browser note".
//!   * [`const_int_exponent`] — the `x ^ k` small-integer-power test (repeated multiply vs a `pow`
//!     call), shared so both backends agree on which exponents fuse.

use std::collections::HashSet;

use crate::ast::UnOp;
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
/// `inline_trans` says whether *this backend* inlines `ln`/`sin`/`cos` as polynomials. The native
/// JIT does ([`crate::approx`] / `jit::emit_ln`), so it passes `true` and a `normal`/`exp`/trig
/// graph counts as fusible. The WASM emitter still imports those from the host (a per-draw call
/// across the JS boundary — even costlier than a native libcall), so it passes `false` and such
/// graphs stay on the interpreter, exactly as before. `atan`/`round`/non-integer `pow` are calls on
/// both backends regardless.
pub fn profitable(graph: &RvGraph, root: RvId, inline_trans: bool) -> bool {
    let mut seen = HashSet::new();
    let (mut fusible, mut libcalls) = (0u32, 0u32);
    if walk_cost(
        graph,
        root,
        &mut seen,
        &mut fusible,
        &mut libcalls,
        inline_trans,
    ) {
        fusible > libcalls
    } else {
        false // unsupported (Poisson) → interpreter
    }
}

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
    if !seen.insert(id) {
        return true; // shared node already counted
    }
    match graph.node(id) {
        RvNode::Src(Source::Poisson { .. }) => false, // Knuth loop stays interpreter-only
        RvNode::Gather { .. } => false, // data-dependent addressing stays interpreter-only
        RvNode::Src(Source::Normal { .. }) => {
            charge(2, fusible, libcalls);
            true
        }
        RvNode::Src(Source::Exp { .. }) | RvNode::Src(Source::Geometric { .. }) => {
            charge(1, fusible, libcalls);
            true
        }
        RvNode::Src(_) => {
            *fusible += 1; // uniform / uniform_int: cheap inline draw on every backend
            true
        }
        RvNode::ConstNum(_) | RvNode::ConstBool(_) => true, // neutral
        RvNode::Unary(op, a) => {
            match op {
                UnOp::Sin | UnOp::Cos | UnOp::Ln => charge(1, fusible, libcalls), // inlined on native
                UnOp::Atan | UnOp::Round | UnOp::Exp => *libcalls += 1, // still a call everywhere
                _ => *fusible += 1,
            }
            walk_cost(graph, *a, seen, fusible, libcalls, inline_trans)
        }
        RvNode::Binary(crate::ast::BinOp::Pow, a, b) => {
            if const_int_exponent(graph, *b).is_some() {
                *fusible += 1; // repeated multiply, no call
                walk_cost(graph, *a, seen, fusible, libcalls, inline_trans)
            } else {
                *libcalls += 1; // pow call
                walk_cost(graph, *a, seen, fusible, libcalls, inline_trans)
                    && walk_cost(graph, *b, seen, fusible, libcalls, inline_trans)
            }
        }
        RvNode::Binary(_, a, b) => {
            *fusible += 1;
            walk_cost(graph, *a, seen, fusible, libcalls, inline_trans)
                && walk_cost(graph, *b, seen, fusible, libcalls, inline_trans)
        }
        RvNode::Select { cond, a, b } => {
            *fusible += 1;
            walk_cost(graph, *cond, seen, fusible, libcalls, inline_trans)
                && walk_cost(graph, *a, seen, fusible, libcalls, inline_trans)
                && walk_cost(graph, *b, seen, fusible, libcalls, inline_trans)
        }
    }
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
    let mut seen = HashSet::new();
    if latency_bound(graph, root, &mut seen) {
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
    if !seen.insert(id) {
        return true; // shared node already checked
    }
    match graph.node(id) {
        RvNode::Src(Source::Uniform(_) | Source::UniformInt { .. }) => true,
        RvNode::Src(_) => false, // normal / exp / geometric / poisson: not latency-bound
        RvNode::Gather { .. } => false, // interpreter-only (gated out before this is consulted)
        RvNode::ConstNum(_) | RvNode::ConstBool(_) => true,
        RvNode::Unary(op, a) => {
            !matches!(
                op,
                UnOp::Sin | UnOp::Cos | UnOp::Atan | UnOp::Round | UnOp::Exp | UnOp::Ln
            ) && latency_bound(graph, *a, seen)
        }
        RvNode::Binary(crate::ast::BinOp::Pow, a, b) => {
            const_int_exponent(graph, *b).is_some()
                && latency_bound(graph, *a, seen)
                && latency_bound(graph, *b, seen)
        }
        RvNode::Binary(_, a, b) => latency_bound(graph, *a, seen) && latency_bound(graph, *b, seen),
        RvNode::Select { cond, a, b } => {
            latency_bound(graph, *cond, seen)
                && latency_bound(graph, *a, seen)
                && latency_bound(graph, *b, seen)
        }
    }
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
pub struct NodeCost {
    /// Distinct nodes in the cone — one per-lane operation each.
    pub ops: u64,
    /// Distinct RNG source nodes — one random draw per lane each.
    pub sources: u64,
}

/// Compute the [`NodeCost`] of `root`'s cone (see [`NodeCost`]).
pub fn cost(graph: &RvGraph, root: RvId) -> NodeCost {
    fn go(graph: &RvGraph, id: RvId, seen: &mut HashSet<RvId>, c: &mut NodeCost) {
        if !seen.insert(id) {
            return;
        }
        c.ops += 1;
        match graph.node(id) {
            RvNode::Src(_) => c.sources += 1,
            RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
            RvNode::Unary(_, a) => go(graph, *a, seen, c),
            RvNode::Binary(_, a, b) => {
                go(graph, *a, seen, c);
                go(graph, *b, seen, c);
            }
            RvNode::Select { cond, a, b } => {
                go(graph, *cond, seen, c);
                go(graph, *a, seen, c);
                go(graph, *b, seen, c);
            }
            RvNode::Gather { elems, index } => {
                for &e in elems.iter() {
                    go(graph, e, seen, c);
                }
                go(graph, *index, seen, c);
            }
        }
    }
    let mut seen = HashSet::new();
    let mut c = NodeCost::default();
    go(graph, root, &mut seen, &mut c);
    c
}

/// Number of distinct nodes in the cone of `root` (each `RvId` once) — the count of per-stream
/// value slots a stack-machine backend (WASM) must reserve, since it memoizes every node into a
/// local rather than an SSA value.
pub fn cone_size(graph: &RvGraph, root: RvId) -> usize {
    fn go(graph: &RvGraph, id: RvId, seen: &mut HashSet<RvId>) {
        if !seen.insert(id) {
            return;
        }
        match graph.node(id) {
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
            RvNode::Unary(_, a) => go(graph, *a, seen),
            RvNode::Binary(_, a, b) => {
                go(graph, *a, seen);
                go(graph, *b, seen);
            }
            RvNode::Select { cond, a, b } => {
                go(graph, *cond, seen);
                go(graph, *a, seen);
                go(graph, *b, seen);
            }
            RvNode::Gather { elems, index } => {
                for &e in elems.iter() {
                    go(graph, e, seen);
                }
                go(graph, *index, seen);
            }
        }
    }
    let mut seen = HashSet::new();
    go(graph, root, &mut seen);
    seen.len()
}
