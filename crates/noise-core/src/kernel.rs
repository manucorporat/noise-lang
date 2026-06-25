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
//!     interpreter (the "B2" gate; reused verbatim by the WASM backend, where a kernel's `ln`/`cos`
//!     are also non-inlined calls — see PLAN.md "Browser note").
//!   * [`const_int_exponent`] — the `x ** k` small-integer-power test (repeated multiply vs a `pow`
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

const _: () = assert!(BATCH.is_multiple_of(STREAMS), "BATCH must be a multiple of STREAMS");

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
/// Such `x ** k` lower to repeated multiply (no `pow` call) on every backend.
pub fn const_int_exponent(graph: &RvGraph, id: RvId) -> Option<u32> {
    match graph.node(id) {
        RvNode::ConstNum(x) if x.fract() == 0.0 && *x >= 0.0 && *x <= 64.0 => Some(*x as u32),
        _ => None,
    }
}

/// Whether codegen is expected to outperform the interpreter for this graph: the graph is supported
/// (no `Poisson` — its Knuth loop stays interpreter-only) **and** the count of fused nodes strictly
/// exceeds the transcendental-call weight. See [`walk_cost`] for the calibration. Used as the gate
/// by both the native JIT and the WASM emitter.
pub fn profitable(graph: &RvGraph, root: RvId) -> bool {
    let mut seen = HashSet::new();
    let (mut fusible, mut libcalls) = (0u32, 0u32);
    if walk_cost(graph, root, &mut seen, &mut fusible, &mut libcalls) {
        fusible > libcalls
    } else {
        false // unsupported (Poisson) → interpreter
    }
}

/// Accumulate `(fusible, libcalls)` weights over the distinct nodes of `id`'s cone (each `RvId`
/// counted once, matching CSE). Returns `false` if the cone contains an unsupported node.
///
/// Calibrated against `benches/sampling.rs`: per-draw `ln`/`cos`/`pow` are non-inlined calls and
/// (for `normal`) do ~2× the transcendental work of the interpreter's pair-sharing column fill, so
/// a transcendental-bound graph is faster interpreted while an arithmetic-dominated one is faster
/// fused. `fusible > libcalls` puts the crossover between those measured cases.
pub fn walk_cost(
    graph: &RvGraph,
    id: RvId,
    seen: &mut HashSet<RvId>,
    fusible: &mut u32,
    libcalls: &mut u32,
) -> bool {
    if !seen.insert(id) {
        return true; // shared node already counted
    }
    match graph.node(id) {
        RvNode::Src(Source::Poisson { .. }) => false,
        // A normal costs two calls (ln + cos) and draws two uniforms per lane — the heaviest.
        RvNode::Src(Source::Normal { .. }) => {
            *libcalls += 2;
            true
        }
        RvNode::Src(Source::Exp { .. }) | RvNode::Src(Source::Geometric { .. }) => {
            *libcalls += 1;
            true
        }
        RvNode::Src(_) => {
            *fusible += 1; // uniform / uniform_int: cheap inline draw
            true
        }
        RvNode::ConstNum(_) | RvNode::ConstBool(_) => true, // neutral
        RvNode::Unary(op, a) => {
            if matches!(op, UnOp::Sin | UnOp::Cos | UnOp::Atan | UnOp::Round) {
                *libcalls += 1;
            } else {
                *fusible += 1;
            }
            walk_cost(graph, *a, seen, fusible, libcalls)
        }
        RvNode::Binary(crate::ast::BinOp::Pow, a, b) => {
            if const_int_exponent(graph, *b).is_some() {
                *fusible += 1; // repeated multiply, no call
                walk_cost(graph, *a, seen, fusible, libcalls)
            } else {
                *libcalls += 1; // pow call
                walk_cost(graph, *a, seen, fusible, libcalls)
                    && walk_cost(graph, *b, seen, fusible, libcalls)
            }
        }
        RvNode::Binary(_, a, b) => {
            *fusible += 1;
            walk_cost(graph, *a, seen, fusible, libcalls)
                && walk_cost(graph, *b, seen, fusible, libcalls)
        }
        RvNode::Select { cond, a, b } => {
            *fusible += 1;
            walk_cost(graph, *cond, seen, fusible, libcalls)
                && walk_cost(graph, *a, seen, fusible, libcalls)
                && walk_cost(graph, *b, seen, fusible, libcalls)
        }
    }
}

/// Pick the RNG stream count for a graph. Multi-stream pays off only when the kernel is bound by the
/// xoshiro latency chain — i.e. pure inline arithmetic/draws. A transcendental call (`ln`/`cos`/
/// `pow`) doesn't overlap across streams and the extra states add register pressure, so
/// call-bearing graphs measured *slower* multi-stream; those stay single-stream.
pub fn choose_streams(graph: &RvGraph, root: RvId) -> usize {
    let mut seen = HashSet::new();
    let (mut fusible, mut libcalls) = (0u32, 0u32);
    if walk_cost(graph, root, &mut seen, &mut fusible, &mut libcalls) && libcalls == 0 {
        STREAMS
    } else {
        1
    }
}

/// Whether every node in the cone of `root` is something a code generator can emit — after "B2"
/// that is everything except `Poisson`. (The backend selector uses [`profitable`], which also
/// rejects Poisson; this stricter capability check is retained for tests.)
#[cfg(test)]
pub fn supported(graph: &RvGraph, root: RvId) -> bool {
    match graph.node(root) {
        RvNode::Src(Source::Poisson { .. }) => false,
        RvNode::Src(_) => true,
        RvNode::ConstNum(_) | RvNode::ConstBool(_) => true,
        RvNode::Unary(_, a) => supported(graph, *a),
        RvNode::Binary(_, a, b) => supported(graph, *a) && supported(graph, *b),
        RvNode::Select { cond, a, b } => {
            supported(graph, *cond) && supported(graph, *a) && supported(graph, *b)
        }
    }
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
        }
    }
    let mut seen = HashSet::new();
    go(graph, root, &mut seen);
    seen.len()
}
