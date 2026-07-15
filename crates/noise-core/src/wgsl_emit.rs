//! WGSL emitter — the **fourth** lowering of the `RvGraph` (PLAN-WEBGPU G1).
//!
//! Same walk as [`crate::wasm_emit`]: post-order over the simplified cone, one
//! memoized value per node, one column per root. Three things make it *not* a port of those two, and
//! all three come from measurements in `tools/gpu-spike/RESULTS.md` (G0):
//!
//! 1. **`squares64` is emulated on `vec2<u32>`** — WGSL has no `u64`. It reproduces the engine's
//!    draws bit for bit (verified, 4096/4096 lanes), so C0's certification carries onto the GPU
//!    unchanged. Two structural gifts keep it affordable: `rotate_left(x, 32)` on a u64 is a free
//!    half-swap (`x.yx`), and a *wrapping* 64×64 product needs only one wide partial product.
//!
//! 2. **Shaped draws are emitted as a LOOP, not unrolled.** This is the whole reason
//!    [`RvNode::ArrDraw`] exists. Because the hash is emulated, each RNG source inlines ~150 ALU ops,
//!    so shader *compile* time tracks the source count at ~6.5 ms apiece. Unrolled,
//!    `barrier_option`'s 52 weekly normals cost **332 ms** of cold pipeline compile and the GPU
//!    *loses* to the multicore CPU backend end to end; as one draw loop over a block of consecutive ordinals
//!    it costs **31 ms** and wins. The other backends have no such problem and keep unrolling.
//!
//! 3. **The transcendentals are WGSL's built-ins, not `approx.rs`'s polynomials.** The GPU *fuses
//!    multiply-add* (measured: 4095/4096 lanes match a fused result; WGSL permits it and there is no
//!    portable off switch), so bit-identical f32 *arithmetic* with the CPU backends is impossible no
//!    matter what we emit. The polynomial therefore buys no bitwise parity — and measured, no better
//!    accuracy either (both land at 7.15e-7 max deviation on `normal`) — at 1.4× the cost. So the
//!    conformance contract is two-tier, and it is the tier that matters that holds:
//!
//!    | tier | what | claim |
//!    |---|---|---|
//!    | 1 | the draws (integer hash → 24-bit uniforms) | **bit-identical** to every other backend |
//!    | 2 | everything computed from them in f32 | **ULP-close** — ≤1e-6 absolute |
//!
//! Ops whose CPU semantics differ from WGSL's built-in of the same name (`round` is ties-**away**
//! here but ties-to-even in WGSL; `%` is **floored** here but truncated in WGSL) are lowered
//! explicitly — those would disagree by a *whole unit*, which is not a rounding difference and would
//! not be caught by a ULP bound.

// Nothing in the library *calls* this yet: the reduce-driver integration and the profitability gate
// are G2, and until they land the emitter is reachable only from its own tests and `tools/gpu-spike`.
// That is deliberate — G1 is "the emitter is correct", G2 is "the engine uses it" — but it does mean
// the whole module reads as dead code to the compiler until then.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::ast::{BinOp, UnOp};
use crate::dist::{RvGraph, RvId, RvKind, RvNode, Source};
use crate::kernel::{const_int_exponent, elem_ordinal, elem_source, source_ordinals};

/// Invocations per workgroup. 64 = two SIMD groups on Apple silicon and a safe divisor everywhere.
pub const WORKGROUP: u32 = 64;

/// Least number of element reads of one [`RvNode::ArrDraw`] block that earns a draw **loop**.
///
/// Below it the loop's own overhead (and the fact that it must draw the *whole* block, including
/// elements nobody reads) costs more than just inlining the few draws that are wanted. Above it, the
/// compile-time collapse is worth far more than any of that — 8 sources is already ~50 ms of cold
/// pipeline compile unrolled.
const LOOP_MIN_READERS: usize = 8;

/// A cone this emitter can't lower. The caller falls back exactly as `wasm_host` does — to a correct,
/// slower backend, never to an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

/// Emit a compute shader producing one f32 column per root: `out[j * n + i]` is root `j`, lane `i`.
///
/// `graph` must already be simplified (the caller does that, so the cache key and the ordinals agree
/// with every other backend — see [`source_ordinals`]).
pub fn emit(graph: &RvGraph, roots: &[RvId]) -> Result<String, Unsupported> {
    let ords = source_ordinals(graph);
    let stream_ords = crate::kernel::cell_stream_ordinals(graph);
    let plan = plan_blocks(graph, roots)?;
    let mut e = Emitter {
        graph,
        ords: &ords,
        stream_ords: &stream_ords,
        plan,
        memo: HashMap::new(),
        arrays_emitted: HashSet::new(),
        taint: crate::simplify::taint_set(graph),
        scan_finals: HashMap::new(),
        body: String::new(),
    };

    // Draw loops first: a block's elements must exist before any expression reads them.
    let mut blocks: Vec<RvId> = e.plan.looped.iter().copied().collect();
    blocks.sort_by_key(|id| id.0); // deterministic output — the shader text is a cache key
    for arr in blocks {
        e.emit_draw_loop(arr);
    }

    let mut cols = String::new();
    for (j, &root) in roots.iter().enumerate() {
        let v = e.emit_node(root)?;
        let _ = writeln!(cols, "    out[{j}u * P.n + i] = {v};");
    }

    Ok(format!(
        "{PRELUDE}\n@compute @workgroup_size({WORKGROUP})\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
         \x20   let i = gid.x;\n\
         \x20   if (i >= P.n) {{ return; }}\n\
         \x20   let lane = P.lane0 + i;\n\
         \x20   let key = P.key;\n\
         {}{cols}}}\n",
        e.body
    ))
}

/// Which [`RvNode::ArrDraw`] blocks get a draw loop, and which get their reads inlined.
struct Plan {
    /// Blocks emitted as `var a{id}: array<f32, n>` filled by one loop.
    looped: HashSet<RvId>,
}

/// Decide the loop/inline split, and reject any cone this backend can't lower.
///
/// A block earns a loop when enough of it is actually read ([`LOOP_MIN_READERS`], and at least half
/// the block — a `~[10000]` draw whose program only reads `zs[0]` must not dispatch 10,000 draws per
/// lane to save a few milliseconds of compile).
fn plan_blocks(graph: &RvGraph, roots: &[RvId]) -> Result<Plan, Unsupported> {
    let mut readers: HashMap<RvId, usize> = HashMap::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<RvId> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match graph.node(id) {
            // Still out of scope: `Poisson`'s Knuth loop counts f64 uniforms in the interpreter and
            // the count is data-dependent — a divergent loop with an f64 contract this backend can't
            // reproduce. Stays interpreter-only.
            RvNode::Src(Source::Poisson { .. }) => return Err(Unsupported("poisson")),
            RvNode::ArrDraw { src, .. } => {
                if matches!(src, Source::Poisson { .. }) {
                    return Err(Unsupported("poisson"));
                }
            }
            // Array-valued sources, both leaves here (their only inputs are key/lane/stream).
            // `Permutation` is integer (Fisher–Yates) and lowers bit-identically. `Rotation` fills
            // f32 normals and Gram–Schmidts them in a bounded loop (G4b): the interpreter does the
            // same in f64, so the *matrix* is not bit-identical — but it is a valid Haar rotation, and
            // the programs that draw it (turboquant) are distributional, so this backend takes them
            // under a distribution-level contract rather than the lane-for-lane one. Both compile to a
            // small shader because the work is a *loop*, not unrolled.
            RvNode::Permutation { .. } | RvNode::Rotation { .. } => {}
            // `ArrIndex` reads an array-valued source; `Gather` reads a table of scalar nodes. Both
            // push their operands so the walk reaches them — which is also how an `ArrIndex` over a
            // (declined) `Rotation` correctly declines the whole cone.
            RvNode::ArrIndex { arr, index } => {
                stack.push(*arr);
                stack.push(*index);
            }
            RvNode::Gather { elems, index } => {
                for &e in elems.iter() {
                    stack.push(e);
                }
                stack.push(*index);
            }
            RvNode::ArrElem { arr, .. } => {
                *readers.entry(*arr).or_default() += 1;
                stack.push(*arr);
            }
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
            // Host input uniforms lower to a WGSL uniform read in P1; for now the GPU declines any
            // cone that carries one and falls back to the interpreter (PLAN-UNIFORM-INPUTS P0).
            RvNode::Input { .. } => return Err(Unsupported("input")),
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
            // G4c: a Scan rolls into a real WGSL loop. Walk its body (inits + the nexts cone) so any
            // unsupported node in the loop declines the whole cone; placeholders are leaves.
            RvNode::Scan { body } => {
                for &init in body.inits.iter() {
                    stack.push(init);
                }
                for &nx in body.nexts.iter() {
                    stack.push(nx);
                }
            }
            RvNode::ScanOut { scan, .. } => stack.push(*scan),
            RvNode::Placeholder { .. } => {}
        }
    }

    let looped = readers
        .iter()
        .filter(|(&arr, &count)| {
            let RvNode::ArrDraw { n, .. } = graph.node(arr) else {
                unreachable!("readers are keyed by ArrDraw")
            };
            count >= LOOP_MIN_READERS && count * 2 >= *n as usize
        })
        .map(|(&arr, _)| arr)
        .collect();
    Ok(Plan { looped })
}

struct Emitter<'g> {
    graph: &'g RvGraph,
    ords: &'g [u32],
    stream_ords: &'g [u32],
    plan: Plan,
    memo: HashMap<RvId, String>,
    /// Array-valued nodes already materialized as a `var … : array<f32, n>` in the body — a
    /// `Permutation` read by several `ArrIndex`es is filled once, not once per read.
    arrays_emitted: HashSet<RvId>,
    /// Ids whose value depends on a loop placeholder (rebuilt inside a loop, not hoisted) — G4c.
    taint: HashSet<RvId>,
    /// A rolled `Scan`'s final per-slot `var` names, cached so its several `ScanOut` readers roll the
    /// loop once.
    scan_finals: HashMap<RvId, Vec<String>>,
    body: String,
}

impl Emitter<'_> {
    /// Fill a shaped draw's whole block with ONE loop — the point of the entire exercise. The block's
    /// ordinals are contiguous (`base + j`, by construction in [`source_ordinals`]), which is exactly
    /// what makes the loop expressible: the draw index *is* the loop counter.
    fn emit_draw_loop(&mut self, arr: RvId) {
        let RvNode::ArrDraw { n, src } = *self.graph.node(arr) else {
            unreachable!("looped blocks are ArrDraw nodes")
        };
        let base = self.ords[arr.0 as usize];
        let a = format!("a{}", arr.0);
        let draw = draw_expr(&src, "base + j", "lane");
        let _ = write!(
            self.body,
            "    var {a}: array<f32, {n}u>;\n\
             \x20   {{\n\
             \x20       let base = {base}u;\n\
             \x20       for (var j = 0u; j < {n}u; j = j + 1u) {{\n\
             \x20           {a}[j] = {draw};\n\
             \x20       }}\n\
             \x20   }}\n"
        );
    }

    /// Bind `expr` to a fresh `let`, so a shared node is computed once (the CSE guarantee every
    /// backend gives) and the generated text stays flat rather than exponentially nested.
    fn bind(&mut self, id: RvId, expr: String) -> String {
        let name = format!("v{}", id.0);
        let _ = writeln!(self.body, "    let {name} = {expr};");
        self.memo.insert(id, name.clone());
        name
    }

    fn emit_node(&mut self, id: RvId) -> Result<String, Unsupported> {
        if let Some(v) = self.memo.get(&id) {
            return Ok(v.clone());
        }
        // Iterative post-order: children first, so a 200k-deep chain can't overflow the emitter's
        // stack the way a recursive walk would (finding A4 — the same reason the interpreter's
        // lowering is iterative).
        let mut stack = vec![(id, false)];
        while let Some((id, expanded)) = stack.pop() {
            if self.memo.contains_key(&id) {
                continue;
            }
            if !expanded {
                stack.push((id, true));
                for c in self.children(id) {
                    if !self.memo.contains_key(&c) {
                        stack.push((c, false));
                    }
                }
                continue;
            }
            let expr = self.expr(id)?;
            self.bind(id, expr);
        }
        Ok(self.memo[&id].clone())
    }

    /// The operand nodes of `id` that must be emitted before it. An `ArrDraw` is deliberately NOT a
    /// child of its `ArrElem`s: a looped block is emitted up front, and an inlined one emits nothing
    /// at all (its readers each spell their own draw).
    fn children(&self, id: RvId) -> Vec<RvId> {
        match self.graph.node(id) {
            RvNode::Unary(_, a) => vec![*a],
            RvNode::Binary(_, a, b) => vec![*a, *b],
            RvNode::Select { cond, a, b } => vec![*cond, *a, *b],
            // The *scalar* operands. An `ArrIndex`'s array parent is materialized on demand by
            // `ensure_array` (not a scalar `let`), so it is deliberately not listed; only `index` is.
            RvNode::ArrIndex { index, .. } => vec![*index],
            RvNode::Gather { elems, index } => {
                let mut c = elems.to_vec();
                c.push(*index);
                c
            }
            // A `ScanOut`'s Scan is rolled on demand by `emit_scan` (it emits a `var`/loop, not a
            // scalar `let`), so it is not a scalar child. `Scan`/`Placeholder` are never children.
            _ => vec![],
        }
    }

    /// Materialize an array-valued node as a `var … : array<f32, n>` in the body, once. Returns the
    /// variable name. Only `Permutation` reaches here — `Rotation` is declined in `plan_blocks`, and
    /// `ArrDraw` blocks go through `emit_draw_loop`.
    fn ensure_array(&mut self, arr: RvId) -> Result<String, Unsupported> {
        let name = format!("p{}", arr.0);
        if !self.arrays_emitted.insert(arr) {
            return Ok(name);
        }
        match *self.graph.node(arr) {
            // Fisher–Yates high-to-low over the node's cell stream, byte-for-byte `Inst::Permutation`:
            // identity, then for j = n-1 … 1 swap element j with element `next_bounded(j+1)`. The
            // draw index counts up from 0 as the interpreter's `CellStream.j` does, so the consumed
            // stream — and thus the permutation — is identical lane for lane.
            RvNode::Permutation { n } => {
                let stream = self.stream_ords[arr.0 as usize];
                let _ = write!(
                    self.body,
                    "    var {name}: array<f32, {n}u>;\n\
                     \x20   {{\n\
                     \x20       for (var t = 0u; t < {n}u; t = t + 1u) {{ {name}[t] = f32(t); }}\n\
                     \x20       var dj = 0u;\n\
                     \x20       var jj = {n}u - 1u;\n\
                     \x20       loop {{\n\
                     \x20           if (jj < 1u) {{ break; }}\n\
                     \x20           let b = cell_bits48(key, {stream}u, lane, dj);\n\
                     \x20           dj = dj + 1u;\n\
                     \x20           let i = bounded48(b, jj + 1u);\n\
                     \x20           let tmp = {name}[i]; {name}[i] = {name}[jj]; {name}[jj] = tmp;\n\
                     \x20           jj = jj - 1u;\n\
                     \x20       }}\n\
                     \x20   }}\n"
                );
                Ok(name)
            }
            // A Haar rotation (G4b): fill d² standard normals from the node's cell stream, then
            // modified Gram–Schmidt the d rows in place — the same algorithm as `Inst::Rotation`, but
            // in **f32** (the interpreter uses f64 scratch, which WGSL has no equivalent for). The
            // matrix is therefore *not* bit-identical to the interpreter's; it is a valid orthonormal
            // rotation drawn from the same Haar distribution, which is the contract turboquant needs.
            //
            // Normals match `Inst::Rotation` op for op: two u48 draws per Box–Muller evaluation feed
            // BOTH branches (cos → even entry, sin → odd), so entry k costs one draw pair per two
            // entries and the per-lane draw budget matches the interpreter's (d² u48s for d² normals).
            // theta ∈ [0, 2π) stays inside the built-in trig's guaranteed range, as in `src_normal`.
            RvNode::Rotation { d } => {
                let stream = self.stream_ords[arr.0 as usize];
                let dd = d * d;
                let _ = write!(
                    self.body,
                    "    var {name}: array<f32, {dd}u>;\n\
                     \x20   {{\n\
                     \x20       var e = 0u;\n\
                     \x20       var dj = 0u;\n\
                     \x20       loop {{\n\
                     \x20           if (e >= {dd}u) {{ break; }}\n\
                     \x20           let b0 = cell_bits48(key, {stream}u, lane, dj);\n\
                     \x20           let b1 = cell_bits48(key, {stream}u, lane, dj + 1u);\n\
                     \x20           dj = dj + 2u;\n\
                     \x20           let u1 = (f32(b0.y) + 0.5) * SCALE24;\n\
                     \x20           let rr = sqrt(-2.0 * log(u1));\n\
                     \x20           let th = 6.28318530717958647692 * (f32(b1.y) * SCALE24);\n\
                     \x20           {name}[e] = rr * cos(th);\n\
                     \x20           if (e + 1u < {dd}u) {{ {name}[e + 1u] = rr * sin(th); }}\n\
                     \x20           e = e + 2u;\n\
                     \x20       }}\n\
                     \x20       for (var row = 0u; row < {d}u; row = row + 1u) {{\n\
                     \x20           for (var p = 0u; p < row; p = p + 1u) {{\n\
                     \x20               var dot = 0.0;\n\
                     \x20               for (var c = 0u; c < {d}u; c = c + 1u) {{ dot = dot + {name}[row * {d}u + c] * {name}[p * {d}u + c]; }}\n\
                     \x20               for (var c = 0u; c < {d}u; c = c + 1u) {{ {name}[row * {d}u + c] = {name}[row * {d}u + c] - dot * {name}[p * {d}u + c]; }}\n\
                     \x20           }}\n\
                     \x20           var nsq = 0.0;\n\
                     \x20           for (var c = 0u; c < {d}u; c = c + 1u) {{ nsq = nsq + {name}[row * {d}u + c] * {name}[row * {d}u + c]; }}\n\
                     \x20           let inv = 1.0 / sqrt(nsq);\n\
                     \x20           for (var c = 0u; c < {d}u; c = c + 1u) {{ {name}[row * {d}u + c] = {name}[row * {d}u + c] * inv; }}\n\
                     \x20       }}\n\
                     \x20   }}\n"
                );
                Ok(name)
            }
            ref other => unreachable!("not an array-valued node: {other:?}"),
        }
    }

    /// A per-lane read `array[round(clamp(index))]` with the engine's exact index semantics
    /// (`Inst::ArrIndex` / `Inst::Gather`): ties-away round, clamp into `0..=len-1`, NaN index → NaN
    /// (never element 0). `len >= 1` always (the evaluator builds no zero-length array).
    fn arr_read(name: &str, len: u32, idx: &str) -> String {
        let last = len - 1;
        format!(
            "select({name}[min(u32(clamp(sign({idx}) * floor(abs({idx}) + 0.5), 0.0, {last}.0)), \
             {last}u)], bitcast<f32>(NAN_BITS | (P.n & 0u)), nz_isnan({idx}))"
        )
    }

    /// Roll a `Scan` into a real WGSL `for` loop and return the final `var` name of each carried slot
    /// (G4c). This is what keeps `prisoners` off the pathological unrolled shader — its cycle-following
    /// becomes one loop of a handful of statements, not ~15,000 dependent reads.
    ///
    /// The shape: a `var` per carried slot from the inits, then a `for` over `trip`. Loop-*invariant*
    /// dependencies — above all the permutation array, a *source* that must be drawn once — are hoisted
    /// before the loop. The carried placeholders are bound to the slot `var`s and the index to the
    /// counter by injecting them into `memo`, so the body's index/carried-dependent nodes emit as
    /// ordinary `let`s inside the loop. A nested loop rolls recursively when its `ScanOut` is reached,
    /// landing inside the enclosing loop.
    fn emit_scan(&mut self, scan_id: RvId) -> Result<Vec<String>, Unsupported> {
        if let Some(f) = self.scan_finals.get(&scan_id) {
            return Ok(f.clone());
        }
        let RvNode::Scan { body } = self.graph.node(scan_id).clone() else {
            unreachable!("emit_scan on a non-Scan")
        };
        // Slot vars, initialised from the carried inits (emitted before the loop — an init is either a
        // hoisted invariant or, when nested, an enclosing loop's `var`/counter already bound in memo).
        let mut slots = Vec::with_capacity(body.inits.len());
        for (s, &init) in body.inits.iter().enumerate() {
            let init = self.emit_node(init)?;
            let v = format!("sc{}_{}", scan_id.0, s);
            let _ = writeln!(self.body, "    var {v} = {init};");
            slots.push(v);
        }
        // Hoist every loop-invariant node the body reads to BEFORE the loop, so it is computed once —
        // and a permutation/rotation source is *drawn* once, not re-drawn per iteration.
        for id in self.body_invariants(&body.nexts) {
            if matches!(self.graph.node(id), RvNode::Permutation { .. } | RvNode::Rotation { .. }) {
                self.ensure_array(id)?;
            } else {
                self.emit_node(id)?;
            }
        }
        let j = format!("sj{}", scan_id.0);
        let _ = writeln!(
            self.body,
            "    for (var {j} = 0u; {j} < {}u; {j} = {j} + 1u) {{",
            body.trip
        );
        // Bind placeholders → slot vars, index → the counter (as f32), by injecting into `memo`.
        let mut injected: Vec<RvId> = Vec::new();
        for (s, &ph) in body.placeholders.iter().enumerate() {
            self.memo.insert(ph, slots[s].clone());
            injected.push(ph);
        }
        if let Some(iph) = body.index_ph {
            self.memo.insert(iph, format!("f32({j})"));
            injected.push(iph);
        }
        // Snapshot AFTER injecting: the memo keys added while emitting *this* loop's body are exactly
        // its own dynamic `let`s, which are the only ones to drop at the end (a nested loop cleans up
        // its own — dropping *all* tainted nodes globally would wipe an enclosing loop's live locals).
        let before: HashSet<RvId> = self.memo.keys().copied().collect();
        // Emit the recurrence: the body's dynamic nodes land as `let`s inside the loop.
        let mut nexts = Vec::with_capacity(body.nexts.len());
        for &nx in body.nexts.iter() {
            nexts.push(self.emit_node(nx)?);
        }
        for (s, nx) in nexts.iter().enumerate() {
            let _ = writeln!(self.body, "        {} = {};", slots[s], nx);
        }
        let _ = writeln!(self.body, "    }}");
        // The loop's `let`s and its placeholder bindings are scoped to the loop; drop them from `memo`
        // (this loop's additions only) so nothing outside references an out-of-scope local.
        let added: Vec<RvId> = self
            .memo
            .keys()
            .copied()
            .filter(|id| !before.contains(id) && self.taint.contains(id))
            .collect();
        for id in added.into_iter().chain(injected) {
            self.memo.remove(&id);
        }
        self.scan_finals.insert(scan_id, slots.clone());
        Ok(slots)
    }

    /// The loop-invariant nodes reachable from `roots` that a loop must hoist — descend *through*
    /// tainted (loop-dependent) nodes to find the invariant sub-expressions they read, and stop at each
    /// invariant (its whole cone is invariant, so the normal emitter handles it).
    fn body_invariants(&self, roots: &[RvId]) -> Vec<RvId> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let mut stack: Vec<RvId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if self.taint.contains(&id) {
                self.push_operands(id, &mut stack);
            } else {
                out.push(id);
            }
        }
        out
    }

    /// All direct operand ids of a node (including an `ArrIndex`/`Scan`'s array/loop inputs, unlike
    /// `children`). Used to walk a loop body for hoisting.
    fn push_operands(&self, id: RvId, stack: &mut Vec<RvId>) {
        match self.graph.node(id) {
            RvNode::Unary(_, a) | RvNode::ArrElem { arr: a, .. } => stack.push(*a),
            RvNode::Binary(_, a, b) | RvNode::ArrIndex { arr: a, index: b } => {
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Select { cond, a, b } => {
                stack.push(*cond);
                stack.push(*a);
                stack.push(*b);
            }
            RvNode::Gather { elems, index } => {
                stack.extend(elems.iter().copied());
                stack.push(*index);
            }
            RvNode::Scan { body } => {
                stack.extend(body.inits.iter().copied());
                stack.extend(body.nexts.iter().copied());
            }
            RvNode::ScanOut { scan, .. } => stack.push(*scan),
            RvNode::Src(_)
            | RvNode::ConstNum(_)
            | RvNode::ConstBool(_)
            | RvNode::Input { .. }
            | RvNode::Permutation { .. }
            | RvNode::Rotation { .. }
            | RvNode::ArrDraw { .. }
            | RvNode::Placeholder { .. } => {}
        }
    }

    /// The WGSL expression for one node, over its already-emitted children.
    fn expr(&mut self, id: RvId) -> Result<String, Unsupported> {
        let node = self.graph.node(id).clone();
        Ok(match node {
            RvNode::ConstNum(x) => f32c(x as f32),
            RvNode::ConstBool(b) => if b { "1.0" } else { "0.0" }.to_string(),

            // A scalar source: its ordinal is fixed, so the counter is a compile-time constant.
            RvNode::Src(src) => draw_expr(&src, &format!("{}u", self.ords[id.0 as usize]), "lane"),

            // Element `k` of a shaped draw. If its block was looped, this is a free array read;
            // otherwise the block is inlined and this reader spells its own draw at ordinal base+k.
            RvNode::ArrElem { arr, k } => {
                if self.plan.looped.contains(&arr) {
                    format!("a{}[{k}u]", arr.0)
                } else {
                    let ord = elem_ordinal(self.ords, arr, k);
                    draw_expr(&elem_source(self.graph, arr), &format!("{ord}u"), "lane")
                }
            }

            RvNode::Unary(op, a) => {
                let x = self.memo[&a].clone();
                match op {
                    UnOp::Neg => format!("(-{x})"),
                    // Bools are f32 0/1 on every backend, so `not` is "is it zero".
                    UnOp::Not => format!("select(0.0, 1.0, {x} == 0.0)"),
                    UnOp::Sign => format!("sign({x})"),
                    UnOp::Floor => format!("floor({x})"),
                    UnOp::Ceil => format!("ceil({x})"),
                    // NOT WGSL's `round`: that is ties-to-EVEN, while the engine's `round` is
                    // ties-AWAY (`f64::round`). They disagree by a whole unit at every .5, which no
                    // ULP bound would catch — so it is spelled out.
                    UnOp::Round => format!("(sign({x}) * floor(abs({x}) + 0.5))"),
                    UnOp::Sqrt => format!("sqrt({x})"),
                    // NOT WGSL's `sin`/`cos`: those are only *guaranteed* on [-pi, pi], and Metal's
                    // return 0 for sin(1e12) — a wrong answer, not a rounding gap. `nz_sin`/`nz_cos`
                    // reduce with integer Payne-Hanek and run the engine's own kernels, so they hold
                    // for every f32 argument (G1b).
                    UnOp::Sin => format!("nz_sin({x})"),
                    UnOp::Cos => format!("nz_cos({x})"),
                    UnOp::Exp => format!("exp({x})"),
                    UnOp::Atan => format!("atan({x})"),
                    // WGSL `log` is the natural log. Its domain guards already match
                    // `approx::ln_guarded_f32`: log(0) = -inf, log(x<0) = NaN.
                    UnOp::Ln => format!("log({x})"),
                }
            }

            RvNode::Binary(BinOp::Pow, a, b) => {
                // A small constant integer exponent fuses to repeated multiplies on every backend —
                // `const_int_exponent` is the shared test, so all four agree on which ones fuse.
                if let Some(k) = const_int_exponent(self.graph, b) {
                    let x = self.memo[&a].clone();
                    match k {
                        0 => "1.0".to_string(),
                        1 => x,
                        _ => {
                            let prod = std::iter::repeat_n(x.as_str(), k as usize)
                                .collect::<Vec<_>>()
                                .join(" * ");
                            format!("({prod})")
                        }
                    }
                } else {
                    let (x, y) = (self.memo[&a].clone(), self.memo[&b].clone());
                    format!("pow({x}, {y})")
                }
            }

            RvNode::Binary(op, a, b) => {
                let (x, y) = (self.memo[&a].clone(), self.memo[&b].clone());
                // IEEE ordering: any comparison involving a NaN is false, except `!=`, which is
                // true. Both operands are explicitly screened with `nz_isnan` rather than left to
                // the hardware — under the vendors' default fast-math the float comparison alone is
                // not trustworthy in the presence of NaNs (see `nz_isnan`).
                let ordered = format!("!nz_isnan({x}) && !nz_isnan({y})");
                let cmp = |c: &str| format!("select(0.0, 1.0, {ordered} && {x} {c} {y})");
                match op {
                    BinOp::Add => format!("({x} + {y})"),
                    BinOp::Sub => format!("({x} - {y})"),
                    BinOp::Mul => format!("({x} * {y})"),
                    BinOp::Div => format!("({x} / {y})"),
                    // NOT WGSL's `%`: that is a TRUNCATED remainder, while the engine's `mod` is
                    // FLOORED (`-1 mod 3 == 2`, `7 mod -3 == -2`). Whole-unit disagreement again.
                    BinOp::Mod => format!("({x} - {y} * floor({x} / {y}))"),
                    BinOp::Eq => cmp("=="),
                    // The one inverted case: `NaN != anything` is TRUE.
                    BinOp::Ne => format!(
                        "select(0.0, 1.0, nz_isnan({x}) || nz_isnan({y}) || {x} != {y})"
                    ),
                    BinOp::Lt => cmp("<"),
                    BinOp::Gt => cmp(">"),
                    BinOp::Le => cmp("<="),
                    BinOp::Ge => cmp(">="),
                    BinOp::And => format!("select(0.0, 1.0, {x} != 0.0 && {y} != 0.0)"),
                    BinOp::Or => format!("select(0.0, 1.0, {x} != 0.0 || {y} != 0.0)"),
                    BinOp::Pow => unreachable!("handled above"),
                }
            }

            RvNode::Select { cond, a, b } => {
                let (c, x, y) = (
                    self.memo[&cond].clone(),
                    self.memo[&a].clone(),
                    self.memo[&b].clone(),
                );
                // WGSL's `select(false_val, true_val, cond)` — argument order is the trap here.
                format!("select({y}, {x}, {c} != 0.0)")
            }

            // A per-lane read of an array-valued source. Materialize the array (a `Permutation`;
            // a `Rotation` would have been declined), then read it at the lane's rounded index.
            RvNode::ArrIndex { arr, index } => {
                let name = self.ensure_array(arr)?;
                let idx = self.memo[&index].clone();
                let RvKind::Arr(len) = self.graph.kind(arr) else {
                    unreachable!("ArrIndex parent is array-kinded")
                };
                Self::arr_read(&name, len, &idx)
            }

            // A per-lane read of a *table of scalar nodes* (`xs[i]` with `i` an RV). The elements are
            // already emitted as `let`s (they are children); pack them into a local array and read it
            // with the same index semantics as `ArrIndex`.
            RvNode::Gather { elems, index } => {
                let idx = self.memo[&index].clone();
                let len = elems.len() as u32;
                let name = format!("g{}", id.0);
                let _ = writeln!(self.body, "    var {name}: array<f32, {len}u>;");
                for (k, e) in elems.iter().enumerate() {
                    let v = self.memo[e].clone();
                    let _ = writeln!(self.body, "    {name}[{k}u] = {v};");
                }
                Self::arr_read(&name, len, &idx)
            }

            // A read of a rolled loop's final carried value: roll the loop (once) and name the slot.
            RvNode::ScanOut { scan, slot } => self.emit_scan(scan)?[slot as usize].clone(),

            other => unreachable!("plan_blocks rejects {other:?} before we get here"),
        })
    }
}

/// One draw, as a WGSL expression: `ctr` is the source ordinal (a literal, or the loop's `base + j`)
/// and `lane` the lane. The `src_*` helpers live in [`PRELUDE`].
fn draw_expr(src: &Source, ctr: &str, lane: &str) -> String {
    match src {
        Source::Uniform(u) => {
            // The bounds are folded to f32 ONCE, exactly as `rng::fill_uniform` folds them, so the
            // arithmetic agrees op-for-op with the CPU fills.
            let (loc, span) = (u.lo as f32, (u.hi - u.lo) as f32);
            format!("src_unif(key, {ctr}, {lane}, {}, {})", f32c(loc), f32c(span))
        }
        Source::UniformInt { lo, hi } => {
            let count = (hi - lo + 1.0).max(1.0) as u32;
            format!(
                "src_unif_int(key, {ctr}, {lane}, {}, {count}u)",
                f32c(*lo as f32)
            )
        }
        Source::Normal { mu, sigma } => format!(
            "src_normal(key, {ctr}, {lane}, {}, {})",
            f32c(*mu as f32),
            f32c(*sigma as f32)
        ),
        Source::Exp { rate } => {
            format!("src_exp(key, {ctr}, {lane}, {})", f32c(*rate as f32))
        }
        Source::Geometric { p } => {
            // The compile-time denominator is the f64 `ln` rounded to f32 — the exact constant the
            // other emitters bake in (`rng::fill_geometric`).
            let denom = (1.0 - p).ln() as f32;
            format!("src_geom(key, {ctr}, {lane}, {})", f32c(denom))
        }
        Source::Poisson { .. } => unreachable!("plan_blocks rejects Poisson"),
    }
}

/// Spell an f32 as an **exact** WGSL constant, by its bit pattern.
///
/// A decimal literal is a re-rounding, and the engine's constants are specified as f32 *values* — so
/// writing `0.33333334` in the shader is a different number than the CPU folded. Bit-for-bit draw
/// parity is decided by exactly this, so the emitter never hand-rounds a constant.
///
/// **Non-finite constants go through a runtime zero.** WGSL forbids a *const-expression* that
/// evaluates to NaN or ±Inf, and Tint (Chrome/Dawn's WebGPU) enforces it — `naga` (native wgpu) does
/// not, which is why native GPU tests never caught this and only a live browser did (PLAN-WEBGPU G3).
/// `P.n & 0u` is a runtime zero, so `bits | (P.n & 0u)` is a *non-const* expression carrying the same
/// bits; `bitcast` of it is the identical NaN/Inf at runtime but is no longer a rejected const NaN.
fn f32c(x: f32) -> String {
    if x.is_finite() {
        format!("bitcast<f32>({:#010x}u)", x.to_bits())
    } else {
        format!("bitcast<f32>({:#010x}u | (P.n & 0u))", x.to_bits())
    }
}

/// The fixed prelude: params, the `vec2<u32>` u64 emulation, `squares64`, and the draw sources.
///
/// The consumption contract falls out beautifully here. C0's rule is "the top 24 bits of each u32
/// half", so `draw48`'s two 24-bit halves are just `w.x >> 8` and `w.y >> 8` — the 48-bit value is
/// never assembled at all for the pair-shared sources.
const PRELUDE: &str = r#"
struct Params {
    key: vec2<u32>,
    lane0: u32,
    n: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (mid << 16u) | (p00 & 0xffffu);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// Low 64 bits of a 64x64 product: the two high partial products are discarded, so only ONE
// 32x32->64 multiply is wide.
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul_wide(a.x, b.x);
    return vec2<u32>(ll.x, ll.y + a.x * b.y + a.y * b.x);
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

// squares64 (Widynski). `x.yx` IS rotate_left(x, 32) on a u64 — the half-swap is free.
fn squares64(ctr: vec2<u32>, key: vec2<u32>) -> vec2<u32> {
    var x = mul64(ctr, key);
    let y = x;
    let z = add64(y, key);
    x = add64(mul64(x, x), y); x = x.yx;
    x = add64(mul64(x, x), z); x = x.yx;
    x = add64(mul64(x, x), y); x = x.yx;
    let t = add64(mul64(x, x), z);
    x = t.yx;
    let f = add64(mul64(x, x), y);
    return vec2<u32>(t.x ^ f.y, t.y);   // t ^ (f >> 32)
}

// The pair-shared draw: counter (source << 36) + (lane >> 1), which in words is just
// (lane >> 1, source << 4). One hash feeds a lane PAIR — even lane takes the low 24 bits, odd the
// high 24 (rng::pair_ctr / lo24 / hi24).
fn pair_bits(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    let w = squares64(vec2<u32>(lane >> 1u, source << 4u), key);
    return vec2<u32>(w.x >> 8u, w.y >> 8u);
}

// unif_int's per-lane draw: counter (source << 36) + lane, all 48 bits spent on one lane.
fn lane_bits48(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    let w = squares64(vec2<u32>(lane, source << 4u), key);
    return vec2<u32>(w.x >> 8u, w.y >> 8u);   // (lo24, hi24)
}

// A CellStream draw (rng::CellStream): the counter region is (1<<63) | (stream<<49) | (lane<<17) | j,
// disjoint from the source region above (bit 63 set), so a Permutation's shuffle and a plain source
// never collide. `j` is the draw index within the lane. Returns the 48 consumed bits as (lo24, hi24),
// exactly what draw48 packs: lo24 = w[8:32], hi24 = w[40:64].
fn cell_bits48(key: vec2<u32>, stream: u32, lane: u32, j: u32) -> vec2<u32> {
    let lo = (lane << 17u) | j;
    let hi = 0x80000000u | (stream << 17u) | (lane >> 15u);
    let w = squares64(vec2<u32>(lo, hi), key);
    return vec2<u32>(w.x >> 8u, w.y >> 8u);
}

// Lemire bounded reduction of a (lo24, hi24) draw: (bits48 * count) >> 48, in 32-bit pieces. Same
// arithmetic as src_unif_int, factored out for the Fisher–Yates in a Permutation fill.
fn bounded48(b: vec2<u32>, count: u32) -> u32 {
    let t = mul_wide(count, b.x);
    let u = mul_wide(count, b.y);
    let tsh = (t.x >> 24u) | (t.y << 8u);
    let s = add64(u, vec2<u32>(tsh, 0u));
    return (s.x >> 24u) | (s.y << 8u);
}

// NaN, detected by BITS — not by `x != x`.
//
// The vendor compilers enable fast-math by default, which assumes NaNs do not exist: on Metal,
// `x == x` folds to `true` outright. That is not academic here — `log(X) == log(X)` is how the
// engine expresses "is X in the domain", and folding it turned a 0.6 into a 1.0 (caught by the
// conformance harness). An exponent-and-mantissa bit test states the question in integer terms, so
// there is no float identity for the optimizer to exploit.
fn nz_isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

const SCALE24: f32 = 0x1p-24;
fn unit24(b: u32) -> f32 { return f32(b) * SCALE24; }
fn my_half(b: vec2<u32>, lane: u32) -> u32 { return select(b.x, b.y, (lane & 1u) == 1u); }

fn src_unif(key: vec2<u32>, source: u32, lane: u32, loc: f32, span: f32) -> f32 {
    return loc + span * unit24(my_half(pair_bits(key, source, lane), lane));
}

// Box-Muller over the lane pair: u1 from the low 24 bits, u2 from the high. `1 - u1` is exactly
// representable on the 2^-24 grid and lies in [2^-24, 1], so the ln(0) guard is structural.
// Even lane takes the cos branch, odd the sin (rng::normal_pair).
fn src_normal(key: vec2<u32>, source: u32, lane: u32, mu: f32, sigma: f32) -> f32 {
    let b = pair_bits(key, source, lane);
    let r = sqrt(-2.0 * log(1.0 - unit24(b.x)));
    let theta = 6.28318530717958647692 * unit24(b.y);
    let z = select(cos(theta), sin(theta), (lane & 1u) == 1u);
    return mu + sigma * r * z;
}

fn src_exp(key: vec2<u32>, source: u32, lane: u32, rate: f32) -> f32 {
    let u = unit24(my_half(pair_bits(key, source, lane), lane));
    return -log(1.0 - u) / rate;
}

fn src_geom(key: vec2<u32>, source: u32, lane: u32, denom: f32) -> f32 {
    let u = unit24(my_half(pair_bits(key, source, lane), lane));
    return floor(log(1.0 - u) / denom);
}

// Lemire multiply-high on the 48 consumed bits, in 32-bit pieces. With bits48 = hi*2^24 + lo,
// (bits48 * count) >> 48 == (count*hi + ((count*lo) >> 24)) >> 24.
fn src_unif_int(key: vec2<u32>, source: u32, lane: u32, loc: f32, count: u32) -> f32 {
    let b = lane_bits48(key, source, lane);
    let t = mul_wide(count, b.x);
    let u = mul_wide(count, b.y);
    let tsh = (t.x >> 24u) | (t.y << 8u);
    let s = add64(u, vec2<u32>(tsh, 0u));
    let k = (s.x >> 24u) | (s.y << 8u);
    return loc + f32(k);
}

// ---------------------------------------------------------------------------
// Explicit sin/cos (PLAN-WEBGPU G1b).
//
// The engine's contract is: below `approx::TRIG_MAX_F32` (1024), a 2-term Cody-Waite reduction and
// an inline poly; at or above it, compute in **f64** and round to f32. WGSL has no f64, so the
// fallback half cannot be reproduced literally -- which is why G1 declined this node outright.
//
// The reason it can't just be `x % TAU` is worth stating, because it is the whole design: the
// quotient `x / (pi/2)` at x = 1e12 is ~6.4e11, which needs 40 bits of mantissa. f32 has 24. So the
// quotient is rounded *before* the subtraction, and `x - k*(pi/2)` then cancels two nearly-equal
// large numbers whose difference is dominated by that rounding -- the reduced argument comes out
// wrong by ~x*2^-24 radians, i.e. it carries no information at all. (Metal's built-in `sin` gives up
// the same way: it returns 0 for sin(1e12*X) against the engine's 0.0056.)
//
// **Payne-Hanek** is the answer, and it is an INTEGER algorithm -- which is exactly why it is
// reachable here: the wide-multiply machinery it needs is already in this prelude, built for
// squares64 because WGSL has no u64 either. An f32 is `mant * 2^e2` with a 24-bit integer mantissa,
// so `x * 2/pi` is an exact integer product against the bits of 2/pi -- provided you have enough of
// them, and take the right window. No f64 anywhere, and exact for EVERY f32 argument, not just the
// ones below some threshold.
//
// Bits of 2/pi, most significant first. TWO_OVER_PI[0] is a zero pad: 2/pi < 1, so every bit before
// the binary point is zero, and the pad lets a small (even negative) exponent index the table
// without a special case. Word k>=1 holds fraction bits 32(k-1)+1 ..= 32k. (These are fdlibm's
// `two_over_pi`, regrouped from 24-bit to 32-bit limbs.)
const TWO_OVER_PI: array<u32, 13> = array<u32, 13>(
    0x00000000u,
    0xa2f9836eu, 0x4e441529u, 0xfc2757d1u, 0xf534ddc0u, 0xdb629599u, 0x3c439041u,
    0xfe5163abu, 0xdebbc561u, 0xb7246e3au, 0x424dd2e0u, 0x06492eeau, 0x09d1921cu,
);

// pi/2, split so that `k * PIO2_HI` is exact in f32 for every k the small path admits (the high part
// carries only 8 significant mantissa bits). Same split as `approx::PIO2_{HI,LO}_F32`.
const PIO2_HI: f32 = 0x1.92p+0;
const PIO2_LO: f32 = 0x1.fb5444p-12;
const PI_4: f32 = 0x1.921fb6p-1;
// The quiet-NaN bit pattern. Used only as `NAN_BITS | (P.n & 0u)` — the OR with a runtime zero makes
// the whole expression non-const, because WGSL forbids a const-expression that evaluates to NaN and
// Tint (Chrome/Dawn) enforces it while `naga` (native wgpu) does not (PLAN-WEBGPU G3). Same NaN at
// runtime, on both backends.
const NAN_BITS: u32 = 0x7fc00000u;

// Reduce |x| to (r, k) with r in [-pi/4, pi/4] and x ~ r + k*(pi/2). `ax` must be finite and >= 0.
//
// **There is no Cody-Waite fast path here, and that is deliberate.** The obvious structure -- cheap
// 2-term reduction below `approx::TRIG_MAX_F32`, exact reduction above -- was written, measured, and
// removed: it is off by 1e-5 at x = 1000, a hundred times its budget. The reason is that Cody-Waite
// depends on `(ax - k*HI) - k*LO` being evaluated *in that order*, with HI carrying only 8 mantissa
// bits so `k*HI` is exact. Fast-math is permitted to reassociate, and Metal does, collapsing it into
// `ax - k*(HI + LO)` -- which rounds HI+LO back to a single f32 pi/2 and throws away precisely the
// bits the split exists to protect. The error is then `k * ulp(pi/2)`, which is what was measured.
//
// It is the same lesson as `nz_isnan`: a float identity the optimizer is allowed to "simplify" is
// not a thing you can build a contract on. State it in integers, where there is nothing to exploit.
// So every argument takes the exact path, and there is one code path instead of two.
fn trig_reduce(ax: f32) -> vec2<f32> {
    // Already reduced: |x| < pi/4 means k = 0 and r = x, which is what the engine's own reduction
    // yields there too. This also keeps `ax` normal and bounds the exponent below, so the table
    // index cannot run off the front (see `p0`).
    if (ax < PI_4) { return vec2<f32>(ax, 0.0); }

    // Payne-Hanek. Write ax = mant * 2^e2, mant a 24-bit integer. Then
    //
    //     ax * 2/pi = mant * sum_i TP[i] * 2^(e2 - i)          (TP[i] = bit i of 2/pi's fraction)
    //
    // Terms with e2 - i >= 2 are integer multiples of 4, so they vanish mod 4 -- and mod 4 is all a
    // quadrant needs. Terms below the window are under 2^-38 of the fraction. So only a 96-bit
    // window of 2/pi matters, starting at fraction bit i0 = e2 - 1, and within it
    //
    //     ax * 2/pi  ==  mant * F / 2^94   (mod 4),   F = those 96 bits as an integer.
    //
    // That product is 120 bits; we want bits 94 (the quadrant) and just below (the fraction).
    let bits = bitcast<u32>(ax);
    let mant = (bits & 0x007fffffu) | 0x00800000u;
    let e2 = i32((bits >> 23u) & 0xffu) - 127 - 23;

    // Extract the 96-bit window. `p0` is the padded bit offset of i0; the pad word is what keeps it
    // non-negative for the small exponents (ax >= pi/4 gives e2 >= -24, so p0 >= 6).
    let p0 = u32(e2 + 30);
    let wi = p0 >> 5u;
    let sh = p0 & 31u;
    // A 32-bit shift is undefined in WGSL, so splice with a select rather than `>> (32 - sh)`.
    let w0 = TWO_OVER_PI[wi]; let w1 = TWO_OVER_PI[wi + 1u];
    let w2 = TWO_OVER_PI[wi + 2u]; let w3 = TWO_OVER_PI[wi + 3u];
    let f2 = (w0 << sh) | select(w1 >> (32u - sh), 0u, sh == 0u);
    let f1 = (w1 << sh) | select(w2 >> (32u - sh), 0u, sh == 0u);
    let f0 = (w2 << sh) | select(w3 >> (32u - sh), 0u, sh == 0u);

    // mant * F, keeping the two high partial products. Dropping mant*f0 costs < 2^56 out of a 2^94
    // scale -- 2^-38 of the fraction, ~1e-11 radians in r.
    let a = mul_wide(mant, f2);   // weight 2^64
    let b = mul_wide(mant, f1);   // weight 2^32
    // S = (mant*F) >> 32. Only its low two words are ever read: the quadrant lives at bits 62,63 and
    // everything at bit 64 and above is a multiple of 4, so the whole high word (and the carry out of
    // s1, which WGSL wraps away for free) vanishes mod 4. That is the point of working mod 4.
    let s0 = b.x;
    let s1 = a.x + b.y;

    // Quadrant = bits 62,63 of S = bits 30,31 of s1. Fraction = the rest, over 2^62.
    let q = f32(s1 >> 30u);
    var frac = f32(s1 & 0x3fffffffu) * 0x1p-30 + f32(s0) * 0x1p-62;

    // Fold to (-1/2, 1/2] so r lands in [-pi/4, pi/4], exactly as the Cody-Waite branch does.
    var k = q;
    if (frac > 0.5) { frac = frac - 1.0; k = k + 1.0; }
    return vec2<f32>(frac * PIO2_HI + frac * PIO2_LO, k);
}

// approx::{SIN,COS}_COEFFS_F32, Horner'd. Written as exact hex floats: a decimal literal is a
// request for the compiler to round, and these must be the same f32s the CPU kernels hold.
fn sin_kernel(r: f32) -> f32 {
    let z = r * r;
    let p = ((0x1.71de3ap-19 * z - 0x1.a01a02p-13) * z + 0x1.111112p-7) * z - 0x1.555556p-3;
    return r + r * z * p;
}

fn cos_kernel(r: f32) -> f32 {
    let z = r * r;
    let p = ((-0x1.27e4fcp-22 * z + 0x1.a01a02p-16) * z - 0x1.6c16c2p-10) * z + 0x1.555556p-5;
    return 1.0 - 0.5 * z + z * z * p;
}

// Pick the kernel and sign for quadrant k mod 4 (approx::quadrant_f32).
fn trig_quadrant(kq: u32, s: f32, c: f32, is_cos: bool) -> f32 {
    let q0 = select(s, c, is_cos);
    let q1 = select(c, -s, is_cos);
    let q2 = select(-s, -c, is_cos);
    let q3 = select(-c, s, is_cos);
    var res = q0;
    if (kq == 1u) { res = q1; }
    if (kq == 2u) { res = q2; }
    if (kq == 3u) { res = q3; }
    return res;
}

// sin(+-inf) and sin(NaN) are NaN. Screened by bits, for the same fast-math reason as `nz_isnan`.
fn nz_sin(x: f32) -> f32 {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) { return bitcast<f32>(NAN_BITS | (P.n & 0u)); }
    let rk = trig_reduce(abs(x));
    let kq = u32(rk.y) & 3u;
    let v = trig_quadrant(kq, sin_kernel(rk.x), cos_kernel(rk.x), false);
    return select(v, -v, x < 0.0);   // sin is odd; the reduction ran on |x|
}

fn nz_cos(x: f32) -> f32 {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) { return bitcast<f32>(NAN_BITS | (P.n & 0u)); }
    let rk = trig_reduce(abs(x));
    let kq = u32(rk.y) & 3u;
    return trig_quadrant(kq, sin_kernel(rk.x), cos_kernel(rk.x), true);   // cos is even
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{RvKind, Uniform};

    /// Just the kernel body — the prelude defines `fn src_normal(...)` etc., so counting draw calls
    /// over the whole shader would count every definition as a use.
    fn body(src: &str) -> &str {
        src.split("fn main").nth(1).expect("emitted shader has a main")
    }

    fn g_with(f: impl FnOnce(&mut RvGraph) -> RvId) -> (RvGraph, RvId) {
        let mut g = RvGraph::default();
        let r = f(&mut g);
        (g, r)
    }

    /// The headline structural claim of G1: a shaped draw becomes **one loop**, and the shader
    /// therefore contains **one** `squares64`-driven source call — not `n` of them.
    ///
    /// This is the difference between 332 ms and 31 ms of cold pipeline compile on
    /// `barrier_option`'s shape, i.e. between losing to the CPU and beating it. If a future change
    /// makes the emitter unroll again, the shader still *works* and every conformance test still
    /// passes — it just quietly becomes slower than the backend it exists to beat. Hence a test.
    #[test]
    fn a_shaped_draw_emits_one_loop_not_n_inlined_hashes() {
        let (g, root) = g_with(|g| {
            let arr = g.push(
                RvNode::ArrDraw { n: 52, src: Source::Normal { mu: 0.0, sigma: 1.0 } },
                RvKind::Arr(52),
            );
            let mut acc = g.push(RvNode::ConstNum(0.0), RvKind::Num);
            for k in 0..52 {
                let e = g.push(RvNode::ArrElem { arr, k }, RvKind::Num);
                acc = g.push(RvNode::Binary(BinOp::Add, acc, e), RvKind::Num);
            }
            acc
        });
        let src = emit(&g, &[root]).expect("lowers");
        let b = body(&src);
        assert_eq!(b.matches("for (var j").count(), 1, "expected exactly one draw loop");
        assert_eq!(
            b.matches("src_normal(").count(),
            1,
            "the 52 draws must collapse to ONE call inside the loop — an unrolled emitter would \
             put 52 here and cost 10x the pipeline compile:\n{src}"
        );
        assert!(src.contains("var a"), "the block must be materialized: {src}");
    }

    /// A block that is barely read must NOT be looped: drawing all 10,000 elements per lane to read
    /// one of them would trade a few ms of compile for an enormous amount of GPU work.
    #[test]
    fn a_barely_read_block_is_inlined_not_looped() {
        let (g, root) = g_with(|g| {
            let arr = g.push(
                RvNode::ArrDraw { n: 10_000, src: Source::Normal { mu: 0.0, sigma: 1.0 } },
                RvKind::Arr(10_000),
            );
            g.push(RvNode::ArrElem { arr, k: 0 }, RvKind::Num)
        });
        let src = emit(&g, &[root]).expect("lowers");
        let b = body(&src);
        assert!(!b.contains("for (var j"), "a 1-of-10000 read must not emit a draw loop:\n{src}");
        assert_eq!(b.matches("src_normal(").count(), 1);
    }

    /// The two ops whose WGSL built-in has *different semantics* from the engine's must not be
    /// lowered to that built-in. Both would disagree by a whole unit — far outside the ULP tier that
    /// covers everything else, and so invisible to an accuracy check.
    #[test]
    fn round_and_mod_are_not_the_wgsl_builtins() {
        let (g, root) = g_with(|g| {
            let u = g.push(
                RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                RvKind::Num,
            );
            let r = g.push(RvNode::Unary(UnOp::Round, u), RvKind::Num);
            let three = g.push(RvNode::ConstNum(3.0), RvKind::Num);
            g.push(RvNode::Binary(BinOp::Mod, r, three), RvKind::Num)
        });
        let src = emit(&g, &[root]).expect("lowers");
        // Scoped to the kernel body: the prelude is prose as well as code, and a `%` inside a comment
        // is not a use of WGSL's `%`.
        let b = body(&src);
        // ties-AWAY, not WGSL's ties-to-even `round`.
        assert!(b.contains("floor(abs("), "round must be ties-away: {src}");
        assert!(!b.contains("round("), "must not call WGSL's round: {src}");
        // FLOORED mod, not WGSL's truncated `%`.
        assert!(b.contains("floor(v"), "mod must be floored: {src}");
        assert!(!b.contains(" % "), "must not use WGSL's truncated %: {src}");
    }

    /// The cones this backend can't lower are declined, not mis-emitted — the caller falls back to a
    /// correct backend exactly as `wasm_host` does. (G4 is where WGSL *beats* the CPU codegen on
    /// these: it has array indexing and divergent loops.)
    #[test]
    fn unsupported_cones_are_declined() {
        // Poisson is the last node this backend declines: its Knuth loop counts f64 uniforms a
        // data-dependent number of times, which WGSL's f32 can't reproduce. It declines whether read
        // directly or through a shaped block, so the whole cone falls back — not a partial lowering.
        let (g, root) = g_with(|g| {
            g.push(RvNode::Src(Source::Poisson { lambda: 3.0 }), RvKind::Num)
        });
        assert_eq!(emit(&g, &[root]), Err(Unsupported("poisson")));

        let (g, root) = g_with(|g| {
            let arr = g.push(
                RvNode::ArrDraw { n: 4, src: Source::Poisson { lambda: 2.0 } },
                RvKind::Arr(4),
            );
            g.push(RvNode::ArrElem { arr, k: 1 }, RvKind::Num)
        });
        assert_eq!(emit(&g, &[root]), Err(Unsupported("poisson")));
    }

    /// A `Permutation` read through `ArrIndex` lowers to a Fisher–Yates fill plus an array read — the
    /// G4 unlock for `prisoners`. Integer-only, so it is bit-identical to the interpreter (proved on
    /// device by `a_permutation_kernel_matches_the_interpreter`); this pins the emitted shape.
    #[test]
    fn a_permutation_lowers_to_a_fisher_yates_fill() {
        let (g, root) = g_with(|g| {
            let perm = g.push(RvNode::Permutation { n: 8 }, RvKind::Arr(8));
            let k = g.push(RvNode::ConstNum(3.0), RvKind::Num);
            g.push(RvNode::ArrIndex { arr: perm, index: k }, RvKind::Num)
        });
        let src = emit(&g, &[root]).expect("a permutation must lower now, not decline");
        let b = body(&src);
        assert!(b.contains("cell_bits48("), "the shuffle must draw from a cell stream:\n{src}");
        assert!(b.contains("bounded48("), "Fisher–Yates needs the bounded draw:\n{src}");
        assert!(b.contains("array<f32, 8u>"), "the permutation array must be materialized:\n{src}");
    }

    /// A `Rotation` lowers to a Gram–Schmidt **loop** (G4b) — the property that keeps it a small
    /// shader. A d=20 rotation is 400 normals + O(d³) MGS flops; unrolled that is the pathological
    /// shape that keeps `prisoners` off the GPU, but as nested `for`s it is a handful of statements
    /// whatever d is, so it compiles cheap and the GPU wins big on turboquant.
    #[test]
    fn a_rotation_lowers_to_a_gram_schmidt_loop() {
        let (g, root) = g_with(|g| {
            let rot = g.push(RvNode::Rotation { d: 20 }, RvKind::Arr(400));
            let k = g.push(RvNode::ConstNum(0.0), RvKind::Num);
            g.push(RvNode::ArrIndex { arr: rot, index: k }, RvKind::Num)
        });
        let src = emit(&g, &[root]).expect("a rotation must lower now, not decline");
        let b = body(&src);
        assert!(b.contains("array<f32, 400u>"), "the matrix must be materialized:\n{src}");
        assert!(b.contains("cell_bits48("), "the normal fill draws from a cell stream:\n{src}");
        // The whole point: a bounded loop, NOT 400 unrolled draws. `barrier_option`'s lesson applied
        // to Gram–Schmidt — a d=20 shader must not scale its statement count with d³.
        assert!(
            b.matches("cell_bits48(").count() <= 2,
            "the normal fill must be ONE loop (<=2 draw calls in its body), not unrolled:\n{src}"
        );
        assert!(b.len() < 4000, "a d=20 rotation shader must stay small (loops), got {} bytes", b.len());
    }

    /// A `Gather` (`xs[i]`, a *random* index into a table of scalar nodes) lowers to a local table
    /// plus the same clamped/NaN-screened read as `ArrIndex`.
    #[test]
    fn a_gather_lowers_to_a_table_read() {
        let (g, root) = g_with(|g| {
            let a = g.push(RvNode::ConstNum(10.0), RvKind::Num);
            let bb = g.push(RvNode::ConstNum(20.0), RvKind::Num);
            let idx = g.push(
                RvNode::Src(Source::UniformInt { lo: 0.0, hi: 1.0 }),
                RvKind::Num,
            );
            g.push(RvNode::Gather { elems: Box::new([a, bb]), index: idx }, RvKind::Num)
        });
        let src = emit(&g, &[root]).expect("a gather must lower");
        let b = body(&src);
        assert!(b.contains("array<f32, 2u>"), "the table must be materialized:\n{src}");
        assert!(b.contains("nz_isnan("), "the read must screen a NaN index:\n{src}");
    }

    /// Comparisons must screen NaN by **bits**, never by a float identity.
    ///
    /// The vendors enable fast-math by default, which assumes NaNs don't exist — on Metal `x == x`
    /// folds to `true`. `log(X) == log(X)` is exactly how the language asks "is X in the domain", and
    /// that fold silently turned a 0.6 into a 1.0 until the GPU conformance harness caught it. An
    /// integer bit test has no float identity for the optimizer to exploit.
    #[test]
    fn comparisons_screen_nan_by_bits_not_by_float_identity() {
        let (g, root) = g_with(|g| {
            let u = g.push(
                RvNode::Src(Source::Uniform(Uniform { lo: -2.0, hi: 3.0 })),
                RvKind::Num,
            );
            let l = g.push(RvNode::Unary(UnOp::Ln, u), RvKind::Num);
            g.push(RvNode::Binary(BinOp::Eq, l, l), RvKind::Bool)
        });
        let src = emit(&g, &[root]).expect("lowers");
        assert!(
            body(&src).contains("nz_isnan("),
            "an equality must screen its operands for NaN, or fast-math folds it to `true`:\n{src}"
        );
    }

    /// Explicit `sin`/`cos` must NOT lower to WGSL's built-ins (G1b).
    ///
    /// The built-in is only *guaranteed* on [-pi, pi], and past that it is not merely imprecise:
    /// Metal returns 0 for `sin(1e12)` against the engine's 0.0056. `nz_sin`/`nz_cos` reduce with
    /// integer Payne-Hanek — exact for every f32 — and then run the engine's own kernels. This test
    /// pins the substitution; `payne_hanek_holds_where_the_builtin_gives_up` proves it on device.
    #[test]
    fn explicit_trig_lowers_to_the_exact_reduction_not_the_builtin() {
        let (g, root) = g_with(|g| {
            let u = g.push(
                RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                RvKind::Num,
            );
            g.push(RvNode::Unary(UnOp::Sin, u), RvKind::Num)
        });
        let src = emit(&g, &[root]).expect("explicit trig lowers");
        let b = body(&src);
        assert!(b.contains("nz_sin("), "must call the exact reduction:\n{src}");
        assert!(
            !b.contains(" sin(") && !b.contains("=sin("),
            "must not call WGSL's built-in `sin` — it is wrong past [-pi, pi]:\n{src}"
        );

        // The Gaussian draw's trig is different: it lives inside the prelude's `src_normal` with
        // theta in [0, 2pi), always inside the built-in's guaranteed range. It stays a built-in.
        let (g, root) = g_with(|g| {
            g.push(
                RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }),
                RvKind::Num,
            )
        });
        let src = emit(&g, &[root]).expect("a Gaussian draw lowers");
        assert!(body(&src).contains("src_normal("));
    }

    /// Constants are transcribed by their **bits**, never by a decimal literal — a decimal is a
    /// re-rounding, and draw parity is decided at exactly that precision.
    #[test]
    fn constants_are_emitted_exactly() {
        let (g, root) = g_with(|g| g.push(RvNode::ConstNum(1.0 / 3.0), RvKind::Num));
        let src = emit(&g, &[root]).expect("lowers");
        let want = format!("{:#010x}", (1.0f32 / 3.0).to_bits());
        assert!(src.contains(&want), "expected exact bits {want} in:\n{src}");
    }

    /// Multi-root (the joint drivers): one column per root, sharing one instruction stream — so a
    /// source feeding two roots is drawn ONCE and both roots see the same lane value.
    #[test]
    fn joint_roots_emit_one_column_each_and_share_their_draws() {
        let (g, roots) = {
            let mut g = RvGraph::default();
            let x = g.push(
                RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }),
                RvKind::Num,
            );
            let two = g.push(RvNode::ConstNum(2.0), RvKind::Num);
            let y = g.push(RvNode::Binary(BinOp::Mul, x, two), RvKind::Num);
            (g, vec![x, y])
        };
        let src = emit(&g, &roots).expect("lowers");
        assert!(src.contains("out[0u * P.n + i]") && src.contains("out[1u * P.n + i]"));
        assert_eq!(
            body(&src).matches("src_normal(").count(),
            1,
            "the shared source must be drawn ONCE for both roots (joint sampling):\n{src}"
        );
    }
}

/// **The conformance harness: run the shaders we generate, on a real GPU, against the interpreter.**
///
/// A shader that merely *parses* proves nothing, and a hand-transcribed WGSL kernel (what the G0
/// spike used) proves something about a shader we don't ship. These tests take the emitter's own
/// output, dispatch it, and compare it lane-for-lane with the columnar interpreter — the oracle every
/// other backend is checked against.
///
/// Held to the two tiers G0 established, and it matters which is which:
///   * **Tier 1 — the draws are bit-identical.** Integer arithmetic; nothing to contract. This is the
///     tier the RNG certification lives in, so it is asserted as exact equality.
///   * **Tier 2 — lane arithmetic is ULP-close.** The GPU fuses multiply-add and there is no portable
///     way to stop it, so exact equality is *unattainable* here and demanding it would be a test that
///     can only fail. Bounded instead, at 1e-6 absolute.
///
/// Skipped (not failed) when no GPU adapter exists, so CI on a headless box stays green.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests {
    use super::*;
    use crate::backend::{Backend, InterpBackend};
    use crate::conformance;
    use crate::eval::Engine;
    use crate::simplify::simplify;

    const LANES: u32 = 4096;

    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    /// `None` when the machine has no GPU — the tests then skip rather than fail.
    fn gpu() -> Option<Gpu> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some(Gpu { device, queue })
    }

    /// Compile and dispatch a generated shader; return its column.
    fn run(gpu: &Gpu, wgsl: &str, key: crate::rng::Key, lane0: u32, n: u32) -> Vec<f32> {
        gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(err) = pollster::block_on(gpu.device.pop_error_scope()) {
            panic!("the emitter produced invalid WGSL:\n{err}\n\n{wgsl}");
        }

        let bytes = u64::from(n) * 4;
        let params: [u32; 4] = [key.k0, key.k1, lane0, n];
        // SAFETY: `[u32; 4]` is 16 contiguous bytes, no padding, every bit pattern valid as `u8`.
        let pbytes = unsafe { std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16) };
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&ubuf, 0, pbytes);
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, bytes);
        gpu.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let data = slice.get_mapped_range();
        let col: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        drop(data);
        staging.unmap();
        col
    }

    /// The interpreter's column for the same program and lanes — the oracle.
    fn interp_col(eng: &Engine, root: RvId, seed: u64, n: usize) -> Vec<f32> {
        let g = eng.graph();
        let prog = InterpBackend.compile(g, root, n);
        let mut r = prog.runner(std::sync::Arc::from(&[] as &[f64]));
        let mut out = Vec::with_capacity(n);
        let cap = r.batch_cap();
        let mut lane = 0u32;
        while out.len() < n {
            r.position(seed, lane);
            let take = cap.min(n - out.len());
            out.extend_from_slice(&r.next_batch(cap)[..take]);
            lane = lane.wrapping_add(cap as u32);
        }
        out
    }

    /// Emit `src`'s cone as WGSL and run it, alongside the interpreter's answer for the same lanes.
    fn both(gpu: &Gpu, src: &str, seed: u64) -> Option<(Vec<f32>, Vec<f32>)> {
        let mut eng = Engine::new();
        let root = match eng.run_rv(src).expect("program builds") {
            crate::Value::Dist(id) => id,
            other => panic!("expected a random variable, got {other:?}"),
        };
        let (sg, sroot) = simplify(eng.graph(), root);
        let wgsl = emit(&sg, &[sroot]).ok()?; // declined cones are not a failure
        let gpu_col = run(gpu, &wgsl, crate::rng::Key::from_seed(seed), 0, LANES);
        let cpu_col = interp_col(&eng, root, seed, LANES as usize);
        Some((gpu_col, cpu_col))
    }

    /// **Tier 1.** The draws themselves must be bit-identical to the interpreter's.
    ///
    /// Each case reads a source straight out, so no arithmetic stands between the hash and the
    /// comparison and there is nothing for the GPU to contract. This is what carries C0's 1 TB
    /// PractRand certification onto the GPU: the shader consumes the identical stream.
    #[test]
    fn tier1_the_draws_are_bit_identical_to_the_interpreter() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        // `unif(0,1)` is `0 + 1*u`, and `unif_int` is integer-valued: both are exact, so a
        // discrepancy here can only be the hash, the counter layout or the pair split.
        for (label, src) in [
            ("unif", "use rand; X ~ unif(0,1); X"),
            ("unif_int", "use rand; X ~ unif_int(1,6); X"),
            ("shaped unif", "use rand; xs ~[16] unif(0,1); xs[9]"),
            ("shaped unif_int", "use rand; xs ~[16] unif_int(1,6); xs[3]"),
        ] {
            let Some((g, c)) = both(&gpu, src, 7) else {
                panic!("{label}: the emitter declined a supported cone")
            };
            let exact = g.iter().zip(&c).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            assert_eq!(
                exact,
                LANES as usize,
                "{label}: only {exact}/{LANES} lanes bit-identical — the GPU is drawing a \
                 DIFFERENT stream than the interpreter, which would void the RNG certification"
            );
        }
    }

    /// **Tier 2.** Everything built on top of the draws, over the shared cross-backend corpus.
    ///
    /// Bounded, not exact: the GPU fuses multiply-add (G0), so `mu + sigma*z`, `a*b + c` and every
    /// Horner step round once where the CPU rounds twice. What must hold is that the gap stays at the
    /// level of f32 rounding rather than being a real algorithmic divergence — a wrong `round` tie
    /// rule or a truncated `%` would land whole units away, not 1e-6.
    #[test]
    fn tier2_lane_arithmetic_is_ulp_close_over_the_shared_corpus() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        let mut checked = 0;
        for (label, src) in conformance::CONST_CASES {
            let Some((g, c)) = both(&gpu, src, 3) else { continue }; // declined: fine, not this backend's job
            checked += 1;
            for (i, (&a, &b)) in g.iter().zip(&c).enumerate() {
                // Exact agreement (including the infinities `sqrt(+inf)` and `1/0` produce) passes
                // outright — `inf - inf` is NaN, not a difference, and would fail a naive tolerance.
                if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                    continue;
                }
                assert!(
                    a.is_finite() && b.is_finite(),
                    "{label}, lane {i}: gpu {a} vs interp {b} — one is non-finite and they disagree"
                );
                let d = (a - b).abs();
                let tol = 1e-6 * b.abs().max(1.0); // relative where the value is large
                assert!(
                    d <= tol,
                    "{label}, lane {i}: gpu {a} vs interp {b} (|Δ| {d:e} > {tol:e}) — that is not \
                     f32 rounding, it is a semantic difference"
                );
            }
        }
        assert!(checked > 10, "only {checked} corpus cases reached the GPU — the gate is too tight");
    }

    /// **Payne-Hanek, on device, exactly where the built-in gives up (G1b).**
    ///
    /// This is the test the reduction exists for. `sin`/`cos` are only *guaranteed* on [-pi, pi];
    /// past that a vendor may do anything, and Metal does the worst thing — it returns a confident
    /// wrong answer (0, for `sin(1e12)`). The naive fix, `x % TAU`, cannot work either: at 1e12 the
    /// quotient `x/(pi/2)` needs 40 mantissa bits and f32 has 24, so the reduced argument is pure
    /// rounding noise. Only an exact integer reduction survives here.
    ///
    /// So: sweep the decades, well past f32's ability to even *represent* the neighbouring multiple
    /// of pi/2, and demand agreement with the interpreter (which computes in f64 and rounds — the
    /// engine's stated contract). A backend that quietly used the built-in fails at 1e12; one that
    /// used `%` fails around 1e5.
    #[test]
    fn payne_hanek_holds_where_the_builtin_gives_up() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        // Straddle TRIG_MAX (1024) so both reduction branches run, and go up to 1e30 — where a
        // single f32 ulp is ~1e23, vastly wider than a period.
        for mag in ["1e3", "1e5", "1e7", "1e12", "1e18", "1e24", "1e30"] {
            for f in ["sin", "cos"] {
                // `unif(0,1)` spreads the arguments across the decade, so this is thousands of
                // distinct large arguments per case, not one lucky point.
                let src = format!("use rand; use math; X ~ unif(0,1); math::{f}({mag} * X)");
                let (g, c) = both(&gpu, &src, 7).expect("explicit trig must lower now, not decline");
                for (i, (&a, &b)) in g.iter().zip(&c).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-6,
                        "{f}({mag} * X), lane {i}: gpu {a} vs interp {b} — the reduction lost the \
                         argument (|Δ| {:e})",
                        (a - b).abs()
                    );
                }
            }
        }
    }

    /// The RNG half of the corpus: the emitted kernel must agree with the interpreter *in
    /// distribution* (the draws are shared, so the means should track closely).
    #[test]
    fn tier2_rng_cases_match_the_interpreter_in_distribution() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        for (label, src, seed) in conformance::RNG_CASES {
            let Some((g, c)) = both(&gpu, src, *seed) else { continue };
            let mean = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
            let (mg, mc) = (mean(&g), mean(&c));
            // Both columns are the SAME draws, so this is far tighter than a two-sample test: the
            // per-lane values agree to ~1e-6, hence so must the means.
            let scale = mc.abs().max(1.0);
            assert!(
                (mg - mc).abs() / scale < 1e-4,
                "{label}: gpu mean {mg} vs interp mean {mc} — same draws, so these must nearly \
                 coincide; a gap means the arithmetic diverged"
            );
        }
    }

    /// The whole point, end to end: `barrier_option`'s shape. 52 shaped normals folded by a sum —
    /// one draw loop in the shader, and it must still be the interpreter's answer.
    #[test]
    fn a_shaped_draw_kernel_agrees_with_the_interpreter() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        let src = "use rand; use vec; zs ~[52] normal(0, 1); vec::sum(zs)";
        let (g, c) = both(&gpu, src, 11).expect("a shaped normal cone must lower");
        let worst = g
            .iter()
            .zip(&c)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-3,
            "max |Δ| {worst:e} on a sum of 52 normals (|x| ~ 7) — f32 rounding over 52 terms is \
             ~1e-5; anything larger means the loop is drawing the wrong ordinals"
        );
    }

    /// **The G4 unlock, on device and BIT-IDENTICAL: a permutation.** `prisoners` is a program of
    /// permutations, and the reason it can move to the GPU under the *same* lane-for-lane contract as
    /// everything else — rather than the looser statistical one a rotation would need — is that
    /// Fisher–Yates is integer arithmetic: the same cell stream, the same Lemire bounded draws, the
    /// same swaps. So the shuffled deck must match the interpreter's exactly, not merely in
    /// distribution. A read of *every* element pins the whole permutation, not just one entry.
    #[test]
    fn a_permutation_kernel_matches_the_interpreter() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        // Read each of the 20 boxes; if any lane's shuffle diverged, some element read disagrees.
        for k in [0usize, 1, 7, 13, 19] {
            let src = format!("use rand; d ~ rand::permutation(20); d[{k}]");
            let (g, c) = both(&gpu, &src, 5).expect("a permutation cone must lower now");
            let exact = g.iter().zip(&c).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            assert_eq!(
                exact, LANES as usize,
                "perm[{k}]: only {exact}/{LANES} lanes bit-identical — the GPU Fisher–Yates diverged \
                 from the interpreter's"
            );
        }
    }

    /// **A rotation is a DISTRIBUTION-contract draw, and this is the test that says so (G4b).**
    ///
    /// Every other supported node is checked lane-for-lane — bit-identical for the integer draws,
    /// ULP-close for the arithmetic. Rotation cannot be: the two backends run the same f32 Gram–
    /// Schmidt, but the normal fill's `ln`/`sin`/`cos` are WGSL built-ins on one side and `approx`
    /// polynomials on the other (ULPs apart), the GPU may fuse the MGS dots where the CPU does not,
    /// and — measured — those low bits do **not** stay small: MGS occasionally lands on a near-singular
    /// Gaussian matrix where it is ill-conditioned, and there it turns a ULP into a finite rotation of
    /// the output basis. The worst lane here differs by ~2.6e-2, not 1e-6.
    ///
    /// So the contract is distributional: the two are the *same random rotation law*, not the same
    /// matrix. This test pins exactly that — element moments agree (mean 0, second moment 1/d, the
    /// Haar signature) even though the elements themselves don't — which is the property turboquant
    /// (and any honest use of a random rotation) actually depends on.
    #[test]
    fn a_rotation_matches_the_interpreter_in_distribution() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        let mean = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
        let m2 = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / v.len() as f64;
        let mut lanewise_worst = 0.0f32;
        for (i, j) in [(0usize, 0usize), (2, 3), (4, 4), (1, 4), (3, 0)] {
            let src = format!("use rand; Pi ~ rand::rotation(5); Pi[{i}][{j}]");
            let (g, c) = both(&gpu, &src, 5).expect("a rotation cone must lower");
            // Same distribution: element mean ≈ 0 and second moment ≈ 1/d = 0.2, and the two backends
            // agree on both to within Monte Carlo error (SE ~ 0.007 on 4096 lanes).
            assert!((mean(&g) - mean(&c)).abs() < 0.03, "element ({i},{j}) means diverge: {} vs {}", mean(&g), mean(&c));
            assert!((m2(&g) - m2(&c)).abs() < 0.03, "element ({i},{j}) 2nd moments diverge: {} vs {}", m2(&g), m2(&c));
            assert!((m2(&g) - 0.2).abs() < 0.03, "element ({i},{j}) 2nd moment {} != 1/d", m2(&g));
            let w = g.iter().zip(&c).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
            lanewise_worst = lanewise_worst.max(w);
        }
        // And confirm the point of the whole test: lane-for-lane they genuinely differ (this is a
        // distribution contract, not a lane one). If this ever collapses to ~0, the backends became
        // bit-identical and the softer distributional asserts above are hiding a stronger truth.
        assert!(
            lanewise_worst > 1e-4,
            "rotations are unexpectedly lane-identical ({lanewise_worst:e}); revisit the contract"
        );
    }

    /// A `Gather` (`xs[i]`, random index) on device: the drawn index and the clamped, NaN-screened
    /// read are all integer/exact, so this too is bit-identical to the interpreter.
    #[test]
    fn a_gather_kernel_matches_the_interpreter() {
        let Some(gpu) = gpu() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        let src = "use rand; i ~ unif_int(0, 3); [10, 20, 30, 40][i]";
        let (g, c) = both(&gpu, src, 9).expect("a gather cone must lower");
        let exact = g.iter().zip(&c).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        assert_eq!(exact, LANES as usize, "gather: only {exact}/{LANES} lanes bit-identical");
    }
}
