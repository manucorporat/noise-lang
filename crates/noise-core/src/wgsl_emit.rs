//! WGSL emitter — the **fourth** lowering of the `RvGraph` (PLAN-WEBGPU G1).
//!
//! Same walk as [`crate::jit`] and [`crate::wasm_emit`]: post-order over the simplified cone, one
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
//!    *loses* to the multicore JIT end to end; as one draw loop over a block of consecutive ordinals
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
use crate::dist::{RvGraph, RvId, RvNode, Source};
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
    let plan = plan_blocks(graph, roots)?;
    let mut e = Emitter {
        graph,
        ords: &ords,
        plan,
        memo: HashMap::new(),
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
            // Out of scope for G1 — the gate keeps them off this backend, and G4 is where the GPU
            // *beats* the CPU codegen on exactly these (WGSL has array indexing and divergent loops).
            RvNode::Src(Source::Poisson { .. }) => return Err(Unsupported("poisson")),
            RvNode::Permutation { .. } => return Err(Unsupported("permutation")),
            RvNode::Rotation { .. } => return Err(Unsupported("rotation")),
            RvNode::ArrIndex { .. } => return Err(Unsupported("array index")),
            RvNode::Gather { .. } => return Err(Unsupported("gather")),
            RvNode::ArrDraw { src, .. } => {
                if matches!(src, Source::Poisson { .. }) {
                    return Err(Unsupported("poisson"));
                }
            }
            RvNode::ArrElem { arr, .. } => {
                *readers.entry(*arr).or_default() += 1;
                stack.push(*arr);
            }
            // **Explicit `sin`/`cos` are declined — G1's one scope cut, and it is a correctness
            // cut, not a performance one.** The engine's contract for large arguments is "compute in
            // f64 and round to f32": past `approx::TRIG_MAX_F32` the 2-term Cody-Waite reduction
            // falls apart, so all three CPU backends hand off to the f64 library (finding C3).
            // **WGSL has no f64**, so that fallback cannot be reproduced — and WGSL only *guarantees*
            // its built-in `sin`/`cos` on [-pi, pi] anyway. Measured, `sin(1e12 * X)` returns 0 on
            // Metal against the interpreter's 0.0056: not a rounding gap, a wrong answer.
            //
            // Declining is the safe answer (the cone falls back to a correct backend, exactly as
            // `wasm_host` does for what it can't emit) and it is *tested*, so it cannot rot into a
            // silent divergence. The fix is an exact integer Payne-Hanek range reduction in the
            // shader, which would close this and hand `am_vs_fm` to the GPU — that is G1b.
            //
            // Note this does NOT block the normal draw: Box-Muller's `sin`/`cos` live inside the
            // prelude's `src_normal` with theta in [0, 2pi), always inside the built-in's guaranteed
            // range. `barrier_option` and every Gaussian model still lower.
            RvNode::Unary(UnOp::Sin | UnOp::Cos, _) => return Err(Unsupported("trig")),
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
    plan: Plan,
    memo: HashMap<RvId, String>,
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
            _ => vec![],
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
                    // Declined in `plan_blocks` — see the note there. (`src_normal`'s Box-Muller
                    // trig is in the prelude, not a graph node, and is unaffected.)
                    UnOp::Sin | UnOp::Cos => unreachable!("plan_blocks declines explicit trig"),
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
fn f32c(x: f32) -> String {
    format!("bitcast<f32>({:#010x}u)", x.to_bits())
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
        // ties-AWAY, not WGSL's ties-to-even `round`.
        assert!(src.contains("floor(abs("), "round must be ties-away: {src}");
        assert!(!src.contains("round("), "must not call WGSL's round: {src}");
        // FLOORED mod, not WGSL's truncated `%`.
        assert!(src.contains("floor(v"), "mod must be floored: {src}");
        assert!(!src.contains(" % "), "must not use WGSL's truncated %: {src}");
    }

    /// The cones this backend can't lower are declined, not mis-emitted — the caller falls back to a
    /// correct backend exactly as `wasm_host` does. (G4 is where WGSL *beats* the CPU codegen on
    /// these: it has array indexing and divergent loops.)
    #[test]
    fn unsupported_cones_are_declined() {
        let (g, root) = g_with(|g| {
            g.push(RvNode::Src(Source::Poisson { lambda: 3.0 }), RvKind::Num)
        });
        assert_eq!(emit(&g, &[root]), Err(Unsupported("poisson")));

        let (g, root) = g_with(|g| g.push(RvNode::Permutation { n: 8 }, RvKind::Arr(8)));
        assert_eq!(emit(&g, &[root]), Err(Unsupported("permutation")));
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

    /// Explicit `sin`/`cos` are declined — G1's one scope cut, and a *correctness* one.
    ///
    /// The engine's large-argument trig computes in f64 and rounds (finding C3). WGSL has no f64, so
    /// that cannot be reproduced, and the built-in is only guaranteed on [-pi, pi]: measured,
    /// `sin(1e12 * X)` returns 0 on Metal against the interpreter's 0.0056. Declining sends the cone
    /// to a correct backend; the fix is an exact Payne-Hanek reduction in the shader (G1b).
    ///
    /// It must NOT take the normal draw with it: Box-Muller's trig lives in the prelude with theta in
    /// [0, 2pi), so every Gaussian model still lowers. That is the second half of this test.
    #[test]
    fn explicit_trig_is_declined_but_the_normal_draw_is_not() {
        let (g, root) = g_with(|g| {
            let u = g.push(
                RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
                RvKind::Num,
            );
            g.push(RvNode::Unary(UnOp::Sin, u), RvKind::Num)
        });
        assert_eq!(emit(&g, &[root]), Err(Unsupported("trig")));

        let (g, root) = g_with(|g| {
            g.push(
                RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }),
                RvKind::Num,
            )
        });
        let src = emit(&g, &[root]).expect("a Gaussian draw must still lower — its trig is internal");
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
        let mut r = prog.runner();
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
}
