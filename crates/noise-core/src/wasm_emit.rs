//! WASM-emitter backend (PLAN.md Phase 4 "Browser note") — the browser's twin of [`crate::jit`].
//!
//! A WASM sandbox can't emit or run native code, so the Cranelift JIT (B1/B2) is native-only. What
//! *is* portable is the [`RvGraph`] IR. This module is the second lowering of that IR: it walks the
//! identical graph the identical post-order way and emits the identical fused, multi-stream kernel —
//! only the per-node encoding differs (`f64.mul` instead of `mulsd`, an inlined xoshiro step in
//! either). The shared "what the graph means" lives in [`crate::kernel`] (the cost/profitability
//! gate, stream policy, seed layout); this file is just "how to spell it in wasm bytes".
//!
//! **Output.** [`emit`] produces a complete WebAssembly module (`Vec<u8>`) exporting:
//!   * `memory` — one linear memory the host reads/writes.
//!   * `kernel(out: i32, n: i32, state: i32)` — fills `out[0..n]` with fresh root draws (`f64`s at
//!     `out`), reading/writing the `4 * streams`-word xoshiro state at `state`. Same ABI shape as the
//!     native kernel, but addresses are `i32` offsets into the module's linear memory.
//!
//! `ln`/`sin`/`cos` are **inlined as polynomials** ([`emit_ln`]/[`emit_trig`], the same
//! [`crate::approx`] reference the JIT uses) — wasm has no native ones, and a host call would cross
//! the JS boundary per draw. The module imports `atan`/`round`/`pow` from module `"m"`, plus
//! `sin`/`cos` (the large-argument fallback past `approx::TRIG_MAX` — finding C3) and `exp` (matched
//! to the interpreter — finding C9); the browser supplies them via `Math.*` and the test harness via
//! Rust `f64` methods. The inline `sin`/`cos` polynomial handles every ordinary argument, so those
//! imports fire only on the rare huge-argument path. Because the transcendentals are inlined,
//! [`emit_for`] passes `inline_trans = true` to
//! [`kernel::profitable`] — the gate decision matches the native JIT, and the 2× win reaches the
//! browser. [`kernel::choose_streams`] still keeps these single-stream (throughput-, not
//! latency-bound).
//!
//! **Why no host execution yet.** Running an emitted module means a JS host `instantiate`s it and
//! drives it (the `Backend`/`Program`/`Runner` seam, browser-side). That wiring is a follow-up; this
//! cut is the emitter plus a native parity harness (tests run emitted modules through an embedded
//! wasm interpreter and check distribution parity with the interpreter, mirroring `jit`'s tests).
//!
//! Scope mirrors the JIT: `unif`/`unif_int`/`normal`/`exp`/`geometric` sources, `+ - * /`,
//! integer-constant `^`, comparisons, `&& ||`, unary `- !` and the math ufuncs, and lifted `if`.
//! `Poisson` stays interpreter-only (rejected by the gate). `unif_int` uses the float method
//! (`lo + floor(u * count)`) rather than the native kernel's Lemire multiply-high — wasm has no
//! 64×64→128 `mulhi`, and the float form is identical in distribution (what the parity tests check).

use std::collections::HashMap;

use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    Ieee64, ImportSection, InstructionSink, MemArg, MemorySection, MemoryType, Module, TypeSection,
    ValType,
};

use crate::ast::{BinOp, UnOp};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId, RvNode, Source};
use crate::kernel::{choose_streams, cone_size, const_int_exponent};

// --- imported-function indices (declared in import order, before the local `kernel`) ---
// `ln` is inlined as a polynomial (`emit_ln`, mirroring `jit`). `sin`/`cos` are inlined too, but
// the module re-imports them for the **large-argument fallback**: past `approx::TRIG_MAX` the 2-term
// reduction degrades, so `emit_trig` defers to the host's accurate `sin`/`cos` there (finding C3).
// `exp` is imported (rather than lowered as `pow(e, x)`) so it matches the interpreter's `exp`
// (finding C9). The host (`wasm_host` / the test linker) supplies all of these via `Math.*`.
const ATAN: u32 = 0;
const ROUND: u32 = 1;
const POW: u32 = 2;
const SIN: u32 = 3;
const COS: u32 = 4;
const EXP: u32 = 5;
const N_IMPORTS: u32 = 6;

// --- fixed local indices (params occupy 0..3) ---
const OUT: u32 = 0; // param: output base pointer
const N: u32 = 1; // param: sample count
const STATE: u32 = 2; // param: xoshiro-state base pointer
const I: u32 = 3; // loop counter (sample index)
                  // Four scratch i64s for one xoshiro step (shared across streams — the step is straight-line).
const RES: u32 = 4;
const T: u32 = 5;
const S2A: u32 = 6;
const S3A: u32 = 7;
/// First local index of a stream's four state words; stream `j` word `k` is at `STATE_BASE + j*4 + k`.
const STATE_BASE: u32 = 8;

/// 8-byte aligned access (f64/i64) into memory 0 at an absolute address already on the stack.
fn mem8(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 3,
        memory_index: 0,
    }
}

/// f64 literal as the `Ieee64` the encoder wants.
fn f64c(x: f64) -> Ieee64 {
    Ieee64::from(x)
}

/// Local holding stream `j`'s xoshiro word `k`.
fn sl(j: usize, k: usize) -> u32 {
    STATE_BASE + (j * 4 + k) as u32
}

/// Scratch locals the inlined transcendentals reuse: 2 i64 + 6 f64, shared across every `ln`/`sin`/
/// `cos` evaluation (a transcendental is emitted atomically — it `local.set`s its input, computes
/// from these, and leaves one result — so a single shared pool is safe, like the xoshiro scratch).
const T_I64: u32 = 2;
const T_F64: u32 = 6;

/// Per-emission constants shared across the whole kernel body.
struct Ctx<'g> {
    graph: &'g RvGraph,
    /// Base index of the f64 value-slot block; node memo locals live above this.
    fbase: u32,
    /// Distinct nodes per stream (each stream gets its own contiguous block of f64 slots).
    cone: u32,
    /// Base index of the [`T_I64`] transcendental i64 scratch locals.
    ti: u32,
    /// Base index of the [`T_F64`] transcendental f64 scratch locals.
    tf: u32,
}

/// Emit a complete WASM module computing `root` with the given RNG stream count. `streams` must
/// divide [`BATCH`] (so a batch is a whole number of loop iterations) and be ≥ 1. The graph must be
/// codegen-supported (no `Poisson`); it **panics** (assert / `unreachable!`) on an ungated graph, so
/// it is `pub(crate)` — the public entry point is the gate-honoring [`emit_for`] (finding C10).
pub(crate) fn emit(graph: &RvGraph, root: RvId, streams: usize) -> Vec<u8> {
    assert!(
        streams >= 1 && BATCH.is_multiple_of(streams),
        "streams must divide BATCH"
    );
    let cone = cone_size(graph, root) as u32;

    // --- types: (f64)->f64 for the unary shims, (f64,f64)->f64 for pow, kernel sig ---
    let mut types = TypeSection::new();
    types.ty().function([ValType::F64], [ValType::F64]); // 0: unary math
    types
        .ty()
        .function([ValType::F64, ValType::F64], [ValType::F64]); // 1: pow
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], []); // 2: kernel

    // --- imports from module "m": unary `atan`/`round`/`sin`/`cos`/`exp` (type 0) + `pow` (type 1).
    // `sin`/`cos` are the large-argument fallback (finding C3); `exp` matches the interpreter (C9).
    // The declaration order fixes the indices `ATAN..EXP` above — keep them in sync.
    let mut imports = ImportSection::new();
    for name in ["atan", "round"] {
        imports.import("m", name, EntityType::Function(0));
    }
    imports.import("m", "pow", EntityType::Function(1));
    for name in ["sin", "cos", "exp"] {
        imports.import("m", name, EntityType::Function(0));
    }

    // --- the local kernel function (index N_IMPORTS, type 2) ---
    let mut functions = FunctionSection::new();
    functions.function(2);

    // --- one linear memory, sized to hold the largest output we emit (BATCH f64s) plus headroom ---
    let mut memories = MemorySection::new();
    let bytes = 4096 + BATCH * 8; // host convention: state low, output at/after 4096
    let pages = bytes.div_ceil(64 * 1024).max(1) as u64;
    memories.memory(MemoryType {
        minimum: pages,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("kernel", ExportKind::Func, N_IMPORTS);

    // --- code: declare locals, then emit the loop body ---
    let i64_count = 4 + 4 * streams as u32; // 4 scratch + 4 state words per stream
    let f64_count = cone * streams as u32; // one value slot per node, per stream
    let mut func = Function::new([
        (1, ValType::I32),                 // I (counter)
        (i64_count + T_I64, ValType::I64), // xoshiro scratch + state words + transcendental i64
        (T_F64 + f64_count, ValType::F64), // transcendental f64 scratch + node value slots
    ]);
    // Layout (indices): I=3, then the i64 block from `RES` (4): 4 xoshiro scratch, `4*streams` state
    // words, then `T_I64` transcendental scratch; then the f64 block: `T_F64` transcendental scratch,
    // then the node value slots. Each group is contiguous in declaration order.
    let ti = RES + i64_count; // transcendental i64 scratch (after xoshiro scratch + state words)
    let tf = ti + T_I64; // first f64 local (transcendental f64 scratch)
    let ctx = Ctx {
        graph,
        fbase: tf + T_F64,
        cone,
        ti,
        tf,
    };
    {
        let mut s = func.instructions();
        emit_kernel(&mut s, &ctx, root, streams);
    }

    let mut code = CodeSection::new();
    code.function(&func);

    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&functions)
        .section(&memories)
        .section(&exports)
        .section(&code);
    module.finish()
}

/// Emit a module for the default stream count chosen by the shared policy ([`kernel::choose_streams`]
/// — multi-stream only for purely-inline graphs). Returns `None` for a graph the backend won't emit
/// (e.g. `Poisson`); the browser keeps the interpreter for those, exactly like the native gate.
///
/// `inline_trans = true`: this emitter inlines `ln`/`sin`/`cos` as polynomials (`emit_ln`/`emit_trig`,
/// same `crate::approx` reference as the JIT), so `normal`/`exp`/trig graphs are fusible here too and
/// the 2× transcendental win reaches the browser — the gate decision now matches the native JIT.
pub fn emit_for(graph: &RvGraph, root: RvId) -> Option<(Vec<u8>, usize)> {
    if !crate::kernel::profitable(graph, root, /* inline_trans */ true) {
        return None;
    }
    let streams = choose_streams(graph, root);
    Some((emit(graph, root, streams), streams))
}

/// The kernel skeleton: load each stream's state, run a counted loop emitting `streams` samples per
/// iteration, store the state back. Mirrors `jit::build_with`.
fn emit_kernel(s: &mut InstructionSink, ctx: &Ctx, root: RvId, streams: usize) {
    // Load each stream's four state words from the strided seed layout `state[(k*streams+j)]`.
    for j in 0..streams {
        for k in 0..4 {
            let off = ((k * streams + j) * 8) as u64;
            s.local_get(STATE).i64_load(mem8(off)).local_set(sl(j, k));
        }
    }
    s.i32_const(0).local_set(I);

    // block { loop { if I >= N: break; <body>; I += streams; continue } }
    s.block(BlockType::Empty);
    s.loop_(BlockType::Empty);
    s.local_get(I).local_get(N).i32_ge_s().br_if(1); // break to the block end

    for j in 0..streams {
        // Address of out[I + j]: out + (I + j) * 8, pushed before the value so f64.store sees both.
        s.local_get(OUT)
            .local_get(I)
            .i32_const(j as i32)
            .i32_add()
            .i32_const(8)
            .i32_mul()
            .i32_add();
        // Emit this stream's draw (own memo → independent draws), leaving its value in a local.
        let mut memo: HashMap<RvId, u32> = HashMap::new();
        let mut slot = 0u32;
        let lroot = emit_node(s, ctx, j, root, &mut memo, &mut slot);
        s.local_get(lroot).f64_store(mem8(0));
    }

    s.local_get(I)
        .i32_const(streams as i32)
        .i32_add()
        .local_set(I);
    s.br(0); // continue the loop
    s.end(); // end loop
    s.end(); // end block

    // Write each stream's advanced state back to its slot.
    for j in 0..streams {
        for k in 0..4 {
            let off = ((k * streams + j) * 8) as u64;
            s.local_get(STATE).local_get(sl(j, k)).i64_store(mem8(off));
        }
    }
    s.end(); // end function
}

/// Emit node `id` for stream `j`, memoizing each `RvId` into its own f64 local so a shared sub-RV is
/// computed once (the CSE guarantee the interpreter/JIT also give). Children are emitted first (each
/// `local.set`s its slot and leaves the stack untouched); the parent then `local.get`s them. Returns
/// the local index holding this node's value. Each stream uses a fresh memo (independent draws).
fn emit_node(
    s: &mut InstructionSink,
    ctx: &Ctx,
    j: usize,
    id: RvId,
    memo: &mut HashMap<RvId, u32>,
    slot: &mut u32,
) -> u32 {
    if let Some(&l) = memo.get(&id) {
        return l;
    }
    // Each branch leaves exactly one f64 on the stack; we `local.set` it into this node's slot below.
    match ctx.graph.node(id) {
        RvNode::Src(Source::Uniform(u)) => emit_uniform(s, j, u.lo, u.hi),
        RvNode::Src(Source::UniformInt { lo, hi }) => emit_uniform_int(s, j, *lo, *hi),
        RvNode::Src(Source::Normal { mu, sigma }) => emit_normal(s, ctx, j, *mu, *sigma),
        RvNode::Src(Source::Exp { rate }) => emit_exp(s, ctx, j, *rate),
        RvNode::Src(Source::Geometric { p }) => emit_geometric(s, ctx, j, *p),
        RvNode::Src(Source::Poisson { .. }) => unreachable!("profitable() excludes Poisson"),
        RvNode::Gather { .. } => unreachable!("profitable() excludes Gather"),
        RvNode::ConstNum(x) => {
            s.f64_const(f64c(*x));
        }
        RvNode::ConstBool(b) => {
            s.f64_const(f64c(if *b { 1.0 } else { 0.0 }));
        }
        RvNode::Unary(op, a) => {
            let la = emit_node(s, ctx, j, *a, memo, slot);
            emit_unary(s, ctx, *op, la);
        }
        RvNode::Binary(BinOp::Pow, a, b) => {
            let la = emit_node(s, ctx, j, *a, memo, slot);
            match const_int_exponent(ctx.graph, *b) {
                Some(k) => emit_pow(s, la, k), // repeated multiply, no call
                None => {
                    let lb = emit_node(s, ctx, j, *b, memo, slot);
                    s.local_get(la).local_get(lb).call(POW);
                }
            }
        }
        RvNode::Binary(op, a, b) => {
            let la = emit_node(s, ctx, j, *a, memo, slot);
            let lb = emit_node(s, ctx, j, *b, memo, slot);
            emit_binary(s, *op, la, lb);
        }
        RvNode::Select { cond, a, b } => {
            let lc = emit_node(s, ctx, j, *cond, memo, slot);
            let la = emit_node(s, ctx, j, *a, memo, slot);
            let lb = emit_node(s, ctx, j, *b, memo, slot);
            // wasm `select` pops [a, b, cond_i32] → a if cond != 0 else b.
            s.local_get(la)
                .local_get(lb)
                .local_get(lc)
                .f64_const(f64c(0.0))
                .f64_ne()
                .select();
        }
    }
    let l = ctx.fbase + (j as u32) * ctx.cone + *slot;
    *slot += 1;
    s.local_set(l);
    memo.insert(id, l);
    l
}

/// One xoshiro256++ step on stream `j`'s state locals, leaving the raw `u64` output on the stack.
/// Mirrors `Rng::next_u64` / `jit::emit_next_u64` (capturing the pre-mutation words before overwrite).
fn emit_next_u64(s: &mut InstructionSink, j: usize) {
    // res = rotl(s0 + s3, 23) + s0
    s.local_get(sl(j, 0))
        .local_get(sl(j, 3))
        .i64_add()
        .i64_const(23)
        .i64_rotl()
        .local_get(sl(j, 0))
        .i64_add()
        .local_set(RES);
    // t = s1 << 17
    s.local_get(sl(j, 1)).i64_const(17).i64_shl().local_set(T);
    // s2a = s2 ^ s0 ; s3a = s3 ^ s1  (capture before any overwrite)
    s.local_get(sl(j, 2))
        .local_get(sl(j, 0))
        .i64_xor()
        .local_set(S2A);
    s.local_get(sl(j, 3))
        .local_get(sl(j, 1))
        .i64_xor()
        .local_set(S3A);
    // s1 ^= s2a ; s0 ^= s3a  (s1 set first; original s0 still intact for s0's update)
    s.local_get(sl(j, 1))
        .local_get(S2A)
        .i64_xor()
        .local_set(sl(j, 1));
    s.local_get(sl(j, 0))
        .local_get(S3A)
        .i64_xor()
        .local_set(sl(j, 0));
    // s2 ^= t ; s3 = rotl(s3a, 45)
    s.local_get(S2A).local_get(T).i64_xor().local_set(sl(j, 2));
    s.local_get(S3A)
        .i64_const(45)
        .i64_rotl()
        .local_set(sl(j, 3));
    s.local_get(RES);
}

/// Uniform `f64` in `[0, 1)` from the top 53 bits — mirrors `Rng::next_f64`.
fn emit_next_f64(s: &mut InstructionSink, j: usize) {
    emit_next_u64(s, j);
    s.i64_const(11)
        .i64_shr_u()
        .f64_convert_i64_u()
        .f64_const(f64c(1.0 / ((1u64 << 53) as f64)))
        .f64_mul();
}

fn emit_uniform(s: &mut InstructionSink, j: usize, lo: f64, hi: f64) {
    emit_next_f64(s, j);
    s.f64_const(f64c(hi - lo))
        .f64_mul()
        .f64_const(f64c(lo))
        .f64_add();
}

/// `unif_int(lo, hi)` as `lo + min(floor(u * count), count - 1)` — uniform over `lo..=hi`. (The
/// native kernel uses Lemire multiply-high, but wasm lacks a 64×64→128 high-multiply; this is
/// identical in distribution.) The `min(·, count - 1)` clamp caps the top face: `u` is `< 1` but
/// `floor(u * count)` can still round up to `count` for a huge `count` where Lemire cannot, which
/// would yield an out-of-range `hi + 1` (finding C9). One extra `f64.min`.
fn emit_uniform_int(s: &mut InstructionSink, j: usize, lo: f64, hi: f64) {
    let count = (hi - lo + 1.0).max(1.0);
    emit_next_f64(s, j);
    s.f64_const(f64c(count))
        .f64_mul()
        .f64_floor()
        .f64_const(f64c(count - 1.0))
        .f64_min()
        .f64_const(f64c(lo))
        .f64_add();
}

/// Horner `Σ c[i]·z^i` (coeffs low→high) with `z` already in a local — mirrors `crate::approx::horner`
/// and `jit::emit_horner`. Leaves the polynomial value on the stack.
fn emit_horner(s: &mut InstructionSink, z: u32, coeffs: &[f64]) {
    s.f64_const(f64c(*coeffs.last().unwrap()));
    for &c in coeffs.iter().rev().skip(1) {
        s.local_get(z).f64_mul().f64_const(f64c(c)).f64_add();
    }
}

/// Inlined `ln(x)` for `x > 0` — wasm transcription of [`crate::approx::ln`] / `jit::emit_ln`.
/// Consumes the f64 input on top of the stack and leaves `ln(x)`, computing through the shared
/// transcendental scratch (so the surrounding stack is untouched).
fn emit_ln(s: &mut InstructionSink, ctx: &Ctx) {
    use crate::approx::LN_COEFFS;
    use std::f64::consts::{LN_2, SQRT_2};
    let (bits, e0) = (ctx.ti, ctx.ti + 1); // i64 scratch
    let (m0, m, ef, f, f2) = (ctx.tf, ctx.tf + 1, ctx.tf + 2, ctx.tf + 3, ctx.tf + 4); // f64 scratch
                                                                                       // bits = reinterpret(x)  (consumes the input f64)
    s.i64_reinterpret_f64().local_set(bits);
    // e0 = (bits >> 52 & 0x7ff) - 1023
    s.local_get(bits)
        .i64_const(52)
        .i64_shr_u()
        .i64_const(0x7ff)
        .i64_and()
        .i64_const(1023)
        .i64_sub()
        .local_set(e0);
    // m0 = reinterpret((bits & MANT) | EXP_ONE)  ∈ [1, 2)
    s.local_get(bits)
        .i64_const(0x000f_ffff_ffff_ffff)
        .i64_and()
        .i64_const(0x3ff0_0000_0000_0000_u64 as i64)
        .i64_or()
        .f64_reinterpret_i64()
        .local_set(m0);
    // m = (m0 > √2) ? m0*0.5 : m0   (select pops a, b, cond)
    s.local_get(m0).f64_const(f64c(0.5)).f64_mul();
    s.local_get(m0);
    s.local_get(m0).f64_const(f64c(SQRT_2)).f64_gt();
    s.select().local_set(m);
    // ef = ((m0 > √2) ? e0+1 : e0) as f64
    s.local_get(e0).i64_const(1).i64_add();
    s.local_get(e0);
    s.local_get(m0).f64_const(f64c(SQRT_2)).f64_gt();
    s.select().f64_convert_i64_s().local_set(ef);
    // f = (m-1)/(m+1); f2 = f*f
    s.local_get(m).f64_const(f64c(1.0)).f64_sub();
    s.local_get(m).f64_const(f64c(1.0)).f64_add();
    s.f64_div().local_set(f);
    s.local_get(f).local_get(f).f64_mul().local_set(f2);
    // ln(x) = 2·f·Σ cₖf²ᵏ + e·ln2
    s.f64_const(f64c(2.0)).local_get(f).f64_mul();
    emit_horner(s, f2, &LN_COEFFS);
    s.f64_mul();
    s.local_get(ef).f64_const(f64c(LN_2)).f64_mul();
    s.f64_add();
}

/// Range-guarded `cos`/`sin` — the inline polynomial ([`emit_trig_poly`]) for `|x| < TRIG_MAX`, else
/// the imported library `sin`/`cos` (finding C3), mirroring `jit::emit_trig` and
/// [`crate::approx::sin`]. Consumes the f64 input on top of the stack and leaves the result. (The
/// Box–Muller draw path calls [`emit_trig_poly`] directly — its argument is always `< 2π`.)
fn emit_trig(s: &mut InstructionSink, ctx: &Ctx, is_cos: bool) {
    use crate::approx::TRIG_MAX;
    let tx = ctx.tf; // stash the input (reused as the poly's accumulator in the else arm)
    s.local_set(tx);
    // if |x| >= TRIG_MAX { host sin/cos } else { inline poly }
    s.local_get(tx).f64_abs().f64_const(f64c(TRIG_MAX)).f64_ge();
    s.if_(BlockType::Result(ValType::F64));
    s.local_get(tx).call(if is_cos { COS } else { SIN });
    s.else_();
    emit_trig_poly(s, ctx, tx, is_cos);
    s.end();
}

/// The inline `cos`/`sin` polynomial body operating on the input already stashed in local `tx` —
/// wasm transcription of [`crate::approx::cos`]/`sin`: Cody–Waite reduce to `[-π/4, π/4]`, evaluate
/// both reduced kernels, pick by quadrant. Leaves the result on the stack. (`tx` is reused as the
/// quadrant-select accumulator once the input is dead.)
fn emit_trig_poly(s: &mut InstructionSink, ctx: &Ctx, tx: u32, is_cos: bool) {
    use crate::approx::{COS_COEFFS, PIO2_HI, PIO2_LO, SIN_COEFFS};
    use std::f64::consts::FRAC_2_PI;
    let ki = ctx.ti; // i64 quadrant
    let (kf, r, z, sinr, cosr) = (ctx.tf + 1, ctx.tf + 2, ctx.tf + 3, ctx.tf + 4, ctx.tf + 5);
    // kf = round(x·2/π); r = (x - kf·π/2_hi) - kf·π/2_lo
    s.local_get(tx)
        .f64_const(f64c(FRAC_2_PI))
        .f64_mul()
        .f64_nearest()
        .local_set(kf);
    s.local_get(tx)
        .local_get(kf)
        .f64_const(f64c(PIO2_HI))
        .f64_mul()
        .f64_sub()
        .local_get(kf)
        .f64_const(f64c(PIO2_LO))
        .f64_mul()
        .f64_sub()
        .local_set(r);
    s.local_get(r).local_get(r).f64_mul().local_set(z);
    // sin(r) = r + r·z·P_sin(z)
    s.local_get(r);
    s.local_get(r).local_get(z).f64_mul();
    emit_horner(s, z, &SIN_COEFFS);
    s.f64_mul().f64_add().local_set(sinr);
    // cos(r) = 1 - z/2 + z²·P_cos(z)
    s.f64_const(f64c(1.0))
        .local_get(z)
        .f64_const(f64c(0.5))
        .f64_mul()
        .f64_sub();
    s.local_get(z).local_get(z).f64_mul();
    emit_horner(s, z, &COS_COEFFS);
    s.f64_mul().f64_add().local_set(cosr);
    // kq = (kf as i64) & 3 — pick the kernel + sign per quadrant. Reuse `tx` as the accumulator.
    s.local_get(kf)
        .i64_trunc_sat_f64_s()
        .i64_const(3)
        .i64_and()
        .local_set(ki);
    // (kernel, negate) for quadrants 0..3.  cos: c,-s,-c,s   sin: s,c,-s,-c
    let q = if is_cos {
        [(cosr, false), (sinr, true), (cosr, true), (sinr, false)]
    } else {
        [(sinr, false), (cosr, false), (sinr, true), (cosr, true)]
    };
    let push_q = |s: &mut InstructionSink, (l, neg): (u32, bool)| {
        s.local_get(l);
        if neg {
            s.f64_neg();
        }
    };
    push_q(s, q[0]);
    s.local_set(tx); // res = q0
    for (n, &qn) in q.iter().enumerate().skip(1) {
        push_q(s, qn); // a = qn
        s.local_get(tx); // b = current res
        s.local_get(ki).i64_const(n as i64).i64_eq(); // cond = (kq == n)
        s.select().local_set(tx);
    }
    s.local_get(tx);
}

/// `N(mu, sigma^2)` via Box–Muller, one normal per draw (cosine arm) — mirrors `jit::emit_normal`.
/// `ln`/`cos` are inlined polynomials ([`emit_ln`]/[`emit_trig`]); `sqrt` is native.
fn emit_normal(s: &mut InstructionSink, ctx: &Ctx, j: usize, mu: f64, sigma: f64) {
    use std::f64::consts::TAU;
    // r = sqrt(-2 * ln(1 - u1))
    s.f64_const(f64c(1.0));
    emit_next_f64(s, j);
    s.f64_sub();
    emit_ln(s, ctx);
    s.f64_const(f64c(-2.0)).f64_mul().f64_sqrt();
    // result = mu + sigma * r * cos(TAU * u2). The angle is `TAU * u2 ∈ [0, 2π)` — always inside the
    // polynomial's range — so stash it and call `emit_trig_poly` directly (skip the range guard) to
    // keep the hot Box–Muller draw lean.
    emit_next_f64(s, j);
    s.f64_const(f64c(TAU)).f64_mul().local_set(ctx.tf);
    emit_trig_poly(s, ctx, ctx.tf, true);
    s.f64_mul()
        .f64_const(f64c(sigma))
        .f64_mul()
        .f64_const(f64c(mu))
        .f64_add();
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `Rng::fill_exp`.
fn emit_exp(s: &mut InstructionSink, ctx: &Ctx, j: usize, rate: f64) {
    s.f64_const(f64c(1.0));
    emit_next_f64(s, j);
    s.f64_sub();
    emit_ln(s, ctx);
    s.f64_neg().f64_const(f64c(rate)).f64_div();
}

/// `Geometric(p)` via `floor(ln(1 - u) / ln(1 - p))` — mirrors `Rng::fill_geometric`. `ln(1 - p)` is
/// a compile-time constant; `p == 1` makes it `-inf`, so every draw floors to `0` (the point mass).
fn emit_geometric(s: &mut InstructionSink, ctx: &Ctx, j: usize, p: f64) {
    let denom = (1.0 - p).ln();
    s.f64_const(f64c(1.0));
    emit_next_f64(s, j);
    s.f64_sub();
    emit_ln(s, ctx);
    s.f64_const(f64c(denom)).f64_div().f64_floor();
}

/// `base ^ k` for a small non-negative integer `k`, as repeated multiply (`k == 0` → `1.0`).
fn emit_pow(s: &mut InstructionSink, base: u32, k: u32) {
    if k == 0 {
        s.f64_const(f64c(1.0));
        return;
    }
    s.local_get(base);
    for _ in 1..k {
        s.local_get(base).f64_mul();
    }
}

fn emit_unary(s: &mut InstructionSink, ctx: &Ctx, op: UnOp, a: u32) {
    match op {
        UnOp::Neg => {
            s.local_get(a).f64_neg();
        }
        UnOp::Not => {
            // logical not of a 0/1 value: (a == 0) ? 1 : 0
            s.local_get(a)
                .f64_const(f64c(0.0))
                .f64_eq()
                .f64_convert_i32_u();
        }
        UnOp::Sin => {
            s.local_get(a);
            emit_trig(s, ctx, false);
        }
        UnOp::Cos => {
            s.local_get(a);
            emit_trig(s, ctx, true);
        }
        UnOp::Atan => {
            s.local_get(a).call(ATAN);
        }
        UnOp::Round => {
            s.local_get(a).call(ROUND);
        }
        UnOp::Floor => {
            s.local_get(a).f64_floor();
        }
        UnOp::Ceil => {
            s.local_get(a).f64_ceil();
        }
        UnOp::Ln => {
            use crate::approx::{LN_SUBNORMAL_CORR, LN_SUBNORMAL_SCALE};
            // Full-domain ln: the inlined poly (positive finite inputs only) behind guards that
            // match `f64::ln` — x > 0 → poly, 0 → -inf, negative/NaN → NaN, +inf → +inf (the
            // poly's exponent bit-surgery would misread ±inf/NaN/negatives). Mirrors
            // `jit::emit_ln_guarded`.
            //
            // Subnormal positive inputs are first scaled into the normal range (their zero exponent
            // field would corrupt the mantissa bit-surgery) and corrected by `54·ln2` (finding C9):
            //   a_in = (a < MIN_POSITIVE) ? a * SCALE : a
            s.local_get(a).f64_const(f64c(LN_SUBNORMAL_SCALE)).f64_mul();
            s.local_get(a);
            s.local_get(a).f64_const(f64c(f64::MIN_POSITIVE)).f64_lt();
            s.select(); // a_in
            emit_ln(s, ctx); // poly_raw (consumes a_in)
            s.local_set(ctx.tf); // stash poly_raw (transcendental scratch is dead after emit_ln)
                                 //   poly = (a < MIN_POSITIVE) ? poly_raw - 54·ln2 : poly_raw
            s.local_get(ctx.tf)
                .f64_const(f64c(LN_SUBNORMAL_CORR))
                .f64_sub();
            s.local_get(ctx.tf);
            s.local_get(a).f64_const(f64c(f64::MIN_POSITIVE)).f64_lt();
            s.select(); // stack: poly
                        // non_pos = (a == 0) ? -inf : NaN
            s.f64_const(f64c(f64::NEG_INFINITY))
                .f64_const(f64c(f64::NAN));
            s.local_get(a).f64_const(f64c(0.0)).f64_eq();
            s.select(); // stack: poly, non_pos
                        // r = (a > 0) ? poly : non_pos
            s.local_get(a).f64_const(f64c(0.0)).f64_gt();
            s.select(); // stack: r
                        // (a != +inf) ? r : +inf — patch the poly's mangled +inf back.
            s.f64_const(f64c(f64::INFINITY));
            s.local_get(a).f64_const(f64c(f64::INFINITY)).f64_ne();
            s.select();
        }
        UnOp::Exp => {
            // e^x via the imported library `exp` — matches the interpreter's `exp` (finding C9; the
            // old `pow(e, x)` could differ in the last bit).
            s.local_get(a).call(EXP);
        }
        UnOp::Sign => {
            // (a > 0) - (a < 0): -1 / 0 / +1, exactly 0 at 0 (matches `apply_un`, unlike signum).
            s.local_get(a)
                .f64_const(f64c(0.0))
                .f64_gt()
                .f64_convert_i32_u();
            s.local_get(a)
                .f64_const(f64c(0.0))
                .f64_lt()
                .f64_convert_i32_u();
            s.f64_sub();
        }
    }
}

fn emit_binary(s: &mut InstructionSink, op: BinOp, a: u32, b: u32) {
    // `&&`/`||` use the interpreter/JIT's `(a != 0) & (b != 0)` semantics — NOT `f64.min`/`max`,
    // which return NaN if either operand is NaN whereas `(NaN != 0)` is `true` (finding C8). Both
    // operands are 0/1 events and the result is 0/1, so this is exact and branch-free.
    if matches!(op, BinOp::And | BinOp::Or) {
        s.local_get(a).f64_const(f64c(0.0)).f64_ne(); // (a != 0) : i32
        s.local_get(b).f64_const(f64c(0.0)).f64_ne(); // (b != 0) : i32
        if matches!(op, BinOp::And) {
            s.i32_and();
        } else {
            s.i32_or();
        }
        s.f64_convert_i32_u();
        return;
    }
    // Floored modulo needs both operands twice (`a − b·floor(a/b)`), so it builds its own stack.
    if matches!(op, BinOp::Mod) {
        s.local_get(a);
        s.local_get(b);
        s.local_get(a).local_get(b).f64_div().f64_floor();
        s.f64_mul();
        s.f64_sub();
        return;
    }
    s.local_get(a).local_get(b);
    match op {
        BinOp::Add => {
            s.f64_add();
        }
        BinOp::Sub => {
            s.f64_sub();
        }
        BinOp::Mul => {
            s.f64_mul();
        }
        BinOp::Div => {
            s.f64_div();
        }
        BinOp::Lt => {
            s.f64_lt().f64_convert_i32_u();
        }
        BinOp::Gt => {
            s.f64_gt().f64_convert_i32_u();
        }
        BinOp::Le => {
            s.f64_le().f64_convert_i32_u();
        }
        BinOp::Ge => {
            s.f64_ge().f64_convert_i32_u();
        }
        BinOp::Eq => {
            s.f64_eq().f64_convert_i32_u();
        }
        BinOp::Ne => {
            s.f64_ne().f64_convert_i32_u();
        }
        BinOp::And | BinOp::Or => {
            unreachable!("And/Or are handled before the generic two-operand path")
        }
        BinOp::Mod => unreachable!("Mod is handled before the generic two-operand path"),
        BinOp::Pow => unreachable!("Pow is handled before emit_binary"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, InterpBackend};
    use crate::kernel::{supported, STREAMS};
    use crate::sampler::moments;
    use wasmi::{Engine, Linker, Module as WasmModule, Store};

    // The shared cross-backend conformance corpus (finding C2), also consumed by `jit`.
    use crate::conformance;

    /// Instantiate an emitted kernel in `wasmi`, seed it, run one batch, and return `out[0]`. For an
    /// RNG-free graph every lane is identical, so `[0]` fully characterizes the backend's output.
    fn first_emitted(bytes: &[u8], streams: usize, seed: u64) -> f64 {
        let engine = Engine::default();
        let module = WasmModule::new(&engine, bytes).expect("emitted module must validate");
        let mut store = Store::new(&engine, ());
        let mut linker = <Linker<()>>::new(&engine);
        linker.func_wrap("m", "atan", |x: f64| x.atan()).unwrap();
        linker.func_wrap("m", "round", |x: f64| x.round()).unwrap();
        linker
            .func_wrap("m", "pow", |a: f64, b: f64| a.powf(b))
            .unwrap();
        linker.func_wrap("m", "sin", |x: f64| x.sin()).unwrap();
        linker.func_wrap("m", "cos", |x: f64| x.cos()).unwrap();
        linker.func_wrap("m", "exp", |x: f64| x.exp()).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let memory = instance.get_memory(&store, "memory").unwrap();
        let kernel = instance
            .get_typed_func::<(i32, i32, i32), ()>(&store, "kernel")
            .unwrap();
        let (state_ptr, out_ptr) = (0i32, 4096i32);
        let state = crate::kernel::seed_state(seed, streams);
        let mut state_bytes = Vec::with_capacity(state.len() * 8);
        for w in &state {
            state_bytes.extend_from_slice(&w.to_le_bytes());
        }
        memory
            .write(&mut store, state_ptr as usize, &state_bytes)
            .unwrap();
        kernel
            .call(&mut store, (out_ptr, BATCH as i32, state_ptr))
            .unwrap();
        let mut b = [0u8; 8];
        memory.read(&store, out_ptr as usize, &mut b).unwrap();
        f64::from_le_bytes(b)
    }

    /// **Const-graph exact-equality suite (finding C2).** For every RNG-free program the emitted WASM
    /// kernel (run through `wasmi`) must be **bit-identical** to the interpreter oracle — no
    /// Monte-Carlo noise to hide a divergence. Pins the C3 (large-arg trig), C8 (`&&`/`||` relowering),
    /// and C9 (`exp`) fixes at the bit level; since the JIT suite checks interp↔JIT identically, all
    /// three backends agree.
    #[test]
    fn conformance_const_graphs_bit_identical_interp_vs_wasm() {
        for (label, src) in conformance::CONST_CASES {
            let (eng, id) = graph_of(src);
            let g = eng.graph();
            let mut ir = InterpBackend.compile(g, id).runner();
            ir.reseed(0);
            let cap = ir.batch_cap();
            let interp = ir.next_batch(cap)[0];
            let wasm = first_emitted(&emit(g, id, 1), 1, 0);
            assert_eq!(
                interp.to_bits(),
                wasm.to_bits(),
                "{label}: interp {interp} ({:#018x}) != wasm {wasm} ({:#018x})",
                interp.to_bits(),
                wasm.to_bits()
            );
        }
    }

    /// The RNG half of the shared corpus: the emitted WASM kernel must agree with the interpreter in
    /// distribution on every case (mean within tolerance) — the same superset the JIT runs (C2).
    #[test]
    fn conformance_rng_cases_match_interp() {
        for (_label, src, seed) in conformance::RNG_CASES {
            assert_wasm_matches_interp(src, *seed);
        }
    }

    /// Run an emitted kernel through the `wasmi` interpreter for `batches` batches, returning the
    /// mean of every sample produced. The host supplies the six transcendental imports (Rust `f64`
    /// methods) and a linear memory; state persists in memory across calls (the kernel reads it at
    /// entry and writes it back at exit), exactly as a browser host would drive it.
    fn run_emitted(bytes: &[u8], streams: usize, seed: u64, batches: usize) -> (f64, u64) {
        let engine = Engine::default();
        let module = WasmModule::new(&engine, bytes).expect("emitted module must validate");
        let mut store = Store::new(&engine, ());
        let mut linker = <Linker<()>>::new(&engine);
        // `ln` is inlined; `sin`/`cos` are inlined for `|x| < TRIG_MAX` but imported for the
        // large-argument fallback (finding C3); `exp` is imported to match the interpreter (C9).
        // These are Rust `f64` methods so the in-test kernel is bit-identical to the interpreter
        // oracle; the browser supplies the same names via `Math.*`.
        linker.func_wrap("m", "atan", |x: f64| x.atan()).unwrap();
        linker.func_wrap("m", "round", |x: f64| x.round()).unwrap();
        linker
            .func_wrap("m", "pow", |a: f64, b: f64| a.powf(b))
            .unwrap();
        linker.func_wrap("m", "sin", |x: f64| x.sin()).unwrap();
        linker.func_wrap("m", "cos", |x: f64| x.cos()).unwrap();
        linker.func_wrap("m", "exp", |x: f64| x.exp()).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let memory = instance.get_memory(&store, "memory").unwrap();
        let kernel = instance
            .get_typed_func::<(i32, i32, i32), ()>(&store, "kernel")
            .unwrap();

        // Host memory convention: state at offset 0, output column at 4096.
        let state_ptr: i32 = 0;
        let out_ptr: i32 = 4096;
        let cap = BATCH;

        // Seed the state (strided layout) into memory as little-endian u64s.
        let state = crate::kernel::seed_state(seed, streams);
        let mut state_bytes = Vec::with_capacity(state.len() * 8);
        for w in &state {
            state_bytes.extend_from_slice(&w.to_le_bytes());
        }
        memory
            .write(&mut store, state_ptr as usize, &state_bytes)
            .unwrap();

        let mut sum = 0.0f64;
        let mut count = 0u64;
        let mut out_bytes = vec![0u8; cap * 8];
        for _ in 0..batches {
            kernel
                .call(&mut store, (out_ptr, cap as i32, state_ptr))
                .unwrap();
            memory
                .read(&store, out_ptr as usize, &mut out_bytes)
                .unwrap();
            for chunk in out_bytes.chunks_exact(8) {
                sum += f64::from_le_bytes(chunk.try_into().unwrap());
                count += 1;
            }
        }
        (sum / count as f64, count)
    }

    fn graph_of(src: &str) -> (crate::Engine, RvId) {
        let mut eng = crate::Engine::new();
        let id = match eng.run_rv(src).unwrap() {
            crate::Value::Dist(id) => id,
            other => panic!("expected a dist, got {other:?}"),
        };
        (eng, id)
    }

    /// The emitted WASM kernel and the interpreter must agree *in distribution* (compared via the
    /// mean, like `jit`'s parity harness — RNG consumption order differs by design).
    fn assert_wasm_matches_interp(src: &str, seed: u64) {
        let (eng, id) = graph_of(src);
        let graph = eng.graph();
        assert!(
            supported(graph, id),
            "case must be codegen-supported: {src}"
        );
        let (bytes, streams) = (emit(graph, id, STREAMS), STREAMS);
        let (wasm_mean, count) = run_emitted(&bytes, streams, seed, 16);
        let interp_mean = moments(graph, id, count as usize, seed).mean;
        assert!(
            (wasm_mean - interp_mean).abs() < 0.05 + 0.05 * interp_mean.abs(),
            "{src}: wasm_mean={wasm_mean} interp_mean={interp_mean}"
        );
    }

    #[test]
    fn wasm_uniform_arithmetic_matches_interp() {
        assert_wasm_matches_interp("use rand; X ~ unif(0,1); 2*X + 3", 1);
        assert_wasm_matches_interp("use rand; X ~ unif(-1,1); X*X*X + X", 2);
    }

    #[test]
    fn wasm_dice_and_indicator_match_interp() {
        assert_wasm_matches_interp("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B", 3);
        assert_wasm_matches_interp("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X^2 + Y^2 < 1", 4);
    }

    #[test]
    fn wasm_continuous_sources_match_interp() {
        assert_wasm_matches_interp("use rand; Z ~ normal(2,3); Z", 5);
        assert_wasm_matches_interp("use rand; X ~ exponential(2); X", 6);
        assert_wasm_matches_interp("use rand; G ~ geometric(0.25); G", 7);
        assert_wasm_matches_interp("use rand; Z ~ normal_int(10,3); Z", 8);
        // Second moment of N(0,1) must be ≈1 — a far tighter probe of the inlined `ln`/`cos` than the
        // mean: it pins the Box–Muller *spread* (radius from `ln`, angle from `cos`), so a biased
        // approximation shows up here even though E[Z]=0 hides it.
        assert_wasm_matches_interp("use rand; Z ~ normal(0,1); Z*Z", 15);
        assert_wasm_matches_interp("use rand; Z ~ normal(0,1); Z*Z*Z*Z", 16); // 4th moment ≈ 3
    }

    #[test]
    fn wasm_ufuncs_and_nonconst_pow_match_interp() {
        assert_wasm_matches_interp("use rand; use math; X ~ unif(0,1); sin(X) + cos(X)", 9);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-1,1); atan(X)", 10);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-1,1); sign(X)", 11);
        assert_wasm_matches_interp("use rand; A ~ unif(1,2); B ~ unif(1,2); A ^ B", 12);
        // Trig over a wide argument range (multiple periods) — exercises the Cody–Waite range
        // reduction and quadrant selection, not just the small-angle kernel. E[sin²]=0.5.
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-8,8); sin(X)*sin(X)", 17);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-8,8); cos(X)*cos(X)", 18);
    }

    #[test]
    fn wasm_mod_floor_ceil_match_interp() {
        // BinOp::Mod (a − b·floor(a/b)) and UnOp::Floor/Ceil (native f64.floor/f64.ceil) must match
        // the interpreter — the wasm `Mod` builds its own stack (both operands twice).
        assert_wasm_matches_interp("use rand; X ~ unif(0,10); X % 3", 19);
        assert_wasm_matches_interp("use rand; X ~ unif(-5,5); X % 4", 20);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-3,3); math::floor(X)", 21);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-3,3); math::ceil(X)", 22);
    }

    #[test]
    fn wasm_exp_ln_match_interp() {
        // exp lowers to the imported pow(e, x): E[e^X] over N(0,1) = e^0.5 (the lognormal mean).
        assert_wasm_matches_interp("use rand; use math; X ~ normal(0,1); exp(X)", 23);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-1,1); exp(X)", 24);
        // ln of a strictly positive draw hits the inlined poly (guard's x > 0 arm only).
        assert_wasm_matches_interp("use rand; use math; X ~ unif(0.5, 3); log(X)", 25);
        // Domain guard, negative lanes: log(x<0) is NaN and NaN == NaN is false, so the indicator
        // mean is P(X > 0) = 0.6 — matching the interpreter's f64::ln lane-for-lane.
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-2, 3); log(X) == log(X)", 26);
        // Domain guard, zero lanes: log(0) = -inf < -100, P = 1/5 over unif_int(0,4).
        assert_wasm_matches_interp(
            "use rand; use math; X ~ unif_int(0, 4); log(X) < 0 - 100",
            27,
        );
        // Domain guard, +inf: X/0 = +inf per lane; log(+inf) = +inf > 100 surely.
        assert_wasm_matches_interp("use rand; use math; X ~ unif(1, 2); log(X / 0) > 100", 28);
    }

    #[test]
    fn wasm_shared_draw_is_cse_not_independent() {
        // X + X reuses ONE draw per lane (mean 2*E[X]=1), proving the emitter memoizes shared nodes
        // rather than drawing twice. (Two draws would still give mean 1 but the structure differs;
        // the X - X == 0 check below is the decisive one.)
        assert_wasm_matches_interp("use rand; X ~ unif(0,1); X + X", 13);
        // X - X must be exactly 0 everywhere — only true if both references are the same draw.
        let (eng, id) = graph_of("use rand; X ~ unif(0,1); X - X");
        let (mean, _) = run_emitted(&emit(eng.graph(), id, STREAMS), STREAMS, 14, 8);
        assert_eq!(mean, 0.0, "X - X must be identically zero (shared draw)");
    }

    /// The default policy ([`emit_for`]) picks the multi-stream kernel for a purely-inline RNG graph
    /// and a 1-stream kernel for any graph carrying a transcendental (now inlined, but
    /// throughput-bound) or a call — and all must match the interpreter's mean. With `ln`/`cos`
    /// inlined, `normal` is now *emitted* (single-stream), not gated out as it was when it meant a
    /// per-draw host call.
    #[test]
    fn stream_choice_and_distribution() {
        // RNG-bound, all inline → multi-stream.
        let (eng, id) = graph_of("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B");
        let (bytes, streams) = emit_for(eng.graph(), id).expect("inline graph should emit");
        assert_eq!(streams, STREAMS, "inline graph should be multi-stream");
        let (wasm_mean, count) = run_emitted(&bytes, streams, 0xABCDEF, 64);
        let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).mean;
        assert!(
            (wasm_mean - interp).abs() < 0.05,
            "dice-sum: wasm={wasm_mean} interp={interp}"
        );

        // `normal` now emits (inlined `ln`/`cos`), but is throughput-bound → single-stream.
        let (eng, id) = graph_of("use rand; X ~ normal(0,1); Y ~ normal(0,1); X + Y");
        let (bytes, streams) = emit_for(eng.graph(), id).expect("normal graph should now emit");
        assert_eq!(streams, 1, "transcendental graph should be single-stream");
        let (wasm_mean, count) = run_emitted(&bytes, streams, 0xABCDEF, 64);
        let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).mean;
        assert!(
            (wasm_mean - interp).abs() < 0.05,
            "normal-sum: wasm={wasm_mean} interp={interp}"
        );

        // Arithmetic-dominated but carries a `pow` call (non-const exponent) → still profitable
        // (fusible > libcalls), but choose_streams keeps it single-stream (the call won't overlap).
        let (eng, id) = graph_of("use rand; A ~ unif(1,2); B ~ unif(1,2); A ^ B");
        let (bytes, streams) = emit_for(eng.graph(), id).expect("pow graph should emit");
        assert_eq!(streams, 1, "call-bearing graph should be single-stream");
        let (wasm_mean, count) = run_emitted(&bytes, streams, 0xABCDEF, 64);
        let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).mean;
        assert!(
            (wasm_mean - interp).abs() < 0.05,
            "pow: wasm={wasm_mean} interp={interp}"
        );
    }

    #[test]
    fn poisson_is_not_emitted() {
        // Poisson stays interpreter-only: the gate returns None (the browser would keep the interp).
        let (eng, id) = graph_of("use rand; K ~ poisson(3); K");
        assert!(!supported(eng.graph(), id));
        assert!(
            emit_for(eng.graph(), id).is_none(),
            "Poisson must not be emitted"
        );
    }

    /// Dump an emitted kernel to disk so the JS host protocol can be validated against real bytes in
    /// a JS engine (see `packages/www`/the Node check). Ignored — run with an explicit out path:
    /// `NOISE_KERNEL_OUT=/tmp/k.wasm cargo test -p noise-core --release -- --ignored dump_kernel`
    #[test]
    #[ignore]
    fn dump_kernel() {
        let path = std::env::var("NOISE_KERNEL_OUT").unwrap_or_else(|_| "/tmp/kernel.wasm".into());
        let src = std::env::var("NOISE_KERNEL_SRC")
            .unwrap_or_else(|_| "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B".into());
        let (eng, id) = graph_of(&src);
        let (bytes, streams) = emit_for(eng.graph(), id).expect("should emit");
        std::fs::write(&path, &bytes).unwrap();
        // streams alongside, so the JS side can seed the right state width.
        std::fs::write(format!("{path}.streams"), streams.to_string()).unwrap();
        eprintln!("wrote {} bytes ({streams} streams) to {path}", bytes.len());
    }

    /// Stream count must not change the distribution: 1-stream and STREAMS-stream kernels estimate
    /// the same mean (each stream is an i.i.d. substream), mirroring `jit`'s invariant.
    #[test]
    fn stream_count_preserves_distribution() {
        let (eng, id) = graph_of("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B");
        for streams in [1usize, STREAMS] {
            let (mean, _) = run_emitted(&emit(eng.graph(), id, streams), streams, 99, 64);
            assert!((mean - 7.0).abs() < 0.05, "@{streams} streams: mean={mean}");
        }
    }
}
