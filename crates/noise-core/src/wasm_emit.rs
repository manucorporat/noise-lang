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
//! The module imports its transcendentals (`ln`/`sin`/`cos`/`atan`/`round`/`pow`) from module `"m"`,
//! exactly as the native kernel calls the `nz_*` shims — wasm has no native `ln`/`cos`. The host
//! supplies them (the browser via `Math.*`; the test harness via Rust `f64` methods). The "B2"
//! finding recurs here — those imports are non-inlined calls — so [`kernel::profitable`] and
//! [`kernel::choose_streams`] gate this backend too (see [`emit_for`]).
//!
//! **Why no host execution yet.** Running an emitted module means a JS host `instantiate`s it and
//! drives it (the `Backend`/`Program`/`Runner` seam, browser-side). That wiring is a follow-up; this
//! cut is the emitter plus a native parity harness (tests run emitted modules through an embedded
//! wasm interpreter and check distribution parity with the interpreter, mirroring `jit`'s tests).
//!
//! Scope mirrors the JIT: `unif`/`unif_int`/`normal`/`exp`/`geometric` sources, `+ - * /`,
//! integer-constant `**`, comparisons, `&& ||`, unary `- !` and the math ufuncs, and lifted `if`.
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
const LN: u32 = 0;
const SIN: u32 = 1;
const COS: u32 = 2;
const ATAN: u32 = 3;
const ROUND: u32 = 4;
const POW: u32 = 5;
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
    MemArg { offset, align: 3, memory_index: 0 }
}

/// f64 literal as the `Ieee64` the encoder wants.
fn f64c(x: f64) -> Ieee64 {
    Ieee64::from(x)
}

/// Local holding stream `j`'s xoshiro word `k`.
fn sl(j: usize, k: usize) -> u32 {
    STATE_BASE + (j * 4 + k) as u32
}

/// Per-emission constants shared across the whole kernel body.
struct Ctx<'g> {
    graph: &'g RvGraph,
    /// Base index of the f64 value-slot block; node memo locals live above this.
    fbase: u32,
    /// Distinct nodes per stream (each stream gets its own contiguous block of f64 slots).
    cone: u32,
}

/// Emit a complete WASM module computing `root` with the given RNG stream count. `streams` must
/// divide [`BATCH`] (so a batch is a whole number of loop iterations) and be ≥ 1. The graph must be
/// codegen-supported (no `Poisson`); callers use [`emit_for`] to honor the profitability gate.
pub fn emit(graph: &RvGraph, root: RvId, streams: usize) -> Vec<u8> {
    assert!(streams >= 1 && BATCH.is_multiple_of(streams), "streams must divide BATCH");
    let cone = cone_size(graph, root) as u32;

    // --- types: (f64)->f64 for the unary shims, (f64,f64)->f64 for pow, kernel sig ---
    let mut types = TypeSection::new();
    types.ty().function([ValType::F64], [ValType::F64]); // 0: unary math
    types.ty().function([ValType::F64, ValType::F64], [ValType::F64]); // 1: pow
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], []); // 2: kernel

    // --- imports: the six transcendentals from module "m" ---
    let mut imports = ImportSection::new();
    for name in ["ln", "sin", "cos", "atan", "round"] {
        imports.import("m", name, EntityType::Function(0));
    }
    imports.import("m", "pow", EntityType::Function(1));

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
        (1, ValType::I32),          // I (counter)
        (i64_count, ValType::I64),  // scratch + state words
        (f64_count, ValType::F64),  // node value slots
    ]);
    // The f64 value-slot block begins right after the i64 block, which starts at `RES` (index 4).
    let ctx = Ctx { graph, fbase: RES + i64_count, cone };
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
/// (e.g. `Poisson`); the browser would keep the interpreter for those, exactly like the native gate.
pub fn emit_for(graph: &RvGraph, root: RvId) -> Option<(Vec<u8>, usize)> {
    if !crate::kernel::profitable(graph, root) {
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

    s.local_get(I).i32_const(streams as i32).i32_add().local_set(I);
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
        RvNode::Src(Source::Normal { mu, sigma }) => emit_normal(s, j, *mu, *sigma),
        RvNode::Src(Source::Exp { rate }) => emit_exp(s, j, *rate),
        RvNode::Src(Source::Geometric { p }) => emit_geometric(s, j, *p),
        RvNode::Src(Source::Poisson { .. }) => unreachable!("profitable() excludes Poisson"),
        RvNode::ConstNum(x) => {
            s.f64_const(f64c(*x));
        }
        RvNode::ConstBool(b) => {
            s.f64_const(f64c(if *b { 1.0 } else { 0.0 }));
        }
        RvNode::Unary(op, a) => {
            let la = emit_node(s, ctx, j, *a, memo, slot);
            emit_unary(s, *op, la);
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
            s.local_get(la).local_get(lb).local_get(lc).f64_const(f64c(0.0)).f64_ne().select();
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
    s.local_get(sl(j, 2)).local_get(sl(j, 0)).i64_xor().local_set(S2A);
    s.local_get(sl(j, 3)).local_get(sl(j, 1)).i64_xor().local_set(S3A);
    // s1 ^= s2a ; s0 ^= s3a  (s1 set first; original s0 still intact for s0's update)
    s.local_get(sl(j, 1)).local_get(S2A).i64_xor().local_set(sl(j, 1));
    s.local_get(sl(j, 0)).local_get(S3A).i64_xor().local_set(sl(j, 0));
    // s2 ^= t ; s3 = rotl(s3a, 45)
    s.local_get(S2A).local_get(T).i64_xor().local_set(sl(j, 2));
    s.local_get(S3A).i64_const(45).i64_rotl().local_set(sl(j, 3));
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
    s.f64_const(f64c(hi - lo)).f64_mul().f64_const(f64c(lo)).f64_add();
}

/// `unif_int(lo, hi)` as `lo + floor(u * count)` — uniform over `lo..=hi`. (The native kernel uses
/// Lemire multiply-high, but wasm lacks a 64×64→128 high-multiply; this is identical in distribution.)
fn emit_uniform_int(s: &mut InstructionSink, j: usize, lo: f64, hi: f64) {
    let count = (hi - lo + 1.0).max(1.0);
    emit_next_f64(s, j);
    s.f64_const(f64c(count)).f64_mul().f64_floor().f64_const(f64c(lo)).f64_add();
}

/// `N(mu, sigma^2)` via Box–Muller, one normal per draw (cosine arm) — mirrors `jit::emit_normal`.
fn emit_normal(s: &mut InstructionSink, j: usize, mu: f64, sigma: f64) {
    use std::f64::consts::TAU;
    // r = sqrt(-2 * ln(1 - u1))
    s.f64_const(f64c(1.0));
    emit_next_f64(s, j);
    s.f64_sub().call(LN).f64_const(f64c(-2.0)).f64_mul().f64_sqrt();
    // result = mu + sigma * r * cos(TAU * u2)
    emit_next_f64(s, j);
    s.f64_const(f64c(TAU))
        .f64_mul()
        .call(COS)
        .f64_mul()
        .f64_const(f64c(sigma))
        .f64_mul()
        .f64_const(f64c(mu))
        .f64_add();
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `Rng::fill_exp`.
fn emit_exp(s: &mut InstructionSink, j: usize, rate: f64) {
    s.f64_const(f64c(1.0));
    emit_next_f64(s, j);
    s.f64_sub().call(LN).f64_neg().f64_const(f64c(rate)).f64_div();
}

/// `Geometric(p)` via `floor(ln(1 - u) / ln(1 - p))` — mirrors `Rng::fill_geometric`. `ln(1 - p)` is
/// a compile-time constant; `p == 1` makes it `-inf`, so every draw floors to `0` (the point mass).
fn emit_geometric(s: &mut InstructionSink, j: usize, p: f64) {
    let denom = (1.0 - p).ln();
    s.f64_const(f64c(1.0));
    emit_next_f64(s, j);
    s.f64_sub().call(LN).f64_const(f64c(denom)).f64_div().f64_floor();
}

/// `base ** k` for a small non-negative integer `k`, as repeated multiply (`k == 0` → `1.0`).
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

fn emit_unary(s: &mut InstructionSink, op: UnOp, a: u32) {
    match op {
        UnOp::Neg => {
            s.local_get(a).f64_neg();
        }
        UnOp::Not => {
            // logical not of a 0/1 value: (a == 0) ? 1 : 0
            s.local_get(a).f64_const(f64c(0.0)).f64_eq().f64_convert_i32_u();
        }
        UnOp::Sin => {
            s.local_get(a).call(SIN);
        }
        UnOp::Cos => {
            s.local_get(a).call(COS);
        }
        UnOp::Atan => {
            s.local_get(a).call(ATAN);
        }
        UnOp::Round => {
            s.local_get(a).call(ROUND);
        }
        UnOp::Sign => {
            // (a > 0) - (a < 0): -1 / 0 / +1, exactly 0 at 0 (matches `apply_un`, unlike signum).
            s.local_get(a).f64_const(f64c(0.0)).f64_gt().f64_convert_i32_u();
            s.local_get(a).f64_const(f64c(0.0)).f64_lt().f64_convert_i32_u();
            s.f64_sub();
        }
    }
}

fn emit_binary(s: &mut InstructionSink, op: BinOp, a: u32, b: u32) {
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
        // `&&`/`||` operands are always 0/1 events, so min/max realize the logic with no branch.
        BinOp::And => {
            s.f64_min();
        }
        BinOp::Or => {
            s.f64_max();
        }
        BinOp::Pow => unreachable!("Pow is handled before emit_binary"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{supported, STREAMS};
    use crate::sampler::moments;
    use wasmi::{Engine, Linker, Module as WasmModule, Store};

    /// Run an emitted kernel through the `wasmi` interpreter for `batches` batches, returning the
    /// mean of every sample produced. The host supplies the six transcendental imports (Rust `f64`
    /// methods) and a linear memory; state persists in memory across calls (the kernel reads it at
    /// entry and writes it back at exit), exactly as a browser host would drive it.
    fn run_emitted(bytes: &[u8], streams: usize, seed: u64, batches: usize) -> (f64, u64) {
        let engine = Engine::default();
        let module = WasmModule::new(&engine, bytes).expect("emitted module must validate");
        let mut store = Store::new(&engine, ());
        let mut linker = <Linker<()>>::new(&engine);
        linker.func_wrap("m", "ln", |x: f64| x.ln()).unwrap();
        linker.func_wrap("m", "sin", |x: f64| x.sin()).unwrap();
        linker.func_wrap("m", "cos", |x: f64| x.cos()).unwrap();
        linker.func_wrap("m", "atan", |x: f64| x.atan()).unwrap();
        linker.func_wrap("m", "round", |x: f64| x.round()).unwrap();
        linker.func_wrap("m", "pow", |a: f64, b: f64| a.powf(b)).unwrap();
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
        memory.write(&mut store, state_ptr as usize, &state_bytes).unwrap();

        let mut sum = 0.0f64;
        let mut count = 0u64;
        let mut out_bytes = vec![0u8; cap * 8];
        for _ in 0..batches {
            kernel.call(&mut store, (out_ptr, cap as i32, state_ptr)).unwrap();
            memory.read(&store, out_ptr as usize, &mut out_bytes).unwrap();
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
        assert!(supported(graph, id), "case must be codegen-supported: {src}");
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
        assert_wasm_matches_interp("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X**2 + Y**2 < 1", 4);
    }

    #[test]
    fn wasm_continuous_sources_match_interp() {
        assert_wasm_matches_interp("use rand; Z ~ normal(2,3); Z", 5);
        assert_wasm_matches_interp("use rand; X ~ exp(2); X", 6);
        assert_wasm_matches_interp("use rand; G ~ geometric(0.25); G", 7);
        assert_wasm_matches_interp("use rand; Z ~ normal_int(10,3); Z", 8);
    }

    #[test]
    fn wasm_ufuncs_and_nonconst_pow_match_interp() {
        assert_wasm_matches_interp("use rand; use math; X ~ unif(0,1); sin(X) + cos(X)", 9);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-1,1); atan(X)", 10);
        assert_wasm_matches_interp("use rand; use math; X ~ unif(-1,1); sign(X)", 11);
        assert_wasm_matches_interp("use rand; A ~ unif(1,2); B ~ unif(1,2); A ** B", 12);
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

    /// The default policy ([`emit_for`]) picks the multi-stream kernel for a purely-inline graph and
    /// a 1-stream kernel for a profitable graph that still carries a call — and both must match the
    /// interpreter's mean. (A transcendental-*bound* graph like `normal(2,3)` is gated out entirely;
    /// that's `poisson_is_not_emitted`'s sibling case, covered by the `< libcalls` branch.)
    #[test]
    fn stream_choice_and_distribution() {
        // RNG-bound, all inline → multi-stream.
        let (eng, id) = graph_of("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B");
        let (bytes, streams) = emit_for(eng.graph(), id).expect("inline graph should emit");
        assert_eq!(streams, STREAMS, "inline graph should be multi-stream");
        let (wasm_mean, count) = run_emitted(&bytes, streams, 0xABCDEF, 64);
        let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).mean;
        assert!((wasm_mean - interp).abs() < 0.05, "dice-sum: wasm={wasm_mean} interp={interp}");

        // Arithmetic-dominated but carries a `pow` call (non-const exponent) → still profitable
        // (fusible > libcalls), but choose_streams keeps it single-stream (the call won't overlap).
        let (eng, id) = graph_of("use rand; A ~ unif(1,2); B ~ unif(1,2); A ** B");
        let (bytes, streams) = emit_for(eng.graph(), id).expect("pow graph should emit");
        assert_eq!(streams, 1, "call-bearing graph should be single-stream");
        let (wasm_mean, count) = run_emitted(&bytes, streams, 0xABCDEF, 64);
        let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).mean;
        assert!((wasm_mean - interp).abs() < 0.05, "pow: wasm={wasm_mean} interp={interp}");
    }

    #[test]
    fn poisson_is_not_emitted() {
        // Poisson stays interpreter-only: the gate returns None (the browser would keep the interp).
        let (eng, id) = graph_of("use rand; K ~ poisson(3); K");
        assert!(!supported(eng.graph(), id));
        assert!(emit_for(eng.graph(), id).is_none(), "Poisson must not be emitted");
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
