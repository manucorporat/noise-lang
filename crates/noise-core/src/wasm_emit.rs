//! WASM-emitter backend (PLAN.md Phase 4 "Browser note") — the browser's twin of [`crate::jit`].
//!
//! A WASM sandbox can't emit or run native code, so the Cranelift JIT (B1/B2) is native-only. What
//! *is* portable is the [`RvGraph`] IR. This module is the second lowering of that IR: it walks the
//! identical graph the identical post-order way and emits the identical fused counter-keyed kernel
//! (PLAN-PREGPU Track C) — only the per-node encoding differs (`f64.mul` instead of `mulsd`, the
//! same inlined pcg4d-3r hash in either). The shared "what the graph means" lives in
//! [`crate::kernel`] (the cost/profitability gate); this file is just "how to spell it in wasm
//! bytes".
//!
//! **Output.** [`emit`] produces a complete WebAssembly module (`Vec<u8>`) exporting:
//!   * `memory` — one linear memory the host reads/writes.
//!   * `kernel(out: i32, n: i32, key_lo: i32, key_hi: i32, lane0: i32)` — fills `out[0..n]` with
//!     the root draws for global lanes `lane0 .. lane0 + n`. Stateless — same arguments, same
//!     column, bit-identical to the interpreter and the native JIT under the same seed.
//!
//! `ln`/`sin`/`cos` are **inlined as polynomials** ([`emit_ln`]/[`emit_trig`], the same
//! [`crate::approx`] reference the JIT uses) — wasm has no native ones, and a host call would cross
//! the JS boundary per draw. The module imports `atan`/`round`/`pow` from module `"m"`, plus
//! `sin`/`cos` (the large-argument fallback past `approx::TRIG_MAX` — finding C3), `exp` (matched
//! to the interpreter — finding C9) and `sqrt` (V8/arm64 regresses on inline `f64.sqrt` in large
//! kernel bodies — see `UnOp::Sqrt` in [`emit_unop`]); the browser supplies them via `Math.*` and
//! the test harness via Rust `f64` methods. The inline `sin`/`cos` polynomial handles every ordinary argument, so those
//! imports fire only on the rare huge-argument path. Because the transcendentals are inlined,
//! [`emit_for`] passes `inline_trans = true` to
//! [`kernel::profitable`] — the gate decision matches the native JIT, and the 2× win reaches the
//! browser.
//!
//! **Why no host execution yet.** Running an emitted module means a JS host `instantiate`s it and
//! drives it (the `Backend`/`Program`/`Runner` seam, browser-side). That wiring is a follow-up; this
//! cut is the emitter plus a native parity harness (tests run emitted modules through an embedded
//! wasm interpreter and check distribution parity with the interpreter, mirroring `jit`'s tests).
//!
//! Scope mirrors the JIT: `unif`/`unif_int`/`normal`/`exp`/`geometric` sources, `+ - * /`,
//! integer-constant `^`, comparisons, `&& ||`, unary `- !` and the math ufuncs, lifted `if`, and
//! `Gather` — a const table becomes an active **data segment** after the output columns (indexed
//! load; the `empirical`/bootstrap shape), a small non-const one a compare/select chain
//! ([`crate::kernel::gather_class`]). `Poisson` and large non-const gathers stay interpreter-only
//! (rejected by the gate). `unif_int` computes the same 48-bit Lemire multiply-high as the
//! interpreter/JIT via a split multiply (wasm has no `mulhi`) — exact, hence bit-identical, for
//! `count < 2^39` (every post-Track-B count; the 2^24 cap is far below), with the old
//! float method (`lo + floor(u·count)`, identical in distribution) as the huge-count fallback.

// The emitter and its host-import indices are exercised on the wasm32 target and by the wasmi
// parity tests; on a native non-test build they read as dead. This module was previously `pub`,
// which masked the same warnings.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, Ieee64, ImportSection, InstructionSink, MemArg, MemorySection,
    MemoryType, Module, TypeSection, ValType,
};

use crate::ast::{BinOp, UnOp};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId, RvNode, Source};
use crate::kernel::{cone_size_roots, const_int_exponent, gather_class, GatherClass};

// --- imported-function indices (declared in import order, before the local `kernel`) ---
// `ln` is inlined as a polynomial (`emit_ln`, mirroring `jit`). `sin`/`cos` are inlined too, but
// the module re-imports them for the **large-argument fallback**: past `approx::TRIG_MAX` the 2-term
// reduction degrades, so `emit_trig` defers to the host's accurate `sin`/`cos` there (finding C3).
// `exp` is imported (rather than lowered as `pow(e, x)`) so it matches the interpreter's `exp`
// (finding C9). `sqrt` is imported for `UnOp::Sqrt` — NOT the inline `f64.sqrt` instruction:
// V8/arm64 regresses ~30% on large single-block kernel bodies when the call sites become inline
// sqrt (measured 2026-07-14, `am_vs_fm` +21% run-level; the import calls act as live-range split
// points for V8's regalloc). JSC prefers inline; revisit if V8's regalloc improves. `Math.sqrt`
// is IEEE-exact, so semantics are unchanged. The host (`wasm_host` / the test linker) supplies
// all of these via `Math.*`.
const ATAN: u32 = 0;
const ROUND: u32 = 1;
const POW: u32 = 2;
const SIN: u32 = 3;
const COS: u32 = 4;
const EXP: u32 = 5;
const SQRT: u32 = 6;
const N_IMPORTS: u32 = 7;

// --- fixed local indices (params occupy 0..4) ---
const OUT: u32 = 0; // param: output base pointer
const N: u32 = 1; // param: sample count
const K0: u32 = 2; // param: draw-key word 0
const K1: u32 = 3; // param: draw-key word 1
const LANE0: u32 = 4; // param: first global lane index
const I: u32 = 5; // loop counter (sample index)
const LANE: u32 = 6; // current global lane
// The four hash words of the cell being computed (i32) — scratch for `emit_cell`, shared by every
// source (a cell is computed atomically: set from the key/lane/source, mixed in place, consumed).
const V0: u32 = 7;
const V1: u32 = 8;
const V2: u32 = 9;
const V3: u32 = 10;
const N_I32_LOCALS: u32 = 6; // I, LANE, V0..V3

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
    /// Absolute byte address of each const gather table ([`GatherClass::ConstTable`]) in linear
    /// memory — the data-segment region after the output columns (see [`collect_gather_tables`]).
    gather_tables: HashMap<RvId, u64>,
}

/// Collect the const gather tables in the union cone of `roots`: absolute byte address per gather
/// node (starting at `tables_base`, the first 8-aligned byte after the output columns) plus the
/// concatenated little-endian f64 bytes for ONE active data segment at `tables_base`. Deduped by
/// `RvId`, matching the emitters' per-node handling (a gather shared across streams/roots reads one
/// table). Traversal skips const-table elems exactly like [`cone_size_roots`] — they are leaves.
fn collect_gather_tables(
    graph: &RvGraph,
    roots: &[RvId],
    tables_base: u64,
) -> (HashMap<RvId, u64>, Vec<u8>) {
    let mut seen = HashSet::new();
    let mut stack: Vec<RvId> = roots.to_vec();
    let mut addrs = HashMap::new();
    let mut data = Vec::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match graph.node(id) {
            RvNode::Gather { elems, index } => {
                if gather_class(graph, elems) == Some(GatherClass::ConstTable) {
                    addrs.insert(id, tables_base + data.len() as u64);
                    for &e in elems.iter() {
                        let RvNode::ConstNum(x) = graph.node(e) else {
                            unreachable!("ConstTable gather has only ConstNum elems");
                        };
                        data.extend_from_slice(&x.to_le_bytes());
                    }
                } else {
                    stack.extend(elems.iter().copied());
                }
                stack.push(*index);
            }
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
            // Interpreter-only (the gate rejects these cones before emission); keep the walk total.
            RvNode::Permutation { .. } | RvNode::Rotation { .. } => {}
            RvNode::ArrIndex { arr, index } => {
                stack.push(*arr);
                stack.push(*index);
            }
        }
    }
    (addrs, data)
}

/// Emit a complete WASM module computing `root`. The graph must be codegen-supported (no
/// `Poisson`); it **panics** (assert / `unreachable!`) on an ungated graph, so it is `pub(crate)`
/// — the public entry point is the gate-honoring [`emit_for`] (finding C10).
pub(crate) fn emit(graph: &RvGraph, root: RvId) -> Vec<u8> {
    emit_roots(graph, &[root])
}

/// Multi-root [`emit`]: ONE kernel computing every root per lane from a shared per-lane memo
/// (shared sources drawn once per lane — the roots stay *jointly* sampled), writing root `r`'s
/// draws into its own BATCH-strided output column at `out + r*BATCH*8`. A single root emits
/// exactly the module [`emit`] always emitted. Memory is sized for `k` columns after the first
/// page (host convention: columns at/after 4096; the counter kernel keeps no state region).
pub(crate) fn emit_roots(graph: &RvGraph, roots: &[RvId]) -> Vec<u8> {
    let cone = cone_size_roots(graph, roots) as u32;

    // --- types: (f64)->f64 for the unary shims, (f64,f64)->f64 for pow, kernel sig ---
    let mut types = TypeSection::new();
    types.ty().function([ValType::F64], [ValType::F64]); // 0: unary math
    types
        .ty()
        .function([ValType::F64, ValType::F64], [ValType::F64]); // 1: pow
    types.ty().function(
        [ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        [],
    ); // 2: kernel(out, n, key_lo, key_hi, lane0)

    // --- imports from module "m": unary `atan`/`round`/`sin`/`cos`/`exp`/`sqrt` (type 0) + `pow`
    // (type 1). `sin`/`cos` are the large-argument fallback (finding C3); `exp` matches the
    // interpreter (C9); `sqrt` sidesteps the V8/arm64 inline-`f64.sqrt` regression (see the index
    // constants above). The declaration order fixes the indices `ATAN..SQRT` above — keep them in
    // sync (`sqrt` was appended last so the earlier indices never shifted).
    let mut imports = ImportSection::new();
    for name in ["atan", "round"] {
        imports.import("m", name, EntityType::Function(0));
    }
    imports.import("m", "pow", EntityType::Function(1));
    for name in ["sin", "cos", "exp", "sqrt"] {
        imports.import("m", name, EntityType::Function(0));
    }

    // --- the local kernel function (index N_IMPORTS, type 2) ---
    let mut functions = FunctionSection::new();
    functions.function(2);

    // --- one linear memory: one BATCH-f64 column per root at/after 4096, then gather tables ---
    // Host convention: columns at/after 4096 (the counter kernel keeps no state region; the first
    // page stays reserved so the host layout is unchanged). Const gather tables live after the
    // columns — `cols_end` is 8-aligned by construction — initialized by an active data segment,
    // so the host needs no wiring at all.
    let mut memories = MemorySection::new();
    let cols_end = 4096 + roots.len() * BATCH * 8;
    let (gather_tables, table_data) = collect_gather_tables(graph, roots, cols_end as u64);
    let bytes = cols_end + table_data.len();
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
    let f64_count = cone; // one value slot per node
    let mut func = Function::new([
        (N_I32_LOCALS, ValType::I32),      // I, LANE, V0..V3 (hash scratch)
        (T_I64, ValType::I64),             // transcendental i64 scratch
        (T_F64 + f64_count, ValType::F64), // transcendental f64 scratch + node value slots
    ]);
    // Layout (indices): params 0..4, then the i32 block (I, LANE, V0..V3), then `T_I64`
    // transcendental i64 scratch, then the f64 block: `T_F64` transcendental scratch, then the
    // node value slots. Each group is contiguous in declaration order.
    let ti = LANE0 + 1 + N_I32_LOCALS; // transcendental i64 scratch (after the i32 block)
    let tf = ti + T_I64; // first f64 local (transcendental f64 scratch)
    let ctx = Ctx {
        graph,
        fbase: tf + T_F64,
        cone,
        ti,
        tf,
        gather_tables,
    };
    {
        let mut s = func.instructions();
        emit_kernel(&mut s, &ctx, roots);
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
    // Const gather tables as ONE active data segment after the columns (binary section order puts
    // data after code). Instantiation copies it in; the kernel only ever reads it.
    if !table_data.is_empty() {
        let mut data = DataSection::new();
        data.active(
            0,
            &ConstExpr::i32_const(cols_end as i32),
            table_data.iter().copied(),
        );
        module.section(&data);
    }
    module.finish()
}

/// Gate-honoring module emission. Returns `None` for a graph the backend won't emit (e.g.
/// `Poisson`); the browser keeps the interpreter for those, exactly like the native gate.
///
/// `inline_trans = true`: this emitter inlines `ln`/`sin`/`cos` as polynomials (`emit_ln`/`emit_trig`,
/// same `crate::approx` reference as the JIT), so `normal`/`exp`/trig graphs are fusible here too and
/// the 2× transcendental win reaches the browser — the gate decision now matches the native JIT.
pub fn emit_for(graph: &RvGraph, root: RvId, draws: usize) -> Option<Vec<u8>> {
    emit_for_roots(graph, &[root], draws)
}

/// Multi-root [`emit_for`] — the gate-honoring entry point behind the joint drivers
/// (`scatter`/`describe`/`corr`/`fan`). One gate decision over the *union* cone, then one
/// multi-column kernel ([`emit_roots`]).
pub fn emit_for_roots(graph: &RvGraph, roots: &[RvId], draws: usize) -> Option<Vec<u8>> {
    // `draws` gates emit-vs-interpret the same way it does natively: emitting + instantiating a
    // module is a fixed cost, fusion is a per-draw saving, so a short query is faster interpreted.
    //
    // The threshold is the browser's own (`MIN_DRAWS_WASM`), an order of magnitude below the JIT's:
    // instantiating a ~1 KB kernel is far cheaper than a Cranelift compile, so emission pays back
    // sooner here. Same rule, each backend's measured constant.
    if !crate::kernel::profitable_roots(
        graph,
        roots,
        /* inline_trans */ true,
        draws,
        crate::kernel::MIN_DRAWS_WASM,
    ) {
        return None;
    }
    Some(emit_roots(graph, roots))
}

/// The kernel skeleton: a counted loop emitting one lane per iteration (each lane computing every
/// root from one shared memo — joint draws — into its BATCH-strided column). Stateless — nothing
/// to load or store around the loop. Mirrors `jit::build_kernel`.
fn emit_kernel(s: &mut InstructionSink, ctx: &Ctx, roots: &[RvId]) {
    s.i32_const(0).local_set(I);
    s.local_get(LANE0).local_set(LANE);

    // block { loop { if I >= N: break; <body>; I += 1; LANE += 1; continue } }
    s.block(BlockType::Empty);
    s.loop_(BlockType::Empty);
    s.local_get(I).local_get(N).i32_ge_s().br_if(1); // break to the block end

    // One memo per lane, shared across all roots (joint draws).
    let mut memo: HashMap<RvId, u32> = HashMap::new();
    let mut slot = 0u32;
    for (r, &root) in roots.iter().enumerate() {
        // Address of out[I]: out + I * 8, pushed before the value so f64.store sees both; root
        // `r`'s column stride is the constant store offset `r * BATCH * 8`.
        s.local_get(OUT)
            .local_get(I)
            .i32_const(8)
            .i32_mul()
            .i32_add();
        // Emit this root's draw (stack-neutral), leaving its value in a local.
        let lroot = emit_node(s, ctx, root, &mut memo, &mut slot);
        s.local_get(lroot).f64_store(mem8((r * BATCH * 8) as u64));
    }

    s.local_get(I).i32_const(1).i32_add().local_set(I);
    s.local_get(LANE).i32_const(1).i32_add().local_set(LANE);
    s.br(0); // continue the loop
    s.end(); // end loop
    s.end(); // end block
    s.end(); // end function
}

/// Emit node `id`, memoizing each `RvId` into its own f64 local so a shared sub-RV is computed
/// once (the CSE guarantee the interpreter/JIT also give — and, with counter keying, the same
/// *draw*: one hash per node per lane). Children are emitted first (each `local.set`s its slot and
/// leaves the stack untouched); the parent then `local.get`s them. Returns the local index holding
/// this node's value.
fn emit_node(
    s: &mut InstructionSink,
    ctx: &Ctx,
    id: RvId,
    memo: &mut HashMap<RvId, u32>,
    slot: &mut u32,
) -> u32 {
    if let Some(&l) = memo.get(&id) {
        return l;
    }
    // Each branch leaves exactly one f64 on the stack; we `local.set` it into this node's slot below.
    match ctx.graph.node(id) {
        RvNode::Src(Source::Uniform(u)) => emit_uniform(s, id.0, u.lo, u.hi),
        RvNode::Src(Source::UniformInt { lo, hi }) => emit_uniform_int(s, ctx, id.0, *lo, *hi),
        RvNode::Src(Source::Normal { mu, sigma }) => emit_normal(s, ctx, id.0, *mu, *sigma),
        RvNode::Src(Source::Exp { rate }) => emit_exp(s, ctx, id.0, *rate),
        RvNode::Src(Source::Geometric { p }) => emit_geometric(s, ctx, id.0, *p),
        RvNode::Src(Source::Poisson { .. }) => unreachable!("profitable() excludes Poisson"),
        RvNode::Permutation { .. } | RvNode::Rotation { .. } | RvNode::ArrIndex { .. } => {
            unreachable!("profitable() excludes the array-valued draw nodes")
        }
        RvNode::Gather { elems, index } => {
            let lx = emit_node(s, ctx, *index, memo, slot);
            match ctx.gather_tables.get(&id) {
                // Const table (strategy A): round ties-away, clamp to [0, last], one indexed
                // 8-byte load from the data segment — bit-identical to the interpreter's
                // `Inst::Gather`. See `jit::emit_gather_table` for the edge-case reasoning
                // (ties-away via `nearest` + `d == 0.5` correction; ±inf clamp to the ends;
                // `i32.trunc_sat` maps NaN to 0 where plain trunc would trap, and the final
                // `x != x` select restores NaN-index → NaN, never element 0).
                Some(&addr) => {
                    let last = (elems.len() - 1) as f64; // table never empty (eval rejects [])
                    let r0 = ctx.tf; // scratch: nearest(x) (free — no nested emission here)
                    s.f64_const(f64c(f64::NAN)); // NaN-guard arm (select pops a, b, cond)
                    s.local_get(lx).f64_nearest().local_set(r0);
                    // r = r0 + ((x - r0 == 0.5) ? 1 : 0) — ties-away correction
                    s.local_get(r0);
                    s.f64_const(f64c(1.0));
                    s.f64_const(f64c(0.0));
                    s.local_get(lx)
                        .local_get(r0)
                        .f64_sub()
                        .f64_const(f64c(0.5))
                        .f64_eq();
                    s.select();
                    s.f64_add();
                    // clamp to [0, last] (f64.min/max propagate NaN; trunc_sat then yields 0)
                    s.f64_const(f64c(0.0))
                        .f64_max()
                        .f64_const(f64c(last))
                        .f64_min();
                    s.i32_trunc_sat_f64_s().i32_const(8).i32_mul();
                    s.f64_load(mem8(addr));
                    // result = (x != x) ? NaN : loaded
                    s.local_get(lx).local_get(lx).f64_ne();
                    s.select();
                }
                // Small non-const table (strategy B): compare/select chain, no rounding — see
                // `jit::emit_gather_chain`. The condition is flipped (`x >= i+0.5 ? acc : e[i]`)
                // so the accumulator stays below the operand on the stack; for non-NaN `x` that
                // is the same chain, and NaN lanes (which fail every compare and would fall
                // through to e[0] here) are overridden by the final NaN guard anyway.
                None => {
                    let les: Vec<u32> = elems
                        .iter()
                        .map(|&e| emit_node(s, ctx, e, memo, slot))
                        .collect();
                    let last = les.len() - 1;
                    s.f64_const(f64c(f64::NAN)); // NaN-guard arm
                    s.local_get(les[last]); // acc = e[last]
                    for i in (0..last).rev() {
                        s.local_get(les[i]);
                        s.local_get(lx).f64_const(f64c(i as f64 + 0.5)).f64_ge();
                        s.select(); // acc = (x >= i+0.5) ? acc : e[i]
                    }
                    s.local_get(lx).local_get(lx).f64_ne();
                    s.select(); // result = (x != x) ? NaN : acc
                }
            }
        }
        RvNode::ConstNum(x) => {
            s.f64_const(f64c(*x));
        }
        RvNode::ConstBool(b) => {
            s.f64_const(f64c(if *b { 1.0 } else { 0.0 }));
        }
        RvNode::Unary(op, a) => {
            let la = emit_node(s, ctx, *a, memo, slot);
            emit_unary(s, ctx, *op, la);
        }
        RvNode::Binary(BinOp::Pow, a, b) => {
            let la = emit_node(s, ctx, *a, memo, slot);
            match const_int_exponent(ctx.graph, *b) {
                Some(k) => emit_pow(s, la, k), // repeated multiply, no call
                None => {
                    let lb = emit_node(s, ctx, *b, memo, slot);
                    s.local_get(la).local_get(lb).call(POW);
                }
            }
        }
        RvNode::Binary(op, a, b) => {
            let la = emit_node(s, ctx, *a, memo, slot);
            let lb = emit_node(s, ctx, *b, memo, slot);
            emit_binary(s, *op, la, lb);
        }
        RvNode::Select { cond, a, b } => {
            let lc = emit_node(s, ctx, *cond, memo, slot);
            let la = emit_node(s, ctx, *a, memo, slot);
            let lb = emit_node(s, ctx, *b, memo, slot);
            // wasm `select` pops [a, b, cond_i32] → a if cond != 0 else b.
            s.local_get(la)
                .local_get(lb)
                .local_get(lc)
                .f64_const(f64c(0.0))
                .f64_ne()
                .select();
        }
    }
    let l = ctx.fbase + *slot;
    *slot += 1;
    s.local_set(l);
    memo.insert(id, l);
    l
}

/// The draw-cell hash for one `(lane, source)` — the wasm transcription of [`crate::rng::cell`]
/// (pcg4d-3r) in pure i32 arithmetic, mixed in the `V0..V3` locals: LCG per word, then three
/// dependent-product rounds with a per-word xorshift between. Leaves the words in `V0..V3`.
fn emit_cell(s: &mut InstructionSink, src: u32) {
    emit_cell_at(s, src, false);
}

/// [`emit_cell`], hashing either this lane or the pair's even lane (`LANE & !1` — Box–Muller
/// pairing, see [`emit_normal`]).
fn emit_cell_at(s: &mut InstructionSink, src: u32, even_lane: bool) {
    // v = {k0, k1, lane, src}, each through v*1664525 + 1013904223 (i32 ops wrap like Rust).
    for (v, init) in [(V0, K0), (V1, K1), (V2, LANE)] {
        s.local_get(init);
        if v == V2 && even_lane {
            s.i32_const(!1).i32_and();
        }
        s.i32_const(1664525)
            .i32_mul()
            .i32_const(1013904223)
            .i32_add()
            .local_set(v);
    }
    s.i32_const(src as i32)
        .i32_const(1664525)
        .i32_mul()
        .i32_const(1013904223)
        .i32_add()
        .local_set(V3);
    for round in 0..3 {
        if round > 0 {
            for v in [V0, V1, V2, V3] {
                s.local_get(v)
                    .local_get(v)
                    .i32_const(16)
                    .i32_shr_u()
                    .i32_xor()
                    .local_set(v);
            }
        }
        for (dst, a, b) in [(V0, V1, V3), (V1, V2, V0), (V2, V0, V1), (V3, V1, V2)] {
            s.local_get(dst)
                .local_get(a)
                .local_get(b)
                .i32_mul()
                .i32_add()
                .local_set(dst);
        }
    }
}

/// The consumed 48 bits of a word pair as an i64 on the stack (`((w0 >> 8) << 24) | (w1 >> 8)`)
/// — mirrors the integer `rng::unit_f64` / `jit::emit_bits48` start from.
fn emit_bits48(s: &mut InstructionSink, w0: u32, w1: u32) {
    s.local_get(w0)
        .i32_const(8)
        .i32_shr_u()
        .i64_extend_i32_u()
        .i64_const(24)
        .i64_shl()
        .local_get(w1)
        .i32_const(8)
        .i32_shr_u()
        .i64_extend_i32_u()
        .i64_or();
}

/// Uniform `f64` in `[0, 1)` from a word pair — mirrors `rng::unit_f64` (`bits · 2⁻⁴⁸`).
fn emit_unit48(s: &mut InstructionSink, w0: u32, w1: u32) {
    emit_bits48(s, w0, w1);
    s.f64_convert_i64_u()
        .f64_const(f64c(1.0 / ((1u64 << 48) as f64)))
        .f64_mul();
}

fn emit_uniform(s: &mut InstructionSink, src: u32, lo: f64, hi: f64) {
    emit_cell(s, src);
    emit_unit48(s, V0, V1);
    s.f64_const(f64c(hi - lo))
        .f64_mul()
        .f64_const(f64c(lo))
        .f64_add();
}

/// Below this count, `unif_int` computes the exact 48-bit Lemire multiply-high (bit-identical to
/// the interpreter/JIT); at or above it — unreachable once Track B's 2^24 cap lands — it falls
/// back to the float method (`lo + min(floor(u·count), count-1)`, identical in distribution).
/// The bound keeps the split multiply's `b_hi·count` term under 2^63.
const LEMIRE_MAX_COUNT: u64 = 1 << 39;

/// `unif_int(lo, hi)` via the same 48-bit Lemire multiply-high as the interpreter/JIT. Wasm has no
/// `mulhi`, so split `bits = b_hi·2^24 + b_lo` and use
/// `(bits·count) >> 48 = (b_hi·count + ((b_lo·count) >> 24)) >> 24` — exact (standard nested-floor
/// identity), and every term fits i64 for `count < 2^39`.
fn emit_uniform_int(s: &mut InstructionSink, ctx: &Ctx, src: u32, lo: f64, hi: f64) {
    let count = (hi - lo + 1.0).max(1.0);
    emit_cell(s, src);
    if (count as u64) < LEMIRE_MAX_COUNT {
        let bits = ctx.ti; // i64 scratch
        emit_bits48(s, V0, V1);
        s.local_set(bits);
        s.local_get(bits)
            .i64_const(24)
            .i64_shr_u()
            .i64_const(count as u64 as i64)
            .i64_mul(); // b_hi * count
        s.local_get(bits)
            .i64_const(0xFF_FFFF)
            .i64_and()
            .i64_const(count as u64 as i64)
            .i64_mul()
            .i64_const(24)
            .i64_shr_u(); // (b_lo * count) >> 24
        s.i64_add()
            .i64_const(24)
            .i64_shr_u()
            .f64_convert_i64_u()
            .f64_const(f64c(lo))
            .f64_add();
    } else {
        emit_unit48(s, V0, V1);
        s.f64_const(f64c(count))
            .f64_mul()
            .f64_floor()
            .f64_const(f64c(count - 1.0))
            .f64_min()
            .f64_const(f64c(lo))
            .f64_add();
    }
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

/// `N(mu, sigma^2)` via Box–Muller over lane pairs — mirrors `rng::normal_pair` bit-for-bit:
/// hash the pair's EVEN lane (`LANE & !1`), `u1` from words 0+1 offset by half a 48-bit-grid ulp,
/// `u2` from words 2+3; even lane takes the cos branch, odd the sin branch (parity select).
/// `ln`/trig are the shared inlined polynomials; `sqrt` is native.
fn emit_normal(s: &mut InstructionSink, ctx: &Ctx, src: u32, mu: f64, sigma: f64) {
    use std::f64::consts::TAU;
    emit_cell_at(s, src, true);
    // r = sqrt(-2 * ln((bits48 + 0.5) * 2^-48))
    emit_bits48(s, V0, V1);
    s.f64_convert_i64_u()
        .f64_const(f64c(0.5))
        .f64_add()
        .f64_const(f64c(1.0 / ((1u64 << 48) as f64)))
        .f64_mul();
    emit_ln(s, ctx);
    s.f64_const(f64c(-2.0)).f64_mul().f64_sqrt();
    // z = parity-selected Box–Muller branch of angle TAU * u2 ∈ [0, 2π) — always inside the
    // polynomial's range, so call `emit_trig_poly` directly (skip the range guard).
    emit_unit48(s, V2, V3);
    s.f64_const(f64c(TAU)).f64_mul().local_set(ctx.tf);
    // wasm select pops [t1, t2, cond] and picks t1 when cond != 0: push cos (even) first.
    // `emit_trig_poly` consumes its input local as the quadrant accumulator, so recompute the
    // angle (the cell words are still live in V2/V3) before the sin call.
    emit_trig_poly(s, ctx, ctx.tf, true); // cos branch (even lanes)
    emit_unit48(s, V2, V3);
    s.f64_const(f64c(TAU)).f64_mul().local_set(ctx.tf);
    emit_trig_poly(s, ctx, ctx.tf, false); // sin branch (odd lanes)
    s.local_get(LANE).i32_const(1).i32_and().i32_eqz();
    s.select(); // (lane even) ? cos : sin
    s.f64_mul()
        .f64_const(f64c(sigma))
        .f64_mul()
        .f64_const(f64c(mu))
        .f64_add();
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `rng::fill_exp` bit-for-bit.
fn emit_exp(s: &mut InstructionSink, ctx: &Ctx, src: u32, rate: f64) {
    emit_cell(s, src);
    s.f64_const(f64c(1.0));
    emit_unit48(s, V0, V1);
    s.f64_sub();
    emit_ln(s, ctx);
    s.f64_neg().f64_const(f64c(rate)).f64_div();
}

/// `Geometric(p)` via `floor(ln(1 - u) / ln(1 - p))` — mirrors `rng::fill_geometric` bit-for-bit.
/// `ln(1 - p)` is a compile-time constant; `p == 1` makes it `-inf`, so every draw floors to `0`
/// (the point mass).
fn emit_geometric(s: &mut InstructionSink, ctx: &Ctx, src: u32, p: f64) {
    let denom = (1.0 - p).ln();
    emit_cell(s, src);
    s.f64_const(f64c(1.0));
    emit_unit48(s, V0, V1);
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
        UnOp::Sqrt => {
            // Imported `Math.sqrt`, NOT the inline `f64.sqrt` instruction: V8/arm64 regresses
            // ~30% on large single-block kernel bodies when these become inline sqrt (measured
            // 2026-07-14, `am_vs_fm` +21% run-level — the import calls serve as live-range split
            // points for V8's regalloc). JSC prefers inline; revisit if V8 improves. `Math.sqrt`
            // is IEEE correctly rounded, bit-identical to the interpreter's `f64::sqrt` on the
            // whole domain (incl. -0.0 → -0.0, x<0 → NaN) — never `pow(x, 0.5)`, which disagrees
            // at -0.0 and -inf (PLAN-PERF-2 §5). The native JIT keeps its inline `sqrt`.
            s.local_get(a).call(SQRT);
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

    /// These tests exercise the *emitted kernel itself*, so they must compile regardless of the
    /// amortization gate (`kernel::break_even_draws`), which would otherwise interpret a short run.
    /// A draw count this large always clears it.
    const ENOUGH_DRAWS: usize = usize::MAX;

    use crate::backend::{Backend, InterpBackend};
    use crate::kernel::supported;
    use crate::rng::Key;
    use crate::sampler::moments;
    use wasmi::{Engine, Linker, Module as WasmModule, Store};

    // The shared cross-backend conformance corpus (finding C2), also consumed by `jit`.
    use crate::conformance;

    /// Instantiate an emitted kernel in `wasmi`, run one batch at lane 0, and return `out[0]`. For
    /// an RNG-free graph every lane is identical, so `[0]` fully characterizes the backend's output.
    fn first_emitted(bytes: &[u8], seed: u64) -> f64 {
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
        linker.func_wrap("m", "sqrt", |x: f64| x.sqrt()).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let memory = instance.get_memory(&store, "memory").unwrap();
        let kernel = instance
            .get_typed_func::<(i32, i32, i32, i32, i32), ()>(&store, "kernel")
            .unwrap();
        let out_ptr = 4096i32;
        let key = Key::from_seed(seed);
        kernel
            .call(
                &mut store,
                (out_ptr, BATCH as i32, key.k0 as i32, key.k1 as i32, 0),
            )
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
            let mut ir = InterpBackend.compile(g, id, ENOUGH_DRAWS).runner();
            ir.position(0, 0);
            let cap = ir.batch_cap();
            let interp = ir.next_batch(cap)[0];
            let wasm = first_emitted(&emit(g, id), 0);
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
    /// mean of every sample produced. The host supplies the transcendental imports (Rust `f64`
    /// methods) and advances the lane cursor across calls — the kernel itself is stateless —
    /// exactly as a browser host would drive it.
    fn run_emitted(bytes: &[u8], seed: u64, batches: usize) -> (f64, u64) {
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
        linker.func_wrap("m", "sqrt", |x: f64| x.sqrt()).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let memory = instance.get_memory(&store, "memory").unwrap();
        let kernel = instance
            .get_typed_func::<(i32, i32, i32, i32, i32), ()>(&store, "kernel")
            .unwrap();

        // Host memory convention: output column at 4096.
        let out_ptr: i32 = 4096;
        let cap = BATCH;
        let key = Key::from_seed(seed);

        let mut sum = 0.0f64;
        let mut count = 0u64;
        let mut out_bytes = vec![0u8; cap * 8];
        for b in 0..batches {
            let lane0 = (b * cap) as u32;
            kernel
                .call(
                    &mut store,
                    (
                        out_ptr,
                        cap as i32,
                        key.k0 as i32,
                        key.k1 as i32,
                        lane0 as i32,
                    ),
                )
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

    /// The emitted WASM kernel and the interpreter must agree DRAW-FOR-DRAW (counter keying makes
    /// the streams bit-identical — the PLAN-PREGPU parity contract), plus a mean check over a
    /// longer run to guard the host-side lane advance.
    fn assert_wasm_matches_interp(src: &str, seed: u64) {
        let (eng, id) = graph_of(src);
        let graph = eng.graph();
        assert!(
            supported(graph, id),
            "case must be codegen-supported: {src}"
        );
        let bytes = emit(graph, id);

        // Bitwise: first batch, lane 0, against the interpreter oracle.
        let mut ir = InterpBackend.compile(graph, id, ENOUGH_DRAWS).runner();
        ir.position(seed, 0);
        let interp_col: Vec<f64> = ir.next_batch(BATCH).to_vec();
        let wasm0 = first_emitted(&bytes, seed);
        assert_eq!(
            interp_col[0].to_bits(),
            wasm0.to_bits(),
            "{src}: lane 0: interp {} ({:#018x}) != wasm {wasm0} ({:#018x})",
            interp_col[0],
            interp_col[0].to_bits(),
            wasm0.to_bits()
        );

        let (wasm_mean, count) = run_emitted(&bytes, seed, 16);
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
        let (mean, _) = run_emitted(&emit(eng.graph(), id), 14, 8);
        assert_eq!(mean, 0.0, "X - X must be identically zero (shared draw)");
    }

    /// The gate-honoring [`emit_for`] emits every supported graph shape — RNG-bound, inlined
    /// transcendentals, call-bearing `pow` — and each must match the interpreter's mean.
    #[test]
    fn gate_and_distribution() {
        for (label, src) in [
            ("dice-sum", "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B"),
            ("normal-sum", "use rand; X ~ normal(0,1); Y ~ normal(0,1); X + Y"),
            ("pow", "use rand; A ~ unif(1,2); B ~ unif(1,2); A ^ B"),
        ] {
            let (eng, id) = graph_of(src);
            let bytes = emit_for(eng.graph(), id, ENOUGH_DRAWS).expect("graph should emit");
            let (wasm_mean, count) = run_emitted(&bytes, 0xABCDEF, 64);
            let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).mean;
            assert!(
                (wasm_mean - interp).abs() < 0.05,
                "{label}: wasm={wasm_mean} interp={interp}"
            );
        }
    }

    /// Drive a multi-column joint kernel ([`emit_roots`]) through `wasmi`: run `batches` batches
    /// (advancing the lane cursor), return the `k` concatenated columns. The kernel writes column
    /// `r` at `out + r*BATCH*8` — the same layout `wasm_host::nz_kernel_run_cols` copies out.
    fn run_emitted_joint(bytes: &[u8], k: usize, seed: u64, batches: usize) -> Vec<Vec<f64>> {
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
        linker.func_wrap("m", "sqrt", |x: f64| x.sqrt()).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let memory = instance.get_memory(&store, "memory").unwrap();
        let kernel = instance
            .get_typed_func::<(i32, i32, i32, i32, i32), ()>(&store, "kernel")
            .unwrap();
        let out_ptr = 4096i32;
        let key = Key::from_seed(seed);
        let mut cols = vec![Vec::new(); k];
        let mut out_bytes = vec![0u8; k * BATCH * 8];
        for b in 0..batches {
            let lane0 = (b * BATCH) as u32;
            kernel
                .call(
                    &mut store,
                    (
                        out_ptr,
                        BATCH as i32,
                        key.k0 as i32,
                        key.k1 as i32,
                        lane0 as i32,
                    ),
                )
                .unwrap();
            memory
                .read(&store, out_ptr as usize, &mut out_bytes)
                .unwrap();
            for (j, col) in cols.iter_mut().enumerate() {
                for chunk in out_bytes[j * BATCH * 8..(j + 1) * BATCH * 8].chunks_exact(8) {
                    col.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
        }
        cols
    }

    /// THE joint invariant, on the emitted multi-column kernel: roots sharing a source read the
    /// same per-lane draw.
    #[test]
    fn joint_kernel_shares_draws() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(crate::dist::Uniform { lo: 0.0, hi: 1.0 })),
            crate::dist::RvKind::Num,
        );
        let two = g.push(RvNode::ConstNum(2.0), crate::dist::RvKind::Num);
        let x2 = g.push(RvNode::Binary(BinOp::Mul, x, two), crate::dist::RvKind::Num);
        let bytes =
            emit_for_roots(&g, &[x, x2], ENOUGH_DRAWS).expect("pure-unif joint cone should emit");
        let cols = run_emitted_joint(&bytes, 2, 11, 4);
        for i in 0..cols[0].len() {
            let (a, b) = (cols[0][i], cols[1][i]);
            assert!(
                (0.0..1.0).contains(&a),
                "lane {i}: unif draw out of range: {a}"
            );
            assert_eq!(b, 2.0 * a, "lane {i}: columns must pair on one shared draw");
        }
        assert!(cols[0].windows(2).any(|w| w[0] != w[1]), "draws must vary");
    }

    /// Joint kernel with a transcendental cone (single-stream) must match the interpreter in
    /// distribution per column — the multi-column twin of `assert_wasm_matches_interp`.
    #[test]
    fn joint_kernel_matches_interp_marginals() {
        let mut g = RvGraph::default();
        let z = g.push(
            RvNode::Src(Source::Normal {
                mu: 1.0,
                sigma: 2.0,
            }),
            crate::dist::RvKind::Num,
        );
        let one = g.push(RvNode::ConstNum(1.0), crate::dist::RvKind::Num);
        let z1 = g.push(RvNode::Binary(BinOp::Add, z, one), crate::dist::RvKind::Num);
        let bytes =
            emit_for_roots(&g, &[z, z1], ENOUGH_DRAWS).expect("normal joint cone should emit");
        let cols = run_emitted_joint(&bytes, 2, 12, 32);
        let n = cols[0].len() as f64;
        let (m0, m1) = (
            cols[0].iter().sum::<f64>() / n,
            cols[1].iter().sum::<f64>() / n,
        );
        assert!((m0 - 1.0).abs() < 0.05, "E[Z]≈1, got {m0}");
        assert!((m1 - 2.0).abs() < 0.05, "E[Z+1]≈2, got {m1}");
        for i in 0..cols[0].len() {
            assert_eq!(cols[1][i], cols[0][i] + 1.0, "lane {i} must pair");
        }
    }

    /// The joint gate rejects an unsupported node anywhere in the union cone (here `Poisson` in the
    /// second root), exactly like the single-root gate.
    #[test]
    fn joint_gate_rejects_unsupported_union() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(crate::dist::Uniform { lo: 0.0, hi: 1.0 })),
            crate::dist::RvKind::Num,
        );
        let p = g.push(
            RvNode::Src(Source::Poisson { lambda: 3.0 }),
            crate::dist::RvKind::Num,
        );
        assert!(
            emit_for_roots(&g, &[x, p], ENOUGH_DRAWS).is_none(),
            "a Poisson root must keep the whole joint pass on the interpreter"
        );
    }

    /// A const-table gather is a LEAF over its elems ([`crate::kernel::gather_class`]): a 10k-point
    /// `empirical`-shaped table takes no value slots and doesn't trip `MAX_CODEGEN_NODES` — the gate
    /// emits it, the table rides the active data segment after the output columns, the cone stays
    /// latency-bound (multi-stream), and the kernel reproduces the uniform-over-the-data mean.
    #[test]
    fn const_table_gather_emits_as_leaf() {
        let mut g = RvGraph::default();
        let elems: Vec<RvId> = (0..10_000)
            .map(|i| g.push(RvNode::ConstNum(i as f64), crate::dist::RvKind::Num))
            .collect();
        let idx = g.push(
            RvNode::Src(Source::UniformInt {
                lo: 0.0,
                hi: 9999.0,
            }),
            crate::dist::RvKind::Num,
        );
        let root = g.push(
            RvNode::Gather {
                elems: elems.into_boxed_slice(),
                index: idx,
            },
            crate::dist::RvKind::Num,
        );
        // The counted cone (= the wasm value-slot pool) is just {gather, index}.
        assert_eq!(crate::kernel::cone_size(&g, root), 2);
        let bytes = emit_for(&g, root, ENOUGH_DRAWS).expect("const-table gather should emit");
        let (mean, _) = run_emitted(&bytes, 42, 16);
        // E over unif_int(0, 9999) of table[i] = i is 4999.5; fixed seed, generous tolerance.
        assert!((mean - 4999.5).abs() < 100.0, "mean={mean}");
    }

    #[test]
    fn poisson_is_not_emitted() {
        // Poisson stays interpreter-only: the gate returns None (the browser would keep the interp).
        let (eng, id) = graph_of("use rand; K ~ poisson(3); K");
        assert!(!supported(eng.graph(), id));
        assert!(
            emit_for(eng.graph(), id, ENOUGH_DRAWS).is_none(),
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
        let bytes = emit_for(eng.graph(), id, ENOUGH_DRAWS).expect("should emit");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("wrote {} bytes to {path}", bytes.len());
    }
}
