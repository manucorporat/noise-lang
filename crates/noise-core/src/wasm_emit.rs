//! WASM-emitter backend (PLAN.md Phase 4 "Browser note") — the browser's twin of [`crate::jit`].
//!
//! A WASM sandbox can't emit or run native code, so the Cranelift JIT (B1/B2) is native-only. What
//! *is* portable is the [`RvGraph`] IR. This module is the second lowering of that IR: it walks the
//! identical graph the identical post-order way and emits the identical fused counter-keyed kernel
//! (PLAN-PREGPU Track C) — only the per-node encoding differs (`f32.mul` instead of `mulss`, the
//! same inlined squares64 hash in either). The shared "what the graph means" lives in
//! [`crate::kernel`] (the cost/profitability gate); this file is just "how to spell it in wasm
//! bytes".
//!
//! **Output.** [`emit`] produces a complete WebAssembly module (`Vec<u8>`) exporting:
//!   * `memory` — one linear memory the host reads/writes.
//!   * `kernel(out: i32, n: i32, key_lo: i32, key_hi: i32, lane0: i32)` — fills `out[0..n]` with
//!     the root draws for global lanes `lane0 .. lane0 + n`, as **f32** (PLAN-PREGPU Track B).
//!     Stateless — same arguments, same column, bit-identical to the interpreter and the native
//!     JIT under the same seed.
//!
//! **f32 lanes.** Values are `f32` throughout and one squares64 draw feeds a whole lane PAIR (48
//! consumed bits = two 24-bit uniforms), so the pair-unrolled loop hashes once per two lanes for
//! every source but `unif_int`. The transcendental *imports* stay `f64` (that is what `Math.*` is):
//! call sites promote and demote around them, which is exactly the contract the interpreter and the
//! JIT's shims follow — so the fallback paths agree bit-for-bit too.
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
    Function, FunctionSection, Ieee32, ImportSection, InstructionSink, MemArg, MemorySection,
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
/// i32 scratch (2, at `ctx.tj`): the f32 `ln`'s bit-surgery fields, and the trig quadrant — never
/// live at the same time (trig runs after `ln` has finished).
const T_I32: u32 = 2;
const N_I32_LOCALS: u32 = 2 + T_I32; // I, LANE, + the i32 scratch pair

/// 4-byte aligned access (f32) into memory 0 at an absolute address already on the stack.
fn mem4(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 2,
        memory_index: 0,
    }
}

/// f32 literal as the `Ieee32` the encoder wants.
fn f32c(x: f32) -> Ieee32 {
    Ieee32::from(x)
}

/// Scratch locals: 3 i64 for the inlined squares64 draw plus the persistent 64-bit key (`ti + 3`,
/// set once in the prologue); 8 f32 — `ln`/trig scratch in `tf .. tf+5`, the Box–Muller radius at
/// `tf + 6` and the Box–Muller angle at `tf + 7` (both must survive the pair's two trig
/// evaluations, which consume their input local as the quadrant accumulator).
const T_I64: u32 = 4;
const T_F32: u32 = 8;

/// Cones at most this many nodes pair-unroll the kernel loop (two lanes per iteration, so a
/// `Normal`'s hash + ln + trig pair is computed once and split across the even/odd lane); larger
/// ones keep the single-lane parity-select loop — the unroll doubles the value-slot local pool,
/// and huge straight-line bodies are memory-bound anyway. Mirrors `jit::PAIR_UNROLL_MAX_NODES`.
const PAIR_UNROLL_MAX_NODES: usize = 2048;

/// How the lane being emitted relates to the **shared pair draw** (Track B): with f32 lanes one
/// squares64 output serves two lanes, so `unif`, `normal`, `exp` and `geometric` all hash once per
/// pair. `Even` banks each such node's odd-lane value into a dedicated local (`bank[id]`); `Odd`
/// reads it back; `Single` (the non-unrolled loop) hashes the pair's counter and selects its half
/// by lane parity — same values, twice the hashing.
enum PairMode {
    Single,
    Even {
        bank: HashMap<RvId, u32>,
        next_bank: u32,
    },
    Odd(HashMap<RvId, u32>),
}

/// Which half of the pair draw this emission is for — the static form of the parity select.
#[derive(Clone, Copy, PartialEq)]
enum Half {
    /// Not statically known (the non-unrolled loop): select by `LANE & 1`.
    ByParity,
    /// The even lane's half (low 24 bits).
    Lo,
    /// The odd lane's half (high 24 bits).
    Hi,
}

/// Per-emission constants shared across the whole kernel body.
struct Ctx<'g> {
    graph: &'g RvGraph,
    /// Base index of the f64 value-slot block; node memo locals live above this.
    fbase: u32,
    /// Distinct nodes per stream (each stream gets its own contiguous block of f64 slots).
    cone: u32,
    /// Base index of the [`T_I32`] i32 scratch locals (`ln`'s bit-surgery / the trig quadrant).
    tj: u32,
    /// Base index of the [`T_I64`] i64 scratch locals (the persistent key sits at `ti + 3`).
    ti: u32,
    /// Base index of the [`T_F32`] transcendental f32 scratch locals.
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
                        // f32 lanes: the table holds the same `x as f32` the other backends bake in.
                        data.extend_from_slice(&(*x as f32).to_le_bytes());
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
    // The imports stay f64 — they ARE `Math.*` — and the f32 call sites promote/demote around
    // them. That is the shared contract (the JIT's shims compute in f64 and round too).
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

    // --- one linear memory: one BATCH-f32 column per root at/after 4096, then gather tables ---
    // Host convention: columns at/after 4096 (the counter kernel keeps no state region; the first
    // page stays reserved so the host layout is unchanged). Const gather tables live after the
    // columns — `cols_end` is 4-aligned by construction — initialized by an active data segment,
    // so the host needs no wiring at all.
    let mut memories = MemorySection::new();
    let cols_end = 4096 + roots.len() * BATCH * 4;
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
    // Pair-unrolled kernels need two value-slot pools (even + odd lane memos) plus a bank pool
    // for the odd lane's Normal values (≤ one per node) — 3×cone stays far under engine local
    // limits at the unroll cap.
    let unroll = (cone as usize) <= PAIR_UNROLL_MAX_NODES;
    let f32_count = if unroll { 3 * cone } else { cone };
    let mut func = Function::new([
        (N_I32_LOCALS, ValType::I32),      // I, LANE, + 2 i32 scratch
        (T_I64, ValType::I64),             // draw scratch + the persistent 64-bit key
        (T_F32 + f32_count, ValType::F32), // transcendental f32 scratch + node value slots
    ]);
    // Layout (indices): params 0..4, then the i32 block (I, LANE, scratch), then the i64 block
    // (scratch + key), then the f32 block: `T_F32` transcendental scratch, then the node value
    // slots. Each group is contiguous in declaration order.
    let tj = LANE + 1; // i32 scratch (after I, LANE)
    let ti = LANE0 + 1 + N_I32_LOCALS; // i64 scratch (after the i32 block)
    let tf = ti + T_I64; // first f32 local (transcendental scratch)
    let ctx = Ctx {
        graph,
        fbase: tf + T_F32,
        cone,
        tj,
        ti,
        tf,
        gather_tables,
    };
    {
        let mut s = func.instructions();
        emit_kernel(&mut s, &ctx, roots, unroll);
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

/// The kernel skeleton: a counted loop (each lane computing every root from one shared memo —
/// joint draws — into its BATCH-strided column). Stateless — nothing to load or store around the
/// loop. Mirrors `jit::build_kernel`, including the pair-unroll: small cones emit TWO lanes per
/// iteration so a `Normal`'s Box–Muller pair — hash, ln, both trig branches — is computed once
/// and split cos/sin across the even/odd lane (`n` is always the even BATCH from every runner);
/// large cones keep the single-lane loop with parity-select normals — same draws.
fn emit_kernel(s: &mut InstructionSink, ctx: &Ctx, roots: &[RvId], unroll: bool) {
    // Reassemble the squares key: two u32 params → one persistent i64 local.
    let k64 = ctx.ti + 3;
    s.local_get(K1)
        .i64_extend_i32_u()
        .i64_const(32)
        .i64_shl()
        .local_get(K0)
        .i64_extend_i32_u()
        .i64_or()
        .local_set(k64);
    s.i32_const(0).local_set(I);
    s.local_get(LANE0).local_set(LANE);
    let step = if unroll { 2 } else { 1 };

    // block { loop { if I >= N: break; <body>; I += step; LANE += step; continue } }
    s.block(BlockType::Empty);
    s.loop_(BlockType::Empty);
    s.local_get(I).local_get(N).i32_ge_s().br_if(1); // break to the block end

    // Store this lane's roots: address of out[I + lane_off] pushed before the value so
    // f64.store sees both; root `r`'s column stride is the constant store offset.
    let emit_lane = |s: &mut InstructionSink,
                     memo: &mut HashMap<RvId, u32>,
                     slot: &mut u32,
                     pair: &mut PairMode,
                     lane_off: u64| {
        for (r, &root) in roots.iter().enumerate() {
            s.local_get(OUT)
                .local_get(I)
                .i32_const(4)
                .i32_mul()
                .i32_add();
            let lroot = emit_node(s, ctx, root, memo, slot, pair);
            s.local_get(lroot)
                .f32_store(mem4((r * BATCH) as u64 * 4 + lane_off * 4));
        }
    };

    let mut slot = 0u32;
    if unroll {
        // Even lane: fresh memo; Normal nodes bank their sin branch (bank pool sits above the
        // two value-slot pools).
        let mut memo: HashMap<RvId, u32> = HashMap::new();
        let mut pair = PairMode::Even {
            bank: HashMap::new(),
            next_bank: ctx.fbase + 2 * ctx.cone,
        };
        emit_lane(s, &mut memo, &mut slot, &mut pair, 0);
        // Odd lane: LANE += 1, fresh memo (slots continue into the second pool), Normal nodes
        // read the bank.
        s.local_get(LANE).i32_const(1).i32_add().local_set(LANE);
        let PairMode::Even { bank, .. } = pair else {
            unreachable!()
        };
        let mut pair = PairMode::Odd(bank);
        let mut memo: HashMap<RvId, u32> = HashMap::new();
        emit_lane(s, &mut memo, &mut slot, &mut pair, 1);
    } else {
        let mut memo: HashMap<RvId, u32> = HashMap::new();
        let mut pair = PairMode::Single;
        emit_lane(s, &mut memo, &mut slot, &mut pair, 0);
    }

    s.local_get(I).i32_const(step).i32_add().local_set(I);
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
#[allow(clippy::too_many_arguments)] // the emitter's full walking state; a struct would just rename it
fn emit_node(
    s: &mut InstructionSink,
    ctx: &Ctx,
    id: RvId,
    memo: &mut HashMap<RvId, u32>,
    slot: &mut u32,
    pair: &mut PairMode,
) -> u32 {
    if let Some(&l) = memo.get(&id) {
        return l;
    }
    // Each branch leaves exactly one f64 on the stack; we `local.set` it into this node's slot below.
    match ctx.graph.node(id) {
        // The four pair-shared sources all follow one protocol (Track B): in `Even` phase the
        // node hashes its pair ONCE, banks the odd lane's finished value in a dedicated local and
        // leaves the even lane's on the stack; in `Odd` phase it just reads the bank; in `Single`
        // it hashes the pair's counter and picks its half by lane parity.
        RvNode::Src(Source::Uniform(u)) => {
            let (lo, hi) = (u.lo, u.hi);
            emit_paired(s, ctx, id, pair, |s, ctx, half| {
                emit_uniform(s, ctx, id.0, lo, hi, half)
            });
        }
        RvNode::Src(Source::UniformInt { lo, hi }) => emit_uniform_int(s, ctx, id.0, *lo, *hi),
        RvNode::Src(Source::Normal { mu, sigma }) => match pair {
            PairMode::Single => emit_normal_single(s, ctx, id.0, *mu, *sigma),
            PairMode::Even { bank, next_bank } => {
                let bank_local = *next_bank;
                *next_bank += 1;
                bank.insert(id, bank_local);
                emit_normal_pair(s, ctx, id.0, *mu, *sigma, bank_local);
            }
            PairMode::Odd(bank) => {
                s.local_get(bank[&id]);
            }
        },
        RvNode::Src(Source::Exp { rate }) => {
            let rate = *rate;
            emit_paired(s, ctx, id, pair, |s, ctx, half| {
                emit_exp(s, ctx, id.0, rate, half)
            });
        }
        RvNode::Src(Source::Geometric { p }) => {
            let p = *p;
            emit_paired(s, ctx, id, pair, |s, ctx, half| {
                emit_geometric(s, ctx, id.0, p, half)
            });
        }
        RvNode::Src(Source::Poisson { .. }) => unreachable!("profitable() excludes Poisson"),
        RvNode::Permutation { .. } | RvNode::Rotation { .. } | RvNode::ArrIndex { .. } => {
            unreachable!("profitable() excludes the array-valued draw nodes")
        }
        RvNode::Gather { elems, index } => {
            let lx = emit_node(s, ctx, *index, memo, slot, pair);
            match ctx.gather_tables.get(&id) {
                // Const table (strategy A): round ties-away, clamp to [0, last], one indexed
                // 8-byte load from the data segment — bit-identical to the interpreter's
                // `Inst::Gather`. See `jit::emit_gather_table` for the edge-case reasoning
                // (ties-away via `nearest` + `d == 0.5` correction; ±inf clamp to the ends;
                // `i32.trunc_sat` maps NaN to 0 where plain trunc would trap, and the final
                // `x != x` select restores NaN-index → NaN, never element 0).
                Some(&addr) => {
                    let last = (elems.len() - 1) as f32; // table never empty (eval rejects [])
                    let r0 = ctx.tf; // scratch: nearest(x) (free — no nested emission here)
                    s.f32_const(f32c(f32::NAN)); // NaN-guard arm (select pops a, b, cond)
                    s.local_get(lx).f32_nearest().local_set(r0);
                    // r = r0 + ((x - r0 == 0.5) ? 1 : 0) — ties-away correction
                    s.local_get(r0);
                    s.f32_const(f32c(1.0));
                    s.f32_const(f32c(0.0));
                    s.local_get(lx)
                        .local_get(r0)
                        .f32_sub()
                        .f32_const(f32c(0.5))
                        .f32_eq();
                    s.select();
                    s.f32_add();
                    // clamp to [0, last] (f32.min/max propagate NaN; trunc_sat then yields 0)
                    s.f32_const(f32c(0.0))
                        .f32_max()
                        .f32_const(f32c(last))
                        .f32_min();
                    s.i32_trunc_sat_f32_s().i32_const(4).i32_mul();
                    s.f32_load(mem4(addr));
                    // result = (x != x) ? NaN : loaded
                    s.local_get(lx).local_get(lx).f32_ne();
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
                        .map(|&e| emit_node(s, ctx, e, memo, slot, pair))
                        .collect();
                    let last = les.len() - 1;
                    s.f32_const(f32c(f32::NAN)); // NaN-guard arm
                    s.local_get(les[last]); // acc = e[last]
                    for i in (0..last).rev() {
                        s.local_get(les[i]);
                        s.local_get(lx).f32_const(f32c(i as f32 + 0.5)).f32_ge();
                        s.select(); // acc = (x >= i+0.5) ? acc : e[i]
                    }
                    s.local_get(lx).local_get(lx).f32_ne();
                    s.select(); // result = (x != x) ? NaN : acc
                }
            }
        }
        // Graph constants are f64; a lane holds `x as f32` (the same rounding the interpreter and
        // the JIT apply).
        RvNode::ConstNum(x) => {
            s.f32_const(f32c(*x as f32));
        }
        RvNode::ConstBool(b) => {
            s.f32_const(f32c(if *b { 1.0 } else { 0.0 }));
        }
        RvNode::Unary(op, a) => {
            let la = emit_node(s, ctx, *a, memo, slot, pair);
            emit_unary(s, ctx, *op, la);
        }
        RvNode::Binary(BinOp::Pow, a, b) => {
            let la = emit_node(s, ctx, *a, memo, slot, pair);
            match const_int_exponent(ctx.graph, *b) {
                Some(k) => emit_pow(s, la, k), // repeated multiply, no call
                None => {
                    let lb = emit_node(s, ctx, *b, memo, slot, pair);
                    emit_pow_call(s, la, lb);
                }
            }
        }
        RvNode::Binary(op, a, b) => {
            let la = emit_node(s, ctx, *a, memo, slot, pair);
            let lb = emit_node(s, ctx, *b, memo, slot, pair);
            emit_binary(s, *op, la, lb);
        }
        RvNode::Select { cond, a, b } => {
            let lc = emit_node(s, ctx, *cond, memo, slot, pair);
            let la = emit_node(s, ctx, *a, memo, slot, pair);
            let lb = emit_node(s, ctx, *b, memo, slot, pair);
            // wasm `select` pops [a, b, cond_i32] → a if cond != 0 else b.
            s.local_get(la)
                .local_get(lb)
                .local_get(lc)
                .f32_const(f32c(0.0))
                .f32_ne()
                .select();
        }
    }
    let l = ctx.fbase + *slot;
    *slot += 1;
    s.local_set(l);
    memo.insert(id, l);
    l
}

/// Run a pair-shared source through the current [`PairMode`] (Track B). `emit` writes ONE lane's
/// value for the requested [`Half`], leaving it on the stack. In `Even` phase the odd lane's value
/// is emitted first into `bank_local` (a second hash of the same cell — wasm has no cheap way to
/// keep two stack values across the node protocol, and the hash is far cheaper than the memory
/// traffic of a spill pool); `Odd` then reads the bank; `Single` selects its half by parity.
fn emit_paired(
    s: &mut InstructionSink,
    ctx: &Ctx,
    id: RvId,
    pair: &mut PairMode,
    emit: impl Fn(&mut InstructionSink, &Ctx, Half),
) {
    match pair {
        PairMode::Single => emit(s, ctx, Half::ByParity),
        PairMode::Even { bank, next_bank } => {
            let bank_local = *next_bank;
            *next_bank += 1;
            bank.insert(id, bank_local);
            emit(s, ctx, Half::Hi); // the odd lane's value
            s.local_set(bank_local);
            emit(s, ctx, Half::Lo); // the even lane's value, left on the stack
        }
        PairMode::Odd(bank) => {
            s.local_get(bank[&id]);
        }
    }
}

/// One squares64 draw at the counter already on the stack (an i64) — the wasm transcription of
/// `rng::squares64` + the 48-bit consumption contract (bit-identical): five middle-square rounds,
/// leaving `((w >> 40) << 24) | ((w >> 8) & 0xFFFFFF)` on the stack as an i64. Computes through
/// the `X/Y/Z` i64 scratch (`ti .. ti+2`).
fn emit_draw48_at(s: &mut InstructionSink, ctx: &Ctx) {
    let (x, y, z) = (ctx.ti, ctx.ti + 1, ctx.ti + 2);
    let k64 = ctx.ti + 3;
    // x = ctr * key; y = x; z = x + key.
    s.local_get(k64).i64_mul().local_tee(x).local_set(y);
    s.local_get(x).local_get(k64).i64_add().local_set(z);
    // Three rounds: x = rotl32(x*x + {y, z, y}).
    for w in [y, z, y] {
        s.local_get(x)
            .local_get(x)
            .i64_mul()
            .local_get(w)
            .i64_add()
            .i64_const(32)
            .i64_rotl()
            .local_set(x);
    }
    // t = x*x + z (stashed in z — its last use); x = rotl32(t); w = t ^ ((x*x + y) >> 32).
    s.local_get(x)
        .local_get(x)
        .i64_mul()
        .local_get(z)
        .i64_add()
        .local_set(z);
    s.local_get(z).i64_const(32).i64_rotl().local_set(x);
    s.local_get(z);
    s.local_get(x)
        .local_get(x)
        .i64_mul()
        .local_get(y)
        .i64_add()
        .i64_const(32)
        .i64_shr_u()
        .i64_xor();
    // bits48 = ((w >> 40) << 24) | ((w >> 8) & 0xFFFFFF)   (w stashed in x)
    s.local_tee(x)
        .i64_const(40)
        .i64_shr_u()
        .i64_const(24)
        .i64_shl()
        .local_get(x)
        .i64_const(8)
        .i64_shr_u()
        .i64_const(0xFF_FFFF)
        .i64_and()
        .i64_or();
}

/// The pair draw for this lane's PAIR — `rng::pair_bits`: counter `(src << 36) + (LANE >> 1)`, so
/// both lanes of a pair hash the same cell. Leaves the 48 consumed bits on the stack.
fn emit_pair_draw(s: &mut InstructionSink, ctx: &Ctx, src: u32) {
    s.local_get(LANE)
        .i32_const(1)
        .i32_shr_u()
        .i64_extend_i32_u()
        .i64_const(((src as u64) << 36) as i64)
        .i64_add();
    emit_draw48_at(s, ctx);
}

/// The per-lane draw for `unif_int` — `rng::scalar_ctr`: counter `(src << 36) + LANE`, all 48 bits
/// spent on this lane (Lemire needs them).
fn emit_lane_draw(s: &mut InstructionSink, ctx: &Ctx, src: u32) {
    s.local_get(LANE)
        .i64_extend_i32_u()
        .i64_const(((src as u64) << 36) as i64)
        .i64_add();
    emit_draw48_at(s, ctx);
}

/// This lane's 24 bits of the pair draw on top of the stack (`rng::lo24` / `rng::hi24`): the
/// requested [`Half`], or a parity select when the phase isn't statically known. Consumes the
/// 48-bit i64 and leaves the 24-bit i64.
fn emit_half(s: &mut InstructionSink, ctx: &Ctx, half: Half) {
    match half {
        Half::Lo => {
            s.i64_const(0xFF_FFFF).i64_and();
        }
        Half::Hi => {
            s.i64_const(24).i64_shr_u();
        }
        Half::ByParity => {
            let bits = ctx.ti; // the draw's own scratch is free once its bits are on the stack
            s.local_set(bits);
            s.local_get(bits).i64_const(24).i64_shr_u(); // hi24 (odd lane)
            s.local_get(bits).i64_const(0xFF_FFFF).i64_and(); // lo24 (even lane)
            s.local_get(LANE).i32_const(1).i32_and(); // cond: lane is odd
            s.select(); // select pops [a, b, cond] → a when cond != 0
        }
    }
}

/// The pair draw's requested half, as an i64 on the stack.
fn emit_pair_half(s: &mut InstructionSink, ctx: &Ctx, src: u32, half: Half) {
    emit_pair_draw(s, ctx, src);
    emit_half(s, ctx, half);
}

/// Uniform f32 in `[0, 1)` from the 24-bit draw on the stack — `rng::unit24` (`bits · 2⁻²⁴`, exact).
fn emit_unit24(s: &mut InstructionSink) {
    s.f32_convert_i64_u()
        .f32_const(f32c(1.0 / ((1u32 << 24) as f32)))
        .f32_mul();
}

/// `unif(lo, hi)` for one lane: `lo + (hi - lo) · u`. The constants are the f64 bounds rounded
/// ONCE, exactly as `rng::fill_uniform` rounds them.
fn emit_uniform(s: &mut InstructionSink, ctx: &Ctx, src: u32, lo: f64, hi: f64, half: Half) {
    emit_pair_half(s, ctx, src, half);
    emit_unit24(s);
    s.f32_const(f32c((hi - lo) as f32))
        .f32_mul()
        .f32_const(f32c(lo as f32))
        .f32_add();
}

/// Below this count, `unif_int` computes the exact 48-bit Lemire multiply-high (bit-identical to
/// the interpreter/JIT); at or above it — unreachable under Track B's 2^24 range cap — it falls
/// back to the float method (`lo + min(floor(u·count), count-1)`, identical in distribution).
/// The bound keeps the split multiply's `b_hi·count` term under 2^63.
const LEMIRE_MAX_COUNT: u64 = 1 << 39;

/// `unif_int(lo, hi)` via the same 48-bit Lemire multiply-high as the interpreter/JIT — the one
/// source that is NOT pair-shared (24 bits would put the bias at `count / 2^24`). Wasm has no
/// `mulhi`, so split `bits = b_hi·2^24 + b_lo` and use
/// `(bits·count) >> 48 = (b_hi·count + ((b_lo·count) >> 24)) >> 24` — exact (standard nested-floor
/// identity), and every term fits i64 for `count < 2^39`.
fn emit_uniform_int(s: &mut InstructionSink, ctx: &Ctx, src: u32, lo: f64, hi: f64) {
    let count = (hi - lo + 1.0).max(1.0);
    if (count as u64) < LEMIRE_MAX_COUNT {
        // The draw's own scratch (`ti..ti+2`) is free once the bits are on the stack.
        let bits = ctx.ti;
        emit_lane_draw(s, ctx, src);
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
            .f32_convert_i64_u()
            .f32_const(f32c(lo as f32))
            .f32_add();
    } else {
        emit_lane_draw(s, ctx, src);
        emit_unit24(s);
        s.f32_const(f32c(count as f32))
            .f32_mul()
            .f32_floor()
            .f32_const(f32c((count - 1.0) as f32))
            .f32_min()
            .f32_const(f32c(lo as f32))
            .f32_add();
    }
}

/// Horner `Σ c[i]·z^i` (coeffs low→high) with `z` already in a local — mirrors
/// `crate::approx::horner_f32` and `jit::emit_horner`. Leaves the polynomial value on the stack.
fn emit_horner(s: &mut InstructionSink, z: u32, coeffs: &[f32]) {
    s.f32_const(f32c(*coeffs.last().unwrap()));
    for &c in coeffs.iter().rev().skip(1) {
        s.local_get(z).f32_mul().f32_const(f32c(c)).f32_add();
    }
}

/// Inlined f32 `ln(x)` for `x > 0` — wasm transcription of [`crate::approx::ln_f32`] /
/// `jit::emit_ln`. Consumes the f32 input on top of the stack and leaves `ln(x)`, computing through
/// the shared scratch (so the surrounding stack is untouched).
fn emit_ln(s: &mut InstructionSink, ctx: &Ctx) {
    use crate::approx::LN_COEFFS_F32;
    use std::f32::consts::{LN_2, SQRT_2};
    let (bits, e0) = (ctx.tj, ctx.tj + 1); // i32 scratch
    let (m0, m, ef, f, f2) = (ctx.tf, ctx.tf + 1, ctx.tf + 2, ctx.tf + 3, ctx.tf + 4); // f32 scratch
                                                                                       // bits = reinterpret(x)  (consumes the input f32)
    s.i32_reinterpret_f32().local_set(bits);
    // e0 = (bits >> 23 & 0xff) - 127
    s.local_get(bits)
        .i32_const(23)
        .i32_shr_u()
        .i32_const(0xff)
        .i32_and()
        .i32_const(127)
        .i32_sub()
        .local_set(e0);
    // m0 = reinterpret((bits & MANT) | EXP_ONE)  ∈ [1, 2)
    s.local_get(bits)
        .i32_const(0x007f_ffff)
        .i32_and()
        .i32_const(0x3f80_0000)
        .i32_or()
        .f32_reinterpret_i32()
        .local_set(m0);
    // m = (m0 > √2) ? m0*0.5 : m0   (select pops a, b, cond)
    s.local_get(m0).f32_const(f32c(0.5)).f32_mul();
    s.local_get(m0);
    s.local_get(m0).f32_const(f32c(SQRT_2)).f32_gt();
    s.select().local_set(m);
    // ef = ((m0 > √2) ? e0+1 : e0) as f32
    s.local_get(e0).i32_const(1).i32_add();
    s.local_get(e0);
    s.local_get(m0).f32_const(f32c(SQRT_2)).f32_gt();
    s.select().f32_convert_i32_s().local_set(ef);
    // f = (m-1)/(m+1); f2 = f*f
    s.local_get(m).f32_const(f32c(1.0)).f32_sub();
    s.local_get(m).f32_const(f32c(1.0)).f32_add();
    s.f32_div().local_set(f);
    s.local_get(f).local_get(f).f32_mul().local_set(f2);
    // ln(x) = 2·f·Σ cₖf²ᵏ + e·ln2
    s.f32_const(f32c(2.0)).local_get(f).f32_mul();
    emit_horner(s, f2, &LN_COEFFS_F32);
    s.f32_mul();
    s.local_get(ef).f32_const(f32c(LN_2)).f32_mul();
    s.f32_add();
}

/// Call an f64 host import on an f32 value already on the stack: promote, call, demote. The whole
/// f32/f64 seam of this backend, in one place — and the exact shape `bytecode::apply_un` and the
/// JIT's shims compute, so the three agree bit-for-bit.
fn call_f64_import(s: &mut InstructionSink, func: u32) {
    s.f64_promote_f32().call(func).f32_demote_f64();
}

/// Range-guarded `cos`/`sin` — the inline polynomial ([`emit_trig_poly`]) for `|x| < TRIG_MAX_F32`,
/// else the imported library `sin`/`cos` in f64, rounded back (finding C3's f32 twin), mirroring
/// `jit::emit_trig` and [`crate::approx::sin_f32`]. Consumes the f32 input on top of the stack and
/// leaves the result. (The Box–Muller draw path calls [`emit_trig_poly`] directly — its argument is
/// always `< 2π`.)
fn emit_trig(s: &mut InstructionSink, ctx: &Ctx, is_cos: bool) {
    use crate::approx::TRIG_MAX_F32;
    let tx = ctx.tf; // stash the input (reused as the poly's accumulator in the else arm)
    s.local_set(tx);
    // if |x| >= TRIG_MAX_F32 { host sin/cos } else { inline poly }
    s.local_get(tx)
        .f32_abs()
        .f32_const(f32c(TRIG_MAX_F32))
        .f32_ge();
    s.if_(BlockType::Result(ValType::F32));
    s.local_get(tx);
    call_f64_import(s, if is_cos { COS } else { SIN });
    s.else_();
    emit_trig_poly(s, ctx, tx, is_cos);
    s.end();
}

/// The inline f32 `cos`/`sin` polynomial body operating on the input already stashed in local `tx`
/// — wasm transcription of [`crate::approx::cos_f32`]/`sin_f32`: Cody–Waite reduce to `[-π/4, π/4]`,
/// evaluate both reduced kernels, pick by quadrant. Leaves the result on the stack. (`tx` is reused
/// as the quadrant-select accumulator once the input is dead.)
fn emit_trig_poly(s: &mut InstructionSink, ctx: &Ctx, tx: u32, is_cos: bool) {
    use crate::approx::{COS_COEFFS_F32, PIO2_HI_F32, PIO2_LO_F32, SIN_COEFFS_F32};
    use std::f32::consts::FRAC_2_PI;
    let ki = ctx.tj; // i32 quadrant (the `ln` scratch is dead by here)
    let (kf, r, z, sinr, cosr) = (ctx.tf + 1, ctx.tf + 2, ctx.tf + 3, ctx.tf + 4, ctx.tf + 5);
    // kf = round(x·2/π); r = (x - kf·π/2_hi) - kf·π/2_lo
    s.local_get(tx)
        .f32_const(f32c(FRAC_2_PI))
        .f32_mul()
        .f32_nearest()
        .local_set(kf);
    s.local_get(tx)
        .local_get(kf)
        .f32_const(f32c(PIO2_HI_F32))
        .f32_mul()
        .f32_sub()
        .local_get(kf)
        .f32_const(f32c(PIO2_LO_F32))
        .f32_mul()
        .f32_sub()
        .local_set(r);
    s.local_get(r).local_get(r).f32_mul().local_set(z);
    // sin(r) = r + r·z·P_sin(z)
    s.local_get(r);
    s.local_get(r).local_get(z).f32_mul();
    emit_horner(s, z, &SIN_COEFFS_F32);
    s.f32_mul().f32_add().local_set(sinr);
    // cos(r) = 1 - z/2 + z²·P_cos(z)
    s.f32_const(f32c(1.0))
        .local_get(z)
        .f32_const(f32c(0.5))
        .f32_mul()
        .f32_sub();
    s.local_get(z).local_get(z).f32_mul();
    emit_horner(s, z, &COS_COEFFS_F32);
    s.f32_mul().f32_add().local_set(cosr);
    // kq = (kf as i32) & 3 — pick the kernel + sign per quadrant. Reuse `tx` as the accumulator.
    s.local_get(kf)
        .i32_trunc_sat_f32_s()
        .i32_const(3)
        .i32_and()
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
            s.f32_neg();
        }
    };
    push_q(s, q[0]);
    s.local_set(tx); // res = q0
    for (n, &qn) in q.iter().enumerate().skip(1) {
        push_q(s, qn); // a = qn
        s.local_get(tx); // b = current res
        s.local_get(ki).i32_const(n as i32).i32_eq(); // cond = (kq == n)
        s.select().local_set(tx);
    }
    s.local_get(tx);
}

/// The shared head of a Box–Muller pair — mirrors `rng::normal_pair` bit-for-bit: ONE pair draw,
/// `u1` from its low 24 bits, `u2` from its high 24; leaves `r = sqrt(-2·ln(1 - u1))` in the
/// `tf + 6` scratch and the angle `TAU·u2` in `ctx.tf` (ready for [`emit_trig_poly`]).
fn emit_normal_head(s: &mut InstructionSink, ctx: &Ctx, src: u32) {
    use std::f32::consts::TAU;
    // The pair draw is needed twice (both halves), so stash the 48 bits in the draw's own scratch.
    let bits = ctx.ti;
    emit_pair_draw(s, ctx, src);
    s.local_set(bits);
    // r = sqrt(-2 * ln(1 - u1)),  u1 = lo24 · 2^-24  (1 - u1 ∈ [2^-24, 1], exact — no ln(0)).
    s.f32_const(f32c(1.0));
    s.local_get(bits).i64_const(0xFF_FFFF).i64_and();
    emit_unit24(s);
    s.f32_sub();
    emit_ln(s, ctx);
    s.f32_const(f32c(-2.0))
        .f32_mul()
        .f32_sqrt()
        .local_set(BM_R(ctx));
    // Angle TAU * u2 ∈ [0, 2π) — u2 = hi24 · 2^-24. Banked in the ANG local (`emit_trig_poly`
    // consumes its input local, and the pair needs the angle twice). Always inside the
    // polynomial's range, so callers use `emit_trig_poly` directly.
    s.local_get(bits).i64_const(24).i64_shr_u();
    emit_unit24(s);
    s.f32_const(f32c(TAU)).f32_mul().local_set(BM_ANG(ctx));
    s.local_get(BM_ANG(ctx)).local_set(ctx.tf);
}

/// Re-stage the banked Box–Muller angle into the trig input local (`emit_trig_poly`
/// consumes its input as the quadrant accumulator).
fn emit_restage_angle(s: &mut InstructionSink, ctx: &Ctx) {
    s.local_get(BM_ANG(ctx)).local_set(ctx.tf);
}

/// `mu + sigma * (r * <branch on stack>)` — the tail both branches share.
fn emit_normal_tail(s: &mut InstructionSink, ctx: &Ctx, mu: f64, sigma: f64) {
    s.local_get(BM_R(ctx))
        .f32_mul()
        .f32_const(f32c(sigma as f32))
        .f32_mul()
        .f32_const(f32c(mu as f32))
        .f32_add();
}

/// The Box–Muller radius scratch local (survives both trig evaluations).
#[allow(non_snake_case)]
fn BM_R(ctx: &Ctx) -> u32 {
    ctx.tf + 6
}

/// The Box–Muller angle scratch local (same lifetime story as the radius).
#[allow(non_snake_case)]
fn BM_ANG(ctx: &Ctx) -> u32 {
    ctx.tf + 7
}

/// `N(mu, sigma^2)` for the single-lane loop (cones too big to pair-unroll): hash the pair,
/// compute both branches, select by lane parity — the same values the unrolled loop computes, at
/// twice the per-lane transcendental cost.
fn emit_normal_single(s: &mut InstructionSink, ctx: &Ctx, src: u32, mu: f64, sigma: f64) {
    emit_normal_head(s, ctx, src);
    // wasm select pops [t1, t2, cond] and picks t1 when cond != 0: push cos (even) first.
    emit_trig_poly(s, ctx, ctx.tf, true); // cos branch (even lanes)
    emit_restage_angle(s, ctx);
    emit_trig_poly(s, ctx, ctx.tf, false); // sin branch (odd lanes)
    s.local_get(LANE).i32_const(1).i32_and().i32_eqz();
    s.select(); // (lane even) ? cos : sin
    emit_normal_tail(s, ctx, mu, sigma);
}

/// `N(mu, sigma^2)` for the pair-unrolled loop's EVEN phase: hash the pair once, bank the odd
/// lane's finished value (`mu + sigma·r·sin θ`) into `bank_local`, and leave the even lane's
/// (`… cos θ`) on the stack.
fn emit_normal_pair(
    s: &mut InstructionSink,
    ctx: &Ctx,
    src: u32,
    mu: f64,
    sigma: f64,
    bank_local: u32,
) {
    emit_normal_head(s, ctx, src);
    emit_trig_poly(s, ctx, ctx.tf, false); // sin branch → the odd lane's value, banked
    emit_normal_tail(s, ctx, mu, sigma);
    s.local_set(bank_local);
    emit_restage_angle(s, ctx);
    emit_trig_poly(s, ctx, ctx.tf, true); // cos branch → the even lane's value, left on stack
    emit_normal_tail(s, ctx, mu, sigma);
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `rng::fill_exp` bit-for-bit
/// (`1 - u ∈ [2⁻²⁴, 1]`, exact on the f32 uniform grid).
fn emit_exp(s: &mut InstructionSink, ctx: &Ctx, src: u32, rate: f64, half: Half) {
    s.f32_const(f32c(1.0));
    emit_pair_half(s, ctx, src, half);
    emit_unit24(s);
    s.f32_sub();
    emit_ln(s, ctx);
    s.f32_neg().f32_const(f32c(rate as f32)).f32_div();
}

/// `Geometric(p)` via `floor(ln(1 - u) / ln(1 - p))` — mirrors `rng::fill_geometric` bit-for-bit.
/// `ln(1 - p)` is a compile-time constant (the f64 `ln` rounded to f32, as the interpreter rounds
/// it); `p == 1` makes it `-inf`, so every draw floors to `0` (the point mass).
fn emit_geometric(s: &mut InstructionSink, ctx: &Ctx, src: u32, p: f64, half: Half) {
    let denom = (1.0 - p).ln() as f32;
    s.f32_const(f32c(1.0));
    emit_pair_half(s, ctx, src, half);
    emit_unit24(s);
    s.f32_sub();
    emit_ln(s, ctx);
    s.f32_const(f32c(denom)).f32_div().f32_floor();
}

/// `base ^ k` for a small non-negative integer `k`, as repeated multiply (`k == 0` → `1.0`).
fn emit_pow(s: &mut InstructionSink, base: u32, k: u32) {
    if k == 0 {
        s.f32_const(f32c(1.0));
        return;
    }
    s.local_get(base);
    for _ in 1..k {
        s.local_get(base).f32_mul();
    }
}

fn emit_unary(s: &mut InstructionSink, ctx: &Ctx, op: UnOp, a: u32) {
    match op {
        UnOp::Neg => {
            s.local_get(a).f32_neg();
        }
        UnOp::Not => {
            // logical not of a 0/1 value: (a == 0) ? 1 : 0
            s.local_get(a)
                .f32_const(f32c(0.0))
                .f32_eq()
                .f32_convert_i32_u();
        }
        UnOp::Sin => {
            s.local_get(a);
            emit_trig(s, ctx, false);
        }
        UnOp::Cos => {
            s.local_get(a);
            emit_trig(s, ctx, true);
        }
        // `atan`/`round`/`exp` have no pinnable f32 form, so all three backends compute them in
        // f64 and round: here that is promote → `Math.*` → demote (see `call_f64_import`).
        UnOp::Atan => {
            s.local_get(a);
            call_f64_import(s, ATAN);
        }
        UnOp::Round => {
            s.local_get(a);
            call_f64_import(s, ROUND);
        }
        UnOp::Floor => {
            s.local_get(a).f32_floor();
        }
        UnOp::Ceil => {
            s.local_get(a).f32_ceil();
        }
        UnOp::Sqrt => {
            // Imported `Math.sqrt` (f64), NOT the inline `f32.sqrt` instruction: V8/arm64 regresses
            // ~30% on large single-block kernel bodies when these become inline sqrt (measured
            // 2026-07-14, `am_vs_fm` +21% run-level — the import calls serve as live-range split
            // points for V8's regalloc). JSC prefers inline; revisit if V8 improves. Promoting to
            // f64, taking an IEEE-exact sqrt and demoting is *bit-identical* to a correctly-rounded
            // f32 sqrt (f64 carries far more than 2·24+2 bits, so the double rounding is benign) —
            // so this matches the interpreter's `f32::sqrt` and the JIT's inline one exactly. Never
            // `pow(x, 0.5)`, which disagrees at -0.0 and -inf (PLAN-PERF-2 §5).
            s.local_get(a);
            call_f64_import(s, SQRT);
        }
        UnOp::Ln => {
            use crate::approx::{LN_SUBNORMAL_CORR_F32, LN_SUBNORMAL_SCALE_F32};
            // Full-domain ln: the inlined poly (positive finite inputs only) behind guards that
            // match `f32::ln` — x > 0 → poly, 0 → -inf, negative/NaN → NaN, +inf → +inf (the
            // poly's exponent bit-surgery would misread ±inf/NaN/negatives). Mirrors
            // `jit::emit_ln_guarded`.
            //
            // Subnormal positive inputs are first scaled into the normal range (their zero exponent
            // field would corrupt the mantissa bit-surgery) and corrected by `25·ln2`:
            //   a_in = (a < MIN_POSITIVE) ? a * SCALE : a
            s.local_get(a)
                .f32_const(f32c(LN_SUBNORMAL_SCALE_F32))
                .f32_mul();
            s.local_get(a);
            s.local_get(a).f32_const(f32c(f32::MIN_POSITIVE)).f32_lt();
            s.select(); // a_in
            emit_ln(s, ctx); // poly_raw (consumes a_in)
            s.local_set(ctx.tf); // stash poly_raw (transcendental scratch is dead after emit_ln)
                                 //   poly = (a < MIN_POSITIVE) ? poly_raw - 25·ln2 : poly_raw
            s.local_get(ctx.tf)
                .f32_const(f32c(LN_SUBNORMAL_CORR_F32))
                .f32_sub();
            s.local_get(ctx.tf);
            s.local_get(a).f32_const(f32c(f32::MIN_POSITIVE)).f32_lt();
            s.select(); // stack: poly
                        // non_pos = (a == 0) ? -inf : NaN
            s.f32_const(f32c(f32::NEG_INFINITY))
                .f32_const(f32c(f32::NAN));
            s.local_get(a).f32_const(f32c(0.0)).f32_eq();
            s.select(); // stack: poly, non_pos
                        // r = (a > 0) ? poly : non_pos
            s.local_get(a).f32_const(f32c(0.0)).f32_gt();
            s.select(); // stack: r
                        // (a != +inf) ? r : +inf — patch the poly's mangled +inf back.
            s.f32_const(f32c(f32::INFINITY));
            s.local_get(a).f32_const(f32c(f32::INFINITY)).f32_ne();
            s.select();
        }
        UnOp::Exp => {
            s.local_get(a);
            call_f64_import(s, EXP);
        }
        UnOp::Sign => {
            // (a > 0) - (a < 0): -1 / 0 / +1, exactly 0 at 0 (matches `apply_un`, unlike signum).
            s.local_get(a)
                .f32_const(f32c(0.0))
                .f32_gt()
                .f32_convert_i32_u();
            s.local_get(a)
                .f32_const(f32c(0.0))
                .f32_lt()
                .f32_convert_i32_u();
            s.f32_sub();
        }
    }
}

fn emit_binary(s: &mut InstructionSink, op: BinOp, a: u32, b: u32) {
    // `&&`/`||` use the interpreter/JIT's `(a != 0) & (b != 0)` semantics — NOT `f32.min`/`max`,
    // which return NaN if either operand is NaN whereas `(NaN != 0)` is `true` (finding C8). Both
    // operands are 0/1 events and the result is 0/1, so this is exact and branch-free.
    if matches!(op, BinOp::And | BinOp::Or) {
        s.local_get(a).f32_const(f32c(0.0)).f32_ne(); // (a != 0) : i32
        s.local_get(b).f32_const(f32c(0.0)).f32_ne(); // (b != 0) : i32
        if matches!(op, BinOp::And) {
            s.i32_and();
        } else {
            s.i32_or();
        }
        s.f32_convert_i32_u();
        return;
    }
    // Floored modulo needs both operands twice (`a − b·floor(a/b)`), so it builds its own stack.
    if matches!(op, BinOp::Mod) {
        s.local_get(a);
        s.local_get(b);
        s.local_get(a).local_get(b).f32_div().f32_floor();
        s.f32_mul();
        s.f32_sub();
        return;
    }
    s.local_get(a).local_get(b);
    match op {
        BinOp::Add => {
            s.f32_add();
        }
        BinOp::Sub => {
            s.f32_sub();
        }
        BinOp::Mul => {
            s.f32_mul();
        }
        BinOp::Div => {
            s.f32_div();
        }
        BinOp::Lt => {
            s.f32_lt().f32_convert_i32_u();
        }
        BinOp::Gt => {
            s.f32_gt().f32_convert_i32_u();
        }
        BinOp::Le => {
            s.f32_le().f32_convert_i32_u();
        }
        BinOp::Ge => {
            s.f32_ge().f32_convert_i32_u();
        }
        BinOp::Eq => {
            s.f32_eq().f32_convert_i32_u();
        }
        BinOp::Ne => {
            s.f32_ne().f32_convert_i32_u();
        }
        BinOp::And | BinOp::Or => {
            unreachable!("And/Or are handled before the generic two-operand path")
        }
        BinOp::Mod => unreachable!("Mod is handled before the generic two-operand path"),
        BinOp::Pow => unreachable!("Pow is handled before emit_binary"),
    }
}

/// Non-integer `^`: the f64 `Math.pow` import over two promoted operands, demoted back — the same
/// promote/call/demote contract as the unary shims (`num::fold_binop_f32`'s `Pow` arm).
fn emit_pow_call(s: &mut InstructionSink, a: u32, b: u32) {
    s.local_get(a).f64_promote_f32();
    s.local_get(b).f64_promote_f32();
    s.call(POW).f32_demote_f64();
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
    fn first_emitted(bytes: &[u8], seed: u64) -> f32 {
        first_batch_emitted(bytes, seed)[0]
    }

    /// Full first batch of an emitted kernel (lane 0), for whole-column bitwise parity checks —
    /// the even/odd lanes take different paths in the pair-unrolled loop, so `out[0]` alone
    /// would only ever exercise the cos branch.
    fn first_batch_emitted(bytes: &[u8], seed: u64) -> Vec<f32> {
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
        let mut b = vec![0u8; BATCH * 4];
        memory.read(&store, out_ptr as usize, &mut b).unwrap();
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
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
                "{label}: interp {interp} ({:#010x}) != wasm {wasm} ({:#010x})",
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
        let mut out_bytes = vec![0u8; cap * 4];
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
            for chunk in out_bytes.chunks_exact(4) {
                sum += f32::from_le_bytes(chunk.try_into().unwrap()) as f64;
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

        // Bitwise: the whole first batch against the interpreter oracle (both branches of the
        // pair-unrolled loop get exercised, not just lane 0's cos arm).
        let mut ir = InterpBackend.compile(graph, id, ENOUGH_DRAWS).runner();
        ir.position(seed, 0);
        let interp_col: Vec<f32> = ir.next_batch(BATCH).to_vec();
        let wasm_col = first_batch_emitted(&bytes, seed);
        for (lane, (a, b)) in interp_col.iter().zip(wasm_col.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{src}: lane {lane}: interp {a} ({:#010x}) != wasm {b} ({:#010x})",
                a.to_bits(),
                b.to_bits()
            );
        }

        let (wasm_mean, count) = run_emitted(&bytes, seed, 16);
        let interp_mean = moments(graph, id, count as usize, seed).unwrap().mean;
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
            let interp = moments(eng.graph(), id, count as usize, 0xABCDEF).unwrap().mean;
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
        let mut out_bytes = vec![0u8; k * BATCH * 4];
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
                for chunk in out_bytes[j * BATCH * 4..(j + 1) * BATCH * 4].chunks_exact(4) {
                    col.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
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
            // The pairing identity holds in the LANE type: `z + 1` is one f32 add (Track B), so
            // comparing against an f64 add of the widened draw would be off by a rounding step.
            assert_eq!(
                cols[1][i] as f32,
                cols[0][i] as f32 + 1.0f32,
                "lane {i} must pair"
            );
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
