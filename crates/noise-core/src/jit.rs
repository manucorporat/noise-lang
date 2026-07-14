//! Cranelift native JIT backend (PLAN.md Phase 4; counter RNG per PLAN-PREGPU Track C).
//!
//! Compiles the sample-DAG into ONE fused machine-code kernel: a loop that draws its sources,
//! computes the whole expression keeping intermediates in registers, and stores the root `f64`s —
//! so, unlike the columnar interpreter, **no intermediate column is materialized to memory**. That
//! fusion is the win on arithmetic-dense graphs (see the `poly_*` benches).
//!
//! **Counter-based RNG.** Every draw is a pure hash of `(key, global_lane, source_offset)`
//! (pcg4d-3r — see `rng.rs` and tools/rng-cert), inlined as straight u32 arithmetic; there is no
//! RNG state chain, so the old multi-stream interleave (built to overlap xoshiro's serial state
//! latency) is gone — every loop iteration is independent and the out-of-order core overlaps them
//! on its own. Because the interpreter keys draws identically, the JIT and interpreter agree
//! **bit-for-bit** under a shared seed (the PLAN-PREGPU draw-stream-parity contract), not merely
//! in distribution.
//!
//! Scope: `unif` / `unif_int` / `normal` / `exp` / `geometric` sources, `+ - * /`, integer-constant
//! `^`, comparisons, `&& ||`, unary `- !` and the math ufuncs, lifted `if` (`Select`), and `Gather`
//! over a const table (an indexed load from a program-owned table — the `empirical`/bootstrap
//! shape) or a small non-const one (a compare/select chain; see [`crate::kernel::gather_class`]).
//! `Poisson` (Knuth's variable-length per-lane loop) and large non-const gathers stay
//! interpreter-only; transcendental-bound graphs that the interpreter samples faster also stay
//! there (see [`crate::kernel::profitable`]).

use std::collections::HashMap;
use std::sync::Arc;

use cranelift::codegen::ir::{BlockArg, UserFuncName};
use cranelift::prelude::settings::Configurable;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use crate::ast::{BinOp, UnOp};
use crate::backend::{Backend, InterpBackend, JointProgram, JointRunner, Program, Runner};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId, RvNode, Source};
use crate::kernel::{self, const_int_exponent, gather_class, profitable, GatherClass};
use crate::rng::Key;

/// `extern "C"` signature of a generated kernel: `kernel(out_ptr, n, key_lo, key_hi, lane0)` fills
/// `out[0..n]` with the root draws for global lanes `lane0 .. lane0 + n` (key words and lane index
/// are the low 32 bits of their `i64` arguments). Stateless: the same arguments always produce the
/// same column.
type KernelFn = unsafe extern "C" fn(*mut f64, i64, i64, i64, i64);

// Scalar math the kernel can't express as a single CLIF instruction is delegated to these
// `extern "C"` shims (registered as JIT symbols, called per draw). `sqrt`/`floor` are NOT here —
// they are native CLIF instructions on the targets we run, and neither are `ln`/`sin`/`cos`, which
// are inlined as polynomials (see `emit_ln`/`emit_trig` and [`crate::approx`]). Names are prefixed
// `nz_` to avoid any clash with libm symbols the module might resolve itself.
extern "C" fn nz_atan(x: f64) -> f64 {
    x.atan()
}
extern "C" fn nz_round(x: f64) -> f64 {
    x.round()
}
extern "C" fn nz_pow(a: f64, b: f64) -> f64 {
    a.powf(b)
}
// Accurate library `sin`/`cos` for the large-argument fallback: the inlined polynomial degrades
// past `approx::TRIG_MAX`, so `emit_trig` calls these there to stay in agreement with the
// interpreter's libm across the whole range (finding C3). Rare path — the compare almost always
// picks the inline poly, so the shim is only a defended edge, not a hot cost.
extern "C" fn nz_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn nz_cos(x: f64) -> f64 {
    x.cos()
}
// `e^x` via the exact library `exp`, so the JIT matches the interpreter's `f64::exp` bit-for-bit
// (finding C9 — the old `pow(e, x)` lowering could differ in the last bit from `x.exp()`).
extern "C" fn nz_exp(x: f64) -> f64 {
    x.exp()
}

/// Module-level `FuncId`s for the math shims (declared once per build).
struct MathIds {
    atan: cranelift_module::FuncId,
    round: cranelift_module::FuncId,
    pow: cranelift_module::FuncId,
    sin: cranelift_module::FuncId,
    cos: cranelift_module::FuncId,
    exp: cranelift_module::FuncId,
}

/// In-function `FuncRef`s for the math shims (resolved once at the top of the kernel body, then
/// reused at every call site). `Copy`, so threading it through emission is free.
#[derive(Clone, Copy)]
struct MathRefs {
    atan: cranelift::codegen::ir::FuncRef,
    round: cranelift::codegen::ir::FuncRef,
    pow: cranelift::codegen::ir::FuncRef,
    sin: cranelift::codegen::ir::FuncRef,
    cos: cranelift::codegen::ir::FuncRef,
    exp: cranelift::codegen::ir::FuncRef,
}

/// Const gather tables materialized during emission ([`GatherClass::ConstTable`]). Each table is a
/// `Box<[f64]>` — a stable heap address the emitted code references as an `iconst` — deduped by
/// `RvId` so a gather shared across streams/roots (per-stream memos don't cover it) is built once.
/// The boxes land in [`JitProgramInner::_tables`], which pins their lifetime to the code's.
struct GatherTables {
    /// Native pointer type, for emitting the base-address `iconst`.
    ptr_ty: Type,
    boxes: Vec<Box<[f64]>>,
    by_node: HashMap<RvId, usize>,
}

impl GatherTables {
    fn new(ptr_ty: Type) -> Self {
        GatherTables {
            ptr_ty,
            boxes: Vec::new(),
            by_node: HashMap::new(),
        }
    }

    /// Base pointer of node `id`'s table, materializing it from the `ConstNum` elems on first use.
    fn base_ptr(&mut self, graph: &RvGraph, id: RvId, elems: &[RvId]) -> *const f64 {
        let slot = *self.by_node.entry(id).or_insert_with(|| {
            let table: Box<[f64]> = elems
                .iter()
                .map(|&e| match graph.node(e) {
                    RvNode::ConstNum(x) => *x,
                    _ => unreachable!("ConstTable gather has only ConstNum elems"),
                })
                .collect();
            self.boxes.push(table);
            self.boxes.len() - 1
        });
        self.boxes[slot].as_ptr()
    }
}

/// The Cranelift JIT backend. Construction is cheap; the work happens in [`Self::compile`].
#[derive(Default)]
pub struct JitBackend;

impl JitBackend {
    pub fn new() -> Self {
        JitBackend
    }
}

impl Backend for JitBackend {
    fn compile(&self, graph: &RvGraph, root: RvId, draws: usize) -> Box<dyn Program> {
        // Use the JIT only where it's expected to *win* — see `kernel::profitable`. We inline the
        // transcendentals (`emit_ln`/`emit_trig`), so `inline_trans = true`: `normal`/`exp`/trig
        // graphs are fusible here and worth compiling. Only graphs still dominated by a real call
        // (`atan`/`round`/non-integer `pow`) stay on the interpreter. Any codegen failure also falls
        // back. Either way correctness is never at risk, only the speedup.
        if !profitable(
            graph,
            root,
            /* inline_trans */ true,
            draws,
            kernel::MIN_DRAWS_JIT,
        ) {
            return InterpBackend.compile(graph, root, draws);
        }
        match build(graph, root) {
            Ok(program) => Box::new(program),
            Err(_) => InterpBackend.compile(graph, root, draws),
        }
    }

    /// The joint (multi-root) path: one fused kernel computing every root per lane from a *shared*
    /// memo — shared sources are drawn once per lane, so the roots stay jointly sampled exactly as
    /// the multi-root interpreter samples them. Gated like [`Self::compile`], on the union cone.
    fn compile_joint(
        &self,
        graph: &RvGraph,
        roots: &[RvId],
        draws: usize,
    ) -> Box<dyn JointProgram> {
        if !kernel::profitable_roots(
            graph,
            roots,
            /* inline_trans */ true,
            draws,
            kernel::MIN_DRAWS_JIT,
        ) {
            return InterpBackend.compile_joint(graph, roots, draws);
        }
        match build_joint(graph, roots) {
            Ok(program) => Box::new(program),
            Err(_) => InterpBackend.compile_joint(graph, roots, draws),
        }
    }
}

/// The immutable JIT artifact: the finalized module (owns the executable memory) and the entry
/// pointer. Shared behind an `Arc`.
struct JitProgramInner {
    // `func` points into `_module`'s code memory; the module is kept alive for the program's life.
    //
    // We deliberately do NOT call `JITModule::free_memory()` on drop. Doing so was a
    // **use-after-free**: it unmapped this module's executable pages, and those pages were then
    // recycled by the *next* module's mmap while other threads were still executing kernels — the
    // corruption showed up as wrong values and NaNs (`E[cos X]` came back 0.495 instead of 0.607),
    // non-deterministically, only under CPU load, and only on the JIT. Rust's lifetimes never
    // caught it, and they never could: they prove no runner of the *freed* module survives, which
    // is true and beside the point. `free_memory`'s safety contract is about the whole process's
    // JIT code, not one module's borrows.
    //
    // Repro (fails 25/25 with the free, 0/12 without it) — the corruption needs a process compiling
    // and dropping many kernels while others execute, plus memory pressure to get the pages recycled:
    //     cargo test --release -p noise-core --features jit --test signals   # under CPU load
    // Narrower attempts (a churn thread + a reader thread, or 8 threads each compiling and running)
    // do NOT reproduce it, which is exactly what a use-after-free looks like: whether the stale page
    // has been handed out again yet is an allocator/timing accident, not a property of the program.
    // Hence no unit-level regression test — an unreliable one would imply coverage that isn't there.
    //
    // So the module simply lives as long as the program does, and cranelift-jit's own
    // `Memory::drop` (which `mem::forget`s the executable pages) leaks them. That reverses finding
    // C4: a long-lived REPL/server now retains a few KB per *distinct* compiled kernel rather than
    // reclaiming it. That is the right trade — a bounded leak beats silently wrong answers — but it
    // is a real regression in memory behavior, so the fix is to stop churning modules (cache
    // compiled programs per Engine) rather than to reinstate the free.
    _module: JITModule,
    func: KernelFn,
    /// Const gather tables the kernel loads from ([`GatherClass::ConstTable`]): each `Box<[f64]>`
    /// is a stable heap allocation whose address was baked into the code as an `iconst`, so the
    /// program must own them exactly as long as the code (moving the `Vec` moves only the box
    /// pointers, never the table storage). Read-only after build — see the Send/Sync note below.
    _tables: Vec<Box<[f64]>>,
}

// SAFETY: after `finalize_definitions`, the module's code is immutable and we never touch the
// module again (only keep it mapped, then free it exactly once on `Drop`). The kernel has NO global
// mutable state — its RNG state and output buffer are passed in per call, and the gather tables
// (`_tables`) are only ever *read* by the code after build — so concurrent calls from multiple
// threads with distinct arguments are data-race-free. Hence the artifact is safe to send and share
// between threads.
unsafe impl Send for JitProgramInner {}
unsafe impl Sync for JitProgramInner {}

/// A compiled JIT program (one `Arc`-shared kernel, spun up into cheap per-worker runners).
struct JitProgram {
    inner: Arc<JitProgramInner>,
}

impl Program for JitProgram {
    fn runner(&self) -> Box<dyn Runner> {
        Box::new(JitRunner {
            inner: self.inner.clone(),
            buf: vec![0.0; BATCH],
            key: Key::from_seed(0),
            lane: 0,
        })
    }
}

/// A per-worker JIT runner: a clone of the shared kernel `Arc`, its own output buffer, and the
/// draw key + lane cursor (the kernel itself is stateless).
struct JitRunner {
    inner: Arc<JitProgramInner>,
    buf: Vec<f64>,
    key: Key,
    lane: u32,
}

impl Runner for JitRunner {
    fn position(&mut self, seed: u64, lane: u32) {
        self.key = Key::from_seed(seed);
        self.lane = lane;
    }

    fn next_batch(&mut self, len: usize) -> &[f64] {
        // Always fill the full BATCH (constant lane consumption per call), then slice to `len`.
        let n = self.buf.len() as i64;
        // SAFETY: `func` is a finalized kernel with this exact ABI; `buf` holds `n` f64s, valid
        // for the duration of the call.
        unsafe {
            (self.inner.func)(
                self.buf.as_mut_ptr(),
                n,
                self.key.k0 as i64,
                self.key.k1 as i64,
                self.lane as i64,
            );
        }
        self.lane = self.lane.wrapping_add(BATCH as u32);
        &self.buf[..len]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}

/// A compiled multi-root JIT program: the same `Arc`-shared kernel artifact, but the kernel fills
/// one BATCH-strided column per root (column `r` at `out[r*BATCH ..]`), all drawn jointly per lane.
struct JitJointProgram {
    inner: Arc<JitProgramInner>,
    /// Number of roots (output columns) the kernel writes.
    k: usize,
}

impl JointProgram for JitJointProgram {
    fn runner(&self) -> Box<dyn JointRunner> {
        Box::new(JitJointRunner {
            inner: self.inner.clone(),
            k: self.k,
            buf: vec![0.0; self.k * BATCH],
            key: Key::from_seed(0),
            lane: 0,
        })
    }
}

/// Per-worker joint runner: one flat `k×BATCH` column buffer plus the draw key + lane cursor.
struct JitJointRunner {
    inner: Arc<JitProgramInner>,
    k: usize,
    buf: Vec<f64>,
    key: Key,
    lane: u32,
}

impl JointRunner for JitJointRunner {
    fn position(&mut self, seed: u64, lane: u32) {
        self.key = Key::from_seed(seed);
        self.lane = lane;
    }

    fn next_batch(&mut self) {
        debug_assert_eq!(self.buf.len(), self.k * BATCH);
        // SAFETY: `func` is a finalized kernel with this exact ABI; `buf` holds `k * BATCH` f64s
        // (the kernel writes lanes `0..BATCH` of each of the `k` BATCH-strided columns), valid for
        // the duration of the call.
        unsafe {
            (self.inner.func)(
                self.buf.as_mut_ptr(),
                BATCH as i64,
                self.key.k0 as i64,
                self.key.k1 as i64,
                self.lane as i64,
            );
        }
        self.lane = self.lane.wrapping_add(BATCH as u32);
    }

    fn col(&self, j: usize) -> &[f64] {
        &self.buf[j * BATCH..(j + 1) * BATCH]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}

/// Build the JIT module + kernel for `root` (the caller falls back to the interpreter on error).
fn build(graph: &RvGraph, root: RvId) -> Result<JitProgram, String> {
    Ok(JitProgram {
        inner: Arc::new(build_kernel(graph, &[root])?),
    })
}

/// Multi-root [`build_kernel`], wrapped as a [`JointProgram`].
fn build_joint(graph: &RvGraph, roots: &[RvId]) -> Result<JitJointProgram, String> {
    Ok(JitJointProgram {
        inner: Arc::new(build_kernel(graph, roots)?),
        k: roots.len(),
    })
}

/// Build the stateless counter-keyed kernel. The skeleton is: reduce the key/lane arguments to
/// i32 → counted loop emitting one lane per iteration (each draw a pure hash of
/// `(key, lane, source)`, so iterations are independent and the OoO core overlaps them) → return.
///
/// Multi-root: each lane evaluates **every** root from one shared memo — a source feeding two
/// roots is drawn once, so the roots are sampled *jointly* — and root `r`'s value is stored into
/// its own BATCH-strided column at `out[r*BATCH + i]`. A single root (`roots == &[root]`) emits
/// exactly the plain single-column kernel.
fn build_kernel(graph: &RvGraph, roots: &[RvId]) -> Result<JitProgramInner, String> {
    // --- ISA + module setup ---
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    flags
        .set("use_colocated_libcalls", "false")
        .map_err(|e| e.to_string())?;
    flags.set("is_pic", "false").map_err(|e| e.to_string())?;
    let isa = cranelift_native::builder()
        .map_err(|e| e.to_string())?
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())?;
    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    // Bind the math shim names to their Rust function pointers so the JIT can resolve the calls.
    builder.symbol("nz_atan", nz_atan as *const u8);
    builder.symbol("nz_round", nz_round as *const u8);
    builder.symbol("nz_pow", nz_pow as *const u8);
    builder.symbol("nz_sin", nz_sin as *const u8);
    builder.symbol("nz_cos", nz_cos as *const u8);
    builder.symbol("nz_exp", nz_exp as *const u8);
    let mut module = JITModule::new(builder);

    // Declare the math shims as imports (f64->f64, except `pow` which is f64,f64->f64).
    let math_ids = declare_math(&mut module)?;

    // kernel(out: *mut f64, n: i64, key_lo: i64, key_hi: i64, lane0: i64)
    let ptr = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // out
    sig.params.push(AbiParam::new(types::I64)); // n
    sig.params.push(AbiParam::new(types::I64)); // key_lo (low 32 bits significant)
    sig.params.push(AbiParam::new(types::I64)); // key_hi
    sig.params.push(AbiParam::new(types::I64)); // lane0
    let func_id = module
        .declare_function("kernel", Linkage::Export, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, func_id.as_u32());

    // --- function body ---
    // Const gather tables are allocated as emission encounters them; the boxes move into the
    // finished program below, which keeps every baked-in `iconst` base address alive.
    let mut tables = GatherTables::new(ptr);
    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        // Resolve the math shims into this function once; `MathRefs` is Copy, reused at each call.
        let math = MathRefs {
            atan: module.declare_func_in_func(math_ids.atan, fb.func),
            round: module.declare_func_in_func(math_ids.round, fb.func),
            pow: module.declare_func_in_func(math_ids.pow, fb.func),
            sin: module.declare_func_in_func(math_ids.sin, fb.func),
            cos: module.declare_func_in_func(math_ids.cos, fb.func),
            exp: module.declare_func_in_func(math_ids.exp, fb.func),
        };

        let entry = fb.create_block();
        let header = fb.create_block();
        let body = fb.create_block();
        let exit = fb.create_block();

        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);
        let out = fb.block_params(entry)[0];
        let n = fb.block_params(entry)[1];
        // Key words + starting lane arrive as i64; the hash world is pure i32.
        let k0_64 = fb.block_params(entry)[2];
        let k1_64 = fb.block_params(entry)[3];
        let lane0_64 = fb.block_params(entry)[4];
        let k0 = fb.ins().ireduce(types::I32, k0_64);
        let k1 = fb.ins().ireduce(types::I32, k1_64);
        let lane0 = fb.ins().ireduce(types::I32, lane0_64);

        // Loop counter `i` (i64 sample index) and the current global lane (i32, wraps with the
        // 2^32-lane forcing cap).
        let i_var = fb.declare_var(types::I64);
        let lane_var = fb.declare_var(types::I32);
        let zero_i = fb.ins().iconst(types::I64, 0);
        fb.def_var(i_var, zero_i);
        fb.def_var(lane_var, lane0);
        fb.ins().jump(header, &[]);

        // header: if i < n goto body else exit  (left unsealed — body adds the back-edge)
        fb.switch_to_block(header);
        let iv = fb.use_var(i_var);
        let cond = fb.ins().icmp(IntCC::SignedLessThan, iv, n);
        fb.ins().brif(cond, body, &[], exit, &[]);

        // body: emit the fused DAG for this lane (one shared memo across roots — joint draws) and
        // store root `r` at out[r*BATCH + i] (the column stride is a constant store offset); then
        // i += 1, lane += 1, loop.
        fb.switch_to_block(body);
        fb.seal_block(body);
        let iv = fb.use_var(i_var);
        let lane = fb.use_var(lane_var);
        let ctx = DrawCtx { k0, k1, lane };
        let mut memo: HashMap<RvId, Value> = HashMap::new();
        let off = fb.ins().imul_imm(iv, 8);
        let addr = fb.ins().iadd(out, off);
        for (r, &root) in roots.iter().enumerate() {
            let result = emit_node(&mut fb, graph, root, ctx, &math, &mut memo, &mut tables);
            fb.ins()
                .store(MemFlags::trusted(), result, addr, (r * BATCH * 8) as i32);
        }
        let inext = fb.ins().iadd_imm(iv, 1);
        fb.def_var(i_var, inext);
        let lnext = fb.ins().iadd_imm(lane, 1);
        fb.def_var(lane_var, lnext);
        fb.ins().jump(header, &[]);
        fb.seal_block(header); // both preds (entry, body) now known

        // exit: nothing to persist — the kernel is stateless.
        fb.switch_to_block(exit);
        fb.seal_block(exit);
        fb.ins().return_(&[]);
        fb.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(func_id);
    // SAFETY: `code` is a finalized function with exactly the `KernelFn` ABI declared above; the
    // module is moved into the program so the code stays mapped for the pointer's lifetime.
    let func: KernelFn = unsafe { std::mem::transmute::<*const u8, KernelFn>(code) };

    Ok(JitProgramInner {
        _module: module,
        func,
        _tables: tables.boxes,
    })
}

/// Declare the math shims as module imports and return their `FuncId`s. (Errors are stringified —
/// `ModuleError` is large, and the caller only ever falls back on failure.)
fn declare_math(module: &mut JITModule) -> Result<MathIds, String> {
    let sig1 = {
        let mut s = module.make_signature();
        s.params.push(AbiParam::new(types::F64));
        s.returns.push(AbiParam::new(types::F64));
        s
    };
    let sig2 = {
        let mut s = module.make_signature();
        s.params.push(AbiParam::new(types::F64));
        s.params.push(AbiParam::new(types::F64));
        s.returns.push(AbiParam::new(types::F64));
        s
    };
    let decl = |module: &mut JITModule, name: &str, sig: &cranelift::codegen::ir::Signature| {
        module
            .declare_function(name, Linkage::Import, sig)
            .map_err(|e| e.to_string())
    };
    Ok(MathIds {
        atan: decl(module, "nz_atan", &sig1)?,
        round: decl(module, "nz_round", &sig1)?,
        pow: decl(module, "nz_pow", &sig2)?,
        sin: decl(module, "nz_sin", &sig1)?,
        cos: decl(module, "nz_cos", &sig1)?,
        exp: decl(module, "nz_exp", &sig1)?,
    })
}

/// `f(x)` for a one-arg shim `FuncRef`.
fn call1(fb: &mut FunctionBuilder, f: cranelift::codegen::ir::FuncRef, x: Value) -> Value {
    let c = fb.ins().call(f, &[x]);
    fb.inst_results(c)[0]
}

/// `f(x, y)` for a two-arg shim `FuncRef`.
fn call2(
    fb: &mut FunctionBuilder,
    f: cranelift::codegen::ir::FuncRef,
    x: Value,
    y: Value,
) -> Value {
    let c = fb.ins().call(f, &[x, y]);
    fb.inst_results(c)[0]
}

// --- DAG → CLIF emission (per draw, in the loop body, for one lane) ---

/// Per-iteration draw context: the run's two key words and this lane's global index, all `I32`
/// SSA values defined in the entry/body blocks. Every source hash takes these plus its own
/// compile-time `source_offset` (the node's `RvId` — the cross-backend keying contract).
#[derive(Clone, Copy)]
struct DrawCtx {
    k0: Value,
    k1: Value,
    lane: Value,
}

/// Emit the value of node `id` as an `f64` SSA value for the current lane, memoizing by `RvId` so
/// a shared sub-RV (e.g. `X` in `X + X`) is emitted ONCE — the same CSE guarantee the interpreter
/// gets (and, with counter keying, the same *draw*: one hash per node per lane).
fn emit_node(
    fb: &mut FunctionBuilder,
    graph: &RvGraph,
    id: RvId,
    ctx: DrawCtx,
    math: &MathRefs,
    memo: &mut HashMap<RvId, Value>,
    tables: &mut GatherTables,
) -> Value {
    if let Some(v) = memo.get(&id) {
        return *v;
    }
    let v = match graph.node(id) {
        RvNode::Src(Source::Uniform(u)) => emit_uniform(fb, ctx, id.0, u.lo, u.hi),
        RvNode::Src(Source::UniformInt { lo, hi }) => emit_uniform_int(fb, ctx, id.0, *lo, *hi),
        RvNode::Src(Source::Normal { mu, sigma }) => emit_normal(fb, ctx, id.0, *mu, *sigma),
        RvNode::Src(Source::Exp { rate }) => emit_exp(fb, ctx, id.0, *rate),
        RvNode::Src(Source::Geometric { p }) => emit_geometric(fb, ctx, id.0, *p),
        RvNode::Src(Source::Poisson { .. }) => unreachable!("profitable() excludes Poisson"),
        RvNode::Permutation { .. } | RvNode::Rotation { .. } | RvNode::ArrIndex { .. } => {
            unreachable!("profitable() excludes the array-valued draw nodes")
        }
        RvNode::Gather { elems, index } => {
            let xv = emit_node(fb, graph, *index, ctx, math, memo, tables);
            match gather_class(graph, elems) {
                Some(GatherClass::ConstTable) => {
                    let base = tables.base_ptr(graph, id, elems);
                    emit_gather_table(fb, tables.ptr_ty, xv, base, elems.len())
                }
                Some(GatherClass::SelectChain) => {
                    let evs: Vec<Value> = elems
                        .iter()
                        .map(|&e| emit_node(fb, graph, e, ctx, math, memo, tables))
                        .collect();
                    emit_gather_chain(fb, xv, &evs)
                }
                None => unreachable!("profitable() excludes large non-const Gather"),
            }
        }
        RvNode::ConstNum(x) => fb.ins().f64const(*x),
        RvNode::ConstBool(b) => fb.ins().f64const(if *b { 1.0 } else { 0.0 }),
        RvNode::Unary(op, a) => {
            let av = emit_node(fb, graph, *a, ctx, math, memo, tables);
            emit_unary(fb, math, *op, av)
        }
        RvNode::Binary(BinOp::Pow, a, b) => {
            let av = emit_node(fb, graph, *a, ctx, math, memo, tables);
            match const_int_exponent(graph, *b) {
                // Small non-negative integer power → repeated multiply (no libcall).
                Some(k) => emit_pow(fb, av, k),
                // Any other exponent → a `pow` libcall over both operands.
                None => {
                    let bv = emit_node(fb, graph, *b, ctx, math, memo, tables);
                    call2(fb, math.pow, av, bv)
                }
            }
        }
        RvNode::Binary(op, a, b) => {
            let av = emit_node(fb, graph, *a, ctx, math, memo, tables);
            let bv = emit_node(fb, graph, *b, ctx, math, memo, tables);
            emit_binary(fb, *op, av, bv)
        }
        RvNode::Select { cond, a, b } => {
            let cv = emit_node(fb, graph, *cond, ctx, math, memo, tables);
            let av = emit_node(fb, graph, *a, ctx, math, memo, tables);
            let bv = emit_node(fb, graph, *b, ctx, math, memo, tables);
            let zero = fb.ins().f64const(0.0);
            let cb = fb.ins().fcmp(FloatCC::NotEqual, cv, zero);
            fb.ins().select(cb, av, bv)
        }
    };
    memo.insert(id, v);
    v
}

/// Const-table gather ([`GatherClass::ConstTable`]): round the index ties-away, clamp to
/// `[0, last]`, load the 8-byte element from the program-owned table at `base` — the exact
/// semantics of the interpreter's `Inst::Gather` (bytecode.rs), bit-for-bit:
///
///   * ties-away rounding without a libcall: `r = nearest(x)` (ties-even) then `+1` exactly when
///     `x - r == 0.5` — for positive `x` that corrects precisely the ties `nearest` sent down, so
///     `r == f64::round(x)` there; a negative tie may land one off `f64::round` but every negative
///     `r` clamps to index 0 anyway (the interpreter also clamps *after* rounding). `+inf` passes
///     through (`d = NaN`, compare false) and clamps to `last`; `-inf` clamps to 0.
///   * `fcvt_to_sint` would TRAP on NaN, so the saturating form converts (NaN → 0) and the final
///     `select(x != x, NaN, loaded)` guard restores the interpreter's NaN-index → NaN result
///     (never element 0). Cranelift `FloatCC::NotEqual` is unordered-or-unequal: true iff NaN.
fn emit_gather_table(
    fb: &mut FunctionBuilder,
    ptr_ty: Type,
    xv: Value,
    base: *const f64,
    len: usize,
) -> Value {
    debug_assert!(len > 0, "gather table is never empty (eval rejects [])");
    let last = (len - 1) as f64; // exact: table lengths are far below 2^53
    let r0 = fb.ins().nearest(xv);
    let d = fb.ins().fsub(xv, r0);
    let half = fb.ins().f64const(0.5);
    let tie = fb.ins().fcmp(FloatCC::Equal, d, half);
    let one = fb.ins().f64const(1.0);
    let zero = fb.ins().f64const(0.0);
    let corr = fb.ins().select(tie, one, zero);
    let r = fb.ins().fadd(r0, corr);
    // Clamp in the float domain (fmax/fmin propagate NaN; the sat-convert then maps NaN to 0,
    // which the NaN guard below overrides).
    let rlo = fb.ins().fmax(r, zero);
    let lastc = fb.ins().f64const(last);
    let rcl = fb.ins().fmin(rlo, lastc);
    let idx = fb.ins().fcvt_to_sint_sat(types::I64, rcl);
    let off = fb.ins().imul_imm(idx, 8);
    let basec = fb.ins().iconst(ptr_ty, base as i64);
    let addr = fb.ins().iadd(basec, off);
    let loaded = fb
        .ins()
        .load(types::F64, MemFlags::trusted().with_readonly(), addr, 0);
    let is_nan = fb.ins().fcmp(FloatCC::NotEqual, xv, xv);
    let nan = fb.ins().f64const(f64::NAN);
    fb.ins().select(is_nan, nan, loaded)
}

/// Small non-const gather ([`GatherClass::SelectChain`]): a branch-free compare/select chain that
/// needs NO rounding — `acc = e[last]; for i in (0..last).rev() { acc = x < i+0.5 ? e[i] : acc }`
/// picks `e[min i: x < i+0.5]`, which equals round-ties-away-then-clamp for every non-NaN `x`
/// (a tie `x = i+0.5` fails the strict `<` and rounds away to `i+1`; anything below `0.5`,
/// including `-inf`, takes `e[0]`; `+inf` falls through to `e[last]`). `i + 0.5` is exact in f64
/// for any table within [`kernel::GATHER_SELECT_MAX`]. NaN fails every compare (falls through to
/// `e[last]`) and the final guard replaces it with NaN, matching the interpreter.
fn emit_gather_chain(fb: &mut FunctionBuilder, xv: Value, evs: &[Value]) -> Value {
    let last = evs.len() - 1; // table never empty (eval rejects [])
    let mut acc = evs[last];
    for i in (0..last).rev() {
        let t = fb.ins().f64const(i as f64 + 0.5);
        let c = fb.ins().fcmp(FloatCC::LessThan, xv, t);
        acc = fb.ins().select(c, evs[i], acc);
    }
    let is_nan = fb.ins().fcmp(FloatCC::NotEqual, xv, xv);
    let nan = fb.ins().f64const(f64::NAN);
    fb.ins().select(is_nan, nan, acc)
}

/// One pcg4d-3r round: the four dependent add-multiplies (`v0 += v1·v3; v1 += v2·v0; …`).
fn emit_cell_round(fb: &mut FunctionBuilder, v: &mut [Value; 4]) {
    let m = fb.ins().imul(v[1], v[3]);
    v[0] = fb.ins().iadd(v[0], m);
    let m = fb.ins().imul(v[2], v[0]);
    v[1] = fb.ins().iadd(v[1], m);
    let m = fb.ins().imul(v[0], v[1]);
    v[2] = fb.ins().iadd(v[2], m);
    let m = fb.ins().imul(v[1], v[2]);
    v[3] = fb.ins().iadd(v[3], m);
}

/// Per-word xorshift `v ^= v >> 16` over the four words.
fn emit_cell_xs(fb: &mut FunctionBuilder, v: &mut [Value; 4]) {
    for x in v.iter_mut() {
        let sh = fb.ins().ushr_imm(*x, 16);
        *x = fb.ins().bxor(*x, sh);
    }
}

/// The draw-cell hash for one `(lane, source)` — a straight-line transcription of
/// [`crate::rng::cell`] (pcg4d-3r) in pure `I32` arithmetic, so the JIT's draws are bit-identical
/// to the interpreter's.
fn emit_cell(fb: &mut FunctionBuilder, ctx: DrawCtx, src: u32) -> [Value; 4] {
    let srcv = fb.ins().iconst(types::I32, src as i32 as i64);
    let mut v = [ctx.k0, ctx.k1, ctx.lane, srcv];
    // LCG step per word (i32 mul/add wrap like the Rust `wrapping_*`).
    for x in v.iter_mut() {
        let m = fb.ins().imul_imm(*x, 1664525);
        *x = fb.ins().iadd_imm(m, 1013904223);
    }
    emit_cell_round(fb, &mut v);
    emit_cell_xs(fb, &mut v);
    emit_cell_round(fb, &mut v);
    emit_cell_xs(fb, &mut v);
    emit_cell_round(fb, &mut v);
    v
}

/// The consumed 48 bits of a word pair as an `I64` (`((w0 >> 8) << 24) | (w1 >> 8)`) — the
/// integer the f64-uniform conversion and the Lemire bounded draw both start from.
fn emit_bits48(fb: &mut FunctionBuilder, w0: Value, w1: Value) -> Value {
    let a = fb.ins().ushr_imm(w0, 8);
    let b = fb.ins().ushr_imm(w1, 8);
    let a64 = fb.ins().uextend(types::I64, a);
    let b64 = fb.ins().uextend(types::I64, b);
    let hi = fb.ins().ishl_imm(a64, 24);
    fb.ins().bor(hi, b64)
}

/// Uniform `f64` in `[0, 1)` from a word pair — mirrors `rng::unit_f64` (`bits · 2⁻⁴⁸`).
fn emit_unit48(fb: &mut FunctionBuilder, w0: Value, w1: Value) -> Value {
    let bits = emit_bits48(fb, w0, w1);
    let f = fb.ins().fcvt_from_uint(types::F64, bits);
    let scale = fb.ins().f64const(1.0 / ((1u64 << 48) as f64));
    fb.ins().fmul(f, scale)
}

fn emit_uniform(fb: &mut FunctionBuilder, ctx: DrawCtx, src: u32, lo: f64, hi: f64) -> Value {
    let w = emit_cell(fb, ctx, src);
    let u = emit_unit48(fb, w[0], w[1]);
    let loc = fb.ins().f64const(lo);
    let span = fb.ins().f64const(hi - lo);
    let scaled = fb.ins().fmul(span, u);
    fb.ins().fadd(loc, scaled)
}

/// `unif_int(lo, hi)` as `f64` via Lemire's multiply-high on the 48 consumed bits:
/// `umulhi(bits48 << 16, count)` = `(bits48 · count) >> 48`, exactly the interpreter's
/// `(bits as u128 * count) >> 48` — uniform in `0..count`, bias ≤ `count / 2^48`.
/// `count >= 1`, so `count == 1` always gives `k == 0` (point mass at `lo`).
fn emit_uniform_int(fb: &mut FunctionBuilder, ctx: DrawCtx, src: u32, lo: f64, hi: f64) -> Value {
    let count = (hi - lo + 1.0).max(1.0) as u64;
    let w = emit_cell(fb, ctx, src);
    let bits = emit_bits48(fb, w[0], w[1]);
    let bits_hi = fb.ins().ishl_imm(bits, 16);
    let count_c = fb.ins().iconst(types::I64, count as i64);
    let k = fb.ins().umulhi(bits_hi, count_c);
    let kf = fb.ins().fcvt_from_uint(types::F64, k); // k < count ≤ 2^53 → exact
    let loc = fb.ins().f64const(lo);
    fb.ins().fadd(loc, kf)
}

/// Horner evaluation of `Σ c[i]·z^i` (coeffs low→high) as straight-line CLIF — the exact reduction
/// `crate::approx::horner` performs, so the inlined transcendentals match the reference op-for-op.
fn emit_horner(fb: &mut FunctionBuilder, z: Value, coeffs: &[f64]) -> Value {
    let mut acc = fb.ins().f64const(*coeffs.last().unwrap());
    for &c in coeffs.iter().rev().skip(1) {
        let mul = fb.ins().fmul(acc, z);
        let cc = fb.ins().f64const(c);
        acc = fb.ins().fadd(mul, cc);
    }
    acc
}

/// Inlined `ln(x)` for `x > 0` — straight-line transcription of [`crate::approx::ln`] (no libcall,
/// and the same polynomial the interpreter's fills now compute, so draws stay bit-identical).
fn emit_ln(fb: &mut FunctionBuilder, x: Value) -> Value {
    use std::f64::consts::{LN_2, SQRT_2};
    // x = m·2^e: pull the IEEE-754 exponent and mantissa fields out by bit-surgery.
    let bits = fb.ins().bitcast(types::I64, MemFlags::new(), x);
    let exp_raw = fb.ins().ushr_imm(bits, 52);
    let exp_masked = fb.ins().band_imm(exp_raw, 0x7ff);
    let e0 = fb.ins().iadd_imm(exp_masked, -1023);
    let mant = fb.ins().band_imm(bits, 0x000f_ffff_ffff_ffff);
    let mbits = fb.ins().bor_imm(mant, 0x3ff0_0000_0000_0000u64 as i64);
    let m0 = fb.ins().bitcast(types::F64, MemFlags::new(), mbits); // [1, 2)
                                                                   // Recenter when m0 > √2 (branchless): m = m0/2, e = e0 + 1, so |f| ≤ 0.172.
    let sqrt2 = fb.ins().f64const(SQRT_2);
    let big = fb.ins().fcmp(FloatCC::GreaterThan, m0, sqrt2);
    let half = fb.ins().f64const(0.5);
    let m0_half = fb.ins().fmul(m0, half);
    let m = fb.ins().select(big, m0_half, m0);
    let e0_p1 = fb.ins().iadd_imm(e0, 1);
    let e = fb.ins().select(big, e0_p1, e0);
    let ef = fb.ins().fcvt_from_sint(types::F64, e);
    // f = (m-1)/(m+1); ln(m) = 2·f·Σ cₖf²ᵏ; ln(x) = ln(m) + e·ln2.
    let one = fb.ins().f64const(1.0);
    let num = fb.ins().fsub(m, one);
    let den = fb.ins().fadd(m, one);
    let f = fb.ins().fdiv(num, den);
    let f2 = fb.ins().fmul(f, f);
    let p = emit_horner(fb, f2, &crate::approx::LN_COEFFS);
    let fp = fb.ins().fmul(f, p);
    let two = fb.ins().f64const(2.0);
    let two_fp = fb.ins().fmul(two, fp);
    let ln2 = fb.ins().f64const(LN_2);
    let e_ln2 = fb.ins().fmul(ef, ln2);
    fb.ins().fadd(two_fp, e_ln2)
}

/// Range-guarded `cos`/`sin`: the inlined polynomial ([`emit_trig_poly`]) for `|x| < TRIG_MAX`, else
/// the accurate library `sin`/`cos` shim (finding C3) — the 2-term reduction degrades past that, so
/// this keeps the emitted trig in agreement with the interpreter's libm on large arguments (e.g.
/// `sin(1e12 * X)`). Mirrors the `|x| < TRIG_MAX ? poly : call` shape in [`crate::approx::sin`].
/// (The Box–Muller draw path calls [`emit_trig_poly`] directly — its argument is always `< 2π`.)
fn emit_trig(fb: &mut FunctionBuilder, math: &MathRefs, x: Value, is_cos: bool) -> Value {
    let ax = fb.ins().fabs(x);
    let thresh = fb.ins().f64const(crate::approx::TRIG_MAX);
    let big = fb.ins().fcmp(FloatCC::GreaterThanOrEqual, ax, thresh);

    let big_block = fb.create_block();
    let poly_block = fb.create_block();
    let merge = fb.create_block();
    fb.append_block_param(merge, types::F64);
    fb.ins().brif(big, big_block, &[], poly_block, &[]);

    // |x| >= TRIG_MAX: defer to the library shim (rare — this branch is predicted not-taken).
    fb.switch_to_block(big_block);
    fb.seal_block(big_block);
    let shim = if is_cos { math.cos } else { math.sin };
    let big_v = call1(fb, shim, x);
    fb.ins().jump(merge, &[BlockArg::from(big_v)]);

    // |x| < TRIG_MAX: the inline polynomial.
    fb.switch_to_block(poly_block);
    fb.seal_block(poly_block);
    let poly_v = emit_trig_poly(fb, x, is_cos);
    fb.ins().jump(merge, &[BlockArg::from(poly_v)]);

    fb.switch_to_block(merge);
    fb.seal_block(merge);
    fb.block_params(merge)[0]
}

/// Inlined `cos(x)` (`is_cos`) / `sin(x)` for `|x| < TRIG_MAX` — transcription of
/// [`crate::approx::cos`]/[`crate::approx::sin`]: Cody–Waite reduce to `[-π/4, π/4]`, evaluate both
/// reduced kernels, then pick by quadrant. `nearest`/`fcvt_to_sint_sat` are native instructions.
fn emit_trig_poly(fb: &mut FunctionBuilder, x: Value, is_cos: bool) -> Value {
    use crate::approx::{COS_COEFFS, PIO2_HI, PIO2_LO, SIN_COEFFS};
    use std::f64::consts::FRAC_2_PI;
    // k = round(x·2/π); r = (x - k·π/2_hi) - k·π/2_lo ∈ [-π/4, π/4].
    let two_pi_inv = fb.ins().f64const(FRAC_2_PI);
    let scaled = fb.ins().fmul(x, two_pi_inv);
    let kf = fb.ins().nearest(scaled);
    let hi = fb.ins().f64const(PIO2_HI);
    let khi = fb.ins().fmul(kf, hi);
    let x1 = fb.ins().fsub(x, khi);
    let lo = fb.ins().f64const(PIO2_LO);
    let klo = fb.ins().fmul(kf, lo);
    let r = fb.ins().fsub(x1, klo);
    let z = fb.ins().fmul(r, r);
    // sin(r) = r + r·z·P_sin(z)
    let sp = emit_horner(fb, z, &SIN_COEFFS);
    let rz = fb.ins().fmul(r, z);
    let rzsp = fb.ins().fmul(rz, sp);
    let sin_r = fb.ins().fadd(r, rzsp);
    // cos(r) = 1 - z/2 + z²·P_cos(z)
    let cp = emit_horner(fb, z, &COS_COEFFS);
    let zz = fb.ins().fmul(z, z);
    let zzcp = fb.ins().fmul(zz, cp);
    let one = fb.ins().f64const(1.0);
    let half = fb.ins().f64const(0.5);
    let halfz = fb.ins().fmul(half, z);
    let onemhz = fb.ins().fsub(one, halfz);
    let cos_r = fb.ins().fadd(onemhz, zzcp);
    // Pick the kernel + sign for quadrant kq = (k as i64) & 3 (saturating cvt = Rust's `as`).
    let ki = fb.ins().fcvt_to_sint_sat(types::I64, kf);
    let kq = fb.ins().band_imm(ki, 3);
    let neg_sin = fb.ins().fneg(sin_r);
    let neg_cos = fb.ins().fneg(cos_r);
    let (q0, q1, q2, q3) = if is_cos {
        (cos_r, neg_sin, neg_cos, sin_r)
    } else {
        (sin_r, cos_r, neg_sin, neg_cos)
    };
    let c1 = fb.ins().icmp_imm(IntCC::Equal, kq, 1);
    let r1 = fb.ins().select(c1, q1, q0);
    let c2 = fb.ins().icmp_imm(IntCC::Equal, kq, 2);
    let r2 = fb.ins().select(c2, q2, r1);
    let c3 = fb.ins().icmp_imm(IntCC::Equal, kq, 3);
    fb.ins().select(c3, q3, r2)
}

/// `N(mu, sigma^2)` via Box–Muller over lane pairs — a straight-line transcription of
/// `rng::normal_pair` (bit-identical draws): hash the pair's EVEN lane (`lane & !1`), `u1` from
/// words 0+1 offset by half a 48-bit-grid ulp (dodges `ln(0)`), `u2` from words 2+3; the even lane
/// takes the cos branch, the odd lane the sin branch (selected by lane parity — both kernels are
/// computed by the shared reduction anyway). `sqrt` is native; `ln`/trig are the shared inlined
/// polynomials the interpreter also computes ([`emit_ln`]/[`emit_trig_poly`]).
fn emit_normal(fb: &mut FunctionBuilder, ctx: DrawCtx, src: u32, mu: f64, sigma: f64) -> Value {
    use std::f64::consts::TAU;
    let even_lane = fb.ins().band_imm(ctx.lane, !1i64 & 0xFFFF_FFFF);
    let pair_ctx = DrawCtx { lane: even_lane, ..ctx };
    let w = emit_cell(fb, pair_ctx, src);
    let bits1 = emit_bits48(fb, w[0], w[1]);
    let f1 = fb.ins().fcvt_from_uint(types::F64, bits1);
    let half = fb.ins().f64const(0.5);
    let f1h = fb.ins().fadd(f1, half);
    let scale = fb.ins().f64const(1.0 / ((1u64 << 48) as f64));
    let u1 = fb.ins().fmul(f1h, scale); // (0, 1)
    let u2 = emit_unit48(fb, w[2], w[3]);
    let lnv = emit_ln(fb, u1);
    let neg2 = fb.ins().f64const(-2.0);
    let inner = fb.ins().fmul(neg2, lnv);
    let r = fb.ins().sqrt(inner);
    let tau = fb.ins().f64const(TAU);
    let ang = fb.ins().fmul(tau, u2);
    // Argument is `TAU * u2 ∈ [0, 2π)` — always well inside the polynomial's range, so call it
    // directly (no range guard) to keep the hot Box–Muller draw lean.
    let c = emit_trig_poly(fb, ang, true);
    let sn = emit_trig_poly(fb, ang, false);
    let parity = fb.ins().band_imm(ctx.lane, 1);
    let is_odd = fb.ins().icmp_imm(IntCC::NotEqual, parity, 0);
    let branch = fb.ins().select(is_odd, sn, c);
    let rc = fb.ins().fmul(r, branch);
    let sig = fb.ins().f64const(sigma);
    let scaled = fb.ins().fmul(sig, rc);
    let mu_c = fb.ins().f64const(mu);
    fb.ins().fadd(mu_c, scaled)
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `rng::fill_exp` bit-for-bit
/// (`1 - u ∈ [2⁻⁴⁸, 1]` keeps `ln` finite and in the polynomial's normal range).
fn emit_exp(fb: &mut FunctionBuilder, ctx: DrawCtx, src: u32, rate: f64) -> Value {
    let w = emit_cell(fb, ctx, src);
    let u = emit_unit48(fb, w[0], w[1]);
    let one = fb.ins().f64const(1.0);
    let om = fb.ins().fsub(one, u); // (0, 1]
    let lnv = emit_ln(fb, om);
    let neg = fb.ins().fneg(lnv);
    let rate_c = fb.ins().f64const(rate);
    fb.ins().fdiv(neg, rate_c)
}

/// `Geometric(p)` (failures before first success) via `floor(ln(1 - u) / ln(1 - p))` — mirrors
/// `rng::fill_geometric` bit-for-bit. `ln(1 - p)` is a compile-time constant (folded in Rust);
/// `p == 1` makes it `-inf`, so every draw floors to `0`, matching the interpreter's point mass.
fn emit_geometric(fb: &mut FunctionBuilder, ctx: DrawCtx, src: u32, p: f64) -> Value {
    let denom = (1.0 - p).ln();
    let w = emit_cell(fb, ctx, src);
    let u = emit_unit48(fb, w[0], w[1]);
    let one = fb.ins().f64const(1.0);
    let om = fb.ins().fsub(one, u); // (0, 1]
    let lnv = emit_ln(fb, om);
    let denom_c = fb.ins().f64const(denom);
    let q = fb.ins().fdiv(lnv, denom_c);
    fb.ins().floor(q)
}

/// `base ^ k` for a small non-negative integer `k`, as repeated multiply (k = 0 → 1.0).
fn emit_pow(fb: &mut FunctionBuilder, base: Value, k: u32) -> Value {
    if k == 0 {
        return fb.ins().f64const(1.0);
    }
    let mut acc = base;
    for _ in 1..k {
        acc = fb.ins().fmul(acc, base);
    }
    acc
}

/// Full-domain `ln(x)` — the inlined [`emit_ln`] polynomial (positive finite inputs only) wrapped
/// in the guards that make it agree with `f64::ln` everywhere the *user* can steer an RV:
/// `x > 0` → poly, `x == 0` → `-inf`, `x < 0` / NaN → NaN, `+inf` → `+inf`. (The raw poly is only
/// ever fed positive uniforms by the draw kernels, so they keep calling it unguarded.)
fn emit_ln_guarded(fb: &mut FunctionBuilder, a: Value) -> Value {
    // Subnormal positive inputs (exponent field 0) would corrupt `emit_ln`'s mantissa bit-surgery,
    // so scale them into the normal range and correct by `54·ln2` (finding C9). `is_sub` is also
    // true for `a <= 0`, but those lanes are overwritten by the domain selects below, so the bogus
    // scaled value is harmless. Normal inputs pass through unscaled (select picks the raw poly).
    let min_pos = fb.ins().f64const(f64::MIN_POSITIVE);
    let is_sub = fb.ins().fcmp(FloatCC::LessThan, a, min_pos);
    let scale = fb.ins().f64const(crate::approx::LN_SUBNORMAL_SCALE);
    let a_scaled = fb.ins().fmul(a, scale);
    let a_in = fb.ins().select(is_sub, a_scaled, a);
    let poly_raw = emit_ln(fb, a_in);
    let corr = fb.ins().f64const(crate::approx::LN_SUBNORMAL_CORR);
    let poly_corr = fb.ins().fsub(poly_raw, corr);
    let poly = fb.ins().select(is_sub, poly_corr, poly_raw);
    let zero = fb.ins().f64const(0.0);
    let neg_inf = fb.ins().f64const(f64::NEG_INFINITY);
    let nan = fb.ins().f64const(f64::NAN);
    // Non-positive lanes: 0 → -inf; negative or NaN input → NaN.
    let is_zero = fb.ins().fcmp(FloatCC::Equal, a, zero);
    let non_pos = fb.ins().select(is_zero, neg_inf, nan);
    let is_pos = fb.ins().fcmp(FloatCC::GreaterThan, a, zero);
    let r = fb.ins().select(is_pos, poly, non_pos);
    // The poly mangles +inf (exponent bit-surgery reads it as 2^1024) — patch it back to +inf.
    let inf = fb.ins().f64const(f64::INFINITY);
    let is_inf = fb.ins().fcmp(FloatCC::Equal, a, inf);
    fb.ins().select(is_inf, inf, r)
}

fn emit_unary(fb: &mut FunctionBuilder, math: &MathRefs, op: UnOp, a: Value) -> Value {
    match op {
        UnOp::Neg => fb.ins().fneg(a),
        UnOp::Not => {
            // logical not over a 0/1 column: (a == 0) ? 1 : 0
            let zero = fb.ins().f64const(0.0);
            let is_zero = fb.ins().fcmp(FloatCC::Equal, a, zero);
            bool_to_f64(fb, is_zero)
        }
        UnOp::Sin => emit_trig(fb, math, a, false),
        UnOp::Cos => emit_trig(fb, math, a, true),
        UnOp::Atan => call1(fb, math.atan, a),
        UnOp::Round => call1(fb, math.round, a),
        UnOp::Floor => fb.ins().floor(a),
        UnOp::Ceil => fb.ins().ceil(a),
        // Native sqrt instruction — IEEE correctly rounded, so bit-identical to the interpreter's
        // `f64::sqrt` on the whole domain (incl. sqrt(-0.0) = -0.0, sqrt(x<0) = NaN). This being a
        // single fused instruction (not a `pow` libcall) is the point of `UnOp::Sqrt`
        // (PLAN-PERF-2 §5).
        UnOp::Sqrt => fb.ins().sqrt(a),
        UnOp::Ln => emit_ln_guarded(fb, a),
        // e^x via the library `exp` shim — bit-identical to the interpreter's `f64::exp` (the old
        // `pow(e, x)` could differ in the last bit; finding C9). Whole domain handled by libm.
        UnOp::Exp => call1(fb, math.exp, a),
        UnOp::Sign => {
            // -1 / 0 / +1 as (a > 0) - (a < 0), matching `apply_un` (0 exactly at 0, unlike signum).
            let zero = fb.ins().f64const(0.0);
            let gt = fb.ins().fcmp(FloatCC::GreaterThan, a, zero);
            let lt = fb.ins().fcmp(FloatCC::LessThan, a, zero);
            let gtf = bool_to_f64(fb, gt);
            let ltf = bool_to_f64(fb, lt);
            fb.ins().fsub(gtf, ltf)
        }
    }
}

fn emit_binary(fb: &mut FunctionBuilder, op: BinOp, a: Value, b: Value) -> Value {
    match op {
        BinOp::Add => fb.ins().fadd(a, b),
        BinOp::Sub => fb.ins().fsub(a, b),
        BinOp::Mul => fb.ins().fmul(a, b),
        BinOp::Div => fb.ins().fdiv(a, b),
        BinOp::Mod => {
            // floored modulo: a − b·floor(a/b)
            let q = fb.ins().fdiv(a, b);
            let fq = fb.ins().floor(q);
            let bf = fb.ins().fmul(b, fq);
            fb.ins().fsub(a, bf)
        }
        BinOp::Lt => cmp_to_f64(fb, FloatCC::LessThan, a, b),
        BinOp::Gt => cmp_to_f64(fb, FloatCC::GreaterThan, a, b),
        BinOp::Le => cmp_to_f64(fb, FloatCC::LessThanOrEqual, a, b),
        BinOp::Ge => cmp_to_f64(fb, FloatCC::GreaterThanOrEqual, a, b),
        BinOp::Eq => cmp_to_f64(fb, FloatCC::Equal, a, b),
        BinOp::Ne => cmp_to_f64(fb, FloatCC::NotEqual, a, b),
        BinOp::And => logic_to_f64(fb, a, b, true),
        BinOp::Or => logic_to_f64(fb, a, b, false),
        BinOp::Pow => unreachable!("Pow is handled before emit_binary"),
    }
}

/// Float compare → `1.0`/`0.0` column.
fn cmp_to_f64(fb: &mut FunctionBuilder, cc: FloatCC, a: Value, b: Value) -> Value {
    let c = fb.ins().fcmp(cc, a, b);
    bool_to_f64(fb, c)
}

/// `&&` / `||` over 0/1 columns: `(a != 0) op (b != 0)` → `1.0`/`0.0`.
fn logic_to_f64(fb: &mut FunctionBuilder, a: Value, b: Value, and: bool) -> Value {
    let zero = fb.ins().f64const(0.0);
    let an = fb.ins().fcmp(FloatCC::NotEqual, a, zero);
    let bn = fb.ins().fcmp(FloatCC::NotEqual, b, zero);
    let r = if and {
        fb.ins().band(an, bn)
    } else {
        fb.ins().bor(an, bn)
    };
    bool_to_f64(fb, r)
}

/// Select `1.0` when the boolean (any nonzero int) is true, else `0.0`.
fn bool_to_f64(fb: &mut FunctionBuilder, cond: Value) -> Value {
    let one = fb.ins().f64const(1.0);
    let zero = fb.ins().f64const(0.0);
    fb.ins().select(cond, one, zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests exercise the *emitted kernel itself*, so they must compile regardless of the
    /// amortization gate (`kernel::break_even_draws`), which would otherwise interpret a short run.
    /// A draw count this large always clears it.
    const ENOUGH_DRAWS: usize = usize::MAX;

    use crate::kernel::supported;
    use crate::sampler::moments;

    // The shared cross-backend conformance corpus (finding C2), also consumed by `wasm_emit`.
    use crate::conformance;

    /// First root sample from a compiled program (seeded to 0). For an RNG-free graph every lane is
    /// identical and seed-independent, so `[0]` fully characterizes the backend's output.
    fn first_sample(program: &dyn Program) -> f64 {
        let mut r = program.runner();
        r.position(0, 0);
        let cap = r.batch_cap();
        r.next_batch(cap)[0]
    }

    /// **Const-graph exact-equality suite (finding C2).** For every RNG-free program the JIT kernel
    /// must be **bit-identical** to the interpreter oracle — there is no Monte-Carlo noise to hide a
    /// divergence, so this is the strongest check of the "backend only changes speed" contract. It
    /// pins the C3 (large-arg trig), C8 (`&&`/`||`), and C9 (`exp`) fixes at the bit level.
    #[test]
    fn conformance_const_graphs_bit_identical_interp_vs_jit() {
        for (label, src) in conformance::CONST_CASES {
            let mut eng = crate::Engine::new();
            let id = match eng.run_rv(src).unwrap() {
                crate::Value::Dist(id) => id,
                other => panic!("{label}: expected a dist, got {other:?}"),
            };
            let g = eng.graph();
            let interp = first_sample(&*InterpBackend.compile(g, id, ENOUGH_DRAWS));
            let jit = first_sample(&build(g, id).expect("jit build failed"));
            assert_eq!(
                interp.to_bits(),
                jit.to_bits(),
                "{label}: interp {interp} ({:#018x}) != jit {jit} ({:#018x})",
                interp.to_bits(),
                jit.to_bits()
            );
        }
    }

    /// The RNG half of the shared corpus: the JIT must agree with the interpreter in distribution on
    /// every case (mean within tolerance), including the higher-moment and wide/large-argument trig
    /// probes the JIT corpus previously lacked (finding C2).
    #[test]
    fn conformance_rng_cases_match_interp() {
        for (_label, src, seed) in conformance::RNG_CASES {
            assert_jit_matches_interp(src, *seed);
        }
    }

    /// JIT and interpreter must agree *in distribution* on a graph the JIT supports. We compare
    /// moments (not draw-for-draw — the RNG consumption order differs by design).
    fn assert_jit_matches_interp(src: &str, seed: u64) {
        let mut eng = crate::Engine::new();
        let rv = eng.run_rv(src).unwrap();
        let id = match rv {
            crate::Value::Dist(id) => id,
            other => panic!("expected a dist, got {other:?}"),
        };
        let graph = eng.graph();
        assert!(supported(graph, id), "case must be JIT-supported: {src}");

        // Force the generated kernel (bypass the profitability gate, which may prefer the
        // interpreter for transcendental-bound graphs) so this test always validates codegen.
        let program = build(graph, id).expect("jit build failed");
        let mut jit = program.runner();
        jit.position(seed, 0);
        let cap = jit.batch_cap();

        // Counter keying makes the JIT and interpreter agree DRAW-FOR-DRAW (the PLAN-PREGPU
        // parity contract), so compare the first batch bit-for-bit against the interpreter —
        // far stronger than the old in-distribution check.
        let mut ir = InterpBackend.compile(graph, id, ENOUGH_DRAWS).runner();
        ir.position(seed, 0);
        {
            let interp_col: Vec<f64> = ir.next_batch(cap).to_vec();
            let jit_col = jit.next_batch(cap);
            for (lane, (a, b)) in interp_col.iter().zip(jit_col.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{src}: lane {lane}: interp {a} ({:#018x}) != jit {b} ({:#018x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }

        // And the mean over a larger run still matches the sampler-driven estimate (guards the
        // lane advance across batches, which the single-batch bitwise check can't see).
        let mut sum = 0.0;
        let mut count = 0u64;
        jit.position(seed, 0);
        for _ in 0..16 {
            for &x in jit.next_batch(cap) {
                sum += x;
                count += 1;
            }
        }
        let jit_mean = sum / count as f64;
        let interp_mean = moments(graph, id, count as usize, seed).mean;
        assert!(
            (jit_mean - interp_mean).abs() < 0.05 + 0.05 * interp_mean.abs(),
            "{src}: jit_mean={jit_mean} interp_mean={interp_mean}"
        );
    }

    #[test]
    fn jit_uniform_arithmetic_matches_interp() {
        assert_jit_matches_interp("use rand; X ~ unif(0,1); 2*X + 3", 1);
        assert_jit_matches_interp("use rand; X ~ unif(-1,1); X*X*X + X", 2);
    }

    #[test]
    fn jit_dice_and_indicator_match_interp() {
        assert_jit_matches_interp("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B", 3);
        assert_jit_matches_interp("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X^2 + Y^2 < 1", 4);
    }

    #[test]
    fn jit_continuous_sources_match_interp() {
        // B2: normal/exp/geometric are now emitted directly (ln/cos shims + native sqrt/floor).
        assert_jit_matches_interp("use rand; Z ~ normal(2,3); Z", 5);
        assert_jit_matches_interp("use rand; X ~ exponential(2); X", 6);
        assert_jit_matches_interp("use rand; G ~ geometric(0.25); G", 7);
        // _int variants exercise the Round ufunc on top of a continuous source.
        assert_jit_matches_interp("use rand; Z ~ normal_int(10,3); Z", 8);
    }

    #[test]
    fn jit_ufuncs_and_nonconst_pow_match_interp() {
        // sin/cos/atan/sign ufuncs and a non-constant `^` (pow libcall) over an RV.
        assert_jit_matches_interp("use rand; use math; X ~ unif(0,1); sin(X) + cos(X)", 9);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-1,1); atan(X)", 10);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-1,1); sign(X)", 11);
        assert_jit_matches_interp("use rand; A ~ unif(1,2); B ~ unif(1,2); A ^ B", 12);
    }

    #[test]
    fn jit_mod_floor_ceil_match_interp() {
        // The new VM ops (BinOp::Mod via floor, UnOp::Floor/Ceil native instructions) must agree
        // with the interpreter draw-for-distribution.
        assert_jit_matches_interp("use rand; X ~ unif(0,10); X % 3", 13);
        assert_jit_matches_interp("use rand; X ~ unif(-5,5); X % 4", 14);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-3,3); math::floor(X)", 15);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-3,3); math::ceil(X)", 16);
        // floored modulo of a negative dividend: a − b·floor(a/b) (composed op chain)
        assert_jit_matches_interp("use rand; use math; X ~ unif(0,8); math::floor(X) % 3", 17);
    }

    #[test]
    fn jit_exp_ln_match_interp() {
        // exp lowers to pow(e, x): E[e^X] over N(0,1) = e^0.5 (the lognormal mean).
        assert_jit_matches_interp("use rand; use math; X ~ normal(0,1); exp(X)", 18);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-1,1); exp(X)", 19);
        // ln of a strictly positive draw hits the inlined poly (E[ln U(0.5,3)] via the guard's
        // x > 0 arm only).
        assert_jit_matches_interp("use rand; use math; X ~ unif(0.5, 3); log(X)", 20);
        // Domain guard, negative lanes: log(x<0) is NaN and NaN == NaN is false, so the indicator
        // mean is P(X > 0) = 0.6 — matching the interpreter's f64::ln semantics lane-for-lane.
        assert_jit_matches_interp("use rand; use math; X ~ unif(-2, 3); log(X) == log(X)", 21);
        // Domain guard, zero lanes: log(0) = -inf < -100, P = 1/5 over unif_int(0,4) (an
        // indicator, so the mean stays finite and comparable).
        assert_jit_matches_interp(
            "use rand; use math; X ~ unif_int(0, 4); log(X) < 0 - 100",
            22,
        );
        // Domain guard, +inf: X/0 = +inf per lane; log(+inf) = +inf > 100 surely.
        assert_jit_matches_interp("use rand; use math; X ~ unif(1, 2); log(X / 0) > 100", 23);
    }

    /// Codegen-quality probe: race the Cranelift JIT against a hand-written, LLVM-compiled fused
    /// kernel computing the *identical* graph (same inlined xoshiro, same arithmetic). Isolates
    /// Cranelift's codegen vs LLVM's. Ignored by default; run with:
    /// `cargo test -p noise-core --features jit --release -- --ignored --nocapture bench_cranelift_vs_llvm`
    #[test]
    #[ignore]
    fn bench_cranelift_vs_llvm() {
        use std::time::Instant;

        let n = 8_000_000usize;
        let src = "use rand; X ~ unif(0,1); ((X*X+X)*X - X)*X + X*X - X + 1";
        let mut eng = crate::Engine::new();
        let id = match eng.run_rv(src).unwrap() {
            crate::Value::Dist(id) => id,
            _ => unreachable!(),
        };
        let program = build(eng.graph(), id).expect("jit build");
        let mut runner = program.runner();
        runner.position(0xC0FFEE, 0);
        let cap = runner.batch_cap();
        for _ in 0..64 {
            runner.next_batch(cap);
        }
        let t = Instant::now();
        let (mut acc, mut done) = (0.0f64, 0usize);
        while done < n {
            let col = runner.next_batch(cap);
            acc += col[0] + col[cap - 1];
            done += cap;
        }
        let jit_mps = done as f64 / t.elapsed().as_secs_f64() / 1e6;
        std::hint::black_box(acc);

        // Hand-written, fused, LLVM-compiled equivalent — also fills a column (same memory
        // behavior as the JIT), so the comparison is pure codegen quality, not store traffic.
        let key = crate::rng::Key::from_seed(0xC0FFEE);
        let mut buf = vec![0.0f64; cap];
        let mut lane = 0u32;
        let t = Instant::now();
        let (mut acc2, mut done2) = (0.0f64, 0usize);
        while done2 < n {
            for slot in buf.iter_mut() {
                let x = crate::rng::unit_uniform(key, lane, 0);
                lane = lane.wrapping_add(1);
                *slot = ((x * x + x) * x - x) * x + x * x - x + 1.0;
            }
            acc2 += buf[0] + buf[cap - 1];
            done2 += cap;
        }
        let llvm_mps = done2 as f64 / t.elapsed().as_secs_f64() / 1e6;
        std::hint::black_box(acc2);

        println!("\n  fused poly_deep kernel, both column-filling (single thread, M elem/s):");
        println!(
            "    cranelift(jit) {jit_mps:7.0}   llvm(rust) {llvm_mps:7.0}   llvm/jit {:.2}x",
            llvm_mps / jit_mps
        );
    }

    /// Compiling and dropping many kernels must run cleanly. This is the *sequential* half of the
    /// story and it always passed — including while `Drop` still called `free_memory()`, which was
    /// unsound. It takes concurrent execution to expose that; see
    /// `a_live_kernel_survives_other_modules_being_dropped` for the case that actually caught it.
    #[test]
    fn compile_and_drop_many_kernels_is_clean() {
        let mut eng = crate::Engine::new();
        let id = match eng
            .run_rv("use rand; X ~ unif(0,1); X*X + 2*X - 1")
            .unwrap()
        {
            crate::Value::Dist(id) => id,
            _ => unreachable!(),
        };
        let g = eng.graph();
        for _ in 0..200 {
            let program = build(g, id).expect("jit build failed");
            // Touch the code memory before it is freed on drop.
            let mut r = program.runner();
            r.position(1, 0);
            let cap = r.batch_cap();
            let x = r.next_batch(cap)[0];
            assert!(x.is_finite());
            drop(r);
            drop(program); // Drop → free_memory (must be UB-free)
        }
    }

    /// A const-table gather is a LEAF over its elems ([`kernel::gather_class`]): a 10k-point
    /// `empirical`-shaped table neither counts toward `MAX_CODEGEN_NODES` nor blocks the gate —
    /// `profitable` accepts it at high draws, the cone stays latency-bound (multi-stream), and the
    /// compiled kernel's indexed load reproduces the uniform-over-the-data distribution.
    #[test]
    fn const_table_gather_is_profitable_and_matches() {
        use crate::dist::RvKind;
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
        assert_eq!(crate::kernel::cone_size(&g, root), 2);
        assert!(supported(&g, root));
        assert!(profitable(
            &g,
            root,
            true,
            ENOUGH_DRAWS,
            kernel::MIN_DRAWS_JIT
        ));
        let program = build(&g, root).expect("jit build failed");
        let mut r = program.runner();
        r.position(42, 0);
        let cap = r.batch_cap();
        let (mut sum, mut count) = (0.0f64, 0u64);
        for _ in 0..16 {
            for &x in r.next_batch(cap) {
                sum += x;
                count += 1;
            }
        }
        let mean = sum / count as f64;
        // E over unif_int(0, 9999) of table[i] = i is 4999.5; fixed seed, generous tolerance.
        assert!((mean - 4999.5).abs() < 100.0, "mean={mean}");
    }

    #[test]
    fn poisson_still_falls_back_to_interpreter() {
        // Poisson keeps the interpreter (Knuth's variable-length per-lane loop); the fallback must
        // still produce correct draws (mean ≈ lambda = 3).
        let mut eng = crate::Engine::new();
        let rv = eng.run_rv("use rand; K ~ poisson(3); K").unwrap();
        let id = match rv {
            crate::Value::Dist(id) => id,
            _ => unreachable!(),
        };
        assert!(!supported(eng.graph(), id));
        let m = moments(eng.graph(), id, 200_000, 7);
        assert!((m.mean - 3.0).abs() < 0.05, "fallback mean = {}", m.mean);
    }

    /// **The codegen amortization curve** — the measurement the profitability gate was missing.
    ///
    /// `profitable()` decides emit-vs-interpret from the graph's *shape* alone. It has no idea how
    /// many samples the query will draw, so it happily compiles a 20k-node kernel to take 3,000
    /// draws — and Cranelift's compile time then dwarfs everything the fused kernel saves. That is
    /// exactly what `examples/noise_colors.noise` (14 forcings × 3k samples) and
    /// `examples/turboquant.noise` (10 × 10k) do, and both are *slower* with the JIT on.
    ///
    /// This prints, per cone size: JIT compile time, the per-draw rate of each backend, and the
    /// resulting **break-even draw count** — below which compiling is a net loss. Those numbers are
    /// what `kernel::BREAK_EVEN_*` are fitted to.
    ///
    /// `cargo test -p noise-core --features jit --release -- --ignored --nocapture bench_amortization`
    #[test]
    #[ignore]
    fn bench_amortization() {
        use crate::backend::{Backend, InterpBackend};
        use std::time::Instant;

        // A cone of ~`k` distinct nodes that CSE can't collapse: (X+1)*(X+2)*...*(X+k).
        fn src_of(k: usize) -> String {
            let terms: Vec<String> = (1..=k).map(|i| format!("(X+{i})")).collect();
            format!("use rand; X ~ unif(0,1); {}", terms.join("*"))
        }

        // Median wall time to compile the cone with `backend`, over `reps`.
        fn compile_ms(backend: &dyn Backend, g: &RvGraph, root: RvId, reps: usize) -> f64 {
            let mut ts: Vec<f64> = (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    let p = backend.compile(g, root, ENOUGH_DRAWS);
                    std::hint::black_box(&p);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ts[reps / 2]
        }

        // Nanoseconds per draw, steady state (compile excluded, cache warm).
        fn ns_per_draw(program: &dyn crate::backend::Program, batches: usize) -> f64 {
            let mut r = program.runner();
            r.position(0xC0FFEE, 0);
            let cap = r.batch_cap();
            for _ in 0..8 {
                r.next_batch(cap);
            }
            let t = Instant::now();
            let mut acc = 0.0f64;
            for _ in 0..batches {
                acc += r.next_batch(cap)[0];
            }
            std::hint::black_box(acc);
            t.elapsed().as_secs_f64() * 1e9 / (batches * cap) as f64
        }

        println!(
            "\n{:>7}{:>9}{:>14}{:>12}{:>12}{:>14}",
            "k", "NODES", "JIT COMPILE", "INTERP", "JIT", "BREAK-EVEN"
        );
        println!(
            "{:>7}{:>9}{:>14}{:>12}{:>12}{:>14}",
            "", "", "(ms)", "(ns/draw)", "(ns/draw)", "(draws)"
        );
        for k in [2usize, 8, 32, 128, 512, 2048, 8192] {
            let src = src_of(k);
            let mut eng = crate::Engine::new();
            let v = eng.run_rv(&src).expect("build");
            let crate::Value::Dist(root) = v else {
                panic!("not a dist")
            };
            // Measure the *simplified* cone — that's what the backends actually lower.
            let (g, root) = crate::simplify::simplify(eng.graph(), root);
            let nodes = crate::kernel::cost(&g, root).ops;

            let reps = if k >= 2048 { 3 } else { 9 };
            let jit_ms = compile_ms(&JitBackend::new(), &g, root, reps);
            let interp_ms = compile_ms(&InterpBackend, &g, root, reps);

            let batches = (2_000_000 / (k.max(1) * BATCH)).max(4);
            let i_ns = ns_per_draw(&*InterpBackend.compile(&g, root, ENOUGH_DRAWS), batches);
            let j_ns = ns_per_draw(&*JitBackend::new().compile(&g, root, ENOUGH_DRAWS), batches);

            // Compiling is worth it only once the per-draw saving has refunded the extra compile.
            let saving = i_ns - j_ns;
            let extra_compile_ns = (jit_ms - interp_ms) * 1e6;
            let breakeven = if saving > 0.0 {
                format!("{:.0}", extra_compile_ns / saving)
            } else {
                "never".into()
            };
            println!("{k:>7}{nodes:>9}{jit_ms:>14.2}{i_ns:>12.2}{j_ns:>12.2}{breakeven:>14}");
        }
    }
}
