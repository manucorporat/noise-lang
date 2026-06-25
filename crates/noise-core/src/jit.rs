//! Cranelift native JIT backend (PLAN.md Phase 4, step "B1").
//!
//! Compiles the sample-DAG into ONE fused machine-code kernel: a per-lane loop that draws its
//! sources, computes the whole expression keeping intermediates in registers, and stores a single
//! `f64` per lane — so, unlike the columnar interpreter, **no intermediate column is materialized
//! to memory**. That fusion is the win on arithmetic-dense graphs (see the `poly_*` benches).
//!
//! The PRNG (xoshiro256++ / SplitMix64, mirroring `rng`) is **inlined into the generated code** —
//! no per-lane call back into Rust, which would otherwise dominate. Because the kernel consumes
//! the RNG stream per-lane (interleaved across sources) rather than column-by-column like the
//! interpreter, the JIT and the interpreter agree *in distribution* but not draw-for-draw under a
//! shared seed; the per-source-substream fix (PLAN "fork (b)") is deferred.
//!
//! Scope of B1: `unif` / `unif_int` sources, `+ - * /`, integer-constant `**`, comparisons,
//! `&& ||`, unary `- !`, and lifted `if` (`Select`). Anything else (continuous distributions,
//! transcendental ufuncs, non-constant `**`) makes [`supported`] return `false`, and
//! [`JitBackend::compile`] falls back to the interpreter — so enabling `jit` never changes a
//! result, it only accelerates the graphs it can handle.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cranelift::codegen::ir::UserFuncName;
use cranelift::prelude::settings::Configurable;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use crate::ast::{BinOp, UnOp};
use crate::backend::{Backend, InterpBackend, Program, Runner};
use crate::bytecode::BATCH;
use crate::dist::{RvGraph, RvId, RvNode, Source};

/// `extern "C"` signature of a generated kernel: `kernel(out_ptr, n, state_ptr)` fills `out[0..n]`
/// with fresh root draws, reading and writing the 4-word xoshiro state through `state_ptr`.
type KernelFn = unsafe extern "C" fn(*mut f64, i64, *mut u64);

// Scalar math the kernel can't express as a single CLIF instruction is delegated to these
// `extern "C"` shims (registered as JIT symbols, called per lane). `sqrt`/`floor` are NOT here —
// they are native CLIF instructions on the targets we run. Names are prefixed `nz_` to avoid any
// clash with libm symbols the module might resolve itself.
extern "C" fn nz_ln(x: f64) -> f64 {
    x.ln()
}
extern "C" fn nz_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn nz_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn nz_atan(x: f64) -> f64 {
    x.atan()
}
extern "C" fn nz_round(x: f64) -> f64 {
    x.round()
}
extern "C" fn nz_pow(a: f64, b: f64) -> f64 {
    a.powf(b)
}

/// Module-level `FuncId`s for the math shims (declared once per build).
struct MathIds {
    ln: cranelift_module::FuncId,
    sin: cranelift_module::FuncId,
    cos: cranelift_module::FuncId,
    atan: cranelift_module::FuncId,
    round: cranelift_module::FuncId,
    pow: cranelift_module::FuncId,
}

/// In-function `FuncRef`s for the math shims (resolved once at the top of the kernel body, then
/// reused at every call site). `Copy`, so threading it through emission is free.
#[derive(Clone, Copy)]
struct MathRefs {
    ln: cranelift::codegen::ir::FuncRef,
    sin: cranelift::codegen::ir::FuncRef,
    cos: cranelift::codegen::ir::FuncRef,
    atan: cranelift::codegen::ir::FuncRef,
    round: cranelift::codegen::ir::FuncRef,
    pow: cranelift::codegen::ir::FuncRef,
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
    fn compile(&self, graph: &RvGraph, root: RvId) -> Box<dyn Program> {
        // Use the JIT only where it's expected to *win* — see `jit_profitable`. Transcendental-
        // bound graphs (per-lane `ln`/`cos` libcalls, no pair-sharing) are slower than the
        // interpreter's vectorized column fills, so they stay on the interpreter. Any codegen
        // failure also falls back. Either way correctness is never at risk, only the speedup.
        if !jit_profitable(graph, root) {
            return InterpBackend.compile(graph, root);
        }
        match build(graph, root) {
            Ok(program) => Box::new(program),
            Err(_) => InterpBackend.compile(graph, root),
        }
    }
}

/// Whether every node in the cone of `root` is something the JIT can emit. After B2 that is
/// everything except `Poisson` (Knuth's variable-length per-lane loop — still interpreter-only).
/// `Pow` with a small non-negative integer constant exponent lowers to repeated multiply; any
/// other exponent lowers to a `pow` libcall. (The backend selector uses [`jit_profitable`], which
/// also rejects Poisson; this stricter capability check is retained for tests.)
#[cfg(test)]
fn supported(graph: &RvGraph, root: RvId) -> bool {
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

/// Whether the JIT is expected to outperform the interpreter for this graph. True iff the graph
/// is supported (no Poisson) **and** the count of fused nodes (arithmetic, comparisons, selects,
/// cheap inline draws) strictly exceeds the transcendental-libcall weight.
///
/// Calibrated against `benches/sampling.rs`: per-lane `ln`/`cos`/`pow` are non-inlined libcalls
/// and (for `normal`) do ~2× the transcendental work of the interpreter's pair-sharing column
/// fill, so a transcendental-bound graph (`normal_sum`, `exp_tail`) is faster interpreted, while
/// a graph where arithmetic dominates (`normal_poly` — one normal feeding a deep polynomial) is
/// ~1.4× faster fused. `fusible > libcalls` puts the crossover between those measured cases.
fn jit_profitable(graph: &RvGraph, root: RvId) -> bool {
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
fn walk_cost(
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
        // A normal costs two libcalls (ln + cos) and draws two uniforms per lane — the heaviest.
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
        RvNode::Binary(BinOp::Pow, a, b) => {
            if const_int_exponent(graph, *b).is_some() {
                *fusible += 1; // repeated multiply, no libcall
                walk_cost(graph, *a, seen, fusible, libcalls)
            } else {
                *libcalls += 1; // pow libcall
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

/// If node `id` is a constant non-negative integer in `0..=64`, return it as an exponent count.
fn const_int_exponent(graph: &RvGraph, id: RvId) -> Option<u32> {
    match graph.node(id) {
        RvNode::ConstNum(x) if x.fract() == 0.0 && *x >= 0.0 && *x <= 64.0 => Some(*x as u32),
        _ => None,
    }
}

/// Replicate `Rng::seed_from_u64`'s SplitMix64 expansion to a raw 4-word xoshiro state, so the
/// JIT seeds identically to the interpreter's RNG (the per-lane consumption order still differs).
fn splitmix_state(seed: u64) -> [u64; 4] {
    let mut z = seed;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    [next(), next(), next(), next()]
}

/// The immutable JIT artifact: the finalized module (owns the executable memory) and the entry
/// pointer. Shared across worker threads behind an `Arc`.
struct JitProgramInner {
    // `func` points into `_module`'s code memory; the module is kept alive for the program's life.
    _module: JITModule,
    func: KernelFn,
}

// SAFETY: after `finalize_definitions`, the module's code is immutable and we never touch the
// module again (only keep it mapped). The kernel has NO global mutable state — its RNG state and
// output buffer are passed in per call — so concurrent calls from multiple threads with distinct
// arguments are data-race-free. Hence the artifact is safe to send and share between threads.
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
            state: splitmix_state(0),
        })
    }
}

/// A per-worker JIT runner: a clone of the shared kernel `Arc`, its own output buffer, and the
/// xoshiro state carried across batches.
struct JitRunner {
    inner: Arc<JitProgramInner>,
    buf: Vec<f64>,
    state: [u64; 4],
}

impl Runner for JitRunner {
    fn reseed(&mut self, seed: u64) {
        self.state = splitmix_state(seed);
    }

    fn next_batch(&mut self, len: usize) -> &[f64] {
        // Always fill the full BATCH (constant RNG consumption per call), then slice to `len`.
        let n = self.buf.len() as i64;
        // SAFETY: `func` is a finalized kernel with this exact ABI; `buf` holds `n` f64s and
        // `state` holds the 4-word RNG state, both valid for the duration of the call.
        unsafe {
            (self.inner.func)(self.buf.as_mut_ptr(), n, self.state.as_mut_ptr());
        }
        &self.buf[..len]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}

/// Build the JIT module + kernel for `root`. Returns an error on any Cranelift failure (the
/// caller then falls back to the interpreter).
fn build(graph: &RvGraph, root: RvId) -> Result<JitProgram, String> {
    // --- ISA + module setup ---
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    flags.set("use_colocated_libcalls", "false").map_err(|e| e.to_string())?;
    flags.set("is_pic", "false").map_err(|e| e.to_string())?;
    let isa = cranelift_native::builder()
        .map_err(|e| e.to_string())?
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())?;
    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    // Bind the math shim names to their Rust function pointers so the JIT can resolve the calls.
    builder.symbol("nz_ln", nz_ln as *const u8);
    builder.symbol("nz_sin", nz_sin as *const u8);
    builder.symbol("nz_cos", nz_cos as *const u8);
    builder.symbol("nz_atan", nz_atan as *const u8);
    builder.symbol("nz_round", nz_round as *const u8);
    builder.symbol("nz_pow", nz_pow as *const u8);
    let mut module = JITModule::new(builder);

    // Declare the math shims as imports (f64->f64, except `pow` which is f64,f64->f64).
    let math_ids = declare_math(&mut module)?;

    // kernel(out: *mut f64, n: i64, state: *mut u64)
    let ptr = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // out
    sig.params.push(AbiParam::new(types::I64)); // n
    sig.params.push(AbiParam::new(ptr)); // state
    let func_id = module
        .declare_function("kernel", Linkage::Export, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, func_id.as_u32());

    // --- function body ---
    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        // Resolve the math shims into this function once; `MathRefs` is Copy, reused at each call.
        let math = MathRefs {
            ln: module.declare_func_in_func(math_ids.ln, fb.func),
            sin: module.declare_func_in_func(math_ids.sin, fb.func),
            cos: module.declare_func_in_func(math_ids.cos, fb.func),
            atan: module.declare_func_in_func(math_ids.atan, fb.func),
            round: module.declare_func_in_func(math_ids.round, fb.func),
            pow: module.declare_func_in_func(math_ids.pow, fb.func),
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
        let state_ptr = fb.block_params(entry)[2];

        // Loop counter `i` and the four xoshiro words live in mutable Variables.
        let i_var = fb.declare_var(types::I64);
        let s: [Variable; 4] = [
            fb.declare_var(types::I64),
            fb.declare_var(types::I64),
            fb.declare_var(types::I64),
            fb.declare_var(types::I64),
        ];
        let zero_i = fb.ins().iconst(types::I64, 0);
        fb.def_var(i_var, zero_i);
        for (k, v) in s.iter().enumerate() {
            let w = fb.ins().load(types::I64, MemFlags::trusted(), state_ptr, (8 * k) as i32);
            fb.def_var(*v, w);
        }
        fb.ins().jump(header, &[]);

        // header: if i < n goto body else exit  (left unsealed — body adds the back-edge)
        fb.switch_to_block(header);
        let iv = fb.use_var(i_var);
        let cond = fb.ins().icmp(IntCC::SignedLessThan, iv, n);
        fb.ins().brif(cond, body, &[], exit, &[]);

        // body: emit the fused DAG, store out[i], i += 1, loop.
        fb.switch_to_block(body);
        fb.seal_block(body);
        let mut memo: HashMap<RvId, Value> = HashMap::new();
        let result = emit_node(&mut fb, graph, root, &s, &math, &mut memo);
        let iv = fb.use_var(i_var);
        let off = fb.ins().imul_imm(iv, 8);
        let addr = fb.ins().iadd(out, off);
        fb.ins().store(MemFlags::trusted(), result, addr, 0);
        let inext = fb.ins().iadd_imm(iv, 1);
        fb.def_var(i_var, inext);
        fb.ins().jump(header, &[]);
        fb.seal_block(header); // both preds (entry, body) now known

        // exit: write the advanced RNG state back, return.
        fb.switch_to_block(exit);
        fb.seal_block(exit);
        for (k, v) in s.iter().enumerate() {
            let w = fb.use_var(*v);
            fb.ins().store(MemFlags::trusted(), w, state_ptr, (8 * k) as i32);
        }
        fb.ins().return_(&[]);
        fb.finalize();
    }

    module.define_function(func_id, &mut ctx).map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(func_id);
    // SAFETY: `code` is a finalized function with exactly the `KernelFn` ABI declared above; the
    // module is moved into the program so the code stays mapped for the pointer's lifetime.
    let func: KernelFn = unsafe { std::mem::transmute::<*const u8, KernelFn>(code) };

    Ok(JitProgram {
        inner: Arc::new(JitProgramInner { _module: module, func }),
    })
}

/// Declare the six math shims as module imports and return their `FuncId`s. (Errors are
/// stringified — `ModuleError` is large, and the caller only ever falls back on failure.)
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
        module.declare_function(name, Linkage::Import, sig).map_err(|e| e.to_string())
    };
    Ok(MathIds {
        ln: decl(module, "nz_ln", &sig1)?,
        sin: decl(module, "nz_sin", &sig1)?,
        cos: decl(module, "nz_cos", &sig1)?,
        atan: decl(module, "nz_atan", &sig1)?,
        round: decl(module, "nz_round", &sig1)?,
        pow: decl(module, "nz_pow", &sig2)?,
    })
}

/// `f(x)` for a one-arg shim `FuncRef`.
fn call1(fb: &mut FunctionBuilder, f: cranelift::codegen::ir::FuncRef, x: Value) -> Value {
    let c = fb.ins().call(f, &[x]);
    fb.inst_results(c)[0]
}

/// `f(x, y)` for a two-arg shim `FuncRef`.
fn call2(fb: &mut FunctionBuilder, f: cranelift::codegen::ir::FuncRef, x: Value, y: Value) -> Value {
    let c = fb.ins().call(f, &[x, y]);
    fb.inst_results(c)[0]
}

// --- DAG → CLIF emission (per lane, in the loop body) ---

/// Emit the value of node `id` as an `f64` SSA value, memoizing by `RvId` so a shared sub-RV
/// (e.g. `X` in `X + X`) is emitted ONCE — the same CSE guarantee the interpreter gets.
fn emit_node(
    fb: &mut FunctionBuilder,
    graph: &RvGraph,
    id: RvId,
    s: &[Variable; 4],
    math: &MathRefs,
    memo: &mut HashMap<RvId, Value>,
) -> Value {
    if let Some(v) = memo.get(&id) {
        return *v;
    }
    let v = match graph.node(id) {
        RvNode::Src(Source::Uniform(u)) => emit_uniform(fb, s, u.lo, u.hi),
        RvNode::Src(Source::UniformInt { lo, hi }) => emit_uniform_int(fb, s, *lo, *hi),
        RvNode::Src(Source::Normal { mu, sigma }) => emit_normal(fb, s, math, *mu, *sigma),
        RvNode::Src(Source::Exp { rate }) => emit_exp(fb, s, math, *rate),
        RvNode::Src(Source::Geometric { p }) => emit_geometric(fb, s, math, *p),
        RvNode::Src(Source::Poisson { .. }) => unreachable!("supported() excludes Poisson"),
        RvNode::ConstNum(x) => fb.ins().f64const(*x),
        RvNode::ConstBool(b) => fb.ins().f64const(if *b { 1.0 } else { 0.0 }),
        RvNode::Unary(op, a) => {
            let av = emit_node(fb, graph, *a, s, math, memo);
            emit_unary(fb, math, *op, av)
        }
        RvNode::Binary(BinOp::Pow, a, b) => {
            let av = emit_node(fb, graph, *a, s, math, memo);
            match const_int_exponent(graph, *b) {
                // Small non-negative integer power → repeated multiply (no libcall).
                Some(k) => emit_pow(fb, av, k),
                // Any other exponent → a `pow` libcall over both operands.
                None => {
                    let bv = emit_node(fb, graph, *b, s, math, memo);
                    call2(fb, math.pow, av, bv)
                }
            }
        }
        RvNode::Binary(op, a, b) => {
            let av = emit_node(fb, graph, *a, s, math, memo);
            let bv = emit_node(fb, graph, *b, s, math, memo);
            emit_binary(fb, *op, av, bv)
        }
        RvNode::Select { cond, a, b } => {
            let cv = emit_node(fb, graph, *cond, s, math, memo);
            let av = emit_node(fb, graph, *a, s, math, memo);
            let bv = emit_node(fb, graph, *b, s, math, memo);
            let zero = fb.ins().f64const(0.0);
            let cb = fb.ins().fcmp(FloatCC::NotEqual, cv, zero);
            fb.ins().select(cb, av, bv)
        }
    };
    memo.insert(id, v);
    v
}

/// One xoshiro256++ step, mutating the state Variables; returns the raw `u64` output.
/// Mirrors `Rng::next_u64` exactly (capturing pre-mutation words in locals).
fn emit_next_u64(fb: &mut FunctionBuilder, s: &[Variable; 4]) -> Value {
    let s0 = fb.use_var(s[0]);
    let s1 = fb.use_var(s[1]);
    let s2 = fb.use_var(s[2]);
    let s3 = fb.use_var(s[3]);

    let sum = fb.ins().iadd(s0, s3);
    let rot = fb.ins().rotl_imm(sum, 23);
    let result = fb.ins().iadd(rot, s0);

    let t = fb.ins().ishl_imm(s1, 17);
    let s2a = fb.ins().bxor(s2, s0); // s2 ^= s0
    let s3a = fb.ins().bxor(s3, s1); // s3 ^= s1
    let s1a = fb.ins().bxor(s1, s2a); // s1 ^= s2 (updated)
    let s0a = fb.ins().bxor(s0, s3a); // s0 ^= s3 (updated)
    let s2b = fb.ins().bxor(s2a, t); // s2 ^= t
    let s3b = fb.ins().rotl_imm(s3a, 45); // s3 = rotl(s3, 45)

    fb.def_var(s[0], s0a);
    fb.def_var(s[1], s1a);
    fb.def_var(s[2], s2b);
    fb.def_var(s[3], s3b);
    result
}

/// Uniform `f64` in `[0, 1)` from the top 53 bits — mirrors `Rng::next_f64`.
fn emit_next_f64(fb: &mut FunctionBuilder, s: &[Variable; 4]) -> Value {
    let bits = emit_next_u64(fb, s);
    let shifted = fb.ins().ushr_imm(bits, 11);
    let f = fb.ins().fcvt_from_uint(types::F64, shifted);
    let scale = fb.ins().f64const(1.0 / ((1u64 << 53) as f64));
    fb.ins().fmul(f, scale)
}

fn emit_uniform(fb: &mut FunctionBuilder, s: &[Variable; 4], lo: f64, hi: f64) -> Value {
    let u = emit_next_f64(fb, s);
    let loc = fb.ins().f64const(lo);
    let span = fb.ins().f64const(hi - lo);
    let scaled = fb.ins().fmul(span, u);
    fb.ins().fadd(loc, scaled)
}

fn emit_uniform_int(fb: &mut FunctionBuilder, s: &[Variable; 4], lo: f64, hi: f64) -> Value {
    let count = (hi - lo + 1.0).max(1.0);
    let u = emit_next_f64(fb, s);
    let nc = fb.ins().f64const(count);
    let m = fb.ins().fmul(u, nc);
    let fl = fb.ins().floor(m);
    let loc = fb.ins().f64const(lo);
    fb.ins().fadd(loc, fl)
}

/// `N(mu, sigma^2)` via Box–Muller, **one normal per lane**: draw two uniforms and keep the
/// cosine arm (the interpreter keeps both arms of a pair; per-lane we discard the sine one). The
/// extra uniform is cheap and avoids carrying a cross-lane cache. `sqrt` is a native instruction;
/// `ln`/`cos` are shims.
fn emit_normal(fb: &mut FunctionBuilder, s: &[Variable; 4], math: &MathRefs, mu: f64, sigma: f64) -> Value {
    use std::f64::consts::TAU;
    let one = fb.ins().f64const(1.0);
    let n1 = emit_next_f64(fb, s);
    let u1 = fb.ins().fsub(one, n1); // (0, 1] keeps ln finite
    let u2 = emit_next_f64(fb, s);
    let lnv = call1(fb, math.ln, u1);
    let neg2 = fb.ins().f64const(-2.0);
    let inner = fb.ins().fmul(neg2, lnv);
    let r = fb.ins().sqrt(inner);
    let tau = fb.ins().f64const(TAU);
    let ang = fb.ins().fmul(tau, u2);
    let c = call1(fb, math.cos, ang);
    let rc = fb.ins().fmul(r, c);
    let sig = fb.ins().f64const(sigma);
    let scaled = fb.ins().fmul(sig, rc);
    let mu_c = fb.ins().f64const(mu);
    fb.ins().fadd(mu_c, scaled)
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `Rng::fill_exp`.
fn emit_exp(fb: &mut FunctionBuilder, s: &[Variable; 4], math: &MathRefs, rate: f64) -> Value {
    let one = fb.ins().f64const(1.0);
    let u = emit_next_f64(fb, s);
    let om = fb.ins().fsub(one, u); // (0, 1]
    let lnv = call1(fb, math.ln, om);
    let neg = fb.ins().fneg(lnv);
    let rate_c = fb.ins().f64const(rate);
    fb.ins().fdiv(neg, rate_c)
}

/// `Geometric(p)` (failures before first success) via `floor(ln(1 - u) / ln(1 - p))` — mirrors
/// `Rng::fill_geometric`. `ln(1 - p)` is a compile-time constant (folded in Rust); `p == 1` makes
/// it `-inf`, so every lane floors to `0`, matching the interpreter's point mass.
fn emit_geometric(fb: &mut FunctionBuilder, s: &[Variable; 4], math: &MathRefs, p: f64) -> Value {
    let denom = (1.0 - p).ln();
    let one = fb.ins().f64const(1.0);
    let u = emit_next_f64(fb, s);
    let om = fb.ins().fsub(one, u); // (0, 1]
    let lnv = call1(fb, math.ln, om);
    let denom_c = fb.ins().f64const(denom);
    let q = fb.ins().fdiv(lnv, denom_c);
    fb.ins().floor(q)
}

/// `base ** k` for a small non-negative integer `k`, as repeated multiply (k = 0 → 1.0).
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

fn emit_unary(fb: &mut FunctionBuilder, math: &MathRefs, op: UnOp, a: Value) -> Value {
    match op {
        UnOp::Neg => fb.ins().fneg(a),
        UnOp::Not => {
            // logical not over a 0/1 column: (a == 0) ? 1 : 0
            let zero = fb.ins().f64const(0.0);
            let is_zero = fb.ins().fcmp(FloatCC::Equal, a, zero);
            bool_to_f64(fb, is_zero)
        }
        UnOp::Sin => call1(fb, math.sin, a),
        UnOp::Cos => call1(fb, math.cos, a),
        UnOp::Atan => call1(fb, math.atan, a),
        UnOp::Round => call1(fb, math.round, a),
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
    let r = if and { fb.ins().band(an, bn) } else { fb.ins().bor(an, bn) };
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
    use crate::sampler::moments;

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
        jit.reseed(seed);
        let cap = jit.batch_cap();
        // Drive a couple of batches by hand and accumulate mean to sanity-check the kernel runs.
        let mut sum = 0.0;
        let mut count = 0u64;
        for _ in 0..16 {
            for &x in jit.next_batch(cap) {
                sum += x;
                count += 1;
            }
        }
        let jit_mean = sum / count as f64;

        let interp_mean = moments(graph, id, count as usize, seed).mean;
        // Both estimate the same true mean; with ~16k samples they land close.
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
        assert_jit_matches_interp("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X**2 + Y**2 < 1", 4);
    }

    #[test]
    fn jit_continuous_sources_match_interp() {
        // B2: normal/exp/geometric are now emitted directly (ln/cos shims + native sqrt/floor).
        assert_jit_matches_interp("use rand; Z ~ normal(2,3); Z", 5);
        assert_jit_matches_interp("use rand; X ~ exp(2); X", 6);
        assert_jit_matches_interp("use rand; G ~ geometric(0.25); G", 7);
        // _int variants exercise the Round ufunc on top of a continuous source.
        assert_jit_matches_interp("use rand; Z ~ normal_int(10,3); Z", 8);
    }

    #[test]
    fn jit_ufuncs_and_nonconst_pow_match_interp() {
        // sin/cos/atan/sign ufuncs and a non-constant `**` (pow libcall) over an RV.
        assert_jit_matches_interp("use rand; use math; X ~ unif(0,1); sin(X) + cos(X)", 9);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-1,1); atan(X)", 10);
        assert_jit_matches_interp("use rand; use math; X ~ unif(-1,1); sign(X)", 11);
        assert_jit_matches_interp("use rand; A ~ unif(1,2); B ~ unif(1,2); A ** B", 12);
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
}
