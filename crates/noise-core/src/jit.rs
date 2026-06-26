//! Cranelift native JIT backend (PLAN.md Phase 4, steps "B1"/"B2" + multi-stream RNG).
//!
//! Compiles the sample-DAG into ONE fused machine-code kernel: a loop that draws its sources,
//! computes the whole expression keeping intermediates in registers, and stores the root `f64`s —
//! so, unlike the columnar interpreter, **no intermediate column is materialized to memory**. That
//! fusion is the win on arithmetic-dense graphs (see the `poly_*` benches).
//!
//! **Multi-stream RNG.** xoshiro256++ is a serial dependency chain — each `next_u64` waits on the
//! previous mutating the state, and because the state threads through every loop iteration the
//! *whole* loop is one chain. On RNG-bound graphs (`dice_sum`, `pi_indicator`) that latency, not
//! the arithmetic, is the ceiling. The kernel therefore runs [`STREAMS`] **independent** xoshiro
//! states, emitting that many samples per iteration; the out-of-order core overlaps the independent
//! chains for ~2× (measured — see `bench_streams`). This is the scalar form of SIMD, and it wins
//! where a hand-rolled `f64x2` Cranelift kernel lost (NEON's 2-wide ops — 3-instruction `rotl`, no
//! native `u64→f64` — couldn't beat what the OoO core already extracts from scalar code).
//!
//! The PRNG (xoshiro256++ / SplitMix64, mirroring `rng`) is **inlined into the generated code** —
//! no per-lane call back into Rust, which would otherwise dominate. Because the kernel consumes the
//! RNG per-stream rather than column-by-column like the interpreter, the JIT and interpreter agree
//! *in distribution* but not draw-for-draw under a shared seed; that's by design.
//!
//! Scope: `unif` / `unif_int` / `normal` / `exp` / `geometric` sources, `+ - * /`, integer-constant
//! `**`, comparisons, `&& ||`, unary `- !` and the math ufuncs, and lifted `if` (`Select`).
//! `Poisson` (Knuth's variable-length per-lane loop) stays interpreter-only; transcendental-bound
//! graphs that the interpreter samples faster also stay there (see [`crate::kernel::profitable`]).

use std::collections::HashMap;
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
use crate::kernel::{choose_streams, const_int_exponent, profitable, seed_state};

/// `extern "C"` signature of a generated kernel: `kernel(out_ptr, n, state_ptr)` fills `out[0..n]`
/// with fresh root draws, reading and writing the xoshiro state (`4 * STREAMS` words) via
/// `state_ptr`.
type KernelFn = unsafe extern "C" fn(*mut f64, i64, *mut u64);

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

/// Module-level `FuncId`s for the math shims (declared once per build).
struct MathIds {
    atan: cranelift_module::FuncId,
    round: cranelift_module::FuncId,
    pow: cranelift_module::FuncId,
}

/// In-function `FuncRef`s for the math shims (resolved once at the top of the kernel body, then
/// reused at every call site). `Copy`, so threading it through emission is free.
#[derive(Clone, Copy)]
struct MathRefs {
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
        // Use the JIT only where it's expected to *win* — see `kernel::profitable`. We inline the
        // transcendentals (`emit_ln`/`emit_trig`), so `inline_trans = true`: `normal`/`exp`/trig
        // graphs are fusible here and worth compiling. Only graphs still dominated by a real call
        // (`atan`/`round`/non-integer `pow`) stay on the interpreter. Any codegen failure also falls
        // back. Either way correctness is never at risk, only the speedup.
        if !profitable(graph, root, /* inline_trans */ true) {
            return InterpBackend.compile(graph, root);
        }
        match build(graph, root) {
            Ok(program) => Box::new(program),
            Err(_) => InterpBackend.compile(graph, root),
        }
    }
}

/// The immutable JIT artifact: the finalized module (owns the executable memory), the entry
/// pointer, and the stream count (so runners size their RNG state). Shared behind an `Arc`.
struct JitProgramInner {
    // `func` points into `_module`'s code memory; the module is kept alive for the program's life.
    _module: JITModule,
    func: KernelFn,
    streams: usize,
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

#[cfg(test)]
impl JitProgram {
    /// Number of independent RNG streams the kernel was built with — lets tests assert the
    /// multi-stream kernel emitted (no silent fallback to a narrower build).
    fn streams(&self) -> usize {
        self.inner.streams
    }
}

impl Program for JitProgram {
    fn runner(&self) -> Box<dyn Runner> {
        Box::new(JitRunner {
            inner: self.inner.clone(),
            buf: vec![0.0; BATCH],
            state: seed_state(0, self.inner.streams),
        })
    }
}

/// A per-worker JIT runner: a clone of the shared kernel `Arc`, its own output buffer, and the
/// xoshiro state (`4 * streams` words) carried across batches.
struct JitRunner {
    inner: Arc<JitProgramInner>,
    buf: Vec<f64>,
    state: Vec<u64>,
}

impl Runner for JitRunner {
    fn reseed(&mut self, seed: u64) {
        self.state = seed_state(seed, self.inner.streams);
    }

    fn next_batch(&mut self, len: usize) -> &[f64] {
        // Always fill the full BATCH (constant RNG consumption per call), then slice to `len`.
        let n = self.buf.len() as i64;
        // SAFETY: `func` is a finalized kernel with this exact ABI; `buf` holds `n` f64s and
        // `state` holds the `4 * streams`-word RNG state, both valid for the duration of the call.
        unsafe {
            (self.inner.func)(self.buf.as_mut_ptr(), n, self.state.as_mut_ptr());
        }
        &self.buf[..len]
    }

    fn batch_cap(&self) -> usize {
        BATCH
    }
}

/// Build the JIT module + kernel for `root` (the caller falls back to the interpreter on error),
/// choosing the stream count from the graph via [`choose_streams`].
fn build(graph: &RvGraph, root: RvId) -> Result<JitProgram, String> {
    build_with(graph, root, choose_streams(graph, root))
}

/// Build the kernel with `streams` independent xoshiro states interleaved per loop iteration.
/// `streams == 1` is the plain fused kernel; higher values overlap the RNG latency chains. The
/// skeleton is: load each stream's state → counted loop emitting `streams` samples per iteration →
/// store state back → return.
fn build_with(graph: &RvGraph, root: RvId, streams: usize) -> Result<JitProgram, String> {
    assert!(streams >= 1 && BATCH.is_multiple_of(streams), "streams must divide BATCH");

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
    let func_id =
        module.declare_function("kernel", Linkage::Export, &sig).map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, func_id.as_u32());

    // --- function body ---
    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        // Resolve the math shims into this function once; `MathRefs` is Copy, reused at each call.
        let math = MathRefs {
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

        // Loop counter `i` (a sample index, stepped by `streams`) and `streams` independent xoshiro
        // states, each four `I64` Variables. The OoO core overlaps the independent state chains.
        let i_var = fb.declare_var(types::I64);
        let states: Vec<[Variable; 4]> = (0..streams)
            .map(|_| {
                [
                    fb.declare_var(types::I64),
                    fb.declare_var(types::I64),
                    fb.declare_var(types::I64),
                    fb.declare_var(types::I64),
                ]
            })
            .collect();
        let zero_i = fb.ins().iconst(types::I64, 0);
        fb.def_var(i_var, zero_i);
        // Load stream `j` word `k` from the strided slot `state[k*streams + j]` (seed layout).
        for (j, st) in states.iter().enumerate() {
            for (k, v) in st.iter().enumerate() {
                let off = ((k * streams + j) * 8) as i32;
                let w = fb.ins().load(types::I64, MemFlags::trusted(), state_ptr, off);
                fb.def_var(*v, w);
            }
        }
        fb.ins().jump(header, &[]);

        // header: if i < n goto body else exit  (left unsealed — body adds the back-edge)
        fb.switch_to_block(header);
        let iv = fb.use_var(i_var);
        let cond = fb.ins().icmp(IntCC::SignedLessThan, iv, n);
        fb.ins().brif(cond, body, &[], exit, &[]);

        // body: for each stream emit the fused DAG (own memo — independent draws) and store the
        // result at out[i + j]; then i += streams, loop.
        fb.switch_to_block(body);
        fb.seal_block(body);
        let iv = fb.use_var(i_var);
        for (j, st) in states.iter().enumerate() {
            let mut memo: HashMap<RvId, Value> = HashMap::new();
            let result = emit_node(&mut fb, graph, root, st, &math, &mut memo);
            let idx = fb.ins().iadd_imm(iv, j as i64);
            let off = fb.ins().imul_imm(idx, 8);
            let addr = fb.ins().iadd(out, off);
            fb.ins().store(MemFlags::trusted(), result, addr, 0);
        }
        let inext = fb.ins().iadd_imm(iv, streams as i64);
        fb.def_var(i_var, inext);
        fb.ins().jump(header, &[]);
        fb.seal_block(header); // both preds (entry, body) now known

        // exit: write each stream's advanced state back to its slot, return.
        fb.switch_to_block(exit);
        fb.seal_block(exit);
        for (j, st) in states.iter().enumerate() {
            for (k, v) in st.iter().enumerate() {
                let off = ((k * streams + j) * 8) as i32;
                let w = fb.use_var(*v);
                fb.ins().store(MemFlags::trusted(), w, state_ptr, off);
            }
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

    Ok(JitProgram { inner: Arc::new(JitProgramInner { _module: module, func, streams }) })
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

// --- DAG → CLIF emission (per draw, in the loop body, for one stream `s`) ---

/// Emit the value of node `id` as an `f64` SSA value for stream `s`, memoizing by `RvId` so a
/// shared sub-RV (e.g. `X` in `X + X`) is emitted ONCE — the same CSE guarantee the interpreter
/// gets. (Each stream uses a fresh memo, since its draws are independent.)
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
        RvNode::Src(Source::Normal { mu, sigma }) => emit_normal(fb, s, *mu, *sigma),
        RvNode::Src(Source::Exp { rate }) => emit_exp(fb, s, *rate),
        RvNode::Src(Source::Geometric { p }) => emit_geometric(fb, s, *p),
        RvNode::Src(Source::Poisson { .. }) => unreachable!("supported() excludes Poisson"),
        RvNode::Gather { .. } => unreachable!("profitable() excludes Gather"),
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

/// `unif_int(lo, hi)` as `f64` via Lemire's multiply-high (`umulhi(bits, count)` = the top 64 bits
/// of the 128-bit product) — uniform in `0..count` with no `u64→f64`/`floor` round-trip, mirroring
/// `Rng::fill_uniform_int`. `count >= 1`, so `count == 1` always gives `k == 0` (point mass at `lo`).
fn emit_uniform_int(fb: &mut FunctionBuilder, s: &[Variable; 4], lo: f64, hi: f64) -> Value {
    let count = (hi - lo + 1.0).max(1.0) as u64;
    let bits = emit_next_u64(fb, s);
    let count_c = fb.ins().iconst(types::I64, count as i64);
    let k = fb.ins().umulhi(bits, count_c); // high 64 bits of bits * count → [0, count)
    let kf = fb.ins().fcvt_from_uint(types::F64, k); // k < count, small → exact
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

/// Inlined `ln(x)` for `x > 0` — straight-line transcription of [`crate::approx::ln`] (no libcall).
/// Removing the call is what lets `normal`/`exp`/`geometric` graphs run multi-stream (the libcall
/// previously serialized the lanes and pinned them to a single stream — see [`crate::kernel`]).
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

/// Inlined `cos(x)` (`is_cos`) / `sin(x)` for finite `x` — transcription of
/// [`crate::approx::cos`]/[`crate::approx::sin`]: Cody–Waite reduce to `[-π/4, π/4]`, evaluate both
/// reduced kernels, then pick by quadrant. `nearest`/`fcvt_to_sint_sat` are native instructions.
fn emit_trig(fb: &mut FunctionBuilder, x: Value, is_cos: bool) -> Value {
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

/// `N(mu, sigma^2)` via Box–Muller, **one normal per draw**: draw two uniforms and keep the cosine
/// arm (the interpreter keeps both arms of a pair; per-draw we discard the sine one). The extra
/// uniform is cheap and avoids carrying a cross-draw cache. `sqrt` is native; `ln`/`cos` are
/// inlined polynomials ([`emit_ln`]/[`emit_trig`]).
fn emit_normal(fb: &mut FunctionBuilder, s: &[Variable; 4], mu: f64, sigma: f64) -> Value {
    use std::f64::consts::TAU;
    let one = fb.ins().f64const(1.0);
    let n1 = emit_next_f64(fb, s);
    let u1 = fb.ins().fsub(one, n1); // (0, 1] keeps ln finite
    let u2 = emit_next_f64(fb, s);
    let lnv = emit_ln(fb, u1);
    let neg2 = fb.ins().f64const(-2.0);
    let inner = fb.ins().fmul(neg2, lnv);
    let r = fb.ins().sqrt(inner);
    let tau = fb.ins().f64const(TAU);
    let ang = fb.ins().fmul(tau, u2);
    let c = emit_trig(fb, ang, true);
    let rc = fb.ins().fmul(r, c);
    let sig = fb.ins().f64const(sigma);
    let scaled = fb.ins().fmul(sig, rc);
    let mu_c = fb.ins().f64const(mu);
    fb.ins().fadd(mu_c, scaled)
}

/// `Exp(rate)` via inverse-CDF `-ln(1 - u) / rate` — mirrors `Rng::fill_exp`.
fn emit_exp(fb: &mut FunctionBuilder, s: &[Variable; 4], rate: f64) -> Value {
    let one = fb.ins().f64const(1.0);
    let u = emit_next_f64(fb, s);
    let om = fb.ins().fsub(one, u); // (0, 1]
    let lnv = emit_ln(fb, om);
    let neg = fb.ins().fneg(lnv);
    let rate_c = fb.ins().f64const(rate);
    fb.ins().fdiv(neg, rate_c)
}

/// `Geometric(p)` (failures before first success) via `floor(ln(1 - u) / ln(1 - p))` — mirrors
/// `Rng::fill_geometric`. `ln(1 - p)` is a compile-time constant (folded in Rust); `p == 1` makes
/// it `-inf`, so every draw floors to `0`, matching the interpreter's point mass.
fn emit_geometric(fb: &mut FunctionBuilder, s: &[Variable; 4], p: f64) -> Value {
    let denom = (1.0 - p).ln();
    let one = fb.ins().f64const(1.0);
    let u = emit_next_f64(fb, s);
    let om = fb.ins().fsub(one, u); // (0, 1]
    let lnv = emit_ln(fb, om);
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
        UnOp::Sin => emit_trig(fb, a, false),
        UnOp::Cos => emit_trig(fb, a, true),
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
    use crate::kernel::supported;
    use crate::kernel::STREAMS;

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

    /// The default kernel must actually interleave `STREAMS` RNG streams (not silently build a
    /// 1-stream kernel) — that's where the RNG-bound speedup comes from.
    #[test]
    fn default_kernel_is_multi_stream() {
        let mut eng = crate::Engine::new();
        let id = match eng
            .run_rv("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B")
            .unwrap()
        {
            crate::Value::Dist(id) => id,
            _ => unreachable!(),
        };
        let program = build(eng.graph(), id).expect("jit build failed");
        assert_eq!(program.streams(), STREAMS);
    }

    /// The stream count must not change the distribution: a 1-stream and an `STREAMS`-stream kernel
    /// estimate the same moments (each stream is an independent, identically-distributed substream).
    #[test]
    fn stream_count_preserves_distribution() {
        let cases = [
            ("use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B", 7.0),
            ("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X**2 + Y**2 < 1", std::f64::consts::FRAC_PI_4),
        ];
        for (src, expected) in cases {
            let mut eng = crate::Engine::new();
            let id = match eng.run_rv(src).unwrap() {
                crate::Value::Dist(id) => id,
                _ => unreachable!(),
            };
            for streams in [1usize, STREAMS] {
                let program = build_with(eng.graph(), id, streams).expect("jit build failed");
                let mut runner = program.runner();
                runner.reseed(0xABCDEF);
                let cap = runner.batch_cap();
                let (mut sum, mut count) = (0.0f64, 0u64);
                for _ in 0..256 {
                    for &x in runner.next_batch(cap) {
                        sum += x;
                        count += 1;
                    }
                }
                let mean = sum / count as f64;
                assert!(
                    (mean - expected).abs() < 0.02,
                    "{src} @ {streams} streams: mean={mean}, want≈{expected}"
                );
            }
        }
    }

    /// Same-process A/B of kernel throughput by stream count, single-threaded — the clean
    /// measurement of the multi-stream win (no multicore / criterion-baseline noise). Ignored by
    /// default; run with:
    /// `cargo test -p noise-core --features jit --release -- --ignored --nocapture bench_streams`
    #[test]
    #[ignore]
    fn bench_streams() {
        use std::time::Instant;

        fn drive(program: &JitProgram, batches: usize) -> f64 {
            let mut runner = program.runner();
            runner.reseed(0xC0FFEE);
            let cap = runner.batch_cap();
            for _ in 0..64 {
                runner.next_batch(cap);
            }
            let t = Instant::now();
            let mut acc = 0.0f64;
            for _ in 0..batches {
                let col = runner.next_batch(cap);
                acc += col[0] + col[cap - 1];
            }
            std::hint::black_box(acc);
            (batches * cap) as f64 / t.elapsed().as_secs_f64() / 1e6
        }

        let cases = [
            ("pi_indicator", "use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X**2 + Y**2 < 1"),
            ("dice_sum", "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B"),
            ("poly_deep", "use rand; X ~ unif(0,1); ((X*X+X)*X - X)*X + X*X - X + 1"),
            ("normal_poly", "use rand; Z ~ normal(0,1); ((Z*Z+Z)*Z - Z)*Z + Z*Z"),
            // Transcendental-bound now that ln/cos are inlined: the multi-stream win should appear.
            ("normal_sum", "use rand; X ~ normal(0,1); Y ~ normal(0,1); X + Y"),
            ("exp_tail", "use rand; X ~ exp(2); X > 1"),
            ("sin_wave", "use rand; use math; X ~ unif(0,1); sin(6.283*X) + cos(6.283*X)"),
        ];
        let batches = 4000;
        println!("\n  kernel throughput by stream count (single thread, M elem/s):");
        for (name, src) in cases {
            let mut eng = crate::Engine::new();
            let id = match eng.run_rv(src).unwrap() {
                crate::Value::Dist(id) => id,
                _ => unreachable!(),
            };
            let g = eng.graph();
            print!("    {name:14}");
            for streams in [1usize, 2, 4, 8] {
                let p = build_with(g, id, streams).expect("build");
                print!("  s{streams}={:6.0}", drive(&p, batches));
            }
            println!();
        }
    }

    /// Codegen-quality probe: race the Cranelift JIT against a hand-written, LLVM-compiled fused
    /// kernel computing the *identical* graph (same inlined xoshiro, same arithmetic). Isolates
    /// Cranelift's codegen vs LLVM's. Ignored by default; run with:
    /// `cargo test -p noise-core --features jit --release -- --ignored --nocapture bench_cranelift_vs_llvm`
    #[test]
    #[ignore]
    fn bench_cranelift_vs_llvm() {
        use crate::rng::Rng;
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
        runner.reseed(0xC0FFEE);
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
        let mut rng = Rng::seed_from_u64(0xC0FFEE);
        let mut buf = vec![0.0f64; cap];
        for _ in 0..64 {
            for x in buf.iter_mut() {
                *x = rng.next_f64();
            }
        }
        let t = Instant::now();
        let (mut acc2, mut done2) = (0.0f64, 0usize);
        while done2 < n {
            for slot in buf.iter_mut() {
                let x = rng.next_f64();
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
