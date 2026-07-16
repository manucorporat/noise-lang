//! The WebGPU backend's driver seam (PLAN-WEBGPU G2) — native, behind `--features gpu`.
//!
//! **Why this doesn't hook into `Runner` like the other three backends do.** `Runner::next_batch` is
//! a synchronous pull of 1024 lanes. Both halves are wrong for a GPU: a dispatch wants ≥256k lanes to
//! be worth its ~1.2 ms fixed latency, and per-1024-lane dispatches would pay that floor ~250×. So the
//! GPU hooks one level *up*, in [`crate::reduce::run_reduction`]: if the gate accepts, this drives the
//! whole forcing itself — dispatching big lane ranges, folding each 16,384-sample chunk into the
//! caller's `Reducer` in chunk order, and handing back the accumulator. `Program` / `Runner` /
//! `Reducer` are all untouched, and the interpreter/wasm paths never see this module.
//!
//! Counter keying is what makes that legal: a chunk is *just a lane range* (the draw at lane `i` is a
//! pure function of `(seed, i, source)`), so the GPU can compute any range independently and the fold
//! stays chunk-ordered and deterministic — the same guarantee the threaded CPU reducer gives.
//!
//! Every failure path falls back to a CPU backend rather than erroring: no adapter, an unsupported
//! cone, a shader that won't compile, a device loss. The GPU may only ever change *speed*.
//!
//! **Two backends behind one driver (G3).** Everything above the dispatch — simplify, WGSL emit, the
//! cost gate, the chunk-ordered fold — is portable and shared. Only the three primitives that touch a
//! real device differ by target: [`available`], [`prepare`] (compile/cache a pipeline), and
//! [`dispatch`] (run a lane range, read back `cols × n` f32). Native drives `wgpu` directly and blocks
//! on a poll (G0–G2). wasm can't block on the GPU — WebGPU is async and a worker that waits on its own
//! `mapAsync` self-deadlocks — so the wasm backend hands the shader + params to the **main thread**
//! over a `SharedArrayBuffer` and blocks on `Atomics.wait` until it writes the result column back
//! (G3). The fold stays in wasm either way, so the answer is bit-identical to native.

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::dist::{RvGraph, RvId};
use crate::error::{NoiseError, Result};
use crate::exec::CancelToken;
use crate::reduce::{MomentsMode, Reduced, Reducer};
use crate::wgsl_emit;
use web_time::Instant;

/// Lanes per dispatch. Big enough to bury the ~1.2 ms fixed cost of a dispatch + readback (G0), and a
/// whole number of 16,384-sample reducer chunks, so the fold below is chunk-for-chunk identical to
/// what the CPU reducer would have produced. 1M lanes is a 4 MB readback.
const GPU_DISPATCH: usize = 1 << 20;

/// Output-buffer element ceiling for the **joint** driver (D4b): a joint dispatch reads back `k × n`
/// f32 (one column per root), so with many roots the per-dispatch lane count must shrink to keep the
/// readback bounded. 8 M elements is a 32 MB readback — comfortable, and the joint introspection/plot
/// passes have moderate `n` (40k–200k), so a `k`-column forcing is usually one or a few dispatches.
const GPU_JOINT_ELEMS: usize = 8 << 20;

/// The **joint** gate's extra term (D4b), on top of the shared [`profitable`] one. A joint forcing
/// reads back and folds `k × n` f32 (one column per root), a cost the single-root gate never sees; the
/// GPU only wins when the per-lane *compute* it saves (∝ `ops × n`) dominates that readback+fold
/// (∝ `k × n`) — i.e. when the **per-root** cone is fat, `ops / k` large. Measured on the corpus:
/// the winners are fat plots (am_vs_fm `plot::line` at ops/draw 567–882 over k≈64 → ~9–14 ops/root;
/// barrier `plot::fan` 315 over k≈52 → ~6), the losers are wide-but-thin passes (nyquist/birthday at
/// ops/draw ~60 over k≈60 → ~1 op/root) whose ~30–40 ms readback+fold buried a <5 ms CPU pass. 4 sits
/// cleanly between. Below it the joint pass declines to the (batch-streaming, no-readback) interpreter.
const JOINT_MIN_OPS_PER_ROOT: u64 = 4;

/// The reducer's chunk size — shared with `reduce` so the GPU folds on exactly the same boundaries
/// (the index-ordered fold's determinism guarantee is about *chunks*, not threads).
const CHUNK_SAMPLES: usize = crate::reduce::CHUNK_SAMPLES;

/// Reduce-mode dispatch sizing (PLAN-PRECISION Track F): with the per-lane readback gone, bigger
/// dispatches only amortize the fixed dispatch cost better — what bounds them is GPU *time*, since
/// the soft-stop/deadline check sits between dispatches. One dispatch is sized to roughly this many
/// cost-model units (`ops + ~150·sources` per lane, the same proxy [`emitted_instrs`] uses for the
/// hash emulation), clamped to [`REDUCE_DISPATCH_MIN`]..[`REDUCE_DISPATCH_MAX`] lanes and
/// chunk-aligned. On an M4 Pro this lands around 5–20 ms per dispatch across the corpus's cones.
const REDUCE_WORK_PER_DISPATCH: f64 = 2e9;
/// Never dispatch fewer lanes than this in reduce mode (16 chunks): below it the fixed dispatch
/// cost dominates whatever the sizing model hoped to save.
const REDUCE_DISPATCH_MIN: usize = 16 * CHUNK_SAMPLES;
/// …and never more than this (16M lanes — still only a 192 KB readback of partials).
const REDUCE_DISPATCH_MAX: usize = 1 << 24;

/// Below this many lanes a moments forcing stays on the **column** path even though reduce mode
/// could fold it. Each reduce-mode thread folds 64 lanes sequentially, so a small-n forcing fills
/// only `n/64` threads — a 15k-lane dispatch runs on 256 threads. Measured (`noise_colors`, 67k
/// ops/draw at n=15,000) the two modes tie anyway (~3.5 s warm — that cone is ALU-bound, not
/// occupancy-bound), but column readback is `4·n` bytes, which is exactly the term that is *cheap*
/// at small n — so small forcings keep the calibrated, battle-tested column economics and reduce
/// mode takes over where its readback win is real. 1M lanes = a 16,384-thread reduce dispatch and
/// 4 MB of column readback: comfortably past both concerns.
const REDUCE_MIN_LANES: usize = 1 << 20;

/// Emitted-instruction ceiling. **Not a node cap** — the unit that matters is what the *shader
/// compiler* sees, and each RNG draw call inlines ~150 ALU ops of emulated `squares64` (WGSL has no
/// `u64`). G0 measured cold pipeline compile against exactly this quantity: ~5k instructions is
/// ~325 ms, 17.6k is ~1.9 s, 45k is ~8.9 s. Past this ceiling the compile can no longer pay for
/// itself no matter how many draws follow it.
///
/// **G4 sharpened this: a long chain of *statements* can be as costly as the sources.** `prisoners`
/// lowers now (its permutation and gathers are supported), but its 100×50 cycle-following unrolls
/// into ~15,000 data-dependent `ArrIndex` reads — one gigantic dependent basic block, on which the
/// Metal compiler goes *super-linear*: 2.2 s cold, versus 127 ms for an 12k-instruction shader with
/// ordinary parallelism. So the cap earns its keep on statement volume too, not only on draw calls,
/// and `prisoners` is correctly declined — it needs the cycle loop *re-rolled* (structured control
/// flow the IR doesn't yet carry past graph-build), which is G4c, not just node support.
const MAX_WGSL_INSTRS: usize = 8_000;

/// **The gate, calibrated against the corpus** (`example_times`, M4 Pro, `--features jit,gpu`).
///
/// The discriminator turned out not to be the total work — it is the **cone size per draw**, and the
/// corpus separates on it with no overlap at all:
///
/// | example | ops/draw | GPU vs multicore JIT |
/// |---|---|---|
/// | `secretary` | 124 | **11.7×** |
/// | `barrier_option` | 401 | **4.3×** |
/// | `birthday` | 784 | **2.4×** |
/// | `am_vs_fm` | 845 | **2.0×** |
/// | `noise_colors` | 2,035 | **1.65×** |
/// | `st_petersburg` | 58 | 0.86× |
/// | `beta_bernoulli` | 37 | 0.57× |
/// | `bootstrap` | 30 | 0.98× |
/// | `kelly` | 6 | 0.42× |
/// | `conditional_bayes` | 5 | 0.26× |
///
/// That is exactly the plan's thesis, arrived at from the other end: a fat cone is a lane's worth of
/// independent ALU work, which is what a GPU is *for*, and it amortizes the dispatch + compile over
/// the cone rather than over the draw count. A thin cone (`kelly`: six ops and a couple of uniforms)
/// is pure RNG-and-memory, where a warmed-up multicore CPU is simply hard to beat and the pipeline
/// compile can never be earned back.
///
/// **Recalibrated against the interpreter floor** (PLAN-DROP-JIT D4a). The gate was tuned against the
/// retired multicore JIT, whose floor was ~1.18× faster; against the interpreter the GPU wins on
/// *thinner* cones than 100. The measured band (M4 Pro, gated-vs-`NOISE_FORCE_GPU` confusion matrix,
/// per forcing at n=1M): `bootstrap` tops out at 41 ops/draw and loses on the GPU (its CPU reduce is
/// ~1–3.5 ms — cheap gathers), while `beta_bernoulli` starts at 47 and wins (~8 ms CPU reduce —
/// transcendental-heavy, so its true per-op cost is higher than the node count shows). 45 sits in
/// that empty band, recovering `beta_bernoulli` (−22 ms) without accepting `bootstrap`/`kelly`/
/// `conditional_bayes`/`dithering` (all ≤ 41 ops/draw). The `MIN_WORK_GPU` term still guards small-n.
const MIN_CONE_OPS: u64 = 45;

/// …and enough total work to pay for the pipeline compile at all. `noise_colors` is the tightest
/// winner (2,035 ops/draw but only 3,000 draws per forcing → 6.1e6) and still returns 1.65×.
const MIN_WORK_GPU: f64 = 3e6;

/// Estimate what a shader costs the *compiler*: one unit per statement, plus ~150 for every draw call
/// (the inlined, emulated `squares64` — WGSL has no `u64`). Counted off the emitted text, so it can't
/// drift from what is actually handed to the driver — and a shaped draw's loop contributes ONE call
/// however wide it is, which is the entire point of `ArrDraw`.
fn emitted_instrs(wgsl: &str) -> usize {
    let body = wgsl.split("fn main").nth(1).unwrap_or("");
    body.matches("src_").count() * 150 + body.matches(";\n").count()
}

/// Whether the GPU is expected to beat the CPU on this forcing, compile included. See
/// [`MIN_CONE_OPS`] for the calibration this is built on.
///
/// Three terms, because a GPU forcing has three ways to lose: a cone too big to compile, a cone too
/// *thin* to be worth dispatching, and a forcing too short to earn the compile back. The **configurable
/// cost model** ([`crate::kernel::prefer_runtime`]) drops that last (amortization) term for an
/// interactive host that reuses the pipeline across runs — a fat cone then goes to the GPU even for a
/// short forcing, because its *runtime* wins and the compile is paid once. The feasibility cap
/// (`MAX_WGSL_INSTRS`) and the runtime term (`MIN_CONE_OPS`) always hold: a shader that takes seconds
/// to compile blocks the first run regardless, and a thin cone is slower on the GPU however it is used.
fn profitable(instrs: usize, ops_per_draw: u64, n: usize) -> bool {
    let work_ok =
        crate::kernel::prefer_runtime() || (n as f64 * ops_per_draw as f64) >= MIN_WORK_GPU;
    instrs <= MAX_WGSL_INSTRS && ops_per_draw >= MIN_CONE_OPS && work_ok
}

/// The **reduce-mode** gate's extra acceptance region (PLAN-PRECISION Track F). [`MIN_CONE_OPS`]
/// was a readback-era number: its calibration measured a 4-byte-per-lane readback plus a CPU-side
/// fold of every lane, which is why thin cones could never win at any `n`. Reduce mode reads back
/// 12 bytes per 4096 lanes and folds ~nothing, so a thin cone becomes purely an **amortization**
/// question: enough total work to pay the pipeline compile and beat the multicore CPU outright.
///
/// Calibrated on M4 Pro (release, pi's ~7-op cone — `bench_thin_cone_gpu` /
/// `bench_thin_cone_cpu`): warm-pipeline the GPU wins from n ≈ 1M (3.0 vs 4.3 ms) and holds ~9× at
/// 256M (109 vs 993 ms); a **fully cold** pipeline compile costs ~0.5 s (Metal's on-disk shader
/// cache usually absorbs it after the first-ever run, but the gate must price the worst case). The
/// strict floor sits where the CPU alternative costs several × the cold compile; `prefer_runtime`
/// (interactive hosts that keep pipelines alive) uses the warm crossover instead.
const MIN_WORK_GPU_REDUCE: f64 = 1e9;
/// The warm-pipeline (interactive-host) floor for reduce mode — dispatch overhead + the CPU's
/// small-n advantage, no compile term.
const MIN_WORK_REDUCE_RUNTIME: f64 = 1e7;

/// Whether reduce mode is expected to beat the CPU on this forcing. Everything the column gate
/// accepted stays accepted (reduce mode strictly dominates it — same shader body, ~no readback), and
/// thin cones additionally win once total work clears the amortization floor above.
fn profitable_reduce(instrs: usize, ops_per_draw: u64, n: usize) -> bool {
    if profitable(instrs, ops_per_draw, n) {
        return true;
    }
    let floor = if crate::kernel::prefer_runtime() {
        MIN_WORK_REDUCE_RUNTIME
    } else {
        MIN_WORK_GPU_REDUCE
    };
    instrs <= MAX_WGSL_INSTRS && (n as f64 * ops_per_draw as f64) >= floor
}

/// The reduce-mode gate decision, with the failing term (`NOISE_PROFILE=1`). Mirrors
/// [`profitable_reduce`] exactly.
fn gate_reason_reduce(instrs: usize, ops_per_draw: u64, n: usize) -> String {
    let runtime = crate::kernel::prefer_runtime();
    let floor = if runtime {
        MIN_WORK_REDUCE_RUNTIME
    } else {
        MIN_WORK_GPU_REDUCE
    };
    if instrs > MAX_WGSL_INSTRS {
        format!("gate(reduce): DECLINE — cone too big ({instrs} instrs > {MAX_WGSL_INSTRS})")
    } else if profitable(instrs, ops_per_draw, n) {
        format!(
            "gate(reduce): ACCEPT (column-gate terms) — {instrs} instrs, {ops_per_draw} ops/draw, {n} draws"
        )
    } else if (n as f64 * ops_per_draw as f64) < floor {
        format!(
            "gate(reduce): DECLINE — work too small ({:.2e} < {floor:.0e})",
            n as f64 * ops_per_draw as f64
        )
    } else {
        let mode = if runtime { " (prefer-runtime)" } else { "" };
        format!("gate(reduce): ACCEPT{mode} — {instrs} instrs, {ops_per_draw} ops/draw, {n} draws")
    }
}

/// The gate decision with the reason the failing term (for `NOISE_PROFILE=1`, PLAN-DROP-JIT D0): the
/// D4a recalibration needs to see *which* of the three terms declines each forcing, not just that one
/// did. Mirrors [`profitable`] exactly.
fn gate_reason(instrs: usize, ops_per_draw: u64, n: usize) -> String {
    if instrs > MAX_WGSL_INSTRS {
        format!("gate: DECLINE — cone too big ({instrs} instrs > {MAX_WGSL_INSTRS})")
    } else if ops_per_draw < MIN_CONE_OPS {
        format!("gate: DECLINE — cone too thin ({ops_per_draw} ops/draw < {MIN_CONE_OPS})")
    } else if !crate::kernel::prefer_runtime() && (n as f64 * ops_per_draw as f64) < MIN_WORK_GPU {
        format!(
            "gate: DECLINE — work too small ({:.2e} < {MIN_WORK_GPU:.0e})",
            n as f64 * ops_per_draw as f64
        )
    } else {
        let mode = if crate::kernel::prefer_runtime() {
            " (prefer-runtime)"
        } else {
            ""
        };
        format!("gate: ACCEPT{mode} — {instrs} instrs, {ops_per_draw} ops/draw, {n} draws")
    }
}

/// Force the GPU regardless of the gate — for the benchmark harness, which needs to measure what the
/// GPU *would* do on cones the gate declines on cost grounds. Never set in normal operation.
static FORCE: OnceLock<bool> = OnceLock::new();
pub fn force_gpu() {
    let _ = FORCE.set(true);
}
fn forced() -> bool {
    // `NOISE_FORCE_GPU=1` is how the benchmark harness measures what the GPU *would* do on cones the
    // gate declines on cost grounds — the data the gate itself is calibrated from.
    static ENV: OnceLock<bool> = OnceLock::new();
    *FORCE.get().unwrap_or(&false)
        || *ENV.get_or_init(|| std::env::var("NOISE_FORCE_GPU").is_ok_and(|v| v == "1"))
}

// ---------------------------------------------------------------------------
// Native backend: `wgpu` device + pipeline cache (process-wide: acquiring a device is slow, and a
// compiled pipeline is exactly the thing G0 says we must not pay for twice).
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Compiled pipelines, keyed by the shader text itself — which is a complete description of the
    /// artifact, so this is a content-addressed cache and can never serve a stale kernel.
    pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
}

/// `None` on a machine with no usable adapter — the caller then simply uses a CPU backend.
#[cfg(not(target_arch = "wasm32"))]
fn device() -> Option<&'static Device> {
    static DEV: OnceLock<Option<Device>> = OnceLock::new();
    DEV.get_or_init(|| {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some(Device {
            device,
            queue,
            pipelines: Mutex::new(HashMap::new()),
        })
    })
    .as_ref()
}

#[cfg(not(target_arch = "wasm32"))]
impl Device {
    /// Whether the pipeline for `wgsl` is already compiled (for the `NOISE_PROFILE=1` hit/miss note,
    /// PLAN-DROP-JIT D0). A pure probe — it never compiles.
    fn pipeline_cached(&self, wgsl: &str) -> bool {
        self.pipelines.lock().is_ok_and(|p| p.contains_key(wgsl))
    }

    /// Compile (or reuse) the pipeline for `wgsl`. `None` if the driver rejects the shader — which
    /// would be our bug, but is still a fallback rather than a crash in a user's face.
    fn pipeline(&self, wgsl: &str) -> Option<wgpu::ComputePipeline> {
        if let Some(p) = self.pipelines.lock().ok()?.get(wgsl) {
            return Some(p.clone());
        }
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if pollster::block_on(self.device.pop_error_scope()).is_some() {
            return None;
        }
        self.pipelines
            .lock()
            .ok()?
            .insert(wgsl.to_string(), pipeline.clone());
        Some(pipeline)
    }

    /// Dispatch lanes `lane0 .. lane0 + n` and read back what the shader wrote — see
    /// [`dispatch_shape`] for the two output layouts (`cols >= 1` column mode, `cols == 0` reduce
    /// mode).
    fn dispatch(
        &self,
        pipe: &wgpu::ComputePipeline,
        key: crate::rng::Key,
        lane0: u32,
        n: u32,
        cols: u32,
    ) -> Vec<f32> {
        let (out_len, workgroups) = dispatch_shape(n, cols);
        let bytes = out_len as u64 * 4;
        let params: [u32; 4] = [key.k0, key.k1, lane0, n];
        // SAFETY: `[u32; 4]` is 16 contiguous bytes, no padding; every bit pattern is a valid `u8`.
        let pbytes = unsafe { std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16) };

        let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf, 0, pbytes);
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, bytes);
        {
            let _s = crate::profile::span("gpu.dispatch");
            self.queue.submit([enc.finish()]);
        }

        // Readback: map + blocking poll + copy out. Native uses a blocking poll — this is why G0–G2
        // need no async at all; the browser host (G3) is the only thing that can't block, and the only
        // consumer of an async spine.
        let _s = crate::profile::span("gpu.readback");
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let data = slice.get_mapped_range();
        let col: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        drop(data);
        staging.unmap();
        col
    }
}

// ---------------------------------------------------------------------------
// Backend seam. `try_reduce` / `try_joint` below are target-agnostic; these three primitives are the
// only device-touching operations, implemented once per target.
// ---------------------------------------------------------------------------

/// The output length and workgroup count of one dispatch, from the `cols` word that already rides
/// the wasm bridge (`gpu-host.ts` mirrors this function — if one drifts, the answers diverge):
///
/// * `cols >= 1` — **column mode**: one thread per lane, `cols × n` f32 out
///   (root `j`'s lane `i` at `out[j*n + i]`, the layout [`wgsl_emit::emit`] writes).
/// * `cols == 0` — **reduce mode** (Track F): one thread per [`wgsl_emit::REDUCE_LANES_PER_THREAD`]
///   lanes, one `(Σx, Σx², count)` triple per workgroup ([`wgsl_emit::emit_reduce`]'s layout).
fn dispatch_shape(n: u32, cols: u32) -> (usize, u32) {
    if cols == 0 {
        let wgs = n.div_ceil(wgsl_emit::REDUCE_WG_LANES);
        (wgs as usize * wgsl_emit::REDUCE_COLS as usize, wgs)
    } else {
        (cols as usize * n as usize, n.div_ceil(wgsl_emit::WORKGROUP))
    }
}

/// A compiled, ready-to-dispatch pipeline. Native carries the `wgpu` object; wasm carries the shader
/// text (the main-thread host is keyed on it — a content-addressed pipeline cache lives over there).
#[cfg(not(target_arch = "wasm32"))]
struct Prepared(wgpu::ComputePipeline);
#[cfg(target_arch = "wasm32")]
struct Prepared(String);

/// Is a GPU backend usable on this thread? Native: an adapter exists. wasm: the JS host installed a
/// device (`crossOriginIsolated && navigator.gpu`, main-thread device acquired) — 0 on a rayon
/// sub-worker or a non-isolated page, which then declines to the CPU.
#[cfg(not(target_arch = "wasm32"))]
fn available() -> bool {
    device().is_some()
}
#[cfg(target_arch = "wasm32")]
fn available() -> bool {
    nz_gpu_available() == 1
}

/// Compile (or reuse) the pipeline for `wgsl`, returning it plus whether it was already cached (for
/// the profile hit/miss note). `None` when the driver rejects the shader — a decline, never a fail.
#[cfg(not(target_arch = "wasm32"))]
fn prepare(wgsl: &str) -> Option<(Prepared, bool)> {
    let dev = device()?;
    let hit = dev.pipeline_cached(wgsl);
    Some((Prepared(dev.pipeline(wgsl)?), hit))
}
#[cfg(target_arch = "wasm32")]
fn prepare(wgsl: &str) -> Option<(Prepared, bool)> {
    // The host compiles + caches the pipeline on the main thread; a shader it rejects declines here,
    // before the fold loop — the same "decline, never fail" contract as native. The hit/miss bit is
    // the host's, not ours; report miss (the browser profile path doesn't rely on it).
    (nz_gpu_prepare(wgsl) == 1).then(|| (Prepared(wgsl.to_string()), false))
}

/// Dispatch lanes `lane0 .. lane0 + n` and read back what the shader wrote — [`dispatch_shape`]
/// gives the layout per `cols` (`>= 1` column mode, `== 0` reduce mode).
#[cfg(not(target_arch = "wasm32"))]
fn dispatch(prep: &Prepared, key: crate::rng::Key, lane0: u32, n: u32, cols: u32) -> Vec<f32> {
    device()
        .expect("available() was true")
        .dispatch(&prep.0, key, lane0, n, cols)
}
#[cfg(target_arch = "wasm32")]
fn dispatch(prep: &Prepared, key: crate::rng::Key, lane0: u32, n: u32, cols: u32) -> Vec<f32> {
    // One span covers the whole round-trip — write request, wake the main thread, block on
    // `Atomics.wait`, copy the column back — since the worker is blocked for all of it (the native
    // path splits this into `gpu.dispatch` + `gpu.readback`, which don't exist as phases here).
    let _s = crate::profile::span("gpu.dispatch");
    let (out_len, _wgs) = dispatch_shape(n, cols);
    let mut out = vec![0.0f32; out_len];
    let ok = nz_gpu_dispatch(&prep.0, &mut out, n, cols, key.k0, key.k1, lane0);
    debug_assert_eq!(
        ok, 1,
        "nz_gpu_dispatch failed after prepare succeeded (device loss?)"
    );
    out
}

/// The wasm bridge to the main-thread WebGPU host. A tiny `inline_js` shim forwards each call to
/// `globalThis.__noiseGpu`, which `worker.ts` installs on the engine worker (and leaves unset on the
/// rayon sub-workers, so `available()` is 0 there). Keeping the SAB / `postMessage` / `Atomics.wait`
/// dance in TS (not here) is deliberate: the browser-only, iterate-in-devtools half stays in JS.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
export function nz_gpu_available() { const g = globalThis.__noiseGpu; return g ? g.available() : 0; }
export function nz_gpu_prepare(wgsl) { const g = globalThis.__noiseGpu; return g ? g.prepare(wgsl) : 0; }
export function nz_gpu_dispatch(wgsl, out, n, cols, k0, k1, lane0) {
  const g = globalThis.__noiseGpu;
  return g ? g.dispatch(wgsl, out, n, cols, k0, k1, lane0) : 0;
}
"#)]
extern "C" {
    fn nz_gpu_available() -> i32;
    fn nz_gpu_prepare(wgsl: &str) -> i32;
    fn nz_gpu_dispatch(
        wgsl: &str,
        out: &mut [f32],
        n: u32,
        cols: u32,
        k0: u32,
        k1: u32,
        lane0: u32,
    ) -> i32;
}

/// The outcome of offering a forcing to the GPU: it either ran the whole range ([`Done`]) or
/// declined and handed the caller's carry accumulator back untouched ([`Declined`]) — the caller
/// then proceeds exactly as if this module did not exist, so a decline is never an error and never
/// changes an answer.
///
/// [`Done`]: GpuReduce::Done
/// [`Declined`]: GpuReduce::Declined
pub enum GpuReduce<A> {
    Done(Reduced<A>),
    Declined(A),
}

/// Try to run this whole forcing (epoch-local lanes `lanes`, PLAN-PRECISION Track A/E) on the GPU,
/// folding into `carry` on the reducer's own chunk boundaries.
///
/// `Err` is reserved for **hard cancellation**, which must propagate: a cancelled forcing has
/// folded only some of its chunks, and that partial answer must never escape as though it were an
/// estimate. A **soft** stop (the token's soft flag, or the run deadline — checked between
/// dispatches, ~1M-lane granularity) folds what was read back and returns it with its cause,
/// exactly like the CPU driver (Track H).
#[allow(clippy::too_many_arguments)]
pub fn try_reduce<R: Reducer>(
    graph: &RvGraph,
    root: RvId,
    lanes: std::ops::Range<u64>,
    seed: u64,
    r: &R,
    token: Option<&CancelToken>,
    deadline: Option<Instant>,
    carry: R::Acc,
) -> Result<GpuReduce<R::Acc>> {
    if !available() {
        crate::profile::note("gpu: no adapter → cpu");
        return Ok(GpuReduce::Declined(carry));
    }

    // The same simplify the other backends get, so the cone — and therefore the draw ordinals — are
    // identical to what the interpreter would compute.
    let (g, root) = {
        let _s = crate::profile::span("gpu.simplify");
        crate::simplify::simplify(graph, root)
    };

    // A big-n moments fold (P/E/Var and their conditional twins) runs in **reduce mode** (Track F):
    // the fold happens in the shader and only per-workgroup partials come back, which removes the
    // readback term that made thin cones GPU-hostile at scale. Small-n forcings stay on the column
    // path below — their readback is cheap and a reduce dispatch would fill only n/64 threads
    // ([`REDUCE_MIN_LANES`]) — as does everything non-moments (Q's collect).
    if (lanes.end - lanes.start) as usize >= REDUCE_MIN_LANES {
        if let Some(mode) = r.moments_mode() {
            return reduce_on_gpu(&g, root, mode, lanes, seed, r, token, deadline, carry);
        }
    }

    let emitted = {
        let _s = crate::profile::span("gpu.emit");
        wgsl_emit::emit(&g, &[root])
    };
    let Ok(wgsl) = emitted else {
        crate::profile::note("gpu: cone unsupported (Poisson/Rotation) → cpu");
        return Ok(GpuReduce::Declined(carry)); // Poisson / Rotation (f64 Gram–Schmidt) → CPU
    };

    let n = (lanes.end - lanes.start) as usize;
    let cost = crate::kernel::cost(&g, root);
    crate::profile::set_ops(cost.ops);
    let instrs = emitted_instrs(&wgsl);
    if !forced() && !profitable(instrs, cost.ops, n) {
        crate::profile::note(gate_reason(instrs, cost.ops, n));
        return Ok(GpuReduce::Declined(carry));
    }
    crate::profile::note(gate_reason(instrs, cost.ops, n));
    let prepared = {
        let _s = crate::profile::span("gpu.pipeline");
        prepare(&wgsl)
    };
    let Some((pipe, hit)) = prepared else {
        crate::profile::note("gpu: driver rejected shader → cpu");
        return Ok(GpuReduce::Declined(carry)); // driver rejected: fall back rather than fail
    };
    crate::profile::note(if hit {
        "gpu.pipeline: cache HIT"
    } else {
        "gpu.pipeline: cache MISS (compiled)"
    });

    // Same counters the CPU path records, and off the same simplified cone — so the playground's
    // ops / random-numbers readout doesn't change just because the GPU took the query.
    crate::stats::record(n, cost.ops, cost.sources);

    let key = crate::rng::Key::from_seed(seed);
    let mut acc = carry;
    let mut done = 0usize;
    let mut stopped = None;
    while done < n {
        if token.is_some_and(CancelToken::is_cancelled) {
            return Err(NoiseError::cancelled());
        }
        // Soft stop / deadline between dispatches (~1M-lane / ~ms granularity, Track H): keep the
        // chunks already folded, report the cause.
        if let Some(cause) = crate::exec::stop_cause_of(token, deadline) {
            stopped = Some(cause);
            break;
        }
        let take = GPU_DISPATCH.min(n - done);
        // Epoch-local lanes fit u32 by construction (the reduce driver split at 2³² boundaries).
        let lane0 =
            u32::try_from(lanes.start + done as u64).expect("epoch-local lane exceeds 2^32");
        let col = dispatch(&pipe, key, lane0, take as u32, 1);

        // Fold on the reducer's OWN chunk boundaries, in order, into the running carry — so the
        // accumulation is the same sequence of `absorb`/`merge` calls the CPU reducer would have
        // made, and the answer doesn't depend on how big a dispatch happened to be.
        let _s = crate::profile::span("gpu.fold");
        for slice in col.chunks(CHUNK_SAMPLES) {
            let mut chunk = r.identity();
            r.absorb(&mut chunk, slice);
            acc = r.merge(acc, chunk);
        }
        done += take;
    }

    Ok(GpuReduce::Done(Reduced { acc, stopped }))
}

/// The reduce-mode driver (PLAN-PRECISION Track F): dispatch big lane ranges whose shader folds
/// `(Σx, Σx², count)` per workgroup ([`wgsl_emit::emit_reduce`]), then fold the partials into the
/// carry **in workgroup (= lane) order**. A workgroup covers a fixed 4096-lane slice and stage
/// boundaries are chunk-aligned (16,384 = 4 workgroups), so a staged adaptive run folds exactly the
/// partials a single run would — the staged-==-single bit-identity holds on the GPU's own fold.
///
/// The partial sums are f32 on the device (tier 2: deterministic per device, ~1e-6-relative of the
/// CPU's f64 fold); the cross-partial fold here is f64, so precision loss is bounded per 4096-lane
/// slice and does not grow with `n`.
#[allow(clippy::too_many_arguments)]
fn reduce_on_gpu<R: Reducer>(
    g: &RvGraph,
    root: RvId,
    mode: MomentsMode,
    lanes: std::ops::Range<u64>,
    seed: u64,
    r: &R,
    token: Option<&CancelToken>,
    deadline: Option<Instant>,
    carry: R::Acc,
) -> Result<GpuReduce<R::Acc>> {
    let emitted = {
        let _s = crate::profile::span("gpu.emit");
        wgsl_emit::emit_reduce(g, root, mode == MomentsMode::SkipNan)
    };
    let Ok(wgsl) = emitted else {
        crate::profile::note("gpu: cone unsupported (Poisson/Input) → cpu");
        return Ok(GpuReduce::Declined(carry));
    };

    let n = (lanes.end - lanes.start) as usize;
    let cost = crate::kernel::cost(g, root);
    crate::profile::set_ops(cost.ops);
    let instrs = emitted_instrs(&wgsl);
    if !forced() && !profitable_reduce(instrs, cost.ops, n) {
        crate::profile::note(gate_reason_reduce(instrs, cost.ops, n));
        return Ok(GpuReduce::Declined(carry));
    }
    crate::profile::note(gate_reason_reduce(instrs, cost.ops, n));
    let prepared = {
        let _s = crate::profile::span("gpu.pipeline");
        prepare(&wgsl)
    };
    let Some((pipe, hit)) = prepared else {
        crate::profile::note("gpu: driver rejected shader → cpu");
        return Ok(GpuReduce::Declined(carry));
    };
    crate::profile::note(if hit {
        "gpu.pipeline: cache HIT"
    } else {
        "gpu.pipeline: cache MISS (compiled)"
    });
    crate::stats::record(n, cost.ops, cost.sources);

    // Size each dispatch to roughly constant GPU *time* (the stop/deadline check granularity),
    // using the same per-lane cost proxy as the compile estimate: the cone's ops plus ~150 emulated
    // ALU per draw. Chunk-aligned so the fold boundaries stay nested in reducer chunks.
    let per_lane = (cost.ops + 150 * cost.sources).max(1) as f64;
    let dispatch_lanes = ((REDUCE_WORK_PER_DISPATCH / per_lane) as usize)
        .clamp(REDUCE_DISPATCH_MIN, REDUCE_DISPATCH_MAX)
        / CHUNK_SAMPLES
        * CHUNK_SAMPLES;

    let key = crate::rng::Key::from_seed(seed);
    let mut acc = carry;
    let mut done = 0usize;
    let mut stopped = None;
    while done < n {
        if token.is_some_and(CancelToken::is_cancelled) {
            return Err(NoiseError::cancelled());
        }
        if let Some(cause) = crate::exec::stop_cause_of(token, deadline) {
            stopped = Some(cause);
            break;
        }
        let take = dispatch_lanes.min(n - done);
        // Epoch-local lanes fit u32 by construction (the reduce driver split at 2³² boundaries).
        let lane0 =
            u32::try_from(lanes.start + done as u64).expect("epoch-local lane exceeds 2^32");
        let partials = dispatch(&pipe, key, lane0, take as u32, 0);

        // One `(Σx, Σx², count)` triple per workgroup, in workgroup order — which IS lane order,
        // so folding them sequentially into the carry keeps the fold a pure function of the lane
        // ranges, independent of how the range was split into dispatches or stages.
        let _s = crate::profile::span("gpu.fold");
        for p in partials.chunks_exact(wgsl_emit::REDUCE_COLS as usize) {
            let mut wg = r.identity();
            // The count is exact: a workgroup covers ≤ 4096 lanes, far inside f32's integer range.
            r.absorb_moments(&mut wg, p[2] as u64, f64::from(p[0]), f64::from(p[1]));
            acc = r.merge(acc, wg);
        }
        done += take;
    }

    Ok(GpuReduce::Done(Reduced { acc, stopped }))
}

/// Try to run a **joint** forcing (several roots drawn together) on the GPU — the D4b driver behind
/// the introspection/plot passes (`describe`/`corr`/`hist`/`fan`/`scatter`, `plot::line`/`plot::fan`)
/// that `sampler::for_each_joint_batch` otherwise runs on the CPU interpreter. Those passes were the
/// single biggest CPU pool the corpus profile surfaced (am_vs_fm's two `plot::line` = 200 ms,
/// barrier_option's `plot::fan` = 96 ms), and never touched the GPU before this.
///
/// `wgsl_emit::emit` already lowers several roots into ONE shader writing `out[j*n + i]` (root `j`,
/// lane `i`) with shared draws, so this is the same dispatch as [`try_reduce`] with a `k`-column
/// output: one pipeline compile, one dispatch stream for all roots, folded through the caller's
/// `sink` per chunk in lane order. `Ok(None)` (thin/unsupported cone, no adapter, gate decline) falls
/// back to the CPU exactly like [`try_reduce`]; the result stays under the two-tier contract (tier-1
/// draws bit-identical, tier-2 f32 stats ULP-close), and the fold is single-threaded and in lane
/// order, so it is deterministic across runs — matching the (also single-threaded) CPU joint loop.
///
/// `sink(cols, take)` is called with `cols[j]` = root `j`'s column for a chunk of `take` lanes, the
/// same contract `for_each_joint_batch`'s CPU loop uses — so the two paths share the accumulation
/// code verbatim.
pub fn try_joint<F: FnMut(&[&[f32]], usize)>(
    graph: &RvGraph,
    roots: &[RvId],
    n: usize,
    seed: u64,
    mut sink: F,
    token: Option<&CancelToken>,
) -> Result<Option<()>> {
    if !available() {
        crate::profile::note("gpu(joint): no adapter → cpu");
        return Ok(None);
    }
    // Same simplify the CPU joint path gets (union cone, cross-root sharing preserved), so the draw
    // ordinals — and thus the joint pairing — are identical to what the interpreter would compute.
    let (g, roots) = {
        let _s = crate::profile::span("gpu.simplify");
        crate::simplify::simplify_roots(graph, roots)
    };
    let emitted = {
        let _s = crate::profile::span("gpu.emit");
        wgsl_emit::emit(&g, &roots)
    };
    let Ok(wgsl) = emitted else {
        crate::profile::note("gpu(joint): cone unsupported → cpu");
        return Ok(None);
    };
    let cost = crate::kernel::cost_roots(&g, &roots);
    crate::profile::set_ops(cost.ops);
    let instrs = emitted_instrs(&wgsl);
    let k = roots.len();
    // The shared single-root gate, PLUS the joint-only per-root fatness term (see
    // [`JOINT_MIN_OPS_PER_ROOT`]): a wide-but-thin joint pass (many roots, tiny per-element cone)
    // reads back `k × n` f32 and folds it on the CPU, which costs more than the fast interpreter it
    // would replace — so it declines even though the *union* cone clears the single-root gate.
    let per_root_ok = cost.ops >= JOINT_MIN_OPS_PER_ROOT * k as u64;
    if !forced() && (!profitable(instrs, cost.ops, n) || !per_root_ok) {
        crate::profile::note(if per_root_ok {
            gate_reason(instrs, cost.ops, n)
        } else {
            format!(
                "gate: DECLINE — joint cone too thin per root ({} ops / {k} roots < {JOINT_MIN_OPS_PER_ROOT})",
                cost.ops
            )
        });
        return Ok(None);
    }
    crate::profile::note(gate_reason(instrs, cost.ops, n));
    let Some((pipe, _hit)) = prepare(&wgsl) else {
        crate::profile::note("gpu(joint): driver rejected shader → cpu");
        return Ok(None);
    };
    crate::stats::record(n, cost.ops, cost.sources);

    let key = crate::rng::Key::from_seed(seed);
    // Fewer lanes per dispatch when there are more columns, so the `k × take` readback stays bounded.
    let max_lanes = (GPU_JOINT_ELEMS / k.max(1)).max(wgsl_emit::WORKGROUP as usize);
    let mut done = 0usize;
    while done < n {
        if token.is_some_and(CancelToken::is_cancelled) {
            return Err(NoiseError::cancelled());
        }
        // Soft stop / deadline (Track H): a stopped plot pass renders what it collected — the
        // caller's sink already holds the completed chunks, a valid smaller-sample pass.
        if crate::exec::stop_cause().is_some() {
            break;
        }
        let take = max_lanes.min(n - done);
        let lane0 = u32::try_from(done).expect("forcing exceeds 2^32 lanes");
        let buf = dispatch(&pipe, key, lane0, take as u32, k as u32);
        // `buf[j*take + i]` is root `j`, lane `i` — present the k columns for this chunk (lane order
        // preserved), and hand them to the caller's fold exactly as the CPU batch loop would.
        let _s = crate::profile::span("gpu.fold");
        let cols: Vec<&[f32]> = (0..k).map(|j| &buf[j * take..(j + 1) * take]).collect();
        sink(&cols, take);
        done += take;
    }
    Ok(Some(()))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::reduce::{MomentsReducer, Reducer};

    /// Track F: the reduce-mode fold is a pure function of the lane ranges. Workgroup partials
    /// cover fixed 4096-lane slices and fold in lane order, so a range split into stages folds
    /// exactly the partials the single range would — **bit-identical**, the same carry-fold
    /// invariant the CPU driver gives the adaptive precision driver. The ranges here are big
    /// enough that the reduce-mode gate accepts each stage *naturally* (no test-only forcing,
    /// which would leak into this whole test binary through the `FORCE` OnceLock).
    #[test]
    fn reduce_mode_staged_fold_is_bit_identical_to_single() {
        if !available() {
            return; // no adapter: nothing to test, and try_reduce would just decline
        }
        let mut eng = crate::eval::Engine::new();
        let v = eng
            .run_rv("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X*X + Y*Y < 1")
            .unwrap();
        let crate::Value::Dist(root) = v else {
            panic!("expected a dist")
        };
        let g = eng.graph();
        let r = MomentsReducer;
        let seed = 11u64;
        let (mid, end) = (1u64 << 28, (1u64 << 28) + (1u64 << 28)); // 2 × 268M lanes, both gate-accepted

        let run = |lanes: std::ops::Range<u64>, carry| match try_reduce(
            g, root, lanes, seed, &r, None, None, carry,
        )
        .unwrap()
        {
            GpuReduce::Done(out) => {
                assert!(out.stopped.is_none());
                out.acc
            }
            GpuReduce::Declined(_) => panic!("gate must accept a 268M-lane thin cone"),
        };
        let single = run(0..end, r.identity());
        let staged = {
            let s1 = run(0..mid, r.identity());
            run(mid..end, s1)
        };
        assert_eq!(staged.count(), single.count());
        let (a, b) = (staged.into_moments(), single.into_moments());
        assert_eq!(
            a.mean.to_bits(),
            b.mean.to_bits(),
            "staged mean must match bit-for-bit"
        );
        assert_eq!(
            a.variance.to_bits(),
            b.variance.to_bits(),
            "staged variance must match"
        );
        // …and it is still π/4.
        assert!(
            (a.mean - std::f64::consts::FRAC_PI_4).abs() < 1e-4,
            "mean = {}",
            a.mean
        );
    }
}
