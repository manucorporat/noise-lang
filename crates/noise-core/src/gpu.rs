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

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::dist::{RvGraph, RvId};
use crate::error::{NoiseError, Result};
use crate::exec::CancelToken;
use crate::reduce::Reducer;
use crate::wgsl_emit;

/// Lanes per dispatch. Big enough to bury the ~1.2 ms fixed cost of a dispatch + readback (G0), and a
/// whole number of 16,384-sample reducer chunks, so the fold below is chunk-for-chunk identical to
/// what the CPU reducer would have produced. 1M lanes is a 4 MB readback.
const GPU_DISPATCH: usize = 1 << 20;

/// The reducer's chunk size — mirrored from `reduce` so the GPU folds on exactly the same boundaries
/// (`combine_in_order`'s determinism guarantee is about *chunks*, not threads).
const CHUNK_SAMPLES: usize = 16 * crate::bytecode::BATCH;

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
/// is pure RNG-and-memory, where a warmed-up multicore JIT is simply hard to beat and the pipeline
/// compile can never be earned back.
///
/// Sits at 100 — comfortably inside the empty band between 58 and 124.
const MIN_CONE_OPS: u64 = 100;

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
/// *thin* to be worth dispatching, and a forcing too short to earn the compile back.
fn profitable(instrs: usize, ops_per_draw: u64, n: usize) -> bool {
    instrs <= MAX_WGSL_INSTRS
        && ops_per_draw >= MIN_CONE_OPS
        && (n as f64 * ops_per_draw as f64) >= MIN_WORK_GPU
}

/// The gate decision with the reason the failing term (for `NOISE_PROFILE=1`, PLAN-DROP-JIT D0): the
/// D4a recalibration needs to see *which* of the three terms declines each forcing, not just that one
/// did. Mirrors [`profitable`] exactly.
fn gate_reason(instrs: usize, ops_per_draw: u64, n: usize) -> String {
    if instrs > MAX_WGSL_INSTRS {
        format!("gate: DECLINE — cone too big ({instrs} instrs > {MAX_WGSL_INSTRS})")
    } else if ops_per_draw < MIN_CONE_OPS {
        format!("gate: DECLINE — cone too thin ({ops_per_draw} ops/draw < {MIN_CONE_OPS})")
    } else if (n as f64 * ops_per_draw as f64) < MIN_WORK_GPU {
        format!(
            "gate: DECLINE — work too small ({:.2e} < {MIN_WORK_GPU:.0e})",
            n as f64 * ops_per_draw as f64
        )
    } else {
        format!("gate: ACCEPT — {instrs} instrs, {ops_per_draw} ops/draw, {n} draws")
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
// Device + pipeline cache (process-wide: acquiring a device is slow, and a compiled pipeline is
// exactly the thing G0 says we must not pay for twice).
// ---------------------------------------------------------------------------

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Compiled pipelines, keyed by the shader text itself — which is a complete description of the
    /// artifact, so this is a content-addressed cache and can never serve a stale kernel.
    pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
}

/// `None` on a machine with no usable adapter — the caller then simply uses a CPU backend.
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
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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

    /// Dispatch lanes `lane0 .. lane0 + n` and read the column back.
    fn dispatch(&self, pipe: &wgpu::ComputePipeline, key: crate::rng::Key, lane0: u32, n: u32) -> Vec<f32> {
        let bytes = u64::from(n) * 4;
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
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(wgsl_emit::WORKGROUP), 1, 1);
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

/// Try to run this whole forcing on the GPU.
///
/// `Ok(None)` means "not this backend's job" — no adapter, an unsupported cone, or the gate saying
/// the CPU finishes first. The caller then proceeds exactly as if this module did not exist, so a
/// decline is never an error and never changes an answer.
///
/// `Err` is reserved for **cancellation**, which must propagate: a cancelled forcing has folded only
/// some of its chunks, and that partial answer must never escape as though it were an estimate.
pub fn try_reduce<R: Reducer>(
    graph: &RvGraph,
    root: RvId,
    n: usize,
    seed: u64,
    r: &R,
    token: Option<&CancelToken>,
) -> Result<Option<R::Acc>> {
    let Some(dev) = device() else {
        crate::profile::note("gpu: no adapter → cpu");
        return Ok(None);
    };

    // The same simplify the other backends get, so the cone — and therefore the draw ordinals — are
    // identical to what the interpreter would compute.
    let (g, root) = {
        let _s = crate::profile::span("gpu.simplify");
        crate::simplify::simplify(graph, root)
    };
    let emitted = {
        let _s = crate::profile::span("gpu.emit");
        wgsl_emit::emit(&g, &[root])
    };
    let Ok(wgsl) = emitted else {
        crate::profile::note("gpu: cone unsupported (Poisson/Rotation) → cpu");
        return Ok(None); // Poisson / Rotation (f64 Gram–Schmidt) → CPU; see wgsl_emit::plan_blocks
    };

    let cost = crate::kernel::cost(&g, root);
    crate::profile::set_ops(cost.ops);
    let instrs = emitted_instrs(&wgsl);
    if !forced() && !profitable(instrs, cost.ops, n) {
        crate::profile::note(gate_reason(instrs, cost.ops, n));
        return Ok(None);
    }
    crate::profile::note(gate_reason(instrs, cost.ops, n));
    let (pipe, hit) = {
        let _s = crate::profile::span("gpu.pipeline");
        let hit = dev.pipeline_cached(&wgsl);
        (dev.pipeline(&wgsl), hit)
    };
    let Some(pipe) = pipe else {
        crate::profile::note("gpu: driver rejected shader → cpu");
        return Ok(None); // a shader the driver rejected: fall back rather than fail the program
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
    let mut accs: Vec<R::Acc> = Vec::new();
    let mut done = 0usize;
    while done < n {
        if token.is_some_and(CancelToken::is_cancelled) {
            return Err(NoiseError::cancelled());
        }
        let take = GPU_DISPATCH.min(n - done);
        // One u32 of lane index caps a forcing at 2^32 draws — the same documented boundary the CPU
        // reducer has.
        let lane0 = u32::try_from(done).expect("forcing exceeds 2^32 lanes");
        let col = dev.dispatch(&pipe, key, lane0, take as u32);

        // Fold on the reducer's OWN chunk boundaries, in order — so the accumulation is the same
        // sequence of `absorb`/`merge` calls the CPU reducer would have made, and the answer doesn't
        // depend on how big a dispatch happened to be.
        let _s = crate::profile::span("gpu.fold");
        for slice in col.chunks(CHUNK_SAMPLES) {
            let mut acc = r.identity();
            r.absorb(&mut acc, slice);
            accs.push(acc);
        }
        done += take;
    }

    let mut acc = r.identity();
    for a in accs {
        acc = r.merge(acc, a);
    }
    Ok(Some(acc))
}
