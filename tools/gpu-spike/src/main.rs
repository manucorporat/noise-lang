//! PLAN-WEBGPU **G0** — the spike that kills or scales the plan.
//!
//! Four questions, and nothing else:
//!
//!   1. **Does the certified RNG survive WGSL?** `squares64` is five 64-bit wrapping multiplies and
//!      WGSL has no `u64`. It must be emulated on `vec2<u32>` — and the plan's stated risk is that
//!      this costs "~70–90 ALU ops per uniform vs ~10" for a GPU-native hash, enough to dominate a
//!      normal-dense kernel. Measured by A/B: the same shape over squares vs a cheap u32 hash vs no
//!      hash at all.
//!   2. **Do the giant shaders compile?** `turboquant` is ~17.6k nodes per draw and `prisoners`
//!      ~45k. Pipeline-compile time vs statement count is the plan's biggest unknown — a 10-second
//!      compile turns a 10x win into a loss, and a hard compiler limit kills those two demos.
//!   3. **What does a dispatch actually cost?** The floor under the profitability gate.
//!   4. **Are the draws bit-identical to the engine's?** Everything above is meaningless if the GPU
//!      is benchmarking a different generator. Checked against `noise_core::rng` before any timing.
//!
//! Run: `cargo run --release` from `tools/gpu-spike`.

mod shapes;
mod wgsl;

use std::time::{Duration, Instant};

use noise_core::rng::Key;
use wgsl::{Rng, Trans};

// ---------------------------------------------------------------------------
// The wgpu harness: compile a WGSL string, dispatch N lanes, read the column back.
// ---------------------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: String,
}

/// What compiling one shader cost, split the way the gate will need it: Naga's parse/validate
/// (ours, on the CPU, unavoidable) vs the backend's own compile (Metal/Tint — the part that scares
/// the plan).
struct CompileCost {
    naga: Duration,
    backend: Duration,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no WebGPU adapter — is this a headless box?");
        let ai = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .expect("no device");
        let info = format!(
            "{} ({:?}, {:?}) — max storage binding {} MB",
            ai.name,
            ai.backend,
            ai.device_type,
            limits.max_storage_buffer_binding_size / (1 << 20)
        );
        Gpu { device, queue, info }
    }

    /// Compile a shader into a pipeline, timing Naga and the backend separately.
    ///
    /// `create_shader_module` runs Naga; `create_compute_pipeline` is where Metal actually compiles
    /// the MSL and builds the PSO. Both are wrapped in an error scope so a WGSL mistake surfaces as
    /// a message rather than a panic 200 lines later.
    fn compile(&self, src: &str) -> Result<(wgpu::ComputePipeline, CompileCost), String> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let t = Instant::now();
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let naga = t.elapsed();
        let t = Instant::now();
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
        let backend = t.elapsed();
        if let Some(err) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(err.to_string());
        }
        Ok((pipeline, CompileCost { naga, backend }))
    }

    /// Dispatch `n` lanes and read the column back, reusing pre-allocated buffers.
    ///
    /// Buffer creation is *outside* the timed region deliberately: a real backend allocates its
    /// chunk buffers once and reuses them across dispatches, so folding allocation into every
    /// measurement would inflate the dispatch floor — the exact number the profitability gate is
    /// built on. (The first version of this spike did that and reported a ~1.2 ms floor that was
    /// mostly `create_buffer`.)
    fn run(&self, pipeline: &wgpu::ComputePipeline, b: &Bufs, key: Key, lane0: u32, n: u32) -> (Vec<f32>, Duration) {
        let bytes = u64::from(n) * 4;
        let params = [key.k0, key.k1, lane0, n];
        let t = Instant::now();
        self.queue.write_buffer(&b.ubuf, 0, as_bytes(&params));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &b.bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(wgsl::WORKGROUP), 1, 1);
        }
        enc.copy_buffer_to_buffer(&b.out, 0, &b.staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let staging = &b.staging;

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).expect("poll");
        let data = slice.get_mapped_range();
        let samples: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        drop(data);
        staging.unmap();
        (samples, t.elapsed())
    }
}

/// The per-dispatch resources a real backend would allocate once and reuse: the params uniform, the
/// output column, the mappable staging copy, and the bind group over them.
struct Bufs {
    ubuf: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl Bufs {
    fn new(gpu: &Gpu, pipeline: &wgpu::ComputePipeline, n: u32) -> Self {
        let bytes = u64::from(n) * 4;
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        Bufs { ubuf, out, staging, bind }
    }
}

/// `&[u32; 4] -> &[u8]`, so the spike doesn't take a bytemuck dependency for four words.
fn as_bytes(words: &[u32; 4]) -> &[u8] {
    // SAFETY: `[u32; 4]` is 16 contiguous bytes with no padding, and every bit pattern is a valid
    // `u8`. The borrow is tied to `words` by the signature.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 16) }
}

// ---------------------------------------------------------------------------
// 1. Conformance, in the two tiers the FMA probe leaves us.
// ---------------------------------------------------------------------------

/// **Tier 1 (bitwise, mandatory).** The raw 24-bit draws must be the engine's, exactly.
///
/// This is the tier that survives contraction, because it is all *integer* arithmetic: the squares64
/// rounds, the counter layout, the 48-bit consumption, the pair split. It is also the tier that
/// carries the RNG certification — if these bits match, C0's 1 TB PractRand evidence applies to the
/// GPU unchanged, because the GPU is consuming the identical stream.
///
/// The draw is written out as an f32 *integer* (every value < 2^24 is exact in f32), so no rounding
/// stands between the hash and the comparison.
fn draw_conformance(gpu: &Gpu) -> bool {
    use noise_core::rng;
    const N: u32 = 4096;
    let key = Key::from_seed(0);
    let mut ok = true;

    for (kind, src_id) in [("pair draws (unif/normal/exp)", 0u32), ("lane draws (unif_int)", 5)] {
        let s = shapes::raw_bits(src_id, kind.starts_with("pair"));
        let wg = wgsl::shader(Rng::Squares, Trans::Poly, &s.body, &s.root);
        let (pipe, _) = gpu.compile(&wg).expect("raw-bits shader compiles");
        let bufs = Bufs::new(gpu, &pipe, N);
        let (got, _) = gpu.run(&pipe, &bufs, key, 0, N);

        // The CPU's answer for the same lanes, straight from the generator.
        let want: Vec<f32> = (0..N)
            .map(|lane| {
                let b = if kind.starts_with("pair") {
                    let (lo, hi) = rng::pair_bits(key, src_id, lane);
                    if lane % 2 == 0 { lo } else { hi }
                } else {
                    let bits = rng::draw48(key, rng::scalar_ctr(src_id, lane));
                    rng::lo24(bits)
                };
                b as f32
            })
            .collect();

        let exact = got.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        ok &= exact == N as usize;
        println!(
            "  {kind:<30} {}  {exact}/{N} bit-identical",
            if exact == N as usize { "EXACT   " } else { "MISMATCH" }
        );
    }
    ok
}

/// **Tier 2 (ULP-bounded).** The distributions built on top of those draws.
///
/// These cannot be bitwise — the FMA probe proved the GPU contracts, so `mu + sigma*z`, every Horner
/// step and every user-level `a*b + c` round once where the CPU rounds twice. What we *can* pin is
/// that the difference stays at the level of f32 rounding rather than a real algorithmic divergence,
/// which is what a ULP bound says. Reported, not asserted — the number is the finding.
fn lane_conformance(gpu: &Gpu, trans: Trans) {
    use noise_core::rng;
    const N: u32 = 4096;
    let key = Key::from_seed(0);

    let cases: [(&str, fn(Key, &mut [f32])); 4] = [
        ("unif(0,1)", |k, out| rng::fill_uniform(k, 0, 0, 0.0, 1.0, out)),
        ("unif(2,5)", |k, out| rng::fill_uniform(k, 0, 0, 2.0, 5.0, out)),
        ("normal(0,1)", |k, out| rng::fill_normal(k, 2, 0, 0.0, 1.0, out)),
        ("exp(1)", |k, out| rng::fill_exp(k, 3, 0, 1.0, out)),
    ];

    for (kind, fill) in cases {
        let s = shapes::conformance(kind);
        let wg = wgsl::shader(Rng::Squares, trans, &s.body, &s.root);
        let (pipe, _) = gpu.compile(&wg).expect("conformance shader compiles");
        let bufs = Bufs::new(gpu, &pipe, N);
        let (got, _) = gpu.run(&pipe, &bufs, key, 0, N);
        let mut want = vec![0.0f32; N as usize];
        fill(key, &mut want);

        let exact = got.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let max_ulp = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (i64::from(a.to_bits()) - i64::from(b.to_bits())).abs())
            .max()
            .unwrap_or(0);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "  {kind:<14} {exact:>4}/{N} bitwise   max {max_ulp:>4} ulp   max |Δ| {max_abs:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Throughput, and the two A/Bs the plan's risks turn on.
// ---------------------------------------------------------------------------

/// Best-of-`reps` round-trip time for a shape at `n` lanes, with the buffers allocated once (as a
/// real backend would) so the number is dispatch cost, not allocation cost.
fn time_shape(
    gpu: &Gpu,
    s: &shapes::Shape,
    rng: Rng,
    trans: Trans,
    n: u32,
    reps: usize,
) -> Option<(CompileCost, Duration)> {
    let src = wgsl::shader(rng, trans, &s.body, &s.root);
    let (pipe, cost) = match gpu.compile(&src) {
        Ok(v) => v,
        Err(e) => {
            println!("    COMPILE FAILED: {}", e.lines().next().unwrap_or(""));
            return None;
        }
    };
    let bufs = Bufs::new(gpu, &pipe, n);
    gpu.run(&pipe, &bufs, Key::from_seed(1), 0, n); // warm
    let best = (0..reps)
        .map(|_| gpu.run(&pipe, &bufs, Key::from_seed(1), 0, n).1)
        .min()
        .unwrap();
    Some((cost, best))
}

fn label(rng: Rng, trans: Trans) -> String {
    let r = match rng {
        Rng::Squares => "squares64",
        Rng::Cheap => "cheap-u32",
        Rng::None => "no-hash",
    };
    let t = match trans {
        Trans::Native => "builtin",
        Trans::Poly => "poly",
    };
    format!("{r}/{t}")
}

/// Does this GPU contract `a*b + c` into a fused multiply-add?
///
/// Everything about the cross-backend numeric contract hinges on the answer. Both candidate results
/// are computed on the CPU (`a*b + c` with two roundings, and `f32::mul_add` with one) and the GPU's
/// output is matched against each. No inference — the hardware says which it did.
fn fma_probe(gpu: &Gpu) {
    const N: u32 = 4096;
    let s = shapes::fma_probe();
    let src = wgsl::shader(Rng::None, Trans::Native, &s.body, &s.root);
    let (pipe, _) = gpu.compile(&src).expect("fma probe compiles");
    let bufs = Bufs::new(gpu, &pipe, N);
    let (got, _) = gpu.run(&pipe, &bufs, Key::from_seed(0), 0, N);

    let (mut two_roundings, mut fused) = (0usize, 0usize);
    for (i, &g) in got.iter().enumerate() {
        let a = 1.0 + i as f32 * 0.000_000_1;
        let b = 1.0 + i as f32 * 0.000_000_3;
        if g.to_bits() == (a * b - 1.0).to_bits() {
            two_roundings += 1;
        }
        if g.to_bits() == a.mul_add(b, -1.0).to_bits() {
            fused += 1;
        }
    }
    println!("  a*b + c matches the CPU's mul-then-add in {two_roundings}/{N} lanes");
    println!("  a*b + c matches a FUSED multiply-add   in {fused}/{N} lanes");
    println!(
        "  => {}",
        if fused > two_roundings {
            "this GPU CONTRACTS. Bit-identical f32 *arithmetic* with the CPU backends is off the\n     table — the contract's bitwise tier is the DRAWS (integer ops, uncontractable), and lane\n     arithmetic is ULP-close. (FMA is the more accurate of the two, for what it's worth.)"
        } else {
            "no contraction — bit-identical lane arithmetic is achievable"
        }
    );
}

/// The head-to-head: the SAME kernel, over the SAME draws, on the CPU (Cranelift JIT, multicore
/// reducer) and on the GPU.
///
/// Every other number in this spike is an absolute rate, which says nothing on its own. This is the
/// one that answers "is the GPU worth building" — and it is deliberately run against the *fastest*
/// CPU backend we have, not the interpreter.
fn head_to_head(gpu: &Gpu) {
    const STEPS: usize = 100;
    const DRAWS: u32 = 1_000_000;

    let src = format!(
        "use rand;\nuse vec;\nengine::set_max_samples({DRAWS});\nengine::set_max_opts(1000000000000);\n\
         zs ~[{STEPS}] normal(0, 1);\nE(vec::sum(zs), {DRAWS})"
    );
    let t = Instant::now();
    let cpu = noise_core::run(&src);
    let cpu_secs = t.elapsed().as_secs_f64();
    match &cpu {
        Ok(v) => println!("  CPU (jit, multicore)   {:>7.1} ms   E[sum] = {v:?}", cpu_secs * 1e3),
        Err(e) => {
            println!("  CPU run failed: {e}");
            return;
        }
    }

    // Before timing anything: both GPU kernels must compute the same thing the CPU computes. A loop
    // that drew the wrong stream would be fast and *wrong*, and it would sail through a benchmark.
    //
    // The bar is a tolerance, not bitwise — and the reason is finding G0-1 (the FMA probe): the two
    // codegen shapes contract multiply-adds differently, so even unrolled-vs-looped disagree in the
    // last bits. What must hold is that they agree to f32 rounding on a value of magnitude ~sqrt(100),
    // which is a very different claim from "the draws are right" (a wrong stream would be O(1) off,
    // not O(1e-5)).
    {
        const N: u32 = 4096;
        let key = Key::from_seed(0);
        // The CPU's per-lane answer, accumulated in the lane type, source by source.
        let mut want = vec![0.0f32; N as usize];
        let mut col = vec![0.0f32; N as usize];
        for s in 0..STEPS as u32 {
            noise_core::rng::fill_normal(key, s, 0, 0.0, 1.0, &mut col);
            for (w, &c) in want.iter_mut().zip(&col) {
                *w += c;
            }
        }
        for (label, shape) in [
            ("unrolled", shapes::sum_normals(STEPS)),
            ("looped  ", shapes::sum_normals_looped(STEPS)),
        ] {
            let src = wgsl::shader(Rng::Squares, Trans::Native, &shape.body, &shape.root);
            let (pipe, _) = gpu.compile(&src).expect("compiles");
            let bufs = Bufs::new(gpu, &pipe, N);
            let (got, _) = gpu.run(&pipe, &bufs, key, 0, N);
            let max_abs = got.iter().zip(&want).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("  vs CPU, {label}: max |Δ| {max_abs:.2e} on a sum of {STEPS} normals (|x| ~ 10)");
            assert!(
                max_abs < 1e-3,
                "{label} does not agree with the CPU — this is a WRONG DRAW STREAM, not rounding"
            );
        }
    }

    // Two emitter strategies for the *same* graph, so the compile tax is priced against the thing it
    // actually buys. The cold compile has to be measured with a salt or Metal's on-disk cache serves
    // it and the number is a fiction.
    for (label, s) in [
        ("unrolled", shapes::sum_normals(STEPS)),
        ("LOOPED  ", shapes::sum_normals_looped(STEPS)),
    ] {
        let src = wgsl::shader(Rng::Squares, Trans::Native, &s.body, &s.root);
        let cold = gpu
            .compile(&wgsl::shader_salted(Rng::Squares, Trans::Native, &s.body, &s.root, fresh_salt()))
            .map(|(_, c)| (c.naga + c.backend).as_secs_f64())
            .unwrap_or(f64::NAN);
        let Ok((pipe, _)) = gpu.compile(&src) else { continue };
        let bufs = Bufs::new(gpu, &pipe, DRAWS);
        gpu.run(&pipe, &bufs, Key::from_seed(1), 0, DRAWS);
        let best = (0..5)
            .map(|_| gpu.run(&pipe, &bufs, Key::from_seed(1), 0, DRAWS).1)
            .min()
            .unwrap()
            .as_secs_f64();
        println!(
            "  GPU {label}          {:>7.1} ms   ({:.0} ms cold compile + {:.1} ms dispatch)   \
             dispatch alone: {:.0}x",
            (cold + best) * 1e3,
            cold * 1e3,
            best * 1e3,
            cpu_secs / best,
        );
        let verdict = if cold + best < cpu_secs {
            format!("{:.1}x FASTER than the CPU end-to-end", cpu_secs / (cold + best))
        } else {
            format!("{:.1}x SLOWER than the CPU end-to-end — the compile eats the win", (cold + best) / cpu_secs)
        };
        println!("       => {verdict}");
    }
    println!("     ({} normal draws per forcing, {} per lane)", DRAWS as usize * STEPS, STEPS);
}

/// A salt unique to this process, so every compile-time measurement is genuinely cold (see
/// `wgsl::shader_salted` — Metal's shader cache lives on disk and survives across runs).
fn fresh_salt() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let base = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    base.wrapping_mul(2_654_435_761) ^ SEQ.fetch_add(1, Ordering::Relaxed)
}

fn main() {
    let gpu = Gpu::new();
    println!("device: {}\n", gpu.info);

    println!("== 0. does the GPU fuse multiply-add? (decides the whole conformance tier) ==");
    fma_probe(&gpu);
    println!();

    println!("== 1a. TIER 1 — the raw draws must be bit-identical to noise_core::rng ==");
    let ok = draw_conformance(&gpu);
    println!(
        "  => {}\n",
        if ok {
            "the WGSL draws ARE the engine's draws — squares64 survives WGSL intact, and C0's\n     certification carries onto the GPU (same counters, same 48 bits, same order)"
        } else {
            "DRAWS DIVERGE — every number below is meaningless until this is fixed"
        }
    );

    println!("== 1b. TIER 2 — the distributions on top of them (poly transcendentals) ==");
    lane_conformance(&gpu, Trans::Poly);
    println!("\n== 1c. TIER 2 with the WGSL built-in log/sin/cos instead ==");
    lane_conformance(&gpu, Trans::Native);
    println!();

    println!("== 2. dispatch + readback floor (pi kernel, buffers pre-allocated) ==");
    let tiny = shapes::pi();
    for n in [1u32, 1_024, 65_536, 1 << 20] {
        if let Some((_, t)) = time_shape(&gpu, &tiny, Rng::Squares, Trans::Poly, n, 50) {
            println!("  {n:>9} lanes  {:>8.3} ms", t.as_secs_f64() * 1e3);
        }
    }
    println!();

    // The two A/Bs that settle "keep the certified hash?" and "keep the shared polynomials?".
    //
    // Configs are timed **round-robin**, not one-after-another, and each keeps its own min. The GPU
    // drifts (clocks, thermals) over the tens of seconds this section takes, and a sequential sweep
    // charges that drift to whichever config ran last — an earlier version of this table swung
    // between "squares64 costs 1.9x" and "squares64 is free" across runs purely from that.
    println!("== 3. RNG + transcendental A/B (barrier, 100 steps, 1M lanes; round-robin best-of-24) ==");
    println!("  {:<20} {:>10} {:>13} {:>10}", "config", "time", "samples/s", "speedup");
    let bar = shapes::barrier(100);
    let n = 1 << 20;
    let configs = [
        (Rng::Squares, Trans::Poly),
        (Rng::Squares, Trans::Native),
        (Rng::Cheap, Trans::Poly),
        (Rng::Cheap, Trans::Native),
        (Rng::None, Trans::Native),
    ];
    let built: Vec<_> = configs
        .iter()
        .filter_map(|&(rng, trans)| {
            let src = wgsl::shader(rng, trans, &bar.body, &bar.root);
            let (pipe, _) = gpu.compile(&src).ok()?;
            let bufs = Bufs::new(&gpu, &pipe, n);
            gpu.run(&pipe, &bufs, Key::from_seed(1), 0, n); // warm
            Some((label(rng, trans), pipe, bufs))
        })
        .collect();
    let mut best = vec![Duration::MAX; built.len()];
    for _ in 0..24 {
        for (i, (_, pipe, bufs)) in built.iter().enumerate() {
            let t = gpu.run(pipe, bufs, Key::from_seed(1), 0, n).1;
            best[i] = best[i].min(t);
        }
    }
    let base = best[0].as_secs_f64();
    for (i, (name, _, _)) in built.iter().enumerate() {
        let secs = best[i].as_secs_f64();
        println!(
            "  {name:<20} {:>7.2} ms {:>10.0} M/s {:>9.2}x",
            secs * 1e3,
            f64::from(n) / secs / 1e6,
            base / secs
        );
    }
    println!("  (speedup is vs the squares64/poly config — i.e. what we'd gain by giving up\n   certification, or by giving up the shared polynomials)\n");

    println!("== 4. pipeline compile vs shader size ==");
    println!(
        "  {:>8} {:>9} {:>10} {:>10} {:>12}",
        "stmts", "naga", "backend", "COLD", "warm (cached)"
    );
    // Each size gets a fresh salt, so the Metal on-disk shader cache cannot serve it; then the
    // identical source is compiled a second time to price the cache hit a repeat visitor gets.
    for (i, n) in [100usize, 1_000, 5_000, 17_600, 45_000].into_iter().enumerate() {
        let _ = i;
        let s = shapes::chain(n, 0);
        let src = wgsl::shader_salted(Rng::Squares, Trans::Native, &s.body, &s.root, fresh_salt());
        match gpu.compile(&src) {
            Ok((_, cold)) => {
                let (_, warm) = gpu.compile(&src).expect("recompile");
                println!(
                    "  {:>8} {:>6.1} ms {:>7.1} ms {:>7.1} ms {:>9.1} ms",
                    s.stmts,
                    cold.naga.as_secs_f64() * 1e3,
                    cold.backend.as_secs_f64() * 1e3,
                    (cold.naga + cold.backend).as_secs_f64() * 1e3,
                    (warm.naga + warm.backend).as_secs_f64() * 1e3,
                );
            }
            Err(e) => println!("  {:>8}  did not compile: {}", s.stmts, e.lines().next().unwrap_or("")),
        }
    }
    println!("  (naga is ours and is never cached; the backend compile is what Metal caches on disk)\n");

    // The anomaly that reframes the whole compile risk: `sum_normals(100)` is 201 source statements
    // but compiles as slowly as a 17,602-statement chain. The reason is that `squares64` is emulated,
    // so every RNG source INLINES into ~150 ALU ops — compile cost tracks emitted instructions, not
    // graph nodes. If that is true, the cost must scale with the SOURCE count here, and the cheap
    // u32 hash (a ~12-op inline) must be dramatically cheaper to compile at the same source count.
    println!("== 4b. compile vs SOURCE count (the emulated-hash inlining tax) ==");
    println!("  {:>8} {:>8} {:>14} {:>14} {:>8}", "sources", "stmts", "squares64", "cheap-u32", "ratio");
    for n in [1usize, 10, 50, 100, 200] {
        let s = shapes::sum_normals(n);
        let sq = wgsl::shader_salted(Rng::Squares, Trans::Native, &s.body, &s.root, fresh_salt());
        let ch = wgsl::shader_salted(Rng::Cheap, Trans::Native, &s.body, &s.root, fresh_salt());
        let (Ok((_, a)), Ok((_, b))) = (gpu.compile(&sq), gpu.compile(&ch)) else {
            println!("  {n:>8}  did not compile");
            continue;
        };
        let (ta, tb) = (
            (a.naga + a.backend).as_secs_f64() * 1e3,
            (b.naga + b.backend).as_secs_f64() * 1e3,
        );
        println!("  {:>8} {:>8} {:>11.1} ms {:>11.1} ms {:>7.1}x", n, s.stmts, ta, tb, ta / tb);
    }
    println!();

    println!("== 6. HEAD TO HEAD: same kernel, same draws, CPU vs GPU ==");
    head_to_head(&gpu);
    println!();

    println!("== 5. demo-shaped kernels (squares64 / poly — the shippable config) ==");
    println!(
        "  {:<28} {:>7} {:>10} {:>10} {:>12}",
        "shape", "stmts", "compile", "run", "samples/s"
    );
    let cases: Vec<(shapes::Shape, u32)> = vec![
        (shapes::barrier(100), 175_000),
        (shapes::signal(20), 40_000),
        (shapes::barrier(100), 1 << 20),
        (shapes::signal(20), 1 << 20),
        (shapes::chain(17_600, 0), 10_000),
    ];
    for (s, n) in cases {
        if let Some((c, t)) = time_shape(&gpu, &s, Rng::Squares, Trans::Poly, n, 5) {
            let secs = t.as_secs_f64();
            println!(
                "  {:<28} {:>7} {:>7.1} ms {:>7.2} ms {:>10.1} M/s",
                format!("{} x {} lanes", s.name, n),
                s.stmts,
                (c.naga + c.backend).as_secs_f64() * 1e3,
                secs * 1e3,
                f64::from(n) / secs / 1e6
            );
        }
    }
}
