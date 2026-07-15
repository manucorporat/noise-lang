//! `noise-core` — the portable core of the Noise probabilistic language.
//!
//! Noise is an expression-based probabilistic language: a binding holds a *probability
//! distribution*, and you compute probabilities / expectations / summaries by Monte Carlo. This
//! crate is the whole language and its runtime — lexer, parser, evaluator, and a batched sampler —
//! with no OS / thread / time dependencies, so it compiles cleanly to `wasm32`. The `noise` CLI and
//! the browser playground are thin hosts over this crate.
//!
//! Pipeline: [`lexer`](self) → `parser` (a hand-written Pratt parser) → `ast` → [`eval`] (a
//! tree-walking [`Engine`] that lowers `~`-drawn randomness into an append-only sample-DAG) → a
//! columnar batched sampler — with an optional Cranelift JIT and a WASM emitter as alternate
//! backends — which forces `P` / `E` / `var` / `describe` and other queries.
//!
//! ## Public surface
//!
//! The curated entry points are [`Engine`] (build, run, and introspect programs) and [`run`]
//! (parse-and-evaluate a string with a fresh engine). The supporting public modules are [`error`],
//! [`value`], [`eval`], [`input`], [`doc`], [`introspect`], [`frontmatter`], and [`stats`].
//! Everything else — the lexer, parser, AST, bytecode VM, the sampler, and the JIT / WASM codegen
//! backends — is a `pub(crate)` implementation detail and deliberately outside the semver surface.
//!
//! ## Example
//!
//! ```
//! use noise_core::{run, Value};
//!
//! // A tiny deterministic program. The builtin modules are strictly scoped, so `math` must be
//! // brought in with `use` before `sqrt` resolves — a bare `sqrt` is an undefined-name error.
//! let value = run("use math; sqrt(2) ^ 2").unwrap();
//! assert!(matches!(value, Value::Num(n) if (n - 2.0).abs() < 1e-9));
//! ```

// `approx` and `rng` are the *numeric contract* every backend transcribes, so the measurement
// harnesses under `tools/` have to reach them: `gpu-spike` proves its WGSL draws are bit-for-bit
// the engine's by linking the real generator, and a transcribed copy that silently drifted would
// make every number it reports a claim about a hash we don't ship. The `internals` feature is that
// door, and only that: off by default, `#[doc(hidden)]`, no semver promise.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod approx;
#[cfg(not(feature = "internals"))]
pub(crate) mod approx;
pub(crate) mod ast;
pub(crate) mod backend;
pub(crate) mod builtins;
pub(crate) mod bytecode;
pub(crate) mod compile_cache;
/// Shared cross-backend conformance corpus (finding C2), consumed by the `jit` and `wasm_emit`
/// test modules. Test-only data — no runtime footprint.
#[cfg(test)]
mod conformance;
pub(crate) mod dist;
pub mod doc;
pub mod error;
pub mod eval;
pub mod exec;
pub(crate) mod flint;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod frontmatter;
pub mod input;
pub mod introspect;
#[cfg(feature = "jit")]
pub(crate) mod jit;
pub(crate) mod kernel;
pub(crate) mod lexer;
pub(crate) mod num;
pub(crate) mod parser;
pub(crate) mod reduce;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod rng;
#[cfg(not(feature = "internals"))]
pub(crate) mod rng;
pub(crate) mod sampler;
pub(crate) mod signal;
pub(crate) mod simplify;
pub mod stats;
pub mod value;
pub(crate) mod wasm_emit;
// The WGSL emitter (PLAN-WEBGPU G1) is pure text generation with no GPU dependency, so it builds on
// every target; `internals` exposes it to `tools/gpu-spike`, which owns the wgpu harness that runs
// the generated shaders against the interpreter oracle.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod wgsl_emit;
#[cfg(not(feature = "internals"))]
pub(crate) mod wgsl_emit;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm_host;

/// Re-exported only so the [`eval`] facade's [`Engine::graph`] accessor can name it. The sample-DAG
/// is an implementation detail, not part of the curated surface — hence `#[doc(hidden)]`.
#[doc(hidden)]
pub use dist::RvGraph;
pub use dist::RvId;
pub use doc::Document;
pub use error::{NoiseError, Result};
pub use eval::{set_loop_capture, Emission, Engine, Output};
/// Chart lowering used by the WASM playground host to turn an [`introspect::Summary`] into a
/// `flint-chart` spec. Re-exported (rather than a public `flint` module) so the rest of that module
/// stays internal.
pub use flint::{to_flint, Plot};
pub use frontmatter::Frontmatter;
pub use input::{InputKind, InputSpec, InputValue, ResolvedInput};
pub use sampler::Moments;
pub use stats::RunStats;
pub use value::Value;

/// Parse and evaluate a source string with a fresh [`Engine`], returning the value of the last
/// statement.
///
/// A convenience wrapper over [`Engine::new`] followed by [`Engine::run`]. Two behaviours are worth
/// knowing before reaching for it (finding E4):
///
/// - **Emissions are discarded.** The engine is dropped when this returns, so any `print` lines,
///   `plot::*` charts, or `input::*` controls the program produced are thrown away. Use
///   [`Engine::run`] together with [`Engine::take_output`] — or [`Engine::run_to_document`] — when
///   you need the output stream.
/// - **Builtin modules are strictly scoped.** The `rand` / `math` / `vec` / `signal` modules must
///   be brought into scope first: a bare `sqrt(2)` or `normal(0, 1)` is an *undefined name* error
///   until the program says `use math;` (or writes a qualified path like `math::sqrt(2)`). Only the
///   default `builtin` module (`P` / `E` / `var` / `print` / `range` / …) is active without a `use`.
pub fn run(src: &str) -> Result<Value> {
    Engine::new().run(src)
}

/// Force every gated forcing onto the WebGPU backend, bypassing its profitability gate.
///
/// For the GPU test suite and the benchmark harness only. Without it those tests would pass
/// *vacuously*: the native multicore JIT is already fast enough that the gate correctly declines most
/// of the corpus, so the tests would be measuring the JIT while claiming to test the GPU.
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub fn gpu_force_for_tests() {
    gpu::force_gpu();
}

/// The one benchmark that stays in-crate after the E3 test relocation: it reaches into
/// `pub(crate)` internals (`kernel::cone_size`, `sampler::moments`) that the public integration
/// tests can no longer see. Ignored by default; run with:
/// `cargo test -p noise-core [--features jit] --release -- --ignored --nocapture bench_turboquant`
#[cfg(test)]
mod bench {
    use crate::{Engine, Value};

    /// Throughput of the TurboQuant workload (examples/turboquant.noise) — by far the heaviest
    /// example in the corpus, and the most interesting one to benchmark. Every Monte Carlo sample
    /// builds a *fresh* random orthonormal rotation of a d=20 vector (modified Gram–Schmidt over
    /// d² = 400 Gaussian draws), then quantizes the rotated coordinates and rotates back.
    ///
    /// The point this measures: `@`/`rotation` carry no GEMM loop. They expand
    /// element-by-element into the *same* scalar `RvGraph`, so the whole d×d linear algebra of a
    /// sample collapses into one straight-line fused kernel (CSE'd, then drawn + reduced across all
    /// cores like any other query). That is ideal at small d — everything lives in registers, no
    /// cache traffic, no loop overhead — and is the opposite end of the size spectrum from `pi`'s
    /// tiny kernel. We print the fused kernel size (`cone_size`) so the expansion is visible.
    #[test]
    #[ignore]
    fn bench_turboquant() {
        use std::time::Instant;

        let setup = "use rand; use math; use vec;
            d = 20;
            x = vec::normalize(vec::ones(d));
            y = vec::normalize(vec::iota(d));
            s = 1 / sqrt(d);
            L3 = [-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520] * s;
            L4 = [-2.7326, -2.0690, -1.6181, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                   0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6181,  2.0690,  2.7326] * s;";

        // D_mse b=4: rotate, snap to the 4-bit codebook, rotate back; distortion ||x - xhat||².
        let d_mse = format!(
            "{setup}
             Pi ~ rotation(d); rot = Pi @ x; PiT = vec::transpose(Pi);
             mse4 = PiT @ vec::quantize(rot, L4);
             vec::normsq(x - mse4)"
        );
        // D_prod b=4: the headline estimator — 3-bit MSE stage + a 1-bit QJL residual sketch,
        // two independent d×d matrices per sample (a rotation R and a Gaussian projection S).
        let d_prod = format!(
            "{setup}
             true_ip = y @ x;
             S ~[d, d] normal(0, 1);  R ~ rotation(d);
             m = vec::transpose(R) @ vec::quantize(R @ x, L3);
             est = y @ m
                 + sqrt(pi / 2) / d * vec::norm(x - m)
                   * (y @ (vec::transpose(S) @ sign(S @ (x - m))));
             (est - true_ip) ^ 2"
        );

        let n = 1_000_000usize;
        for (label, src, target) in [
            ("D_mse  b=4", d_mse.as_str(), 0.009),
            ("D_prod b=4", d_prod.as_str(), 0.047 / 20.0),
        ] {
            let mut eng = Engine::new();
            let id = match eng.run_rv(src).unwrap() {
                Value::Dist(id) => id,
                other => panic!("expected an RV, got {other:?}"),
            };
            let g = eng.graph();
            let cone = crate::kernel::cone_size(g, id);

            // Warm up (this compiles + JITs the kernel), then time sampling only.
            let _ = crate::sampler::moments(g, id, 4096, 1);
            let t = Instant::now();
            let m = crate::sampler::moments(g, id, n, 0xC0FFEE).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let mps = n as f64 / secs / 1e6;
            println!(
                "  {label}: fused kernel {cone:4} nodes ({} total)   {mps:6.2} M samples/s   \
                 est {:.4} (paper {target:.4})",
                g.len(),
                m.mean
            );
        }
    }
}
