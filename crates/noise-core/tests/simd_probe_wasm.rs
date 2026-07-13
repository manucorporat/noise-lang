//! The **wasm** half of the SIMD question — the sibling of `simd_probe.rs`, which asks it for NEON.
//!
//! `simd_probe.rs` found that hand-written NEON loses 13-16% to the multi-stream scalar kernel on the
//! RNG-bound graphs Noise actually runs (pi 0.87x, dice 0.84x), because the scalar kernel's integer
//! RNG and FP math occupy disjoint ports the out-of-order core overlaps for free, while a vector
//! kernel makes them contend. That killed the native vector path. But it says nothing about wasm:
//! V8's scalar baseline is weaker than LLVM's, so SIMD could plausibly win there even though it
//! loses natively — and the browser is the target that matters.
//!
//! So this assembles a hand-written `f64x2` pi kernel (`simd_probe_wasm.wat`) to race against the
//! *real* emitted scalar kernel in V8. Measured (M4 Pro, Node 22): **simd/scalar = 0.90-0.97x** —
//! SIMD is slower, and this is with the vector kernel issuing strictly fewer instructions (4 vector
//! xoshiro steps vs 8 scalar, for the same 4 samples). Same verdict as NEON, same reason.
//!
//! Conclusion: no vector emitter. It would cost several hundred lines across `wasm_emit`, plus a
//! vectorized `approx` and a second conformance oracle, to make the dominant graph class slower.
//!
//! Reproduce:
//!   cargo test -p noise-core --release -- --ignored --nocapture dump_simd_probe_wasm
//!   NOISE_KERNEL_OUT=/tmp/scalar.wasm \
//!     NOISE_KERNEL_SRC='use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X^2 + Y^2 < 1' \
//!     cargo test -p noise-core --release --lib -- --ignored dump_kernel
//!   # then drive both through a JS host: kernel(out=4096, n, state=0), time the kernel calls ONLY
//!   # (summing the column in JS costs more than generating it and hides the difference).

#[test]
#[ignore = "dumps a .wasm for the V8 harness; asserts nothing"]
fn dump_simd_probe_wasm() {
    let bytes = wat::parse_str(include_str!("simd_probe_wasm.wat")).expect("probe must assemble");
    let out = std::env::var("NOISE_SIMD_OUT").unwrap_or_else(|_| "/tmp/simd.wasm".into());
    std::fs::write(&out, &bytes).unwrap();
    println!("wrote {out} ({} bytes)", bytes.len());
}
