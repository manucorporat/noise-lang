//! The native WebGPU backend, end to end (PLAN-WEBGPU G2). `--features gpu`.
//!
//! The whole contract of a backend in this engine is: **it may change speed, never results.** The
//! GPU is the hardest case for that promise — it fuses multiply-add, its transcendentals are
//! vendor-defined, and it drives the forcing itself rather than going through `Runner`. So these
//! tests are all one question asked several ways: *does the answer survive?*
//!
//! They run against the same programs with the feature on and off is not possible in one binary, so
//! the oracle here is **the analytic answer plus the standard error the engine itself reports** —
//! which is the strongest available check that doesn't just re-run the CPU: an estimate that is
//! wrong by more than a few standard errors is wrong, whichever backend produced it.

#![cfg(feature = "gpu")]

use noise_core::{Engine, Value};

/// Force every gated forcing onto the GPU, so these tests exercise the backend rather than silently
/// measuring the JIT. Without this, the gate would decline most of them on cost grounds (the native
/// multicore JIT is *already* fast — see the corpus numbers) and the tests would pass vacuously.
fn force() {
    noise_core::gpu_force_for_tests();
}

fn est(src: &str) -> (f64, f64) {
    force();
    match Engine::new().run(src).expect("program runs") {
        Value::Est { val, se } => (val, se),
        other => panic!("expected an estimate, got {other:?}"),
    }
}

/// `4 · P(x² + y² < 1) = π`. The simplest possible end-to-end proof that the GPU is drawing the right
/// stream, doing the right arithmetic, and folding into the right accumulator.
#[test]
fn pi_on_the_gpu() {
    let (v, se) = est("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X^2 + Y^2 < 1, 2000000)");
    assert!(
        (v - std::f64::consts::PI).abs() < 6.0 * se.max(1e-9),
        "pi = {v} +- {se}"
    );
}

/// A Gaussian model — the shape the whole plan is built on (Box–Muller, `ln` + `sincos` + `sqrt` per
/// draw, all inside the shader's prelude). Mean and variance must both land.
#[test]
fn a_gaussian_model_on_the_gpu() {
    let (m, se) = est("use rand; Z ~ normal(3, 2); E(Z, 2000000)");
    assert!((m - 3.0).abs() < 6.0 * se, "mean = {m} +- {se}");

    let (v, _) = est("use rand; Z ~ normal(3, 2); Var(Z, 2000000)");
    assert!((v - 4.0).abs() < 0.05, "var = {v}, want 4");
}

/// **A Haar rotation on the GPU (G4b), tested by the property that DEFINES it.**
///
/// A rotation is the one draw the GPU cannot make bit-identical to the interpreter: the interpreter
/// runs its Gram–Schmidt in f64 scratch and WGSL has no f64, so the two produce *different* matrices
/// from the same seed. What must survive is the thing turboquant actually relies on — that the matrix
/// is orthonormal — so the contract here is distributional, not lane-for-lane.
///
/// Norm preservation is exactly that property, sharpened: a true rotation has `|Pi·x|² = |x|²` for
/// EVERY lane, so with `x` a fixed unit vector `normsq(Pi @ x)` is identically 1 — mean 1 **and**
/// near-zero variance. The variance is the real test: a matrix that was merely unit-*mean* but not
/// orthonormal would pass the mean and fail the spread.
#[test]
fn a_haar_rotation_on_the_gpu() {
    let setup = "use rand; use vec; x = normalize(ones(20)); Pi ~ rand::rotation(20);";

    let (m, _) = est(&format!("{setup} E(normsq(Pi @ x), 300000)"));
    assert!((m - 1.0).abs() < 1e-3, "E[|Pi x|²] = {m}, want 1 — the rotation is not norm-preserving");

    // f32 Gram–Schmidt at d=20 holds orthonormality to ~1e-4, so |Pi x|² sits within ~1e-4 of 1 every
    // lane; its variance is thus ~1e-8. Anything like 1e-2 would mean the rows aren't orthonormal.
    let (v, _) = est(&format!("{setup} Var(normsq(Pi @ x), 300000)"));
    assert!(
        v < 1e-4,
        "Var(|Pi x|²) = {v} — a true rotation gives |Pi x|² ≡ 1 (zero variance up to f32); this large \
         a spread means the GPU matrix is not orthonormal"
    );
}

/// **turboquant, end to end on the GPU** — the demo this whole phase is for. Algorithm 1's 1-bit
/// distortion is a known quantity (the paper's ≈0.36), and it is a *distributional* result: it does
/// not depend on drawing the interpreter's exact rotations, only on drawing valid ones from the Haar
/// measure. So the GPU — with its own f32 rotations — must still land it.
#[test]
fn turboquant_distortion_on_the_gpu() {
    let (d, _) = est(
        "use rand; use vec; use math; \
         d = 20; x = normalize(ones(d)); s = 1 / sqrt(d); \
         L1 = [-0.7979, 0.7979] * s; \
         Pi ~ rand::rotation(d); PiT = transpose(Pi); \
         E(normsq(x - (PiT @ quantize(Pi @ x, L1))), 200000)",
    );
    // The paper's Algorithm-1 b=1 distortion is ≈0.36; allow for MC noise and the f32 rotation.
    assert!((d - 0.36).abs() < 0.03, "turboquant b=1 distortion = {d}, want ≈0.36");
}

/// The `barrier_option` shape, and the reason `ArrDraw` exists: 52 shaped normals folded by a sum,
/// emitted as ONE draw loop. `sum(zs) ~ N(0, 52)`, so `P(sum < 0) = 1/2` and `Var(sum) = 52`.
///
/// If the loop drew the wrong ordinals — reading element 0 every iteration, say, or starting from the
/// wrong base — the mean would still be 0 and only the *variance* would give it away. Hence both.
#[test]
fn a_shaped_draw_loop_on_the_gpu() {
    let (p, se) = est("use rand; use vec; zs ~[52] normal(0,1); P(vec::sum(zs) < 0, 2000000)");
    assert!((p - 0.5).abs() < 6.0 * se, "P(sum < 0) = {p} +- {se}");

    let (v, _) = est("use rand; use vec; zs ~[52] normal(0,1); Var(vec::sum(zs), 2000000)");
    assert!(
        (v - 52.0).abs() < 1.0,
        "Var(sum of 52 iid normals) = {v}, want 52 — a loop reading the wrong ordinals would still \
         give mean 0, so this is the assertion that actually pins it"
    );
}

/// Cones the GPU can't lower must still produce correct answers — it declines and a CPU backend
/// takes over. This is the fallback promise, and it is the one that decides whether the feature is
/// safe to ship: a `poisson` (a Knuth loop) and a `permutation` (an array-valued source, and
/// interpreter-only on every backend).
#[test]
fn declined_cones_still_give_the_right_answer() {
    force();

    // Poisson: mean = lambda.
    let (m, se) = est("use rand; K ~ poisson(4.5); E(K, 500000)");
    assert!((m - 4.5).abs() < 6.0 * se, "poisson mean = {m} +- {se}");

    // Permutation: box 0 holds card 0 with probability 1/20 in a uniform permutation of 20.
    let (p, se) = est("use rand; d ~ rand::permutation(20); P(d[0] == 0, 500000)");
    assert!((p - 0.05).abs() < 6.0 * se, "P(perm[0] == 0) = {p} +- {se}, want 0.05");
}

/// Large-argument trig, end to end and now **on** the GPU (G1b) — this cone used to be declined.
///
/// `E[sin(1e6 · X)]` over `X ~ unif(0,1)` is ~0 because the argument sweeps ~159,000 whole periods.
/// Getting that right requires actually resolving where in its period each argument falls, which is
/// exactly what no f32-representable multiple of π/2 can tell you: WGSL's built-in `sin` returns a
/// flat 0 out here, and an `x % TAU` reduction returns noise. Only the exact integer reduction lands.
#[test]
fn large_argument_trig_on_the_gpu() {
    let (m, se) = est("use rand; use math; X ~ unif(0,1); E(math::sin(1000000 * X), 500000)");
    assert!(m.abs() < 6.0 * se.max(1e-3), "E[sin(1e6 X)] = {m} +- {se}, want ~0");

    // The sharper probe: E[sin²] = 1/2 exactly, and unlike the mean it cannot be faked by a
    // backend that returns 0 everywhere — which is precisely what the built-in does here.
    let (v, _) = est(
        "use rand; use math; X ~ unif(0,1); E(math::sin(1000000 * X) * math::sin(1000000 * X), 500000)",
    );
    assert!(
        (v - 0.5).abs() < 0.01,
        "E[sin²(1e6 X)] = {v}, want 0.5 — a backend whose `sin` collapses to 0 out here scores 0.0"
    );
}

/// The GPU's answer must not depend on how the work was split into dispatches — the same
/// chunk-ordered-fold determinism the threaded CPU reducer guarantees. Two draw counts that land on
/// either side of the 1M-lane dispatch boundary must agree to within their own noise.
#[test]
fn the_answer_does_not_depend_on_the_dispatch_split() {
    let (a, sa) = est("use rand; X ~ unif(0,1); E(X * X, 1000000)");
    let (b, sb) = est("use rand; X ~ unif(0,1); E(X * X, 3000000)");
    // Both estimate 1/3, and they share a seed and a stream, so they should be very close.
    assert!((a - 1.0 / 3.0).abs() < 6.0 * sa, "1M: {a} +- {sa}");
    assert!((b - 1.0 / 3.0).abs() < 6.0 * sb, "3M (spans 3 dispatches): {b} +- {sb}");
}


/// **G4c: a rolled loop on the GPU** — prisoners' pointer-chase, the whole reason the Scan IR exists.
/// The loop rolls to a WGSL `for`, and its answer must still be the interpreter's (analytic 0.3118).
#[test]
fn prisoners_on_the_gpu() {
    let (p, _se) = est(
        "use rand; n = 100; boxes ~ permutation(n); all = true; \
         for prisoner in 0..n { box = prisoner; found = false; \
           for hop in 0..50 { box = boxes[box]; found = found || (box == prisoner) }; \
           all = all && found }; P(all, 200000)",
    );
    assert!((p - 0.3118).abs() < 0.01, "prisoners on GPU = {p}, want ~0.3118");
}

/// A single rolled loop on the GPU (pointer-chase over a permutation), forced onto the backend.
#[test]
fn a_rolled_loop_on_the_gpu() {
    let (p, se) = est(
        "use rand; boxes ~ permutation(20); box = 0; found = false; \
         for hop in 0..12 { box = boxes[box]; found = found || (box == 0) }; P(found, 500000)",
    );
    // A cycle of length <=12 through 0 in a random permutation of 20: analytic P = 12/20 = 0.6.
    assert!((p - 0.6).abs() < 6.0 * se.max(1e-3), "P(found) = {p} +- {se}, want 0.6");
}
