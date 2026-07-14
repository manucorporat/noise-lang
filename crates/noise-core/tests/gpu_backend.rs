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
/// safe to ship: a `permutation` (interpreter-only), a `poisson` (Knuth loop), and an explicit `sin`
/// (G1's correctness cut — no f64 for the large-argument reduction).
#[test]
fn declined_cones_still_give_the_right_answer() {
    force();

    // Poisson: mean = lambda.
    let (m, se) = est("use rand; K ~ poisson(4.5); E(K, 500000)");
    assert!((m - 4.5).abs() < 6.0 * se, "poisson mean = {m} +- {se}");

    // Explicit trig, past the range where WGSL's built-in is even defined — the GPU must NOT take
    // this, and the CPU's answer must come through unharmed.
    let (m, _) = est("use rand; use math; X ~ unif(0,1); E(math::sin(1000000 * X), 500000)");
    assert!(m.abs() < 0.01, "sin of a large argument = {m}, want ~0");

    // Permutation (an array-valued source, interpreter-only): box 0 holds card 0 with probability
    // 1/20 in a uniform permutation of 20.
    let (p, se) = est("use rand; d ~ rand::permutation(20); P(d[0] == 0, 500000)");
    assert!((p - 0.05).abs() < 6.0 * se, "P(perm[0] == 0) = {p} +- {se}, want 0.05");
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
