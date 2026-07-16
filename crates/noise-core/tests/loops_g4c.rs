//! G4c: a captured `for` loop (rolled to a Scan, then unrolled for the CPU) must produce
//! byte-for-byte the same result as the classic build-time unroll.
//!
//! Caveat since PLAN-DROP-JIT D4a lowered the GPU gate (`--features gpu`): for a cone whose result is
//! *floating-point arithmetic* fat enough to gate to the GPU, the captured loop emits a ROLLED WGSL
//! loop (`gpu::emit_scan`) while the build-time unroll emits a flat cone, and the two f32 summation
//! orders differ in the last bits — the engine's two-tier contract (tier-2 arithmetic is ULP-close,
//! not byte-identical). Those cases use [`assert_capture_matches_ulp`]. The count/`P(…)` cases stay
//! byte-for-byte: their draws are tier-1 (integer permutations/gathers, bit-identical on every
//! backend), so the GPU changes nothing there.

use noise_core::{set_loop_capture, Engine, Value};

fn run(src: &str) -> f64 {
    match Engine::new().run(src).expect("runs") {
        Value::Est { val, .. } => val,
        Value::Num(x) => x,
        other => panic!("unexpected {other:?}"),
    }
}

/// The same program with capture on and off must agree to the last bit — the whole correctness claim
/// of `unroll_scans`.
fn assert_capture_matches(src: &str) {
    let off = {
        set_loop_capture(false);
        let v = run(src);
        set_loop_capture(true);
        v
    };
    let on = {
        set_loop_capture(true);
        run(src)
    };
    assert_eq!(
        on.to_bits(),
        off.to_bits(),
        "capture changed the result for:\n{src}\n on={on} off={off}"
    );
}

/// The tier-2 twin of [`assert_capture_matches`] for a *floating-point arithmetic* cone that can gate
/// to the GPU, where the captured (rolled WGSL loop) and unrolled (flat cone) forms agree only to f32
/// ULP — see the module note. On a no-GPU build both run on the interpreter and are in fact
/// byte-for-byte, so the loose bound is only ever exercised by the GPU; a real unroll bug (wrong trip
/// count or index) diverges by orders of magnitude more than this.
fn assert_capture_matches_ulp(src: &str) {
    let off = {
        set_loop_capture(false);
        let v = run(src);
        set_loop_capture(true);
        v
    };
    let on = {
        set_loop_capture(true);
        run(src)
    };
    let tol = 1e-6 * on.abs().max(1.0);
    assert!(
        (on - off).abs() <= tol,
        "capture changed the result by more than f32 ULP for:\n{src}\n on={on} off={off} (tol={tol})",
    );
}

#[test]
fn pointer_chase_scan_matches_unroll() {
    // prisoners' inner shape: follow a permutation from a start for K hops, OR-accumulate a hit.
    assert_capture_matches(
        "use rand; boxes ~ permutation(20); box = 0; found = false; \
         for hop in 0..12 { box = boxes[box]; found = found || (box == 0) }; P(found)",
    );
}

#[test]
fn index_using_scan_matches_unroll() {
    // A loop whose body reads the loop counter (needs is_iota + index_ph). Its result is float
    // arithmetic (`E` over a `X*i` sum) fat enough to gate to the GPU, where the rolled and unrolled
    // forms agree only to f32 ULP (the two-tier contract) — see the module note. Byte-for-byte on CPU.
    assert_capture_matches_ulp(
        "use rand; X ~ unif(0,1); acc = 0.0; for i in 0..16 { acc = acc + X * i }; E(acc)",
    );
}

#[test]
fn nested_scan_matches_unroll() {
    // prisoners' full doubly-nested shape, small.
    assert_capture_matches(
        "use rand; boxes ~ permutation(10); all = true; \
         for p in 0..10 { box = p; found = false; \
           for hop in 0..5 { box = boxes[box]; found = found || (box == p) }; \
           all = all && found }; P(all)",
    );
}

#[test]
fn prisoners_analytic() {
    // The real thing: ~31.18% for n=100, opens=50.
    let p = run("use rand; n = 100; boxes ~ permutation(n); all = true; \
         for prisoner in 0..n { box = prisoner; found = false; \
           for hop in 0..50 { box = boxes[box]; found = found || (box == prisoner) }; \
           all = all && found }; P(all, 200000)");
    assert!(
        (p - 0.3118).abs() < 0.01,
        "prisoners P(all) = {p}, want ~0.3118"
    );
}
