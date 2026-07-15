//! G4c: a captured `for` loop (rolled to a Scan, then unrolled for the CPU) must produce
//! byte-for-byte the same result as the classic build-time unroll.

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
    let off = { set_loop_capture(false); let v = run(src); set_loop_capture(true); v };
    let on = { set_loop_capture(true); run(src) };
    assert_eq!(on.to_bits(), off.to_bits(), "capture changed the result for:\n{src}\n on={on} off={off}");
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
    // A loop whose body reads the loop counter (needs is_iota + index_ph).
    assert_capture_matches(
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
    let p = run(
        "use rand; n = 100; boxes ~ permutation(n); all = true; \
         for prisoner in 0..n { box = prisoner; found = false; \
           for hop in 0..50 { box = boxes[box]; found = found || (box == prisoner) }; \
           all = all && found }; P(all, 200000)",
    );
    assert!((p - 0.3118).abs() < 0.01, "prisoners P(all) = {p}, want ~0.3118");
}
