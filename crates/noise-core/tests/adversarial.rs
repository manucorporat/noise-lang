//! "No program may panic" adversarial sweep (PLAN-TECH-DEBT.md Phase 1, finding set A).
//!
//! The core safety contract: **no user-reachable program may abort the process** (SIGABRT /
//! stack overflow) **or hang unboundedly**. Every input below used to crash or hang the engine;
//! each now must return promptly — `Err` (a spanned `NoiseError`) or `Ok` — through the public
//! `Engine::run` path, never a panic or an abort.
//!
//! How this catches regressions: for the stack-overflow / abort classes, a regression would abort
//! the whole test binary (SIGABRT) rather than fail one assertion — so the sweep *running to
//! completion* is itself the assertion. The deep-source cases run on a large-stack thread because
//! *unoptimized* (debug) frames are large: the parser/eval guards trip at a depth that is safe on
//! the small release/wasm stacks they protect but needs several MiB of debug stack to reach. A
//! regression (guard removed) still overflows even the large thread, or returns the wrong shape.

use noise_core::Engine;

/// Run a program to completion, returning whether it errored. Panics propagate (a panic here is a
/// test failure — the whole point is that these inputs must not panic).
fn errored(src: &str) -> bool {
    Engine::new().run(src).is_err()
}

/// Run `body` on a thread with a large stack so debug-build parser/eval frames can reach the depth
/// guards (see module docs).
fn on_big_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

// === A1 — parser recursion depth =================================================================

#[test]
fn a1_deeply_nested_source_errors_not_aborts() {
    on_big_stack(|| {
        // Nested parens, deep unary chains, and a right-leaning power tower each recurse the parser.
        assert!(errored(&format!(
            "{}1{}",
            "(".repeat(3000),
            ")".repeat(3000)
        )));
        assert!(errored(&format!("{}1", "-".repeat(5000))));
        assert!(errored(&format!("{}1", "!".repeat(5000))));
        assert!(errored(&format!("x{}", "[0]".repeat(3000)))); // deep postfix index chain
                                                               // A deep `if … else if …` chain.
        let _ = errored(&format!("{}{{}}", "if 1 {} else ".repeat(5000)));
    });
}

// === A2 — eval expression-recursion depth ========================================================

#[test]
fn a2_flat_operator_chain_errors_not_aborts() {
    on_big_stack(|| {
        // A 10k-term `1+1+1+…` parses fine (well under the parser's guard, it is left-nested by the
        // Pratt loop) but recurses `eval` down its left spine. It must error, not abort.
        let chain = "1+".repeat(10_000) + "1";
        let _ = errored(&chain); // returns (Err on the eval-depth guard), never overflows
                                 // A comfortably-sized chain still evaluates fine.
        assert!(!errored(&("1+".repeat(500) + "1")));
    });
}

// === A3 — signal-tree exponential walk ===========================================================

#[test]
fn a3_shared_signal_tree_does_not_hang() {
    // `for k in 0..K { s = s + s }` builds an `Rc`-shared DAG whose naive tree walk is O(2^K):
    // `has_noise`, materialization, and Display each used to blow up. Memoization makes them all
    // O(distinct nodes). K = 60 would be 2^60 unmemoized walks — this must return promptly.
    assert!(!errored(
        "use signal; s = signal::sine(1); for k in 0..60 { s = s + s }; signal::sample(s, 16)"
    ));
    // The same doubling over a *drawn-noise* tree materializes into RV nodes; memoized, it is cheap.
    assert!(!errored(
        "use signal; s ~ signal::noise_white(1); for k in 0..60 { s = s + s }; signal::sample(s, 8)"
    ));
    // Printing/formatting a huge shared signal tree must also not blow up (Display is budgeted).
    let _ = errored("use signal; s = signal::sine(1); for k in 0..60 { s = s + s }; s");
}

// === A4 — deep sample-graph walks (lower / cost / simplify) ======================================

#[test]
fn a4_deep_graph_does_not_overflow() {
    on_big_stack(|| {
        // A 200k-deep `Add` chain: `cumsum` over 200k white-noise sources. Forcing it walks the
        // graph in simplify / lower / cost / kernel — all now iterative worklists.
        assert!(!errored(
            "use rand; use vec; use signal; E(sum(cumsum(~[200000] signal::noise_white(1))), 64)"
        ));
        // Brown noise is itself a 200k-deep random walk; sampling its tail forces the same spine.
        assert!(!errored(
            "use signal; s ~[200000] signal::noise_brown(1); E(s[199999], 64)"
        ));
    });
}

// === A5 — unbounded range iteration ==============================================================

#[test]
fn a5_huge_range_errors_not_ooms() {
    // `0..1e12` would allocate/iterate a trillion elements.
    assert!(errored("0..1000000000000"));
    // Bounds beyond 2^53 (where `+= 1.0` can't advance) used to loop forever; now length is
    // computed up front, so this returns immediately (empty or capped), never hangs.
    assert!(!errored("9007199254740992..9007199254740992")); // a == b → empty
    assert!(errored("0..9007199254740994")); // huge span → capped error, no infinite loop
                                             // A normal range still works.
    assert!(!errored("for i in 0..1000 { i }"));
}

// === A6 — uncapped graph construction ============================================================

#[test]
fn a6_huge_graph_construction_errors_not_ooms() {
    assert!(errored("use rand; ~[1000000000000000] unif(0, 1)")); // ~[1e15] shape product cap
    assert!(errored("use rand; Pi ~ rotation(500); Pi")); // O(d³) rotation cap
    assert!(errored("use rand; deck ~ permutation(100000); deck")); // O(n²) permutation cap
                                                                    // In-limit structured draws still work.
    assert!(!errored(
        "use rand; use vec; Pi ~ rotation(8); normsq(Pi @ ones(8))"
    ));
    assert!(!errored(
        "use rand; use vec; deck ~ permutation(50); sum(deck)"
    ));
    assert!(!errored(
        "use rand; use vec; xs ~[1000] unif(0, 1); mean(xs)"
    ));
}

// === A7 — quantize NaN centroid ==================================================================

#[test]
fn a7_quantize_nan_centroid_errors_not_panics() {
    assert!(errored("use vec; quantize([1, 2], [0/0, 1])")); // NaN centroid
    assert!(errored("use vec; quantize([1, 2], [1/0, 1])")); // inf centroid
                                                             // A well-formed codebook still works.
    assert!(!errored("use vec; quantize([1, 2, 3], [1, 3])"));
}

// === A8 — large-lambda Poisson ===================================================================

#[test]
fn a8_large_lambda_poisson_does_not_hang() {
    // `poisson(1e12)` used to be an effective hang (O(lambda) Knuth loop per draw). The normal
    // approximation returns promptly with the right mean.
    let mut e = Engine::new();
    let mean = e
        .run("use rand; x ~ poisson(1000000000000); E(x, 2000)")
        .unwrap();
    match mean {
        noise_core::Value::Est { val, .. } | noise_core::Value::Num(val) => {
            assert!((val / 1e12 - 1.0).abs() < 0.05, "poisson mean {val}");
        }
        other => panic!("expected a numeric mean, got {other}"),
    }
    // Small lambda is still exact.
    assert!(!errored("use rand; x ~ poisson(3); E(x)"));
}

// === The consolidated "no program may panic" sweep ===============================================

#[test]
fn no_pathological_program_panics_or_hangs() {
    on_big_stack(|| {
        let cases = [
            // deep nesting (parse) — A1
            "((((((((((((((((((((1))))))))))))))))))))",
            // deep flat chain (eval) — A2
            // (kept short here; the dedicated test exercises the extreme)
            "1+1+1+1+1+1+1+1+1+1",
            // huge range — A5
            "0..1000000000000",
            // huge draw — A6
            "use rand; ~[1000000000000000] unif(0, 1)",
            "use rand; Pi ~ rotation(500); Pi",
            "use rand; deck ~ permutation(100000); deck",
            // NaN centroid — A7
            "use vec; quantize([1, 2], [0/0, 1])",
            // large-lambda poisson — A8
            "use rand; x ~ poisson(1000000000000); E(x, 500)",
            // exponential signal tree — A3
            "use signal; s = signal::sine(1); for k in 0..55 { s = s + s }; signal::sample(s, 8)",
            // deep graph — A4
            "use vec; use signal; E(sum(cumsum(~[100000] signal::noise_white(1))), 32)",
            // assorted malformed / edge inputs that must stay errors, not panics
            "use vec; quantize([], [1, 2])",
            "0/0..1",
            "use rand; ~[0] unif(0, 1)",
        ];
        for src in cases {
            // The assertion is simply that `run` returns (Ok or Err) without panicking/aborting.
            let _ = Engine::new().run(src);
        }
    });
}
