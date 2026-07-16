//! Smoke test for the benchmark corpus (`benches/corpus/mod.rs`): every bench program must still
//! parse, lower, and sample, so plain `cargo test` catches bench-corpus rot (removed syntax,
//! renamed builtins) without anyone running Criterion.

use noise_core::Engine;

#[path = "../benches/corpus/mod.rs"]
mod corpus;

#[test]
fn every_bench_program_still_builds_and_samples() {
    for (label, src) in corpus::CASES {
        let mut eng = Engine::new();
        let rv = eng
            .run_rv(src)
            .unwrap_or_else(|e| panic!("bench program {label:?} failed to build: {e}"));
        // A small draw is enough to prove the program lowers and runs; Criterion owns throughput.
        let m = eng
            .moments(&rv, 1024, 0xC0FFEE)
            .unwrap_or_else(|e| panic!("bench program {label:?} failed to sample: {e}"));
        assert!(
            m.mean.is_finite(),
            "bench program {label:?} produced a non-finite mean"
        );
    }
}

/// Every real example must still run end-to-end (the `examples.rs` bench corpus). Cheap insurance
/// that a language change doesn't silently break a benchmarked program.
#[test]
fn every_example_still_runs_end_to_end() {
    for (label, src) in corpus::EXAMPLES {
        let mut eng = Engine::new();
        let doc = eng.run_to_document(src);
        assert!(
            doc.result.error.is_none(),
            "example {label:?} failed to run: {:?}",
            doc.result.error
        );
    }
}

/// Diagnostic (not an assertion): one warmed end-to-end wall-time per example — a quick per-program
/// readout for comparing two working trees or feature sets without a full Criterion run (which owns
/// the statistically careful numbers, but takes >10 minutes over this corpus).
///
/// `cargo test -p noise-core --release [--features gpu] -- --ignored --nocapture example_times`
#[test]
#[ignore = "diagnostic: prints a table, asserts nothing"]
fn example_times() {
    println!("\n{:<20}{:>12}", "EXAMPLE", "TOTAL ms");
    let mut total = 0.0f64;
    for (label, src) in corpus::EXAMPLES {
        // Warm once (allocator, lazy statics), then time a fresh engine end-to-end.
        let mut eng = Engine::new();
        let _ = eng.run_to_document(src);
        let t0 = std::time::Instant::now();
        let mut eng = Engine::new();
        let _ = eng.run_to_document(src);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        total += ms;
        println!("{label:<20}{ms:>12.1}");
    }
    println!("{:<20}{total:>12.1}", "TOTAL");
}

/// Diagnostic (not an assertion): the *shape* of each real program — how many queries it forces and
/// how many samples each one draws. This is the number that decides whether a codegen backend can
/// ever pay for itself: a program that forces 10 queries at 3k samples each compiles 10 kernels to
/// draw 30k samples total, and no per-draw speedup can refund that.
///
/// `cargo test -p noise-core --release -- --ignored --nocapture example_shapes`
#[test]
#[ignore = "diagnostic: prints a table, asserts nothing"]
fn example_shapes() {
    println!(
        "\n{:<20}{:>10}{:>12}{:>16}{:>14}",
        "EXAMPLE", "FORCINGS", "SAMPLES", "SAMPLES/FORCING", "OPS"
    );
    for (label, src) in corpus::EXAMPLES {
        let mut eng = Engine::new();
        let _ = eng.run_to_document(src);
        let s = eng.stats();

        let per = s.samples.checked_div(s.forcings).unwrap_or_default();
        println!(
            "{label:<20}{:>10}{:>12}{:>16}{:>14}",
            s.forcings, s.samples, per, s.ops
        );
    }
}
