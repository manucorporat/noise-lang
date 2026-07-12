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
