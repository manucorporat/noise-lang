//! Sampler throughput baseline — the Phase 4 measurement harness (PLAN.md "Speed pass").
//!
//! Each case parses + lowers a representative program ONCE (outside the timed loop), then times
//! the columnar VM drawing `N` samples of the root RV. We report `Throughput::Elements(N)` so
//! Criterion prints **samples/sec** directly — the number Phase 4's definition of done is stated
//! in, and the regression gate any future engine (SIMD interpreter, Cranelift JIT, WASM codegen)
//! must beat on the SAME cases.
//!
//! Cases are chosen to separate the two cost regimes the engine has (see the Phase 4 discussion):
//!   * `pi`, `dice_sum`        — tiny graphs: dispatch + RNG-bound, the toy-example regime.
//!   * `normal_sum`            — transcendental-bound (Box–Muller sin/cos/ln per draw).
//!   * `poly_deep`, `poly_wide`— larger arithmetic DAGs: memory-traffic-bound from materializing
//!     every intermediate column. This is where fusion (JIT) should pull away from the
//!     interpreter, so it's the case that justifies — or doesn't — building a codegen backend.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use noise_core::Engine;
use std::hint::black_box;

// The `(label, source)` corpus lives in `corpus/mod.rs`, shared with `tests/bench_corpus.rs`
// so plain `cargo test` verifies every program here still parses + runs.
mod corpus;
use corpus::CASES;

/// Samples per query. 1e6 is the default `P()` budget, so this mirrors a real run's hot loop
/// while staying large enough that one-time compile cost is in the noise.
const N: usize = 1_000_000;
const SEED: u64 = 0xC0FFEE;

fn bench_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler/moments");
    group.throughput(Throughput::Elements(N as u64));
    for (label, src) in CASES {
        // Parse + lower once; the timed closure only re-runs the sampling pass.
        let mut eng = Engine::new();
        let rv = eng
            .run_rv(src)
            .unwrap_or_else(|e| panic!("bench program {label:?} failed to build: {e}"));
        group.bench_with_input(BenchmarkId::from_parameter(label), src, |b, _| {
            b.iter(|| {
                let m = eng.moments(&rv, N, black_box(SEED)).unwrap();
                black_box(m)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sampling);
criterion_main!(benches);
