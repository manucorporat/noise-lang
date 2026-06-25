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

/// Samples per query. 1e6 is the default `P()` budget, so this mirrors a real run's hot loop
/// while staying large enough that one-time compile cost is in the noise.
const N: usize = 1_000_000;
const SEED: u64 = 0xC0FFEE;

/// Representative programs: `(label, source)`. Each ends in an RV expression so `run_rv` yields a
/// `Value::Dist`. `use rand` brings the distribution constructors into scope.
const CASES: &[(&str, &str)] = &[
    // Tiny graph, the π Monte Carlo indicator (2 uniforms, a few arith ops, one compare).
    ("pi_indicator", "use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X**2 + Y**2 < 1"),
    // Two discrete draws + an add — the cheapest realistic graph.
    ("dice_sum", "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B"),
    // Transcendental-bound: Box–Muller dominates, arithmetic is trivial.
    ("normal_sum", "use rand; X ~ normal(0,1); Y ~ normal(0,1); X + Y"),
    // Deep arithmetic chain over ONE draw: ~10 dependent ops, all intermediates materialized.
    (
        "poly_deep",
        "use rand; X ~ unif(0,1); \
         X*X*X*X*X + 2*X*X*X*X - 3*X*X*X + 4*X*X - 5*X + 6",
    ),
    // Wide graph: many independent draws summed — exercises register pressure / column count.
    (
        "poly_wide",
        "use rand; A ~ unif(0,1); B ~ unif(0,1); C ~ unif(0,1); D ~ unif(0,1); \
         E ~ unif(0,1); F ~ unif(0,1); G ~ unif(0,1); H ~ unif(0,1); \
         A*B + C*D + E*F + G*H + A*H + B*G + C*F + D*E",
    ),
    // Single `ln` libcall + a compare — calibrates the cost of one transcendental on the JIT.
    ("exp_tail", "use rand; X ~ exp(2); X > 1"),
    // MIXED: one normal (libcall-heavy source) feeding a deep arithmetic chain — the case that
    // decides whether fusion can outrun the transcendental penalty (the profitability crossover).
    (
        "normal_poly",
        "use rand; Z ~ normal(0,1); \
         Z*Z*Z*Z*Z + 2*Z*Z*Z*Z - 3*Z*Z*Z + 4*Z*Z - 5*Z + 6",
    ),
];

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
