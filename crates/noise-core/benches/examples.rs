//! End-to-end benchmark over the **real** programs in `examples/`.
//!
//! `sampling.rs` times the hot loop of a single hand-written RV expression: compile once, draw a
//! million samples. That isolates throughput, which is what the WASM and WGSL emitters were built to
//! win — but it is *not* the shape of a real program, and it systematically hides the cost those
//! backends add. A real Noise program:
//!
//!   * forces **many** queries, each compiling its own cone (`noise_colors` forces 10, `kelly` 12),
//!   * often draws **few** samples per query (small per-call counts), so a codegen backend
//!     may never amortize its compile,
//!   * spends real time in parse/eval building the graph (a `rotation(20)` expands to ~16k nodes
//!     before a single draw), and
//!   * ends in plots and introspection passes, which take a different sampler path entirely.
//!
//! So this bench times [`Engine::run_to_document`] — exactly what `noise <file>` runs — over every
//! example. It is the honest regression gate: a change that speeds the kernel but slows codegen
//! shows up *here* and nowhere else.
//!
//! Run: `cargo bench -p noise-core --bench examples` (interpreter) and again with `--features gpu`
//! to compare backends on the same real programs.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use noise_core::Engine;
use std::hint::black_box;
use std::time::Duration;

mod corpus;
use corpus::EXAMPLES;

fn bench_examples(c: &mut Criterion) {
    let mut group = c.benchmark_group("examples/end_to_end");
    // Real programs run for milliseconds-to-seconds, not nanoseconds. Criterion's default of 100
    // samples would take many minutes on the heavy ones (`turboquant` alone is seconds per run), so
    // trade statistical tightness for a suite that people will actually run.
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for (label, src) in EXAMPLES {
        group.bench_with_input(BenchmarkId::from_parameter(label), src, |b, src| {
            b.iter(|| {
                // A fresh engine per iteration: a program's cost includes building its graph from
                // scratch, and a reused engine would carry over cached realizations and bindings.
                let mut eng = Engine::new();
                black_box(eng.run_to_document(black_box(src)))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_examples);
criterion_main!(benches);
