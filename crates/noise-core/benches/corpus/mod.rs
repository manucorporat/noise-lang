//! The benchmark corpus, shared between the Criterion bench (`benches/sampling.rs`) and the
//! smoke test (`tests/bench_corpus.rs`) so plain `cargo test` catches bench-corpus rot — a
//! program that stops parsing/running fails CI without anyone remembering to run `cargo bench`.

/// Representative programs: `(label, source)`. Each ends in an RV expression so `run_rv` yields a
/// `Value::Dist`. `use rand` brings the distribution constructors into scope.
pub const CASES: &[(&str, &str)] = &[
    // Tiny graph, the π Monte Carlo indicator (2 uniforms, a few arith ops, one compare).
    (
        "pi_indicator",
        "use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X^2 + Y^2 < 1",
    ),
    // Two discrete draws + an add — the cheapest realistic graph.
    (
        "dice_sum",
        "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B",
    ),
    // Transcendental-bound: Box–Muller dominates, arithmetic is trivial.
    (
        "normal_sum",
        "use rand; X ~ normal(0,1); Y ~ normal(0,1); X + Y",
    ),
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
    ("exp_tail", "use rand; X ~ exponential(2); X > 1"),
    // MIXED: one normal (libcall-heavy source) feeding a deep arithmetic chain — the case that
    // decides whether fusion can outrun the transcendental penalty (the profitability crossover).
    (
        "normal_poly",
        "use rand; Z ~ normal(0,1); \
         Z*Z*Z*Z*Z + 2*Z*Z*Z*Z - 3*Z*Z*Z + 4*Z*Z - 5*Z + 6",
    ),
];
