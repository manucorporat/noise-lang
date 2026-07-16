//! The benchmark corpus, shared between the Criterion benches (`benches/sampling.rs`,
//! `benches/examples.rs`) and the smoke test (`tests/bench_corpus.rs`) so plain `cargo test`
//! catches bench-corpus rot — a program that stops parsing/running fails CI without anyone
//! remembering to run `cargo bench`.
//!
//! Two corpora, measuring two different things:
//!   * [`CASES`] — hand-written *micro-kernels*, each a single RV expression. They isolate the
//!     sampling hot loop: one compile, then `N` draws. This is what `sampling.rs` times.
//!   * [`EXAMPLES`] — the **real programs** in `examples/`, run end-to-end through the same
//!     `run_to_document` path the CLI uses. These include everything the micro-kernels leave out:
//!     parse, eval, *codegen*, every forcing the program performs, plots, and formatting. A
//!     program that forces 10 small queries pays codegen 10 times and draws few samples each —
//!     a regime `CASES` cannot see, and where a backend that wins on throughput can still lose
//!     on wall clock. `examples.rs` times these.

// Each bench binary (`sampling.rs` / `examples.rs`) includes this whole module but drives only its
// own corpus, so the other const reads as dead in that binary.
#![allow(dead_code)]

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
    // Single `ln` libcall + a compare — calibrates the cost of one transcendental under codegen.
    ("exp_tail", "use rand; X ~ exponential(2); X > 1"),
    // MIXED: one normal (libcall-heavy source) feeding a deep arithmetic chain — the case that
    // decides whether fusion can outrun the transcendental penalty (the profitability crossover).
    (
        "normal_poly",
        "use rand; Z ~ normal(0,1); \
         Z*Z*Z*Z*Z + 2*Z*Z*Z*Z - 3*Z*Z*Z + 4*Z*Z - 5*Z + 6",
    ),
];

/// The real-program corpus: every file in `examples/`, embedded at build time so the bench is
/// independent of the working directory. `(label, source)`, run end-to-end.
///
/// These are the programs users actually write, so they are the honest regression gate: they
/// exercise codegen cost, multi-forcing programs, joint/introspection passes, and plots — none of
/// which [`CASES`] touches.
pub const EXAMPLES: &[(&str, &str)] = &[
    ("advantage", include_str!("../../../../examples/advantage.noise")),
    ("am_vs_fm", include_str!("../../../../examples/am_vs_fm.noise")),
    ("barrier_option", include_str!("../../../../examples/barrier_option.noise")),
    ("beta_bernoulli", include_str!("../../../../examples/beta_bernoulli.noise")),
    ("birthday", include_str!("../../../../examples/birthday.noise")),
    ("bootstrap", include_str!("../../../../examples/bootstrap.noise")),
    ("buffon", include_str!("../../../../examples/buffon.noise")),
    ("clt_normal", include_str!("../../../../examples/clt_normal.noise")),
    ("coin_streak", include_str!("../../../../examples/coin_streak.noise")),
    ("conditional_bayes", include_str!("../../../../examples/conditional_bayes.noise")),
    ("dice", include_str!("../../../../examples/dice.noise")),
    ("dice_bet", include_str!("../../../../examples/dice_bet.noise")),
    ("dice_sum", include_str!("../../../../examples/dice_sum.noise")),
    ("dithering", include_str!("../../../../examples/dithering.noise")),
    ("exactly_two_heads", include_str!("../../../../examples/exactly_two_heads.noise")),
    ("functions", include_str!("../../../../examples/functions.noise")),
    ("insurance", include_str!("../../../../examples/insurance.noise")),
    ("irwin_hall", include_str!("../../../../examples/irwin_hall.noise")),
    ("kelly", include_str!("../../../../examples/kelly.noise")),
    ("max_of_dice", include_str!("../../../../examples/max_of_dice.noise")),
    ("monty_hall", include_str!("../../../../examples/monty_hall.noise")),
    ("noise_colors", include_str!("../../../../examples/noise_colors.noise")),
    ("nyquist", include_str!("../../../../examples/nyquist.noise")),
    ("pi", include_str!("../../../../examples/pi.noise")),
    ("prisoners", include_str!("../../../../examples/prisoners.noise")),
    ("qjl_scalar", include_str!("../../../../examples/qjl_scalar.noise")),
    ("rare_event", include_str!("../../../../examples/rare_event.noise")),
    ("reliability", include_str!("../../../../examples/reliability.noise")),
    ("secretary", include_str!("../../../../examples/secretary.noise")),
    ("shor_period", include_str!("../../../../examples/shor_period.noise")),
    ("st_petersburg", include_str!("../../../../examples/st_petersburg.noise")),
    ("turboquant", include_str!("../../../../examples/turboquant.noise")),
];
