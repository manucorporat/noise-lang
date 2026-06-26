//! noise-core — the portable core of the Noise probabilistic language.
//!
//! Pipeline: `lexer` → `parser` (Pratt) → `ast` → `eval` (tree-walking, deterministic).
//! The random-variable runtime (bytecode + batched/SIMD sampler) lands in Phase 2; see
//! PLAN.md. This crate avoids OS/threads so it compiles cleanly to `wasm32`.

pub mod approx;
pub mod ast;
pub mod backend;
pub mod builtins;
pub mod bytecode;
pub mod dist;
pub mod error;
pub mod eval;
#[cfg(feature = "jit")]
pub mod jit;
pub mod kernel;
pub mod lexer;
pub mod parser;
pub mod reduce;
pub mod rng;
pub mod sampler;
pub mod signal;
pub mod simplify;
pub mod wasm_emit;
#[cfg(target_arch = "wasm32")]
pub mod wasm_host;
pub mod value;

pub use dist::RvId;
pub use error::{NoiseError, Result};
pub use eval::Engine;
pub use sampler::Moments;
pub use value::Value;

/// Convenience: parse and evaluate a source string with a fresh engine.
pub fn run(src: &str) -> Result<Value> {
    Engine::new().run(src)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The collections/distribution/math modules are strictly scoped (a `use` or a `mod::name`
    /// path is required). Most of the corpus below predates modules and uses bare names, so the
    /// helpers run programs with all three non-default modules pre-`use`d. This shadows the
    /// crate-level `run` for the whole test module; tests that need the *raw* strict behaviour
    /// (the module-system tests) call [`run_raw`] instead.
    fn with_prelude(src: &str) -> String {
        format!("use rand; use math; use vec; use signal;\n{src}")
    }
    fn run(src: &str) -> Result<Value> {
        super::run(&with_prelude(src))
    }
    /// Run a program with NO prelude — bare `rand`/`math`/`vec` names are out of scope.
    fn run_raw(src: &str) -> Result<Value> {
        super::run(src)
    }

    fn num(src: &str) -> f64 {
        match run(src).unwrap() {
            Value::Num(n) => n,
            // An estimate's central value (used by assertions that check accuracy/tolerance;
            // display precision is checked separately via `display_of`).
            Value::Est { val, .. } => val,
            other => panic!("expected number, got {other:?} for {src:?}"),
        }
    }
    /// The user-visible string a program prints (e.g. an estimate rounded to its precision).
    fn display_of(src: &str) -> String {
        run(src).unwrap().to_string()
    }
    fn boolean(src: &str) -> bool {
        match run(src).unwrap() {
            Value::Bool(b) => b,
            other => panic!("expected bool, got {other:?} for {src:?}"),
        }
    }

    // --- ported from the legacy crate (src/lib.rs) ---
    #[test]
    fn arith() {
        assert_eq!(num("(3*(2+1)+9)/2;"), 9.0);
        assert_eq!(num("3*(2+1)+9/2;"), 13.5);
        assert_eq!(num("3*2+1+9/2;"), 11.5);
        assert_eq!(num("3*2+1-9/2-4;"), -1.5);
    }

    #[test]
    fn assignment() {
        assert_eq!(num("a=1+(2*3);a;"), 7.0);
        assert_eq!(num("a=2;a;"), 2.0);
        assert_eq!(num("a=2;"), 2.0);
        assert_eq!(num("a=1+(2*3);b=a+2;a+b;"), 16.0);
    }

    #[test]
    fn blocks() {
        assert_eq!(num("{1;2;3};"), 3.0);
        assert_eq!(num("a={1;2;3};a;"), 3.0);
        // block-local bindings leak to the enclosing env (legacy semantics)
        assert_eq!(num("a={b=2*3;c=(b+1)/2}; b+c"), 9.5);
    }

    // --- new surface (Phase 1) ---
    #[test]
    fn floats_and_unary() {
        assert_eq!(num("1.5 + 2.5"), 4.0);
        assert_eq!(num("-3 + 1"), -2.0);
        assert_eq!(num("-2 ** 2"), -4.0); // ** binds tighter than unary minus
        assert_eq!(num("2 ** 3 ** 2"), 512.0); // right-assoc
    }

    #[test]
    fn comparisons() {
        assert!(boolean("3 > 2"));
        assert!(boolean("2 <= 2"));
        assert!(!boolean("1 == 2"));
        assert!(boolean("1 != 2"));
        assert!(boolean("!(1 == 2)"));
    }

    #[test]
    fn if_else() {
        assert_eq!(num("if 3 > 2 { 10 } else { 20 }"), 10.0);
        assert_eq!(num("if 1 > 2 { 10 } else { 20 }"), 20.0);
        assert_eq!(num("x = if 5 > 1 { 5 } else { 1 }; x"), 5.0);
    }

    #[test]
    fn errors_dont_panic() {
        assert!(run("undefined_var").is_err());
        assert!(run("1 - \"a\"").is_err()); // only `+` concatenates; `-` on a string is an error
        assert!(run("if 1 { 2 }").is_err()); // non-bool condition
        assert!(run("3 +").is_err()); // parse error
    }

    #[test]
    fn tilde_on_a_point_mass_binds_the_constant() {
        // `~` on a point mass (a number) is a Dirac draw — just the constant, no randomness.
        assert_eq!(num("X ~ 5; X + 1"), 6.0);
    }

    // --- core model §2: recipes (undrawn distributions) vs draws (`~` vs `=`) ---

    #[test]
    fn dist_constructor_returns_an_undrawn_recipe() {
        // A bare distribution constructor is a *recipe*, not a value/RV. It builds no graph node
        // and displays as itself.
        let mut eng = Engine::new();
        let v = eng.run(&with_prelude("unif(0, 1)")).unwrap();
        assert!(matches!(v, Value::Recipe(_)), "got {v:?}");
        assert_eq!(v.to_string(), "unif(0, 1)");
        assert!(eng.graph().is_empty(), "a recipe must not touch the sample-DAG");
    }

    #[test]
    fn arithmetic_on_an_undrawn_distribution_is_an_error() {
        // The load-bearing rule: you can't operate on a recipe — draw it with `~` first.
        assert!(run("unif(0,1) + 3").is_err());
        assert!(run("Die = unif_int(1,6); Die + 1").is_err()); // `=` keeps it undrawn
        assert!(run("-unif(0,1)").is_err());
        assert!(run("if unif(0,1) { 1 } else { 0 }").is_err());
        // and the message points the user at `~`
        let err = run("unif(0,1) + 3").unwrap_err();
        assert!(err.to_string().contains('~'), "message should mention `~`: {err}");
    }

    #[test]
    fn assign_keeps_a_recipe_then_tilde_draws_independent_copies() {
        // `Die = unif_int(1,6)` binds the recipe; each `~ Die` draws an INDEPENDENT die, so
        // P(a == b) = 1/6 (not 1 — they are not the same node).
        let p = run_num("Die = unif_int(1,6); a ~ Die; b ~ Die; P(a == b)");
        assert!((p - 1.0 / 6.0).abs() < 5e-3, "P(a==b) over two draws of one recipe = {p}");
    }

    #[test]
    fn one_draw_reused_is_perfectly_correlated() {
        // Contrast with the above: a single `~` draw reused is one node, so P(X == X) = 1.
        let p = run_num("X ~ unif_int(1,6); P(X == X)");
        assert_eq!(p, 1.0);
    }

    #[test]
    fn forgetting_to_draw_a_recipe_errors_in_a_query() {
        // The canonical mistake: bind a distribution with `=`, then use it as if drawn. The
        // comparison `D == 4` hits an undrawn recipe and errors — caught, not silently wrong.
        assert!(run("D = unif_int(1,6); P(D == 4)").is_err());
        // The fix — draw it with `~` — works.
        let p = run_num("D ~ unif_int(1,6); P(D == 4)");
        assert!((p - 1.0 / 6.0).abs() < 5e-3, "P(D==4) = {p}");
    }

    #[test]
    fn recipe_is_rejected_everywhere_a_concrete_value_is_expected() {
        assert!(run("unif(0,1) < 0.5").is_err()); // comparison operand
        assert!(run("unif(0,1) == 0.5").is_err()); // equality operand
        assert!(run("P(unif(0,1))").is_err()); // not an event
        assert!(run("unif(unif(0,1), 1)").is_err()); // distribution parameter must be a number
        assert!(run("round(unif(0,1), 2)").is_err()); // builtin numeric argument
    }

    #[test]
    fn drawing_through_an_aliased_recipe() {
        // `=` can alias a recipe under a new name; `~` on either name draws.
        let p = run_num("Fair = unif_int(1,6); D = Fair; X ~ D; P(X == X)");
        assert_eq!(p, 1.0);
    }

    #[test]
    fn each_constructor_is_an_undrawn_recipe_that_displays_as_itself() {
        assert_eq!(display_of("unif(-1, 1)"), "unif(-1, 1)");
        assert_eq!(display_of("unif_int(1, 6)"), "unif_int(1, 6)");
        assert_eq!(display_of("bernoulli(0.3)"), "bernoulli(0.3)");
        // unif_int normalizes its (inclusive) integer bounds at construction.
        assert_eq!(display_of("unif_int(1.4, 5.6)"), "unif_int(1, 6)");
    }

    // --- Phase 2: random-variable runtime ---

    use crate::error::ErrorKind;

    /// Run a program, expecting the last value to be a `dist`, and return moments.
    fn moments_of(src: &str, n: usize, seed: u64) -> crate::Moments {
        let mut eng = Engine::new();
        let rv = eng.run_rv(&with_prelude(src)).unwrap();
        eng.moments(&rv, n, seed).unwrap()
    }

    #[test]
    fn uniform_moments() {
        // X ~ unif(-1,1): mean 0, variance (2^2)/12 = 1/3.
        let m = moments_of("X ~ unif(-1,1); X", 1_000_000, 42);
        assert!((m.mean - 0.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!((m.variance - 1.0 / 3.0).abs() < 3e-3, "var = {}", m.variance);
    }

    #[test]
    fn shifted_scaled_uniform() {
        // D ~ unif(1,6): mean 3.5, variance 25/12 ≈ 2.08333.
        let m = moments_of("D ~ unif(1,6); D", 1_000_000, 7);
        assert!((m.mean - 3.5).abs() < 3e-3, "mean = {}", m.mean);
        assert!((m.variance - 25.0 / 12.0).abs() < 5e-3, "var = {}", m.variance);
    }

    #[test]
    fn derived_rv_square() {
        // X ~ unif(-1,1); X**2: E=1/3, Var = 1/5 - 1/9 = 4/45 ≈ 0.08889.
        let m = moments_of("X ~ unif(-1,1); X ** 2", 1_000_000, 42);
        assert!((m.mean - 1.0 / 3.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!((m.variance - 4.0 / 45.0).abs() < 3e-3, "var = {}", m.variance);
    }

    #[test]
    fn affine_lift() {
        // Y = 2*X + 3 for X~unif(0,1): mean 4, variance (2^2)/12 = 1/3.
        let m = moments_of("X ~ unif(0,1); 2*X + 3", 1_000_000, 1);
        assert!((m.mean - 4.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!((m.variance - 1.0 / 3.0).abs() < 3e-3, "var = {}", m.variance);
    }

    #[test]
    fn shared_draw_correctness() {
        // X + X uses ONE draw of X per lane (structural sharing): mean 1, variance 4*Var(X)=1/3.
        // (Independent draws would give variance 1/6 — this proves CSE.)
        let m = moments_of("X ~ unif(0,1); X + X", 1_000_000, 99);
        assert!((m.mean - 1.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!((m.variance - 1.0 / 3.0).abs() < 3e-3, "var = {}", m.variance);

        // X - X is exactly 0 everywhere.
        let m = moments_of("X ~ unif(0,1); X - X", 100_000, 99);
        assert_eq!(m.mean, 0.0);
        assert_eq!(m.variance, 0.0);
    }

    #[test]
    fn comparison_lifts_to_indicator() {
        // X ~ unif(0,1); X < 0.5: an indicator column; mean ≈ 0.5 (pre-wires Phase 3 P()).
        let m = moments_of("X ~ unif(0,1); X < 0.5", 1_000_000, 5);
        assert!((m.mean - 0.5).abs() < 3e-3, "mean = {}", m.mean);

        // Prove indicator semantics, not just the mean: every sampled lane is exactly 0 or 1.
        let mut eng = Engine::new();
        let rv = eng.run_rv(&with_prelude("X ~ unif(0,1); X < 0.5")).unwrap();
        let draws = eng.sample(&rv, 50_000, 5).unwrap();
        assert!(
            draws.iter().all(|&x| x == 0.0 || x == 1.0),
            "comparison RV must produce a strict 0/1 column"
        );
    }

    #[test]
    fn determinism() {
        let mut eng = Engine::new();
        let rv = eng.run_rv(&with_prelude("X ~ unif(-1,1); X")).unwrap();
        let a = eng.sample(&rv, 100_000, 42).unwrap();
        let b = eng.sample(&rv, 100_000, 42).unwrap();
        assert_eq!(a, b, "same seed must give byte-identical draws");
        let c = eng.sample(&rv, 100_000, 43).unwrap();
        assert_ne!(a, c, "different seed must differ");
    }

    #[test]
    fn rv_type_and_arity_errors_stay_spanned() {
        for src in [
            "X ~ unif(0,1); X + \"a\"",
            "unif(1)",
            "unif(\"a\", 2)",
            "X ~ unif(0,1); !X", // Not on a numeric RV
        ] {
            let err = run(src).expect_err(&format!("{src:?} should error"));
            assert!(
                matches!(err.kind, ErrorKind::Runtime(_)),
                "{src:?} should be a Runtime error, got {:?}",
                err.kind
            );
            assert_ne!(
                err.span,
                crate::error::Span::default(),
                "{src:?} should carry a real source span, not 0..0"
            );
        }
    }

    #[test]
    fn deterministic_programs_never_touch_the_graph() {
        // Purely deterministic programs stay Value::Num and leave the graph empty.
        let mut eng = Engine::new();
        let v = eng.run("2*3+1").unwrap();
        assert_eq!(v, Value::Num(7.0));
        assert!(eng.graph().is_empty(), "graph must stay empty for deterministic programs");
    }

    // --- Phase 3: probability queries, discrete distributions, boolean ops ---
    // `P(...)` uses a fixed default seed + N=1e6, so these are deterministic, not flaky.

    /// Run a program expecting a numeric result.
    fn run_num(src: &str) -> f64 {
        num(src)
    }

    #[test]
    fn boolean_and_or_deterministic() {
        assert!(boolean("(1 < 2) && (3 > 2)"));
        assert!(!boolean("(1 < 2) && (3 < 2)"));
        assert!(boolean("(1 < 2) || (3 < 2)"));
        assert!(!boolean("(1 > 2) || (3 < 2)"));
        // precedence: && binds tighter than ||, comparison tighter than both
        assert!(boolean("1 < 2 || 5 < 1 && 9 < 1"));
    }

    #[test]
    fn pi_via_monte_carlo() {
        // The corrected example: π = 4·P(point in unit circle).
        let pi = run_num("X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X**2 + Y**2 < 1)");
        assert!((pi - std::f64::consts::PI).abs() < 0.02, "pi estimate = {pi}");
    }

    #[test]
    fn discrete_die_probability() {
        // unif_int is discrete, so equality is meaningful: P(Dice == 4) = 1/6.
        let p = run_num("Dice ~ unif_int(1,6); P(Dice == 4)");
        assert!((p - 1.0 / 6.0).abs() < 5e-3, "P(Dice==4) = {p}");
    }

    #[test]
    fn two_independent_dice() {
        // Two bindings = two independent dice: P(A==4 && B==4) = 1/36.
        let p = run_num("A ~ unif_int(1,6); B ~ unif_int(1,6); P(A == 4 && B == 4)");
        assert!((p - 1.0 / 36.0).abs() < 3e-3, "P(A==4 && B==4) = {p}");
    }

    #[test]
    fn shared_die_disjunction() {
        // One die (shared): P(D==4 || D==5) = 2/6, NOT (1/6 + 1/6) of two dice — and
        // crucially not P over two draws. Proves sharing flows through `||`.
        let p = run_num("D ~ unif_int(1,6); P(D == 4 || D == 5)");
        assert!((p - 2.0 / 6.0).abs() < 5e-3, "P(D==4 || D==5) = {p}");
    }

    #[test]
    fn bernoulli_probability() {
        let p = run_num("C ~ bernoulli(0.3); P(C)");
        assert!((p - 0.3).abs() < 3e-3, "P(bernoulli 0.3) = {p}");
    }

    #[test]
    fn unif_int_samples_are_integers_in_range() {
        let mut eng = Engine::new();
        let rv = eng.run_rv(&with_prelude("D ~ unif_int(1,6); D")).unwrap();
        let draws = eng.sample(&rv, 50_000, 11).unwrap();
        assert!(
            draws.iter().all(|&x| (1.0..=6.0).contains(&x) && x.fract() == 0.0),
            "unif_int(1,6) must yield integers in 1..=6"
        );
    }

    #[test]
    fn p_of_deterministic_event() {
        assert_eq!(run_num("P(1 == 1)"), 1.0);
        assert_eq!(run_num("P(1 == 2)"), 0.0);
    }

    #[test]
    fn p_rejects_non_event() {
        // P of a numeric RV (not an event) is an error, not a silent number.
        assert!(run("X ~ unif(0,1); P(X)").is_err());
        assert!(run("P(3)").is_err());
        // logical op over numeric RVs is an error
        assert!(run("X ~ unif(0,1); P(X && X)").is_err());
        // bad arity / sample count
        assert!(run("P()").is_err());
        assert!(run("D ~ unif_int(1,6); P(D == 4, 0)").is_err());
    }

    #[test]
    fn p_display_precision_scales_with_samples() {
        // Displayed to confidence precision: at N=1000 the standard error (~0.012) only
        // justifies one decimal -> "0.2"; far more samples reveal more digits.
        assert_eq!(display_of("D ~ unif_int(1,6); P(D == 4, 1000)"), "0.2");
        let fine = display_of("D ~ unif_int(1,6); P(D == 4, 100000000)");
        assert!(fine.len() > 5, "fine should show ~4 digits, got {fine}");
        // The underlying value stays full precision and accurate (rounding is display-only).
        assert!((num("D ~ unif_int(1,6); P(D == 4, 100000000)") - 1.0 / 6.0).abs() < 5e-4);
    }

    #[test]
    fn error_propagates_through_arithmetic() {
        // The intent: error-propagation sets the *number of shown digits*. The exact last digit
        // depends on the seed/sampling/chunking, so we pin the precision (decimal count) and that
        // the value is genuinely π — not a specific string.
        let decimals = |s: &str| s.split_once('.').map_or(0, |(_, frac)| frac.len());

        // P knows itself to ~4 digits at N=1e8, but `4 * P` has 4× the error → 3 decimals shown
        // (e.g. "3.141"/"3.142"), not a spurious "3.1416".
        let pi = display_of("X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X**2 + Y**2 < 1, 100000000)");
        assert_eq!(decimals(&pi), 3, "should propagate to 3 decimals, got {pi}");
        assert!((pi.parse::<f64>().unwrap() - std::f64::consts::PI).abs() < 0.01, "pi = {pi}");

        // Default budget → larger error → coarser: 2 decimals ("3.14").
        let pi_coarse = display_of("X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X**2 + Y**2 < 1)");
        assert_eq!(decimals(&pi_coarse), 2, "default budget should show 2 decimals, got {pi_coarse}");
        assert!(
            (pi_coarse.parse::<f64>().unwrap() - std::f64::consts::PI).abs() < 0.02,
            "pi_coarse = {pi_coarse}"
        );
    }

    #[test]
    fn engine_set_max_loops_sets_the_default_budget() {
        // `engine::set_max_loops(N)` is the global default for P/E/Var/Q — equivalent to passing N
        // as the explicit count, but once, up front. At a small budget the standard error is wide,
        // so the estimate displays to few digits; bumping the budget reveals more.
        let coarse = display_of("engine::set_max_loops(1000); D ~ unif_int(1,6); P(D == 4)");
        assert_eq!(coarse, "0.2", "1000 draws justifies one decimal, got {coarse}");
        let fine = display_of("engine::set_max_loops(100000000); D ~ unif_int(1,6); P(D == 4)");
        assert!(fine.len() > 5, "1e8 draws should reveal ~4 digits, got {fine}");

        // An explicit per-call count still overrides the engine default (here: tighten back up
        // despite the coarse global setting).
        let overridden =
            display_of("engine::set_max_loops(1000); D ~ unif_int(1,6); P(D == 4, 100000000)");
        assert!(overridden.len() > 5, "explicit count should override, got {overridden}");
    }

    #[test]
    fn engine_module_scoping_and_validation() {
        // Reachable both as a `mod::name` path and (with `use`) unqualified, like any module.
        assert!(run_raw("engine::set_max_loops(50); use rand; X ~ unif(0,1); P(X < 0.5)").is_ok());
        assert!(run_raw("use engine; set_max_loops(50); 1").is_ok());
        // Out of scope without a `use`/path, with a fix-it message naming the module.
        let err = run_raw("set_max_loops(50)").unwrap_err().to_string();
        assert!(err.contains("engine") && err.contains("use"), "{err}");
        // The budget must draw at least once.
        assert!(run_raw("engine::set_max_loops(0)").unwrap_err().to_string().contains(">= 1"));
    }

    // --- lifted `if` over a random variable (per-lane select) ---

    #[test]
    fn if_over_rv_selects_per_lane() {
        // payoff = +10 with prob 1/6, -2 with prob 5/6 -> mean 0 exactly, variance 20.
        let m = moments_of("D ~ unif_int(1,6); if D == 6 { 10 } else { 0 - 2 }", 1_000_000, 3);
        assert!(m.mean.abs() < 0.02, "mean = {}", m.mean);
        assert!((m.variance - 20.0).abs() < 0.2, "var = {}", m.variance);
    }

    #[test]
    fn if_branches_read_consistent_draws() {
        // if D == 6 { D } else { 0 } : the branch reuses the SAME per-lane draw of D,
        // so the result is 6 w.p. 1/6 and 0 otherwise -> mean 1.
        let m = moments_of("D ~ unif_int(1,6); if D == 6 { D } else { 0 }", 1_000_000, 4);
        assert!((m.mean - 1.0).abs() < 0.02, "mean = {}", m.mean);
    }

    #[test]
    fn max_via_if() {
        // M = max(A, B) for two d6; P(M == 6) = 1 - (5/6)^2 = 11/36.
        let p = run_num("A ~ unif_int(1,6); B ~ unif_int(1,6); P((if A > B { A } else { B }) == 6)");
        assert!((p - 11.0 / 36.0).abs() < 5e-3, "P(max==6) = {p}");
    }

    #[test]
    fn abs_via_if() {
        // |X| for X ~ unif(-1,1); P(|X| > 0.5) = 0.5.
        let p = run_num("X ~ unif(-1,1); P((if X < 0 { 0 - X } else { X }) > 0.5)");
        assert!((p - 0.5).abs() < 5e-3, "P(|X|>0.5) = {p}");
    }

    #[test]
    fn if_over_rv_errors() {
        // needs an else branch
        assert!(run("D ~ unif_int(1,6); if D == 6 { 10 }").is_err());
        // branches must share a type
        assert!(run("D ~ unif_int(1,6); if D == 6 { 10 } else { 1 < 2 }").is_err());
        // a numeric-RV condition is not an event
        assert!(run("X ~ unif(0,1); if X { 1 } else { 0 }").is_err());
    }

    #[test]
    fn if_else_branches_on_a_probability() {
        // `P(...) > c` is a deterministic bool, so a normal if/else picks one branch — handy
        // for printing a verdict instead of a hardcoded message.
        assert_eq!(num("D ~ unif_int(1,6); if P(D < 4) > 0.4 { 1 } else { 0 }"), 1.0);
        assert_eq!(num("D ~ unif_int(1,6); if P(D == 6) > 0.5 { 1 } else { 0 }"), 0.0);
        // a print in a branch yields unit
        assert_eq!(
            run("if 1 < 2 { Print(\"yes\") } else { Print(\"no\") }").unwrap(),
            Value::Unit
        );
    }

    #[test]
    fn deterministic_if_still_short_circuits() {
        // A deterministic condition takes exactly one branch (no RV, graph untouched).
        let mut eng = Engine::new();
        let v = eng.run("if 3 > 2 { 10 } else { 20 }").unwrap();
        assert_eq!(v, Value::Num(10.0));
        assert!(eng.graph().is_empty());
    }

    // --- strings, print, round (printing messages) ---

    fn string_of(src: &str) -> String {
        match run(src).unwrap() {
            Value::Str(s) => s,
            other => panic!("expected string, got {other:?} for {src:?}"),
        }
    }

    #[test]
    fn string_concatenation() {
        assert_eq!(string_of("\"a\" + \"b\""), "ab");
        assert_eq!(string_of("\"x = \" + 5"), "x = 5");
        assert_eq!(string_of("5 + \" apples\""), "5 apples");
        // chains left-to-right
        assert_eq!(string_of("\"p=\" + 1 + \",q=\" + 2"), "p=1,q=2");
    }

    #[test]
    fn round_builtin() {
        assert_eq!(num("round(3.14159, 2)"), 314.0 / 100.0); // avoid approx_constant lint
        assert_eq!(num("round(0.16666, 3)"), 0.167);
        assert_eq!(num("round(2.5, 0)"), 3.0);
    }

    #[test]
    fn print_returns_unit() {
        // `Print` emits to stdout and yields unit (so it can be the last statement).
        assert_eq!(run("Print(\"hello\")").unwrap(), Value::Unit);
        assert_eq!(run("Print(\"p =\", round(0.5, 2))").unwrap(), Value::Unit);
    }

    #[test]
    fn string_cannot_enter_a_random_variable() {
        // Concatenation is deterministic-only; a string can't be lifted into an RV.
        assert!(run("X ~ unif(0,1); X + \"a\"").is_err());
    }

    // --- core model §4: user-defined functions (`=` deterministic, `~` stochastic) ---

    #[test]
    fn deterministic_function_of_numbers() {
        assert_eq!(num("double(x) = x * 2; double(21)"), 42.0);
        assert_eq!(num("add(a, b) = a + b; add(3, 4)"), 7.0);
        // a body can be an if-expression
        assert_eq!(num("max(a, b) = if a > b { a } else { b }; max(3, 9)"), 9.0);
        // zero-arg function (a thunk)
        assert_eq!(num("answer() = 42; answer() + 1"), 43.0);
        // functions may call other functions
        assert_eq!(num("inc(x) = x + 1; twice(x) = inc(inc(x)); twice(10)"), 12.0);
    }

    #[test]
    fn functions_are_pure_in_their_params_no_outer_capture() {
        // A function body sees only its parameters, not outer variables (no closures yet).
        assert!(run("n = 10; f(x) = x + n; f(1)").is_err());
        // params shadow nothing leaks back out: the call's frame is discarded.
        assert_eq!(num("x = 100; id(x) = x; id(5); x"), 100.0);
    }

    #[test]
    fn deterministic_function_lifts_over_random_variables() {
        // max(A,B) as a *function* (not inline): P(max of 2d6 == 6) = 11/36.
        let p = run_num(
            "max(a, b) = if a > b { a } else { b }; \
             A ~ unif_int(1,6); B ~ unif_int(1,6); P(max(A, B) == 6)",
        );
        assert!((p - 11.0 / 36.0).abs() < 5e-3, "P(max==6) = {p}");

        // square(X) over an RV argument: E[X^2] = 1/3 for X ~ unif(-1,1).
        let m = moments_of("sq(x) = x * x; X ~ unif(-1,1); sq(X)", 1_000_000, 8);
        assert!((m.mean - 1.0 / 3.0).abs() < 3e-3, "mean = {}", m.mean);
    }

    #[test]
    fn stochastic_function_draws_fresh_each_call() {
        // roll() ~ unif_int(1,6): roll() + roll() is two INDEPENDENT dice -> P(sum==7)=1/6.
        let p = run_num("roll() ~ unif_int(1,6); P(roll() + roll() == 7)");
        assert!((p - 1.0 / 6.0).abs() < 5e-3, "P(sum==7) = {p}");

        // The parameter flows into the distribution: top(n) ~ unif_int(1,n); P(top(2)==1)=1/2.
        let p = run_num("top(n) ~ unif_int(1, n); P(top(2) == 1)");
        assert!((p - 0.5).abs() < 5e-3, "P(top(2)==1) = {p}");
    }

    #[test]
    fn stochastic_function_two_calls_are_independent_not_shared() {
        // Unlike a bound draw (X+X = 2X), two CALLS draw independently: Var(roll()+roll())
        // = 2*Var(d6) = 2*35/12 ≈ 5.833 (shared would be 4*35/12 ≈ 11.667).
        let m = moments_of("roll() ~ unif_int(1,6); roll() + roll()", 1_000_000, 12);
        assert!((m.mean - 7.0).abs() < 0.02, "mean = {}", m.mean);
        assert!((m.variance - 2.0 * 35.0 / 12.0).abs() < 0.1, "var = {}", m.variance);
    }

    #[test]
    fn function_call_arity_and_unknown_errors() {
        assert!(run("f(x) = x; f(1, 2)").is_err()); // too many args
        assert!(run("f(x, y) = x; f(1)").is_err()); // too few args
        assert!(run("nope(1)").is_err()); // unknown function
        // a recursive function with no base case is caught, not a stack overflow
        assert!(run("loop(x) = loop(x); loop(1)").is_err());
    }

    #[test]
    fn user_function_shadows_a_builtin() {
        // A user definition wins over a builtin of the same name.
        assert_eq!(num("round(x) = x + 1; round(10)"), 11.0);
    }

    #[test]
    fn defining_a_function_yields_unit_and_touches_no_graph() {
        let mut eng = Engine::new();
        let v = eng.run("sq(x) = x * x").unwrap();
        assert_eq!(v, Value::Unit);
        assert!(eng.graph().is_empty(), "defining a function must not sample");
    }

    // --- degenerate distributions (point masses) and parameter-domain errors ---

    /// All draws of the last-statement RV, for asserting exact support.
    fn draws_of(src: &str, n: usize, seed: u64) -> Vec<f64> {
        let mut eng = Engine::new();
        let rv = eng.run_rv(&with_prelude(src)).unwrap();
        eng.sample(&rv, n, seed).unwrap()
    }

    #[test]
    fn unif_int_degenerate_range_is_a_constant() {
        // unif_int(5,5) is a point mass at 5: P(==5) = 1 and every draw is exactly 5.
        assert_eq!(run_num("D ~ unif_int(5,5); P(D == 5)"), 1.0);
        assert!(draws_of("D ~ unif_int(5,5); D", 5000, 7).iter().all(|&x| x == 5.0));
    }

    #[test]
    fn unif_degenerate_range_is_a_constant() {
        // unif(2,2) has zero span, so every draw is exactly 2.
        assert!(draws_of("X ~ unif(2,2); X", 4096, 9).iter().all(|&x| x == 2.0));
    }

    #[test]
    fn unif_int_supports_negative_and_two_point_ranges() {
        // a negative inclusive range stays integral and in-range
        assert!(draws_of("D ~ unif_int(-3,-1); D", 20_000, 3)
            .iter()
            .all(|&x| (-3.0..=-1.0).contains(&x) && x.fract() == 0.0));
        // a fair coin via unif_int(0,1): P(==1) ≈ 0.5
        let p = run_num("B ~ unif_int(0,1); P(B == 1)");
        assert!((p - 0.5).abs() < 5e-3, "P(B==1) = {p}");
    }

    #[test]
    fn bernoulli_edges_are_exact_probabilities() {
        assert_eq!(run_num("C ~ bernoulli(0); P(C)"), 0.0);
        assert_eq!(run_num("C ~ bernoulli(1); P(C)"), 1.0);
    }

    #[test]
    fn distribution_parameter_domains_are_checked() {
        assert!(run("unif_int(6,1)").is_err()); // lo > hi
        assert!(run("bernoulli(1.5)").is_err()); // p outside [0,1]
        assert!(run("bernoulli(-0.1)").is_err());
        assert!(run("unif()").is_err()); // arity
        assert!(run("unif(1,2,3)").is_err());
    }

    // --- language surface: comments, empty programs, else-if, statement separators ---

    #[test]
    fn empty_and_comment_only_programs_are_unit() {
        assert_eq!(run("").unwrap(), Value::Unit);
        assert_eq!(run("# just a comment").unwrap(), Value::Unit);
        assert_eq!(run("  ;;  ").unwrap(), Value::Unit);
    }

    #[test]
    fn inline_comments_do_not_affect_evaluation() {
        assert_eq!(num("1 + # plus\n 2 // two\n"), 3.0);
    }

    #[test]
    fn else_if_chains_pick_the_right_branch() {
        let def = "sgn(x) = if x < 0 { -1 } else if x == 0 { 0 } else { 1 };";
        assert_eq!(num(&format!("{def} sgn(-5)")), -1.0);
        assert_eq!(num(&format!("{def} sgn(0)")), 0.0);
        assert_eq!(num(&format!("{def} sgn(7)")), 1.0);
    }

    #[test]
    fn trailing_semicolons_are_optional() {
        assert_eq!(num("1 + 1"), 2.0);
        assert_eq!(num("1 + 1;"), 2.0);
        assert_eq!(num("a = 2;; a"), 2.0); // extra separators are fine
    }

    #[test]
    fn string_equality_and_inequality() {
        assert!(boolean("\"a\" == \"a\""));
        assert!(boolean("\"a\" != \"b\""));
        assert!(!boolean("\"a\" == \"b\""));
    }

    // --- continuous distributions, moments, and math builtins (PLAN.md step 3) ---

    #[test]
    fn normal_has_the_right_moments() {
        // Z ~ normal(2, 3): mean 2, variance 9.
        let m = moments_of("Z ~ normal(2, 3); Z", 1_000_000, 1);
        assert!((m.mean - 2.0).abs() < 0.02, "mean = {}", m.mean);
        assert!((m.variance - 9.0).abs() < 0.1, "var = {}", m.variance);
    }

    #[test]
    fn normal_is_drawn_independently_per_binding() {
        // Sum of two independent standard normals is N(0, 2): variance 2, not 4 (shared).
        let m = moments_of("X ~ normal(0,1); Y ~ normal(0,1); X + Y", 1_000_000, 2);
        assert!((m.variance - 2.0).abs() < 0.05, "var = {}", m.variance);
        // tail prob of a standard normal: P(Z > 1) ≈ 0.1587.
        let p = run_num("Z ~ normal(0,1); P(Z > 1)");
        assert!((p - 0.1587).abs() < 3e-3, "P(Z>1) = {p}");
    }

    #[test]
    fn expectation_of_a_numeric_rv() {
        // E of a die is 3.5; E is total over numeric RVs (P only handles events).
        let e = run_num("D ~ unif_int(1,6); E(D)");
        assert!((e - 3.5).abs() < 5e-3, "E(D) = {e}");
        // E of a constant is exact.
        assert_eq!(run_num("E(42)"), 42.0);
        // E of a bool RV coincides with P.
        let e = run_num("C ~ bernoulli(0.3); E(C)");
        assert!((e - 0.3).abs() < 3e-3, "E(C) = {e}");
    }

    #[test]
    fn variance_of_a_numeric_rv() {
        // Var of a fair d6 is 35/12 ≈ 2.9167.
        let v = run_num("D ~ unif_int(1,6); Var(D)");
        assert!((v - 35.0 / 12.0).abs() < 0.02, "Var(D) = {v}");
        assert_eq!(run_num("Var(7)"), 0.0); // a constant has zero variance
    }

    #[test]
    fn sqrt_and_pi_builtins() {
        assert_eq!(run_num("sqrt(16)"), 4.0);
        assert!((run_num("sqrt(2)") - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!((run_num("pi") - std::f64::consts::PI).abs() < 1e-12);
        assert!((run_num("2 * pi") - std::f64::consts::TAU).abs() < 1e-12);
        // `pi` resolves inside a function body (params-only scope) — it's a constant, not a var.
        assert!((run_num("circ(r) = 2 * pi * r; circ(1)") - std::f64::consts::TAU).abs() < 1e-12);
        assert!(run("sqrt(-1)").is_err());
    }

    #[test]
    fn normal_parameter_domain_and_recipe_display() {
        assert!(run("normal(0, -1)").is_err()); // sigma must be >= 0
        assert_eq!(display_of("normal(0, 1)"), "normal(0, 1)"); // undrawn recipe prints itself
        // sigma = 0 is a degenerate point mass at mu.
        assert!(draws_of("Z ~ normal(5, 0); Z", 2000, 4).iter().all(|&x| x == 5.0));
    }

    // --- more `rand` distributions: exp, poisson, geometric, and the `_int` family ---

    #[test]
    fn exponential_has_the_right_moments() {
        // Exp(rate): mean = 1/rate, variance = 1/rate^2. For rate = 2: mean 0.5, var 0.25.
        let m = moments_of("X ~ exp(2); X", 1_000_000, 1);
        assert!((m.mean - 0.5).abs() < 0.01, "mean = {}", m.mean);
        assert!((m.variance - 0.25).abs() < 0.01, "var = {}", m.variance);
        // memoryless tail: P(X > 1) = e^-(rate*1) = e^-2 ≈ 0.1353 for rate = 2.
        let p = run_num("X ~ exp(2); P(X > 1)");
        assert!((p - (-2.0f64).exp()).abs() < 3e-3, "P(X>1) = {p}");
    }

    #[test]
    fn poisson_has_the_right_moments_and_support() {
        // Poisson(lambda): mean = variance = lambda. For lambda = 3.
        let m = moments_of("K ~ poisson(3); K", 1_000_000, 2);
        assert!((m.mean - 3.0).abs() < 0.02, "mean = {}", m.mean);
        assert!((m.variance - 3.0).abs() < 0.05, "var = {}", m.variance);
        // Counts are non-negative integers.
        assert!(draws_of("K ~ poisson(3); K", 20_000, 5)
            .iter()
            .all(|&x| x >= 0.0 && x.fract() == 0.0));
        // P(K == 0) = e^-lambda.
        let p = run_num("K ~ poisson(3); P(K == 0)");
        assert!((p - (-3.0f64).exp()).abs() < 3e-3, "P(K==0) = {p}");
    }

    #[test]
    fn geometric_has_the_right_distribution() {
        // Geometric(p) = failures before first success: mean = (1-p)/p, support {0,1,2,…}.
        // For p = 0.25: mean = 3.
        let m = moments_of("G ~ geometric(0.25); G", 1_000_000, 3);
        assert!((m.mean - 3.0).abs() < 0.05, "mean = {}", m.mean);
        assert!(draws_of("G ~ geometric(0.25); G", 20_000, 6)
            .iter()
            .all(|&x| x >= 0.0 && x.fract() == 0.0));
        // P(G == 0) = p (success on the first trial).
        let p = run_num("G ~ geometric(0.25); P(G == 0)");
        assert!((p - 0.25).abs() < 3e-3, "P(G==0) = {p}");
        // p = 1 is a point mass at 0.
        assert!(draws_of("G ~ geometric(1); G", 2000, 7).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn int_variants_round_to_integers() {
        // normal_int / exp_int draw the continuous distribution then round each lane.
        assert!(draws_of("Z ~ normal_int(0, 5); Z", 20_000, 8)
            .iter()
            .all(|&x| x.fract() == 0.0));
        assert!(draws_of("X ~ exp_int(0.5); X", 20_000, 9)
            .iter()
            .all(|&x| x.fract() == 0.0 && x >= 0.0));
        // Rounding preserves the mean closely: E(normal_int(10, 3)) ≈ 10.
        let m = run_num("Z ~ normal_int(10, 3); E(Z)");
        assert!((m - 10.0).abs() < 0.02, "E(normal_int) = {m}");
    }

    #[test]
    fn new_distribution_domains_and_recipe_display() {
        assert!(run("exp(0)").is_err()); // rate must be > 0
        assert!(run("exp(-1)").is_err());
        assert!(run("poisson(0)").is_err()); // lambda must be > 0
        assert!(run("geometric(0)").is_err()); // p must be > 0
        assert!(run("geometric(1.5)").is_err()); // p must be <= 1
        // undrawn recipes print themselves
        assert_eq!(display_of("exp(2)"), "exp(2)");
        assert_eq!(display_of("poisson(3)"), "poisson(3)");
        assert_eq!(display_of("geometric(0.5)"), "geometric(0.5)");
        assert_eq!(display_of("normal_int(0, 1)"), "normal_int(0, 1)");
        assert_eq!(display_of("exp_int(2)"), "exp_int(2)");
        // recipes bound with `=` stay undrawn; `~` draws independent copies
        let p = run_num("D = poisson(3); a ~ D; b ~ D; P(a == b && a == 0)");
        assert!((p - (-3.0f64).exp().powi(2)).abs() < 3e-3, "P(a==b==0) = {p}");
    }

    // --- Q(): distribution quantiles (companion to E/Var/P) ---

    #[test]
    fn quantile_of_known_distributions() {
        // Median of unif(0,1) is 0.5.
        assert!((run_num("X ~ unif(0, 1); Q(X, 0.5)") - 0.5).abs() < 5e-3);
        // The 97.5th percentile of a standard normal is ≈ 1.96.
        assert!((run_num("Z ~ normal(0, 1); Q(Z, 0.975)") - 1.96).abs() < 0.02);
        // Median of Exp(rate) is ln(2)/rate; for rate = 1 that's ≈ 0.693.
        assert!((run_num("X ~ exp(1); Q(X, 0.5)") - 2.0f64.ln()).abs() < 5e-3);
    }

    #[test]
    fn quantile_endpoints_are_min_and_max() {
        // q = 0 / q = 1 are the smallest / largest draw; for a fair d6 that's 1 and 6.
        assert_eq!(run_num("D ~ unif_int(1, 6); Q(D, 0)"), 1.0);
        assert_eq!(run_num("D ~ unif_int(1, 6); Q(D, 1)"), 6.0);
    }

    #[test]
    fn quantile_of_a_deterministic_value_is_itself() {
        // A point mass has the same quantile at every level.
        assert_eq!(run_num("Q(5, 0.1)"), 5.0);
        assert_eq!(run_num("Q(5, 0.9)"), 5.0);
    }

    #[test]
    fn quantile_domain_and_arity_are_checked() {
        assert!(run("X ~ unif(0, 1); Q(X)").is_err()); // missing level
        assert!(run("X ~ unif(0, 1); Q(X, 1.5)").is_err()); // q out of [0,1]
        assert!(run("X ~ unif(0, 1); Q(X, -0.1)").is_err());
        assert!(run("X ~ unif(0, 1); Q(X, 0.5, 0)").is_err()); // sample count >= 1
        assert!(run("Q(\"a\", 0.5)").is_err()); // non-numeric quantity
    }

    #[test]
    fn quantile_is_always_active_like_other_builtins() {
        // `Q` lives in the always-on `builtin` module — no `use vec/math` needed (mirrors P/E/Var);
        // only `rand` is needed for the distribution constructor.
        match run_raw("use rand; X ~ unif(0, 1); Q(X, 0.5)").unwrap() {
            Value::Num(m) => assert!((m - 0.5).abs() < 5e-3, "median = {m}"),
            other => panic!("expected a number, got {other:?}"),
        }
    }

    // --- Step 4: collections (arrays, indexing, `for`, the library) ---

    /// The number of sample-DAG nodes a program builds (for unroll/determinism assertions).
    fn graph_len(src: &str) -> usize {
        let mut eng = Engine::new();
        eng.run(&with_prelude(src)).unwrap();
        eng.graph().len()
    }

    #[test]
    fn boolean_literals() {
        assert!(boolean("true"));
        assert!(!boolean("false"));
        assert!(boolean("true && (1 < 2)"));
        assert!(!boolean("false || (1 > 2)"));
        assert!(boolean("!false"));
        // they are point masses: a `for`-accumulator over events works (the `any`/`all` shape).
        assert!(boolean("acc = false; for x in [1 < 2, 3 < 4] { acc = acc || x }; acc"));
        // `true`/`false` are reserved keywords, not identifiers.
        assert!(run("true = 5").is_err());
    }

    #[test]
    fn a_bool_is_a_bernoulli_point_mass() {
        // LANG.md §1: a bool is a Bernoulli — `true` ≡ Bernoulli(1), `false` ≡ Bernoulli(0) —
        // and any event (`P`/comparison/`&&`/`||`) is Bernoulli too. So P and E agree across
        // deterministic bools, comparison events, and a drawn bernoulli, with no special-casing.
        assert_eq!(run_num("P(true)"), 1.0);
        assert_eq!(run_num("P(false)"), 0.0);
        assert_eq!(run_num("E(true)"), 1.0); // E of a Bernoulli(1) point mass
        assert_eq!(run_num("E(false)"), 0.0);
        assert_eq!(run_num("Var(true)"), 0.0); // a point mass has zero variance
        // E of an event equals its probability, whether it's a comparison, `&&`/`||`, or drawn.
        let p = run_num("D ~ unif_int(1, 6); P(D > 3)");
        let e = run_num("D ~ unif_int(1, 6); E(D > 3)");
        assert!((p - e).abs() < 1e-9 && (p - 0.5).abs() < 5e-3, "P={p} E={e}");
        let pe = run_num("C ~ bernoulli(0.3); E(C)");
        assert!((pe - 0.3).abs() < 3e-3, "E(bernoulli 0.3) = {pe}");
    }

    #[test]
    fn array_literals_index_and_len() {
        assert_eq!(display_of("[1, 2, 3]"), "[1, 2, 3]");
        assert_eq!(display_of("[]"), "[]");
        assert_eq!(num("[10, 20, 30][1]"), 20.0);
        assert_eq!(num("Len([4, 5, 6, 7])"), 4.0);
        assert_eq!(num("Len([])"), 0.0);
        // nesting: a matrix is an array of arrays; chained indexing M[i][j].
        assert_eq!(num("M = [[1, 2], [3, 4]]; M[1][0]"), 3.0);
        assert_eq!(display_of("[[1, 2], [3, 4]]"), "[[1, 2], [3, 4]]");
        // an array element can be any value (here a string), Displayed in place.
        assert_eq!(display_of("[1, \"a\", 2]"), "[1, a, 2]");
    }

    #[test]
    fn array_index_errors() {
        assert!(run("[1, 2, 3][5]").is_err()); // out of bounds
        assert!(run("[1, 2, 3][1.5]").is_err()); // non-integer index
        assert!(run("[1, 2, 3][0 - 1]").is_err()); // negative index
        assert!(run("5[0]").is_err()); // indexing a non-array
        // A random-variable index is no longer an error — it's a per-lane gather (see
        // `random_index_is_a_gather`). Gathering a matrix row into one lane is still rejected.
        assert!(run("X ~ unif_int(0, 1); [[1, 2], [3, 4]][X]").is_err());
    }

    #[test]
    fn permutation_is_a_uniform_permutation() {
        // `permutation(n)` draws a length-n array that is a permutation of 0..n every lane: no
        // duplicates, the values 0..n-1 (so sum = n(n-1)/2 and max = n-1).
        assert_eq!(num("deck ~ permutation(5); E(has_duplicate(deck))"), 0.0);
        assert_eq!(num("deck ~ permutation(5); E(sum(deck))"), 10.0); // 0+1+2+3+4
        assert_eq!(num("deck ~ permutation(5); E(max(deck))"), 4.0);
        // uniform: element 0 is equally likely to land in any position (P = 1/5 each).
        assert!((num("deck ~ permutation(5); P(deck[0] == 0)") - 0.2).abs() < 0.01);
        // it's a recipe like any distribution: `=` keeps it undrawn, arithmetic on it errors.
        assert!(run("deck = permutation(4); deck + 1").is_err());
    }

    #[test]
    fn random_index_is_a_gather() {
        // A random index selects a per-lane element. Over a constant array it's a plain lookup:
        // a uniform index makes the result uniform over the elements.
        assert!((num("i ~ unif_int(0, 2); E([10, 20, 30][i])") - 20.0).abs() < 0.05);
        assert!((num("i ~ unif_int(0, 2); P([10, 20, 30][i] == 30)") - 1.0 / 3.0).abs() < 0.01);
        // Gathering into an array of random variables works too (mean of 0..3).
        assert!((num("deck ~ permutation(4); i ~ unif_int(0, 3); E(deck[i])") - 1.5).abs() < 0.02);
    }

    #[test]
    fn hundred_prisoners_cycle_following() {
        // The 100 Prisoners Riddle, cycle-following strategy: prisoner i follows the permutation
        // from drawer i, opening `opens` drawers; everyone wins iff the longest cycle <= opens.
        // Small instance (n=6, opens=3) with a known analytic value: 1 - (1/4 + 1/5 + 1/6).
        let p = num(
            "n = 6; opens = 3; deck ~ permutation(n); all_found = true; \
             for i in 0..n { cur = i; hit = false; \
               for s in 0..opens { cur = deck[cur]; hit = hit || (cur == i); }; \
               all_found = all_found && hit; }; \
             P(all_found, 200000)",
        );
        assert!((p - 0.3833).abs() < 0.01, "got {p}, expected ~0.3833");
    }

    #[test]
    fn range_syntax_is_half_open() {
        assert_eq!(display_of("0..5"), "[0, 1, 2, 3, 4]");
        assert_eq!(num("Len(0..23)"), 23.0);
        assert_eq!(display_of("2..2"), "[]"); // empty
        assert_eq!(display_of("3..1"), "[]"); // a >= b
        assert!(run("0..2.5").is_err()); // non-integer bound
        // bounds are full expressions: `1+1 .. 2*3` is `2..6`
        assert_eq!(display_of("1 + 1 .. 2 * 3"), "[2, 3, 4, 5]");
        // a range over an undrawn distribution / non-number is an error
        assert!(run("0..unif(0, 1)").is_err());
    }

    #[test]
    fn reducers_on_constant_arrays() {
        assert_eq!(num("sum([1, 2, 3, 4])"), 10.0);
        assert_eq!(num("sum([])"), 0.0);
        assert_eq!(num("max([3, 1, 4, 1, 5])"), 5.0);
        assert_eq!(num("min([3, 1, 4, 1, 5])"), 1.0);
        assert_eq!(num("mean([2, 4, 6])"), 4.0);
        assert_eq!(num("count([1 < 2, 2 < 1, 3 < 4])"), 2.0); // two true
        assert!(boolean("any([1 > 2, 2 > 1])"));
        assert!(!boolean("any([1 > 2, 3 > 4])"));
        assert!(boolean("all([1 < 2, 3 < 4])"));
        assert!(!boolean("all([1 < 2, 4 < 3])"));
    }

    #[test]
    fn linear_algebra_on_constant_vectors() {
        // hand-computed: dot([1,2,3],[4,5,6]) = 4 + 10 + 18 = 32.
        assert_eq!(num("dot([1, 2, 3], [4, 5, 6])"), 32.0);
        assert_eq!(num("normsq([3, 4])"), 25.0);
        assert_eq!(num("norm([3, 4])"), 5.0); // 3-4-5 triangle
        // scaling a vector is just broadcast multiplication (no `scale` builtin needed)
        assert_eq!(display_of("[1, 2, 3] * 2"), "[2, 4, 6]");
        assert_eq!(display_of("vadd([1, 2], [3, 4])"), "[4, 6]");
        assert_eq!(display_of("vsub([5, 7], [1, 2])"), "[4, 5]");
        // `sign` is a math ufunc that maps over arrays (-1 / 0 / +1)
        assert_eq!(display_of("sign([3, 0 - 2, 5, 0])"), "[1, -1, 1, 0]");
        // matvec: [[1,2],[3,4]] * [1,1] = [3, 7].
        assert_eq!(display_of("matvec([[1, 2], [3, 4]], [1, 1])"), "[3, 7]");
        // normalize yields a unit vector: norm(normalize([3,4])) == 1.
        assert!((num("norm(normalize([3, 4]))") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vector_constructors_and_transpose() {
        assert_eq!(display_of("ones(3)"), "[1, 1, 1]");
        assert_eq!(display_of("zeros(2)"), "[0, 0]");
        assert_eq!(display_of("iota(4)"), "[0, 1, 2, 3]");
        assert_eq!(display_of("iota(4)"), display_of("0..4")); // iota == 0..n
        // transpose swaps rows and columns
        assert_eq!(display_of("transpose([[1, 2, 3], [4, 5, 6]])"), "[[1, 4], [2, 5], [3, 6]]");
        assert_eq!(display_of("transpose([])"), "[]");
        // transpose is an involution on a square matrix
        assert_eq!(
            display_of("transpose(transpose([[1, 2], [3, 4]]))"),
            display_of("[[1, 2], [3, 4]]")
        );
        assert!(run("transpose([[1, 2], [3]])").is_err()); // ragged matrix
        assert!(run("ones(2.5)").is_err()); // non-integer size
    }

    #[test]
    fn matmul_operator_dispatches_on_shape() {
        // `@` is the matrix product, picking the right contraction from the operand shapes.
        assert_eq!(num("[1, 2] @ [3, 4]"), 11.0); // vec·vec = dot
        assert_eq!(display_of("[[1, 2], [3, 4]] @ [1, 1]"), "[3, 7]"); // mat·vec = matvec
        assert_eq!(display_of("[1, 2] @ [[1, 2], [3, 4]]"), "[7, 10]"); // vec·mat
        assert_eq!(
            display_of("[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]"),
            "[[19, 22], [43, 50]]"
        ); // mat·mat
        // `@` binds like `*`, so `1 + M @ v` is `1 + (M @ v)`.
        assert_eq!(display_of("1 + [[1, 2], [3, 4]] @ [1, 1]"), "[4, 8]");
        // It is NOT elementwise `*`, which broadcasts the row by the scalar lane.
        assert_eq!(display_of("[[1, 2], [3, 4]] * [1, 1]"), "[[1, 2], [3, 4]]");
        // It lifts over random variables (each entry is a dot of RV lanes): E([X,1]·[2,3]) = 3.
        assert!((run_num("X ~ normal(0, 1); E([X, 1] @ [2, 3])") - 3.0).abs() < 0.05);
        // A scalar operand is an error — `@` is for linear algebra, `*` for scaling.
        assert!(run("3 @ [1, 2]").is_err());
        assert!(run("[1, 2] @ 3").is_err());
    }

    #[test]
    fn log_and_mse_utilities() {
        // math::log (natural) and math::log10
        assert_eq!(num("log10(1000)"), 3.0);
        assert_eq!(num("log10(1)"), 0.0);
        assert!((num("log(e)") - 1.0).abs() < 1e-12);
        assert!((num("log(e ** 3)") - 3.0).abs() < 1e-12);
        assert!(run("log(0)").is_err()); // domain x > 0
        assert!(run("log10(0 - 5)").is_err());
        // vec::mse — mean squared error between two signals
        assert!((num("mse([1, 2, 3], [1, 2, 5])") - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(num("mse([5, 5], [5, 5])"), 0.0); // identical signals
        assert!(run("mse([1, 2], [1, 2, 3])").is_err()); // length mismatch
        // mse equals the noise power of an additive channel: received = signal + noise.
        let p = run_num(
            "sig = ones(8); noise ~[8] normal(0, 2); E(mse(vadd(sig, noise), sig))",
        );
        assert!((p - 4.0).abs() < 0.1, "additive-noise MSE = {p} (want sigma^2 = 4)");
    }

    #[test]
    fn signal_wave_generators() {
        // Eager two-arg form sine(n, freq) / cosine — a sampled unit waveform array.
        assert_eq!(num("Len(sine(64, 3))"), 64.0);
        // sine starts at 0, cosine at 1; sine(4,1) = [0, 1, ~0, -1] (quarter-cycle steps).
        assert!(num("sine(8, 1)[0]").abs() < 1e-12);
        assert!((num("cosine(8, 1)[0]") - 1.0).abs() < 1e-12);
        assert!((num("sine(4, 1)[1]") - 1.0).abs() < 1e-12);
        assert!((num("sine(4, 1)[3]") + 1.0).abs() < 1e-12);
    }

    #[test]
    fn lazy_signal_is_a_generator() {
        // sine(freq) is a lazy signal (a generator) — type "signal", O(1) until materialized.
        assert_eq!(run("sine(3)").unwrap().type_name(), "signal");
        // scalar arithmetic and trig DEFER (stay a signal); only sampling materializes.
        assert_eq!(run("1 + 0.3 * sine(3)").unwrap().type_name(), "signal");
        assert_eq!(run("cos(2 * sine(1))").unwrap().type_name(), "signal");
        // sample(sig, n) materializes to an n-element array.
        assert_eq!(num("Len(sample(sine(3), 16))"), 16.0);
        // a lazy signal adopts the length of a sized array it meets.
        assert_eq!(num("Len(sine(2) + zeros(6))"), 6.0);
        // the eager 2-arg form equals sampling the lazy 1-arg form.
        assert_eq!(display_of("sample(sine(3), 8)"), display_of("sine(8, 3)"));
        // two bare signals can't combine without a length; an RV operand likewise.
        assert!(run("sine(1) + sine(2)").is_err());
        assert!(run("sample(5, 8)").is_err()); // sample needs a signal
    }

    #[test]
    fn lazy_noise_colors_materialize_to_correlated_normals() {
        // noise_*(…) are lazy, RANDOM generators — type "noise", no length yet.
        assert_eq!(run("noise_white(0.5)").unwrap().type_name(), "noise");
        assert_eq!(run("noise_brown(1)").unwrap().type_name(), "noise");
        assert_eq!(run("noise_ou(1, 8)").unwrap().type_name(), "noise");
        // sample(noise, n) draws an n-vector of zero-mean normals; white has variance sigma^2.
        assert_eq!(num("Len(sample(noise_pink(1), 16))"), 16.0);
        assert!(run_num("z = sample(noise_white(2), 6); E(mean(z), 40000)").abs() < 0.05);
        let v = run_num("z = sample(noise_white(2), 6); Var(z[0], 80000)");
        assert!((v - 4.0).abs() < 0.2, "white Var = {v} (want sigma^2 = 4)");
        // The spectral COLOR is the lag-1 autocorrelation E[x_k x_{k+1}] / E[x_k^2]:
        // white ~ 0, OU = exp(-1/tau), brown ~ 1. We assert the ordering and the OU closed form.
        let corr = "pair(x) = { a = 0; for i in 0..Len(x)-1 { a = a + x[i]*x[i+1] }; a }; \
                    sq(x)   = { a = 0; for i in 0..Len(x)-1 { a = a + x[i]*x[i]   }; a }; \
                    c(x)    = E(pair(x), 4000) / E(sq(x), 4000); ";
        let white = run_num(&format!("{corr} c(sample(noise_white(1), 200))"));
        let ou = run_num(&format!("{corr} c(sample(noise_ou(1, 8), 200))"));
        let brown = run_num(&format!("{corr} c(sample(noise_brown(1), 200))"));
        assert!(white.abs() < 0.05, "white neighbor corr = {white} (want ~0)");
        assert!((ou - (-1.0f64 / 8.0).exp()).abs() < 0.05, "OU corr = {ou} (want exp(-1/8))");
        assert!(brown > 0.9, "brown neighbor corr = {brown} (want ~1)");
        assert!(white < ou && ou < brown, "color should redden: {white} < {ou} < {brown}");
        // bare noise with no length, or bad params, are errors.
        assert!(run("noise_pink(0.5) * 2").is_err()); // needs a sized context
        assert!(run("noise_ou(1, 0)").is_err()); // tau must be > 0
        assert!(run("noise_brown(0 - 1)").is_err()); // sigma must be >= 0
    }

    #[test]
    fn nyquist_aliasing_via_sampling_rate() {
        // Nyquist-Shannon: a freq-7 wave needs > 2*7 = 14 samples. Undersampled at 10 it aliases to
        // the freq-3 wave (7 folds to 10-7 = 3) — identical samples, mse 0. Oversampled, they differ.
        let aliased = num("mse(sample(cosine(7), 10), sample(cosine(3), 10))");
        assert!(aliased < 1e-12, "undersampled should alias (mse 0), got {aliased}");
        let resolved = num("mse(sample(cosine(7), 64), sample(cosine(3), 64))");
        assert!(resolved > 0.5, "oversampled should distinguish them, got {resolved}");
    }

    #[test]
    fn trig_ufuncs_scalar_array_and_lifted() {
        // sin/cos/atan as ufuncs: scalar, elementwise over arrays, and lifted over RVs.
        assert_eq!(num("cos(0)"), 1.0);
        assert!((num("sin(pi / 2)") - 1.0).abs() < 1e-12);
        assert!((num("atan(1) * 4") - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(display_of("cos([0, 0])"), "[1, 1]"); // maps over an array
        // E[cos(X)] for X ~ N(0,1) is exp(-1/2) — proves cos lifts over a random variable.
        let m = run_num("X ~ normal(0, 1); E(cos(X))");
        assert!((m - (-0.5f64).exp()).abs() < 3e-3, "E[cos(X)] = {m} (want e^-0.5)");
        // `sign` is a ufunc too: scalar -1/0/+1, maps over arrays, and lifts over RVs.
        assert_eq!(num("sign(3.2)"), 1.0);
        assert_eq!(num("sign(0 - 7)"), -1.0);
        assert_eq!(num("sign(0)"), 0.0);
        assert_eq!(display_of("sign([2, 0 - 3, 0])"), "[1, -1, 0]");
        // E[sign(X)] for symmetric X ~ N(0,1) is 0 (it lifts over the RV per lane).
        assert!(run_num("X ~ normal(0, 1); E(sign(X))").abs() < 3e-3);
    }

    #[test]
    fn arithmetic_broadcasts_over_arrays() {
        assert_eq!(display_of("1 + [1, 2, 3]"), "[2, 3, 4]"); // scalar ⊕ array
        assert_eq!(display_of("[1, 2, 3] + [10, 20, 30]"), "[11, 22, 33]"); // array ⊕ array
        assert_eq!(display_of("[2, 4, 6] / 2"), "[1, 2, 3]");
        assert_eq!(display_of("[1, 2, 3] ** 2"), "[1, 4, 9]");
        // nested: an array of arrays broadcasts recursively ([I,Q] + [nI,nQ]).
        assert_eq!(display_of("[[1, 2], [3, 4]] + [[10, 20], [30, 40]]"), "[[11, 22], [33, 44]]");
        assert!(run("[1, 2] + [1, 2, 3]").is_err()); // length mismatch
    }

    #[test]
    fn am_vs_fm_modulate_demodulate_pipeline() {
        // Telecom, end to end: modulate a message, add the SAME static, demodulate, and compare
        // with mse. AM hides the message in amplitude (reads noise ~sigma^2); FM hides it in the
        // angle and divides the angle noise by the deviation, so FM recovers the message cleaner —
        // the advantage emerging from the model, not a hand-written formula.
        let lib = "am_mod(m) = [1 + m, 0 * m]; \
                   fm_mod(m, dev) = [cos(dev * m), sin(dev * m)]; \
                   am_demod(iq) = (iq[0] ** 2 + iq[1] ** 2) ** 0.5 - 1; \
                   fm_demod(iq, dev) = atan(iq[1] / iq[0]) / dev; \
                   N = 32; dev = 3; sigma = 0.3; \
                   msg = 0.3 * sin(iota(N) * (2 * pi * 2 / N)); \
                   static = [~[N] normal(0, sigma), ~[N] normal(0, sigma)]; ";
        let am = run_num(&format!(
            "{lib} E(mse(am_demod(am_mod(msg) + static), msg), 30000)"
        ));
        let fm = run_num(&format!(
            "{lib} E(mse(fm_demod(fm_mod(msg, dev) + static, dev), msg), 30000)"
        ));
        assert!((am - 0.09).abs() < 0.02, "AM error = {am} (want ~ sigma^2 = 0.09)");
        // FM should be several times cleaner for the same static (advantage grows with deviation).
        assert!(fm < am * 0.5, "FM error {fm} should be << AM error {am}");
    }

    #[test]
    fn rotation_is_orthonormal() {
        // `rotation(d)` is a random orthonormal matrix, so every sample preserves length exactly:
        // ||Pi x|| = ||x|| = 1 (the mean is exactly 1, hence a tight tolerance at tiny N).
        let nrm =
            run_num("d = 8; x = normalize(ones(d)); Pi ~ rotation(d); E(normsq(matvec(Pi, x)), 100)");
        assert!((nrm - 1.0).abs() < 1e-9, "||Pi x||^2 = {nrm}, want exactly 1");
        // And it round-trips: Pi^T Pi x = x (same Pi reused, so transpose is the inverse).
        let rt = run_num(
            "d = 8; Pi ~ rotation(d); x = normalize(iota(d)); \
             E(normsq(matvec(transpose(Pi), matvec(Pi, x)) - x), 100)",
        );
        assert!(rt < 1e-9, "||Pi^T Pi x - x||^2 = {rt}, want ~0");
    }

    #[test]
    fn rotation_is_a_recipe_drawn_with_tilde() {
        // `rotation(d)` is a recipe like any distribution: `~` draws it, `=` keeps it undrawn.
        // Using an undrawn rotation in arithmetic is the usual "draw it first" error.
        assert!(run("d = 4; Pi = rotation(d); x = ones(d); matvec(Pi, x)").is_err());
        // A shaped draw gives k *independent* rotations: stack two and they should differ, so the
        // squared distance between Pi0 x and Pi1 x is comfortably positive (identical draws → 0).
        let spread = run_num(
            "d = 6; x = normalize(ones(d)); Rs ~[2] rotation(d); \
             E(normsq(matvec(Rs[0], x) - matvec(Rs[1], x)), 2000)",
        );
        assert!(spread > 0.1, "two independent rotations should differ, got spread {spread}");
    }

    #[test]
    fn quantize_snaps_to_nearest_centroid() {
        // Each coordinate snaps to its nearest codebook entry; cell boundaries are the midpoints.
        assert_eq!(display_of("quantize([-2, -0.1, 0.1, 2], [-1, 1])"), "[-1, -1, 1, 1]");
        assert_eq!(display_of("quantize([0.4, 0.6], [0, 1])"), "[0, 1]"); // midpoint at 0.5
        assert_eq!(display_of("quantize([5, -5], [0])"), "[0, 0]"); // single-level codebook
        assert!(run("quantize([1], [])").is_err()); // empty codebook
    }

    #[test]
    fn turboquant_algorithm2_unbiased_and_lower_distortion() {
        // The faithful two-stage TurboQuant (arXiv:2504.19874), by Monte Carlo. Algorithm 1: a
        // random orthonormal rotation Pi, snap each rotated coordinate to its nearest Lloyd-Max
        // level, rotate back. At 1 bit this MSE quantizer reconstructs to ~0.36 distortion and is
        // BIASED by 2/pi for inner products. Algorithm 2: add a 1-bit QJL sketch of the residual
        // -> unbiased, and far lower inner-product error. (Small d/N so the test stays quick.)
        let common = "d = 16; x = normalize(ones(d)); y = normalize(iota(d)); t = dot(y, x); \
                      L1 = [-0.7979, 0.7979] * (1 / sqrt(d)); \
                      Pi ~ rotation(d); mse = matvec(transpose(Pi), quantize(matvec(Pi, x), L1)); \
                      S ~[d, d] normal(0, 1); r = x - mse; \
                      prod = dot(y, mse) \
                           + sqrt(pi / 2) / d * norm(r) \
                             * dot(y, matvec(transpose(S), sign(matvec(S, r)))); ";
        // Algorithm 1's distortion table, b=1 entry: D_mse ~ 0.36.
        let dmse = run_num(&format!("{common} E(normsq(x - mse), 12000)"));
        assert!((dmse - 0.36).abs() < 0.05, "D_mse(b=1) = {dmse}, want ~0.36");
        // The MSE quantizer is biased by 2/pi; the two-stage prod estimate is unbiased.
        let mse_ratio = run_num(&format!("{common} E(dot(y, mse) / t, 12000)"));
        let prod_ratio = run_num(&format!("{common} E(prod / t, 12000)"));
        assert!((mse_ratio - 2.0 / std::f64::consts::PI).abs() < 0.05, "MSE ratio = {mse_ratio}");
        assert!((prod_ratio - 1.0).abs() < 0.05, "prod ratio = {prod_ratio}");
        // The payoff: the unbiased two-stage estimate has far lower mean-squared inner-product error
        // than the biased MSE quantizer, whose error is dominated by its 2/pi bias floor.
        let mse_err = run_num(&format!("{common} E((dot(y, mse) - t) ** 2, 12000)"));
        let prod_err = run_num(&format!("{common} E((prod - t) ** 2, 12000)"));
        assert!(prod_err < mse_err * 0.6, "prod err {prod_err} should be << MSE err {mse_err}");
    }

    #[test]
    fn linear_algebra_length_mismatches_error() {
        assert!(run("dot([1, 2], [1, 2, 3])").is_err());
        assert!(run("vadd([1], [1, 2])").is_err());
        assert!(run("vsub([1, 2, 3], [1, 2])").is_err());
        assert!(run("dot(5, [1, 2])").is_err()); // non-array operand
    }

    #[test]
    fn has_duplicate_correctness() {
        assert!(!boolean("has_duplicate([1, 2, 3])"));
        assert!(boolean("has_duplicate([1, 2, 2, 3])"));
        assert!(!boolean("has_duplicate([])"));
        assert!(!boolean("has_duplicate([7])"));
    }

    #[test]
    fn shaped_draws_are_independent() {
        // Two iid draws are distinct nodes: P(days[0] == days[1]) = 1/6 for a d6.
        let p = run_num("days ~[2] unif_int(1, 6); P(days[0] == days[1])");
        assert!((p - 1.0 / 6.0).abs() < 5e-3, "P(days0==days1) = {p}");
        // Contrast: a single draw reused is perfectly correlated.
        assert_eq!(run_num("X ~ unif_int(1, 6); P(X == X)"), 1.0);
    }

    #[test]
    fn shaped_draw_builds_arrays_and_works_inline() {
        // `~[n]` draws an n-vector: E[sum of 4 uniforms] = 4 * 0.5 = 2.
        assert!((run_num("xs ~[4] unif(0, 1); E(sum(xs))") - 2.0).abs() < 0.05);
        // `~[n, m]` draws a matrix: E[sum over all 2x3 entries] = 6 * 0.5 = 3.
        assert!((run_num("M ~[2, 3] unif(0, 1); E(sum(sum(M)))") - 3.0).abs() < 0.05);
        // The prefix `~` works in expression position too — this is what retires `iid`.
        assert!((run_num("E(sum(~[3] unif(0, 1)))") - 1.5).abs() < 0.05);
        // A bare `~` is a scalar (rank 0); `~[1]` is a length-1 array (rank 1) — NOT equivalent.
        assert!((run_num("x ~ unif(0, 1); E(x)") - 0.5).abs() < 0.05);
        assert!((run_num("xs ~[1] unif(0, 1); E(xs[0])") - 0.5).abs() < 0.05); // indexable
        assert!(run("x ~ unif(0, 1); x[0]").is_err()); // a scalar can't be indexed
        // `~[]` (empty shape) is rejected at parse time.
        assert!(run("xs ~[] unif(0, 1)").is_err());
    }

    #[test]
    fn for_loop_accumulates_via_leaking_scope() {
        // The body's binding leaks into the current frame, so `acc` persists across iterations.
        assert_eq!(num("acc = 0; for x in [1, 2, 3, 4] { acc = acc + x }; acc"), 10.0);
        // nested loops
        assert_eq!(
            num("acc = 0; for i in 0..3 { for j in 0..3 { acc = acc + 1 } }; acc"),
            9.0
        );
        // a zero-iteration loop runs the body zero times and yields unit (graph untouched).
        assert_eq!(run("for i in 0..0 { x ~ unif(0, 1) }").unwrap(), Value::Unit);
        assert_eq!(graph_len("for i in 0..0 { x ~ unif(0, 1) }"), 0);
    }

    #[test]
    fn for_loop_unrolls_at_build_time() {
        // Each iteration draws a fresh node, so a loop of n builds exactly n× the body's nodes —
        // proving unroll (not runtime branching). One `unif` draw is one source node.
        let one = graph_len("x ~ unif(0, 1)");
        assert_eq!(graph_len("for i in 0..5 { x ~ unif(0, 1) }"), 5 * one);
        // `~` inside the loop gives independence: sum of n iid uniforms has variance n/12.
        let m = moments_of("acc = 0; for i in 0..12 { u ~ unif(0, 1); acc = acc + u }; acc", 1_000_000, 7);
        assert!((m.variance - 12.0 / 12.0).abs() < 0.02, "var = {}", m.variance);
    }

    #[test]
    fn for_loop_errors_on_a_non_array() {
        assert!(run("for x in 5 { x }").is_err());
        assert!(run("Len(5)").is_err());
    }

    #[test]
    fn library_matches_a_noise_written_version() {
        // The §0 property: each library function IS expressible in Noise. A Noise-written `sum`,
        // `dot`, and `has_duplicate` must match the builtin on the same inputs.
        let noise_sum = "mysum(xs) = { acc = 0; for x in xs { acc = acc + x }; acc };";
        assert_eq!(num(&format!("{noise_sum} mysum([1, 2, 3, 4])")), num("sum([1, 2, 3, 4])"));

        let noise_dot =
            "mydot(a, b) = { acc = 0; for i in 0..Len(a) { acc = acc + a[i] * b[i] }; acc };";
        assert_eq!(
            num(&format!("{noise_dot} mydot([1, 2, 3], [4, 5, 6])")),
            num("dot([1, 2, 3], [4, 5, 6])")
        );

        let noise_dup = "mydup(xs) = { d = false; for i in 0..Len(xs) { for j in i + 1 .. Len(xs) { d = d || (xs[i] == xs[j]) } }; d };";
        assert!(boolean(&format!("{noise_dup} mydup([1, 2, 2])")));
        assert!(!boolean(&format!("{noise_dup} mydup([1, 2, 3])")));
    }

    #[test]
    fn library_lifts_over_random_variables() {
        // A Noise-written reducer over RVs matches the builtin: P(sum of 2d6 == 7) = 1/6.
        let p = run_num(
            "mysum(xs) = { acc = 0; for x in xs { acc = acc + x }; acc }; \
             A ~ unif_int(1, 6); B ~ unif_int(1, 6); P(mysum([A, B]) == 7)",
        );
        assert!((p - 1.0 / 6.0).abs() < 5e-3, "P(sum==7) = {p}");
    }

    // --- Step 4 golden migrations (the §5 examples, collapsed to one-liners) ---

    #[test]
    fn birthday_problem_for_23() {
        // The headline win: 23 people, one line. Analytic ≈ 0.5073.
        let p = run_num("days ~[23] unif_int(1, 365); P(has_duplicate(days))");
        assert!((p - 0.5073).abs() < 5e-3, "P(shared birthday among 23) = {p}");
    }

    #[test]
    fn generalized_clt_agrees_with_native_normal() {
        // sum of n iid uniforms minus n/2 → ~N(0,1) at n=12; its P(Z>1) matches native normal.
        let clt = run_num("n = 12; clt = sum(~[n] unif(0, 1)) - n / 2; P(clt > 1)");
        let native = run_num("Z ~ normal(0, 1); P(Z > 1)");
        assert!((clt - native).abs() < 5e-3, "CLT {clt} vs native {native}");
        assert!((native - 0.1587).abs() < 3e-3, "native P(Z>1) = {native}");
    }

    #[test]
    fn migrated_one_liners_hit_their_analytic_values() {
        // The §5c table — each was a hand-unrolled example, now one line.
        let cases = [
            ("P(sum(~[2] unif_int(1, 6)) == 7)", 1.0 / 6.0),       // dice_sum
            ("P(all(~[3] bernoulli(0.5)))", 0.125),                // coin_streak
            ("P(count(~[3] bernoulli(0.5)) == 2)", 0.375),         // exactly_two_heads
            ("P(sum(~[3] unif(0, 1)) > 2)", 1.0 / 6.0),            // irwin_hall
            ("P(any(~[3] bernoulli(0.9)))", 0.999),                // reliability
        ];
        for (src, expected) in cases {
            let p = run_num(src);
            assert!((p - expected).abs() < 5e-3, "{src} = {p}, expected {expected}");
        }
    }

    // --- module system: `mod::name` paths, `use`, strict scoping (these bypass the prelude) ---

    #[test]
    fn qualified_paths_resolve_without_use() {
        // A `mod::name` path reaches a module's items even with nothing `use`d.
        assert_eq!(run_raw("math::sqrt(16)").unwrap(), Value::Num(4.0));
        assert!((as_num(run_raw("math::pi").unwrap()) - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(run_raw("vec::sum([1, 2, 3])").unwrap(), Value::Num(6.0));
        assert_eq!(run_raw("vec::dot([1, 2], [3, 4])").unwrap(), Value::Num(11.0));
        // a random constructor path draws fine (the qualified `rand::unif` needs no `use`)
        assert!(run_raw("X ~ rand::unif(0, 1); P(X < 0.5, 1000)").is_ok());
    }

    #[test]
    fn use_brings_a_module_into_unqualified_scope() {
        // After `use`, bare names resolve — like Rust.
        assert_eq!(run_raw("use math; sqrt(25)").unwrap(), Value::Num(5.0));
        assert_eq!(run_raw("use vec; sum([10, 20])").unwrap(), Value::Num(30.0));
        let p = run_raw("use rand; use vec; P(has_duplicate(~[2] unif_int(1, 6)))").unwrap();
        assert!((as_num(p) - 1.0 / 6.0).abs() < 5e-3);
        // `use builtin;` is a harmless no-op (always active anyway).
        assert_eq!(run_raw("use builtin; Len([1, 2])").unwrap().to_string(), "2");
    }

    #[test]
    fn builtin_module_is_active_by_default() {
        // P, E, Var, Print, Len need no `use` (and are capital-only).
        assert_eq!(run_raw("Len([1, 2, 3])").unwrap(), Value::Num(3.0));
        assert_eq!(as_num(run_raw("P(1 == 1)").unwrap()), 1.0);
        assert_eq!(run_raw("Print(\"hi\")").unwrap(), Value::Unit);
        // the old lowercase spellings are gone (no back-compat alias for the core)
        assert!(run_raw("print(\"hi\")").is_err());
        assert!(run_raw("len([1])").is_err());
    }

    #[test]
    fn strict_scoping_rejects_bare_names_until_used() {
        // Without a `use` (or a path), a rand/math/vec name is out of scope — with a message that
        // tells you exactly how to fix it.
        for (src, module) in [
            ("unif(0, 1)", "rand"),
            ("pi", "math"),
            ("sqrt(4)", "math"),
            ("sum([1, 2])", "vec"),
            ("dot([1], [1])", "vec"),
        ] {
            let err = run_raw(src).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains(module) && msg.contains("use"), "{src} -> {msg}");
        }
    }

    #[test]
    fn module_path_errors_are_specific() {
        // wrong module for a real function
        assert!(run_raw("math::unif(0, 1)").unwrap_err().to_string().contains("rand"));
        // unknown module, in a path and in a `use`
        assert!(run_raw("foo::bar()").is_err());
        assert!(run_raw("use foo; 1").is_err());
        // a function name used as a value (not called)
        assert!(run_raw("math::sqrt").is_err());
        // module has no such item
        assert!(run_raw("math::nope()").is_err());
    }

    /// Throughput of the TurboQuant workload (examples/turboquant.noise) — by far the heaviest
    /// example in the corpus, and the most interesting one to benchmark. Every Monte Carlo sample
    /// builds a *fresh* random orthonormal rotation of a d=20 vector (modified Gram–Schmidt over
    /// d² = 400 Gaussian draws), then quantizes the rotated coordinates and rotates back.
    ///
    /// The point this measures: `@`/`rotation`/`matvec` carry no GEMM loop. They expand
    /// element-by-element into the *same* scalar `RvGraph`, so the whole d×d linear algebra of a
    /// sample collapses into one straight-line fused kernel (CSE'd, then drawn + reduced across all
    /// cores like any other query). That is ideal at small d — everything lives in registers, no
    /// cache traffic, no loop overhead — and is the opposite end of the size spectrum from `pi`'s
    /// tiny kernel. We print the fused kernel size (`cone_size`) so the expansion is visible.
    ///
    /// Ignored; run with:
    /// `cargo test -p noise-core [--features jit] --release -- --ignored --nocapture bench_turboquant`
    #[test]
    #[ignore]
    fn bench_turboquant() {
        use std::time::Instant;

        let setup = "use rand; use math; use vec;
            d = 20;
            x = vec::normalize(vec::ones(d));
            y = vec::normalize(vec::iota(d));
            s = 1 / sqrt(d);
            L3 = [-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520] * s;
            L4 = [-2.7326, -2.0690, -1.6181, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                   0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6181,  2.0690,  2.7326] * s;";

        // D_mse b=4: rotate, snap to the 4-bit codebook, rotate back; distortion ||x - xhat||².
        let d_mse = format!(
            "{setup}
             Pi ~ rotation(d); rot = Pi @ x; PiT = vec::transpose(Pi);
             mse4 = PiT @ vec::quantize(rot, L4);
             vec::normsq(x - mse4)"
        );
        // D_prod b=4: the headline estimator — 3-bit MSE stage + a 1-bit QJL residual sketch,
        // two independent d×d matrices per sample (a rotation R and a Gaussian projection S).
        let d_prod = format!(
            "{setup}
             true_ip = y @ x;
             S ~[d, d] normal(0, 1);  R ~ rotation(d);
             m = vec::transpose(R) @ vec::quantize(R @ x, L3);
             est = y @ m
                 + sqrt(pi / 2) / d * vec::norm(x - m)
                   * (y @ (vec::transpose(S) @ sign(S @ (x - m))));
             (est - true_ip) ** 2"
        );

        let n = 1_000_000usize;
        for (label, src, target) in
            [("D_mse  b=4", d_mse.as_str(), 0.009), ("D_prod b=4", d_prod.as_str(), 0.047 / 20.0)]
        {
            let mut eng = Engine::new();
            let id = match eng.run_rv(src).unwrap() {
                Value::Dist(id) => id,
                other => panic!("expected an RV, got {other:?}"),
            };
            let g = eng.graph();
            let cone = crate::kernel::cone_size(g, id);

            // Warm up (this compiles + JITs the kernel), then time sampling only.
            let _ = crate::sampler::moments(g, id, 4096, 1);
            let t = Instant::now();
            let m = crate::sampler::moments(g, id, n, 0xC0FFEE);
            let secs = t.elapsed().as_secs_f64();
            let mps = n as f64 / secs / 1e6;
            println!(
                "  {label}: fused kernel {cone:4} nodes ({} total)   {mps:6.2} M samples/s   \
                 est {:.4} (paper {target:.4})",
                g.len(),
                m.mean
            );
        }
    }

    /// Pull the numeric value out of a `Num`/`Est` (test helper for module assertions).
    fn as_num(v: Value) -> f64 {
        match v {
            Value::Num(n) => n,
            Value::Est { val, .. } => val,
            other => panic!("expected number, got {other:?}"),
        }
    }
}
