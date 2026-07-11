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
pub mod doc;
pub mod error;
pub mod eval;
pub mod flint;
pub mod frontmatter;
pub mod input;
pub mod introspect;
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
pub mod stats;
pub mod value;
pub mod wasm_emit;
#[cfg(target_arch = "wasm32")]
pub mod wasm_host;

pub use dist::RvId;
pub use doc::Document;
pub use error::{NoiseError, Result};
pub use eval::{Emission, Engine, Output};
pub use frontmatter::Frontmatter;
pub use input::{InputKind, InputSpec, InputValue, ResolvedInput};
pub use sampler::Moments;
pub use stats::RunStats;
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
        assert_eq!(num("-2 ^ 2"), -4.0); // ^ binds tighter than unary minus
        assert_eq!(num("2 ^ 3 ^ 2"), 512.0); // right-assoc
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
        assert!(
            eng.graph().is_empty(),
            "a recipe must not touch the sample-DAG"
        );
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
        assert!(
            err.to_string().contains('~'),
            "message should mention `~`: {err}"
        );
    }

    #[test]
    fn assign_keeps_a_recipe_then_tilde_draws_independent_copies() {
        // `Die = unif_int(1,6)` binds the recipe; each `~ Die` draws an INDEPENDENT die, so
        // P(a == b) = 1/6 (not 1 — they are not the same node).
        let p = run_num("Die = unif_int(1,6); a ~ Die; b ~ Die; P(a == b)");
        assert!(
            (p - 1.0 / 6.0).abs() < 5e-3,
            "P(a==b) over two draws of one recipe = {p}"
        );
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
    fn unif_int_with_inverted_dynamic_bounds_stays_in_range() {
        // Finding B4: data-dependent bounds where `hi < lo` on every lane. The lowered draw clamps
        // the width to >= 1, so an inverted lane degenerates to a point mass at `lo` — never a value
        // *below* `lo` (the old `lo + floor((hi-lo+1)·U)` produced out-of-range values as low as 1).
        let mut eng = Engine::new();
        let rv = eng
            .run_rv("use rand; lo ~ unif_int(3, 5); hi ~ unif_int(0, 2); x ~ unif_int(lo, hi); x")
            .unwrap();
        let draws = eng.sample(&rv, 8192, 123).unwrap();
        for &x in &draws {
            assert!(
                (3.0..=5.0).contains(&x),
                "inverted-bounds draw out of range: {x} (should degenerate to lo ∈ 3..=5)"
            );
            assert_eq!(x, x.round(), "draw must be an integer: {x}");
        }
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
        assert!(
            (m.variance - 1.0 / 3.0).abs() < 3e-3,
            "var = {}",
            m.variance
        );
    }

    #[test]
    fn shifted_scaled_uniform() {
        // D ~ unif(1,6): mean 3.5, variance 25/12 ≈ 2.08333.
        let m = moments_of("D ~ unif(1,6); D", 1_000_000, 7);
        assert!((m.mean - 3.5).abs() < 3e-3, "mean = {}", m.mean);
        assert!(
            (m.variance - 25.0 / 12.0).abs() < 5e-3,
            "var = {}",
            m.variance
        );
    }

    #[test]
    fn derived_rv_square() {
        // X ~ unif(-1,1); X^2: E=1/3, Var = 1/5 - 1/9 = 4/45 ≈ 0.08889.
        let m = moments_of("X ~ unif(-1,1); X ^ 2", 1_000_000, 42);
        assert!((m.mean - 1.0 / 3.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!(
            (m.variance - 4.0 / 45.0).abs() < 3e-3,
            "var = {}",
            m.variance
        );
    }

    #[test]
    fn affine_lift() {
        // Y = 2*X + 3 for X~unif(0,1): mean 4, variance (2^2)/12 = 1/3.
        let m = moments_of("X ~ unif(0,1); 2*X + 3", 1_000_000, 1);
        assert!((m.mean - 4.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!(
            (m.variance - 1.0 / 3.0).abs() < 3e-3,
            "var = {}",
            m.variance
        );
    }

    #[test]
    fn shared_draw_correctness() {
        // X + X uses ONE draw of X per lane (structural sharing): mean 1, variance 4*Var(X)=1/3.
        // (Independent draws would give variance 1/6 — this proves CSE.)
        let m = moments_of("X ~ unif(0,1); X + X", 1_000_000, 99);
        assert!((m.mean - 1.0).abs() < 3e-3, "mean = {}", m.mean);
        assert!(
            (m.variance - 1.0 / 3.0).abs() < 3e-3,
            "var = {}",
            m.variance
        );

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

    // --- conditioning: `event | given` (Bayes' rule, scoped to one query) ---

    #[test]
    fn conditional_probability_matches_bayes() {
        // P(D==6 | D>3) = P(D==6 && D>3) / P(D>3) = (1/6)/(1/2) = 1/3.
        let p = run_num("D ~ unif_int(1,6); P(D == 6 | D > 3)");
        assert!((p - 1.0 / 3.0).abs() < 5e-3, "P(D==6 | D>3) = {p}");
    }

    #[test]
    fn conditioning_agrees_with_the_ratio_form() {
        // The new `P(A | C)` matches the hand-written `P(A && C) / P(C)` — same Bayes, less ceremony.
        let bar = run_num("D ~ unif_int(1,6); P(D == 6 | D > 3)");
        let ratio = run_num("D ~ unif_int(1,6); P(D == 6 && D > 3) / P(D > 3)");
        assert!((bar - ratio).abs() < 5e-3, "bar {bar} vs ratio {ratio}");
    }

    #[test]
    fn conditional_expectation_variance_quantile() {
        // D | D>3 is uniform on {4,5,6}: mean 5, variance 2/3, median 5.
        assert!((run_num("D ~ unif_int(1,6); E(D | D > 3)") - 5.0).abs() < 5e-3);
        assert!((run_num("D ~ unif_int(1,6); Var(D | D > 3)") - 2.0 / 3.0).abs() < 5e-3);
        assert!((run_num("D ~ unif_int(1,6); Q(D | D > 3, 0.5)") - 5.0).abs() < 1e-9);
    }

    #[test]
    fn conditioned_value_is_bindable_then_queryable() {
        // `a = X | C` is a first-class value: bind it, query it later (P/E/Var/Q).
        let p = run_num("D ~ unif_int(1,6); a = D == 6 | D > 3; P(a)");
        assert!((p - 1.0 / 3.0).abs() < 5e-3, "P(a) = {p}");
    }

    #[test]
    fn operations_push_into_the_conditioned_quantity() {
        // 2*(D|D>3)+1 is (2D+1) | D>3, mean 2*5+1 = 11.
        assert!((run_num("D ~ unif_int(1,6); b = D | D > 3; E(2*b + 1)") - 11.0).abs() < 5e-3);
        // A comparison on a conditioned number is a conditioned event: P((D|D>3) < 5) = P(D==4 | D>3) = 1/3.
        let p = run_num("D ~ unif_int(1,6); b = D | D > 3; P(b < 5)");
        assert!((p - 1.0 / 3.0).abs() < 5e-3, "P(b<5) = {p}");
        // Unary minus pushes in too: E(-(D|D>3)) = -5.
        assert!((run_num("D ~ unif_int(1,6); E(-(D | D > 3))") + 5.0).abs() < 5e-3);
    }

    #[test]
    fn conditioning_misuse_is_a_spanned_error() {
        for src in [
            // Combining two values conditioned on DIFFERENT events is ill-defined.
            "D ~ unif_int(1,6); b = D | D > 3; c = D | D > 4; E(b + c)",
            // P needs an event; a conditioned NUMBER is for E/Var/Q, not P.
            "D ~ unif_int(1,6); P(D | D > 3)",
            // The condition after `|` must be an event (bool), not a number.
            "D ~ unif_int(1,6); P(D == 6 | D)",
            // A condition that never occurs makes the conditional undefined (0/0).
            "D ~ unif_int(1,6); P(D == 6 | D > 100)",
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
                "{src:?} needs a real span"
            );
        }
    }

    // --- hierarchical models: a random parameter feeding another distribution ---

    #[test]
    fn random_parameter_bernoulli_marginalizes() {
        // p ~ unif(0,1); k ~ bernoulli(p): P(k) = E[p] = 0.5 (the parameter is integrated out).
        let p = run_num("p ~ unif(0,1); k ~ bernoulli(p); P(k)");
        assert!((p - 0.5).abs() < 5e-3, "P(k) = {p}");
    }

    #[test]
    fn random_mean_normal_adds_variance() {
        // mu ~ N(0,1); X ~ N(mu,1): E[X] = 0, Var[X] = Var[mu] + E[sigma^2] = 1 + 1 = 2.
        let m = moments_of("mu ~ normal(0,1); X ~ normal(mu, 1); X", 1_000_000, 7);
        assert!(m.mean.abs() < 5e-3, "mean = {}", m.mean);
        assert!((m.variance - 2.0).abs() < 1e-2, "var = {}", m.variance);
    }

    #[test]
    fn random_scale_uniform_is_drawn_per_lane() {
        // w ~ unif(1,3); X ~ unif(0, w): E[X] = E[w]/2 = 2/2 = 1.
        assert!((run_num("w ~ unif(1,3); X ~ unif(0, w); E(X)") - 1.0).abs() < 5e-3);
        // rate ~ unif(1,3); T ~ exponential(rate): E[T] = E[1/rate] = (ln3 - ln1)/2 ≈ 0.5493.
        let e = run_num("rate ~ unif(1,3); T ~ exponential(rate); E(T)");
        assert!((e - (3.0_f64.ln() / 2.0)).abs() < 5e-3, "E(T) = {e}");
    }

    #[test]
    fn two_draws_of_a_random_param_recipe_are_independent_given_the_param() {
        // p ~ unif(0,1); a,b ~ bernoulli(p): a and b share p but use fresh draws, so they are
        // correlated (independent only GIVEN p). For 0/1 indicators, E[a·b] = P(a && b) = E[p^2] =
        // 1/3, while P(a)·P(b) = 1/4, so the covariance is 1/3 − 1/4 = 1/12 > 0.
        let cov =
            run_num("p ~ unif(0,1); a ~ bernoulli(p); b ~ bernoulli(p); P(a && b) - P(a)*P(b)");
        assert!(
            (cov - 1.0 / 12.0).abs() < 5e-3,
            "cov = {cov} (expected 1/12)"
        );
    }

    #[test]
    fn hierarchical_plus_conditioning_is_bayesian_posterior() {
        // Beta-Bernoulli: prior p~U(0,1), observe one head -> posterior mean E[p|k] = 2/3.
        let post = run_num("p ~ unif(0,1); k ~ bernoulli(p); E(p | k)");
        assert!((post - 2.0 / 3.0).abs() < 5e-3, "E(p | k) = {post}");
        // Observe 7 heads in 10 flips -> posterior mean of the bias = (7+1)/(10+2) = 8/12.
        let post10 = run_num("q ~ unif(0,1); flips ~[10] bernoulli(q); E(q | count(flips) == 7)");
        assert!((post10 - 8.0 / 12.0).abs() < 1e-2, "E(q | 7/10) = {post10}");
    }

    #[test]
    fn random_distribution_parameter_type_errors_stay_spanned() {
        for src in [
            // a bool RV is not a valid numeric parameter
            "b ~ bernoulli(0.5); X ~ normal(b, 1); E(X)",
            // an undrawn recipe as a parameter — must be drawn first
            "k ~ bernoulli(unif(0,1)); P(k)",
            // poisson with a random parameter is not supported yet (only unif/unif_int/normal/exp/bernoulli)
            "L ~ unif(1,3); N ~ poisson(L); E(N)",
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
                "{src:?} needs a real span"
            );
        }
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
        assert!(
            eng.graph().is_empty(),
            "graph must stay empty for deterministic programs"
        );
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
        let pi = run_num("X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X^2 + Y^2 < 1)");
        assert!(
            (pi - std::f64::consts::PI).abs() < 0.02,
            "pi estimate = {pi}"
        );
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
            draws
                .iter()
                .all(|&x| (1.0..=6.0).contains(&x) && x.fract() == 0.0),
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
        let pi = display_of("X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X^2 + Y^2 < 1, 100000000)");
        assert_eq!(decimals(&pi), 3, "should propagate to 3 decimals, got {pi}");
        assert!(
            (pi.parse::<f64>().unwrap() - std::f64::consts::PI).abs() < 0.01,
            "pi = {pi}"
        );

        // Default budget → larger error → coarser: 2 decimals ("3.14").
        let pi_coarse = display_of("X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X^2 + Y^2 < 1)");
        assert_eq!(
            decimals(&pi_coarse),
            2,
            "default budget should show 2 decimals, got {pi_coarse}"
        );
        assert!(
            (pi_coarse.parse::<f64>().unwrap() - std::f64::consts::PI).abs() < 0.02,
            "pi_coarse = {pi_coarse}"
        );
    }

    #[test]
    fn engine_set_max_samples_sets_the_default_budget() {
        // `engine::set_max_samples(N)` is the global default for P/E/Var/Q — equivalent to passing N
        // as the explicit count, but once, up front. At a small budget the standard error is wide,
        // so the estimate displays to few digits; bumping the budget reveals more.
        let coarse = display_of("engine::set_max_samples(1000); D ~ unif_int(1,6); P(D == 4)");
        assert_eq!(
            coarse, "0.2",
            "1000 draws justifies one decimal, got {coarse}"
        );
        let fine = display_of("engine::set_max_samples(100000000); D ~ unif_int(1,6); P(D == 4)");
        assert!(
            fine.len() > 5,
            "1e8 draws should reveal ~4 digits, got {fine}"
        );

        // An explicit per-call count still overrides the engine default (here: tighten back up
        // despite the coarse global setting).
        let overridden =
            display_of("engine::set_max_samples(1000); D ~ unif_int(1,6); P(D == 4, 100000000)");
        assert!(
            overridden.len() > 5,
            "explicit count should override, got {overridden}"
        );
    }

    #[test]
    fn engine_set_max_opts_clamps_query_cost_without_erroring() {
        // `engine::set_max_opts(N)` caps `draws × cone-ops` per query: a tight budget forces few
        // draws, so the estimate displays to fewer digits than the unclamped default. It degrades
        // accuracy gracefully — it never errors, not even when a single draw already exceeds the
        // budget (it floors at one draw).
        let baseline = display_of("D ~ unif_int(1,6); P(D == 4)");
        let clamped = display_of("engine::set_max_opts(10); D ~ unif_int(1,6); P(D == 4)");
        assert!(
            clamped.len() < baseline.len(),
            "a tight op budget should give a coarser estimate: clamped={clamped} baseline={baseline}"
        );
        // Never errors, even at a budget below a single draw's cost (clamps down to one draw).
        assert!(run_raw("use rand; engine::set_max_opts(1); X ~ unif(0,1); P(X < 0.5)").is_ok());
        // Composes with set_max_samples — the query draws the smaller of the two budgets.
        assert!(run_raw(
            "use rand; engine::set_max_samples(100000); engine::set_max_opts(5); X ~ unif(0,1); E(X)"
        )
        .is_ok());
    }

    #[test]
    fn engine_module_scoping_and_validation() {
        // Reachable both as a `mod::name` path and (with `use`) unqualified, like any module.
        assert!(
            run_raw("engine::set_max_samples(50); use rand; X ~ unif(0,1); P(X < 0.5)").is_ok()
        );
        assert!(run_raw("use engine; set_max_samples(50); 1").is_ok());
        assert!(run_raw("use engine; set_max_opts(50); 1").is_ok());
        // Out of scope without a `use`/path, with a fix-it message naming the module.
        let err = run_raw("set_max_samples(50)").unwrap_err().to_string();
        assert!(err.contains("engine") && err.contains("use"), "{err}");
        let err_opts = run_raw("set_max_opts(50)").unwrap_err().to_string();
        assert!(
            err_opts.contains("engine") && err_opts.contains("use"),
            "{err_opts}"
        );
        // Both budgets must be at least 1.
        assert!(run_raw("engine::set_max_samples(0)")
            .unwrap_err()
            .to_string()
            .contains(">= 1"));
        assert!(run_raw("engine::set_max_opts(0)")
            .unwrap_err()
            .to_string()
            .contains(">= 1"));
    }

    #[test]
    fn check_builds_graph_but_skips_monte_carlo() {
        // `check` validates a program — parse + evaluate + build the graph — without sampling.
        // With a 100-million-sample budget a real run would be glacial; `check` returns instantly
        // because P/E/Var/Q hand back placeholders instead of forcing the cone.
        let prelude = "use rand; use math; use vec; use signal;\n";
        let heavy = "engine::set_max_samples(100000000); X ~ normal(0,1); P(X < 0) + E(X) + Var(X) + Q(X, 0.5)";
        assert!(Engine::new().check(&format!("{prelude}{heavy}")).is_ok());

        // The placeholder probability stays in range, so it's safe flowing into a range-checked
        // constructor (this would error if `P` returned NaN or an out-of-[0,1] value).
        assert!(Engine::new()
            .check(&format!(
                "{prelude}X ~ normal(0,1); B ~ bernoulli(P(X < 0))"
            ))
            .is_ok());

        // `check` still surfaces parse, scope, and type errors the way `run` does.
        assert!(Engine::new()
            .check(&format!("{prelude}Y ~ normal(mu, 1)"))
            .is_err());
        assert!(Engine::new().check("X ~ unif(0, 1").is_err());
        assert!(Engine::new()
            .check(&format!("{prelude}X ~ normal(0,1); P(X)"))
            .is_err());
    }

    // --- lifted `if` over a random variable (per-lane select) ---

    #[test]
    fn if_over_rv_selects_per_lane() {
        // payoff = +10 with prob 1/6, -2 with prob 5/6 -> mean 0 exactly, variance 20.
        let m = moments_of(
            "D ~ unif_int(1,6); if D == 6 { 10 } else { 0 - 2 }",
            1_000_000,
            3,
        );
        assert!(m.mean.abs() < 0.02, "mean = {}", m.mean);
        assert!((m.variance - 20.0).abs() < 0.2, "var = {}", m.variance);
    }

    #[test]
    fn if_branches_read_consistent_draws() {
        // if D == 6 { D } else { 0 } : the branch reuses the SAME per-lane draw of D,
        // so the result is 6 w.p. 1/6 and 0 otherwise -> mean 1.
        let m = moments_of(
            "D ~ unif_int(1,6); if D == 6 { D } else { 0 }",
            1_000_000,
            4,
        );
        assert!((m.mean - 1.0).abs() < 0.02, "mean = {}", m.mean);
    }

    #[test]
    fn max_via_if() {
        // M = max(A, B) for two d6; P(M == 6) = 1 - (5/6)^2 = 11/36.
        let p =
            run_num("A ~ unif_int(1,6); B ~ unif_int(1,6); P((if A > B { A } else { B }) == 6)");
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
        assert_eq!(
            num("D ~ unif_int(1,6); if P(D < 4) > 0.4 { 1 } else { 0 }"),
            1.0
        );
        assert_eq!(
            num("D ~ unif_int(1,6); if P(D == 6) > 0.5 { 1 } else { 0 }"),
            0.0
        );
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
        assert_eq!(
            num("inc(x) = x + 1; twice(x) = inc(inc(x)); twice(10)"),
            12.0
        );
    }

    #[test]
    fn functions_are_pure_in_their_params_no_outer_capture() {
        // A function body sees only its parameters, not outer variables (no closures yet).
        assert!(run("n = 10; f(x) = x + n; f(1)").is_err());
        // params shadow nothing leaks back out: the call's frame is discarded.
        assert_eq!(num("x = 100; id(x) = x; id(5); x"), 100.0);
    }

    #[test]
    fn named_arguments_bind_to_user_function_parameters() {
        // named args bind by name, in any order (PLAN-INPUTS §2).
        assert_eq!(num("sub(a, b) = a - b; sub(b: 2, a: 10)"), 8.0);
        // same as the positional call
        assert_eq!(num("sub(a, b) = a - b; sub(10, 2)"), 8.0);
        // a single named arg
        assert_eq!(num("dbl(x) = x + x; dbl(x: 21)"), 42.0);
    }

    #[test]
    fn named_argument_errors() {
        // unknown parameter name
        let e = run("f(a, b) = a + b; f(a: 1, z: 2)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("no parameter named `z`"), "{e}");
        // a parameter left unbound
        let e = run("f(a, b) = a + b; f(a: 1)").unwrap_err().to_string();
        assert!(e.contains("missing argument `b`"), "{e}");
        // a builtin does not accept named arguments
        let e = run("unif(lo: 0, hi: 1)").unwrap_err().to_string();
        assert!(e.contains("does not accept named arguments"), "{e}");
    }

    #[test]
    fn input_resolves_to_its_default_value() {
        // With no host override, an input evaluates to its (clamped/snapped) default — the program
        // runs deterministically, no UI needed (PLAN-INPUTS §1).
        assert_eq!(num("n = input::real(min: 1, max: 100, default: 6); n"), 6.0);
        assert_eq!(
            num("k = input::int(min: 0, max: 10, step: 2, default: 5); k"),
            6.0
        ); // snap 5→6
        assert!(boolean("b = input::bool(default: true); b"));
        // name inference: the LHS names the input, no explicit `name:` needed.
        assert_eq!(
            num("sides = input::real(min: 2, max: 20, default: 6); sides * 2"),
            12.0
        );
    }

    #[test]
    fn input_override_is_clamped_and_snapped() {
        let mut e = Engine::new();
        e.set_input_overrides(vec![("n".into(), crate::input::InputValue::Num(250.0))]);
        // 250 clamps to max 100.
        let v = e
            .run(&with_prelude(
                "n = input::real(min: 1, max: 100, default: 6); n",
            ))
            .unwrap();
        assert!(matches!(v, Value::Num(x) if x == 100.0), "got {v:?}");
    }

    #[test]
    fn input_errors() {
        // a standalone input with no name and no binding LHS
        let e = run("input::real(min: 1, max: 10, default: 5)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("needs a name"), "{e}");
        // missing default
        let e = run("n = input::real(min: 1, max: 10)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("needs a `default`"), "{e}");
        // unknown field
        let e = run("n = input::real(minn: 1, default: 5)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("no field `minn`"), "{e}");
        // unknown input type
        let e = run("n = input::color(default: 5)").unwrap_err().to_string();
        assert!(e.contains("unknown input type"), "{e}");
        // override for an input the program never declares
        let mut eng = Engine::new();
        eng.set_input_overrides(vec![("ghost".into(), crate::input::InputValue::Num(1.0))]);
        let err = eng.run(&with_prelude("x = 1")).unwrap_err().to_string();
        assert!(err.contains("unknown input `ghost`"), "{err}");
    }

    #[test]
    fn input_redeclared_with_different_spec_errors() {
        // Same name, same spec: fine (dedup). Different spec: conflict.
        assert_eq!(
            num("a = input::real(min: 0, max: 9, default: 3); b = input::real(min: 0, max: 9, default: 3); a + b"),
            6.0
        );
        let e = run("a = input::real(min: 0, max: 9, default: 3); c = input::real(name: \"a\", min: 0, max: 9, default: 5); c")
            .unwrap_err()
            .to_string();
        assert!(e.contains("redeclared with a different spec"), "{e}");
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
        assert!(
            (m.variance - 2.0 * 35.0 / 12.0).abs() < 0.1,
            "var = {}",
            m.variance
        );
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
        assert!(
            eng.graph().is_empty(),
            "defining a function must not sample"
        );
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
        assert!(draws_of("D ~ unif_int(5,5); D", 5000, 7)
            .iter()
            .all(|&x| x == 5.0));
    }

    #[test]
    fn unif_degenerate_range_is_a_constant() {
        // unif(2,2) has zero span, so every draw is exactly 2.
        assert!(draws_of("X ~ unif(2,2); X", 4096, 9)
            .iter()
            .all(|&x| x == 2.0));
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
        // Real `sqrt` is IEEE: a negative argument is NaN, not an error (complex sqrt is opt-in via
        // `math::sqrt(-1 + 0*math::i)`).
        assert!(run_num("sqrt(-1)").is_nan());
    }

    #[test]
    fn normal_parameter_domain_and_recipe_display() {
        assert!(run("normal(0, -1)").is_err()); // sigma must be >= 0
        assert_eq!(display_of("normal(0, 1)"), "normal(0, 1)"); // undrawn recipe prints itself
                                                                // sigma = 0 is a degenerate point mass at mu.
        assert!(draws_of("Z ~ normal(5, 0); Z", 2000, 4)
            .iter()
            .all(|&x| x == 5.0));
    }

    // --- more `rand` distributions: exp, poisson, geometric, and the `_int` family ---

    #[test]
    fn exponential_has_the_right_moments() {
        // Exp(rate): mean = 1/rate, variance = 1/rate^2. For rate = 2: mean 0.5, var 0.25.
        let m = moments_of("X ~ exponential(2); X", 1_000_000, 1);
        assert!((m.mean - 0.5).abs() < 0.01, "mean = {}", m.mean);
        assert!((m.variance - 0.25).abs() < 0.01, "var = {}", m.variance);
        // memoryless tail: P(X > 1) = e^-(rate*1) = e^-2 ≈ 0.1353 for rate = 2.
        let p = run_num("X ~ exponential(2); P(X > 1)");
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
        assert!(draws_of("G ~ geometric(1); G", 2000, 7)
            .iter()
            .all(|&x| x == 0.0));
    }

    #[test]
    fn int_variants_round_to_integers() {
        // normal_int / exp_int draw the continuous distribution then round each lane.
        assert!(draws_of("Z ~ normal_int(0, 5); Z", 20_000, 8)
            .iter()
            .all(|&x| x.fract() == 0.0));
        assert!(draws_of("X ~ exponential_int(0.5); X", 20_000, 9)
            .iter()
            .all(|&x| x.fract() == 0.0 && x >= 0.0));
        // Rounding preserves the mean closely: E(normal_int(10, 3)) ≈ 10.
        let m = run_num("Z ~ normal_int(10, 3); E(Z)");
        assert!((m - 10.0).abs() < 0.02, "E(normal_int) = {m}");
    }

    #[test]
    fn new_distribution_domains_and_recipe_display() {
        assert!(run("exponential(0)").is_err()); // rate must be > 0
        assert!(run("exponential(-1)").is_err());
        assert!(run("poisson(0)").is_err()); // lambda must be > 0
        assert!(run("geometric(0)").is_err()); // p must be > 0
        assert!(run("geometric(1.5)").is_err()); // p must be <= 1
                                                 // undrawn recipes print themselves
        assert_eq!(display_of("exponential(2)"), "exponential(2)");
        assert_eq!(display_of("poisson(3)"), "poisson(3)");
        assert_eq!(display_of("geometric(0.5)"), "geometric(0.5)");
        assert_eq!(display_of("normal_int(0, 1)"), "normal_int(0, 1)");
        assert_eq!(display_of("exponential_int(2)"), "exponential_int(2)");
        // recipes bound with `=` stay undrawn; `~` draws independent copies
        let p = run_num("D = poisson(3); a ~ D; b ~ D; P(a == b && a == 0)");
        assert!(
            (p - (-3.0f64).exp().powi(2)).abs() < 3e-3,
            "P(a==b==0) = {p}"
        );
    }

    // --- Q(): distribution quantiles (companion to E/Var/P) ---

    #[test]
    fn quantile_of_known_distributions() {
        // Median of unif(0,1) is 0.5.
        assert!((run_num("X ~ unif(0, 1); Q(X, 0.5)") - 0.5).abs() < 5e-3);
        // The 97.5th percentile of a standard normal is ≈ 1.96.
        assert!((run_num("Z ~ normal(0, 1); Q(Z, 0.975)") - 1.96).abs() < 0.02);
        // Median of Exp(rate) is ln(2)/rate; for rate = 1 that's ≈ 0.693.
        assert!((run_num("X ~ exponential(1); Q(X, 0.5)") - 2.0f64.ln()).abs() < 5e-3);
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
        assert!(boolean(
            "acc = false; for x in [1 < 2, 3 < 4] { acc = acc || x }; acc"
        ));
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
        assert!(
            (p - e).abs() < 1e-9 && (p - 0.5).abs() < 5e-3,
            "P={p} E={e}"
        );
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
        assert_eq!(num("deck ~ permutation(5); E(has_duplicates(deck))"), 0.0);
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
    fn empirical_resamples_the_data() {
        // `empirical(xs)` draws a uniformly-random element of the data: the bootstrap mean is the
        // sample mean, and a value's probability is its multiplicity (exactly-representable data).
        assert!((num("X ~ empirical([1, 2, 3, 4]); E(X)") - 2.5).abs() < 0.01);
        assert!((num("X ~ empirical([1, 2, 2, 5]); P(X == 2)") - 0.5).abs() < 0.01);
        assert!((num("X ~ empirical([1, 2, 2, 5]); P(X == 5)") - 0.25).abs() < 0.01);
        // it's a recipe like any distribution: `=` keeps it undrawn, arithmetic on it errors.
        assert!(run("X = empirical([1, 2, 3]); X + 1").is_err());
    }

    #[test]
    fn empirical_draws_are_iid() {
        // two `~` draws resample independently: P(a == b) = Σ pᵢ² (0.5 for two equal atoms), not 1.
        assert!(
            (num("d = [0, 1]; a ~ empirical(d); b ~ empirical(d); P(a == b)") - 0.5).abs() < 0.01
        );
        // a shaped draw is iid at every leaf — NOT one shared draw repeated (two iid coin values
        // sum to 1 half the time; a shared pair never would).
        assert!((num("xs ~[2] empirical([0, 1]); P(sum(xs) == 1)") - 0.5).abs() < 0.01);
    }

    #[test]
    fn empirical_one_name_one_draw() {
        // binding the draw to a name shares the ONE draw: X - X is identically zero.
        assert_eq!(num("X ~ empirical([1, 2, 3]); P(X - X == 0)"), 1.0);
    }

    #[test]
    fn block_bootstrap_preserves_blocks() {
        // data 0..19 in blocks of 5: the drawn series has the data's length, and inside a block
        // consecutive elements are consecutive data points (difference exactly 1, every lane).
        assert_eq!(num("s ~ block_bootstrap(0..20, 5); Len(s)"), 20.0);
        assert_eq!(
            num("s ~ block_bootstrap(0..20, 5); P(s[1] - s[0] == 1)"),
            1.0
        );
        assert_eq!(
            num("s ~ block_bootstrap(0..20, 5); P(s[4] - s[3] == 1)"),
            1.0
        );
        // across a block boundary the blocks are independent, so a +1 step is rare but possible:
        // start₁ == start₀ + 5, i.e. 11 of the 16² equally-likely start pairs ≈ 0.043.
        let p = num("s ~ block_bootstrap(0..20, 5); P(s[5] - s[4] == 1)");
        assert!(p > 0.02 && p < 0.5, "boundary step probability was {p}");
    }

    #[test]
    fn block_bootstrap_marginals() {
        // data 0..19, b = 5, starts uniform on 0..15 (mean 7.5): element j sits at offset
        // o = j mod 5 inside its block, so its marginal mean is 7.5 + o — NOT the sample mean
        // (non-wrapping starts skew each position toward its offset). Averaged over a whole
        // series the offsets contribute their mean 2, so E(mean(s)) = 9.5 = the sample mean.
        assert!((num("s ~ block_bootstrap(0..20, 5); E(s[1])") - 8.5).abs() < 0.05);
        assert!((num("s ~ block_bootstrap(0..20, 5); E(mean(s), 200000)") - 9.5).abs() < 0.05);
    }

    #[test]
    fn block_bootstrap_degenerate_block_lengths() {
        // block_len == Len(xs): the only valid start is 0, so the series IS the data, every lane.
        assert_eq!(
            num("d = [3, 1, 4, 1, 5]; s ~ block_bootstrap(d, 5); \
                 P(all([for k in 0..5 { s[k] == d[k] }]))"),
            1.0
        );
        // block_len == 1: every element is an independent iid resample (like `empirical`).
        assert!((num("s ~ block_bootstrap([0, 1], 1); P(sum(s) == 1)") - 0.5).abs() < 0.01);
    }

    #[test]
    fn block_bootstrap_draws_are_independent() {
        // two `~` draws pick independent block starts: the first elements agree only when the
        // two starts collide (1/16 for data 0..19, b = 5).
        let p =
            num("a ~ block_bootstrap(0..20, 5); b ~ block_bootstrap(0..20, 5); P(a[0] == b[0])");
        assert!((p - 1.0 / 16.0).abs() < 0.01, "got {p}");
    }

    #[test]
    fn bootstrap_validation_errors() {
        let e = run("X ~ empirical([])").unwrap_err().to_string();
        assert!(e.contains("non-empty"), "{e}");
        // elements must be constant numbers (flat): an RV or a nested array is rejected.
        let e = run("Z ~ normal(0, 1); X ~ empirical([Z, 1])")
            .unwrap_err()
            .to_string();
        assert!(e.contains("constant numbers"), "{e}");
        let e = run("X ~ empirical([[1, 2], [3, 4]])")
            .unwrap_err()
            .to_string();
        assert!(e.contains("constant numbers"), "{e}");
        // block_len must be an integer in 1..=Len(xs).
        for bad in ["0", "4", "1.5"] {
            let e = run(&format!("s ~ block_bootstrap([1, 2, 3], {bad})"))
                .unwrap_err()
                .to_string();
            assert!(e.contains("1 <= block_len <= Len(xs)"), "{e}");
        }
        let e = run("s ~ block_bootstrap([], 1)").unwrap_err().to_string();
        assert!(e.contains("non-empty"), "{e}");
    }

    #[test]
    fn bootstrap_example_core() {
        // the core of examples/bootstrap.noise: bootstrap mean == sample mean, the crash day
        // keeps exactly its 1/24 share of history, and a contiguous resampled week is a real run.
        let rets = "[0.004, -0.006, 0.012, 0.003, -0.002, 0.007, 0.001, -0.005, 0.008, 0.002, \
                     -0.012, -0.04, -0.018, -0.009, 0.011, 0.006, -0.003, 0.009, -0.001, 0.013, \
                     -0.007, 0.002, 0.01, -0.004]";
        let diff = num(&format!(
            "rets = {rets}; r ~ empirical(rets); E(r) - mean(rets)"
        ));
        assert!(
            diff.abs() < 0.001,
            "bootstrap mean off the sample mean by {diff}"
        );
        let p_crash = num(&format!(
            "rets = {rets}; r ~ empirical(rets); P(r <= -0.04)"
        ));
        assert!((p_crash - 1.0 / 24.0).abs() < 0.005, "got {p_crash}");
        // the first 5 elements of a block-5 series form one contiguous historical run, so
        // consecutive differences match some window of the data every lane (spot-check length).
        assert_eq!(
            num(&format!(
                "rets = {rets}; w ~ block_bootstrap(rets, 5); Len(w)"
            )),
            24.0
        );
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
        // vector add/sub are the elementwise `+`/`-` operators
        assert_eq!(display_of("[1, 2] + [3, 4]"), "[4, 6]");
        assert_eq!(display_of("[5, 7] - [1, 2]"), "[4, 5]");
        // `sign` is a math ufunc that maps over arrays (-1 / 0 / +1)
        assert_eq!(display_of("sign([3, 0 - 2, 5, 0])"), "[1, -1, 1, 0]");
        // matrix·vector is the `@` operator: [[1,2],[3,4]] @ [1,1] = [3, 7].
        assert_eq!(display_of("[[1, 2], [3, 4]] @ [1, 1]"), "[3, 7]");
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
        assert_eq!(
            display_of("transpose([[1, 2, 3], [4, 5, 6]])"),
            "[[1, 4], [2, 5], [3, 6]]"
        );
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
        assert_eq!(display_of("[[1, 2], [3, 4]] @ [1, 1]"), "[3, 7]"); // mat·vec
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
        assert!((num("log(e ^ 3)") - 3.0).abs() < 1e-12);
        assert!(run("log(0)").is_err()); // domain x > 0
        assert!(run("log10(0 - 5)").is_err());
        // vec::mse — mean squared error between two signals
        assert!((num("mse([1, 2, 3], [1, 2, 5])") - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(num("mse([5, 5], [5, 5])"), 0.0); // identical signals
        assert!(run("mse([1, 2], [1, 2, 3])").is_err()); // length mismatch
                                                         // mse equals the noise power of an additive channel: received = signal + noise.
        let p = run_num("sig = ones(8); noise ~[8] normal(0, 2); E(mse(sig + noise, sig))");
        assert!(
            (p - 4.0).abs() < 0.1,
            "additive-noise MSE = {p} (want sigma^2 = 4)"
        );
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
        // signal×signal arithmetic DEFERS too (PLAN-SIGNALS §3): a two-tone chord stays lazy…
        assert_eq!(run("sine(3) + sine(7)").unwrap().type_name(), "signal");
        // …and materializes to the sum of the two rendered tones.
        assert_eq!(
            display_of("sample(sine(3) + sine(7), 16)"),
            display_of("a = sample(sine(3), 16); b = sample(sine(7), 16); a + b")
        );
        // math::exp defers into a signal (the deterministic FM building block).
        assert_eq!(
            run("use math; exp(0.5 * sine(3))").unwrap().type_name(),
            "signal"
        );
        let e0 = run_num("use math; sample(exp(0.5 * sine(3)), 4)[0]");
        assert!((e0 - 1.0).abs() < 1e-12, "exp(0.5·sin(0)) = {e0} (want 1)");
        // prefix negation defers as well.
        assert_eq!(
            display_of("sample(-sine(4), 4)"),
            display_of("0 - sine(4, 4)")
        );
        assert!(run("sample(5, 8)").is_err()); // sample needs a signal
    }

    #[test]
    fn lazy_noise_colors_materialize_to_correlated_normals() {
        // noise_*(…) are UNDRAWN generators — recipes, like normal(0, 1); `~` is the only way in.
        assert_eq!(run("noise_white(0.5)").unwrap().type_name(), "noise");
        assert_eq!(run("noise_brown(1)").unwrap().type_name(), "noise");
        assert_eq!(run("noise_ou(1, 8)").unwrap().type_name(), "noise");
        // `w ~[n] noise` pins one realization to length n — an ordinary array of zero-mean
        // normals; white has variance sigma^2.
        assert_eq!(num("w ~[16] noise_pink(1); Len(w)"), 16.0);
        assert!(run_num("z ~[6] noise_white(2); E(mean(z), 40000)").abs() < 0.05);
        let v = run_num("z ~[6] noise_white(2); Var(z[0], 80000)");
        assert!((v - 4.0).abs() < 0.2, "white Var = {v} (want sigma^2 = 4)");
        // The spectral COLOR is the lag-1 autocorrelation E[x_k x_{k+1}] / E[x_k^2]:
        // white ~ 0, OU = exp(-1/tau), brown ~ 1. We assert the ordering and the OU closed form.
        let corr = "pair(x) = { a = 0; for i in 0..Len(x)-1 { a = a + x[i]*x[i+1] }; a }; \
                    sq(x)   = { a = 0; for i in 0..Len(x)-1 { a = a + x[i]*x[i]   }; a }; \
                    c(x)    = E(pair(x), 4000) / E(sq(x), 4000); ";
        let white = run_num(&format!("{corr} w ~[200] noise_white(1); c(w)"));
        let ou = run_num(&format!("{corr} w ~[200] noise_ou(1, 8); c(w)"));
        let brown = run_num(&format!("{corr} w ~[200] noise_brown(1); c(w)"));
        assert!(
            white.abs() < 0.05,
            "white neighbor corr = {white} (want ~0)"
        );
        assert!(
            (ou - (-1.0f64 / 8.0).exp()).abs() < 0.05,
            "OU corr = {ou} (want exp(-1/8))"
        );
        assert!(brown > 0.9, "brown neighbor corr = {brown} (want ~1)");
        assert!(
            white < ou && ou < brown,
            "color should redden: {white} < {ou} < {brown}"
        );
        // an UNDRAWN generator in arithmetic or `sample` is the recipe error, pointing at `~`.
        let e = run("noise_pink(0.5) * 2").unwrap_err().to_string();
        assert!(
            e.contains("undrawn distribution") && e.contains('~'),
            "got: {e}"
        );
        let e = run("sample(noise_white(1), 8)").unwrap_err().to_string();
        assert!(e.contains("undrawn distribution"), "got: {e}");
        // bad params are still errors.
        assert!(run("noise_ou(1, 0)").is_err()); // tau must be > 0
        assert!(run("noise_brown(0 - 1)").is_err()); // sigma must be >= 0
    }

    #[test]
    fn drawn_noise_is_one_realization() {
        // A `~`-drawn noise is ONE realization: every mention is the same noise. Two `sample`
        // calls reuse the same RV nodes, so a[0] - b[0] is exactly 0 (PLAN-SIGNALS §1.1) — this
        // is the hidden-re-draw defect the plan closes (it used to be Var = 2·sigma^2).
        let v = run_num(
            "static ~ noise_white(1); a = sample(static, 4); b = sample(static, 4); \
             Var(a[0] - b[0], 20000)",
        );
        assert!(
            v.abs() < 1e-9,
            "same realization must cancel exactly, got Var = {v}"
        );
        // …and `static - static` is a zero signal, not "two samples".
        let z = run_num("static ~ noise_white(1); Var(sample(static - static, 8)[0], 20000)");
        assert!(z.abs() < 1e-12, "static - static = {z} (want exactly 0)");
        // The plain `~` form stays length-lazy and PINS at first materialization: a later mention
        // at a different length is the length-clash error naming the pinned length.
        let e = run("static ~ noise_white(1); a = sample(static, 8); sample(static, 16)")
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("realized at length 8") && e.contains("16"),
            "got: {e}"
        );
        // A drawn realization composes lazily: signal arithmetic over it defers, and reducers
        // see the same noise both times.
        let v = run_num(
            "static ~ noise_white(1); rx = 2 * static; \
             a = sample(rx, 4); b = sample(rx, 4); Var(a[2] - b[2], 20000)",
        );
        assert!(
            v.abs() < 1e-9,
            "derived signals share the realization, got Var = {v}"
        );
        // An `=`-bound generator stays a recipe: each `~` draw of it is independent.
        let v = run_num("G = noise_white(1); x ~[4] G; y ~[4] G; Var(x[0] - y[0], 60000)");
        assert!(
            (v - 2.0).abs() < 0.15,
            "independent draws should give Var 2, got {v}"
        );
    }

    #[test]
    fn reducers_render_at_the_ambient_resolution() {
        // A reducer meeting a lazy signal renders it at the ambient resolution (PLAN-SIGNALS
        // §1.2) — no inline length anywhere. mean(sin²) over whole cycles is 1/2.
        let m = run_num("mean(sine(3) ^ 2)");
        assert!((m - 0.5).abs() < 1e-9, "mean(sine²) = {m} (want 0.5)");
        // mse renders BOTH sides at the same ambient length (two-tone vs silence).
        let e = run_num("mse(sine(3) + sine(7), 0 * sine(3))");
        assert!((e - 1.0).abs() < 1e-9, "mse(two-tone, 0) = {e} (want 1)");
        // The default resolution is 256: a drawn noise rendered by a reducer pins at 256…
        assert_eq!(
            num("static ~ noise_white(1); x = E(mean(static), 100); Len(sample(static, 256))"),
            256.0
        );
        // …and `engine::set_resolution(N)` changes what reducers use (the realization pins at 8,
        // so re-rendering at 256 is now the length clash).
        let ok = run("engine::set_resolution(8); static ~ noise_white(1); \
             x = E(mean(static), 100); Len(sample(static, 8))");
        assert_eq!(ok.unwrap(), Value::Num(8.0));
        let e = run("engine::set_resolution(8); static ~ noise_white(1); \
             x = E(mean(static), 100); sample(static, 256)")
        .unwrap_err()
        .to_string();
        assert!(e.contains("realized at length 8"), "got: {e}");
        // set_resolution validates its argument like the other engine knobs.
        assert!(run("engine::set_resolution(0)").is_err());
        assert!(run("engine::set_resolution(2.5)").is_err());
    }

    #[test]
    fn complex_signals_by_composition() {
        // signal + i·signal is a COMPLEX SIGNAL — Complex{re: Signal, im: Signal}, no new kind.
        let z = run("sine(3) + math::i * sine(7)").unwrap();
        assert_eq!(z.type_name(), "complex");
        // sample of a complex signal → an array of complex, channels rendered consistently.
        assert_eq!(num("Len(sample(sine(3) + math::i * sine(7), 8))"), 8.0);
        let im1 = run_num("im(sample(sine(2) + math::i, 4)[1])");
        assert!(
            (im1 - 1.0).abs() < 1e-12,
            "constant im channel = {im1} (want 1)"
        );
        // abs/arg of a complex signal stay lazy and demodulate correctly:
        // |e^{i·θ(t)}| = 1 and arg(e^{i·θ(t)}) = θ(t) for a small angle.
        assert_eq!(
            run("abs(exp(i * 0.3 * sine(3)))").unwrap().type_name(),
            "signal"
        );
        let m = run_num("mse(arg(exp(i * 0.3 * sine(3))), 0.3 * sine(3))");
        assert!(m < 1e-18, "lossless FM round-trip, got mse = {m}");
        let a = run_num("mean(abs(exp(i * 0.3 * sine(3))))");
        assert!((a - 1.0).abs() < 1e-12, "unit carrier magnitude, got {a}");
        // plotting a complex signal is a teaching error (no single trace).
        let e = run("plot::line(sine(3) + math::i * sine(7))")
            .unwrap_err()
            .to_string();
        assert!(e.contains("no single trace"), "got: {e}");
    }

    #[test]
    fn noise_white_complex_is_a_drawn_cscg() {
        // noise_white_complex(σ) is undrawn like the rest; `~` draws a lazy complex realization.
        assert_eq!(run("noise_white_complex(1)").unwrap().type_name(), "noise");
        assert!(run("noise_white_complex(1) + math::i").is_err());
        assert_eq!(
            run("z ~ noise_white_complex(1); z").unwrap().type_name(),
            "complex"
        );
        // Total-power convention matches rand::normal_complex: E|z|² = σ² (σ = 2 ⇒ 4),
        // split evenly across the quadratures.
        let p = run_num("z ~[64] noise_white_complex(2); E(normsq(z) / 64, 20000)");
        assert!((p - 4.0).abs() < 0.2, "E|z|² = {p} (want σ² = 4)");
        let re_var = run_num("z ~[8] noise_white_complex(2); Var(re(z[0]), 40000)");
        assert!(
            (re_var - 2.0).abs() < 0.15,
            "per-quadrature var = {re_var} (want σ²/2 = 2)"
        );
        // ONE realization: both channels re-render consistently across two samples.
        let v = run_num(
            "z ~ noise_white_complex(1); a = sample(z, 4); b = sample(z, 4); \
             Var(re(a[0]) - re(b[0]) + im(a[1]) - im(b[1]), 20000)",
        );
        assert!(
            v.abs() < 1e-9,
            "complex realization must be shared, got Var = {v}"
        );
    }

    #[test]
    fn am_vs_fm_all_in_signal_land() {
        // The PLAN-SIGNALS §2 flagship: the whole modulate → add static → demodulate chain stays
        // lazy; the two measurement knobs sit at the top, one per axis. The numbers must land on
        // the sampled-first pipeline's (AM ≈ σ²/2 + a small demod bias, FM ≈ AM/dev² in the
        // small-signal limit — measured 0.0777 / 0.0099 at these settings).
        let src = "
            engine::set_max_samples(8000);
            engine::set_resolution(64);
            am_modulate(m)        = 1 + m;
            fm_modulate(m, dev)   = exp(i * dev * m);
            am_demodulate(z)      = abs(z) - 1;
            fm_demodulate(z, dev) = arg(z) / dev;
            dev   = 3;
            sigma = 0.4;
            msg    = 0.3 * sine(3);
            static ~ noise_white_complex(sigma);
            rec_am = am_demodulate(am_modulate(msg)      + static);
            rec_fm = fm_demodulate(fm_modulate(msg, dev) + static, dev);
            [E(mse(rec_am, msg)), E(mse(rec_fm, msg))]
        ";
        let (am, fm) = match run(src).unwrap() {
            Value::Array(xs) => {
                let get = |v: &Value| match v {
                    Value::Num(n) | Value::Est { val: n, .. } => *n,
                    other => panic!("expected number, got {other:?}"),
                };
                (get(&xs[0]), get(&xs[1]))
            }
            other => panic!("expected array, got {other:?}"),
        };
        assert!((am - 0.0777).abs() < 0.01, "AM err = {am} (want ≈ 0.0777)");
        assert!((fm - 0.0099).abs() < 0.003, "FM err = {fm} (want ≈ 0.0099)");
        let ratio = am / fm;
        assert!(
            ratio > 5.0 && ratio < 9.5,
            "FM advantage ≈ dev² = 9 (a bit under at finite noise), got {ratio}"
        );
    }

    #[test]
    fn nyquist_aliasing_via_sampling_rate() {
        // Nyquist-Shannon: a freq-7 wave needs > 2*7 = 14 samples. Undersampled at 10 it aliases to
        // the freq-3 wave (7 folds to 10-7 = 3) — identical samples, mse 0. Oversampled, they differ.
        let aliased = num("mse(sample(cosine(7), 10), sample(cosine(3), 10))");
        assert!(
            aliased < 1e-12,
            "undersampled should alias (mse 0), got {aliased}"
        );
        let resolved = num("mse(sample(cosine(7), 64), sample(cosine(3), 64))");
        assert!(
            resolved > 0.5,
            "oversampled should distinguish them, got {resolved}"
        );
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
        assert!(
            (m - (-0.5f64).exp()).abs() < 3e-3,
            "E[cos(X)] = {m} (want e^-0.5)"
        );
        // `sign` is a ufunc too: scalar -1/0/+1, maps over arrays, and lifts over RVs.
        assert_eq!(num("sign(3.2)"), 1.0);
        assert_eq!(num("sign(0 - 7)"), -1.0);
        assert_eq!(num("sign(0)"), 0.0);
        assert_eq!(display_of("sign([2, 0 - 3, 0])"), "[1, -1, 0]");
        // E[sign(X)] for symmetric X ~ N(0,1) is 0 (it lifts over the RV per lane).
        assert!(run_num("X ~ normal(0, 1); E(sign(X))").abs() < 3e-3);
    }

    #[test]
    fn exp_log_scalars_and_arrays() {
        // Scalar sanity: exp/log are exact inverses on the deterministic path.
        assert_eq!(num("exp(0)"), 1.0);
        assert!((num("exp(1)") - std::f64::consts::E).abs() < 1e-12);
        assert!((num("log(e)") - 1.0).abs() < 1e-12);
        assert_eq!(num("log(1)"), 0.0);
        assert!((num("exp(log(5))") - 5.0).abs() < 1e-12);
        // Arrays map elementwise: log([1, e, e^2]) ≈ [0, 1, 2], exp keeps shape.
        assert_eq!(num("log([1, e, e ^ 2])[0]"), 0.0);
        assert!((num("log([1, e, e ^ 2])[1]") - 1.0).abs() < 1e-12);
        assert!((num("log([1, e, e ^ 2])[2]") - 2.0).abs() < 1e-12);
        assert_eq!(display_of("exp([0, 0])"), "[1, 1]");
        // log10 rides on the same machinery (scalar fast path stays exact).
        assert_eq!(num("log10(1000)"), 3.0);
        assert!((num("log10([1, 100])[1]") - 2.0).abs() < 1e-12);
    }

    #[test]
    fn exp_log_lift_over_random_variables() {
        // Lognormal mean: X ~ N(0,1) → E[e^X] = e^{1/2} (the PLAN-FINANCE F1 unlock).
        let m = run_num("X ~ normal(0, 1); E(exp(X))");
        let want = (0.5f64).exp();
        assert!(
            (m - want).abs() < 3e-2 * want,
            "E[e^X] = {m} (want e^0.5 = {want})"
        );
        // E[ln U] over U(0,1) = -1.
        let l = run_num("U ~ unif(0, 1); E(log(U))");
        assert!((l + 1.0).abs() < 1e-2, "E[ln U] = {l} (want -1)");
        // Lognormal price: S = 100·e^X with X ~ N(0, 0.25) → E[S] = 100·e^{σ²/2} = 100·e^{0.03125}.
        let s = run_num("X ~ normal(0, 0.25); S = 100 * exp(X); E(S)");
        let want_s = 100.0 * (0.03125f64).exp();
        assert!((s - want_s).abs() < 0.5, "E[S] = {s} (want {want_s})");
        // log10 of an RV lifts as Ln/ln10: a point mass at 100 comes back as exactly-2-ish.
        let l10 = run_num("X ~ unif_int(100, 100); E(log10(X))");
        assert!((l10 - 2.0).abs() < 1e-9, "E[log10(100)] = {l10}");
        // Domain semantics per lane match f64: e^X is surely positive; negative lanes of log are
        // NaN (NaN == NaN is false), so P(log X == log X) = P(X > 0).
        assert_eq!(run_num("X ~ normal(0, 1); P(exp(X) > 0)"), 1.0);
        let p = run_num("X ~ normal(0, 1); P(log(X) == log(X))");
        assert!((p - 0.5).abs() < 3e-3, "P(log X == log X) = {p} (want 0.5)");
        // E[ln|X|] for X ~ N(0,1) = -(γ + ln 2)/2 ≈ -0.6352 — finite despite the |X| → 0 lanes.
        let la = run_num("X ~ normal(0, 1); E(log(abs(X)))");
        assert!(
            (la + 0.6352).abs() < 2e-2,
            "E[ln|X|] = {la} (want ≈ -0.6352)"
        );
        // exp/log map elementwise over an *array of RVs*: E[Σₖ e^{Xₖ}] = 3·e^{1/2}.
        let a = run_num("xs ~[3] normal(0, 1); E(sum(exp(xs)))");
        assert!(
            (a - 3.0 * (0.5f64).exp()).abs() < 0.15,
            "E[Σ e^X] = {a} (want 3·e^0.5)"
        );
        // A *noisy lazy signal* defers exp into its tree and lifts per lane at materialization.
        let sg = run_num("static ~ noise_white(1); E(mean(sample(exp(static), 16)))");
        assert!(
            (sg - (0.5f64).exp()).abs() < 0.05,
            "E[mean e^noise] = {sg} (want e^0.5)"
        );
    }

    #[test]
    fn kelly_log_growth_peaks_at_kelly_fraction() {
        // A 60% coin that doubles-or-nothings the staked fraction f: per-round growth factor is
        // 1+f on a win, 1-f on a loss. The long-run compounding rate is E[log g], and its analytic
        // peak is the Kelly fraction f* = 2p - 1 = 0.2 (mirrors examples/kelly.noise).
        let growth = |f: f64| {
            run_num(&format!(
                "win ~ bernoulli(0.6); g = if win {{ 1 + {f} }} else {{ 1 - {f} }}; E(log(g))"
            ))
        };
        let analytic = |f: f64| 0.6 * (1.0 + f).ln() + 0.4 * (1.0 - f).ln();
        for f in [0.1, 0.2, 0.3] {
            let (got, want) = (growth(f), analytic(f));
            assert!(
                (got - want).abs() < 2e-3,
                "E[log g] at f={f}: {got} (want {want})"
            );
        }
        // The Kelly point beats both neighbours (analytic gaps ≈ 5e-3 >> MC noise).
        assert!(growth(0.2) > growth(0.1) && growth(0.2) > growth(0.3));
    }

    #[test]
    fn arithmetic_broadcasts_over_arrays() {
        assert_eq!(display_of("1 + [1, 2, 3]"), "[2, 3, 4]"); // scalar ⊕ array
        assert_eq!(display_of("[1, 2, 3] + [10, 20, 30]"), "[11, 22, 33]"); // array ⊕ array
        assert_eq!(display_of("[2, 4, 6] / 2"), "[1, 2, 3]");
        assert_eq!(display_of("[1, 2, 3] ^ 2"), "[1, 4, 9]");
        // nested: an array of arrays broadcasts recursively ([I,Q] + [nI,nQ]).
        assert_eq!(
            display_of("[[1, 2], [3, 4]] + [[10, 20], [30, 40]]"),
            "[[11, 22], [33, 44]]"
        );
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
                   am_demod(iq) = (iq[0] ^ 2 + iq[1] ^ 2) ^ 0.5 - 1; \
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
        assert!(
            (am - 0.09).abs() < 0.02,
            "AM error = {am} (want ~ sigma^2 = 0.09)"
        );
        // FM should be several times cleaner for the same static (advantage grows with deviation).
        assert!(fm < am * 0.5, "FM error {fm} should be << AM error {am}");
    }

    #[test]
    fn rotation_is_orthonormal() {
        // `rotation(d)` is a random orthonormal matrix, so every sample preserves length exactly:
        // ||Pi x|| = ||x|| = 1 (the mean is exactly 1, hence a tight tolerance at tiny N).
        let nrm =
            run_num("d = 8; x = normalize(ones(d)); Pi ~ rotation(d); E(normsq(Pi @ x), 100)");
        assert!(
            (nrm - 1.0).abs() < 1e-9,
            "||Pi x||^2 = {nrm}, want exactly 1"
        );
        // And it round-trips: Pi^T Pi x = x (same Pi reused, so transpose is the inverse).
        let rt = run_num(
            "d = 8; Pi ~ rotation(d); x = normalize(iota(d)); \
             E(normsq(transpose(Pi) @ (Pi @ x) - x), 100)",
        );
        assert!(rt < 1e-9, "||Pi^T Pi x - x||^2 = {rt}, want ~0");
    }

    #[test]
    fn rotation_is_a_recipe_drawn_with_tilde() {
        // `rotation(d)` is a recipe like any distribution: `~` draws it, `=` keeps it undrawn.
        // Using an undrawn rotation in arithmetic is the usual "draw it first" error.
        assert!(run("d = 4; Pi = rotation(d); x = ones(d); Pi @ x").is_err());
        // A shaped draw gives k *independent* rotations: stack two and they should differ, so the
        // squared distance between Pi0 x and Pi1 x is comfortably positive (identical draws → 0).
        let spread = run_num(
            "d = 6; x = normalize(ones(d)); Rs ~[2] rotation(d); \
             E(normsq(Rs[0] @ x - Rs[1] @ x), 2000)",
        );
        assert!(
            spread > 0.1,
            "two independent rotations should differ, got spread {spread}"
        );
    }

    #[test]
    fn quantize_snaps_to_nearest_centroid() {
        // Each coordinate snaps to its nearest codebook entry; cell boundaries are the midpoints.
        assert_eq!(
            display_of("quantize([-2, -0.1, 0.1, 2], [-1, 1])"),
            "[-1, -1, 1, 1]"
        );
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
                      Pi ~ rotation(d); mse = transpose(Pi) @ quantize(Pi @ x, L1); \
                      S ~[d, d] normal(0, 1); r = x - mse; \
                      prod = dot(y, mse) \
                           + sqrt(pi / 2) / d * norm(r) \
                             * dot(y, transpose(S) @ sign(S @ r)); ";
        // Algorithm 1's distortion table, b=1 entry: D_mse ~ 0.36.
        let dmse = run_num(&format!("{common} E(normsq(x - mse), 12000)"));
        assert!(
            (dmse - 0.36).abs() < 0.05,
            "D_mse(b=1) = {dmse}, want ~0.36"
        );
        // The MSE quantizer is biased by 2/pi; the two-stage prod estimate is unbiased.
        let mse_ratio = run_num(&format!("{common} E(dot(y, mse) / t, 12000)"));
        let prod_ratio = run_num(&format!("{common} E(prod / t, 12000)"));
        assert!(
            (mse_ratio - 2.0 / std::f64::consts::PI).abs() < 0.05,
            "MSE ratio = {mse_ratio}"
        );
        assert!((prod_ratio - 1.0).abs() < 0.05, "prod ratio = {prod_ratio}");
        // The payoff: the unbiased two-stage estimate has far lower mean-squared inner-product error
        // than the biased MSE quantizer, whose error is dominated by its 2/pi bias floor.
        let mse_err = run_num(&format!("{common} E((dot(y, mse) - t) ^ 2, 12000)"));
        let prod_err = run_num(&format!("{common} E((prod - t) ^ 2, 12000)"));
        assert!(
            prod_err < mse_err * 0.6,
            "prod err {prod_err} should be << MSE err {mse_err}"
        );
    }

    #[test]
    fn linear_algebra_length_mismatches_error() {
        assert!(run("dot([1, 2], [1, 2, 3])").is_err());
        assert!(run("[1] + [1, 2]").is_err());
        assert!(run("[1, 2, 3] - [1, 2]").is_err());
        assert!(run("dot(5, [1, 2])").is_err()); // non-array operand
    }

    #[test]
    fn has_duplicates_correctness() {
        assert!(!boolean("has_duplicates([1, 2, 3])"));
        assert!(boolean("has_duplicates([1, 2, 2, 3])"));
        assert!(!boolean("has_duplicates([])"));
        assert!(!boolean("has_duplicates([7])"));
    }

    #[test]
    fn count_duplicates_correctness() {
        // counts equal pairs i<j: none, one pair, and three pairs among [2,2,2].
        assert_eq!(num("count_duplicates([1, 2, 3])"), 0.0);
        assert_eq!(num("count_duplicates([1, 2, 2, 3])"), 1.0);
        assert_eq!(num("count_duplicates([2, 2, 2])"), 3.0);
        assert_eq!(num("count_duplicates([])"), 0.0);
        assert_eq!(num("count_duplicates([7])"), 0.0);
    }

    // --- PLAN-FINANCE F3: prefix scans (cumsum/cumprod/cummax/cummin) + the prod reducer ---

    #[test]
    fn scans_deterministic() {
        assert_eq!(display_of("cumsum([1, 2, 3, 4])"), "[1, 3, 6, 10]");
        assert_eq!(display_of("cumprod([1, 2, 3, 4])"), "[1, 2, 6, 24]");
        assert_eq!(display_of("cummax([3, 1, 4, 2])"), "[3, 3, 4, 4]");
        assert_eq!(display_of("cummin([3, 1, 4, 2])"), "[3, 1, 1, 1]");
        assert_eq!(num("prod([1, 2, 3, 4])"), 24.0);
        // Scans route through the same `binop` as `sum`, so a matrix scans by elementwise
        // row folds (mirroring `sum(matrix)` = elementwise row-sum vector).
        assert_eq!(display_of("cumsum([[1, 2], [3, 4]])"), "[[1, 2], [4, 6]]");
        assert_eq!(display_of("prod([[1, 2], [3, 4]])"), "[3, 8]");
    }

    #[test]
    fn scan_last_element_matches_reducer() {
        let xs = "xs = [2.5, -1, 4, 0.5];";
        assert_eq!(num(&format!("{xs} cumsum(xs)[3] - sum(xs)")), 0.0);
        assert_eq!(num(&format!("{xs} cumprod(xs)[3] - prod(xs)")), 0.0);
        assert_eq!(num(&format!("{xs} cummax(xs)[3] - max(xs)")), 0.0);
        assert_eq!(num(&format!("{xs} cummin(xs)[3] - min(xs)")), 0.0);
    }

    #[test]
    fn scans_and_prod_on_empty_arrays() {
        // A scan of an empty array is the empty array (every output element summarizes a
        // nonempty prefix, so there are simply no elements to output). `prod`'s empty fold is
        // the multiplicative identity 1, mirroring `sum`'s additive 0 — while the extremum
        // *reducers* keep erroring (no extremum of nothing).
        assert_eq!(display_of("cumsum([])"), "[]");
        assert_eq!(display_of("cumprod([])"), "[]");
        assert_eq!(display_of("cummax([])"), "[]");
        assert_eq!(display_of("cummin([])"), "[]");
        assert_eq!(num("prod([])"), 1.0);
        assert_eq!(num("sum([])"), 0.0);
        assert!(run("max([])").is_err());
        assert!(run("min([])").is_err());
    }

    #[test]
    fn cumsum_of_mixed_const_and_rv_array() {
        // Literal arrays mixing constants and RVs scan like any other: E[1 + U + 2] = 3.5.
        let e = run_num("X ~ unif(0, 1); c = cumsum([1, X, 2]); E(c[2])");
        assert!((e - 3.5).abs() < 0.05, "E(cumsum [1,X,2] last) = {e}");
        // the constant prefix stays a plain number
        assert_eq!(run_num("X ~ unif(0, 1); cumsum([1, X, 2])[0]"), 1.0);
    }

    #[test]
    fn cumsum_over_shaped_draws_is_a_random_walk() {
        // path[t] = Σ steps[0..=t] of iid normal(0.01, 0.1): E[path[t]] = (t+1)·0.01 and
        // Var[path[99]] = 100·0.01 = 1.0 (independent increments add in variance).
        let common = "steps ~[100] normal(0.01, 0.1); path = cumsum(steps);";
        let e99 = run_num(&format!("{common} E(path[99], 200000)"));
        assert!((e99 - 1.0).abs() < 0.02, "E(path[99]) = {e99}");
        let e49 = run_num(&format!("{common} E(path[49], 200000)"));
        assert!((e49 - 0.5).abs() < 0.02, "E(path[49]) = {e49}");
        let v99 = run_num(&format!("{common} Var(path[99], 200000)"));
        assert!((v99 - 1.0).abs() < 0.05, "Var(path[99]) = {v99}");
        // path[9] is ONE random variable reused (the scan shares structure), not a fresh draw:
        // subtracting it from itself is exactly 0 in every lane.
        assert_eq!(
            run_num(&format!("{common} P(path[9] - path[9] == 0, 10000)")),
            1.0
        );
    }

    #[test]
    fn cumprod_gbm_one_liner() {
        // GBM in one line: 52 weekly returns, E[S_52] = 100·(1.001)^52 ≈ 105.33.
        let e = run_num(
            "rets ~[52] normal(0.001, 0.02); path = 100 * cumprod(1 + rets); \
             E(path[51], 200000)",
        );
        let want = 100.0 * 1.001f64.powi(52);
        assert!((e - want).abs() < 0.25, "E(path[51]) = {e}, want ~{want}");
    }

    #[test]
    fn barrier_via_scan_matches_hand_rolled_loop() {
        // P(the walk ever touches the barrier): the scan idiom must agree (statistically —
        // different RNG streams) with the hand-written for-loop accumulator of the SAME model.
        let scan = run_num(
            "zs ~[52] normal(0, 1); \
             path = 100 * cumprod(exp(0.001 + 0.02 * zs)); \
             P(any(path < 90), 100000)",
        );
        let looped = run_num(
            "s = 100; knocked = false; \
             for t in 0..52 { z ~ normal(0, 1); s = s * exp(0.001 + 0.02 * z); \
                              knocked = knocked || (s < 90) }; \
             P(knocked, 100000)",
        );
        assert!(
            (scan - looped).abs() < 0.02,
            "scan barrier P = {scan}, hand-rolled P = {looped}"
        );
        // sanity: the event is neither impossible nor certain
        assert!(scan > 0.05 && scan < 0.95, "P(knocked) = {scan}");
    }

    #[test]
    fn drawdown_one_liner_via_cummax() {
        // Max drawdown = min(path / running peak) - 1: strictly inside (-1, 0) for a walk
        // that moves (it can't gain relative to its own peak, and can't lose everything).
        let dd = run_num(
            "rets ~[52] normal(0.001, 0.02); path = 100 * cumprod(1 + rets); \
             dd = min(path / cummax(path)) - 1; E(dd, 50000)",
        );
        assert!(
            dd > -1.0 && dd < 0.0,
            "E(max drawdown) = {dd}, want in (-1, 0)"
        );
    }

    #[test]
    fn barrier_option_vanilla_leg_matches_black_scholes() {
        // The examples/barrier_option.noise pricing core: exact per-step GBM (log-space cumsum
        // through exp), so the vanilla European call must match Black-Scholes 10.4506 for
        // S0=100, K=100, r=0.05, sigma=0.2, T=1 (any step count — the discretization is exact).
        let price = run_num(
            "s0 = 100; k = 100; r = 0.05; sigma = 0.2; t = 1; n = 52; dt = t / n; \
             zs ~[52] normal(0, 1); \
             logrets = (r - sigma^2 / 2) * dt + sigma * sqrt(dt) * zs; \
             path = s0 * exp(cumsum(logrets)); \
             s_final = path[51]; \
             payoff = if s_final > k { s_final - k } else { 0 }; \
             E(exp(0 - r * t) * payoff, 400000)",
        );
        assert!(
            (price - 10.4506).abs() < 0.2,
            "vanilla call = {price}, BS wants 10.4506"
        );
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
        assert_eq!(
            num("acc = 0; for x in [1, 2, 3, 4] { acc = acc + x }; acc"),
            10.0
        );
        // nested loops
        assert_eq!(
            num("acc = 0; for i in 0..3 { for j in 0..3 { acc = acc + 1 } }; acc"),
            9.0
        );
        // a zero-iteration loop runs the body zero times and yields unit (graph untouched).
        assert_eq!(
            run("for i in 0..0 { x ~ unif(0, 1) }").unwrap(),
            Value::Unit
        );
        assert_eq!(graph_len("for i in 0..0 { x ~ unif(0, 1) }"), 0);
    }

    #[test]
    fn for_loop_unrolls_at_build_time() {
        // Each iteration draws a fresh node, so a loop of n builds exactly n× the body's nodes —
        // proving unroll (not runtime branching). One `unif` draw is one source node.
        let one = graph_len("x ~ unif(0, 1)");
        assert_eq!(graph_len("for i in 0..5 { x ~ unif(0, 1) }"), 5 * one);
        // `~` inside the loop gives independence: sum of n iid uniforms has variance n/12.
        let m = moments_of(
            "acc = 0; for i in 0..12 { u ~ unif(0, 1); acc = acc + u }; acc",
            1_000_000,
            7,
        );
        assert!(
            (m.variance - 12.0 / 12.0).abs() < 0.02,
            "var = {}",
            m.variance
        );
    }

    #[test]
    fn for_loop_errors_on_a_non_array() {
        assert!(run("for x in 5 { x }").is_err());
        assert!(run("Len(5)").is_err());
    }

    #[test]
    fn library_matches_a_noise_written_version() {
        // The §0 property: each library function IS expressible in Noise. A Noise-written `sum`,
        // `dot`, and `has_duplicates` must match the builtin on the same inputs.
        let noise_sum = "mysum(xs) = { acc = 0; for x in xs { acc = acc + x }; acc };";
        assert_eq!(
            num(&format!("{noise_sum} mysum([1, 2, 3, 4])")),
            num("sum([1, 2, 3, 4])")
        );

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
        let p = run_num("days ~[23] unif_int(1, 365); P(has_duplicates(days))");
        assert!(
            (p - 0.5073).abs() < 5e-3,
            "P(shared birthday among 23) = {p}"
        );
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
            ("P(sum(~[2] unif_int(1, 6)) == 7)", 1.0 / 6.0), // dice_sum
            ("P(all(~[3] bernoulli(0.5)))", 0.125),          // coin_streak
            ("P(count(~[3] bernoulli(0.5)) == 2)", 0.375),   // exactly_two_heads
            ("P(sum(~[3] unif(0, 1)) > 2)", 1.0 / 6.0),      // irwin_hall
            ("P(any(~[3] bernoulli(0.9)))", 0.999),          // reliability
        ];
        for (src, expected) in cases {
            let p = run_num(src);
            assert!(
                (p - expected).abs() < 5e-3,
                "{src} = {p}, expected {expected}"
            );
        }
    }

    // --- module system: `mod::name` paths, `use`, strict scoping (these bypass the prelude) ---

    #[test]
    fn qualified_paths_resolve_without_use() {
        // A `mod::name` path reaches a module's items even with nothing `use`d.
        assert_eq!(run_raw("math::sqrt(16)").unwrap(), Value::Num(4.0));
        assert!((as_num(run_raw("math::pi").unwrap()) - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(run_raw("vec::sum([1, 2, 3])").unwrap(), Value::Num(6.0));
        assert_eq!(
            run_raw("vec::dot([1, 2], [3, 4])").unwrap(),
            Value::Num(11.0)
        );
        // a random constructor path draws fine (the qualified `rand::unif` needs no `use`)
        assert!(run_raw("X ~ rand::unif(0, 1); P(X < 0.5, 1000)").is_ok());
    }

    #[test]
    fn use_brings_a_module_into_unqualified_scope() {
        // After `use`, bare names resolve — like Rust.
        assert_eq!(run_raw("use math; sqrt(25)").unwrap(), Value::Num(5.0));
        assert_eq!(run_raw("use vec; sum([10, 20])").unwrap(), Value::Num(30.0));
        let p = run_raw("use rand; use vec; P(has_duplicates(~[2] unif_int(1, 6)))").unwrap();
        assert!((as_num(p) - 1.0 / 6.0).abs() < 5e-3);
        // `use builtin;` is a harmless no-op (always active anyway).
        assert_eq!(
            run_raw("use builtin; Len([1, 2])").unwrap().to_string(),
            "2"
        );
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
            assert!(
                msg.contains(module) && msg.contains("use"),
                "{src} -> {msg}"
            );
        }
    }

    #[test]
    fn module_path_errors_are_specific() {
        // wrong module for a real function
        assert!(run_raw("math::unif(0, 1)")
            .unwrap_err()
            .to_string()
            .contains("rand"));
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
    /// The point this measures: `@`/`rotation` carry no GEMM loop. They expand
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
             (est - true_ip) ^ 2"
        );

        let n = 1_000_000usize;
        for (label, src, target) in [
            ("D_mse  b=4", d_mse.as_str(), 0.009),
            ("D_prod b=4", d_prod.as_str(), 0.047 / 20.0),
        ] {
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

    // ===================== complex numbers (PLAN-COMPLEX) =====================

    /// Pull `(re, im)` out of a constant `Value::Complex` (or promote a real scalar).
    fn complex_of(src: &str) -> (f64, f64) {
        match run(src).unwrap() {
            Value::Complex { re, im } => (as_num(*re), as_num(*im)),
            Value::Num(n) => (n, 0.0),
            other => panic!("expected complex, got {other:?} for {src:?}"),
        }
    }

    #[test]
    fn complex_emerges_from_i_and_displays() {
        // The type emerges from `math::i` + the existing operators (no complex literal).
        assert_eq!(display_of("2 + 3*math::i"), "2 + 3i");
        assert_eq!(display_of("2 - 3*math::i"), "2 - 3i");
        assert_eq!(display_of("3*math::i"), "3i");
        assert_eq!(display_of("-1*math::i"), "-1i");
        assert_eq!(display_of("0*math::i"), "0"); // 0 + 0i collapses to 0
                                                  // `j` is an alias for `i` (electrical-engineering convention).
        assert_eq!(complex_of("math::j"), (0.0, 1.0));
        // a user variable `i` still shadows the constant (vars win over the math fallback).
        assert_eq!(num("i = 5; i"), 5.0);
    }

    #[test]
    fn complex_arithmetic_folds() {
        // (2+3i)(1+1i) = -1 + 5i ; (2+3i)/(1+1i) = 2.5 + 0.5i
        assert_eq!(complex_of("(2 + 3*math::i) * (1 + 1*math::i)"), (-1.0, 5.0));
        assert_eq!(complex_of("(2 + 3*math::i) / (1 + 1*math::i)"), (2.5, 0.5));
        // real promotes to re + 0i
        assert_eq!(complex_of("5 + 2*math::i"), (5.0, 2.0));
        // integer power = repeated multiply: (1+i)^2 = 2i, (1+i)^3 = -2 + 2i
        assert_eq!(complex_of("(1 + math::i) ^ 2"), (0.0, 2.0));
        assert_eq!(complex_of("(1 + math::i) ^ 3"), (-2.0, 2.0));
        // exact (re, im) equality
        assert!(boolean("(2 + 3*math::i) == (2 + 3*math::i)"));
        assert!(boolean("(2 + 3*math::i) != (2 + 4*math::i)"));
    }

    #[test]
    fn complex_has_no_ordering() {
        // ℂ is not totally ordered — `<` etc. are a type error, like comparing one number to none.
        assert!(run("(1 + math::i) < (2 + math::i)").is_err());
        assert!(run("math::i > 0").is_err());
    }

    #[test]
    fn euler_identity_and_magnitude() {
        // e^{iπ} = -1 (Euler). Imag part is sin(π) ≈ 0.
        let (re, im) = complex_of("math::exp(math::i * math::pi)");
        assert!((re + 1.0).abs() < 1e-12, "re = {re}");
        assert!(im.abs() < 1e-12, "im = {im}");
        // |3 + 4i| = 5 ; arg(i) = π/2 ; conj(2+3i) = 2-3i ; re/im selectors
        assert_eq!(num("math::abs(3 + 4*math::i)"), 5.0);
        assert!((num("math::arg(math::i)") - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(complex_of("math::conj(2 + 3*math::i)"), (2.0, -3.0));
        assert_eq!(num("math::re(2 + 3*math::i)"), 2.0);
        assert_eq!(num("math::im(2 + 3*math::i)"), 3.0);
        // principal square root: sqrt(-1 + 0i) = i, sqrt(2i) = 1 + i
        assert_eq!(complex_of("math::sqrt(-1 + 0*math::i)"), (0.0, 1.0));
        let (sr, si) = complex_of("math::sqrt(2*math::i)");
        assert!(
            (sr - 1.0).abs() < 1e-12 && (si - 1.0).abs() < 1e-12,
            "sqrt(2i) = {sr}+{si}i"
        );
        // real sqrt stays IEEE: sqrt(-1.0) is NaN (no auto-promotion to complex)
        assert!(num("math::sqrt(-1)").is_nan());
    }

    #[test]
    fn exp_is_the_function_not_the_distribution() {
        // `exp` is now the exponential FUNCTION (math); the distribution is `rand::exponential`.
        assert!((num("math::exp(1)") - std::f64::consts::E).abs() < 1e-12);
        assert_eq!(num("math::exp(0)"), 1.0);
        // the distribution kept its semantics under the new name
        let m = moments_of("X ~ rand::exponential(2); X", 200_000, 1);
        assert!((m.mean - 0.5).abs() < 0.02, "mean = {}", m.mean);
    }

    #[test]
    fn normal_complex_is_a_cscg() {
        // Circularly-symmetric: E|z|² = σ² (total power), E re = E im = 0.
        let power = run_num("z ~ rand::normal_complex(2); E(math::re(z)^2 + math::im(z)^2)");
        assert!((power - 4.0).abs() < 0.1, "E|z|^2 = {power}");
        let mre = run_num("z ~ rand::normal_complex(2); E(math::re(z))");
        assert!(mre.abs() < 0.05, "E re = {mre}");
        // drawn with ~[n] like any distribution: an array of independent complex RVs
        let s = run_num("zs ~[3] rand::normal_complex(1); E(vec::normsq(zs))");
        assert!((s - 3.0).abs() < 0.1, "E normsq = {s}");
    }

    #[test]
    fn vec_consistency_over_complex() {
        // magnitude-based ops return a REAL: normsq(z) = Σ|zᵢ|², norm, mse.
        assert_eq!(num("vec::normsq([3 + 4*math::i])"), 25.0);
        assert_eq!(num("vec::norm([3 + 4*math::i])"), 5.0);
        assert_eq!(num("vec::mse([1 + 1*math::i], [1 + 0*math::i])"), 1.0); // |i|² = 1
                                                                            // sum/mean lift component-wise (stay complex)
        assert_eq!(
            complex_of("vec::sum([1 + 1*math::i, 2 + 3*math::i])"),
            (3.0, 4.0)
        );
        // dot stays bilinear (no conjugation): [i]·[i] = i·i = -1
        assert_eq!(complex_of("vec::dot([math::i], [math::i])"), (-1.0, 0.0));
        // vdot is Hermitian (conjugates the first arg): conj(i)·i = (-i)(i) = 1
        assert_eq!(complex_of("vec::vdot([math::i], [math::i])"), (1.0, 0.0));
        // outer product builds a matrix; adjoint = conjugate transpose
        assert_eq!(complex_of("vec::outer([math::i], [1])[0][0]"), (0.0, 1.0));
        assert_eq!(complex_of("vec::adjoint([[math::i]])[0][0]"), (0.0, -1.0));
        // max/min are a deliberate type error on ℂ (no ordering)
        assert!(run("vec::max([1 + math::i, 2 + math::i])").is_err());
    }

    #[test]
    fn am_vs_fm_complex_fm_wins() {
        // The headline payoff: FM recovers the message markedly cleaner than AM under identical
        // circularly-symmetric static. (The complex variant of examples/am_vs_fm.noise.)
        let src = "
            engine::set_max_samples(8000);
            am_modulate(m)      = 1 + m;
            fm_modulate(m, dev) = math::exp(math::i*dev*m);
            am_demodulate(z)      = math::abs(z) - 1;
            fm_demodulate(z, dev) = math::arg(z) / dev;
            dev = 3; sigma = 0.3;
            msg = signal::sample(0.3*signal::sine(3), 64);
            static ~[Len(msg)] rand::normal_complex(sigma);
            am_err = E(vec::mse(am_demodulate(am_modulate(msg)      + static),      msg));
            fm_err = E(vec::mse(fm_demodulate(fm_modulate(msg, dev) + static, dev), msg));
            am_err / fm_err";
        let ratio = run_num(src);
        assert!(ratio > 3.0, "FM should be several× cleaner, got {ratio}x");
    }

    // ===================== % operator + floor/ceil (PLAN-COMPLEX §8) =====================

    #[test]
    fn floored_modulo_takes_sign_of_divisor() {
        // floored: result has the sign of `b`, so `x % n ∈ [0, n)` for n > 0 (clock arithmetic).
        assert_eq!(num("7 % 3"), 1.0);
        assert_eq!(num("-1 % 3"), 2.0); // NOT -1 (truncated) — floored
        assert_eq!(num("7 % -3"), -2.0); // sign of divisor
        assert_eq!(num("5.5 % 2"), 1.5);
        assert_eq!(num("13 % 12"), 1.0);
        // x % 0 = NaN (no panic)
        assert!(num("1 % 0").is_nan());
        // precedence: same level as `* /`, looser than `^`
        assert_eq!(num("1 + 7 % 3"), 2.0);
        assert_eq!(num("2 ^ 3 % 5"), 3.0); // (2^3) % 5 = 8 % 5
    }

    #[test]
    fn floor_ceil_ufuncs() {
        assert_eq!(num("math::floor(2.7)"), 2.0);
        assert_eq!(num("math::ceil(2.1)"), 3.0);
        assert_eq!(num("math::floor(-2.1)"), -3.0);
        assert_eq!(num("math::ceil(-2.9)"), -2.0);
        // map over arrays
        assert_eq!(display_of("math::floor([1.9, 2.1, 3.5])"), "[1, 2, 3]");
        // real-only: complex floor is a type error
        assert!(run("math::floor(1 + math::i)").is_err());
        // lifts over a random variable (interp and JIT agree)
        let m = run_num("X ~ rand::unif_int(0, 11); E(X % 4)");
        assert!((m - 1.5).abs() < 0.05, "E(X % 4) = {m}"); // {0,1,2,3} uniform-ish over 0..11
    }

    // ===================== comprehensions (PLAN-COMPLEX §8) =====================

    #[test]
    fn comprehensions_map_and_close_over_outer() {
        assert_eq!(display_of("[for x in 0..5 { x*x }]"), "[0, 1, 4, 9, 16]");
        // body closes over an outer variable (no closures needed)
        assert_eq!(
            display_of("a = 10; [for x in 0..3 { a + x }]"),
            "[10, 11, 12]"
        );
        // a pure 1-to-1 map: result length always equals the iterable's
        assert_eq!(num("Len([for x in 0..7 { x }])"), 7.0);
        // empty array literal still parses (not a comprehension)
        assert_eq!(display_of("[]"), "[]");
        // a multi-statement body block works (and leaks, like every Noise block)
        assert_eq!(
            display_of("[for x in 0..3 { y = x + 1; y * y }]"),
            "[1, 4, 9]"
        );
    }

    // ===================== vec::outer / vec::categorical (PLAN-COMPLEX §8-9) =====================

    #[test]
    fn outer_product_and_categorical() {
        // outer(a, b)[i][j] = a_i * b_j
        assert_eq!(
            display_of("vec::outer([1, 2], [10, 20, 30])"),
            "[[10, 20, 30], [20, 40, 60]]"
        );
        // categorical samples an index proportional to the weights
        let p = run_num("y ~ rand::categorical([0, 0, 1, 0]); E(y)"); // all mass on index 2
        assert!((p - 2.0).abs() < 1e-9, "E(categorical) = {p}");
        let q = run_num("y ~ rand::categorical([3, 1]); E((y == 0))"); // P(index 0) = 3/4
        assert!((q - 0.75).abs() < 3e-3, "P(y==0) = {q}");
    }

    #[test]
    fn gcd_and_modpow_builtins() {
        // gcd (Euclid), defined via absolute values; gcd(0, 0) = 0.
        assert_eq!(num("math::gcd(48, 18)"), 6.0);
        assert_eq!(num("math::gcd(0, 0)"), 0.0);
        assert_eq!(num("math::gcd(-12, 8)"), 4.0);
        assert_eq!(num("math::gcd(17, 5)"), 1.0); // coprime
                                                  // modpow: exact even when base^exp would overflow 2^53.
        assert_eq!(num("math::modpow(7, 4, 15)"), 1.0); // 7^4 = 2401 ≡ 1
        assert_eq!(num("math::modpow(2, 10, 1000)"), 24.0); // 1024 mod 1000
        assert_eq!(num("math::modpow(7, 100, 13)"), 9.0); // 7^100 is astronomically large
        assert_eq!(num("math::modpow(2, 0, 15)"), 1.0); // base^0 = 1
        assert_eq!(num("math::modpow(2, 5, 1)"), 0.0); // anything mod 1 = 0
                                                       // exactness: `2^64 % 13` would lose precision via float `^`; modpow is exact (= 3).
        assert_eq!(num("math::modpow(2, 64, 13)"), 3.0);
        // domain errors: non-integer, negative exponent, non-positive modulus, RV argument
        assert!(run("math::gcd(1.5, 2)").is_err());
        assert!(run("math::modpow(2, -1, 7)").is_err());
        assert!(run("math::modpow(2, 3, 0)").is_err());
        assert!(run("X ~ rand::unif_int(1, 5); math::gcd(X, 10)").is_err());
    }

    #[test]
    fn shor_factors_via_quantum_period_finding() {
        // The full algorithm (examples/shor_period.noise): `shor(N)` returns N's two factors. The
        // quantum step `period(a, N, Q)` reads the period of a^x mod N off the interference comb (its
        // number of spikes), and the factors fall out of gcd(a^(r/2) +- 1, N).
        let lib = "
            onehot(v, width) = [for w in 0..width { if v == w { 1 } else { 0 } }];
            period(a, N, Q) = {
                ks = 0..Q;
                fx = [for x in 0..Q { math::modpow(a, x, N) }];
                Psi = [for x in 0..Q { onehot(fx[x], N) }] / Q^0.5;
                QFT = math::exp(math::i * (-2*math::pi/Q) * vec::outer(ks, ks)) / Q^0.5;
                vec::count([for p in QFT @ Psi { vec::normsq(p) > 0.0001 }])
            };
            shor(N) = {
                Q = 1; for i in 0..16 { if Q < N { Q = Q * 2 } };
                p = 1; q = N;
                for a in 2..N {
                    if p == 1 {
                        if math::gcd(a, N) > 1 { p = math::gcd(a, N); q = N / p }
                        else {
                            r = period(a, N, Q);
                            if r % 2 == 0 { s = math::modpow(a, r/2, N); g = math::gcd(s - 1, N);
                                if g > 1 { if g < N { p = g; q = N / g } } }
                        }
                    }
                };
                [p, q]
            };";
        // the quantum subroutine reads period(7^x mod 15) = 4 off a clean 4-spike comb (4 | Q = 16)
        assert_eq!(run_num(&format!("{lib} period(7, 15, 16)")), 4.0);
        assert_eq!(run_num(&format!("{lib} period(4, 15, 16)")), 2.0); // base 4 -> period 2
                                                                       // shor(15) genuinely factors via the quantum period (a = 2 is coprime, period 4 | 16) -> [3, 5]
        assert_eq!(run_num(&format!("{lib} shor(15)[0]")), 3.0);
        assert_eq!(run_num(&format!("{lib} shor(15)[1]")), 5.0);
        // a few more composites factor correctly (product check)
        assert_eq!(run_num(&format!("{lib} shor(21)[0] * shor(21)[1]")), 21.0);
        assert_eq!(run_num(&format!("{lib} shor(35)[0] * shor(35)[1]")), 35.0);
    }

    // ===================== complex: deeper coverage =====================

    #[test]
    fn complex_subtraction_negation_and_mixed_real() {
        assert_eq!(complex_of("(3 + 4*math::i) - (1 + 2*math::i)"), (2.0, 2.0));
        // unary minus on a complex (constant and the imaginary unit)
        assert_eq!(complex_of("-(2 + 3*math::i)"), (-2.0, -3.0));
        assert_eq!(complex_of("-math::i"), (0.0, -1.0));
        // real ⊕ complex in both orders, promotion to re + 0i
        assert_eq!(complex_of("(2 + 3*math::i) - 5"), (-3.0, 3.0));
        assert_eq!(complex_of("5 - (2 + 3*math::i)"), (3.0, -3.0));
        assert_eq!(complex_of("2 * (2 + 3*math::i)"), (4.0, 6.0));
        assert_eq!(complex_of("(2 + 3*math::i) * 2"), (4.0, 6.0));
        // division by a purely imaginary number: 10 / (2i) = -5i
        assert_eq!(complex_of("10 / (2*math::i)"), (0.0, -5.0));
    }

    #[test]
    fn complex_power_edge_cases() {
        assert_eq!(complex_of("(2 + 3*math::i) ^ 0"), (1.0, 0.0)); // z^0 = 1
        assert_eq!(complex_of("(2 + 3*math::i) ^ 1"), (2.0, 3.0)); // z^1 = z
        assert_eq!(complex_of("(1 + math::i) ^ -1"), (0.5, -0.5)); // reciprocal
                                                                   // a general complex power is rejected (needs a constant integer exponent)
        assert!(run("(1 + math::i) ^ 1.5").is_err());
        assert!(run("(1 + math::i) ^ (1 + math::i)").is_err());
        assert!(run("2 ^ math::i").is_err());
        assert!(run("(1 + math::i) ^ 100000").is_err()); // exponent magnitude cap
    }

    #[test]
    fn complex_type_errors() {
        // ordering, modulo, logical, and bool/string mixing are all rejected on ℂ
        assert!(run("math::i < math::i").is_err());
        assert!(run("(1 + math::i) <= 2").is_err());
        assert!(run("math::i % 2").is_err());
        assert!(run("math::i && true").is_err());
        assert!(run("\"a\" + math::i").is_err());
        // a complex can't be an `if` condition (not a bool)
        assert!(run("if math::i { 1 } else { 0 }").is_err());
        // E/Var of a complex is deferred — a clear, actionable error
        assert!(run("z ~ rand::normal_complex(1); E(z)").is_err());
    }

    #[test]
    fn arg_is_quadrant_correct() {
        let pi = std::f64::consts::PI;
        assert!((num("math::arg(1 + 1*math::i)") - pi / 4.0).abs() < 1e-12);
        assert!((num("math::arg(-1 + 1*math::i)") - 3.0 * pi / 4.0).abs() < 1e-12);
        assert!((num("math::arg(-1 - 1*math::i)") + 3.0 * pi / 4.0).abs() < 1e-12);
        assert!((num("math::arg(1 - 1*math::i)") + pi / 4.0).abs() < 1e-12);
    }

    #[test]
    fn complex_ufuncs_map_over_arrays() {
        // exp / conj / re map elementwise over a complex array
        assert_eq!(
            display_of("math::re([1 + 2*math::i, 3 + 4*math::i])"),
            "[1, 3]"
        );
        assert_eq!(
            display_of("math::im([1 + 2*math::i, 3 + 4*math::i])"),
            "[2, 4]"
        );
        assert_eq!(
            complex_of("math::conj([1 + 2*math::i, 3 + 4*math::i])[1]"),
            (3.0, -4.0)
        );
        // abs over a complex array → real array
        assert_eq!(
            display_of("math::abs([3 + 4*math::i, 0 + 1*math::i])"),
            "[5, 1]"
        );
    }

    #[test]
    fn complex_matmul() {
        // vec @ vec is the bilinear dot over ℂ: (1+i)*1 + 2*i = 1 + 3i
        assert_eq!(complex_of("[1 + 1*math::i, 2] @ [1, math::i]"), (1.0, 3.0));
        // i·I (matrix @ vector): [[i,0],[0,i]] @ [1,1] = [i, i]
        let out = "M = [[math::i, 0], [0, math::i]]; (M @ [1, 1])[0]";
        assert_eq!(complex_of(out), (0.0, 1.0));
    }

    #[test]
    fn complex_random_variable_paths() {
        // |e^{iX}| = 1 for every lane (cos²+sin² = 1): exp lifts the imaginary RV via cos/sin.
        let m = run_num("X ~ rand::unif(0, 1); E(math::abs(math::exp(math::i * X)))");
        assert!((m - 1.0).abs() < 1e-9, "E|e^iX| = {m}");
        // exp of a *random real part* lifts too (PLAN-FINANCE F1): E[e^U] over U(0,1) = e - 1.
        let x = run_num("X ~ rand::unif(0,1); E(math::exp(X))");
        assert!(
            (x - (std::f64::consts::E - 1.0)).abs() < 0.02,
            "E e^U = {x}"
        );
        // sqrt over a real RV is IEEE `^ 0.5`: E[sqrt(U(0,4))] = (1/4)∫₀⁴ √x dx = 4/3
        let s = run_num("X ~ rand::unif(0, 4); E(math::sqrt(X))");
        assert!((s - 4.0 / 3.0).abs() < 0.02, "E sqrt = {s}");
        // arg over an RV phasor near -1 sits at ±π (left half-plane quadrant fix)
        let a = run_num("z ~ rand::normal_complex(0.02); E(math::abs(math::arg((0 - 1) + z)))");
        assert!(
            (a - std::f64::consts::PI).abs() < 0.05,
            "E|arg| near -1 = {a}"
        );
    }

    #[test]
    fn normal_complex_parameters_and_independence() {
        // sigma = 0 is a point mass at 0+0i.
        assert_eq!(run_num("z ~ rand::normal_complex(0); E(math::abs(z))"), 0.0);
        // sigma < 0 is rejected.
        assert!(run("z ~ rand::normal_complex(-1); z").is_err());
        // each channel is N(0, sigma/√2): Var(re) = sigma²/2 = 2 for sigma = 2.
        let v = run_num("z ~ rand::normal_complex(2); Var(math::re(z))");
        assert!((v - 2.0).abs() < 0.1, "Var(re) = {v}");
        // re and im are independent ⇒ E(re·im) ≈ 0.
        let cov = run_num("z ~ rand::normal_complex(2); E(math::re(z) * math::im(z))");
        assert!(cov.abs() < 0.05, "E(re·im) = {cov}");
    }

    #[test]
    fn vec_lifts_and_magnitudes_over_complex() {
        // linear ops lift component-wise (stay complex)
        assert_eq!(
            complex_of("vec::mean([2 + 2*math::i, 4 + 4*math::i])"),
            (3.0, 3.0)
        );
        assert_eq!(
            num("vec::transpose([[1 + 1*math::i, 2], [3, 4*math::i]])[0][1]"),
            3.0
        );
        // normalize a complex vector → unit magnitude
        assert!((num("vec::norm(vec::normalize([3 + 4*math::i]))") - 1.0).abs() < 1e-12);
        // dot is bilinear, vdot is Hermitian: dot([i,i],[i,i]) = -2, vdot = +2 (= normsq)
        assert_eq!(
            complex_of("vec::dot([math::i, math::i], [math::i, math::i])"),
            (-2.0, 0.0)
        );
        assert_eq!(
            complex_of("vec::vdot([math::i, math::i], [math::i, math::i])"),
            (2.0, 0.0)
        );
        // vdot(z, z) == normsq(z) (a real)
        assert_eq!(
            complex_of("vec::vdot([3 + 4*math::i], [3 + 4*math::i])"),
            (25.0, 0.0)
        );
        // on REAL vectors vdot coincides with dot
        assert_eq!(
            num("vec::vdot([1, 2, 3], [4, 5, 6])"),
            num("vec::dot([1, 2, 3], [4, 5, 6])")
        );
        // adjoint = conjugate transpose (rectangular)
        assert_eq!(
            complex_of("vec::adjoint([[1 + 1*math::i, 2 + 1*math::i, 3]])[1][0]"),
            (2.0, -1.0)
        );
    }

    #[test]
    fn complex_scoping_of_i_and_j() {
        // qualified `math::i` resolves with NO `use` (run_raw has no prelude)
        match run_raw("math::abs(3 + 4*math::i)").unwrap() {
            Value::Num(n) => assert_eq!(n, 5.0),
            other => panic!("got {other:?}"),
        }
        // a bare `i` needs `use math;` — out of scope without it
        assert!(run_raw("i").is_err());
    }

    // ===================== % / floor / ceil: deeper coverage =====================

    #[test]
    fn modulo_precedence_and_fractional() {
        // unary minus is looser than `%` here only via grouping; `-7 % 3` parses as `-(7 % 3)`? No:
        // prefix `-` binds tighter than `^`, hence tighter than `%` — so `-7 % 3 = (-7) % 3 = 2`.
        assert_eq!(num("-7 % 3"), 2.0);
        assert_eq!(num("10 % 2.5"), 0.0);
        assert_eq!(num("10.5 % 3"), 1.5);
        // chains left-to-right with `*` and `/`
        assert_eq!(num("13 % 12 * 2"), 2.0); // (13 % 12) * 2
                                             // propagates an estimate's error (E carries one)
        let m = run_num("X ~ rand::unif_int(0, 9); E(X) % 100");
        assert!((m - 4.5).abs() < 0.1, "E(X) % 100 = {m}");
    }

    #[test]
    fn floor_ceil_over_values_and_rvs() {
        assert_eq!(num("math::ceil(-2.9)"), -2.0);
        assert_eq!(display_of("math::ceil([1.1, 2.0, 2.9])"), "[2, 2, 3]");
        // E[floor(U(0,3))] = mean of {0,1,2} weighted by the unit intervals = 1.0
        let m = run_num("X ~ rand::unif(0, 3); E(math::floor(X))");
        assert!((m - 1.0).abs() < 0.05, "E floor = {m}");
        // ceil is real-only
        assert!(run("math::ceil(1 + math::i)").is_err());
    }

    // ===================== comprehensions: deeper coverage =====================

    #[test]
    fn comprehensions_nest_and_over_arrays() {
        // nested comprehension builds a matrix
        assert_eq!(
            display_of("[for i in 0..2 { [for j in 0..3 { i*j }] }]"),
            "[[0, 0, 0], [0, 1, 2]]"
        );
        // iterate a literal array (not just a range)
        assert_eq!(
            display_of("[for x in [10, 20, 30] { x + 1 }]"),
            "[11, 21, 31]"
        );
        // a body that maps each element through several ops
        assert_eq!(display_of("[for x in 0..4 { (2*x) % 3 }]"), "[0, 2, 1, 0]");
        // the loop variable leaks (Noise blocks don't scope) — last value survives
        assert_eq!(num("[for x in 0..3 { x }]; x"), 2.0);
    }

    #[test]
    fn comprehensions_build_random_variables() {
        // a comprehension over an array of RVs builds independent transformed nodes
        let m = run_num("ds ~[3] rand::unif(0, 1); E(vec::sum([for d in ds { 2*d }]))");
        assert!((m - 3.0).abs() < 0.05, "E sum = {m}"); // each 2·E[U]=1, three of them
    }

    #[test]
    fn comprehension_errors() {
        // iterating a non-array
        assert!(run("[for x in 5 { x }]").is_err());
        // an undrawn recipe in the body is rejected like anywhere else
        assert!(run("[for x in 0..3 { rand::unif(0, 1) }]").is_err());
    }

    #[test]
    fn continue_filters_in_comprehensions_and_loops() {
        // filter form: `if bad { continue }; keep` — the imperative idiom
        assert_eq!(
            display_of("[for x in 0..10 { if x % 2 != 0 { continue }; x }]"),
            "[0, 2, 4, 6, 8]"
        );
        // else-continue form
        assert_eq!(
            display_of("[for x in 0..10 { if x % 3 == 0 { x } else { continue } }]"),
            "[0, 3, 6, 9]"
        );
        // all elements skipped → empty array
        assert_eq!(display_of("[for x in 0..5 { continue }]"), "[]");
        // continue in a plain `for` skips that iteration's side effects (here, accumulation)
        assert_eq!(
            num("acc = 0; for x in 0..10 { if x % 2 != 0 { continue }; acc = acc + x }; acc"),
            20.0
        );
        // it propagates up through a nested block
        assert_eq!(
            display_of("[for x in 0..6 { { if x % 2 == 1 { continue } }; x*10 }]"),
            "[0, 20, 40]"
        );
        // a RANDOM continue is rejected (the array length is fixed at build time) — clean error
        assert!(run("B ~ rand::bernoulli(0.5); [for x in 0..3 { if B { continue }; x }]").is_err());
        // misused as a data value → a plain type error (not a panic)
        assert!(run("1 + continue").is_err());
    }

    // ===================== categorical / outer: deeper coverage =====================

    #[test]
    fn categorical_proportions_and_errors() {
        // single-weight is a point mass at index 0
        assert_eq!(run_num("y ~ rand::categorical([5]); E(y)"), 0.0);
        // proportional sampling: weights [1,1,2] → P(index 2) = 1/2
        let p = run_num("y ~ rand::categorical([1, 1, 2]); E((y == 2))");
        assert!((p - 0.5).abs() < 3e-3, "P(y==2) = {p}");
        // domain errors
        assert!(run("y ~ rand::categorical([])").is_err()); // empty
        assert!(run("y ~ rand::categorical([1, -1])").is_err()); // negative weight
        assert!(run("y ~ rand::categorical([0, 0])").is_err()); // zero total
    }

    #[test]
    fn outer_product_shapes() {
        assert_eq!(display_of("vec::outer([1, 2], [3, 4])"), "[[3, 4], [6, 8]]");
        // rectangular: len(a) rows, len(b) cols (`Len` is the always-on builtin, not `vec::`)
        assert_eq!(num("Len(vec::outer([1, 2, 3], [4, 5]))"), 3.0);
        assert_eq!(num("Len(vec::outer([1, 2, 3], [4, 5])[0])"), 2.0);
    }

    #[test]
    fn exponential_distribution_renamed_cleanly() {
        // `exp` is the function (e^2), NOT a distribution; the old name is gone.
        assert!((num("math::exp(2)") - (2.0f64).exp()).abs() < 1e-12);
        assert!(run("X ~ exp_int(2); X").is_err()); // old `exp_int` name removed
                                                    // the distribution lives on as `rand::exponential` with the same tail.
        let p = run_num("X ~ rand::exponential(2); P(X > 1)");
        assert!((p - (-2.0f64).exp()).abs() < 3e-3, "P(X>1) = {p}");
    }

    // ===================== mixing complex with everything else (robustness) =====================

    #[test]
    fn complex_broadcasts_with_arrays() {
        // complex scalar ⊕ real array (both orders), real array ⊕ complex array
        assert_eq!(
            display_of("(2 + 3*math::i) + [1, 2, 3]"),
            "[3 + 3i, 4 + 3i, 5 + 3i]"
        );
        assert_eq!(
            display_of("[1, 2] + [math::i, 2*math::i]"),
            "[1 + 1i, 2 + 2i]"
        );
        // complex array scaled by a real scalar (result may mix complex & real elements)
        assert_eq!(display_of("[1 + 1*math::i, 2] * 2"), "[2 + 2i, 4]");
        // length mismatch is a clean error
        assert!(run("[math::i] + [1, 2]").is_err());
        // real matrix @ complex vector: [[1,2],[3,4]] @ [i,i] = [3i, 7i]
        assert_eq!(
            complex_of("([[1, 2], [3, 4]] @ [math::i, math::i])[0]"),
            (0.0, 3.0)
        );
        assert_eq!(
            complex_of("([[1, 2], [3, 4]] @ [math::i, math::i])[1]"),
            (0.0, 7.0)
        );
    }

    #[test]
    fn complex_real_equality() {
        // a complex with zero imaginary part equals the matching real, and differs otherwise
        assert!(boolean("(2 + 0*math::i) == 2"));
        assert!(boolean("2 == (2 + 0*math::i)"));
        assert!(!boolean("2 == (2 + 3*math::i)"));
        assert!(boolean("(2 + 3*math::i) != 2"));
    }

    #[test]
    fn complex_mixing_errors_cleanly() {
        // a signal + complex is now a COMPLEX SIGNAL (PLAN-SIGNALS §1.3) — the channels stay lazy
        let z = run("signal::sine(3) + math::i").unwrap();
        assert_eq!(z.type_name(), "complex");
        // an UNDRAWN noise generator is still rejected (the message now points at `~`)
        let e = run("signal::noise_white(1) + math::i")
            .unwrap_err()
            .to_string();
        assert!(e.contains("undrawn distribution"), "got: {e}");
        assert!(run("math::i + true").is_err());
        assert!(run("math::i + rand::unif(0, 1)").is_err());
        // complex can't be an array index, a counted event, or an ordered extremum
        assert!(run("[10, 20, 30][math::i]").is_err());
        assert!(run("vec::count([math::i])").is_err());
        assert!(run("vec::max([math::i, 2*math::i])").is_err());
        // a lifted `if` (random condition) with complex branches is unsupported — a clean error
        assert!(run("B ~ rand::bernoulli(0.5); if B { math::i } else { 0*math::i }").is_err());
        // a random gather over complex elements is likewise rejected
        assert!(run("k ~ rand::unif_int(0, 1); [math::i, 2*math::i][k]").is_err());
    }

    #[test]
    fn complex_degenerate_values_dont_panic() {
        // division by 0 + 0i yields NaN channels (no panic)
        let (re, im) = complex_of("(1 + math::i) / (0*math::i)");
        assert!(re.is_nan() && im.is_nan(), "got {re}+{im}i");
        // abs/arg of 0 + 0i are well-defined (0 and 0)
        assert_eq!(num("math::abs(0*math::i)"), 0.0);
        assert_eq!(num("math::arg(0*math::i)"), 0.0);
        // real ufuncs of plain reals (the real branch)
        assert_eq!(num("math::conj(5)"), 5.0);
        assert_eq!(num("math::re(5)"), 5.0);
        assert_eq!(num("math::im(5)"), 0.0);
        assert_eq!(num("math::abs(-7)"), 7.0);
    }

    #[test]
    fn estimates_flow_through_complex() {
        // a `P`/`E` estimate (carries a standard error) promotes into a complex value and through
        // the magnitude ufuncs, floor, and exp — none of which should choke on the `Est` channel.
        assert!(
            (run_num("D ~ rand::unif_int(1,6); math::abs(P(D > 3) + 0*math::i)") - 0.5).abs()
                < 0.01
        );
        // 6.5·P(D>3) ≈ 3.25 (off the integer boundary, so floor is a stable 3 despite MC noise)
        assert_eq!(
            run_num("D ~ rand::unif_int(1,6); math::floor(6.5 * P(D > 3))"),
            3.0
        );
        assert_eq!(
            run_num("D ~ rand::unif_int(1,6); math::exp(0 * P(D > 3))"),
            1.0
        );
        // sum over a mix of real and complex elements stays complex
        assert_eq!(
            complex_of("vec::sum([1, math::i, 2 + 2*math::i])"),
            (3.0, 3.0)
        );
    }

    // ===================== identity fold: correctness guards =====================

    #[test]
    fn identity_fold_preserves_values_and_ieee() {
        // `1*X → X` keeps the SAME draw (structural sharing), so `X - 1*X` is exactly 0.
        let z = run("X ~ rand::unif(0,1); X - 1*X").unwrap();
        // every lane is 0 → mean exactly 0
        assert_eq!(run_num("X ~ rand::unif(0,1); E(X - 1*X)"), 0.0);
        assert!(matches!(z, Value::Dist(_)));
        // `0*X → 0` and `X + 0 → X`
        assert_eq!(run_num("X ~ rand::unif(0,1); E(0 * X)"), 0.0);
        assert!((run_num("X ~ rand::unif(0,1); E(X + 0)") - 0.5).abs() < 0.01);
        // the fold does NOT fire for constants, so IEEE `0 * inf = NaN` is preserved.
        assert!(num("0 * (1 / 0)").is_nan());
        // the fold does NOT fire for a bool RV: `0 * event` is still a type error, not 0.
        assert!(run("B ~ rand::bernoulli(0.5); 0 * B").is_err());
    }

    // ===================== variable introspection (describe/corr/explain) =====================

    use crate::introspect::{Payload, Summary, View};

    /// Run a program whose last expression is an introspection and return its summary.
    fn summary_of(src: &str) -> std::rc::Rc<Summary> {
        match run(src).unwrap() {
            Value::Summary(s) => s,
            other => panic!("expected a summary, got {other:?} for {src:?}"),
        }
    }
    fn one(src: &str) -> crate::introspect::Dist1 {
        match &summary_of(src).payload {
            Payload::One(d) => d.clone(),
            other => panic!("expected a one-variable summary, got {other:?}"),
        }
    }

    /// `describe(X)` recovers the marginal distribution: a uniform's mean/median are ~0.5, its
    /// quantiles span the support, and the histogram covers `[0, 1]`.
    #[test]
    fn describe_recovers_the_marginal_distribution() {
        let d = one("X ~ rand::unif(0, 1); describe(X)");
        assert!((d.mean - 0.5).abs() < 0.01, "mean {}", d.mean);
        assert!((d.q50 - 0.5).abs() < 0.02, "median {}", d.q50);
        assert!(d.min >= 0.0 && d.max <= 1.0);
        assert!(d.hist.lo >= 0.0 && d.hist.hi <= 1.0);
    }

    /// The headline: `describe(bias | data)` is the Bayesian posterior. A flat prior after 7 of 10
    /// heads has posterior mean (h+1)/(n+2) = 8/12 ≈ 0.6667 — read straight off the conditioned
    /// summary, no separate machinery.
    #[test]
    fn describe_of_a_conditioned_value_is_the_posterior() {
        let d = one("use rand; use vec;
             bias ~ unif(0,1); flips ~[10] bernoulli(bias); heads = count(flips);
             describe(bias | heads == 7)");
        assert!((d.mean - 0.6667).abs() < 0.02, "posterior mean {}", d.mean);
        // the posterior is tighter than the flat prior (sd 1/√12 ≈ 0.289).
        assert!(d.sd < 0.2, "posterior sd should be tight, got {}", d.sd);
    }

    /// `corr` is a *joint* (paired) statistic: two independent draws are ~uncorrelated, while a
    /// variable correlates strongly with a sum it appears in. (Separate passes would mis-pair lanes
    /// and break this — the same joint-sampling requirement as conditioning.)
    #[test]
    fn corr_is_a_correct_joint_statistic() {
        let indep =
            match &summary_of("A ~ rand::unif(0,1); B ~ rand::unif(0,1); corr(A, B)").payload {
                Payload::Two(d) => d.corr,
                _ => panic!("expected a two-variable summary"),
            };
        assert!(indep.abs() < 0.02, "independent corr ~0, got {indep}");
        let shared =
            match &summary_of("A ~ rand::unif(0,1); B ~ rand::unif(0,1); corr(A, A + B)").payload {
                Payload::Two(d) => d.corr,
                _ => panic!("expected a two-variable summary"),
            };
        // corr(A, A+B) = sd_A / sd_{A+B} = 1/√2 ≈ 0.707 for two iid uniforms.
        assert!(
            (shared - 0.707).abs() < 0.03,
            "corr(A, A+B) ≈ 0.707, got {shared}"
        );
    }

    /// `explain(Y)` ranks the named upstream variables that drive `Y`. For the posterior predictive
    /// `next | data`, the only upstream model variable is `bias`, so it is the top (and only) driver.
    #[test]
    fn explain_finds_the_upstream_driver() {
        let s = summary_of(
            "use rand; use vec;
             bias ~ unif(0,1); flips ~[10] bernoulli(bias); next ~ bernoulli(bias);
             heads = count(flips);
             explain(next | heads == 7)",
        );
        assert_eq!(s.view, View::Explain);
        match &s.payload {
            Payload::Explain(e) => {
                assert_eq!(e.drivers.first().map(|d| d.name.as_str()), Some("bias"));
                assert!(
                    e.drivers[0].corr > 0.1,
                    "bias should positively drive next: {}",
                    e.drivers[0].corr
                );
            }
            other => panic!("expected an explain summary, got {other:?}"),
        }
    }

    /// The sidecar mechanism the playground rests on: a program's scope **persists across `run`
    /// calls**, so a *separate* follow-up `run("describe(...)")` resolves against the live variables —
    /// inspection without editing the source. Also checks `bindings()` reports the live RVs.
    #[test]
    fn introspection_resolves_against_a_retained_scope() {
        let mut eng = Engine::new();
        eng.run("use rand; use vec; bias ~ unif(0,1); flips ~[10] bernoulli(bias); heads = count(flips);")
            .unwrap();
        // bindings() surfaces the live variables (name + kind) for the picker.
        let binds = eng.bindings();
        assert!(binds
            .iter()
            .any(|(n, k)| n == "bias" && *k == "dist<number>"));
        assert!(binds.iter().any(|(n, _)| n == "heads"));
        // a follow-up run sees `bias`/`heads` — no re-declaration — and yields the posterior.
        match eng.run("describe(bias | heads == 7)").unwrap() {
            Value::Summary(s) => match &s.payload {
                Payload::One(d) => assert!((d.mean - 0.6667).abs() < 0.02, "posterior {}", d.mean),
                other => panic!("expected a one-variable summary, got {other:?}"),
            },
            other => panic!("expected a summary, got {other:?}"),
        }
    }

    /// A condition that never holds is a clean spanned error, not a silent NaN summary.
    #[test]
    fn describe_of_an_impossible_condition_errors() {
        let err = run("X ~ rand::unif(0,1); describe(X | X > 2)").unwrap_err();
        assert!(format!("{err}").contains("never occurred"), "got: {err}");
    }

    /// `describe` is polymorphic: a scalar estimate → a value+CI card (so `pi = 4*P(…)` is
    /// inspectable), and an array → a per-cell grid (vector here).
    #[test]
    fn describe_is_polymorphic_over_value_kinds() {
        // a scalar estimate → value card carrying its standard error
        match &summary_of(
            "X ~ rand::unif(-1,1); Y ~ rand::unif(-1,1); pi = 4*P(X^2+Y^2<1); describe(pi)",
        )
        .payload
        {
            Payload::Value(v) => {
                assert!((v.val - std::f64::consts::PI).abs() < 0.02, "pi {}", v.val);
                assert!(v.se > 0.0, "an estimate should carry uncertainty");
            }
            other => panic!("expected a value card, got {other:?}"),
        }
        // a vector → a 1×n grid of per-element moments
        match &summary_of("xs ~[6] rand::bernoulli(0.7); describe(xs)").payload {
            Payload::Grid(g) => {
                assert_eq!((g.rows, g.cols), (1, 6));
                assert!(
                    g.mean.iter().all(|&m| (m - 0.7).abs() < 0.02),
                    "per-element P(true)≈0.7: {:?}",
                    g.mean
                );
            }
            other => panic!("expected a grid, got {other:?}"),
        }
    }

    /// `corr(vec)` is the element×element correlation matrix: iid draws give an identity (1 on the
    /// diagonal, ~0 off it).
    #[test]
    fn corr_of_a_vector_is_the_correlation_matrix() {
        match &summary_of("xs ~[4] rand::normal(0,1); corr(xs)").payload {
            Payload::CorrMatrix(c) => {
                assert_eq!(c.n, 4);
                for i in 0..4 {
                    assert!((c.corr[i * 4 + i] - 1.0).abs() < 1e-6, "diagonal must be 1");
                    for j in 0..4 {
                        if i != j {
                            assert!(
                                c.corr[i * 4 + j].abs() < 0.02,
                                "iid off-diagonal ~0: {}",
                                c.corr[i * 4 + j]
                            );
                        }
                    }
                }
            }
            other => panic!("expected a correlation matrix, got {other:?}"),
        }
    }

    /// `plot::*` captures charts (like `Print`) and yields unit — a program can ask to *see* several
    /// things, and the host (CLI/playground) drains and renders them.
    #[test]
    fn plot_calls_capture_charts_and_yield_unit() {
        let mut eng = Engine::new();
        let last = eng
            .run("use rand; X ~ normal(0,1); Y ~ normal(0,1); plot::histogram(X); plot::scatter(X, Y)")
            .unwrap();
        assert!(
            matches!(last, Value::Unit),
            "a plot is a statement, not a value"
        );
        let items = eng.take_output();
        let plots: Vec<_> = items
            .iter()
            .filter_map(|o| match &o.output {
                crate::Output::Plot(s) => Some(s),
                crate::Output::Text(_)
                | crate::Output::Note { .. }
                | crate::Output::Input { .. } => None,
            })
            .collect();
        assert_eq!(plots.len(), 2);
        assert!(
            matches!(plots[0].payload, Payload::One(_)),
            "histogram → Dist1"
        );
        assert!(
            matches!(plots[1].payload, Payload::Two(_)),
            "scatter → Dist2"
        );
        // drained: a second take is empty
        assert!(eng.take_output().is_empty());
    }

    /// `describe`/`heatmap` of a 2-D draw → a rectangular grid (rows × cols), one mean per cell.
    #[test]
    fn describe_of_a_matrix_is_a_grid() {
        match &summary_of("M ~[3, 4] rand::normal(0,1); describe(M)").payload {
            Payload::Grid(g) => {
                assert_eq!((g.rows, g.cols), (3, 4));
                assert_eq!(g.mean.len(), 12);
                assert!(!g.is_series(), "a matrix is not a series");
            }
            other => panic!("expected a grid, got {other:?}"),
        }
    }

    // ===================== fan charts (plot::fan of a path) =====================

    /// Run a program (with the module prelude) and return the fan chart it plotted.
    fn fan_plot_of(src: &str) -> std::rc::Rc<Summary> {
        let mut eng = Engine::new();
        eng.run(&with_prelude(src)).unwrap();
        for o in eng.take_output() {
            if let crate::Output::Plot(s) = o.output {
                if matches!(s.payload, Payload::Fan(_)) {
                    return s;
                }
            }
        }
        panic!("expected a fan plot in the output stream for {src:?}");
    }

    /// `plot::fan(path)` of a driftless random walk shows the textbook cone: at EVERY index the
    /// bands are strictly ordered (q05 < q25 < q50 < q75 < q95 — one joint pass, so they never
    /// cross), the median hugs the zero drift line, and the band widens with t (sd = √(t+1)).
    #[test]
    fn fan_of_a_random_walk_is_a_widening_cone() {
        let s = fan_plot_of("steps ~[16] normal(0, 1); path = cumsum(steps); plot::fan(path)");
        let c = match &s.payload {
            Payload::Fan(c) => c,
            other => panic!("expected a fan chart, got {other:?}"),
        };
        assert_eq!(c.cols, 16);
        for t in 0..c.cols {
            assert!(
                c.q05[t] < c.q25[t]
                    && c.q25[t] < c.q50[t]
                    && c.q50[t] < c.q75[t]
                    && c.q75[t] < c.q95[t],
                "bands must be strictly ordered at index {t}"
            );
        }
        assert!(
            c.q50[15].abs() < 0.1,
            "driftless median ≈ 0, got {}",
            c.q50[15]
        );
        let (w0, w1) = (c.q95[0] - c.q05[0], c.q95[15] - c.q05[15]);
        assert!(
            w1 > 2.0 * w0,
            "the cone must widen with t: width {w0} → {w1}"
        );
    }

    /// A plain numeric array is the degenerate fan: every band (and the mean) equals the values —
    /// accepted rather than erroring, so `plot::fan` works on data too.
    #[test]
    fn fan_of_a_deterministic_array_is_the_degenerate_fan() {
        let s = fan_plot_of("plot::fan([1, 2, 3])");
        let c = match &s.payload {
            Payload::Fan(c) => c,
            other => panic!("expected a fan chart, got {other:?}"),
        };
        assert_eq!(c.cols, 3);
        for (i, want) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            for band in [&c.q05, &c.q25, &c.q50, &c.q75, &c.q95, &c.mean] {
                assert_eq!(band[i], want, "a constant's every quantile is itself");
            }
        }
        let shown = s.to_string();
        // An array literal has no source name, so the card falls back to the generic `value`.
        assert_eq!(
            shown,
            "fan(value): 3 steps n=200000 final q05=3 med=3 q95=3"
        );
    }

    /// A scalar or a matrix has no single index to fan over → a friendly spanned error.
    #[test]
    fn fan_rejects_scalars_and_matrices() {
        let scalar = run("X ~ rand::normal(0,1); plot::fan(X)")
            .unwrap_err()
            .to_string();
        assert!(scalar.contains("plot::fan wants a vector"), "got: {scalar}");
        let matrix = run("M ~[2, 2] rand::normal(0,1); plot::fan(M)")
            .unwrap_err()
            .to_string();
        assert!(matrix.contains("plot::fan wants a vector"), "got: {matrix}");
        let num = run("plot::fan(3)").unwrap_err().to_string();
        assert!(num.contains("plot::fan wants a vector"), "got: {num}");
    }

    /// End-to-end: a GBM-style program with `plot::fan(path)` runs, captures an `Output::Plot`, and
    /// that summary emits the three layerable Flint specs the cone is made of — the whole contract
    /// between the engine and any renderer.
    #[test]
    fn plot_fan_of_a_gbm_path_emits_the_layered_cone_spec() {
        let s = fan_plot_of(
            "zs ~[8] normal(0, 1);
             path = 100 * exp(cumsum(0.01 * zs - 0.005));
             plot::fan(path)",
        );
        let plot = crate::flint::to_flint(&s);
        assert_eq!(plot.title, "fan(path)");
        assert_eq!(plot.charts.len(), 3, "two bands + a median line");
        assert!(
            plot.text.starts_with("fan(path): 8 steps"),
            "text fallback: {}",
            plot.text
        );
        // The median line's y is the source variable, so the merged chart's y axis is titled `path`.
        assert_eq!(plot.charts[2]["chart_spec"]["encodings"]["y"], "path");
    }

    // ===================== `stats::*` — the raw-data twins of the charts =====================
    //
    // The point of these builtins is not that they compute *something like* what a chart shows —
    // it is that they compute *exactly* it. Each test below runs a `plot::` call and its `stats::`
    // twin against the same engine, and asserts the numbers are bit-identical. If someone ever
    // reimplements one of the two, these fail.

    /// The numbers in an array value.
    fn nums(v: &Value) -> Vec<f64> {
        match v {
            Value::Array(xs) => xs
                .iter()
                .map(|x| match x {
                    Value::Num(n) => *n,
                    other => panic!("expected a number, got {other:?}"),
                })
                .collect(),
            other => panic!("expected an array, got {other:?}"),
        }
    }

    /// The rows of a matrix value.
    fn rows(v: &Value) -> Vec<Vec<f64>> {
        match v {
            Value::Array(rs) => rs.iter().map(nums).collect(),
            other => panic!("expected a matrix, got {other:?}"),
        }
    }

    /// Run a program, then evaluate `expr` against its retained scope (the same trick the playground
    /// sidecar uses) — so a `plot::` call and its `stats::` twin see the identical graph and roots.
    fn engine_after(src: &str) -> Engine {
        let mut eng = Engine::new();
        eng.run(&with_prelude(src)).unwrap();
        eng
    }

    fn plot_payload(eng: &mut Engine) -> Payload {
        for o in eng.take_output() {
            if let crate::Output::Plot(s) = &o.output {
                return s.payload.clone();
            }
        }
        panic!("expected a plot in the output stream");
    }

    /// `stats::histogram(x)` returns the bins `plot::hist(x)` draws — the same binning, not a second.
    #[test]
    fn stats_histogram_is_the_data_behind_plot_hist() {
        let mut eng = engine_after("X ~ normal(100, 15); plot::hist(X)");
        let d = match plot_payload(&mut eng) {
            Payload::One(d) => d,
            other => panic!("expected a distribution, got {other:?}"),
        };
        let got = rows(&eng.run("stats::histogram(X)").unwrap());
        assert_eq!(got.len(), 2, "[midpoints, counts]");
        assert_eq!(
            got[1],
            d.hist.bins.iter().map(|&c| c as f64).collect::<Vec<_>>()
        );
        assert_eq!(got[0], d.hist.midpoints(false));
        assert_eq!(got[0].len(), 30, "the default bin count is the chart's");
        // The counts are a partition of the sample: nothing binned twice, nothing dropped.
        assert_eq!(got[1].iter().sum::<f64>(), d.n as f64);
    }

    /// The bin count is a knob, and an *event* ignores it: `false`/`true` is all an event can be,
    /// and its buckets are the points 0 and 1, not the halves of `[0, 1]`.
    #[test]
    fn stats_histogram_takes_a_bin_count_and_an_event_takes_two() {
        let mut eng = engine_after("X ~ normal(0, 1); hit = X > 0");
        assert_eq!(
            rows(&eng.run("stats::histogram(X, 8)").unwrap())[0].len(),
            8
        );

        let h = rows(&eng.run("stats::histogram(hit, 8)").unwrap());
        assert_eq!(
            h[0],
            vec![0.0, 1.0],
            "an event bins into false/true, whatever `bins` says"
        );
        let (f, t) = (h[1][0], h[1][1]);
        assert!(
            (t / (f + t) - 0.5).abs() < 0.01,
            "P(X > 0) ≈ 0.5, got {}",
            t / (f + t)
        );
    }

    /// `stats::quantiles` reads the same sample `describe` ranks, at the same levels.
    #[test]
    fn stats_quantiles_match_the_describe_card() {
        let mut eng = engine_after("X ~ normal(0, 1); d = describe(X)");
        let d = match &eng.run("describe(X)").unwrap() {
            Value::Summary(s) => match &s.payload {
                Payload::One(d) => d.clone(),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        let got = nums(
            &eng.run("stats::quantiles(X, [0.05, 0.25, 0.5, 0.75, 0.95])")
                .unwrap(),
        );
        assert_eq!(got, vec![d.q05, d.q25, d.q50, d.q75, d.q95]);
        // …and they are the quantiles of a standard normal, in the order asked.
        assert!((got[2]).abs() < 0.02, "median ≈ 0, got {}", got[2]);
        assert!((got[4] - 1.645).abs() < 0.02, "q95 ≈ 1.645, got {}", got[4]);
    }

    /// `stats::moments(x)` is `describe(x)`'s header line, as data — including the honest `n` of a
    /// conditioned variable (only the in-condition lanes count).
    #[test]
    fn stats_moments_are_the_describe_header_including_a_conditional_n() {
        let mut eng = engine_after("X ~ normal(0, 1)");
        let d = match &eng.run("describe(X)").unwrap() {
            Value::Summary(s) => match &s.payload {
                Payload::One(d) => d.clone(),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        let m = nums(&eng.run("stats::moments(X)").unwrap());
        assert_eq!(m, vec![d.n as f64, d.mean, d.sd, d.min, d.max]);

        // `X | X > 0` keeps ~half the lanes, and its mean is the half-normal's E[X | X>0] = √(2/π).
        let c = nums(&eng.run("stats::moments(X | X > 0)").unwrap());
        assert!(
            (c[0] / m[0] - 0.5).abs() < 0.01,
            "about half the lanes survive: {} of {}",
            c[0],
            m[0]
        );
        assert!(
            (c[1] - (2.0 / std::f64::consts::PI).sqrt()).abs() < 0.01,
            "E[X|X>0] = √(2/π), got {}",
            c[1]
        );
        assert!(c[3] >= 0.0, "no draw below the condition, got min {}", c[3]);
    }

    /// `stats::fan(path)` returns the very bands `plot::fan(path)` shades: 6 rows, one column per
    /// index, in the documented order.
    #[test]
    fn stats_fan_is_the_data_behind_plot_fan() {
        let mut eng = engine_after("zs ~[8] normal(0, 1); path = cumsum(zs); plot::fan(path)");
        let c = match plot_payload(&mut eng) {
            Payload::Fan(c) => c,
            other => panic!("expected a fan, got {other:?}"),
        };
        let got = rows(&eng.run("stats::fan(path)").unwrap());
        assert_eq!(got, vec![c.q05, c.q25, c.q50, c.q75, c.q95, c.mean]);
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|r| r.len() == 8), "one column per index");
        // The bands are ordered at every index, because they came from one joint pass.
        for t in 0..8 {
            assert!(
                got[0][t] < got[1][t]
                    && got[1][t] < got[2][t]
                    && got[2][t] < got[3][t]
                    && got[3][t] < got[4][t]
            );
        }
    }

    /// `stats::corr(a, b)` is the number `corr(a, b)` reports; `stats::corr(v)` is the matrix
    /// `plot::corr(v)` shades.
    #[test]
    fn stats_corr_is_a_number_for_two_variables_and_a_matrix_for_a_vector() {
        let mut eng = engine_after("X ~ normal(0, 1); N ~ normal(0, 1); Y = X + N");
        let expected = match &eng.run("corr(X, Y)").unwrap() {
            Value::Summary(s) => match &s.payload {
                Payload::Two(d) => d.corr,
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        let got = match eng.run("stats::corr(X, Y)").unwrap() {
            Value::Num(c) => c,
            other => panic!("expected a number, got {other:?}"),
        };
        assert_eq!(got, expected);
        assert!(
            (got - 0.5f64.sqrt()).abs() < 0.01,
            "corr(X, X+N) = 1/√2, got {got}"
        );

        // A vector: the element×element matrix. iid ⇒ identity, up to Monte-Carlo noise.
        let mut eng = engine_after("v ~[4] normal(0, 1)");
        let m = rows(&eng.run("stats::corr(v)").unwrap());
        assert_eq!(m.len(), 4);
        for (i, row) in m.iter().enumerate() {
            assert_eq!(row.len(), 4);
            assert!((row[i] - 1.0).abs() < 1e-9, "the diagonal is 1");
            for (j, &c) in row.iter().enumerate() {
                if i != j {
                    assert!(
                        c.abs() < 0.02,
                        "iid elements are uncorrelated: corr[{i}][{j}] = {c}"
                    );
                }
                assert_eq!(c, m[j][i], "the matrix is symmetric");
            }
        }
    }

    /// The results are ordinary arrays: index them, reduce them, feed them back into a plot.
    #[test]
    fn a_stats_result_is_an_ordinary_array() {
        assert_eq!(
            num("b ~ bernoulli(0.25); h = stats::histogram(b, 2); h[0][1]"),
            1.0
        );
        // The counts sum to the sample size, so a share is one division away.
        let p = num("b ~ bernoulli(0.25); h = stats::histogram(b, 2); h[1][1] / sum(h[1])");
        assert!(
            (p - 0.25).abs() < 0.01,
            "P(true) recovered from the counts: {p}"
        );
        // A fan's median row is a path a program can measure.
        let drift =
            num("zs ~[16] rand::normal(0, 1); f = stats::fan(cumsum(zs)); max(f[2]) - min(f[2])");
        assert!(drift > 0.0, "the median band moves");
    }

    /// `stats::` is always qualified (like `plot::`), and says so.
    #[test]
    fn stats_functions_are_always_qualified() {
        let e = run("xs ~[4] normal(0,1); fan(xs)").unwrap_err().to_string();
        assert!(
            e.contains("is in module 'stats'") && e.contains("stats::fan"),
            "{e}"
        );
        // `use stats;` parses (every module does) but grants nothing unqualified.
        let e = super::run("use stats; xs ~[4] rand::normal(0,1); fan(xs)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("stats::fan"), "{e}");
        // A bare `corr(a, b)` is still the introspection summary, not `stats::corr`'s number.
        assert!(matches!(
            run("X ~ normal(0,1); corr(X, X)").unwrap(),
            Value::Summary(_)
        ));
    }

    #[test]
    fn stats_errors_are_specific() {
        let e = run("X ~ normal(0,1); stats::bogus(X)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("unknown 'stats::bogus'"), "{e}");
        let e = run("X ~ normal(0,1); stats::quantiles(X, 0.5)")
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("wants an array") && e.contains("Q(x, 0.5)"),
            "{e}"
        );
        let e = run("X ~ normal(0,1); stats::quantiles(X, [1.5])")
            .unwrap_err()
            .to_string();
        assert!(e.contains("must lie in [0, 1]"), "{e}");
        let e = run("X ~ normal(0,1); stats::histogram(X, 0)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("at least 1 bin"), "{e}");
        let e = run("X ~ normal(0,1); stats::fan(X)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("stats::fan wants a vector"), "{e}");
        let e = run("X ~ normal(0,1); stats::corr(X)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("two variables, or one vector"), "{e}");
        // A condition that never holds has nothing to summarize — the same error `describe` gives.
        let e = run("X ~ normal(0,1); stats::moments(X | X > 100)")
            .unwrap_err()
            .to_string();
        assert!(e.contains("never occurred"), "{e}");
    }

    /// Finding B2 (DEFERRED — captures the *current, biased* behavior, not the correct one).
    ///
    /// The NaN conditioning sentinel conflates "condition false" with "the quantity is NaN on an
    /// in-condition lane". Here `X ~ unif(-1,1)` and the condition `X > -1` holds for ~every lane,
    /// but the quantity `log(X)` is NaN on the `X < 0` half. Those in-condition NaN lanes are
    /// silently dropped, so the estimate collapses to the mean over the `X > 0` lanes,
    /// `∫₀¹ ln(x) dx = -1`.
    ///
    /// The CORRECT answer is `NaN`: an in-condition NaN quantity occurs with probability ~1/2, so
    /// the conditional expectation is undefined. This test pins the biased `-1` so a future
    /// condition-column fix (see `Engine::query_cond`) has a visible before/after. When that lands,
    /// update this assertion to expect a NaN estimate.
    #[test]
    fn conditioning_drops_in_condition_nan_lanes_biased_deferred_b2() {
        let v = run("X ~ unif(-1, 1); E(log(X) | X > -1)").unwrap();
        let val = match v {
            Value::Num(n) => n,
            Value::Est { val, .. } => val,
            other => panic!("expected a number-ish estimate, got {other:?}"),
        };
        // CURRENT (biased) value; the honest value is NaN — see the doc comment above.
        assert!(
            (val - (-1.0)).abs() < 0.05,
            "captured biased conditional mean = {val} (expected ≈ -1 under the current NaN-drop; \
             the correct value is NaN once B2 is fixed)"
        );
    }

    /// `noise validate` must type/shape-check a `stats::` program without drawing a single sample —
    /// so the placeholders it returns have to carry the *right shape*, or indexing them would raise
    /// a phantom error.
    #[test]
    fn check_mode_gives_stats_results_their_real_shape_without_sampling() {
        let mut eng = Engine::new();
        let src = with_prelude(
            "X ~ normal(0,1); zs ~[5] normal(0,1); v ~[3] normal(0,1);
             h = stats::histogram(X, 7); q = stats::quantiles(X, [0.1, 0.9]);
             m = stats::moments(X); f = stats::fan(cumsum(zs)); c = stats::corr(v);
             [Len(h[0]), Len(h[1]), Len(q), Len(m), Len(f), Len(f[0]), Len(c), Len(c[0])]",
        );
        let shape = nums(&eng.check(&src).unwrap());
        assert_eq!(shape, vec![7.0, 7.0, 2.0, 5.0, 6.0, 5.0, 3.0, 3.0]);
        assert_eq!(eng.stats().samples, 0, "check mode must not draw");
    }

    #[test]
    fn two_engines_on_one_thread_keep_independent_stats() {
        // Finding B8: run-time counters are per-engine now, not a thread-local global. Two engines
        // forcing on the same thread must not corrupt each other's stats.
        let mut a = Engine::new();
        let mut b = Engine::new();
        a.run(&with_prelude("X ~ normal(0,1); E(X, 10000)"))
            .unwrap();
        assert_eq!(
            a.stats().samples,
            10000,
            "engine A should count its own 10k draws"
        );
        // B runs a different query; A's counters must be untouched.
        b.run(&with_prelude("Y ~ normal(0,1); E(Y, 50000)"))
            .unwrap();
        assert_eq!(b.stats().samples, 50000);
        assert_eq!(
            a.stats().samples,
            10000,
            "engine A's stats changed when engine B ran on the same thread"
        );
        // Re-running A resets only A (B stays at 50k).
        a.run(&with_prelude("Z ~ normal(0,1); E(Z, 20000)"))
            .unwrap();
        assert_eq!(a.stats().samples, 20000);
        assert_eq!(b.stats().samples, 50000);
    }

    #[test]
    fn joint_introspection_passes_are_counted_in_stats() {
        // Finding B8: the joint-pass drivers (`corr`/grid/fan) now record RunStats via the shared
        // `for_each_joint_batch`. A `stats::corr(v)` pass used to be entirely invisible in the
        // engine readout, contradicting the stats module's contract.
        let mut eng = Engine::new();
        eng.run(&with_prelude("v ~[4] normal(0,1); stats::corr(v)"))
            .unwrap();
        let s = eng.stats();
        assert!(
            s.samples > 0,
            "a joint corr pass must be counted (was invisible before B8): {s:?}"
        );
        assert!(
            s.forcings >= 1,
            "the pass should register as a forcing: {s:?}"
        );
        assert!(s.rng_draws > 0, "the pass draws random numbers: {s:?}");
    }

    // --- inline inputs (PLAN-INPUTS) ---

    /// An input binds its (clamped/snapped) default and the program reads it like any variable.
    #[test]
    fn input_default_binds_as_value() {
        assert_eq!(
            super::run("sides = input::int(min: 1, max: 100, default: 6); sides + 1").unwrap(),
            Value::Num(7.0)
        );
    }

    /// A host override wins over the default and is clamped/snapped by the engine.
    #[test]
    fn input_override_clamped() {
        use crate::input::InputValue;
        let mut eng = Engine::new();
        eng.set_input_overrides(vec![("sides".into(), InputValue::Num(999.0))]);
        // 999 clamps to the max (20).
        let v = eng
            .run("sides = input::int(min: 1, max: 20, default: 6); sides")
            .unwrap();
        assert_eq!(v, Value::Num(20.0));
    }

    /// An override naming an input the program doesn't declare is a clear error, not a silent no-op.
    #[test]
    fn unknown_input_override_errors() {
        use crate::input::InputValue;
        let mut eng = Engine::new();
        eng.set_input_overrides(vec![("nope".into(), InputValue::Num(1.0))]);
        let err = eng
            .run("sides = input::int(default: 6); sides")
            .unwrap_err();
        assert!(format!("{err}").contains("unknown input"), "{err}");
    }

    // --- template blocks (PLAN-LITERATE LT2) ---

    /// A root-position template emits a Note (rendered text) and yields unit; holes render via each
    /// value's display form. Dedent strips the shared indentation and the blank fence-adjacent lines.
    #[test]
    fn template_emits_as_note() {
        let mut eng = Engine::new();
        let src = "x = 42\n`\nanswer = ${x}\ntwo = ${1 + 1}\n`\n";
        let last = eng.run(src).unwrap();
        assert!(matches!(last, Value::Unit), "a root template yields unit");
        let notes: Vec<String> = eng
            .take_output()
            .into_iter()
            .filter_map(|o| match o.output {
                crate::Output::Note { text, .. } => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(notes, vec!["answer = 42\ntwo = 2".to_string()]);
    }

    /// The triple fence carries a syntax tag; the single backtick does not.
    #[test]
    fn template_triple_fence_carries_syntax_tag() {
        let mut eng = Engine::new();
        eng.run("x = 1\n```md\nhi ${x}\n```\n").unwrap();
        match eng.take_output().into_iter().next().unwrap().output {
            crate::Output::Note { text, syntax } => {
                assert_eq!(text, "hi 1");
                assert_eq!(syntax.as_deref(), Some("md"));
            }
            other => panic!("expected a note, got {other:?}"),
        }
    }

    /// Nested (non-root) a template is just a string value — usable with `+`, as a `Print` arg, etc.
    #[test]
    fn template_as_value_is_a_string() {
        assert_eq!(
            super::run("x = 5\n`v=${x}` + \"!\"").unwrap(),
            Value::Str("v=5!".into())
        );
    }

    /// An error inside a `${…}` hole points at the *original* byte offset, not a re-based one.
    #[test]
    fn template_hole_error_span_is_absolute() {
        let src = "y = `val ${nope}`";
        let err = super::run(src).unwrap_err();
        let at = src.find("nope").unwrap();
        assert_eq!(err.span.start, at, "{err}");
    }

    /// An undrawn recipe in a hole is the same error `+` raises — holes are deterministic-only.
    #[test]
    fn template_hole_rejects_undrawn_recipe() {
        let err = super::run("d = rand::unif(0,1)\n`x=${d}`").unwrap_err();
        assert!(format!("{err}").contains("undrawn distribution"), "{err}");
    }

    /// The fence is trivia: a runtime error *after* the block still points at the original byte
    /// offset (spans are not shifted by frontmatter).
    #[test]
    fn spans_survive_the_fence() {
        let src = "---\ntitle: t\n---\nundefined_name\n";
        let err = super::run(src).unwrap_err();
        // `undefined_name` starts at byte 15 in the original source.
        let at = src.find("undefined_name").unwrap();
        assert_eq!(err.span.start, at, "{err}");
    }
}
