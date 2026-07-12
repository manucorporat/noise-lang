//! Sampling: recipes vs draws, distributions, moments, lifting, quantiles, determinism.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
fn two_draws_of_a_random_param_recipe_are_independent_given_the_param() {
    // p ~ unif(0,1); a,b ~ bernoulli(p): a and b share p but use fresh draws, so they are
    // correlated (independent only GIVEN p). For 0/1 indicators, E[a·b] = P(a && b) = E[p^2] =
    // 1/3, while P(a)·P(b) = 1/4, so the covariance is 1/3 − 1/4 = 1/12 > 0.
    let cov = run_num("p ~ unif(0,1); a ~ bernoulli(p); b ~ bernoulli(p); P(a && b) - P(a)*P(b)");
    assert!(
        (cov - 1.0 / 12.0).abs() < 5e-3,
        "cov = {cov} (expected 1/12)"
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
            // After the D2 ErrorKind split, these spanned eval-time errors land in the
            // structured variants (TypeMismatch / NotDrawn / ArityMismatch) as well as the
            // Runtime catch-all — `is_runtime()` accepts the whole runtime family.
            err.kind.is_runtime(),
            "{src:?} should be a spanned runtime-family error, got {:?}",
            err.kind
        );
        assert_ne!(
            err.span,
            noise_core::error::Span::default(),
            "{src:?} should carry a real source span, not 0..0"
        );
    }
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
fn exponential_distribution_renamed_cleanly() {
    // `exp` is the function (e^2), NOT a distribution; the old name is gone.
    assert!((num("math::exp(2)") - (2.0f64).exp()).abs() < 1e-12);
    assert!(run("X ~ exp_int(2); X").is_err()); // old `exp_int` name removed
                                                // the distribution lives on as `rand::exponential` with the same tail.
    let p = run_num("X ~ rand::exponential(2); P(X > 1)");
    assert!((p - (-2.0f64).exp()).abs() < 3e-3, "P(X>1) = {p}");
}
