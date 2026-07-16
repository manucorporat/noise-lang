//! Probability surface: `P`/`E`/`var`, conditioning, hierarchical models, user functions, engine knobs.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
            "{src:?} needs a real span"
        );
    }
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
fn untargeted_queries_default_to_the_auto_precision_target() {
    // PLAN-PRECISION Phase 2: with no per-call argument and no `set_precision`, a query draws to
    // the AUTO target `se <= 5e-3 * max(|est|, sd)`. In the sd-floored regime that solves to
    // m >= 40,000 effective draws — one 65,536-lane pilot for an easy unconditional query, ~25x
    // less than the old fixed 1M default.
    let mut eng = Engine::new();
    let v = eng
        .run(&with_prelude("D ~ unif_int(1,6); P(D == 4)"))
        .unwrap();
    let Value::Est { val, se } = v else {
        panic!("expected an estimate")
    };
    assert!((val - 1.0 / 6.0).abs() < 6.0 * se, "P(D==4) = {val} ± {se}");
    assert_eq!(
        eng.stats().samples,
        65_536,
        "an easy untargeted P should stop at the pilot"
    );

    // A conditional query keeps extending until the *effective* (in-condition) m reaches the
    // target — the honest version of what the old fixed budget silently under-delivered.
    let mut eng = Engine::new();
    let v = eng
        .run(&with_prelude(
            "X ~ unif(0,1); Y ~ unif(0,1); P(Y < 0.5 | X < 0.05)",
        ))
        .unwrap();
    let Value::Est { val, se } = v else {
        panic!("expected an estimate")
    };
    assert!(
        (val - 0.5).abs() < 6.0 * se,
        "P(Y<0.5 | X<0.05) = {val} ± {se}"
    );
    assert!(
        eng.stats().samples > 500_000,
        "a 5%-acceptance conditional must sweep far past the pilot to reach m >= 40k, swept {}",
        eng.stats().samples
    );

    // An explicit per-call count still pins the draw count exactly.
    let mut eng = Engine::new();
    eng.run(&with_prelude("D ~ unif_int(1,6); P(D == 4, 2000000)"))
        .unwrap();
    assert_eq!(eng.stats().samples, 2_000_000, "per-call count is exact");
}

#[test]
fn removed_budget_pragmas_error_with_migration_hints() {
    // PLAN-PRECISION deleted the three budget pragmas. Calling one — qualified, bare, or through
    // `use engine` — is a spanned error naming its replacement, not a generic "unknown function".
    for name in ["set_max_samples", "set_max_ops", "set_max_opts"] {
        for form in [
            format!("engine::{name}(1000); 1"),
            format!("{name}(1000); 1"),
            format!("use engine; {name}(1000); 1"),
        ] {
            let err = run_raw(&form).unwrap_err().to_string();
            assert!(err.contains("was removed"), "{form}: {err}");
            assert!(
                err.contains("set_precision"),
                "{form} should name the replacement: {err}"
            );
        }
    }
}

#[test]
fn engine_module_scoping_and_validation() {
    // `set_precision` is reachable both as a `mod::name` path and (with `use`) unqualified.
    assert!(run_raw("engine::set_precision(0.01); use rand; X ~ unif(0,1); P(X < 0.5)").is_ok());
    assert!(run_raw("use engine; set_precision(0.01); 1").is_ok());
    // Out of scope without a `use`/path, with a fix-it message naming the module.
    let err = run_raw("set_precision(0.01)").unwrap_err().to_string();
    assert!(err.contains("engine") && err.contains("use"), "{err}");
    // Validation: rel/abs >= 0, not both 0; numbers only.
    assert!(run_raw("engine::set_precision(0)")
        .unwrap_err()
        .to_string()
        .contains("not both 0"));
    assert!(run_raw("engine::set_precision(0, 0)")
        .unwrap_err()
        .to_string()
        .contains("not both 0"));
    assert!(run_raw("engine::set_precision(-0.1)").is_err());
    assert!(run_raw("engine::set_precision()").is_err());
    // The abs-only form is legal (rescues quantities whose mean is ~0).
    assert!(run_raw("engine::set_precision(0, 0.001); 1").is_ok());
}

#[test]
fn precision_target_stops_early_and_goes_far() {
    // Validation 2 (PLAN-PRECISION): a loose relative target on a fat probability stops with FAR
    // fewer draws than the fixed 1M default — the stats counters record what was actually swept.
    let mut eng = Engine::new();
    let v = eng.run("use rand; C ~ bernoulli(0.5); P(C, 0.01)").unwrap();
    let (val, se) = match v {
        Value::Est { val, se } => (val, se),
        other => panic!("expected an estimate, got {other:?}"),
    };
    assert!((val - 0.5).abs() < 0.02, "P(C) = {val}");
    assert!(se <= 0.01 * val, "target missed: se {se} vs {}", 0.01 * val);
    let samples = eng.stats().samples;
    assert!(
        samples < 200_000,
        "rel=1e-2 on p=0.5 needs ~1e4 draws, swept {samples}"
    );

    // …and a tight target draws MORE than the old default would (it goes as far as the ask).
    let mut eng = Engine::new();
    let v = eng
        .run("use rand; C ~ bernoulli(0.5); P(C, 0.0005)")
        .unwrap();
    let Value::Est { val, se } = v else {
        panic!("expected an estimate")
    };
    assert!(se <= 0.0005 * val, "target missed: se {se}");
    assert!(
        eng.stats().samples > 1_000_000,
        "rel=5e-4 needs >1M draws, swept {}",
        eng.stats().samples
    );
}

#[test]
fn per_call_argument_splits_count_from_target() {
    // Validation 7: `x >= 1` is an exact count, `0 < x < 1` a relative target, `x <= 0` an error.
    let mut eng = Engine::new();
    eng.run("use rand; X ~ unif(0,1); E(X, 3)").unwrap();
    assert_eq!(eng.stats().samples, 3, "P(e, 3) must draw exactly 3");

    assert!(
        run("X ~ unif(0,1); P(X < 0.5, 0.5)").is_ok(),
        "0.5 is a 50% relative target"
    );
    for bad in ["P(X < 0.5, 0)", "P(X < 0.5, -2)", "E(X, 0)", "Var(X, -0.5)"] {
        let err = run(&format!("X ~ unif(0,1); {bad}"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("sample count") && err.contains("precision"),
            "{bad}: {err}"
        );
    }
}

#[test]
fn set_precision_pragma_governs_untargeted_queries_and_overrides_pin() {
    // Validation 9: the pragma applies when no override is passed…
    let mut eng = Engine::new();
    eng.run("use rand; engine::set_precision(0.01); C ~ bernoulli(0.5); P(C)")
        .unwrap();
    assert!(
        eng.stats().samples < 200_000,
        "pragma target should stop early, swept {}",
        eng.stats().samples
    );
    // …and a host override PINS the setting: the program's tighter pragma no longer changes it.
    let mut eng = Engine::new();
    eng.set_precision(Some((0.05, 0.0)));
    eng.run("use rand; engine::set_precision(0.0001); C ~ bernoulli(0.5); P(C)")
        .unwrap();
    assert!(
        eng.stats().samples < 100_000,
        "the looser (pinned) override must win, swept {}",
        eng.stats().samples
    );
}

#[test]
fn max_time_caps_a_run_with_an_honest_warning() {
    // Validation 4 + 8: an already-expired deadline soft-stops the run at the first statement
    // boundary — the run completes with values-so-far semantics (no error) plus a "stopped early"
    // warning; the query is skipped, never fabricated.
    let mut eng = Engine::new();
    eng.set_max_time(Some(std::time::Duration::from_nanos(1)));
    eng.run("use rand; X ~ unif(0,1); P(X < 0.5, 100000000)")
        .unwrap();
    let warnings = eng.warnings();
    assert!(
        warnings.iter().any(|w| w.contains("stopped early")),
        "{warnings:?}"
    );

    // A deadline generous enough for the pilot but not the full ask: the estimate flows with an
    // honest (wide) se and the document carries the capped warning.
    let mut eng = Engine::new();
    eng.set_max_time(Some(std::time::Duration::from_millis(30)));
    let doc = eng.run_to_document(
        "use rand;\nX ~ unif(0,1);\nP(X < 0.5, 0.00001)", // a ~1e9-draw ask
    );
    assert!(doc.result.error.is_none(), "{:?}", doc.result.error);
    assert!(
        doc.result.warnings.iter().any(|w| w.contains("max_time")),
        "warnings: {:?}",
        doc.result.warnings
    );
}

#[test]
fn check_builds_graph_but_skips_monte_carlo() {
    // `check` validates a program — parse + evaluate + build the graph — without sampling.
    // With a 100-million-sample budget a real run would be glacial; `check` returns instantly
    // because P/E/Var/Q hand back placeholders instead of forcing the cone.
    let prelude = "use rand; use math; use vec; use signal;\n";
    let heavy =
        "X ~ normal(0,1); P(X < 0, 100000000) + E(X, 100000000) + Var(X, 100000000) + Q(X, 0.5, 100000000)";
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

/// Track F gate calibration, CPU half — the multicore-interpreter timings the reduce-mode GPU gate
/// (`MIN_WORK_GPU_REDUCE`) is calibrated against. Run WITHOUT `--features gpu` (a plain build takes
/// every forcing on the CPU); pair with `bench_thin_cone_gpu` in `tests/gpu_backend.rs`. Ignored:
/// `cargo test -p noise-core --release --test probability -- --ignored --nocapture bench_thin_cone_cpu`
#[test]
#[ignore]
fn bench_thin_cone_cpu() {
    for n in [1u64 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28] {
        let src = format!("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X^2 + Y^2 < 1, {n})");
        let _ = noise_core::Engine::new().run(&src).unwrap(); // warm: caches, allocator
        let t = std::time::Instant::now();
        let v = noise_core::Engine::new().run(&src).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let noise_core::Value::Est { val, .. } = v else {
            panic!()
        };
        println!(
            "  cpu n={n:>11}  {ms:8.1} ms  ({:.0} M draws/s)  pi={val:.4}",
            n as f64 / ms / 1e3
        );
    }
}
