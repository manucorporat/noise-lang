//! Core language: parsing, arithmetic, blocks, `if`, strings, booleans, scalar ufuncs, module scoping.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
fn deterministic_if_still_short_circuits() {
    // A deterministic condition takes exactly one branch (no RV, graph untouched).
    let mut eng = Engine::new();
    let v = eng.run("if 3 > 2 { 10 } else { 20 }").unwrap();
    assert_eq!(v, Value::Num(10.0));
    assert!(eng.graph().is_empty());
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

/// Finding F3: `sqrt` is dispatched by `eval::lib_call` (NaN on a negative real), and the old
/// scalar `builtins::call` `sqrt` arm — which *errored* on negatives — was dead code with
/// contradicting semantics. This pins the language-design decision (NaN, not an error) so a
/// dispatch reorder can never silently flip it. Whether `sqrt(-1)` *should* be NaN vs an error is a
/// language choice, settled here as NaN (matching IEEE / the interpreter oracle).
#[test]
fn sqrt_negative_is_nan_not_error() {
    // Both the bare (prelude `use math`) and qualified forms take the same lib_call path.
    assert!(run_num("sqrt(-1)").is_nan());
    assert!(run_raw("use math; sqrt(-4)").unwrap().to_string() == "NaN");
    assert!(run_raw("math::sqrt(-9)").unwrap().to_string() == "NaN");
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
    // lifts over a random variable (interp and codegen agree)
    let m = run_num("X ~ rand::unif_int(0, 11); E(X % 4)");
    assert!((m - 1.5).abs() < 0.05, "E(X % 4) = {m}"); // {0,1,2,3} uniform-ish over 0..11
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
