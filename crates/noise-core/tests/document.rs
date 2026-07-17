//! Document model: `input::*` controls, templates/notes, frontmatter spans.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
fn input_folds_in_builtin_numeric_arguments() {
    // A symbolic input handed to an eager numeric builtin (math::round, a sample count, …) folds
    // to its current value — the structural materialization — instead of a type error. The
    // regression: `math::round(2 * slider - 1, 2)` used to fail with "expected a number, got
    // number".
    assert_eq!(
        num("p = input::real(min: 0.5, max: 1, default: 0.6); math::round(2 * p - 1, 2)"),
        0.2
    );
}

#[test]
fn input_override_is_clamped_and_snapped() {
    let mut e = Engine::new();
    e.set_input_overrides(vec![(
        "n".into(),
        noise_core::input::InputValue::Num(250.0),
    )]);
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
    eng.set_input_overrides(vec![(
        "ghost".into(),
        noise_core::input::InputValue::Num(1.0),
    )]);
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

// --- inline inputs (PLAN-INPUTS) ---

/// An input binds its (clamped/snapped) default and the program reads it like any variable.
#[test]
fn input_default_binds_as_value() {
    assert_eq!(
        noise_core::run("sides = input::int(min: 1, max: 100, default: 6); sides + 1").unwrap(),
        Value::Num(7.0)
    );
}

/// A host override wins over the default and is clamped/snapped by the engine.
#[test]
fn input_override_clamped() {
    use noise_core::input::InputValue;
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
    use noise_core::input::InputValue;
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
            noise_core::Output::Note { text, .. } => Some(text),
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
        noise_core::Output::Note { text, syntax } => {
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
        noise_core::run("x = 5\n`v=${x}` + \"!\"").unwrap(),
        Value::Str("v=5!".into())
    );
}

/// An error inside a `${…}` hole points at the *original* byte offset, not a re-based one.
#[test]
fn template_hole_error_span_is_absolute() {
    let src = "y = `val ${nope}`";
    let err = noise_core::run(src).unwrap_err();
    let at = src.find("nope").unwrap();
    assert_eq!(err.span.start, at, "{err}");
}

/// An undrawn recipe in a hole is the same error `+` raises — holes are deterministic-only.
#[test]
fn template_hole_rejects_undrawn_recipe() {
    let err = noise_core::run("d = rand::unif(0,1)\n`x=${d}`").unwrap_err();
    assert!(format!("{err}").contains("undrawn distribution"), "{err}");
}

/// The fence is trivia: a runtime error *after* the block still points at the original byte
/// offset (spans are not shifted by frontmatter).
#[test]
fn spans_survive_the_fence() {
    let src = "---\ntitle: t\n---\nundefined_name\n";
    let err = noise_core::run(src).unwrap_err();
    // `undefined_name` starts at byte 15 in the original source.
    let at = src.find("undefined_name").unwrap();
    assert_eq!(err.span.start, at, "{err}");
}
