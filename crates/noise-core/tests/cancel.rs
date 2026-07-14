//! Cancellation (PLAN-PREGPU Track A), end to end.
//!
//! The claim under test is not "a flag can be set" — it is that a forcing already *running* stops
//! promptly, that the partial answer never escapes as if it were real, and that a cancellation is
//! distinguishable from a program error. Each test below pins one of those.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use noise_core::error::ErrorKind;
use noise_core::Engine;

/// A query that genuinely runs a long time: **8.2 s uncancelled** (measured, release, M4 Pro).
///
/// The raised op budget is load-bearing. Without it the default `max_opts` ceiling silently clamps
/// the draw count and the whole thing finishes in ~0.3 s — at which point a "cancellation was
/// prompt" assertion would pass whether or not cancellation worked at all. The margin between 8.2 s
/// and the sub-second bounds below is what makes these tests able to fail.
const BIG: &str = "use rand; engine::set_max_opts(1000000000000); \
                   X ~ unif(0,1); Y ~ unif(0,1); Z ~ normal(0,1); \
                   P(X*X + Y*Y + math::sin(Z) * math::cos(Z) < 1.5, 2000000000)";

/// How long a cancelled run may take before we call the per-chunk check broken. Generous against
/// the 8.2 s the query would otherwise need — a regression that stopped checking the token would
/// blow through this by ~10×.
const ABORT_BUDGET: Duration = Duration::from_millis(1500);

#[test]
fn cancelling_mid_forcing_aborts_promptly_with_a_cancelled_error() {
    let mut eng = Engine::new();
    let token = eng.cancel_token();

    // Trip the token shortly after the forcing starts, from another thread — the native story
    // (a Ctrl-C handler, a watchdog, a request timeout).
    let started = Arc::new(AtomicBool::new(false));
    let flag = started.clone();
    let killer = std::thread::spawn(move || {
        while !flag.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        std::thread::sleep(Duration::from_millis(30));
        token.cancel();
    });

    started.store(true, Ordering::Release);
    let t = Instant::now();
    let err = eng.run(BIG).expect_err("a cancelled run must not return a value");
    let elapsed = t.elapsed();
    killer.join().unwrap();

    assert_eq!(err.kind, ErrorKind::Cancelled, "got {err:?}");
    assert_eq!(err.kind.code(), "cancelled");
    // The query needs 8.2 s to complete. Stopping well inside 1.5 s can only mean the reducer saw
    // the token mid-flight — it cannot be the run merely finishing.
    assert!(
        elapsed < ABORT_BUDGET,
        "cancellation took {elapsed:?} — the reducer is not checking the token per chunk"
    );
}

/// A cancellation is NOT a program error: it carries no diagnostic, and `is_runtime()` is false, so
/// a host can tell "the user hit stop" apart from "the program is broken".
#[test]
fn a_cancellation_is_not_a_program_error() {
    let mut eng = Engine::new();
    eng.cancel();
    let err = eng.run("1 + 1").expect_err("a pre-cancelled run must abort");

    assert!(err.kind.is_cancelled());
    assert!(
        !err.kind.is_runtime(),
        "cancellation must not masquerade as a runtime (program) error"
    );
    assert_eq!(err.to_string(), "cancelled", "no span, no diagnostic noise");

    // And a genuine program error is still classified as one.
    let mut eng = Engine::new();
    let bad = eng.run("nope + 1").expect_err("undefined name");
    assert!(bad.kind.is_runtime() && !bad.kind.is_cancelled());
}

/// Cancelling before the run even starts must abort at the first statement boundary — the coarse
/// tier — without waiting for a forcing to begin.
#[test]
fn a_pre_cancelled_engine_aborts_at_the_first_statement() {
    let mut eng = Engine::new();
    eng.cancel();
    let t = Instant::now();
    let err = eng.run(BIG).expect_err("must abort");
    assert!(err.kind.is_cancelled());
    assert!(
        t.elapsed() < Duration::from_millis(200),
        "a pre-cancelled run must not compile or sample anything"
    );
}

/// The partial-answer guarantee: a cancelled forcing has folded only *some* of its chunks, and that
/// half-finished estimate must never be handed back. The only observable outcome is `Err`.
#[test]
fn a_cancelled_forcing_never_yields_a_partial_estimate() {
    let mut eng = Engine::new();
    let token = eng.cancel_token();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(40));
        token.cancel();
    });
    // If a partial fold could escape, this would come back as a plausible-looking (but wrong,
    // under-sampled) probability instead of an error.
    match eng.run(BIG) {
        Err(e) if e.kind.is_cancelled() => {}
        Err(e) => panic!("wrong error: {e:?}"),
        Ok(v) => panic!("a cancelled forcing returned a value: {v:?} — a partial fold escaped"),
    }
}

/// `reset_cancel` makes a cancelled engine usable again — the deliberate escape hatch, with the
/// staleness caveat living in the docs rather than in behavior.
#[test]
fn reset_cancel_makes_the_engine_runnable_again() {
    let mut eng = Engine::new();
    eng.cancel();
    assert!(eng.run("1 + 1").is_err());
    eng.reset_cancel();
    let v = eng.run("use rand; D ~ unif_int(1,6); E(D, 10000)");
    assert!(v.is_ok(), "a reset engine must run: {v:?}");
}

/// An engine that is never cancelled must behave exactly as before — the token costs nothing and
/// changes no result. (Guards against the check-point accidentally aborting a healthy run.)
#[test]
fn an_uncancelled_run_is_unaffected() {
    let mut eng = Engine::new();
    let _token = eng.cancel_token(); // held, never tripped
    let v = eng
        .run("use rand; X ~ unif(-1,1); Y ~ unif(-1,1); 4 * P(X^2 + Y^2 < 1, 500000)")
        .expect("a healthy run must not be cancelled");
    match v {
        noise_core::Value::Est { val, .. } => {
            assert!((val - std::f64::consts::PI).abs() < 0.02, "pi = {val}")
        }
        other => panic!("expected an estimate, got {other:?}"),
    }
}
