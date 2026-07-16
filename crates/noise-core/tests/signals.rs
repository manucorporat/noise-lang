//! Signals & DSP: lazy signal trees, noise generators, trig/exp/log ufuncs, AM/FM, Nyquist.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
fn am_vs_fm_all_in_signal_land() {
    // The PLAN-SIGNALS §2 flagship: the whole modulate → add static → demodulate chain stays
    // lazy; the two measurement knobs sit at the top, one per axis. The numbers must land on
    // the sampled-first pipeline's (AM ≈ σ²/2 + a small demod bias, FM ≈ AM/dev² in the
    // small-signal limit — measured 0.0777 / 0.0099 at these settings).
    // (The `samples` budget moved to per-call counts — PLAN-PRECISION removed the pragma.)
    let src = "
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
        [E(mse(rec_am, msg), 8000), E(mse(rec_fm, msg), 8000)]
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
