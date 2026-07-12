//! Complex numbers: arithmetic, ufuncs, complex random variables, Shor's period finding.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
        (run_num("D ~ rand::unif_int(1,6); math::abs(P(D > 3) + 0*math::i)") - 0.5).abs() < 0.01
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
