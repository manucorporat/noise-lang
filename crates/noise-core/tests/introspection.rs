//! Introspection & plots: `describe`/`corr`/`explain`, `plot::*` and their `stats::*` data twins.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

#[test]
fn one_draw_reused_is_perfectly_correlated() {
    // Contrast with the above: a single `~` draw reused is one node, so P(X == X) = 1.
    let p = run_num("X ~ unif_int(1,6); P(X == X)");
    assert_eq!(p, 1.0);
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
    let indep = match &summary_of("A ~ rand::unif(0,1); B ~ rand::unif(0,1); corr(A, B)").payload {
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
    eng.run(
        "use rand; use vec; bias ~ unif(0,1); flips ~[10] bernoulli(bias); heads = count(flips);",
    )
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
            noise_core::Output::Plot(s) => Some(s),
            // `Output` is `#[non_exhaustive]` (E2): a wildcard covers the non-plot kinds.
            _ => None,
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
    let plot = noise_core::to_flint(&s);
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
    let e = noise_core::run("use stats; xs ~[4] rand::normal(0,1); fan(xs)")
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
