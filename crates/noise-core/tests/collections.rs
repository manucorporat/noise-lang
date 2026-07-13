//! Collections & linear algebra: arrays, reducers, scans, matmul, bootstrap, comprehensions, TurboQuant.
//!
//! Relocated from lib.rs's in-crate `mod tests` (finding E3): an integration test that
//! exercises only the exported crate surface.

mod common;
use common::*;

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
fn permutation_draws_are_independent() {
    // two `~` draws are independent permutations (never CSE-merged into one array): their first
    // elements agree with probability 1/n, not 1 (a merged/shared array would agree always).
    let p = num("a ~ permutation(4); b ~ permutation(4); P(a[0] == b[0], 100000)");
    assert!((p - 0.25).abs() < 0.01, "got {p}");
}

#[test]
fn permutation_graph_is_linear_not_quadratic() {
    // Regression for the eval-time blowup (PLAN-PERF-2 item 3): `permutation(n)` + a random index
    // must build O(n) graph nodes — one array-valued source, n element reads, one ArrIndex for
    // the random index — not the old argsort's O(n²) compare/add nodes (~2800 for n = 30, ~13k
    // per prisoners forcing). The bound is loose (< 200 for n = 30) so only a relapse trips it.
    let mut eng = Engine::new();
    eng.run("use rand; p ~ permutation(30); i ~ unif_int(0, 29); x = p[i]; E(x)")
        .unwrap();
    let nodes = eng.graph().len();
    assert!(nodes < 200, "permutation(30) + p[i] built {nodes} nodes");
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
    assert!((num("d = [0, 1]; a ~ empirical(d); b ~ empirical(d); P(a == b)") - 0.5).abs() < 0.01);
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
    let p = num("a ~ block_bootstrap(0..20, 5); b ~ block_bootstrap(0..20, 5); P(a[0] == b[0])");
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
fn rotation_is_orthonormal() {
    // `rotation(d)` is a random orthonormal matrix, so every sample preserves length exactly:
    // ||Pi x|| = ||x|| = 1 (the mean is exactly 1, hence a tight tolerance at tiny N).
    let nrm = run_num("d = 8; x = normalize(ones(d)); Pi ~ rotation(d); E(normsq(Pi @ x), 100)");
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
fn rotation_graph_is_quadratic_not_cubic() {
    // Regression for the eval-time blowup (PLAN-PERF-2 item 3, stage 2): `rotation(d)` + a matvec
    // must build O(d²) graph nodes — one array-valued source, d² constant-index element reads,
    // ~2d² matvec arithmetic — not the old graph-level Gram–Schmidt's O(d³) dot/sub/normalize
    // chains (~17.5k nodes for turboquant's d = 20, re-interpreted per draw). The bound is loose
    // (< 2500 for d = 16, where the old lowering built >10k) so only a relapse trips it.
    let mut eng = Engine::new();
    eng.run(
        "use rand; use vec; d = 16; Pi ~ rotation(d); x = normalize(ones(d)); E(normsq(Pi @ x))",
    )
    .unwrap();
    let nodes = eng.graph().len();
    assert!(nodes < 2500, "rotation(16) + matvec built {nodes} nodes");
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

/// Finding F8: `continue` is a loop control statement, not a value. Using it in a data position —
/// bound, stored in an array, or passed as an argument — is now caught at that position with a
/// clear "not a value" message, instead of leaking a `Value::Continue` sentinel that later surfaced
/// as a baffling "arithmetic on continue and number" at the wrong place.
#[test]
fn continue_rejected_in_data_positions() {
    // Bound to a name: the error points at the binding, not the later `x + 1`.
    let e = run("x = continue; x + 1").unwrap_err().to_string();
    assert!(e.contains("continue"), "message was: {e}");
    assert!(
        e.contains("not a value") || e.contains("control statement"),
        "message was: {e}"
    );
    // Stored in an array element.
    assert!(run("[1, continue, 3]").is_err());
    // Passed as a call argument.
    assert!(run("use math; sqrt(continue)").is_err());
    // Sampled into a binding.
    assert!(run("x ~ continue").is_err());
    // Legitimate `continue` inside a loop / comprehension still works (regression guard).
    assert_eq!(
        display_of("[for x in 0..6 { if x % 2 != 0 { continue }; x }]"),
        "[0, 2, 4]"
    );
    assert_eq!(
        num("acc = 0; for x in 0..4 { if x == 2 { continue }; acc = acc + x }; acc"),
        4.0
    );
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
