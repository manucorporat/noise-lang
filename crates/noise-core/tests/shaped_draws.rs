//! Shaped draws (`~[n] recipe`) after PLAN-WEBGPU G½ — the `ArrDraw`/`ArrElem` pair.
//!
//! The change replaced `n` independent `Src` nodes with ONE array-valued source owning a contiguous
//! block of `n` draw ordinals, plus `n` cheap element reads. It is meant to be **purely structural**:
//! it exists so the WGSL emitter can see "n draws from one recipe at consecutive ordinals" and emit
//! a loop instead of n inlined `squares64`s (332 ms → 31 ms of cold pipeline compile on
//! `barrier_option`'s 52 weekly normals — `tools/gpu-spike/RESULTS.md`).
//!
//! "Purely structural" is a claim with teeth, and these tests are what hold it to it: the leaves
//! must still be iid, the draws must still be the draws, and every backend must still agree.

use noise_core::{Engine, Value};

fn est(src: &str) -> f64 {
    match Engine::new().run(src).expect("program runs") {
        Value::Est { val, .. } => val,
        Value::Num(x) => x,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// A shaped draw's leaves must be **independent**. This is the property that would break first if
/// the ordinal block were misaddressed — every leaf reading element 0, say, would make `zs` a vector
/// of n copies of one draw, and `var(sum)` would be `n²` instead of `n`.
///
/// `Var(Σ zs) = n` for n iid standard normals; a fully-correlated block would give `n² = 10000`.
#[test]
fn the_leaves_of_a_shaped_draw_are_independent() {
    let v = est("use rand; use vec;\nzs ~[100] normal(0, 1);\nVar(vec::sum(zs), 400000)");
    assert!(
        (v - 100.0).abs() < 5.0,
        "Var(sum of 100 iid normals) = {v}, want ~100 \
         (a fully-correlated block would give ~10000, a constant one ~0)"
    );
}

/// Two shaped draws of the same recipe are independent of each other — the rule that makes an
/// `ArrDraw` a *source* (never CSE-merged), exactly like two `~ normal(0,1)` draws.
///
/// If they were merged, `a - b` would be identically zero and its variance 0 instead of 2.
#[test]
fn two_shaped_draws_of_one_recipe_are_independent() {
    let v = est(
        "use rand; use vec;\n\
         a ~[8] normal(0, 1);\n\
         b ~[8] normal(0, 1);\n\
         Var(vec::sum(a) - vec::sum(b), 400000)",
    );
    assert!(
        (v - 16.0).abs() < 1.0,
        "Var(sum(a) - sum(b)) = {v}, want ~16 (8 + 8); a CSE-merged block would give 0"
    );
}

/// A shaped draw must draw the **same numbers** `n` separate `~` draws would.
///
/// This is the heart of "purely structural". Element `k` of the block gets draw ordinal `base + k`,
/// and ordinals are handed out by walking node ids in order — so a program whose only sources are
/// one `~[3]` block and a program whose only sources are three scalar `~` draws assign ordinals
/// `0, 1, 2` either way, and must therefore produce identical samples. Not "the same distribution":
/// the same values, from the same seed.
#[test]
fn a_shaped_draw_draws_exactly_what_n_separate_draws_draw() {
    let shaped = est("use rand; use vec;\nzs ~[3] normal(0, 1);\nE(vec::sum(zs), 100000)");
    let scalar = est(
        "use rand;\n\
         a ~ normal(0, 1);\n\
         b ~ normal(0, 1);\n\
         c ~ normal(0, 1);\n\
         E(a + b + c, 100000)",
    );
    assert_eq!(
        shaped, scalar,
        "`~[3] normal` must draw the same stream as three `~ normal` draws, not merely the same \
         distribution — the ordinal block is laid out to make this exact"
    );
}

/// The moments survive the rewrite: a shaped uniform is still uniform, a shaped normal still normal.
/// (Cheap, but it is the test that would catch an element read landing on the wrong recipe.)
#[test]
fn shaped_draws_keep_their_distributions() {
    let mean = est("use rand; use vec;\nxs ~[10] unif(0, 1);\nE(vec::sum(xs), 400000)");
    assert!((mean - 5.0).abs() < 0.02, "E[sum of 10 U(0,1)] = {mean}, want 5");

    let var = est("use rand; use vec;\nzs ~[10] normal(0, 3);\nVar(vec::sum(zs), 400000)");
    assert!((var - 90.0).abs() < 2.0, "Var(sum of 10 N(0,9)) = {var}, want 90");
}

/// A multi-dimensional shape is ONE block, not one per row — `~[3, 4]` is a single 12-wide draw.
/// That matters for the GPU (a matrix draw is a single loop, not three) and it must not disturb
/// independence: the 12 leaves are still iid.
#[test]
fn a_matrix_shaped_draw_is_one_block_of_iid_leaves() {
    let v = est(
        "use rand; use vec;\n\
         m ~[3, 4] normal(0, 1);\n\
         Var(vec::sum(m[0]) + vec::sum(m[1]) + vec::sum(m[2]), 400000)",
    );
    assert!(
        (v - 12.0).abs() < 0.8,
        "Var(sum of a 3x4 iid normal draw) = {v}, want ~12 — row-major leaves must stay independent"
    );
}

/// The **derived** recipes must shape too, and this is what the redirection-at-`push_src` design
/// buys: a recipe is a little cone over one or more base sources (`bernoulli(p)` is a `Uniform`
/// under a `<`; `normal_int` is a `Normal` under a `round`; the hierarchical `normal(mu_rv, 1)` is a
/// standard `Normal` under an affine map). None of them are special-cased anywhere, so if the
/// redirection is right they all just work.
#[test]
fn derived_and_hierarchical_recipes_shape_correctly() {
    // bernoulli: a shaped draw of a bool-RV recipe. Element 7 must be a Bernoulli(0.3), which also
    // proves the block is addressed past its first element. (Bools don't sum — hence `P`, not `E`.)
    let b = est("use rand;\nbs ~[10] bernoulli(0.3);\nP(bs[7], 400000)");
    assert!((b - 0.3).abs() < 0.005, "P(bs[7]) = {b}, want 0.3");

    // normal_int: still integers, still mean 0.
    let ni = est("use rand; use vec;\nxs ~[4] normal_int(0, 2);\nE(vec::sum(xs), 200000)");
    assert!(ni.abs() < 0.05, "E[sum of 4 normal_int(0,2)] = {ni}, want ~0");

    // Hierarchical: one shared mu, 5 conditionally-independent draws around it.
    // Var(sum) = Var(5*mu) + 5*Var(noise) = 25*1 + 5*1 = 30.
    let h = est(
        "use rand; use vec;\n\
         mu ~ normal(0, 1);\n\
         xs ~[5] normal(mu, 1);\n\
         Var(vec::sum(xs), 400000)",
    );
    assert!(
        (h - 30.0).abs() < 1.5,
        "Var(sum of 5 draws around a shared mu) = {h}, want ~30 (25 from mu + 5 from the noise) — \
         the shared parameter node must NOT be duplicated per leaf"
    );
}

/// An element of a shaped draw is a value, not a re-draw: `zs[0] + zs[0]` is one draw doubled
/// (variance 4), not two independent draws summed (variance 2). `ArrElem` is CSE-able precisely so
/// that this stays true — it is what `zs[0]` meant back when it was a plain `Src` handle.
#[test]
fn reading_the_same_element_twice_is_one_draw() {
    let v = est("use rand;\nzs ~[4] normal(0, 1);\nVar(zs[0] + zs[0], 400000)");
    assert!(
        (v - 4.0).abs() < 0.15,
        "Var(zs[0] + zs[0]) = {v}, want 4 (one draw doubled). 2 would mean the element re-drew"
    );
}

/// A shaped draw inside a `P`/`E` still reaches the code generators and agrees with the
/// interpreter. `barrier_option`'s shape (a shaped normal folded by a scan) is the one the whole
/// exercise is for, so it gets an end-to-end check with a known answer: the sum of `n` iid standard
/// normals is `N(0, n)`, so `P(sum < 0) = 1/2`.
#[test]
fn a_shaped_draw_survives_the_codegen_gate() {
    // 300k draws: over MIN_DRAWS_JIT (100k), so the JIT compiles this rather than interpreting it.
    let p = est("use rand; use vec;\nzs ~[52] normal(0, 1);\nP(vec::sum(zs) < 0, 300000)");
    assert!(
        (p - 0.5).abs() < 0.01,
        "P(sum of 52 iid normals < 0) = {p}, want 0.5 — a codegen'd shaped draw disagrees with \
         the interpreter oracle"
    );
}
