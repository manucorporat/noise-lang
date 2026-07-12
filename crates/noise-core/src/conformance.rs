//! Shared cross-backend conformance corpus (finding C2). Test-only (`#[cfg(test)] mod conformance`
//! in `lib.rs`).
//!
//! ONE corpus, consumed by BOTH the JIT tests (`src/jit.rs`) and the WASM-emitter tests
//! (`src/wasm_emit.rs`) via `crate::conformance::{CONST_CASES, RNG_CASES}`, so the two backends can
//! no longer drift apart the way the old hand-maintained near-copies did (the wasm side had gained
//! 2nd/4th-moment and wide-trig probes the JIT never did). It holds only `(&str, ...)` data. The
//! per-backend harness (bit comparison for [`CONST_CASES`]; distribution comparison for
//! [`RNG_CASES`]) lives in each file, reusing that file's existing execution machinery
//! (`jit::build`; `wasm_emit::emit` + `wasmi`).
//!
//! Both backends already agree with the interpreter (transitively with each other) on every case:
//! the interpreter is the oracle each is checked against.

/// **Deterministic (RNG-free) programs — the exact-equality suite.**
///
/// Each is pinned to a single value by a **point-mass source**: `unif(c, c)` draws exactly `c`
/// (`lo + (hi - lo)·u = c`) and `unif_int(c, c)` draws exactly `c` (count 1 → offset 0). Making the
/// pinned draw an operand forces the emitters to build and evaluate a *real* graph node (the
/// alternative — a bare constant expression — would fold to a `ConstNum` and never reach a backend),
/// while the output carries no Monte-Carlo noise. So the interpreter, JIT, and WASM backends must be
/// **bit-identical** here (`f64::to_bits`), the strongest form of the "backend only changes speed,
/// never results" contract (`backend.rs`).
///
/// Only ops that are exact across all three backends appear: `+ - * /`, comparisons, `&&`/`||`, `!`,
/// unary `-`, `sign`, `floor`, `ceil`, `%`, lifted `if` (select), non-integer `^` (a `powf` call on
/// every backend), `atan`, `exp` (a library call on every backend — finding C9), and **large-argument
/// `sin`/`cos`** which route to the library fallback past `approx::TRIG_MAX` (finding C3), so they
/// too match libm bit-for-bit. Small-argument `sin`/`cos`/`ln` and integer-`^` are *polynomial /
/// repeated-multiply* approximations that intentionally differ from the interpreter's libm within
/// Monte-Carlo tolerance — those live in [`RNG_CASES`], where the noise hides the ~1e-9 gap.
///
/// (`1e12`-style scientific literals don't lex yet — finding D9 — so large constants are spelled in
/// full decimal.)
pub const CONST_CASES: &[(&str, &str)] = &[
    // --- core arithmetic (Add/Sub/Mul/Div) over pinned operands ---
    ("add", "use rand; X ~ unif(0,0); 3.5 + X"),
    (
        "arith_chain",
        "use rand; X ~ unif(0,0); (7.0 + X) * (3.0 + X) - (10.0 + X) / (4.0 + X)",
    ),
    ("neg", "use rand; X ~ unif(0,0); 0 - (2.5 + X)"),
    // Integer point mass through unif_int, then float arithmetic on it.
    (
        "unif_int_point_mass",
        "use rand; A ~ unif_int(5,5); A + 0.5",
    ),
    // --- comparisons → 0/1 (must be exact) ---
    ("lt_true", "use rand; X ~ unif(0,0); (3.0 + X) < (5.0 + X)"),
    ("ge_eq", "use rand; X ~ unif(0,0); (5.0 + X) >= (5.0 + X)"),
    (
        "ne",
        "use rand; X ~ unif(0,0); (5.0 + X) != (5.0 + X)", // → 0
    ),
    // --- `&&` / `||` (finding C8: wasm now lowers `(a!=0)&(b!=0)`, not min/max) ---
    (
        "and",
        "use rand; X ~ unif(0,0); ((1.0 + X) > (0.0 + X)) && ((2.0 + X) < (1.0 + X))", // T && F → 0
    ),
    (
        "or",
        "use rand; X ~ unif(0,0); ((1.0 + X) < (0.0 + X)) || ((2.0 + X) > (1.0 + X))", // F || T → 1
    ),
    (
        "not",
        "use rand; X ~ unif(0,0); !((3.0 + X) < (1.0 + X))", // !F → 1
    ),
    // --- sign / floor / ceil / mod ---
    (
        "sign_neg",
        "use rand; use math; X ~ unif(0,0); sign((0.0 + X) - 3.0)", // → -1
    ),
    (
        "floor",
        "use rand; use math; X ~ unif(0,0); math::floor(2.7 + X)",
    ),
    (
        "ceil",
        "use rand; use math; X ~ unif(0,0); math::ceil(2.1 + X)",
    ),
    ("mod_pos", "use rand; X ~ unif(0,0); (7.0 + X) % (3.0 + X)"),
    (
        "mod_neg_dividend",
        "use rand; X ~ unif(0,0); (0 - (1.0 + X)) % (3.0 + X)", // floored: -1 % 3 = 2
    ),
    // --- lifted `if` (Select) ---
    (
        "select",
        "use rand; X ~ unif(0,0); if (1.0 + X) > (0.0 + X) { 3.0 + X } else { 9.0 + X }",
    ),
    // --- non-integer `^` → a `powf` call on every backend (exact) ---
    ("pow_frac", "use rand; X ~ unif(0,0); (2.0 + X) ^ (0.5 + X)"),
    // --- exp: a library call on every backend now (finding C9 — was `pow(e,x)`) ---
    ("exp", "use rand; use math; X ~ unif(0,0); exp(1.5 + X)"),
    ("atan", "use rand; use math; X ~ unif(0,0); atan(0.7 + X)"),
    // --- large-argument trig routes to the library fallback (finding C3): matches libm exactly ---
    (
        "sin_huge",
        "use rand; use math; X ~ unif(0,0); sin(1000000000000.0 + X)", // sin(1e12)
    ),
    (
        "cos_huge",
        "use rand; use math; X ~ unif(0,0); cos(1000000000000000.0 + X)", // cos(1e15)
    ),
    (
        "sin_neg_huge",
        "use rand; use math; X ~ unif(0,0); sin((0.0 + X) - 1000000000000.0)",
    ),
];

/// **RNG programs — the distribution suite.** Each backend must agree with the interpreter *in
/// distribution*: the sample mean within `|a - b| < 0.05 + 0.05·|mean|` (the tolerance both
/// harnesses already used). Higher-moment probes bake the moment into the expression (`Z*Z`,
/// `Z*Z*Z*Z`) so a biased transcendental approximation — invisible in `E[Z]=0` — shows up in the
/// spread. `(label, source, seed)`.
///
/// This is the union of what the two corpora tested separately, so both backends now exercise the
/// same superset — including the 2nd/4th-moment and wide-range-trig probes the JIT corpus lacked,
/// and the large-argument trig case (`sin(1e12 * X)`) that motivated the C3 range guard.
pub const RNG_CASES: &[(&str, &str, u64)] = &[
    // uniform arithmetic
    ("uniform_affine", "use rand; X ~ unif(0,1); 2*X + 3", 1),
    ("uniform_cubic", "use rand; X ~ unif(-1,1); X*X*X + X", 2),
    // dice + indicator
    (
        "dice_sum",
        "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B",
        3,
    ),
    (
        "pi_indicator",
        "use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X^2 + Y^2 < 1",
        4,
    ),
    // continuous sources
    ("normal", "use rand; Z ~ normal(2,3); Z", 5),
    ("exponential", "use rand; X ~ exponential(2); X", 6),
    ("geometric", "use rand; G ~ geometric(0.25); G", 7),
    ("normal_int", "use rand; Z ~ normal_int(10,3); Z", 8),
    // higher moments of N(0,1): E[Z²] ≈ 1, E[Z⁴] ≈ 3 — tight probes of the inlined ln/cos
    ("normal_2nd_moment", "use rand; Z ~ normal(0,1); Z*Z", 15),
    (
        "normal_4th_moment",
        "use rand; Z ~ normal(0,1); Z*Z*Z*Z",
        16,
    ),
    // ufuncs + non-const pow
    (
        "sin_plus_cos",
        "use rand; use math; X ~ unif(0,1); sin(X) + cos(X)",
        9,
    ),
    ("atan", "use rand; use math; X ~ unif(-1,1); atan(X)", 10),
    ("sign", "use rand; use math; X ~ unif(-1,1); sign(X)", 11),
    (
        "pow_nonconst",
        "use rand; A ~ unif(1,2); B ~ unif(1,2); A ^ B",
        12,
    ),
    // wide-range trig (multiple periods): E[sin²]=E[cos²]=0.5 — exercises range reduction + quadrant
    (
        "sin_sq_wide",
        "use rand; use math; X ~ unif(-8,8); sin(X)*sin(X)",
        17,
    ),
    (
        "cos_sq_wide",
        "use rand; use math; X ~ unif(-8,8); cos(X)*cos(X)",
        18,
    ),
    // large-argument trig (finding C3): without the range guard the poly is nonsense past ~1e10,
    // so the backend mean would diverge from the interpreter's libm. `1e12 * X` for X ∈ (0,1).
    (
        "sin_large_arg",
        "use rand; use math; X ~ unif(0,1); sin(1000000000000.0 * X)",
        29,
    ),
    // mod / floor / ceil
    ("mod_uniform", "use rand; X ~ unif(0,10); X % 3", 19),
    ("mod_neg", "use rand; X ~ unif(-5,5); X % 4", 20),
    (
        "floor_uniform",
        "use rand; use math; X ~ unif(-3,3); math::floor(X)",
        21,
    ),
    (
        "ceil_uniform",
        "use rand; use math; X ~ unif(-3,3); math::ceil(X)",
        22,
    ),
    // exp / ln domain guards
    (
        "exp_normal",
        "use rand; use math; X ~ normal(0,1); exp(X)",
        23,
    ),
    (
        "exp_uniform",
        "use rand; use math; X ~ unif(-1,1); exp(X)",
        24,
    ),
    (
        "log_positive",
        "use rand; use math; X ~ unif(0.5, 3); log(X)",
        25,
    ),
    (
        "log_domain_neg",
        "use rand; use math; X ~ unif(-2, 3); log(X) == log(X)",
        26,
    ),
    (
        "log_domain_zero",
        "use rand; use math; X ~ unif_int(0, 4); log(X) < 0 - 100",
        27,
    ),
    (
        "log_domain_inf",
        "use rand; use math; X ~ unif(1, 2); log(X / 0) > 100",
        28,
    ),
];
