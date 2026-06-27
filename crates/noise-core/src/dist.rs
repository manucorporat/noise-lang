//! Distributions and the sample-DAG (PLAN.md "Core data model").
//!
//! A random variable is a node in an append-only `RvGraph` arena, referenced by a cheap
//! `RvId` handle. `~` and `unif(a,b)` produce source nodes; operator lifting (in `eval.rs`)
//! produces `Unary`/`Binary` nodes. Structural sharing is *required* for correctness: a
//! variable bound to a `Dist` reuses its single `RvId`, so `X + X` references one draw of X.
//!
//! `Source` is a concrete enum (not `Box<dyn Distribution>`) so `RvNode` stays
//! `Clone`/`PartialEq` and allocation-free; the `Distribution` trait remains the Phase-3
//! extension seam (add a `Source` variant + a struct + impl).

use crate::ast::{BinOp, UnOp};
use crate::rng::Rng;

/// Handle into the engine-owned [`RvGraph`]. Cheap, `Copy`, structural-identity equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RvId(pub u32);

/// A distribution fills a whole column of samples in one call.
pub trait Distribution {
    fn sample_into(&self, rng: &mut Rng, out: &mut [f64]);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    pub lo: f64,
    pub hi: f64,
}

impl Distribution for Uniform {
    #[inline]
    fn sample_into(&self, rng: &mut Rng, out: &mut [f64]) {
        rng.fill_uniform(self.lo, self.hi, out);
    }
}

/// Value-kind carried alongside every `RvId` so lifting can enforce the deterministic type
/// rules (e.g. reject `!num_rv`, reject `str + rv`) with spans, *before* sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvKind {
    Num,
    Bool,
}

impl RvKind {
    pub fn type_name(self) -> &'static str {
        match self {
            RvKind::Num => "dist<number>",
            RvKind::Bool => "dist<bool>",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source {
    Uniform(Uniform),
    /// Discrete uniform over the integers `lo..=hi` (inclusive). `lo`/`hi` are stored as
    /// `f64` for a uniform column representation but are sampled as integers.
    UniformInt { lo: f64, hi: f64 },
    /// Gaussian `N(mu, sigma^2)` (continuous). Sampled via Box–Muller in the column fill.
    Normal { mu: f64, sigma: f64 },
    /// Exponential `Exp(rate)` (continuous, `rate > 0`). Inverse-CDF sampled.
    Exp { rate: f64 },
    /// Poisson `Poisson(lambda)` counts (discrete, `lambda > 0`). Knuth-sampled.
    Poisson { lambda: f64 },
    /// Geometric `Geometric(p)` — failures before the first success (discrete, `0 < p <= 1`).
    Geometric { p: f64 },
}

/// A distribution **parameter that may itself be random** — the building block for hierarchical
/// models (`p ~ unif(0,1); k ~ bernoulli(p)`). `Const` is an ordinary deterministic parameter;
/// `Rv` is a sample-DAG node, so each Monte Carlo lane uses that lane's *draw* of the parameter.
/// Both fields are `Copy`, so a recipe carrying these stays `Copy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistArg {
    Const(f64),
    Rv(RvId),
}

impl std::fmt::Display for DistArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistArg::Const(x) => write!(f, "{x}"),
            DistArg::Rv(id) => write!(f, "<dist #{}>", id.0),
        }
    }
}

/// An **undrawn distribution** — a *recipe*, not a random variable (LANG.md core model §2). A
/// builtin like `unif(0,1)` produces a `Recipe`; it carries no node in the sample-DAG yet.
/// Drawing happens only at `~` (see `Engine::draw`), which instantiates *fresh* source node(s)
/// in the graph and returns a `Value::Dist`. Binding a recipe with `=` keeps it a recipe, so a
/// later `~` on it draws an independent copy. You **cannot do arithmetic on a recipe** — it has
/// no draw to operate on; the evaluator rejects it with "draw it with `~` first."
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Recipe {
    Uniform { lo: f64, hi: f64 },
    UniformInt { lo: f64, hi: f64 },
    Bernoulli { p: f64 },
    Normal { mu: f64, sigma: f64 },
    /// A circularly-symmetric complex Gaussian (CSCG, PLAN-COMPLEX §5): `re`/`im` each
    /// `~ N(0, sigma/√2)`, independent, so `E|z|² = sigma²`. Drawing it yields a `Value::Complex`
    /// whose two channels are independent normal RV nodes — see `Engine::draw`.
    NormalComplex { sigma: f64 },
    Exp { rate: f64 },
    Poisson { lambda: f64 },
    Geometric { p: f64 },
    /// Integer-rounded continuous draws: the `_int` family rounds each draw to the nearest
    /// integer (composed at draw time as `round(continuous)`), so `==`/counts are meaningful.
    NormalInt { mu: f64, sigma: f64 },
    ExpInt { rate: f64 },
    /// A random `d`×`d` orthonormal matrix (a Haar rotation). Unlike the scalar recipes above this
    /// is a *structured, multivariate* draw: `~` instantiates `d²` correlated Gaussian sources and
    /// orthonormalizes them (Gram–Schmidt), yielding a `Value::Array` of arrays rather than a
    /// scalar `Value::Dist`. See `Engine::draw_rotation`.
    Rotation { d: usize },
    /// A uniform random permutation of `0..n` (a length-`n` array, each value once). Like
    /// `Rotation` this is a *structured* draw: `~` instantiates `n` iid uniform keys and takes
    /// their argsort (each element is `rank(keyₖ)`), so every entry is an ordinary RV node and the
    /// `n` entries are jointly a permutation per Monte Carlo lane. See `Engine::draw_permutation`.
    Permutation { n: usize },
    // --- distributions with a possibly-random parameter (hierarchical models) ---
    // Each is drawn (LANG.md core model §2 / "Hierarchical distributions") by lowering to a
    // *standard base draw + a deterministic transform* (location–scale / inverse-CDF / threshold)
    // so the VM, `Source`, and RNG stay unchanged — `~` builds a fresh base draw and reuses the
    // parameter node(s). The constructors emit these only when at least one parameter is an `Rv`;
    // an all-`Const` call still uses the plain variants above. See `Engine::draw`.
    /// `unif(lo, hi)` → `lo + (hi − lo)·U`, `U ~ unif(0,1)`.
    UniformDyn { lo: DistArg, hi: DistArg },
    /// `unif_int(lo, hi)` → `lo + floor((hi − lo + 1)·U)`, `U ~ unif(0,1)`.
    UniformIntDyn { lo: DistArg, hi: DistArg },
    /// `normal(mu, sigma)` → `mu + sigma·Z`, `Z ~ normal(0,1)`. `round` of it gives `normal_int`.
    NormalDyn { mu: DistArg, sigma: DistArg, int: bool },
    /// `exponential(rate)` → `E / rate`, `E ~ Exp(1)`. `round` of it gives `exponential_int`.
    ExpDyn { rate: DistArg, int: bool },
    /// `bernoulli(p)` → `(U < p)`, `U ~ unif(0,1)` (a bool RV true with probability `p`).
    BernoulliDyn { p: DistArg },
}

impl std::fmt::Display for Recipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Recipe::Uniform { lo, hi } => write!(f, "unif({lo}, {hi})"),
            Recipe::UniformInt { lo, hi } => write!(f, "unif_int({lo}, {hi})"),
            Recipe::Bernoulli { p } => write!(f, "bernoulli({p})"),
            Recipe::Normal { mu, sigma } => write!(f, "normal({mu}, {sigma})"),
            Recipe::NormalComplex { sigma } => write!(f, "normal_complex({sigma})"),
            Recipe::Exp { rate } => write!(f, "exponential({rate})"),
            Recipe::Poisson { lambda } => write!(f, "poisson({lambda})"),
            Recipe::Geometric { p } => write!(f, "geometric({p})"),
            Recipe::NormalInt { mu, sigma } => write!(f, "normal_int({mu}, {sigma})"),
            Recipe::ExpInt { rate } => write!(f, "exponential_int({rate})"),
            Recipe::Rotation { d } => write!(f, "rotation({d})"),
            Recipe::Permutation { n } => write!(f, "permutation({n})"),
            Recipe::UniformDyn { lo, hi } => write!(f, "unif({lo}, {hi})"),
            Recipe::UniformIntDyn { lo, hi } => write!(f, "unif_int({lo}, {hi})"),
            Recipe::NormalDyn { mu, sigma, int } => {
                write!(f, "{}({mu}, {sigma})", if *int { "normal_int" } else { "normal" })
            }
            Recipe::ExpDyn { rate, int } => {
                write!(f, "{}({rate})", if *int { "exponential_int" } else { "exponential" })
            }
            Recipe::BernoulliDyn { p } => write!(f, "bernoulli({p})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RvNode {
    /// A distribution source (`RvKind::Num`).
    Src(Source),
    /// A folded deterministic numeric operand.
    ConstNum(f64),
    /// A folded deterministic boolean operand.
    ConstBool(bool),
    Unary(UnOp, RvId),
    Binary(BinOp, RvId, RvId),
    /// Per-lane select (lifted `if`): `cond ? a : b`. `cond` is a bool-RV; `a` and `b` share
    /// the result kind. Pure data-parallel — no sequential state.
    Select { cond: RvId, a: RvId, b: RvId },
    /// Per-lane gather (a *random* array index, `xs[i]` with `i` an RV): each lane reads its own
    /// `index` value, rounds and clamps it into `0..elems.len()`, and selects that element. The
    /// element nodes share the result kind. This is the one node a code generator can't emit
    /// (data-dependent addressing), so it forces the interpreter — see `kernel::walk_cost`.
    Gather { elems: Box<[RvId]>, index: RvId },
}

/// Append-only arena. Structural sharing is REQUIRED for correctness.
#[derive(Debug, Default)]
pub struct RvGraph {
    nodes: Vec<RvNode>,
    kinds: Vec<RvKind>, // parallel to `nodes`; kinds[id] is the value-kind of node id
}

impl RvGraph {
    pub fn push(&mut self, node: RvNode, kind: RvKind) -> RvId {
        let id = RvId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.kinds.push(kind);
        id
    }
    pub fn node(&self, id: RvId) -> &RvNode {
        &self.nodes[id.0 as usize]
    }
    pub fn kind(&self, id: RvId) -> RvKind {
        self.kinds[id.0 as usize]
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
