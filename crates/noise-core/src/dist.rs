//! Distributions and the sample-DAG (PLAN.md "Core data model").
//!
//! A random variable is a node in an append-only `RvGraph` arena, referenced by a cheap
//! `RvId` handle. `~` and `unif(a,b)` produce source nodes; operator lifting (in `eval.rs`)
//! produces `Unary`/`Binary` nodes. Structural sharing is *required* for correctness: a
//! variable bound to a `Dist` reuses its single `RvId`, so `X + X` references one draw of X.
//!
//! `Source` is a concrete enum (not `Box<dyn Distribution>`) so `RvNode` stays
//! `Clone`/`PartialEq` and allocation-free. A new distribution is added by extending the `Source`
//! and `Recipe` enums (and an `Inst` if it samples) — `normal` is the worked example — not by
//! implementing a trait: sampling is done column-at-a-time by the bytecode VM (`bytecode.rs`),
//! never through per-value dynamic dispatch.

use crate::ast::{BinOp, UnOp};

/// Handle into the engine-owned [`RvGraph`]. Cheap, `Copy`, structural-identity equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RvId(pub u32);

/// Handle into the engine-owned **dataset store** (`Engine::datasets`) — the constant data
/// arrays behind `rand::empirical` / `rand::block_bootstrap`. The data is interned so the
/// bootstrap recipes stay `Copy` like every other [`Recipe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataId(pub u32);

/// The `lo`/`hi` bounds of a continuous `unif(lo, hi)` source. A plain data holder inside
/// [`Source::Uniform`]; sampling happens in the columnar VM (`Inst::Uniform`), not here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    pub lo: f64,
    pub hi: f64,
}

/// Value-kind carried alongside every `RvId` so lifting can enforce the deterministic type
/// rules (e.g. reject `!num_rv`, reject `str + rv`) with spans, *before* sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvKind {
    Num,
    Bool,
    /// An **array-valued** node of the carried length ([`RvNode::Permutation`] /
    /// [`RvNode::Rotation`]). Never
    /// user-visible: the evaluator wraps such a node in per-element [`RvNode::ArrIndex`] reads and
    /// hands out those scalar `Value::Dist`s, so an `Arr`-kind id cannot become a forcing root or
    /// enter operator lifting (the scalar drivers' columns are `f64`).
    Arr(u32),
}

impl RvKind {
    pub fn type_name(self) -> &'static str {
        match self {
            RvKind::Num => "dist<number>",
            RvKind::Bool => "dist<bool>",
            RvKind::Arr(_) => "dist<array>",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source {
    Uniform(Uniform),
    /// Discrete uniform over the integers `lo..=hi` (inclusive). `lo`/`hi` are stored as
    /// `f64` for a uniform column representation but are sampled as integers.
    UniformInt {
        lo: f64,
        hi: f64,
    },
    /// Gaussian `N(mu, sigma^2)` (continuous). Sampled via Box–Muller in the column fill.
    Normal {
        mu: f64,
        sigma: f64,
    },
    /// Exponential `Exp(rate)` (continuous, `rate > 0`). Inverse-CDF sampled.
    Exp {
        rate: f64,
    },
    /// Poisson `Poisson(lambda)` counts (discrete, `lambda > 0`). Knuth-sampled.
    Poisson {
        lambda: f64,
    },
    /// Geometric `Geometric(p)` — failures before the first success (discrete, `0 < p <= 1`).
    Geometric {
        p: f64,
    },
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
    Uniform {
        lo: f64,
        hi: f64,
    },
    UniformInt {
        lo: f64,
        hi: f64,
    },
    Bernoulli {
        p: f64,
    },
    Normal {
        mu: f64,
        sigma: f64,
    },
    /// A circularly-symmetric complex Gaussian (CSCG, PLAN-COMPLEX §5): `re`/`im` each
    /// `~ N(0, sigma/√2)`, independent, so `E|z|² = sigma²`. Drawing it yields a `Value::Complex`
    /// whose two channels are independent normal RV nodes — see `Engine::draw`.
    NormalComplex {
        sigma: f64,
    },
    Exp {
        rate: f64,
    },
    Poisson {
        lambda: f64,
    },
    Geometric {
        p: f64,
    },
    /// Integer-rounded continuous draws: the `_int` family rounds each draw to the nearest
    /// integer (composed at draw time as `round(continuous)`), so `==`/counts are meaningful.
    NormalInt {
        mu: f64,
        sigma: f64,
    },
    ExpInt {
        rate: f64,
    },
    /// A random `d`×`d` orthonormal matrix (a Haar rotation). Unlike the scalar recipes above this
    /// is a *structured, multivariate* draw: `~` instantiates ONE array-valued [`RvNode::Rotation`]
    /// source (a per-lane Gaussian fill + modified Gram–Schmidt in the VM) and returns `d` rows of
    /// `d` scalar [`RvNode::ArrIndex`] element reads — a `Value::Array` of arrays rather than a
    /// scalar `Value::Dist`. See `Engine::draw_rotation`.
    Rotation {
        d: usize,
    },
    /// A uniform random permutation of `0..n` (a length-`n` array, each value once). Like
    /// `Rotation` this is a *structured* draw: `~` instantiates ONE array-valued
    /// [`RvNode::Permutation`] source (a per-lane Fisher–Yates in the VM) and returns its `n`
    /// scalar [`RvNode::ArrIndex`] element reads, so every entry is an ordinary RV node and the
    /// `n` entries are jointly a permutation per Monte Carlo lane. See `Engine::draw_permutation`.
    Permutation {
        n: usize,
    },
    /// An **iid bootstrap** over a constant data array (`rand::empirical(xs)`, PLAN-FINANCE F2):
    /// each `~` draws a uniformly-random element of the data — resample history instead of
    /// fitting a distribution. Sugar for `i ~ unif_int(0, Len(xs)-1); xs[i]`, but a true recipe,
    /// so `~[n]` yields iid resamples at every leaf. The data lives in the engine's dataset
    /// store (recipes stay `Copy`); drawing pushes one uniform-integer index source and a
    /// per-lane `Gather` over the constant elements. See `Engine::draw_empirical`.
    Empirical {
        data: DataId,
    },
    /// A **moving-block bootstrap** over a constant data array
    /// (`rand::block_bootstrap(xs, block_len)`): `~` draws a whole `Len(xs)`-long series glued
    /// from random contiguous blocks of the data (block starts iid `unif_int(0, n-b)`,
    /// non-wrapping; the last block truncates when `b ∤ n`), so within-block autocorrelation
    /// survives the resampling. A *structured* draw like `Permutation` — it yields a
    /// `Value::Array`. See `Engine::draw_block_bootstrap`.
    BlockBootstrap {
        data: DataId,
        block_len: usize,
    },
    // --- distributions with a possibly-random parameter (hierarchical models) ---
    // Each is drawn (LANG.md core model §2 / "Hierarchical distributions") by lowering to a
    // *standard base draw + a deterministic transform* (location–scale / inverse-CDF / threshold)
    // so the VM, `Source`, and RNG stay unchanged — `~` builds a fresh base draw and reuses the
    // parameter node(s). The constructors emit these only when at least one parameter is an `Rv`;
    // an all-`Const` call still uses the plain variants above. See `Engine::draw`.
    /// `unif(lo, hi)` → `lo + (hi − lo)·U`, `U ~ unif(0,1)`.
    UniformDyn {
        lo: DistArg,
        hi: DistArg,
    },
    /// `unif_int(lo, hi)` → `lo + floor((hi − lo + 1)·U)`, `U ~ unif(0,1)`.
    UniformIntDyn {
        lo: DistArg,
        hi: DistArg,
    },
    /// `normal(mu, sigma)` → `mu + sigma·Z`, `Z ~ normal(0,1)`. `round` of it gives `normal_int`.
    NormalDyn {
        mu: DistArg,
        sigma: DistArg,
        int: bool,
    },
    /// `exponential(rate)` → `E / rate`, `E ~ Exp(1)`. `round` of it gives `exponential_int`.
    ExpDyn {
        rate: DistArg,
        int: bool,
    },
    /// `bernoulli(p)` → `(U < p)`, `U ~ unif(0,1)` (a bool RV true with probability `p`).
    BernoulliDyn {
        p: DistArg,
    },
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
            Recipe::Empirical { .. } => write!(f, "empirical(data)"),
            Recipe::BlockBootstrap { block_len, .. } => {
                write!(f, "block_bootstrap(data, {block_len})")
            }
            Recipe::UniformDyn { lo, hi } => write!(f, "unif({lo}, {hi})"),
            Recipe::UniformIntDyn { lo, hi } => write!(f, "unif_int({lo}, {hi})"),
            Recipe::NormalDyn { mu, sigma, int } => {
                write!(
                    f,
                    "{}({mu}, {sigma})",
                    if *int { "normal_int" } else { "normal" }
                )
            }
            Recipe::ExpDyn { rate, int } => {
                write!(
                    f,
                    "{}({rate})",
                    if *int {
                        "exponential_int"
                    } else {
                        "exponential"
                    }
                )
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
    Select {
        cond: RvId,
        a: RvId,
        b: RvId,
    },
    /// Per-lane gather (a *random* array index, `xs[i]` with `i` an RV): each lane reads its own
    /// `index` value, rounds and clamps it into `0..elems.len()`, and selects that element. The
    /// element nodes share the result kind. This is the one node a code generator can't emit
    /// (data-dependent addressing), so it forces the interpreter — see `kernel::walk_cost`.
    Gather {
        elems: Box<[RvId]>,
        index: RvId,
    },
    /// A uniform random permutation of `0..n`, drawn whole (one Fisher–Yates per lane) — an
    /// **array-valued SOURCE** (`RvKind::Arr(n)`). This is what keeps `permutation(n)` an `O(n)`
    /// graph: the old argsort lowering spent `n²` compare/add nodes making each element a scalar
    /// RV, which multiplied into ~13k nodes *per forcing* in `examples/prisoners.noise`. Like any
    /// source it is never CSE-merged (two draws stay independent) and, like `Poisson`, it stays
    /// interpreter-only (`kernel::walk_cost` returns false — the shuffle is a per-lane loop with
    /// data-dependent swaps, not a fusible expression).
    Permutation {
        n: u32,
    },
    /// A Haar-random `d`×`d` orthonormal matrix, drawn whole per lane — an **array-valued SOURCE**
    /// (`RvKind::Arr(d²)`, row-major `k = row·d + col`). The VM fills `d²` iid standard normals and
    /// modified-Gram–Schmidt-orthonormalizes the rows in native Rust (`Inst::Rotation`). This is
    /// what keeps `rotation(d)` an `O(d²)` graph: the old lowering ran MGS *in the graph* —
    /// `O(d³)` dot/sub/normalize nodes (~17.5k for turboquant's `d = 20`), re-interpreted per
    /// draw. Like any source it is never CSE-merged (two draws stay independent rotations) and,
    /// like `Permutation`, it stays interpreter-only (`kernel::walk_cost` returns false — a
    /// per-lane triple loop over an array register, not a fusible scalar expression).
    Rotation {
        d: u32,
    },
    /// Scalar per-lane read `arr[index]` of an array-valued node: round the lane's `index`
    /// (ties away from zero), clamp into `0..n`, NaN index → NaN — the EXACT semantics of
    /// `Gather` (bytecode.rs), so `deck[i]` behaves identically whichever node the evaluator
    /// builds. `arr` must be `RvKind::Arr(_)`; the result is `RvKind::Num`.
    ArrIndex {
        arr: RvId,
        index: RvId,
    },
    /// A **shaped draw** `~[n] recipe`: `n` iid draws from one scalar recipe, held as ONE node
    /// (`RvKind::Arr(n)`) instead of `n` independent [`Src`](RvNode::Src) nodes.
    ///
    /// This node emits **no code on any backend**. Its whole job is to own a *contiguous block of
    /// `n` source ordinals* (see [`crate::kernel::source_ordinals`]) that its [`ArrElem`](
    /// RvNode::ArrElem) readers index into — element `k` draws from ordinal `base + k`, so the draw
    /// stream is exactly what `n` separate `Src` nodes would have produced. The interpreter, JIT and
    /// wasm emitters lower each `ArrElem` to the same scalar fill they lower a `Src` to, and never
    /// see this node at all.
    ///
    /// **Why it exists: the WGSL emitter.** WGSL has no `u64`, so `squares64` must be emulated, and
    /// each RNG source inlines ~150 ALU ops — which means shader *compile* time tracks the source
    /// count, at ~6.5 ms each (PLAN-WEBGPU G0, `tools/gpu-spike/RESULTS.md`). Unrolled,
    /// `barrier_option`'s 52 weekly normals cost **332 ms** of cold pipeline compile and the GPU
    /// *loses* to the CPU end to end; emitted as one draw loop over a block of consecutive ordinals
    /// — which is precisely what this node preserves — it costs **31 ms** and wins. The other three
    /// backends compile orders of magnitude faster and keep unrolling, so they neither gain nor lose.
    ///
    /// Like any source it is **never CSE-merged**: two `~[n]` draws of the same recipe are
    /// independent. Only scalar recipes are shaped this way — `permutation`/`rotation` are already
    /// array-valued sources of their own, and `poisson` shapes fine but stays interpreter-only.
    ArrDraw {
        n: u32,
        src: Source,
    },
    /// Element `k` of an [`ArrDraw`](RvNode::ArrDraw) — a **static** index fixed at build time, and
    /// so nothing like [`ArrIndex`](RvNode::ArrIndex) (a per-lane *random* index) or
    /// [`Gather`](RvNode::Gather). `arr` must be an `ArrDraw`; the result is `RvKind::Num`.
    ///
    /// Deterministic given `(arr, k)`, so unlike its parent it **is** CSE-able: `zs[3] + zs[3]` is
    /// one draw doubled, exactly as it was when `zs[3]` was a plain `Src` handle.
    ArrElem {
        arr: RvId,
        k: u32,
    },
    /// A re-rollable **loop** — the compact form of a `for` whose body threads loop-carried scalar
    /// state (PLAN-WEBGPU G4c). `for prisoner … { … for hop … { box = boxes[box]; found = found ||
    /// (box == prisoner) } … }` unrolls at eval time into a flat chain of thousands of dependent
    /// nodes; on the GPU that chain is one gigantic dependent basic block whose shader compile is
    /// super-linear (2.2 s for `prisoners`). This node keeps the loop instead: the CPU backends
    /// **unroll it at lowering** (byte-for-byte the flat form they lower today, so nothing about the
    /// interpreter's answer or draw stream changes), while the WGSL emitter rolls it into an actual
    /// `for` loop over per-thread `var`s.
    ///
    /// The body is an encapsulated sub-graph so the main-graph walks (`cost`, `simplify`, ordinals)
    /// see `Scan` as one opaque node; only the unroll/roll code looks inside. `ScanOut` reads a
    /// carried slot's final value. See [`ScanBody`].
    ///
    /// **v1 scope:** the body draws nothing (no `~` inside the loop), so there are no per-iteration
    /// source ordinals to thread — the sources it reads (a permutation, constants) live outside the
    /// loop and are drawn once. A loop with an internal draw falls back to eval-time unrolling.
    Scan {
        body: Box<ScanBody>,
    },
    /// Final value of carried slot `slot` after a [`Scan`](RvNode::Scan) completes its `trip`
    /// iterations. Distinct nodes per slot so a loop with several live outputs composes and dead
    /// slots are dropped by DCE. `scan` must be a `Scan`; the kind is that slot's carried kind.
    ScanOut {
        scan: RvId,
        slot: u32,
    },
    /// Inside a [`ScanBody`] sub-graph only: the value of carried slot `slot` at the *start* of an
    /// iteration (the recurrence variable the body reads), or — for `slot == INDEX_SLOT` — the
    /// iteration counter `0..trip`. Never appears in the main graph.
    Placeholder {
        slot: u32,
    },
}

/// The body of a [`RvNode::Scan`]: a recurrence over loop-carried scalar slots, run `trip` times.
///
/// **Single id space.** Every id is a node in the *same* [`RvGraph`] the `Scan` lives in — the body
/// nodes are ordinary nodes reachable only through this `Scan`'s `nexts`. That is deliberate: the
/// loop body reads loop-*invariant* values (a permutation, constants) defined outside the loop, and
/// keeping one arena makes those plain id references rather than cross-graph captures.
///
/// Slot `i` evolves `inits[i]` → then each iteration `nexts[i]` with the [`Placeholder`](
/// RvNode::Placeholder) nodes standing for the slots' values at the iteration's start. `placeholders[i]`
/// is slot `i`'s placeholder; `index_ph` (if present) is the iteration counter `0..trip`. `kinds[i]`
/// is slot `i`'s value kind.
///
/// To **unroll** (CPU, [`crate::simplify::unroll_scans`]): substitute placeholders with the previous
/// iteration's values (or `inits` / the concrete index at iteration 0) and splice the body in `trip`
/// times — reproducing exactly the flat DAG eval used to build, so the interpreter's answer and draw
/// stream are byte-for-byte unchanged. To **roll** (WGSL): one `var` per slot from `inits`, a `for`
/// loop of the body.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanBody {
    pub trip: u32,
    /// The carried-slot placeholder node ids (main graph), one per slot.
    pub placeholders: Box<[RvId]>,
    /// Each slot's value before iteration 0 (main graph).
    pub inits: Box<[RvId]>,
    /// Each slot's value at an iteration's end, in terms of the placeholders (main graph).
    pub nexts: Box<[RvId]>,
    /// The iteration-counter placeholder (`0..trip`), if the body reads the loop index.
    pub index_ph: Option<RvId>,
    pub kinds: Box<[RvKind]>,
}

/// The [`Placeholder`](RvNode::Placeholder) slot that carries the iteration counter (`0..trip`)
/// rather than a loop-carried variable. Chosen well above any real carried-slot count.
pub const INDEX_SLOT: u32 = u32::MAX;

/// Append-only arena. Structural sharing is REQUIRED for correctness.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RvGraph {
    nodes: Vec<RvNode>,
    kinds: Vec<RvKind>, // parallel to `nodes`; kinds[id] is the value-kind of node id
}

impl RvGraph {
    pub fn push(&mut self, node: RvNode, kind: RvKind) -> RvId {
        // Checked cast (finding B7): a truncating `as u32` past 2³² nodes would alias an unrelated
        // node and silently corrupt results. Construction is capped well below this (finding A6),
        // and `push` is a build-time path (not the per-lane sample loop), so the check is free
        // insurance. `expect` rather than a debug-only assert so a release build can't truncate.
        let id = RvId(u32::try_from(self.nodes.len()).expect("RvGraph exceeded 2^32 nodes"));
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
