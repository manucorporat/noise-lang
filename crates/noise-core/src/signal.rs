//! Lazy signal expressions (a DSP "signal" value).
//!
//! `sine(freq)` / `cosine(freq)` return a [`SigExpr`] leaf — a *waveform expression*, not an
//! array. Deferred operations (scalar arithmetic, signal×signal arithmetic, and the
//! `sin`/`cos`/`atan`/`exp`/… ufuncs) grow the tree, so a whole processing chain costs O(1)
//! memory regardless of how many samples it will eventually produce. It materializes to a
//! concrete array only when it meets a sized context (an array it is combined with — adopting
//! that length), an explicit `signal::sample(sig, n)`, or a reducer rendering it at the engine's
//! ambient resolution. This mirrors how a random variable stays symbolic until `E`/`P` forces it.
//!
//! A **drawn noise realization** (`static ~ signal::noise_white(1)`) is a [`SigExpr::Noise`]
//! leaf: still lazy (no length yet), but *random* — materializing it consults the engine's
//! realization cache, which pins the length at first materialization and hands every later
//! mention the SAME RV nodes (`static - static` is exactly 0). The undrawn generator
//! (`Value::Noise`) never enters a tree: `~` is the only way in.

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{BinOp, UnOp};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Wave {
    Sine,
    Cosine,
}

/// The spectral *color* of a `noise_*` generator — what distinguishes it is how samples correlate
/// across the window, not their marginal. All are zero-mean with overall strength `sigma`; the
/// engine materializes each into a vector of `normal` RV nodes (see `Engine::materialize_noise`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseKind {
    /// Flat spectrum: independent samples (`noise_white`).
    White,
    /// Circularly-symmetric complex white noise (`noise_white_complex`): independent
    /// `normal(0, sigma/√2)` per quadrature per sample, so `E|z|² = sigma²` (the same total-power
    /// convention as `rand::normal_complex`). Drawing splits it into two real `White` lanes.
    WhiteComplex,
    /// `1/f²` (red): a random walk — the cumulative sum of white (`noise_brown`).
    Brown,
    /// Colored with a single correlation time `tau` (in samples): the Ornstein–Uhlenbeck / AR(1)
    /// process, lag-1 autocorrelation `exp(-1/tau)` (`noise_ou`).
    Ou { tau: f64 },
    /// `~1/f` (pink): a sum of octave-spaced OU processes (`noise_pink`).
    Pink,
}

/// An **undrawn** noise generator: zero-mean noise of strength `sigma` and spectral color `kind`,
/// with no length and no realization yet. Like `normal(0, 1)` it is a recipe, not a value —
/// arithmetic on it is an error; `~` (or `~[n]`) is the only way to a usable realization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSpec {
    pub sigma: f64,
    pub kind: NoiseKind,
}

impl std::fmt::Display for NoiseSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = crate::value::format_num(self.sigma);
        match self.kind {
            NoiseKind::White => write!(f, "noise_white({s})"),
            NoiseKind::WhiteComplex => write!(f, "noise_white_complex({s})"),
            NoiseKind::Brown => write!(f, "noise_brown({s})"),
            NoiseKind::Ou { tau } => write!(f, "noise_ou({s}, {})", crate::value::format_num(tau)),
            NoiseKind::Pink => write!(f, "noise_pink({s})"),
        }
    }
}

/// Identity of one **drawn noise realization** — allocated by the `~` draw, carried by a
/// [`SigExpr::Noise`] leaf, and resolved through the engine's realization cache so every
/// materialization of the same draw sees the same RV nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RealizationId(pub usize);

/// A deferred unary op in a signal tree: the VM ufuncs, plus `exp` (which has no VM node — it is
/// legal only on the deterministic part of a tree and folds to `f64::exp` at materialization).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SigUnOp {
    /// One of the ordinary ufuncs (`neg`/`sin`/`cos`/`atan`/`sign`/`round`/`floor`/`ceil`).
    Un(UnOp),
    /// `math::exp` deferred into the tree.
    Exp,
}

/// A lazy signal: an expression **tree** over waveform leaves, promoted scalars, and drawn noise
/// realizations (PLAN-SIGNALS §3). `Rc` keeps the lazy builder cheap; `Value::Signal` holds
/// `Rc<SigExpr>`. Materializing at length `n` walks the tree per sample index — deterministic
/// subtrees fold as `f64`s, a `Noise` leaf switches the walk to ordinary RV lifting.
#[derive(Debug, Clone, PartialEq)]
pub enum SigExpr {
    /// A `sine`/`cosine` leaf: `freq` cycles over the (as-yet-unknown) sample window.
    Wave { wave: Wave, freq: f64 },
    /// A scalar constant promoted into signal land (`0.3 * sine(3)` carries a `Konst(0.3)`).
    Konst(f64),
    /// A **drawn noise realization** (see [`RealizationId`]). The `spec` says how to render it
    /// the first time; the cache guarantees every mention after that is the same noise.
    Noise { id: RealizationId, spec: NoiseSpec },
    /// A deferred elementwise unary op.
    Unary(SigUnOp, Rc<SigExpr>),
    /// A deferred elementwise binary op — signal×signal arithmetic (`sine(3) + sine(7)`),
    /// comparisons, and the scalar ops (the scalar side is a `Konst` leaf).
    Binop(BinOp, Rc<SigExpr>, Rc<SigExpr>),
    /// Quadrant-correct `atan2(y, x)` over two signals — the phase read-out `math::arg` defers.
    Atan2(Rc<SigExpr>, Rc<SigExpr>),
}

impl SigExpr {
    pub fn wave(wave: Wave, freq: f64) -> Rc<Self> {
        Rc::new(SigExpr::Wave { wave, freq })
    }

    /// Whether the tree contains a drawn-noise leaf (⇒ materializes into RV nodes, needs the
    /// engine; a noise-free tree folds to plain `f64`s via [`SigExpr::sample_f64`]).
    ///
    /// The tree is an `Rc` **DAG** — `for k in 0..40 { s = s + s }` shares one child under both
    /// arms of every `Binop`, so a naive tree walk is `O(2^depth)` and hangs (finding A3). We
    /// memoize per node by `Rc` identity (pointer), making it `O(distinct nodes)`.
    pub fn has_noise(&self) -> bool {
        let mut cache: HashMap<*const SigExpr, bool> = HashMap::new();
        self.has_noise_memo(&mut cache)
    }

    fn has_noise_memo(&self, cache: &mut HashMap<*const SigExpr, bool>) -> bool {
        let key = self as *const SigExpr;
        if let Some(&v) = cache.get(&key) {
            return v;
        }
        let v = match self {
            SigExpr::Wave { .. } | SigExpr::Konst(_) => false,
            SigExpr::Noise { .. } => true,
            SigExpr::Unary(_, a) => a.has_noise_memo(cache),
            SigExpr::Binop(_, a, b) | SigExpr::Atan2(a, b) => {
                a.has_noise_memo(cache) || b.has_noise_memo(cache)
            }
        };
        cache.insert(key, v);
        v
    }

    /// Materialize `n` samples of a **deterministic** tree (no `Noise` leaves — the caller
    /// checks [`SigExpr::has_noise`]; a noise leaf here is a bug, not user error). Each sample
    /// folds the shared-`Rc` DAG with a per-index memo keyed by node identity, so a diamond DAG
    /// costs `O(n · distinct nodes)` rather than `O(n · 2^depth)` (finding A3).
    pub fn sample_f64(&self, n: usize) -> Vec<f64> {
        let mut cache: HashMap<*const SigExpr, f64> = HashMap::new();
        (0..n)
            .map(|k| {
                cache.clear();
                self.eval_at(k, n, &mut cache)
            })
            .collect()
    }

    /// One deterministic sample: the tree folded at index `k` of an `n`-sample window. `cache`
    /// memoizes each node's value at this `k` by `Rc` identity (the caller clears it per index).
    fn eval_at(&self, k: usize, n: usize, cache: &mut HashMap<*const SigExpr, f64>) -> f64 {
        let key = self as *const SigExpr;
        if let Some(&v) = cache.get(&key) {
            return v;
        }
        let v = match self {
            SigExpr::Wave { wave, freq } => {
                let phase = std::f64::consts::TAU * freq * (k as f64) / (n as f64);
                match wave {
                    Wave::Sine => phase.sin(),
                    Wave::Cosine => phase.cos(),
                }
            }
            SigExpr::Konst(c) => *c,
            // Caller contract (finding F10): `sample_f64`/`eval_at` are only ever called on
            // noise-*free* trees — a noise-bearing tree is drawn through `Engine::materialize_sig`
            // (the realization cache), never folded deterministically. A `debug_assert!` documents
            // and checks the contract in tests; in release we degrade to a defined `NaN` sample
            // rather than aborting the process on a would-be-impossible input.
            SigExpr::Noise { .. } => {
                debug_assert!(false, "sample_f64/eval_at reached a Noise leaf — a noise-bearing tree must go through Engine::materialize_sig");
                f64::NAN
            }
            SigExpr::Unary(op, a) => apply_unary(*op, a.eval_at(k, n, cache)),
            SigExpr::Binop(op, a, b) => {
                scalar_binop(*op, a.eval_at(k, n, cache), b.eval_at(k, n, cache))
            }
            SigExpr::Atan2(y, x) => y.eval_at(k, n, cache).atan2(x.eval_at(k, n, cache)),
        };
        cache.insert(key, v);
        v
    }
}

/// Apply one deferred unary step to a single sample value.
#[inline]
pub fn apply_unary(op: SigUnOp, x: f64) -> f64 {
    match op {
        SigUnOp::Exp => x.exp(),
        SigUnOp::Un(UnOp::Neg) => -x,
        SigUnOp::Un(UnOp::Sin) => x.sin(),
        SigUnOp::Un(UnOp::Cos) => x.cos(),
        SigUnOp::Un(UnOp::Atan) => x.atan(),
        SigUnOp::Un(UnOp::Sign) => (x > 0.0) as i32 as f64 - (x < 0.0) as i32 as f64,
        SigUnOp::Un(UnOp::Round) => x.round(),
        SigUnOp::Un(UnOp::Floor) => x.floor(),
        SigUnOp::Un(UnOp::Ceil) => x.ceil(),
        SigUnOp::Un(UnOp::Exp) => x.exp(),
        SigUnOp::Un(UnOp::Ln) => x.ln(),
        SigUnOp::Un(UnOp::Not) => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Scalar binary kernel for the deferred arithmetic (matches the evaluator's IEEE-754 behaviour;
/// comparisons yield 0/1 like every signal-land boolean). A thin alias for the shared kernel
/// (finding F4) so the signal folder can never drift from the VM / const-fold / simplifier.
#[inline]
pub fn scalar_binop(op: BinOp, a: f64, b: f64) -> f64 {
    crate::num::fold_binop(op, a, b)
}

/// Node budget for [`SigExpr`]'s `Display`. The tree is a shared-`Rc` DAG printed *as a tree*, so a
/// diamond DAG (`for k in 0..40 { s = s + s }`) would emit `O(2^depth)` characters and OOM/hang
/// (finding A3). Past this many emitted nodes the formatter prints `…` and stops — a printed signal
/// is a diagnostic, not a serialization, so an abbreviation is the right behavior for a huge tree.
const FMT_NODE_BUDGET: usize = 10_000;

impl std::fmt::Display for SigExpr {
    /// `<signal 1 + 0.3*sine(3)>` — the tree in infix form, so a printed signal reads like the
    /// expression that built it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<signal ")?;
        let mut budget = FMT_NODE_BUDGET;
        fmt_expr(self, f, &mut budget)?;
        write!(f, ">")
    }
}

fn fmt_expr(e: &SigExpr, f: &mut std::fmt::Formatter<'_>, budget: &mut usize) -> std::fmt::Result {
    if *budget == 0 {
        return write!(f, "…");
    }
    *budget -= 1;
    match e {
        SigExpr::Wave {
            wave: Wave::Sine,
            freq,
        } => write!(f, "sine({})", crate::value::format_num(*freq)),
        SigExpr::Wave {
            wave: Wave::Cosine,
            freq,
        } => {
            write!(f, "cosine({})", crate::value::format_num(*freq))
        }
        SigExpr::Konst(c) => write!(f, "{}", crate::value::format_num(*c)),
        SigExpr::Noise { spec, .. } => write!(f, "~{spec}"),
        SigExpr::Unary(op, a) => {
            let name = match op {
                SigUnOp::Exp => "exp",
                SigUnOp::Un(UnOp::Neg) => "neg",
                SigUnOp::Un(UnOp::Not) => "not",
                SigUnOp::Un(UnOp::Sin) => "sin",
                SigUnOp::Un(UnOp::Cos) => "cos",
                SigUnOp::Un(UnOp::Atan) => "atan",
                SigUnOp::Un(UnOp::Sign) => "sign",
                SigUnOp::Un(UnOp::Round) => "round",
                SigUnOp::Un(UnOp::Floor) => "floor",
                SigUnOp::Un(UnOp::Ceil) => "ceil",
                SigUnOp::Un(UnOp::Exp) => "exp",
                SigUnOp::Un(UnOp::Ln) => "log",
            };
            write!(f, "{name}(")?;
            fmt_expr(a, f, budget)?;
            write!(f, ")")
        }
        SigExpr::Binop(op, a, b) => {
            write!(f, "(")?;
            fmt_expr(a, f, budget)?;
            let sym = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Mod => "%",
                BinOp::Pow => "^",
                BinOp::Lt => "<",
                BinOp::Gt => ">",
                BinOp::Le => "<=",
                BinOp::Ge => ">=",
                BinOp::Eq => "==",
                BinOp::Ne => "!=",
                BinOp::And => "&&",
                BinOp::Or => "||",
            };
            write!(f, " {sym} ")?;
            fmt_expr(b, f, budget)?;
            write!(f, ")")
        }
        SigExpr::Atan2(y, x) => {
            write!(f, "atan2(")?;
            fmt_expr(y, f, budget)?;
            write!(f, ", ")?;
            fmt_expr(x, f, budget)?;
            write!(f, ")")
        }
    }
}
