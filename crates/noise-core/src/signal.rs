//! Lazy signal generators (a DSP "signal" value).
//!
//! `sine(freq)` / `cosine(freq)` return a [`SignalSpec`] — a *generator*, not an array. It carries
//! only a base waveform plus a chain of deferred elementwise operations (scalar arithmetic and the
//! `sin`/`cos`/`atan` ufuncs), so it costs O(1) memory regardless of how many samples it will
//! eventually produce. It materializes to a concrete array only when it meets a sized context (an
//! array it is combined with — adopting that length) or an explicit `sample(sig, n)`. This mirrors
//! how a random variable stays symbolic until `E`/`P` forces it.

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
    /// `1/f²` (red): a random walk — the cumulative sum of white (`noise_brown`).
    Brown,
    /// Colored with a single correlation time `tau` (in samples): the Ornstein–Uhlenbeck / AR(1)
    /// process, lag-1 autocorrelation `exp(-1/tau)` (`noise_ou`).
    Ou { tau: f64 },
    /// `~1/f` (pink): a sum of octave-spaced OU processes (`noise_pink`).
    Pink,
}

/// A lazy noise generator: zero-mean noise of strength `sigma` and spectral color `kind`, with no
/// length yet (like a `SignalSpec`, but *random* — it materializes into RV nodes, not floats).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSpec {
    pub sigma: f64,
    pub kind: NoiseKind,
}

impl std::fmt::Display for NoiseSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = crate::value::format_num(self.sigma);
        match self.kind {
            NoiseKind::White => write!(f, "<noise_white({s})>"),
            NoiseKind::Brown => write!(f, "<noise_brown({s})>"),
            NoiseKind::Ou { tau } => write!(f, "<noise_ou({s}, {})>", crate::value::format_num(tau)),
            NoiseKind::Pink => write!(f, "<noise_pink({s})>"),
        }
    }
}

/// A deferred elementwise step applied during sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SigOp {
    /// A scalar arithmetic op with a constant, e.g. `0.3 * x` or `x + 1`. `scalar_on_left`
    /// distinguishes `c - x` from `x - c`.
    Scalar { op: BinOp, c: f64, scalar_on_left: bool },
    /// A transcendental ufunc applied elementwise (`sin`/`cos`/`atan`).
    Unary(UnOp),
}

/// A lazy signal: a base waveform of `freq` cycles over the (as-yet-unknown) sample window, plus a
/// pipeline of deferred elementwise ops. Sampling at length `n` runs the pipeline per sample.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalSpec {
    pub wave: Wave,
    pub freq: f64,
    pub ops: Vec<SigOp>,
}

impl SignalSpec {
    pub fn base(wave: Wave, freq: f64) -> Self {
        SignalSpec { wave, freq, ops: Vec::new() }
    }

    /// Return a copy with one more deferred op appended (the lazy builder).
    pub fn push(&self, op: SigOp) -> Self {
        let mut next = self.clone();
        next.ops.push(op);
        next
    }

    /// Materialize `n` samples: the base waveform over `n` points, then the deferred pipeline.
    pub fn sample(&self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| {
                let phase = std::f64::consts::TAU * self.freq * (k as f64) / (n as f64);
                let mut x = match self.wave {
                    Wave::Sine => phase.sin(),
                    Wave::Cosine => phase.cos(),
                };
                for op in &self.ops {
                    x = apply(*op, x);
                }
                x
            })
            .collect()
    }
}

/// Apply one deferred step to a single sample value.
#[inline]
fn apply(op: SigOp, x: f64) -> f64 {
    match op {
        SigOp::Scalar { op, c, scalar_on_left } => {
            let (a, b) = if scalar_on_left { (c, x) } else { (x, c) };
            scalar_binop(op, a, b)
        }
        SigOp::Unary(UnOp::Neg) => -x,
        SigOp::Unary(UnOp::Sin) => x.sin(),
        SigOp::Unary(UnOp::Cos) => x.cos(),
        SigOp::Unary(UnOp::Atan) => x.atan(),
        SigOp::Unary(UnOp::Sign) => (x > 0.0) as i32 as f64 - (x < 0.0) as i32 as f64,
        SigOp::Unary(UnOp::Round) => x.round(),
        SigOp::Unary(UnOp::Floor) => x.floor(),
        SigOp::Unary(UnOp::Ceil) => x.ceil(),
        SigOp::Unary(UnOp::Not) => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Scalar binary kernel for the deferred arithmetic (matches the evaluator's IEEE-754 behaviour).
#[inline]
fn scalar_binop(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => a - b * (a / b).floor(),
        BinOp::Pow => a.powf(b),
        BinOp::Lt => (a < b) as i32 as f64,
        BinOp::Gt => (a > b) as i32 as f64,
        BinOp::Le => (a <= b) as i32 as f64,
        BinOp::Ge => (a >= b) as i32 as f64,
        BinOp::Eq => (a == b) as i32 as f64,
        BinOp::Ne => (a != b) as i32 as f64,
        BinOp::And => ((a != 0.0) && (b != 0.0)) as i32 as f64,
        BinOp::Or => ((a != 0.0) || (b != 0.0)) as i32 as f64,
    }
}

impl std::fmt::Display for SignalSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base = match self.wave {
            Wave::Sine => "sine",
            Wave::Cosine => "cosine",
        };
        if self.ops.is_empty() {
            write!(f, "<signal {base}({})>", self.freq)
        } else {
            // A transformed signal — show the base and how many deferred ops are pending.
            write!(f, "<signal {base}({}) +{} op(s)>", self.freq, self.ops.len())
        }
    }
}
