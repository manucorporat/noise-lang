//! Exact enumeration of finite-discrete query cones (the Pluck/PClean "solve small discrete
//! subproblems exactly" idea, specialized to Noise's static sample-DAG).
//!
//! When every source feeding a query is finite-discrete, the query's answer is a finite sum —
//! no Monte Carlo needed. `try_enumerate` walks the root's dependency cone, assigns each source
//! a finite set of `(value, probability)` **atoms**, sweeps the joint state space, and returns
//! the exact distribution of the root. `P`/`E`/`Var`/`Q` (and their conditional forms) consult
//! it first and fall back to sampling when it declines, so `P(D == 6 | D > 3)` is exactly `1/3`
//! with zero standard error while `E(X)` over a normal still samples.
//!
//! Two source shapes are enumerable:
//! - `UniformInt { lo, hi }` — the integers `lo..=hi`, each `1/count` (matching
//!   `Rng::fill_uniform_int`'s support exactly).
//! - `Uniform { lo, hi }` used **only through comparisons against constants** — the lowered form
//!   of `bernoulli(p)` (`U < p`). The comparison cut points partition the support into intervals;
//!   each interval is one atom (probability = width/(hi−lo)) represented by its midpoint, which
//!   fixes every comparison's truth value. The cut points themselves have measure zero, so
//!   skipping them keeps the enumeration exact.
//!
//! Everything else (a normal, a Poisson, a uniform that flows into arithmetic) declines, as does
//! a state space bigger than [`MAX_STATES`] or the engine's per-query op budget — enumeration
//! must never cost more than the sampling run it replaces.
//!
//! Evaluation reuses the bytecode lowering ([`crate::bytecode::compile_roots`], so CSE and gather
//! tables come for free) with a scalar executor: source instructions are pre-seeded from the
//! current joint assignment instead of the RNG, and every other instruction applies the VM's own
//! scalar kernels — the enumerated semantics are the sampled semantics by construction.

use std::collections::{HashMap, HashSet};

use crate::ast::BinOp;
use crate::bytecode::{apply_bin, apply_un, compile_roots, Inst};
use crate::dist::{RvGraph, RvId, RvNode, Source};

/// Hard ceiling on the joint state count. Ten dice (6^10 ≈ 60M) is out; eight (1.7M) is out too;
/// twenty coin flips (2^20) is exactly in. Beyond this, sampling is the better tool anyway.
pub const MAX_STATES: u64 = 1 << 20;

/// The exact distribution of a query root: value → probability atoms, sorted by value.
/// A `select(C, x, NaN)` conditioning root carries its out-of-condition mass on a NaN atom;
/// [`Exact::condition`] strips and renormalizes it.
#[derive(Debug, Clone)]
pub struct Exact {
    /// `(value, probability)` pairs, deduplicated, sorted by value (NaN last). Probabilities
    /// sum to 1 (up to float rounding).
    pub atoms: Vec<(f64, f64)>,
}

impl Exact {
    pub fn mean(&self) -> f64 {
        self.atoms.iter().map(|&(v, p)| v * p).sum()
    }

    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        let ex2: f64 = self.atoms.iter().map(|&(v, p)| v * v * p).sum();
        (ex2 - mean * mean).max(0.0)
    }

    /// The exact quantile (inverse CDF) at level `q ∈ [0, 1]`: the smallest atom value whose
    /// cumulative probability reaches `q`. This is the true discrete quantile, not the
    /// interpolated sample statistic the Monte Carlo path reports.
    pub fn quantile(&self, q: f64) -> f64 {
        let mut cum = 0.0;
        for &(v, p) in &self.atoms {
            cum += p;
            // The epsilon absorbs float rounding in the running sum so q = 1 hits the max atom.
            if cum >= q - 1e-12 {
                return v;
            }
        }
        self.atoms.last().map_or(f64::NAN, |&(v, _)| v)
    }

    /// Restrict to the non-NaN atoms and renormalize — the exact analogue of dropping the
    /// out-of-condition lanes of a `select(C, x, NaN)` conditioning root. Returns `None` when
    /// the condition has probability exactly 0 (it can never hold).
    pub fn condition(&self) -> Option<Exact> {
        let total: f64 = self.atoms.iter().filter(|(v, _)| !v.is_nan()).map(|&(_, p)| p).sum();
        if total <= 0.0 {
            return None;
        }
        let atoms = self
            .atoms
            .iter()
            .filter(|(v, _)| !v.is_nan())
            .map(|&(v, p)| (v, p / total))
            .collect();
        Some(Exact { atoms })
    }
}

/// One enumerable source: the graph node plus its finite `(value, probability)` support.
struct SourceAtoms {
    id: RvId,
    atoms: Vec<(f64, f64)>,
}

/// Try to compute the root's exact distribution. `None` means "not enumerable here" (a
/// continuous or unbounded source, or a state space over budget) — the caller falls back to
/// Monte Carlo. `max_opts` is the engine's per-query op budget (`0` = unlimited): the sweep
/// costs ~`states × cone-ops` scalar operations and declines rather than exceed it.
pub fn try_enumerate(graph: &RvGraph, root: RvId, max_opts: u64) -> Option<Exact> {
    let cone = collect_cone(graph, root);
    let sources = classify_sources(graph, &cone, root)?;

    let mut states: u64 = 1;
    for s in &sources {
        states = states.checked_mul(s.atoms.len() as u64)?;
        if states > MAX_STATES {
            return None;
        }
    }
    let cone_ops = cone.len() as u64;
    if max_opts != 0 && states.checked_mul(cone_ops)? > max_opts {
        return None;
    }

    // Lower the cone once, requesting the root and every source as roots so we learn each
    // source's register. The shared memo means sources add no duplicate instructions.
    let mut roots: Vec<RvId> = Vec::with_capacity(1 + sources.len());
    roots.push(root);
    roots.extend(sources.iter().map(|s| s.id));
    let (prog, regs) = compile_roots(graph, &roots);
    let root_reg = regs[0] as usize;
    let src_regs: Vec<usize> = regs[1..].iter().map(|&r| r as usize).collect();
    let src_reg_set: HashSet<usize> = src_regs.iter().copied().collect();

    // Odometer sweep over the joint assignments. `choice[j]` indexes source j's atom list.
    let mut choice = vec![0usize; sources.len()];
    let mut vals = vec![0.0f64; prog.n_regs];
    let mut mass: HashMap<u64, f64> = HashMap::new();
    loop {
        let mut p = 1.0f64;
        for (j, s) in sources.iter().enumerate() {
            let (v, pj) = s.atoms[choice[j]];
            vals[src_regs[j]] = v;
            p *= pj;
        }
        let v = eval_state(&prog, &src_reg_set, &mut vals, root_reg);
        *mass.entry(v.to_bits()).or_insert(0.0) += p;

        // Advance the odometer; done when it wraps.
        let mut j = 0;
        loop {
            if j == sources.len() {
                let mut atoms: Vec<(f64, f64)> =
                    mass.into_iter().map(|(bits, p)| (f64::from_bits(bits), p)).collect();
                atoms.sort_by(|a, b| a.0.total_cmp(&b.0));
                // Renormalize: the per-state products accumulate float rounding (six 1/6's sum
                // to 1 − 1ulp), and a certain event must be exactly 1 — `P(X == X) == 1.0`.
                let total: f64 = atoms.iter().map(|&(_, p)| p).sum();
                if total > 0.0 {
                    for a in &mut atoms {
                        a.1 /= total;
                    }
                }
                return Some(Exact { atoms });
            }
            choice[j] += 1;
            if choice[j] < sources[j].atoms.len() {
                break;
            }
            choice[j] = 0;
            j += 1;
        }
    }
}

/// The transitive dependency cone of `root`.
fn collect_cone(graph: &RvGraph, root: RvId) -> HashSet<RvId> {
    let mut cone = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !cone.insert(id) {
            continue;
        }
        match graph.node(id) {
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
            RvNode::Unary(_, a) => stack.push(*a),
            RvNode::Binary(_, a, b) => stack.extend([*a, *b]),
            RvNode::Select { cond, a, b } => stack.extend([*cond, *a, *b]),
            RvNode::Gather { elems, index } => {
                stack.extend(elems.iter().copied());
                stack.push(*index);
            }
        }
    }
    cone
}

/// Assign every source in the cone a finite atom list, or decline (`None`).
fn classify_sources(
    graph: &RvGraph,
    cone: &HashSet<RvId>,
    root: RvId,
) -> Option<Vec<SourceAtoms>> {
    let mut sources = Vec::new();
    // Iterate in id order so the sweep (and float rounding) is deterministic.
    let mut ids: Vec<RvId> = cone.iter().copied().collect();
    ids.sort_by_key(|id| id.0);
    for id in ids {
        let RvNode::Src(src) = graph.node(id) else { continue };
        let atoms = match *src {
            Source::UniformInt { lo, hi } => {
                // Mirror `Rng::fill_uniform_int`: count = (hi − lo + 1).max(1), values lo + k.
                let count = (hi - lo + 1.0).max(1.0);
                if !count.is_finite() || count > MAX_STATES as f64 {
                    return None;
                }
                let n = count as u64;
                let p = 1.0 / count;
                (0..n).map(|k| (lo + k as f64, p)).collect()
            }
            Source::Uniform(u) => threshold_atoms(graph, cone, root, id, u.lo, u.hi)?,
            // Continuous or unbounded support — not enumerable.
            Source::Normal { .. }
            | Source::Exp { .. }
            | Source::Poisson { .. }
            | Source::Geometric { .. } => return None,
        };
        sources.push(SourceAtoms { id, atoms });
    }
    Some(sources)
}

/// Atoms for a continuous uniform used **only through comparisons against constants** (the
/// lowered `bernoulli(p)` shape, and hand-written forms like `U < 0.3`). The comparison cut
/// points partition `[lo, hi]` into intervals; within an interval every comparison's truth value
/// is fixed, so the interval is one atom, represented by its midpoint, with probability
/// `width / (hi − lo)`. Declines (`None`) if the source reaches anything but such a comparison.
fn threshold_atoms(
    graph: &RvGraph,
    cone: &HashSet<RvId>,
    root: RvId,
    src: RvId,
    lo: f64,
    hi: f64,
) -> Option<Vec<(f64, f64)>> {
    // The root's value is read directly, so a bare-uniform root is a non-comparison use.
    if root == src {
        return None;
    }
    let width = hi - lo;
    if !(width > 0.0) {
        // Degenerate support: a point mass (matches `lo + u·0`-style sampling collapse).
        return Some(vec![(lo, 1.0)]);
    }
    let is_cmp = |op: BinOp| {
        matches!(op, BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne)
    };
    let const_of = |id: RvId| match graph.node(id) {
        RvNode::ConstNum(v) => Some(*v),
        _ => None,
    };
    let mut cuts: Vec<f64> = Vec::new();
    for &id in cone {
        let uses_src = |x: RvId| x == src;
        match graph.node(id) {
            RvNode::Binary(op, a, b) if *a == src || *b == src => {
                if !is_cmp(*op) {
                    return None;
                }
                let other = if *a == src { *b } else { *a };
                let c = const_of(other)?;
                if c > lo && c < hi {
                    cuts.push(c);
                }
            }
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
            RvNode::Unary(_, a) if uses_src(*a) => return None,
            RvNode::Select { cond, a, b } if uses_src(*cond) || uses_src(*a) || uses_src(*b) => {
                return None
            }
            RvNode::Gather { elems, index } if uses_src(*index) || elems.iter().any(|&e| uses_src(e)) => {
                return None
            }
            _ => {}
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    // Interval atoms between consecutive boundaries; the cut points themselves carry no mass.
    let mut bounds = Vec::with_capacity(cuts.len() + 2);
    bounds.push(lo);
    bounds.extend(cuts);
    bounds.push(hi);
    let mut atoms = Vec::with_capacity(bounds.len() - 1);
    for w in bounds.windows(2) {
        let len = w[1] - w[0];
        if len > 0.0 {
            atoms.push((w[0] + len / 2.0, len / width));
        }
    }
    Some(atoms)
}

/// Evaluate one joint assignment: walk the instruction stream with one scalar per register.
/// Source registers are pre-seeded by the caller and skipped here; everything else applies the
/// VM's own scalar kernels, so enumeration and sampling share one semantics.
fn eval_state(
    prog: &crate::bytecode::Program,
    src_regs: &HashSet<usize>,
    vals: &mut [f64],
    root_reg: usize,
) -> f64 {
    for inst in &prog.insts {
        match *inst {
            Inst::Uniform { dst, .. }
            | Inst::UniformInt { dst, .. }
            | Inst::Normal { dst, .. }
            | Inst::Exp { dst, .. }
            | Inst::Poisson { dst, .. }
            | Inst::Geometric { dst, .. } => {
                debug_assert!(
                    src_regs.contains(&(dst as usize)),
                    "every source in an enumerable cone must be pre-seeded"
                );
            }
            Inst::ConstNum { dst, val } | Inst::ConstBool { dst, val } => {
                vals[dst as usize] = val;
            }
            Inst::Un { dst, op, a } => vals[dst as usize] = apply_un(op, vals[a as usize]),
            Inst::Bin { dst, op, a, b } => {
                vals[dst as usize] = apply_bin(op, vals[a as usize], vals[b as usize]);
            }
            Inst::Select { dst, cond, a, b } => {
                vals[dst as usize] =
                    if vals[cond as usize] != 0.0 { vals[a as usize] } else { vals[b as usize] };
            }
            Inst::Gather { dst, table, index } => {
                // Mirror the VM's per-lane clamp exactly (see `bytecode::run_batch`).
                let tbl = &prog.gathers[table as usize];
                let last = tbl.len() - 1;
                let raw = vals[index as usize].round();
                let i = if raw <= 0.0 {
                    0
                } else if raw as usize >= last {
                    last
                } else {
                    raw as usize
                };
                vals[dst as usize] = vals[tbl[i] as usize];
            }
        }
    }
    vals[root_reg]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{RvKind, Uniform};

    fn die(g: &mut RvGraph) -> RvId {
        g.push(RvNode::Src(Source::UniformInt { lo: 1.0, hi: 6.0 }), RvKind::Num)
    }

    #[test]
    fn die_moments_are_exact() {
        let mut g = RvGraph::default();
        let d = die(&mut g);
        let ex = try_enumerate(&g, d, 0).expect("a die is enumerable");
        assert_eq!(ex.atoms.len(), 6);
        assert!((ex.mean() - 3.5).abs() < 1e-12);
        assert!((ex.variance() - 35.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn two_dice_sum_has_exact_distribution() {
        let mut g = RvGraph::default();
        let a = die(&mut g);
        let b = die(&mut g);
        let sum = g.push(RvNode::Binary(BinOp::Add, a, b), RvKind::Num);
        let ex = try_enumerate(&g, sum, 0).unwrap();
        assert_eq!(ex.atoms.len(), 11); // 2..=12
        // P(sum == 7) = 6/36.
        let p7 = ex.atoms.iter().find(|(v, _)| *v == 7.0).unwrap().1;
        assert!((p7 - 6.0 / 36.0).abs() < 1e-12);
    }

    #[test]
    fn shared_node_is_one_draw_not_two() {
        // X - X == 0 exactly: the shared source must be a single enumerated value.
        let mut g = RvGraph::default();
        let x = die(&mut g);
        let diff = g.push(RvNode::Binary(BinOp::Sub, x, x), RvKind::Num);
        let ex = try_enumerate(&g, diff, 0).unwrap();
        assert_eq!(ex.atoms, vec![(0.0, 1.0)]);
    }

    #[test]
    fn bernoulli_lowering_enumerates_via_threshold_atoms() {
        // bernoulli(0.3) lowers to (U < 0.3), U ~ unif(0,1): P(true) must be exactly 0.3.
        let mut g = RvGraph::default();
        let u = g.push(RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })), RvKind::Num);
        let c = g.push(RvNode::ConstNum(0.3), RvKind::Num);
        let k = g.push(RvNode::Binary(BinOp::Lt, u, c), RvKind::Bool);
        let ex = try_enumerate(&g, k, 0).unwrap();
        let p_true = ex.atoms.iter().find(|(v, _)| *v == 1.0).unwrap().1;
        assert!((p_true - 0.3).abs() < 1e-12);
    }

    #[test]
    fn uniform_in_arithmetic_declines() {
        // U + 1 reads the uniform's value directly — not enumerable, must fall back to MC.
        let mut g = RvGraph::default();
        let u = g.push(RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })), RvKind::Num);
        let c = g.push(RvNode::ConstNum(1.0), RvKind::Num);
        let sum = g.push(RvNode::Binary(BinOp::Add, u, c), RvKind::Num);
        assert!(try_enumerate(&g, sum, 0).is_none());
    }

    #[test]
    fn continuous_source_declines() {
        let mut g = RvGraph::default();
        let z = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        assert!(try_enumerate(&g, z, 0).is_none());
    }

    #[test]
    fn state_space_over_budget_declines() {
        // 6^9 ≈ 10M > MAX_STATES: nine dice must decline.
        let mut g = RvGraph::default();
        let mut sum = die(&mut g);
        for _ in 0..8 {
            let d = die(&mut g);
            sum = g.push(RvNode::Binary(BinOp::Add, sum, d), RvKind::Num);
        }
        assert!(try_enumerate(&g, sum, 0).is_none());
    }

    #[test]
    fn conditioning_root_conditions_exactly() {
        // select(D > 3, D == 6, NaN): P(D == 6 | D > 3) = 1/3 exactly.
        let mut g = RvGraph::default();
        let d = die(&mut g);
        let three = g.push(RvNode::ConstNum(3.0), RvKind::Num);
        let six = g.push(RvNode::ConstNum(6.0), RvKind::Num);
        let cond = g.push(RvNode::Binary(BinOp::Gt, d, three), RvKind::Bool);
        let ev = g.push(RvNode::Binary(BinOp::Eq, d, six), RvKind::Bool);
        let nan = g.push(RvNode::ConstNum(f64::NAN), RvKind::Num);
        let root = g.push(RvNode::Select { cond, a: ev, b: nan }, RvKind::Num);
        let ex = try_enumerate(&g, root, 0).unwrap();
        let given = ex.condition().unwrap();
        assert!((given.mean() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_condition_is_none_not_nan() {
        // select(D > 6, ..., NaN): the condition has probability exactly 0.
        let mut g = RvGraph::default();
        let d = die(&mut g);
        let seven = g.push(RvNode::ConstNum(7.0), RvKind::Num);
        let cond = g.push(RvNode::Binary(BinOp::Ge, d, seven), RvKind::Bool);
        let nan = g.push(RvNode::ConstNum(f64::NAN), RvKind::Num);
        let root = g.push(RvNode::Select { cond, a: d, b: nan }, RvKind::Num);
        let ex = try_enumerate(&g, root, 0).unwrap();
        assert!(ex.condition().is_none());
    }

    #[test]
    fn exact_quantile_is_the_discrete_inverse_cdf() {
        let mut g = RvGraph::default();
        let d = die(&mut g);
        let ex = try_enumerate(&g, d, 0).unwrap();
        assert_eq!(ex.quantile(0.0), 1.0);
        assert_eq!(ex.quantile(0.5), 3.0);
        assert_eq!(ex.quantile(1.0), 6.0);
    }
}
