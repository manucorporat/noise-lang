//! Density-weighted conditioning ("observation" of continuous draws).
//!
//! Rejection conditioning keeps the lanes where the condition happened — which is *never* for a
//! continuous equality: `E(mu | Y == 2.5)` has a measure-zero condition, so every lane is
//! rejected and the query errors. But `Y == v` observing a draw with a known density has an
//! exact importance-weighting answer (the `generate` operation of Gen [Cusumano-Towner et al.
//! 2019] / likelihood weighting): **clamp the draw to `v` and weight each lane by the density of
//! `Y` at `v`**. Self-normalized weighted averages over those lanes converge to the exact
//! conditional — Bayesian inference on continuous data, with no surface-language change.
//!
//! [`analyze`] decomposes a condition into `&&`-conjuncts and recognizes the **clampable** ones:
//! `pivot == v` (either side) where `v` is a constant and `pivot` is
//! - a continuous source itself — `Src(Normal/Exp/Uniform)` (the all-const recipes), or
//! - a hierarchical lowered shape (see `Engine::draw`): `mu + sigma·Z`, `E/rate`, `lo + width·U`
//!   — where the parameters may be random. The base draw is solved for (`Z := (v−mu)/sigma`) and
//!   the density (`normpdf((v−mu)/sigma)/|sigma|`, …) is built **as graph nodes**, so each Monte
//!   Carlo lane weights by *its* parameter draw — exactly what a hierarchical posterior needs.
//!
//! Everything unrecognized stays a **residual** condition handled by ordinary rejection, so
//! `E(mu | Y == 2.5 && mu > 0)` mixes weighting and rejection in one query. If *no* conjunct is
//! clampable, `analyze` returns `None` and the query takes the unchanged rejection path.
//!
//! Non-goals (they fall back to rejection): non-invertible pivots (`X + Y == v`,
//! `count(flips) == 7`), the `_int` families (a rounded equality has positive probability, so
//! rejection already works), and discrete draws.

use std::collections::{HashMap, HashSet};

use crate::ast::{BinOp, UnOp};
use crate::dist::{RvGraph, RvId, RvKind, RvNode, Source};

/// `1/sqrt(2π)` — the standard normal density normalizer.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// One recognized observation: replace `source` (a base draw) with `replacement` everywhere, and
/// multiply the lane weight by `weight` (the observed density, an `RvKind::Num` node).
struct Clamp {
    source: RvId,
    replacement: RvId,
    weight: RvId,
}

/// The compiled observation plan for one conditional query.
pub struct Plan {
    subst: HashMap<RvId, RvId>,
    /// Product of the observed densities (an `RvKind::Num` node; per-lane when parameters are
    /// random). NOT yet rewritten — pass it through [`Plan::rewrite`] like every other root.
    pub weight: RvId,
    /// The `&&`-fold of the unrecognized conjuncts (`None` when every conjunct was clamped).
    /// NOT yet rewritten.
    pub residual: Option<RvId>,
}

/// Try to turn `condition` into clamps + weights + a residual. `None` means no conjunct is
/// clampable — take the rejection path unchanged.
pub fn analyze(graph: &mut RvGraph, condition: RvId) -> Option<Plan> {
    let mut conjuncts = Vec::new();
    split_and(graph, condition, &mut conjuncts);

    let mut clamps: Vec<Clamp> = Vec::new();
    let mut clamped: HashSet<RvId> = HashSet::new();
    let mut residual: Vec<RvId> = Vec::new();
    for c in conjuncts {
        match recognize(graph, c) {
            // A second equality on an already-clamped draw stays a residual conjunct: it is
            // evaluated under the first clamp, so a contradiction rejects every lane (a
            // probability-0 conditional), which is the correct outcome.
            Some(cl) if !clamped.contains(&cl.source) => {
                clamped.insert(cl.source);
                clamps.push(cl);
            }
            _ => residual.push(c),
        }
    }
    if clamps.is_empty() {
        return None;
    }

    let mut weight = clamps[0].weight;
    for cl in &clamps[1..] {
        weight = graph.push(RvNode::Binary(BinOp::Mul, weight, cl.weight), RvKind::Num);
    }
    let residual = residual.into_iter().reduce(|a, b| {
        graph.push(RvNode::Binary(BinOp::And, a, b), RvKind::Bool)
    });
    let subst = clamps.into_iter().map(|c| (c.source, c.replacement)).collect();
    Some(Plan { subst, weight, residual })
}

impl Plan {
    /// Rebuild `root`'s cone with every clamped source replaced by its solved value. Replacement
    /// cones are rewritten too (a clamp's parameters may themselves be clamped draws); this
    /// terminates because a replacement only references nodes older than its source — the graph
    /// is append-only, so substitution chains strictly descend.
    pub fn rewrite(&self, graph: &mut RvGraph, root: RvId) -> RvId {
        let mut memo: HashMap<RvId, RvId> = HashMap::new();
        self.go(graph, root, &mut memo)
    }

    fn go(&self, graph: &mut RvGraph, id: RvId, memo: &mut HashMap<RvId, RvId>) -> RvId {
        if let Some(&r) = memo.get(&id) {
            return r;
        }
        if let Some(&target) = self.subst.get(&id) {
            let r = self.go(graph, target, memo);
            memo.insert(id, r);
            return r;
        }
        let kind = graph.kind(id);
        let node = graph.node(id).clone();
        let new = match node {
            // Leaves (unclamped sources and constants) stay in place.
            RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => id,
            RvNode::Unary(op, a) => {
                let a2 = self.go(graph, a, memo);
                if a2 == a { id } else { graph.push(RvNode::Unary(op, a2), kind) }
            }
            RvNode::Binary(op, a, b) => {
                let (a2, b2) = (self.go(graph, a, memo), self.go(graph, b, memo));
                if a2 == a && b2 == b { id } else { graph.push(RvNode::Binary(op, a2, b2), kind) }
            }
            RvNode::Select { cond, a, b } => {
                let (c2, a2, b2) =
                    (self.go(graph, cond, memo), self.go(graph, a, memo), self.go(graph, b, memo));
                if c2 == cond && a2 == a && b2 == b {
                    id
                } else {
                    graph.push(RvNode::Select { cond: c2, a: a2, b: b2 }, kind)
                }
            }
            RvNode::Gather { elems, index } => {
                let elems2: Box<[RvId]> = elems.iter().map(|&e| self.go(graph, e, memo)).collect();
                let i2 = self.go(graph, index, memo);
                if i2 == index && elems2.iter().zip(elems.iter()).all(|(a, b)| a == b) {
                    id
                } else {
                    graph.push(RvNode::Gather { elems: elems2, index: i2 }, kind)
                }
            }
        };
        memo.insert(id, new);
        new
    }
}

/// Flatten nested `&&` into conjuncts (the only decomposition rejection distributes over).
fn split_and(graph: &RvGraph, id: RvId, out: &mut Vec<RvId>) {
    if let RvNode::Binary(BinOp::And, a, b) = *graph.node(id) {
        split_and(graph, a, out);
        split_and(graph, b, out);
    } else {
        out.push(id);
    }
}

/// Recognize one conjunct as a clampable observation `pivot == v`.
fn recognize(graph: &mut RvGraph, conjunct: RvId) -> Option<Clamp> {
    let RvNode::Binary(BinOp::Eq, a, b) = *graph.node(conjunct) else { return None };
    let (pivot, v) = match (graph.node(a), graph.node(b)) {
        (_, RvNode::ConstNum(v)) => (a, *v),
        (RvNode::ConstNum(v), _) => (b, *v),
        _ => return None,
    };
    if !v.is_finite() {
        return None;
    }
    clamp_pivot(graph, pivot, v)
}

// Small node-building helpers, so the density expressions below read like formulas.
fn num(g: &mut RvGraph, v: f64) -> RvId {
    g.push(RvNode::ConstNum(v), RvKind::Num)
}
fn bin(g: &mut RvGraph, op: BinOp, a: RvId, b: RvId) -> RvId {
    let kind = match op {
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne | BinOp::And
        | BinOp::Or => RvKind::Bool,
        _ => RvKind::Num,
    };
    g.push(RvNode::Binary(op, a, b), kind)
}
fn un(g: &mut RvGraph, op: UnOp, a: RvId) -> RvId {
    g.push(RvNode::Unary(op, a), RvKind::Num)
}
/// `|x|` as `x·sign(x)` (no Abs op in the VM).
fn abs(g: &mut RvGraph, x: RvId) -> RvId {
    let s = un(g, UnOp::Sign, x);
    bin(g, BinOp::Mul, x, s)
}
/// `e^x` as `E^x` (`Pow` is the VM's exponential; there is no dedicated Exp op).
fn exp(g: &mut RvGraph, x: RvId) -> RvId {
    let e = num(g, std::f64::consts::E);
    bin(g, BinOp::Pow, e, x)
}

/// Match `pivot` against the invertible draw shapes and build (replacement, density-weight).
fn clamp_pivot(graph: &mut RvGraph, pivot: RvId, v: f64) -> Option<Clamp> {
    match *graph.node(pivot) {
        // --- direct sources (all-const recipes): the density is a plain constant ---
        RvNode::Src(Source::Normal { mu, sigma }) => {
            if !(sigma > 0.0) {
                return None; // a point mass has no density; rejection handles it
            }
            let z = (v - mu) / sigma;
            let w = INV_SQRT_2PI * (-0.5 * z * z).exp() / sigma;
            let replacement = num(graph, v);
            let weight = num(graph, w);
            Some(Clamp { source: pivot, replacement, weight })
        }
        RvNode::Src(Source::Exp { rate }) => {
            let w = if v >= 0.0 { rate * (-rate * v).exp() } else { 0.0 };
            let replacement = num(graph, v);
            let weight = num(graph, w);
            Some(Clamp { source: pivot, replacement, weight })
        }
        RvNode::Src(Source::Uniform(u)) => {
            if !(u.hi > u.lo) {
                return None;
            }
            let w = if v >= u.lo && v < u.hi { 1.0 / (u.hi - u.lo) } else { 0.0 };
            let replacement = num(graph, v);
            let weight = num(graph, w);
            Some(Clamp { source: pivot, replacement, weight })
        }
        // --- hierarchical lowered shapes: solve for the base draw; the density is per-lane ---
        RvNode::Binary(BinOp::Add, x, y) => {
            // normal: mu + sigma·Z, Z ~ N(0,1)  →  Z := (v−mu)/sigma, w = φ(Z)/|sigma|
            if let Some((mu, sigma, z)) = match_scaled(graph, x, y, is_std_normal) {
                let vv = num(graph, v);
                let diff = bin(graph, BinOp::Sub, vv, mu);
                let zval = bin(graph, BinOp::Div, diff, sigma);
                let z2 = bin(graph, BinOp::Mul, zval, zval);
                let half = num(graph, -0.5);
                let arg = bin(graph, BinOp::Mul, half, z2);
                let phi = exp(graph, arg);
                let c = num(graph, INV_SQRT_2PI);
                let numer = bin(graph, BinOp::Mul, c, phi);
                let denom = abs(graph, sigma);
                let weight = bin(graph, BinOp::Div, numer, denom);
                return Some(Clamp { source: z, replacement: zval, weight });
            }
            // uniform: lo + width·U, U ~ unif(0,1)  →  U := (v−lo)/width,
            // w = [0 ≤ U < 1] / |width|
            if let Some((lo, width, u)) = match_scaled(graph, x, y, is_std_uniform) {
                let vv = num(graph, v);
                let diff = bin(graph, BinOp::Sub, vv, lo);
                let uval = bin(graph, BinOp::Div, diff, width);
                let zero = num(graph, 0.0);
                let one = num(graph, 1.0);
                let ge = bin(graph, BinOp::Ge, uval, zero);
                let lt = bin(graph, BinOp::Lt, uval, one);
                // 0/1 product = the in-support indicator, kept numeric for the division.
                let ind = graph.push(RvNode::Binary(BinOp::Mul, ge, lt), RvKind::Num);
                let denom = abs(graph, width);
                let weight = bin(graph, BinOp::Div, ind, denom);
                return Some(Clamp { source: u, replacement: uval, weight });
            }
            None
        }
        // exponential: E/rate, E ~ Exp(1)  →  E := v·rate, w = rate·e^{−rate·v}·[v ≥ 0]
        RvNode::Binary(BinOp::Div, e, rate) => {
            if !matches!(graph.node(e), RvNode::Src(Source::Exp { rate }) if *rate == 1.0) {
                return None;
            }
            if v < 0.0 {
                let replacement = num(graph, 0.0);
                let weight = num(graph, 0.0);
                return Some(Clamp { source: e, replacement, weight });
            }
            let vv = num(graph, v);
            let replacement = bin(graph, BinOp::Mul, vv, rate);
            let neg = un(graph, UnOp::Neg, replacement);
            let ex = exp(graph, neg);
            let weight = bin(graph, BinOp::Mul, rate, ex);
            Some(Clamp { source: e, replacement, weight })
        }
        _ => None,
    }
}

/// Match `param + scale·B` across operand orders: one of `(x, y)` is `Binary(Mul, s, b)` (either
/// order) whose `b` satisfies `is_base`; the other is the location parameter. Returns
/// `(param, scale, base)`.
fn match_scaled(
    graph: &RvGraph,
    x: RvId,
    y: RvId,
    is_base: fn(&RvNode) -> bool,
) -> Option<(RvId, RvId, RvId)> {
    for (param, scaled) in [(x, y), (y, x)] {
        if let RvNode::Binary(BinOp::Mul, p, q) = *graph.node(scaled) {
            for (s, b) in [(p, q), (q, p)] {
                if is_base(graph.node(b)) {
                    return Some((param, s, b));
                }
            }
        }
    }
    None
}

fn is_std_normal(n: &RvNode) -> bool {
    matches!(n, RvNode::Src(Source::Normal { mu, sigma }) if *mu == 0.0 && *sigma == 1.0)
}

fn is_std_uniform(n: &RvNode) -> bool {
    matches!(n, RvNode::Src(Source::Uniform(u)) if u.lo == 0.0 && u.hi == 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Uniform;

    #[test]
    fn direct_normal_equality_is_recognized() {
        let mut g = RvGraph::default();
        let y = g.push(RvNode::Src(Source::Normal { mu: 1.0, sigma: 2.0 }), RvKind::Num);
        let c = num(&mut g, 2.0);
        let eq = bin(&mut g, BinOp::Eq, y, c);
        let plan = analyze(&mut g, eq).expect("normal == const must be clampable");
        assert!(plan.residual.is_none());
        // The weight is the constant normpdf(2; 1, 2).
        let expected = INV_SQRT_2PI * (-0.125f64).exp() / 2.0;
        match g.node(plan.weight) {
            RvNode::ConstNum(w) => assert!((w - expected).abs() < 1e-15),
            other => panic!("expected a constant weight, got {other:?}"),
        }
        // Rewriting the pivot yields the clamped constant.
        let mut g2 = g;
        let rewritten = plan.rewrite(&mut g2, y);
        assert!(matches!(g2.node(rewritten), RvNode::ConstNum(v) if *v == 2.0));
    }

    #[test]
    fn non_invertible_pivot_declines() {
        // X + Y == 1.5 with two sources: not a recognized shape → rejection path.
        let mut g = RvGraph::default();
        let x = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        let y = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        let sum = bin(&mut g, BinOp::Add, x, y);
        let c = num(&mut g, 1.5);
        let eq = bin(&mut g, BinOp::Eq, sum, c);
        assert!(analyze(&mut g, eq).is_none());
    }

    #[test]
    fn mixed_condition_keeps_a_residual() {
        // Y == 2.0 && Z > 0: the equality clamps, the inequality stays for rejection.
        let mut g = RvGraph::default();
        let y = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        let z = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        let c = num(&mut g, 2.0);
        let eq = bin(&mut g, BinOp::Eq, y, c);
        let zero = num(&mut g, 0.0);
        let gt = bin(&mut g, BinOp::Gt, z, zero);
        let both = bin(&mut g, BinOp::And, eq, gt);
        let plan = analyze(&mut g, both).expect("the equality conjunct must clamp");
        assert_eq!(plan.residual, Some(gt));
    }

    #[test]
    fn hierarchical_normal_solves_for_the_base_draw() {
        // mu_node + sigma·Z with Z ~ N(0,1): Z must be substituted, weight is per-lane.
        let mut g = RvGraph::default();
        let mu = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        let z = g.push(RvNode::Src(Source::Normal { mu: 0.0, sigma: 1.0 }), RvKind::Num);
        let sigma = num(&mut g, 1.0);
        let scaled = bin(&mut g, BinOp::Mul, sigma, z);
        let y = bin(&mut g, BinOp::Add, mu, scaled);
        let c = num(&mut g, 1.0);
        let eq = bin(&mut g, BinOp::Eq, y, c);
        let plan = analyze(&mut g, eq).expect("lowered normal must clamp");
        // The clamped source is Z (the base draw), not mu (the parameter).
        assert!(plan.subst.contains_key(&z));
        assert!(!plan.subst.contains_key(&mu));
        // Rewriting Y itself yields mu + sigma·((1 − mu)/sigma) — no Z left in the cone.
        let mut g2 = g;
        let y2 = plan.rewrite(&mut g2, y);
        let mut stack = vec![y2];
        while let Some(id) = stack.pop() {
            assert_ne!(id, z, "the base draw must be substituted away");
            match g2.node(id) {
                RvNode::Unary(_, a) => stack.push(*a),
                RvNode::Binary(_, a, b) => stack.extend([*a, *b]),
                RvNode::Select { cond, a, b } => stack.extend([*cond, *a, *b]),
                RvNode::Gather { elems, index } => {
                    stack.extend(elems.iter().copied());
                    stack.push(*index);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn contradictory_double_clamp_leaves_second_as_residual() {
        let mut g = RvGraph::default();
        let y = g.push(RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })), RvKind::Num);
        let c1 = num(&mut g, 0.3);
        let c2 = num(&mut g, 0.7);
        let eq1 = bin(&mut g, BinOp::Eq, y, c1);
        let eq2 = bin(&mut g, BinOp::Eq, y, c2);
        let both = bin(&mut g, BinOp::And, eq1, eq2);
        let plan = analyze(&mut g, both).unwrap();
        // One clamp, one residual (which will reject every lane — probability 0, correctly).
        assert_eq!(plan.subst.len(), 1);
        assert!(plan.residual.is_some());
    }
}
