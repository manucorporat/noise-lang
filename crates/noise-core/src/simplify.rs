//! Graph-level algebraic simplification (PLAN.md Phase 4 "speed pass").
//!
//! A once-per-compile rewrite of the root's cone that **folds constants**, applies a finite-safe
//! set of **algebraic identities**, and **hash-conses** (common-subexpression elimination) — so
//! every backend (interpreter and JIT alike) lowers a smaller DAG with fewer hot-loop ops and
//! columns. It runs inside [`crate::backend::compile_root`], so the cost is paid once and shared.
//!
//! Pure: it reads the engine's graph and builds a **fresh** one, never mutating the original.
//! Two invariants keep sampling semantics intact:
//!   * **Source nodes are copied 1:1, never deduplicated** — each `~` draw is a distinct random
//!     variable, so two structurally-identical sources must stay independent.
//!   * The rebuild is **post-order**, matching the bytecode lowerer, so the relative order of the
//!     surviving sources (hence their RNG consumption) is preserved; only the ops actually removed
//!     change anything.
//!
//! Identities are restricted to those exact for *all* finite draws (`x+0`, `x*1`, `x/1`, `x^1`,
//! `x^0`, double `-`/`!`): we avoid `x*0 → 0` and `x/x → 1`, which would be wrong for a non-finite
//! lane (`inf*0`, `0/0`) the user could construct.

use std::collections::HashMap;

use crate::ast::{BinOp, UnOp};
use crate::dist::{RvGraph, RvId, RvKind, RvNode};

/// Rewrite the cone of `root` into a fresh, simplified graph. Returns the new graph and the new
/// root id. Nodes unreachable from `root` are dropped.
pub fn simplify(graph: &RvGraph, root: RvId) -> (RvGraph, RvId) {
    let mut b = Builder::default();
    let new_root = b.rewrite(graph, root);
    (b.out, new_root)
}

/// A worklist item for the iterative post-order rewrite (see [`Builder::rewrite`]).
enum Task {
    /// First visit: schedule this node's rebuild after its children.
    Visit(RvId),
    /// Second visit: children are rebuilt (in `done`); rebuild this node.
    Emit(RvId),
}

/// CSE key for the deterministic combinators (operands are already-interned new-graph ids).
#[derive(PartialEq, Eq, Hash)]
enum Key {
    Unary(UnOp, RvId),
    Binary(BinOp, RvId, RvId),
    Select(RvId, RvId, RvId),
}

#[derive(Default)]
struct Builder {
    out: RvGraph,
    done: HashMap<RvId, RvId>, // original id -> new id (memoizes the rewrite)
    cse: HashMap<Key, RvId>,   // structural dedup of Unary/Binary/Select in the new graph
    nums: HashMap<u64, RvId>,  // ConstNum dedup, keyed by bit pattern
    bools: HashMap<bool, RvId>, // ConstBool dedup
}

impl Builder {
    /// Post-order rewrite of `root`'s cone, memoized so a shared `RvId` is rebuilt once.
    ///
    /// **Iterative** post-order with an explicit `Task` worklist, *not* recursion: the cone can be
    /// hundreds of thousands of nodes deep (a `cumsum` over `~[200000]`), which would overflow a
    /// recursive rewriter's stack and abort (finding A4). The worklist reproduces the exact same
    /// post-order — children rebuilt before their parent, left operand before right — so the
    /// **relative order of surviving source nodes (hence their RNG consumption) is preserved**,
    /// which is the correctness invariant this pass promises.
    fn rewrite(&mut self, g: &RvGraph, root: RvId) -> RvId {
        if let Some(&n) = self.done.get(&root) {
            return n;
        }
        let mut stack = vec![Task::Visit(root)];
        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(id) => {
                    if self.done.contains_key(&id) {
                        continue;
                    }
                    stack.push(Task::Emit(id));
                    let done = &self.done;
                    let mut push_child = |c: RvId| {
                        if !done.contains_key(&c) {
                            stack.push(Task::Visit(c));
                        }
                    };
                    match g.node(id) {
                        RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
                        RvNode::Unary(_, a) => push_child(*a),
                        RvNode::Binary(_, l, r) => {
                            push_child(*r);
                            push_child(*l);
                        }
                        RvNode::Select { cond, a, b } => {
                            push_child(*b);
                            push_child(*a);
                            push_child(*cond);
                        }
                        RvNode::Gather { elems, index } => {
                            push_child(*index);
                            for &e in elems.iter().rev() {
                                push_child(e);
                            }
                        }
                    }
                }
                Task::Emit(id) => {
                    if self.done.contains_key(&id) {
                        continue; // reached via another path already
                    }
                    let kind = g.kind(id);
                    let new = match g.node(id) {
                        // A draw: copy as a fresh node. NEVER interned — draws stay independent.
                        RvNode::Src(s) => self.out.push(RvNode::Src(*s), kind),
                        RvNode::ConstNum(x) => self.num(*x),
                        RvNode::ConstBool(b) => self.boolean(*b),
                        RvNode::Unary(op, a) => {
                            let a = self.done[a];
                            self.unary(*op, a, kind)
                        }
                        RvNode::Binary(op, l, r) => {
                            let (l, r) = (self.done[l], self.done[r]);
                            self.binary(*op, l, r, kind)
                        }
                        RvNode::Select { cond, a, b } => {
                            let (cond, a, b) = (self.done[cond], self.done[a], self.done[b]);
                            self.select(cond, a, b, kind)
                        }
                        // Gather: rebuild over the simplified operands. Not interned — distinct
                        // gathers rarely coincide and a wrong merge would corrupt the selection.
                        RvNode::Gather { elems, index } => {
                            let elems: Vec<RvId> = elems.iter().map(|e| self.done[e]).collect();
                            let index = self.done[index];
                            self.out.push(
                                RvNode::Gather {
                                    elems: elems.into_boxed_slice(),
                                    index,
                                },
                                kind,
                            )
                        }
                    };
                    self.done.insert(id, new);
                }
            }
        }
        self.done[&root]
    }

    // --- interning constructors (the only way nodes enter `out`, besides sources) ---

    fn num(&mut self, x: f64) -> RvId {
        if let Some(&id) = self.nums.get(&x.to_bits()) {
            return id;
        }
        let id = self.out.push(RvNode::ConstNum(x), RvKind::Num);
        self.nums.insert(x.to_bits(), id);
        id
    }

    fn boolean(&mut self, b: bool) -> RvId {
        if let Some(&id) = self.bools.get(&b) {
            return id;
        }
        let id = self.out.push(RvNode::ConstBool(b), RvKind::Bool);
        self.bools.insert(b, id);
        id
    }

    fn intern(&mut self, key: Key, node: RvNode, kind: RvKind) -> RvId {
        if let Some(&id) = self.cse.get(&key) {
            return id;
        }
        let id = self.out.push(node, kind);
        self.cse.insert(key, id);
        id
    }

    // --- node-level rewrites: constant fold, then identity, then intern ---

    fn unary(&mut self, op: UnOp, a: RvId, kind: RvKind) -> RvId {
        if let Some(x) = self.as_num(a) {
            match op {
                UnOp::Neg => return self.num(-x),
                UnOp::Sin => return self.num(x.sin()),
                UnOp::Cos => return self.num(x.cos()),
                UnOp::Atan => return self.num(x.atan()),
                UnOp::Sign => return self.num((x > 0.0) as i32 as f64 - (x < 0.0) as i32 as f64),
                UnOp::Round => return self.num(x.round()),
                UnOp::Floor => return self.num(x.floor()),
                UnOp::Ceil => return self.num(x.ceil()),
                UnOp::Exp => return self.num(x.exp()),
                UnOp::Ln => return self.num(x.ln()),
                UnOp::Not => {} // kind-checked away upstream; fall through
            }
        }
        if op == UnOp::Not {
            if let Some(b) = self.as_bool(a) {
                return self.boolean(!b);
            }
        }
        // Involutions: -(-x) and !(!x) collapse to x.
        if let RvNode::Unary(inner_op, inner) = *self.out.node(a) {
            if (op == UnOp::Neg && inner_op == UnOp::Neg)
                || (op == UnOp::Not && inner_op == UnOp::Not)
            {
                return inner;
            }
        }
        self.intern(Key::Unary(op, a), RvNode::Unary(op, a), kind)
    }

    fn binary(&mut self, op: BinOp, l: RvId, r: RvId, kind: RvKind) -> RvId {
        use BinOp::*;
        let (ln, rn) = (self.as_num(l), self.as_num(r));
        // Constant folding over two numeric constants — via the shared scalar kernel (finding F4),
        // so `%`/`^`/comparison semantics can't drift from the VM and the const-fold. Arithmetic
        // yields a `ConstNum`; comparisons yield a `ConstBool` (the kernel returns 0/1, recovered
        // with `!= 0.0`). `And`/`Or` on numbers are never reached (kind-checked upstream).
        if let (Some(a), Some(b)) = (ln, rn) {
            match op {
                Add | Sub | Mul | Div | Mod | Pow => {
                    return self.num(crate::num::fold_binop(op, a, b))
                }
                And | Or => {}
                _ => return self.boolean(crate::num::fold_binop(op, a, b) != 0.0),
            }
        }
        // Constant folding over two boolean constants.
        if let (Some(a), Some(b)) = (self.as_bool(l), self.as_bool(r)) {
            match op {
                And => return self.boolean(a && b),
                Or => return self.boolean(a || b),
                Eq => return self.boolean(a == b),
                Ne => return self.boolean(a != b),
                _ => {}
            }
        }
        // Finite-safe algebraic identities (see module docs — no `x*0`, no `x/x`).
        match op {
            Add if rn == Some(0.0) => return l,
            Add if ln == Some(0.0) => return r,
            Sub if rn == Some(0.0) => return l,
            Mul if rn == Some(1.0) => return l,
            Mul if ln == Some(1.0) => return r,
            Div if rn == Some(1.0) => return l,
            Pow if rn == Some(1.0) => return l,
            Pow if rn == Some(0.0) => return self.num(1.0), // x^0 == 1 (matches powf, incl. inf/nan)
            _ => {}
        }
        self.intern(Key::Binary(op, l, r), RvNode::Binary(op, l, r), kind)
    }

    fn select(&mut self, cond: RvId, a: RvId, b: RvId, kind: RvKind) -> RvId {
        // A constant condition collapses to the taken branch; identical branches collapse too.
        if let Some(c) = self.as_bool(cond) {
            return if c { a } else { b };
        }
        if a == b {
            return a;
        }
        self.intern(Key::Select(cond, a, b), RvNode::Select { cond, a, b }, kind)
    }

    // --- constant inspection of new-graph nodes ---

    fn as_num(&self, id: RvId) -> Option<f64> {
        match self.out.node(id) {
            RvNode::ConstNum(x) => Some(*x),
            _ => None,
        }
    }

    fn as_bool(&self, id: RvId) -> Option<bool> {
        match self.out.node(id) {
            RvNode::ConstBool(b) => Some(*b),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Source;

    fn src(g: &mut RvGraph) -> RvId {
        g.push(
            RvNode::Src(Source::Uniform(crate::dist::Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        )
    }
    fn num(g: &mut RvGraph, x: f64) -> RvId {
        g.push(RvNode::ConstNum(x), RvKind::Num)
    }
    fn bin(g: &mut RvGraph, op: BinOp, a: RvId, b: RvId) -> RvId {
        let k = if matches!(
            op,
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne
        ) {
            RvKind::Bool
        } else {
            RvKind::Num
        };
        g.push(RvNode::Binary(op, a, b), k)
    }

    #[test]
    fn folds_constant_subexpression() {
        // (2 * 3) + X  →  6 + X   (the 2*3 disappears)
        let mut g = RvGraph::default();
        let x = src(&mut g);
        let two = num(&mut g, 2.0);
        let three = num(&mut g, 3.0);
        let prod = bin(&mut g, BinOp::Mul, two, three);
        let root = bin(&mut g, BinOp::Add, prod, x);
        let (out, r) = simplify(&g, root);
        // Root is Add(Const(6), Src); no Mul node remains.
        match out.node(r) {
            RvNode::Binary(BinOp::Add, a, _) => assert_eq!(out.node(*a), &RvNode::ConstNum(6.0)),
            other => panic!("expected Add(6, X), got {other:?}"),
        }
        assert!(!out_has_mul(&out), "the 2*3 multiply should be folded away");
    }

    fn out_has_mul(g: &RvGraph) -> bool {
        (0..g.len() as u32).any(|i| matches!(g.node(RvId(i)), RvNode::Binary(BinOp::Mul, ..)))
    }

    #[test]
    fn folds_constant_exp_and_ln() {
        // exp(1) + X → e + X: the constant Unary(Exp) node folds away (Ln folds symmetrically).
        let mut g = RvGraph::default();
        let x = src(&mut g);
        let one = num(&mut g, 1.0);
        let e = g.push(RvNode::Unary(UnOp::Exp, one), RvKind::Num);
        let lne = g.push(RvNode::Unary(UnOp::Ln, e), RvKind::Num); // ln(exp(1)) folds to 1
        let root = bin(&mut g, BinOp::Add, lne, x);
        let (out, r) = simplify(&g, root);
        match out.node(r) {
            RvNode::Binary(BinOp::Add, a, _) => match out.node(*a) {
                RvNode::ConstNum(v) => assert!((v - 1.0).abs() < 1e-12, "ln(exp(1)) folded to {v}"),
                other => panic!("expected a folded constant, got {other:?}"),
            },
            other => panic!("expected Add(1, X), got {other:?}"),
        }
        let any_unary = (0..out.len() as u32)
            .any(|i| matches!(out.node(RvId(i)), RvNode::Unary(UnOp::Exp | UnOp::Ln, _)));
        assert!(!any_unary, "constant exp/ln nodes must fold away");
    }

    #[test]
    fn applies_identities() {
        // X + 0, X * 1, X ^ 1 all collapse to X (the same source node).
        for (op, c) in [
            (BinOp::Add, 0.0),
            (BinOp::Mul, 1.0),
            (BinOp::Pow, 1.0),
            (BinOp::Div, 1.0),
        ] {
            let mut g = RvGraph::default();
            let x = src(&mut g);
            let c = num(&mut g, c);
            let root = bin(&mut g, op, x, c);
            let (out, r) = simplify(&g, root);
            assert_eq!(
                out.node(r),
                &RvNode::Src(Source::Uniform(crate::dist::Uniform { lo: 0.0, hi: 1.0 })),
                "{op:?} identity should collapse to X"
            );
        }
    }

    #[test]
    fn x_pow_zero_is_one_and_drops_the_draw() {
        // X ^ 0  →  1. The root is the constant; X is no longer reachable from it, so the backend
        // (which lowers only the root cone) never samples it — even though it lingers in the arena.
        let mut g = RvGraph::default();
        let x = src(&mut g);
        let zero = num(&mut g, 0.0);
        let root = bin(&mut g, BinOp::Pow, x, zero);
        let (out, r) = simplify(&g, root);
        assert_eq!(out.node(r), &RvNode::ConstNum(1.0));
        // The root is a leaf constant → nothing reachable, in particular no Src.
        assert!(matches!(out.node(r), RvNode::ConstNum(_)));
    }

    #[test]
    fn cse_merges_identical_subexpressions_but_not_distinct_draws() {
        // (X + Y) compared to itself: the two Add subtrees must merge to ONE node...
        let mut g = RvGraph::default();
        let x = src(&mut g);
        let y = src(&mut g);
        let s1 = bin(&mut g, BinOp::Add, x, y);
        let s2 = bin(&mut g, BinOp::Add, x, y);
        let root = bin(&mut g, BinOp::Eq, s1, s2);
        let (out, r) = simplify(&g, root);
        match out.node(r) {
            RvNode::Binary(BinOp::Eq, a, b) => {
                assert_eq!(a, b, "identical X+Y must CSE to one node")
            }
            other => panic!("expected Eq(s, s), got {other:?}"),
        }
        // ...but the two independent sources X and Y must NOT be merged.
        let n_src = (0..out.len() as u32)
            .filter(|i| matches!(out.node(RvId(*i)), RvNode::Src(_)))
            .count();
        assert_eq!(n_src, 2, "distinct draws must stay distinct");
    }

    /// Reports the cone-size reduction (nodes the backend must lower) on representative graphs.
    /// Ignored; run with: `cargo test -p noise-core simplify -- --ignored --nocapture report_node`
    #[test]
    #[ignore]
    fn report_node_reduction() {
        fn cone(g: &RvGraph, root: RvId) -> usize {
            fn walk(g: &RvGraph, id: RvId, seen: &mut std::collections::HashSet<u32>) {
                if !seen.insert(id.0) {
                    return;
                }
                match g.node(id) {
                    RvNode::Unary(_, a) => walk(g, *a, seen),
                    RvNode::Binary(_, a, b) => {
                        walk(g, *a, seen);
                        walk(g, *b, seen);
                    }
                    RvNode::Select { cond, a, b } => {
                        walk(g, *cond, seen);
                        walk(g, *a, seen);
                        walk(g, *b, seen);
                    }
                    _ => {}
                }
            }
            let mut seen = std::collections::HashSet::new();
            walk(g, root, &mut seen);
            seen.len()
        }

        let cases = [
            (
                "dice_sum",
                "use rand; A ~ unif_int(1,6); B ~ unif_int(1,6); A + B",
            ),
            (
                "pi",
                "use rand; X ~ unif(-1,1); Y ~ unif(-1,1); X^2 + Y^2 < 1",
            ),
            (
                "poly_deep",
                "use rand; X ~ unif(0,1); ((X*X+X)*X - X)*X + X*X - X + 1",
            ),
            // CSE: a subexpression reused several times.
            (
                "cse_reuse",
                "use rand; X ~ unif(0,1); Y ~ unif(0,1); (X+Y)*(X+Y) + (X+Y)*3 - (X+Y)",
            ),
            // Identity-bearing: `* 1`, `+ 0`, `^ 1` that survive to graph nodes.
            (
                "identities",
                "use rand; X ~ unif(0,1); (X * 1 + 0) ^ 1 + X*X",
            ),
        ];
        println!("\n  cone size (nodes the backend lowers): before → after simplify");
        for (name, src) in cases {
            let mut eng = crate::eval::Engine::new();
            let id = match eng.run_rv(src).unwrap() {
                crate::Value::Dist(id) => id,
                _ => continue,
            };
            let before = cone(eng.graph(), id);
            let (out, r) = simplify(eng.graph(), id);
            let after = cone(&out, r);
            println!(
                "    {name:12} {before:3} → {after:3}  (-{})",
                before - after
            );
        }
    }

    #[test]
    fn distinct_sources_with_same_recipe_are_independent() {
        // Two separate unif(0,1) sources must remain two nodes (not CSE'd into one draw).
        let mut g = RvGraph::default();
        let a = src(&mut g);
        let b = src(&mut g);
        let root = bin(&mut g, BinOp::Add, a, b);
        let (out, _) = simplify(&g, root);
        let n_src = (0..out.len() as u32)
            .filter(|i| matches!(out.node(RvId(*i)), RvNode::Src(_)))
            .count();
        assert_eq!(n_src, 2);
    }
}
