//! Graph-level algebraic simplification (PLAN.md Phase 4 "speed pass").
//!
//! A once-per-compile rewrite of the root's cone that **folds constants**, applies a finite-safe
//! set of **algebraic identities**, and **hash-conses** (common-subexpression elimination) — so
//! every backend (interpreter and the codegen emitters alike) lowers a smaller DAG with fewer hot-loop ops and
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

/// Rewrite the union of several roots' cones into ONE fresh, simplified graph — the multi-root
/// twin of [`simplify`], for the joint drivers. A single [`Builder`] (one `done` memo) rewrites
/// every root, so a node feeding more than one root rebuilds to a *single* new node: cross-root
/// sharing — the property that makes a paired statistic (covariance, a scatter point) correct —
/// survives the rewrite. Simplifying each root separately would sever exactly that sharing.
/// Roots are rewritten in input order, so the relative order of surviving sources (hence their
/// RNG consumption) matches the multi-root bytecode lowerer's.
pub fn simplify_roots(graph: &RvGraph, roots: &[RvId]) -> (RvGraph, Vec<RvId>) {
    let mut b = Builder::default();
    let new_roots = roots.iter().map(|&r| b.rewrite(graph, r)).collect();
    (b.out, new_roots)
}

/// Expand every [`RvNode::Scan`] (PLAN-WEBGPU G4c) into flat nodes — the CPU form. Run *before*
/// [`simplify`] in [`crate::backend::compile_root`], so the interpreter and wasm backend lower exactly the
/// DAG eval used to build before this feature existed: the answer and the draw stream are byte-for-
/// byte unchanged whether a loop was captured as a `Scan` or not. The GPU path skips this and rolls
/// the `Scan` into a real WGSL loop instead.
///
/// Returns `None` when there is no `Scan`, so the overwhelmingly common case pays nothing.
///
/// The one correctness rule is that a **loop-invariant** node — above all a source, whose duplication
/// would fork the draw stream — is built **once and shared**, while a node that depends on the loop
/// index or a carried value is rebuilt **per iteration**. [`taint_set`] marks the latter; the
/// `env`-threaded [`Unroller::rebuild`] resolves placeholders and reuses the shared invariants.
pub fn unroll_scans(graph: &RvGraph, root: RvId) -> Option<(RvGraph, RvId)> {
    let has_scan =
        (0..graph.len()).any(|i| matches!(graph.node(RvId(i as u32)), RvNode::Scan { .. }));
    if !has_scan {
        return None;
    }
    let mut u = Unroller {
        graph,
        out: RvGraph::default(),
        taint: taint_set(graph),
        memo: HashMap::new(),
        finals: HashMap::new(),
    };
    let new_root = u.rebuild(root, &HashMap::new());
    Some((u.out, new_root))
}

/// Multi-root [`unroll_scans`] for the joint drivers: one [`Unroller`] (shared invariant memo) over
/// every root, so a source feeding two roots stays a single node. `None` when there is no `Scan`.
pub fn unroll_scans_roots(graph: &RvGraph, roots: &[RvId]) -> Option<(RvGraph, Vec<RvId>)> {
    let has_scan =
        (0..graph.len()).any(|i| matches!(graph.node(RvId(i as u32)), RvNode::Scan { .. }));
    if !has_scan {
        return None;
    }
    let mut u = Unroller {
        graph,
        out: RvGraph::default(),
        taint: taint_set(graph),
        memo: HashMap::new(),
        finals: HashMap::new(),
    };
    let empty = HashMap::new();
    let new_roots = roots.iter().map(|&r| u.rebuild(r, &empty)).collect();
    Some((u.out, new_roots))
}

/// Ids whose value depends on a loop [`Placeholder`](RvNode::Placeholder) — a carried slot or the
/// iteration counter — and so must be rebuilt *per iteration* rather than shared. Computed in one
/// ascending pass because a node's operands always have smaller ids (append-only arena).
pub(crate) fn taint_set(graph: &RvGraph) -> std::collections::HashSet<RvId> {
    let mut taint = std::collections::HashSet::new();
    let t = |taint: &std::collections::HashSet<RvId>, id: &RvId| taint.contains(id);
    for i in 0..graph.len() {
        let id = RvId(i as u32);
        let tainted = match graph.node(id) {
            RvNode::Placeholder { .. } => true,
            // A loop is tainted (its result depends on outer state) iff a carried init is.
            RvNode::Scan { body } => body.inits.iter().any(|x| t(&taint, x)),
            RvNode::ScanOut { scan, .. } => t(&taint, scan),
            RvNode::Src(_)
            | RvNode::ConstNum(_)
            | RvNode::ConstBool(_)
            | RvNode::Input { .. }
            | RvNode::Permutation { .. }
            | RvNode::Rotation { .. }
            | RvNode::ArrDraw { .. } => false,
            RvNode::ArrElem { arr, .. } => t(&taint, arr),
            RvNode::Unary(_, a) => t(&taint, a),
            RvNode::Binary(_, a, b) => t(&taint, a) || t(&taint, b),
            RvNode::Select { cond, a, b } => t(&taint, cond) || t(&taint, a) || t(&taint, b),
            RvNode::Gather { elems, index } => {
                t(&taint, index) || elems.iter().any(|e| t(&taint, e))
            }
            RvNode::ArrIndex { arr, index } => t(&taint, arr) || t(&taint, index),
        };
        if tainted {
            taint.insert(id);
        }
    }
    taint
}

struct Unroller<'g> {
    graph: &'g RvGraph,
    out: RvGraph,
    taint: std::collections::HashSet<RvId>,
    /// Invariant old-id → new-id: built once, shared across iterations (sources must not duplicate).
    memo: HashMap<RvId, RvId>,
    /// Cached final slot values of a *non-tainted* (loop-independent) Scan, so its several `ScanOut`
    /// readers expand it once, not once per slot.
    finals: HashMap<RvId, Vec<RvId>>,
}

impl Unroller<'_> {
    /// Rebuild `id` into the output graph under the current placeholder bindings `env`. Tainted nodes
    /// (loop-dependent) are rebuilt fresh; invariants are memoized and shared.
    fn rebuild(&mut self, id: RvId, env: &HashMap<RvId, RvId>) -> RvId {
        if let Some(&n) = env.get(&id) {
            return n;
        }
        if let Some(&n) = self.memo.get(&id) {
            return n;
        }
        let kind = self.graph.kind(id);
        let new = match self.graph.node(id).clone() {
            RvNode::Placeholder { .. } => {
                unreachable!("a placeholder must be bound by its enclosing Scan's env")
            }
            RvNode::Scan { .. } => unreachable!("a Scan is reached only through its ScanOut"),
            RvNode::ScanOut { scan, slot } => self.scan_finals(scan, env)[slot as usize],
            RvNode::Src(s) => self.out.push(RvNode::Src(s), kind),
            RvNode::ConstNum(x) => self.out.push(RvNode::ConstNum(x), kind),
            RvNode::ConstBool(b) => self.out.push(RvNode::ConstBool(b), kind),
            RvNode::Input { idx } => self.out.push(RvNode::Input { idx }, kind),
            RvNode::Permutation { n } => self.out.push(RvNode::Permutation { n }, kind),
            RvNode::Rotation { d } => self.out.push(RvNode::Rotation { d }, kind),
            RvNode::ArrDraw { n, src } => self.out.push(RvNode::ArrDraw { n, src }, kind),
            RvNode::ArrElem { arr, k } => {
                let arr = self.rebuild(arr, env);
                self.out.push(RvNode::ArrElem { arr, k }, kind)
            }
            RvNode::Unary(op, a) => {
                let a = self.rebuild(a, env);
                self.out.push(RvNode::Unary(op, a), kind)
            }
            RvNode::Binary(op, a, b) => {
                let a = self.rebuild(a, env);
                let b = self.rebuild(b, env);
                self.out.push(RvNode::Binary(op, a, b), kind)
            }
            RvNode::Select { cond, a, b } => {
                let cond = self.rebuild(cond, env);
                let a = self.rebuild(a, env);
                let b = self.rebuild(b, env);
                self.out.push(RvNode::Select { cond, a, b }, kind)
            }
            RvNode::Gather { elems, index } => {
                let elems: Box<[RvId]> = elems.iter().map(|&e| self.rebuild(e, env)).collect();
                let index = self.rebuild(index, env);
                self.out.push(RvNode::Gather { elems, index }, kind)
            }
            RvNode::ArrIndex { arr, index } => {
                let arr = self.rebuild(arr, env);
                let index = self.rebuild(index, env);
                self.out.push(RvNode::ArrIndex { arr, index }, kind)
            }
        };
        if !self.taint.contains(&id) {
            self.memo.insert(id, new);
        }
        new
    }

    /// The final carried values of a Scan after `trip` iterations. A loop-independent (non-tainted)
    /// Scan is expanded once and cached; a loop-dependent one is expanded fresh under the caller's env.
    fn scan_finals(&mut self, scan_id: RvId, env: &HashMap<RvId, RvId>) -> Vec<RvId> {
        if !self.taint.contains(&scan_id) {
            if let Some(f) = self.finals.get(&scan_id) {
                return f.clone();
            }
        }
        let body = match self.graph.node(scan_id).clone() {
            RvNode::Scan { body } => body,
            _ => unreachable!("scan_finals on a non-Scan"),
        };
        // Slot values before iteration 0, in the caller's env (a carried init may itself be tainted,
        // e.g. an inner loop starting from the outer loop's index).
        let mut cur: Vec<RvId> = body.inits.iter().map(|&i| self.rebuild(i, env)).collect();
        for k in 0..body.trip {
            let mut it_env = env.clone();
            for (slot, &ph) in body.placeholders.iter().enumerate() {
                it_env.insert(ph, cur[slot]);
            }
            if let Some(iph) = body.index_ph {
                let c = self.out.push(RvNode::ConstNum(f64::from(k)), RvKind::Num);
                it_env.insert(iph, c);
            }
            cur = body
                .nexts
                .iter()
                .map(|&nx| self.rebuild(nx, &it_env))
                .collect();
        }
        if !self.taint.contains(&scan_id) {
            self.finals.insert(scan_id, cur.clone());
        }
        cur
    }
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
    /// A host input uniform, deduped by slot index — two `Input { idx }` leaves are the same value.
    Input(u32),
    /// Element `k` of a shaped draw. Pure given `(arr, k)` — the parent `ArrDraw` is never interned,
    /// so two independent `~[n]` draws have different `arr` ids and can never collide here.
    ArrElem(RvId, u32),
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
                        RvNode::Src(_)
                        | RvNode::ConstNum(_)
                        | RvNode::ConstBool(_)
                        | RvNode::Input { .. }
                        | RvNode::Permutation { .. }
                        | RvNode::Rotation { .. }
                        | RvNode::ArrDraw { .. } => {}
                        RvNode::ArrElem { arr, .. } => push_child(*arr),
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
                        RvNode::ArrIndex { arr, index } => {
                            push_child(*index);
                            push_child(*arr);
                        }
                        // A Scan's operands, all in this graph: the carried inits, and the body cone
                        // (its `nexts`, which reach the placeholders and loop-invariant nodes). All
                        // are simplified so the rebuilt Scan points into the new graph.
                        RvNode::Scan { body } => {
                            for &init in body.inits.iter() {
                                push_child(init);
                            }
                            for &ph in body.placeholders.iter() {
                                push_child(ph);
                            }
                            if let Some(ph) = body.index_ph {
                                push_child(ph);
                            }
                            for &nx in body.nexts.iter() {
                                push_child(nx);
                            }
                        }
                        RvNode::ScanOut { scan, .. } => push_child(*scan),
                        RvNode::Placeholder { .. } => {}
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
                        // A whole-array draw is a SOURCE too: copied 1:1, never interned, so two
                        // `permutation(n)` draws stay independent permutations.
                        RvNode::Permutation { n } => {
                            self.out.push(RvNode::Permutation { n: *n }, kind)
                        }
                        // Same for the whole-matrix Haar draw: two `rotation(d)` draws must stay
                        // independent rotations.
                        RvNode::Rotation { d } => self.out.push(RvNode::Rotation { d: *d }, kind),
                        RvNode::ConstNum(x) => self.num(*x),
                        RvNode::ConstBool(b) => self.boolean(*b),
                        // A uniform input: interned by idx (deduped), NEVER folded to a constant —
                        // the whole point is that the value is not baked (PLAN-UNIFORM-INPUTS).
                        RvNode::Input { idx } => self.input(*idx),
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
                        // ArrIndex: rebuild over the simplified operands. Not interned, mirroring
                        // Gather — reads rarely coincide and a wrong merge would corrupt selection.
                        RvNode::ArrIndex { arr, index } => {
                            let (arr, index) = (self.done[arr], self.done[index]);
                            self.out.push(RvNode::ArrIndex { arr, index }, kind)
                        }
                        // A shaped draw is a SOURCE: copied 1:1, never interned, so two `~[n]`
                        // draws of the same recipe stay independent (exactly the rule that keeps
                        // two `Src`s of one recipe independent).
                        RvNode::ArrDraw { n, src } => {
                            self.out.push(RvNode::ArrDraw { n: *n, src: *src }, kind)
                        }
                        // But an element READ is deterministic given `(arr, k)`, so it interns like
                        // any pure node: `zs[3] + zs[3]` is one draw doubled — which is what it was
                        // when `zs[3]` was a plain `Src` handle shared by both operands.
                        RvNode::ArrElem { arr, k } => {
                            let arr = self.done[arr];
                            self.arr_elem(arr, *k, kind)
                        }
                        // A Scan is a loop-shaped SOURCE-like node: remap every carried id (inits,
                        // placeholders, nexts, index) into the simplified graph. Not interned — two
                        // loops rarely coincide and a wrong merge would fuse distinct recurrences.
                        RvNode::Scan { body } => {
                            let map = |ids: &[RvId]| -> Box<[RvId]> {
                                ids.iter().map(|i| self.done[i]).collect()
                            };
                            let new_body = crate::dist::ScanBody {
                                trip: body.trip,
                                placeholders: map(&body.placeholders),
                                inits: map(&body.inits),
                                nexts: map(&body.nexts),
                                index_ph: body.index_ph.map(|p| self.done[&p]),
                                kinds: body.kinds.clone(),
                            };
                            self.out.push(
                                RvNode::Scan {
                                    body: Box::new(new_body),
                                },
                                kind,
                            )
                        }
                        RvNode::ScanOut { scan, slot } => {
                            let scan = self.done[scan];
                            self.out.push(RvNode::ScanOut { scan, slot: *slot }, kind)
                        }
                        // A placeholder keeps its identity across simplification (two slots are
                        // distinct); copied as a fresh node, never interned.
                        RvNode::Placeholder { slot } => {
                            self.out.push(RvNode::Placeholder { slot: *slot }, kind)
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

    fn input(&mut self, idx: u32) -> RvId {
        self.intern(Key::Input(idx), RvNode::Input { idx }, RvKind::Num)
    }

    fn arr_elem(&mut self, arr: RvId, k: u32, kind: RvKind) -> RvId {
        self.intern(Key::ArrElem(arr, k), RvNode::ArrElem { arr, k }, kind)
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
                UnOp::Sqrt => return self.num(x.sqrt()),
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
            // `x ^ 0.5 → sqrt(x)` — a pow libcall becomes one native instruction — but ONLY when
            // the base is provably a square/exponential, i.e. its domain is {NaN, +0.0, positive,
            // +inf}. There powf and sqrt agree; on the two excluded inputs they do NOT
            // (`powf(-0.0, 0.5) = +0.0` vs `sqrt(-0.0) = -0.0` — a *finite* lane, observable via
            // `1/x`; `powf(-inf, 0.5) = +inf` vs `sqrt(-inf) = NaN`), so an unconditional rewrite
            // would break this module's exact-for-all-draws charter. General `x ^ 0.5` keeps C99
            // powf semantics; `math::sqrt`/`vec::norm`/complex `abs` build `UnOp::Sqrt` directly
            // at eval time and don't rely on this rewrite (PLAN-PERF-2 §5).
            Pow if rn == Some(0.5) && self.nonneg_base(l) => {
                return self.unary(UnOp::Sqrt, l, kind)
            }
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

    /// Structural proof (in the new graph) that a node's per-lane value can only be in
    /// {NaN, +0.0, positive, +inf} — the domain where `powf(x, 0.5)` and `sqrt(x)` agree exactly,
    /// which is what licenses the `x ^ 0.5 → sqrt(x)` rewrite in [`Self::binary`]:
    ///   * `a * a` with **the same id** on both sides (hash-consing makes a user's `x*x` share one
    ///     id, and one id = one draw): `(±0)² = +0.0`, finite² ≥ 0 (or +inf on overflow),
    ///     `(±inf)² = +inf`, NaN propagates. Never `-0.0`, never `-inf`.
    ///   * `exp(_)`: range is [+0.0, +inf] ∪ {NaN} by construction.
    ///
    /// Deliberately NOT `sqrt(_)` (its own output includes `-0.0`) and not a general sign
    /// analysis — a conservative allowlist keeps the charter auditable.
    fn nonneg_base(&self, id: RvId) -> bool {
        match self.out.node(id) {
            RvNode::Binary(BinOp::Mul, a, b) => a == b,
            RvNode::Unary(UnOp::Exp, _) => true,
            _ => false,
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

    /// PLAN-UNIFORM-INPUTS' load-bearing invariant: an `Input` uniform is **opaque to constant
    /// folding**. `Input * 2` must stay `Binary(Mul, Input, Const(2))` — folding it back to a
    /// `ConstNum` would re-bake the input value and defeat the whole feature. (A deduping intern and
    /// the sound identities like `Input + 0 → Input` are fine; only value-folding is forbidden, and
    /// it's forbidden by construction because `as_num` never recognizes an `Input`.)
    #[test]
    fn input_is_opaque_to_constant_folding() {
        let mut g = RvGraph::default();
        let inp = g.push(RvNode::Input { idx: 0 }, RvKind::Num);
        let two = num(&mut g, 2.0);
        let root = bin(&mut g, BinOp::Mul, inp, two);
        let (out, r) = simplify(&g, root);
        match out.node(r) {
            RvNode::Binary(BinOp::Mul, a, b) => {
                assert_eq!(
                    out.node(*a),
                    &RvNode::Input { idx: 0 },
                    "the Input leaf must survive"
                );
                assert_eq!(out.node(*b), &RvNode::ConstNum(2.0));
            }
            other => panic!("Input*2 must stay a Mul over the Input, got {other:?}"),
        }
        // No ConstNum ever stands in for the Input (that would be a re-bake).
        let input_count = (0..out.len() as u32)
            .filter(|&i| matches!(out.node(RvId(i)), RvNode::Input { idx: 0 }))
            .count();
        assert_eq!(
            input_count, 1,
            "exactly one Input leaf, deduped, never folded"
        );
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
    fn pow_half_becomes_sqrt_only_for_provably_nonneg_bases() {
        // (X * X) ^ 0.5 → sqrt(X * X): the base is a square (never -0.0 / -inf), so powf and
        // sqrt agree on its whole domain and the rewrite is exact.
        let mut g = RvGraph::default();
        let x = src(&mut g);
        let sq = bin(&mut g, BinOp::Mul, x, x);
        let half = num(&mut g, 0.5);
        let root = bin(&mut g, BinOp::Pow, sq, half);
        let (out, r) = simplify(&g, root);
        assert!(
            matches!(out.node(r), RvNode::Unary(UnOp::Sqrt, _)),
            "square base: expected Sqrt, got {:?}",
            out.node(r)
        );
        // Plain X ^ 0.5 must stay a Pow: X could draw -0.0 or -inf, where powf(x, 0.5) and
        // sqrt(x) disagree (+0.0 vs -0.0; +inf vs NaN) — the rewrite would change semantics.
        let mut g = RvGraph::default();
        let x = src(&mut g);
        let half = num(&mut g, 0.5);
        let root = bin(&mut g, BinOp::Pow, x, half);
        let (out, r) = simplify(&g, root);
        assert!(
            matches!(out.node(r), RvNode::Binary(BinOp::Pow, ..)),
            "unproven base: expected Pow to survive, got {:?}",
            out.node(r)
        );
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
    fn permutation_draws_are_never_merged() {
        // Two structurally-identical `Permutation` draws are independent random variables (like
        // any Src) and must survive as TWO nodes; their element reads (`ArrIndex`, also identical
        // in shape once the const indices intern to one node) must not be CSE'd either.
        let mut g = RvGraph::default();
        let p1 = g.push(RvNode::Permutation { n: 4 }, RvKind::Arr(4));
        let p2 = g.push(RvNode::Permutation { n: 4 }, RvKind::Arr(4));
        let c1 = num(&mut g, 0.0);
        let c2 = num(&mut g, 0.0);
        let e1 = g.push(RvNode::ArrIndex { arr: p1, index: c1 }, RvKind::Num);
        let e2 = g.push(RvNode::ArrIndex { arr: p2, index: c2 }, RvKind::Num);
        let root = bin(&mut g, BinOp::Eq, e1, e2);
        let (out, r) = simplify(&g, root);
        let n_perm = (0..out.len() as u32)
            .filter(|i| matches!(out.node(RvId(*i)), RvNode::Permutation { .. }))
            .count();
        assert_eq!(
            n_perm, 2,
            "independent permutation draws must stay distinct"
        );
        match out.node(r) {
            RvNode::Binary(BinOp::Eq, a, b) => {
                assert_ne!(a, b, "element reads of distinct draws must not merge")
            }
            other => panic!("expected Eq(e1, e2), got {other:?}"),
        }
    }

    #[test]
    fn rotation_draws_are_never_merged() {
        // Mirror of `permutation_draws_are_never_merged` for the whole-matrix Haar draw: two
        // structurally-identical `Rotation` sources must survive as TWO nodes (independent
        // rotations), and their same-index element reads must not be CSE'd into one.
        let mut g = RvGraph::default();
        let r1 = g.push(RvNode::Rotation { d: 3 }, RvKind::Arr(9));
        let r2 = g.push(RvNode::Rotation { d: 3 }, RvKind::Arr(9));
        let c1 = num(&mut g, 0.0);
        let c2 = num(&mut g, 0.0);
        let e1 = g.push(RvNode::ArrIndex { arr: r1, index: c1 }, RvKind::Num);
        let e2 = g.push(RvNode::ArrIndex { arr: r2, index: c2 }, RvKind::Num);
        let root = bin(&mut g, BinOp::Eq, e1, e2);
        let (out, r) = simplify(&g, root);
        let n_rot = (0..out.len() as u32)
            .filter(|i| matches!(out.node(RvId(*i)), RvNode::Rotation { .. }))
            .count();
        assert_eq!(n_rot, 2, "independent rotation draws must stay distinct");
        match out.node(r) {
            RvNode::Binary(BinOp::Eq, a, b) => {
                assert_ne!(a, b, "element reads of distinct draws must not merge")
            }
            other => panic!("expected Eq(e1, e2), got {other:?}"),
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
