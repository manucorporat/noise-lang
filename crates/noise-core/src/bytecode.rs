//! Lowering (sample-DAG → flat bytecode with CSE) and the columnar VM (PLAN.md
//! "Performance thesis").
//!
//! The RV expression compiles ONCE to a flat list of [`Inst`] over column registers. A
//! register is a contiguous `[f64; BATCH]` buffer. `run_batch` walks the instruction list,
//! filling each instruction's whole `dst` column before moving on — so one pass evaluates
//! `BATCH` draws, never tree-walking per draw. CSE via `HashMap<RvId, Reg>` guarantees a
//! shared sub-RV (e.g. `X` in `X + X`) compiles to ONE register/instruction, so both lanes
//! read the SAME per-batch draw.

use std::collections::HashMap;

use crate::ast::{BinOp, UnOp};
use crate::dist::{RvGraph, RvId, RvNode, Source};

/// Batch / column width. 8 KiB per f64 column: small enough to stay in cache, large enough
/// to amortize dispatch and give the autovectorizer long runs. Tunable in Phase 4.
pub const BATCH: usize = 1024;

/// Register index into the column file.
pub type Reg = u32;

#[derive(Debug, Clone, Copy)]
pub enum Inst {
    Uniform {
        dst: Reg,
        lo: f64,
        hi: f64,
    },
    UniformInt {
        dst: Reg,
        lo: f64,
        hi: f64,
    },
    Normal {
        dst: Reg,
        mu: f64,
        sigma: f64,
    },
    Exp {
        dst: Reg,
        rate: f64,
    },
    Poisson {
        dst: Reg,
        lambda: f64,
    },
    Geometric {
        dst: Reg,
        p: f64,
    },
    ConstNum {
        dst: Reg,
        val: f64,
    },
    ConstBool {
        dst: Reg,
        val: f64,
    }, // 0.0 / 1.0
    Un {
        dst: Reg,
        op: UnOp,
        a: Reg,
    },
    Bin {
        dst: Reg,
        op: BinOp,
        a: Reg,
        b: Reg,
    },
    /// Per-lane select: `dst = cond ? a : b` (lifted `if`).
    Select {
        dst: Reg,
        cond: Reg,
        a: Reg,
        b: Reg,
    },
    /// Per-lane gather: `dst = table[round(clamp(index))]`. `table` indexes `Program::gathers`
    /// (the element registers); `index` is the register holding the per-lane index. Kept out of
    /// `Inst` itself so the enum stays `Copy`.
    Gather {
        dst: Reg,
        table: u32,
        index: Reg,
    },
}

pub struct Program {
    pub insts: Vec<Inst>,
    pub n_regs: usize,
    pub root: Reg,
    /// Element-register tables for `Inst::Gather`, indexed by its `table` field.
    pub gathers: Vec<Box<[Reg]>>,
}

/// Lower the transitive cone of `root` to flat bytecode.
///
/// Post-order DFS with a `HashMap<RvId, Reg>` memo: a shared `RvId` compiles to ONE register
/// and ONE instruction (CSE). First cut allocates one register per distinct node, no reuse —
/// register-liveness reuse is deferred to Phase 4 (BATCH×n_regs memory is fine here).
pub fn compile(graph: &RvGraph, root: RvId) -> Program {
    let mut memo: HashMap<RvId, Reg> = HashMap::new();
    let mut insts: Vec<Inst> = Vec::new();
    let mut gathers: Vec<Box<[Reg]>> = Vec::new();
    let root_reg = lower(graph, root, &mut memo, &mut insts, &mut gathers);
    Program {
        n_regs: insts.len(),
        insts,
        root: root_reg,
        gathers,
    }
}

/// Lower several roots into ONE shared instruction stream (a single `lower` memo), then return the
/// program plus the register holding each root, in input order. The shared memo is the whole point:
/// any source feeding more than one root compiles to a *single* instruction, so every root reads the
/// **same** per-lane draw of it — i.e. the roots are sampled *jointly*. This is what makes a paired
/// statistic (covariance, correlation, a scatter point) correct: two separately-compiled roots would
/// place their shared sources at different stream positions and so would not pair lane-for-lane (the
/// same joint-sampling requirement as conditioning). `Program::root` is set to the first root (a
/// don't-care for multi-root reads, which index `regs` directly). Like [`compile`], this lowers the
/// raw graph (no simplify pass) so cross-root source sharing is preserved verbatim.
pub fn compile_roots(graph: &RvGraph, roots: &[RvId]) -> (Program, Vec<Reg>) {
    let mut memo: HashMap<RvId, Reg> = HashMap::new();
    let mut insts: Vec<Inst> = Vec::new();
    let mut gathers: Vec<Box<[Reg]>> = Vec::new();
    let regs: Vec<Reg> = roots
        .iter()
        .map(|&r| lower(graph, r, &mut memo, &mut insts, &mut gathers))
        .collect();
    let root = regs.first().copied().unwrap_or(0);
    (
        Program {
            n_regs: insts.len(),
            insts,
            root,
            gathers,
        },
        regs,
    )
}

/// A worklist item for the iterative post-order lowering (below).
enum Task {
    /// First visit: schedule this node's emission after its children.
    Visit(RvId),
    /// Second visit: all children are lowered (in `memo`); emit this node's instruction.
    Emit(RvId),
}

/// Lower `id`'s cone into `insts`, memoizing each `RvId` → `Reg` (CSE).
///
/// **Iterative** post-order DFS with an explicit `Task` worklist, *not* recursion: a graph can be
/// hundreds of thousands of nodes deep (`cumsum(~[200000] noise_white(1))` builds a 200k-deep `Add`
/// chain), which would overflow a recursive lowerer's call stack and abort (finding A4). The
/// worklist models the same post-order — children emit before their parent, left operand before
/// right — so register numbering and instruction order are identical to the old recursive lowerer.
fn lower(
    graph: &RvGraph,
    id: RvId,
    memo: &mut HashMap<RvId, Reg>,
    insts: &mut Vec<Inst>,
    gathers: &mut Vec<Box<[Reg]>>,
) -> Reg {
    if let Some(&reg) = memo.get(&id) {
        return reg;
    }
    let mut stack = vec![Task::Visit(id)];
    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(id) => {
                if memo.contains_key(&id) {
                    continue;
                }
                // Emit `id` only after its children; push children in reverse so they pop (and so
                // emit) in operand order — matching the recursive lowerer's register assignment.
                stack.push(Task::Emit(id));
                let push_child = |stack: &mut Vec<Task>, c: RvId| {
                    if !memo.contains_key(&c) {
                        stack.push(Task::Visit(c));
                    }
                };
                match graph.node(id) {
                    RvNode::Src(_) | RvNode::ConstNum(_) | RvNode::ConstBool(_) => {}
                    RvNode::Unary(_, a) => push_child(&mut stack, *a),
                    RvNode::Binary(_, a, b) => {
                        push_child(&mut stack, *b);
                        push_child(&mut stack, *a);
                    }
                    RvNode::Select { cond, a, b } => {
                        push_child(&mut stack, *b);
                        push_child(&mut stack, *a);
                        push_child(&mut stack, *cond);
                    }
                    RvNode::Gather { elems, index } => {
                        push_child(&mut stack, *index);
                        for &e in elems.iter().rev() {
                            push_child(&mut stack, e);
                        }
                    }
                }
            }
            Task::Emit(id) => {
                if memo.contains_key(&id) {
                    continue; // reached via another path already
                }
                // Checked cast (finding B7): a truncating `as Reg` past 2³² instructions would alias
                // an unrelated register. Compile-time path (not per-lane), so `try_from` is free.
                let dst = Reg::try_from(insts.len()).expect("bytecode exceeded 2^32 instructions");
                match graph.node(id).clone() {
                    RvNode::Src(Source::Uniform(u)) => {
                        insts.push(Inst::Uniform {
                            dst,
                            lo: u.lo,
                            hi: u.hi,
                        });
                    }
                    RvNode::Src(Source::UniformInt { lo, hi }) => {
                        insts.push(Inst::UniformInt { dst, lo, hi });
                    }
                    RvNode::Src(Source::Normal { mu, sigma }) => {
                        insts.push(Inst::Normal { dst, mu, sigma });
                    }
                    RvNode::Src(Source::Exp { rate }) => insts.push(Inst::Exp { dst, rate }),
                    RvNode::Src(Source::Poisson { lambda }) => {
                        insts.push(Inst::Poisson { dst, lambda });
                    }
                    RvNode::Src(Source::Geometric { p }) => insts.push(Inst::Geometric { dst, p }),
                    RvNode::ConstNum(v) => insts.push(Inst::ConstNum { dst, val: v }),
                    RvNode::ConstBool(b) => insts.push(Inst::ConstBool {
                        dst,
                        val: if b { 1.0 } else { 0.0 },
                    }),
                    RvNode::Unary(op, a) => {
                        let ra = memo[&a];
                        insts.push(Inst::Un { dst, op, a: ra });
                    }
                    RvNode::Binary(op, a, b) => {
                        let (ra, rb) = (memo[&a], memo[&b]);
                        insts.push(Inst::Bin {
                            dst,
                            op,
                            a: ra,
                            b: rb,
                        });
                    }
                    RvNode::Select { cond, a, b } => {
                        let (rc, ra, rb) = (memo[&cond], memo[&a], memo[&b]);
                        insts.push(Inst::Select {
                            dst,
                            cond: rc,
                            a: ra,
                            b: rb,
                        });
                    }
                    RvNode::Gather { elems, index } => {
                        let table: Vec<Reg> = elems.iter().map(|e| memo[e]).collect();
                        let ri = memo[&index];
                        // Checked cast (finding B7): the gather-table index must not truncate.
                        let tbl = u32::try_from(gathers.len())
                            .expect("bytecode exceeded 2^32 gather tables");
                        gathers.push(table.into_boxed_slice());
                        insts.push(Inst::Gather {
                            dst,
                            table: tbl,
                            index: ri,
                        });
                    }
                }
                memo.insert(id, dst);
            }
        }
    }
    memo[&id]
}

/// Run one batch: fill every register column for `BATCH` lanes by walking the instructions.
///
/// Because first-cut allocation is one-register-per-node, `dst` is always distinct from
/// `a`/`b`. We borrow per-iteration scalars (not slices) so the borrow checker is satisfied
/// without splitting the register vector.
pub fn run_batch(program: &Program, regs: &mut [Box<[f64]>], rng: &mut crate::rng::Rng) {
    for inst in &program.insts {
        match *inst {
            Inst::Uniform { dst, lo, hi } => {
                rng.fill_uniform(lo, hi, &mut regs[dst as usize]);
            }
            Inst::UniformInt { dst, lo, hi } => {
                rng.fill_uniform_int(lo, hi, &mut regs[dst as usize]);
            }
            Inst::Normal { dst, mu, sigma } => {
                rng.fill_normal(mu, sigma, &mut regs[dst as usize]);
            }
            Inst::Exp { dst, rate } => {
                rng.fill_exp(rate, &mut regs[dst as usize]);
            }
            Inst::Poisson { dst, lambda } => {
                rng.fill_poisson(lambda, &mut regs[dst as usize]);
            }
            Inst::Geometric { dst, p } => {
                rng.fill_geometric(p, &mut regs[dst as usize]);
            }
            Inst::ConstNum { dst, val } => {
                for x in regs[dst as usize].iter_mut() {
                    *x = val;
                }
            }
            Inst::ConstBool { dst, val } => {
                for x in regs[dst as usize].iter_mut() {
                    *x = val;
                }
            }
            Inst::Un { dst, op, a } => {
                let (dst, a) = (dst as usize, a as usize);
                for k in 0..BATCH {
                    let x = regs[a][k];
                    regs[dst][k] = apply_un(op, x);
                }
            }
            Inst::Bin { dst, op, a, b } => {
                let (dst, a, b) = (dst as usize, a as usize, b as usize);
                for k in 0..BATCH {
                    let xa = regs[a][k];
                    let xb = regs[b][k];
                    regs[dst][k] = apply_bin(op, xa, xb);
                }
            }
            Inst::Select { dst, cond, a, b } => {
                let (dst, cond, a, b) = (dst as usize, cond as usize, a as usize, b as usize);
                for k in 0..BATCH {
                    regs[dst][k] = if regs[cond][k] != 0.0 {
                        regs[a][k]
                    } else {
                        regs[b][k]
                    };
                }
            }
            Inst::Gather { dst, table, index } => {
                // Per lane: round the index, clamp into `0..len`, copy that element's lane value.
                // Gather to a scratch column first so the immutable element reads don't alias the
                // mutable `dst` write (one-register-per-node guarantees they're distinct anyway).
                //
                // NaN index → NaN result (finding B5). A NaN index selects no element, so the honest
                // answer is NaN — propagating it the way every other IEEE op here does. This is a
                // SEMANTIC CHOICE: the previous code let `NaN as usize == 0` silently read element 0,
                // fabricating a real value from an undefined index. (±inf still clamp to the ends:
                // `-inf` below `0`, `+inf` at/above `last`, which is the sensible saturating read.)
                let tbl = &program.gathers[table as usize];
                let last = tbl.len() - 1; // `gathers` never holds an empty table (eval rejects it)
                let index = index as usize;
                let mut scratch = [0.0f64; BATCH];
                for k in 0..BATCH {
                    let raw = regs[index][k].round();
                    if raw.is_nan() {
                        scratch[k] = f64::NAN;
                        continue;
                    }
                    let i = if raw <= 0.0 {
                        0
                    } else if raw as usize >= last {
                        last
                    } else {
                        raw as usize
                    };
                    scratch[k] = regs[tbl[i] as usize][k];
                }
                regs[dst as usize].copy_from_slice(&scratch);
            }
        }
    }
}

/// Scalar unary op. `Not` is logical not over a 0/1 bool column.
#[inline]
fn apply_un(op: UnOp, x: f64) -> f64 {
    match op {
        UnOp::Neg => -x,
        UnOp::Not => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
        UnOp::Sin => x.sin(),
        UnOp::Cos => x.cos(),
        UnOp::Atan => x.atan(),
        // sign: -1 / 0 / +1 (0 at exactly zero, unlike f64::signum which is ±1 at ±0.0).
        UnOp::Sign => (x > 0.0) as i32 as f64 - (x < 0.0) as i32 as f64,
        UnOp::Round => x.round(),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        // The interpreter is the exact oracle: full-precision libm `exp`/`ln` (IEEE semantics —
        // ln(0) = -inf, ln(x<0) = NaN). The code generators approximate these within MC noise.
        UnOp::Exp => x.exp(),
        UnOp::Ln => x.ln(),
    }
}

/// Scalar binary op. Matches the deterministic evaluator's IEEE-754 behavior; comparisons
/// produce 0.0/1.0 columns (the bool convention from PLAN.md, pre-wiring Phase 3's `P()`).
#[inline]
fn apply_bin(op: BinOp, a: f64, b: f64) -> f64 {
    // The single shared scalar kernel (finding F4) — same computation the signal folder, the
    // `eval` constant-fold, and the graph simplifier use, so the VM can never drift from them.
    crate::num::fold_binop(op, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{RvGraph, RvId, RvKind, RvNode, Source, Uniform};

    #[test]
    fn cse_shares_a_repeated_subexpression() {
        // X + X: the shared `X` must compile to ONE register, so total = 2 (X, the Add) not 3.
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let sum = g.push(RvNode::Binary(BinOp::Add, x, x), RvKind::Num);
        let prog = compile(&g, sum);
        assert_eq!(prog.n_regs, 2, "X must be shared (CSE), not duplicated");
    }

    #[test]
    fn comparison_and_logical_kernels_yield_0_or_1() {
        assert_eq!(apply_bin(BinOp::Lt, 1.0, 2.0), 1.0);
        assert_eq!(apply_bin(BinOp::Lt, 2.0, 1.0), 0.0);
        assert_eq!(apply_bin(BinOp::Eq, 3.0, 3.0), 1.0);
        assert_eq!(apply_bin(BinOp::Ne, 3.0, 3.0), 0.0);
        assert_eq!(apply_bin(BinOp::And, 1.0, 0.0), 0.0);
        assert_eq!(apply_bin(BinOp::And, 1.0, 1.0), 1.0);
        assert_eq!(apply_bin(BinOp::Or, 0.0, 0.0), 0.0);
        assert_eq!(apply_bin(BinOp::Or, 1.0, 0.0), 1.0);
    }

    #[test]
    fn arithmetic_kernels_match_ieee() {
        assert_eq!(apply_bin(BinOp::Add, 2.0, 3.0), 5.0);
        assert_eq!(apply_bin(BinOp::Pow, 2.0, 10.0), 1024.0);
        assert_eq!(apply_un(UnOp::Neg, 4.0), -4.0);
        // Not is logical over a 0/1 column.
        assert_eq!(apply_un(UnOp::Not, 0.0), 1.0);
        assert_eq!(apply_un(UnOp::Not, 1.0), 0.0);
    }

    #[test]
    fn gather_with_a_nan_index_yields_nan() {
        // Finding B5 (SEMANTIC CHOICE): a NaN per-lane index propagates NaN, rather than silently
        // reading element 0 (`NaN as usize == 0`). A finite index still selects; ±inf saturates.
        let mut g = RvGraph::default();
        let elems: Vec<RvId> = [10.0, 20.0, 30.0]
            .iter()
            .map(|&v| g.push(RvNode::ConstNum(v), RvKind::Num))
            .collect();
        // A per-lane NaN index: ln(-1) = NaN (kept as a node — compile lowers the raw graph).
        let neg = g.push(RvNode::ConstNum(-1.0), RvKind::Num);
        let nan_idx = g.push(RvNode::Unary(UnOp::Ln, neg), RvKind::Num);
        let gather = g.push(
            RvNode::Gather {
                elems: elems.into_boxed_slice(),
                index: nan_idx,
            },
            RvKind::Num,
        );
        let prog = compile(&g, gather);
        let mut buf: Vec<Box<[f64]>> = (0..prog.n_regs)
            .map(|_| vec![0.0f64; BATCH].into_boxed_slice())
            .collect();
        let mut rng = crate::rng::Rng::seed_from_u64(0);
        run_batch(&prog, &mut buf, &mut rng);
        let out = &buf[prog.root as usize];
        assert!(
            out.iter().all(|x| x.is_nan()),
            "NaN index must gather NaN, not element 0; got {:?}",
            &out[..4]
        );
    }

    #[test]
    fn gather_with_a_finite_index_selects_that_element() {
        // Companion to the NaN case: an ordinary integer index still reads its element.
        let mut g = RvGraph::default();
        let elems: Vec<RvId> = [10.0, 20.0, 30.0]
            .iter()
            .map(|&v| g.push(RvNode::ConstNum(v), RvKind::Num))
            .collect();
        let idx = g.push(RvNode::ConstNum(1.0), RvKind::Num);
        let gather = g.push(
            RvNode::Gather {
                elems: elems.into_boxed_slice(),
                index: idx,
            },
            RvKind::Num,
        );
        let prog = compile(&g, gather);
        let mut buf: Vec<Box<[f64]>> = (0..prog.n_regs)
            .map(|_| vec![0.0f64; BATCH].into_boxed_slice())
            .collect();
        let mut rng = crate::rng::Rng::seed_from_u64(0);
        run_batch(&prog, &mut buf, &mut rng);
        assert!(buf[prog.root as usize].iter().all(|&x| x == 20.0));
    }

    #[test]
    fn mod_floor_ceil_kernels() {
        // Floored modulo: result takes the sign of the divisor.
        assert_eq!(apply_bin(BinOp::Mod, 7.0, 3.0), 1.0);
        assert_eq!(apply_bin(BinOp::Mod, -1.0, 3.0), 2.0);
        assert_eq!(apply_bin(BinOp::Mod, 7.0, -3.0), -2.0);
        assert_eq!(apply_bin(BinOp::Mod, 5.5, 2.0), 1.5);
        assert!(apply_bin(BinOp::Mod, 1.0, 0.0).is_nan());
        assert_eq!(apply_un(UnOp::Floor, 2.7), 2.0);
        assert_eq!(apply_un(UnOp::Floor, -2.1), -3.0);
        assert_eq!(apply_un(UnOp::Ceil, 2.1), 3.0);
        assert_eq!(apply_un(UnOp::Ceil, -2.9), -2.0);
    }
}
