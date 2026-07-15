//! Lowering (sample-DAG → flat bytecode with CSE) and the columnar VM (PLAN.md
//! "Performance thesis").
//!
//! The RV expression compiles ONCE to a flat list of [`Inst`] over column registers. A
//! register is a contiguous `[f32; BATCH]` buffer (PLAN-PREGPU Track B: **lanes are f32**,
//! aggregation stays f64 — see [`crate::reduce`]). `run_batch` walks the instruction list,
//! filling each instruction's whole `dst` column before moving on — so one pass evaluates
//! `BATCH` draws, never tree-walking per draw. CSE via `HashMap<RvId, Reg>` guarantees a
//! shared sub-RV (e.g. `X` in `X + X`) compiles to ONE register/instruction, so both lanes
//! read the SAME per-batch draw.

use std::collections::HashMap;

use crate::ast::{BinOp, UnOp};
use crate::dist::{RvGraph, RvId, RvNode, Source};

/// Batch / column width. 4 KiB per f32 column (8 KiB before Track B halved the lane type):
/// small enough to stay in cache, large enough to amortize dispatch and give the autovectorizer
/// long runs. Always EVEN — the pair-shared draws (`rng::pair_ctr`) need every fill range to
/// start on an even lane, and every real range start is a multiple of this.
pub const BATCH: usize = 1024;

/// Register index into the column file.
pub type Reg = u32;

/// Register index into the **array** column file (`Program::arrays` gives each register's element
/// count; a runner allocates `n × BATCH` f64s per register). A separate namespace from [`Reg`] so
/// `Inst` stays `Copy` and the scalar file stays homogeneous.
pub type ArrReg = u32;

#[derive(Debug, Clone, Copy)]
pub enum Inst {
    Uniform {
        dst: Reg,
        src: u32,
        lo: f64,
        hi: f64,
    },
    UniformInt {
        dst: Reg,
        src: u32,
        lo: f64,
        hi: f64,
    },
    Normal {
        dst: Reg,
        src: u32,
        mu: f64,
        sigma: f64,
    },
    Exp {
        dst: Reg,
        src: u32,
        rate: f64,
    },
    Poisson {
        dst: Reg,
        src: u32,
        /// Per-program cell-stream ordinal (the Knuth loop's counter region).
        stream: u32,
        lambda: f64,
    },
    Geometric {
        dst: Reg,
        src: u32,
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
    /// Per-lane Fisher–Yates: fill array register `dst` (element count `n`) with an independent
    /// uniform permutation of `0..n` per lane. A SOURCE: consumes exactly `(n-1) × BATCH` RNG
    /// draws per batch (full batch, like every other source), via the same Lemire multiply-high
    /// bounded draw `fill_uniform_int` uses.
    Permutation {
        dst: ArrReg,
        /// Per-program cell-stream ordinal (see `rng::CellStream`).
        stream: u32,
        n: u32,
    },
    /// Per-lane Haar rotation: fill array register `dst` (element count `d²`, row-major
    /// `k = row·d + col`) with an independent random `d`×`d` orthonormal matrix per lane — `d²`
    /// iid `N(0,1)` draws via the same Box–Muller fill every `Normal` source uses, then modified
    /// Gram–Schmidt over the rows in place. A SOURCE: consumes exactly `2·⌈d²/2⌉ × BATCH` RNG
    /// draws per batch (Box–Muller consumes uniforms in pairs — a fixed count even when `d²` is
    /// odd; full batch, like every other source).
    Rotation {
        dst: ArrReg,
        /// Per-program cell-stream ordinal (see `rng::CellStream`).
        stream: u32,
        d: u32,
    },
    /// Per-lane read `dst = arr[round(clamp(index))]` of an array register — `Inst::Gather`'s
    /// exact index semantics (ties-away round; NaN → NaN; clamp into `0..n`), but the table is a
    /// per-lane *random* array instead of a list of element registers.
    ArrIndex {
        dst: Reg,
        arr: ArrReg,
        index: Reg,
    },
}

pub struct Program {
    pub insts: Vec<Inst>,
    pub n_regs: usize,
    pub root: Reg,
    /// Element-register tables for `Inst::Gather`, indexed by its `table` field.
    pub gathers: Vec<Box<[Reg]>>,
    /// Element count of each **array register** (`Inst::Permutation`/`Inst::ArrIndex`), indexed by
    /// [`ArrReg`]. A runner allocates `arrays[a] × BATCH` f32s per register, **lane-major**
    /// (`buf[k*n + j]` is lane `k`'s element `j`): the Fisher–Yates fill writes a whole lane's
    /// array at a time (contiguous swaps stay in one cache line for realistic `n`), where an
    /// element-major layout would stride every swap `BATCH × 8` bytes apart. The ArrIndex read of
    /// one element per lane strides either way, so the writer's layout wins.
    pub arrays: Vec<u32>,
}

/// Lower the transitive cone of `root` to flat bytecode.
///
/// Post-order DFS with a `HashMap<RvId, Reg>` memo: a shared `RvId` compiles to ONE register
/// and ONE instruction (CSE). First cut allocates one register per distinct node, no reuse —
/// register-liveness reuse is deferred to Phase 4 (BATCH×n_regs memory is fine here).
pub fn compile(graph: &RvGraph, root: RvId) -> Program {
    let mut memo: HashMap<RvId, Reg> = HashMap::new();
    let mut arr_memo: HashMap<RvId, ArrReg> = HashMap::new();
    let mut insts: Vec<Inst> = Vec::new();
    let mut gathers: Vec<Box<[Reg]>> = Vec::new();
    let mut arrays: Vec<u32> = Vec::new();
    let ords = crate::kernel::source_ordinals(graph);
    let stream_ords = crate::kernel::cell_stream_ordinals(graph);
    let root_reg = lower(
        graph,
        root,
        &mut memo,
        &mut arr_memo,
        &mut insts,
        &mut gathers,
        &mut arrays,
        &stream_ords,
        &ords,
    );
    Program {
        n_regs: insts.len(),
        insts,
        root: root_reg,
        gathers,
        arrays,
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
    let mut arr_memo: HashMap<RvId, ArrReg> = HashMap::new();
    let mut insts: Vec<Inst> = Vec::new();
    let mut gathers: Vec<Box<[Reg]>> = Vec::new();
    let mut arrays: Vec<u32> = Vec::new();
    // One ordinal map for the whole joint program, so a source shared by two roots draws ONE stream
    // — the property the joint drivers (`corr`, `scatter`) exist for.
    let ords = crate::kernel::source_ordinals(graph);
    let stream_ords = crate::kernel::cell_stream_ordinals(graph);
    let regs: Vec<Reg> = roots
        .iter()
        .map(|&r| {
            lower(
                graph,
                r,
                &mut memo,
                &mut arr_memo,
                &mut insts,
                &mut gathers,
                &mut arrays,
                &stream_ords,
                &ords,
            )
        })
        .collect();
    let root = regs.first().copied().unwrap_or(0);
    (
        Program {
            n_regs: insts.len(),
            insts,
            root,
            gathers,
            arrays,
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
#[allow(clippy::too_many_arguments)] // the lowerer's full output state; a struct would just rename it
fn lower(
    graph: &RvGraph,
    id: RvId,
    memo: &mut HashMap<RvId, Reg>,
    arr_memo: &mut HashMap<RvId, ArrReg>,
    insts: &mut Vec<Inst>,
    gathers: &mut Vec<Box<[Reg]>>,
    arrays: &mut Vec<u32>,
    stream_ords: &[u32],
    ords: &[u32],
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
                    RvNode::Src(_)
                    | RvNode::ConstNum(_)
                    | RvNode::ConstBool(_)
                    | RvNode::Permutation { .. }
                    | RvNode::Rotation { .. }
                    // A shaped draw emits NOTHING (see the Emit arm): it owns a block of source
                    // ordinals, and its readers lower to ordinary scalar fills against that block.
                    // So it is a leaf — and `ArrElem` is one too, taking its recipe and ordinal
                    // straight from the graph rather than from an emitted parent.
                    | RvNode::ArrDraw { .. }
                    | RvNode::ArrElem { .. } => {}
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
                    RvNode::ArrIndex { arr, index } => {
                        push_child(&mut stack, *index);
                        push_child(&mut stack, *arr);
                    }
                    // G4c: `unroll_scans` expands every Scan into flat nodes before the interpreter
                    // lowers, so the interpreter never sees one — bit-for-bit today's DAG.
                    RvNode::Scan { .. } | RvNode::ScanOut { .. } | RvNode::Placeholder { .. } => {
                        unreachable!("Scan is unrolled before bytecode lowering (unroll_scans)")
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
                            src: ords[id.0 as usize],
                            lo: u.lo,
                            hi: u.hi,
                        });
                    }
                    RvNode::Src(Source::UniformInt { lo, hi }) => {
                        insts.push(Inst::UniformInt { dst, src: ords[id.0 as usize], lo, hi });
                    }
                    RvNode::Src(Source::Normal { mu, sigma }) => {
                        insts.push(Inst::Normal { dst, src: ords[id.0 as usize], mu, sigma });
                    }
                    RvNode::Src(Source::Exp { rate }) => {
                        insts.push(Inst::Exp { dst, src: ords[id.0 as usize], rate })
                    }
                    RvNode::Src(Source::Poisson { lambda }) => {
                        let stream = stream_ords[id.0 as usize];
                        insts.push(Inst::Poisson { dst, src: ords[id.0 as usize], stream, lambda });
                    }
                    RvNode::Src(Source::Geometric { p }) => {
                        insts.push(Inst::Geometric { dst, src: ords[id.0 as usize], p })
                    }
                    // The shaped-draw pair (PLAN-WEBGPU G-half). `ArrDraw` lowers to NOTHING: it is
                    // a pure ordinal-block owner, so it never occupies an instruction slot and
                    // never enters `memo` (nothing reads it — `ArrElem` gets its recipe and its
                    // ordinal straight from the graph). `ArrElem` lowers to exactly the fill its
                    // recipe's scalar `Src` lowers to, at ordinal `base + k` — so the interpreter's
                    // hot loop is bit-for-bit what it was before this node existed, and `~[n] d` is
                    // indistinguishable from n separate `~ d` draws.
                    //
                    // `continue` (not a pushed no-op): the `dst == insts.len()` register numbering
                    // invariant only requires that every pushed inst take the next slot, so a node
                    // that pushes nothing simply doesn't participate.
                    RvNode::ArrDraw { .. } => continue,
                    RvNode::ArrElem { arr, k } => {
                        let src = crate::kernel::elem_ordinal(ords, arr, k);
                        match crate::kernel::elem_source(graph, arr) {
                            Source::Uniform(u) => insts.push(Inst::Uniform { dst, src, lo: u.lo, hi: u.hi }),
                            Source::UniformInt { lo, hi } => {
                                insts.push(Inst::UniformInt { dst, src, lo, hi })
                            }
                            Source::Normal { mu, sigma } => {
                                insts.push(Inst::Normal { dst, src, mu, sigma })
                            }
                            Source::Exp { rate } => insts.push(Inst::Exp { dst, src, rate }),
                            Source::Geometric { p } => insts.push(Inst::Geometric { dst, src, p }),
                            Source::Poisson { lambda } => {
                                let stream = crate::kernel::elem_stream(stream_ords, arr, k);
                                insts.push(Inst::Poisson { dst, src, stream, lambda });
                            }
                        }
                    }
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
                    RvNode::Permutation { n } => {
                        // An array-valued source: its result lives in an ARRAY register, but it
                        // still occupies one instruction slot, so the `dst == inst index` register
                        // numbering invariant (`n_regs == insts.len()`) holds — the scalar column
                        // at `dst` is simply never written. `arr_memo` maps the node to its array
                        // register for the ArrIndex readers.
                        let a = ArrReg::try_from(arrays.len())
                            .expect("bytecode exceeded 2^32 array registers");
                        arrays.push(n);
                        arr_memo.insert(id, a);
                        let stream = stream_ords[id.0 as usize];
                        insts.push(Inst::Permutation { dst: a, stream, n });
                    }
                    RvNode::Rotation { d } => {
                        // Same array-valued-source shape as Permutation: one instruction slot (the
                        // scalar column at `dst` stays unwritten, keeping `n_regs == insts.len()`),
                        // result in a fresh `d²`-element array register.
                        let a = ArrReg::try_from(arrays.len())
                            .expect("bytecode exceeded 2^32 array registers");
                        arrays.push(d * d);
                        arr_memo.insert(id, a);
                        let stream = stream_ords[id.0 as usize];
                        insts.push(Inst::Rotation { dst: a, stream, d });
                    }
                    RvNode::ArrIndex { arr, index } => {
                        let (a, ri) = (arr_memo[&arr], memo[&index]);
                        insts.push(Inst::ArrIndex {
                            dst,
                            arr: a,
                            index: ri,
                        });
                    }
                    // G4c: the interpreter unrolls a Scan into the flat instruction stream (below).
                    // These arms don't fit the one-inst-per-Emit shape, so they're handled specially
                    // and never reach here — see the note. (Wired in the next step.)
                    RvNode::Scan { .. } | RvNode::ScanOut { .. } | RvNode::Placeholder { .. } => {
                        unreachable!("G4c Scan lowering not yet wired")
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
///
/// `arrs` is the **array register file** (one `arrays[a] × BATCH` buffer per [`ArrReg`], sized by
/// `Program::arrays`) — a separate parameter, not part of `regs`, so the scalar file stays
/// uniform-width and programs without array nodes pass `&mut []` for free.
pub fn run_batch(
    program: &Program,
    regs: &mut [Box<[f32]>],
    arrs: &mut [Box<[f32]>],
    key: crate::rng::Key,
    lane0: u32,
) {
    use crate::rng;
    for inst in &program.insts {
        match *inst {
            Inst::Uniform { dst, src, lo, hi } => {
                rng::fill_uniform(key, src, lane0, lo, hi, &mut regs[dst as usize]);
            }
            Inst::UniformInt { dst, src, lo, hi } => {
                rng::fill_uniform_int(key, src, lane0, lo, hi, &mut regs[dst as usize]);
            }
            Inst::Normal { dst, src, mu, sigma } => {
                rng::fill_normal(key, src, lane0, mu, sigma, &mut regs[dst as usize]);
            }
            Inst::Exp { dst, src, rate } => {
                rng::fill_exp(key, src, lane0, rate, &mut regs[dst as usize]);
            }
            Inst::Poisson {
                dst,
                src,
                stream,
                lambda,
            } => {
                rng::fill_poisson(key, src, stream, lane0, lambda, &mut regs[dst as usize]);
            }
            Inst::Geometric { dst, src, p } => {
                rng::fill_geometric(key, src, lane0, p, &mut regs[dst as usize]);
            }
            // The graph's constants are f64 (deterministic values stay f64); a lane holds
            // `val as f32` — the same rounding both emitters bake into their f32 constants.
            // A magnitude past f32's range becomes ±inf here: a documented Track B boundary.
            Inst::ConstNum { dst, val } => {
                for x in regs[dst as usize].iter_mut() {
                    *x = val as f32;
                }
            }
            Inst::ConstBool { dst, val } => {
                for x in regs[dst as usize].iter_mut() {
                    *x = val as f32;
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
                    regs[dst][k] = if regs[cond][k] != 0.0f32 {
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
                let mut scratch = [0.0f32; BATCH];
                for k in 0..BATCH {
                    let raw = regs[index][k].round();
                    if raw.is_nan() {
                        scratch[k] = f32::NAN;
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
            Inst::Permutation { dst, stream, n } => {
                // Per lane: identity, then Fisher–Yates high-to-low with the same Lemire
                // multiply-high bounded draw `fill_uniform_int` uses (48 consumed bits, bias ≤
                // count/2⁴⁸), drawing from the lane's `CellStream` chain. Lane-major
                // (`buf[k*n + j]`, see `Program::arrays`): the shuffle's swaps stay within lane
                // `k`'s contiguous n-element run. Per-cell keying makes consumption a pure
                // function of `(lane, src)` — partial batches can't change the stream.
                let n = n as usize;
                let buf = &mut arrs[dst as usize];
                for k in 0..BATCH {
                    let mut draws = rng::CellStream::new(key, stream, lane0.wrapping_add(k as u32));
                    let lane = &mut buf[k * n..(k + 1) * n];
                    for (j, x) in lane.iter_mut().enumerate() {
                        *x = j as f32;
                    }
                    for j in (1..n).rev() {
                        let i = draws.next_bounded(j as u64 + 1) as usize;
                        lane.swap(i, j);
                    }
                }
            }
            Inst::Rotation { dst, stream, d } => {
                // Per lane: fill the lane's contiguous `d²` run (row-major, `m[r*d + c]`) with iid
                // standard normals from the lane's `CellStream` chain, then modified Gram–Schmidt the
                // rows in place — subtract each earlier (already unit) row's projection, then
                // normalize. Per-cell keying makes consumption a pure function of `(lane, src)`, so a
                // partial batch can't change the stream. A zero-norm row (probability 0 for Gaussians)
                // yields inf/NaN, the same IEEE answer the graph-level `normalize` produced.
                //
                // **All f32 (PLAN-WEBGPU G4b).** This mirrors the WGSL rotation kernel *op for op* —
                // the same 24-bit uniforms, the same Box–Muller pairing (two u48 draws feed both
                // branches, cos → even entry, sin → odd), the same MGS loop order — so the two
                // backends draw the same rotation to within f32 transcendental ULPs and FMA. The
                // interpreter used to run this in f64 scratch (orthonormality ~1e-7 vs f32's ~2e-5),
                // but the lane type is f32 everywhere else and the extra precision changed no result:
                // turboquant's b=1 distortion is 0.347221 either way. Keeping it f32 makes rotation
                // consistent with the rest of the engine and lets the GPU take turboquant without a
                // precision fork (the f64 reference lives in git history if it is ever wanted back).
                let d = d as usize;
                let dd = d * d;
                let buf = &mut arrs[dst as usize];
                const SCALE24: f32 = 1.0 / (1u32 << 24) as f32;
                let mut m = vec![0.0f32; dd];
                for k in 0..BATCH {
                    let mut draws = rng::CellStream::new(key, stream, lane0.wrapping_add(k as u32));
                    // Box–Muller in f32, matching `wgsl_emit`'s rotation fill: `u1` from the top 24
                    // bits of the first u48 (+0.5 keeps it off zero, so `ln` is safe), `u2` from the
                    // second; theta ∈ [0, 2π) stays inside `approx::TRIG_MAX_F32`.
                    let mut e = 0;
                    while e < dd {
                        let hi0 = (draws.next_u48() >> 24) as u32;
                        let hi1 = (draws.next_u48() >> 24) as u32;
                        let u1 = (hi0 as f32 + 0.5) * SCALE24;
                        let r = (-2.0f32 * crate::approx::ln_f32(u1)).sqrt();
                        let theta = std::f32::consts::TAU * (hi1 as f32 * SCALE24);
                        m[e] = r * crate::approx::cos_f32(theta);
                        if e + 1 < dd {
                            m[e + 1] = r * crate::approx::sin_f32(theta);
                        }
                        e += 2;
                    }
                    for r in 0..d {
                        for p in 0..r {
                            let mut dot = 0.0f32;
                            for c in 0..d {
                                dot += m[r * d + c] * m[p * d + c];
                            }
                            for c in 0..d {
                                m[r * d + c] -= dot * m[p * d + c];
                            }
                        }
                        let mut normsq = 0.0f32;
                        for c in 0..d {
                            normsq += m[r * d + c] * m[r * d + c];
                        }
                        let inv = 1.0 / normsq.sqrt();
                        for c in 0..d {
                            m[r * d + c] *= inv;
                        }
                    }
                    buf[k * dd..(k + 1) * dd].copy_from_slice(&m);
                }
            }
            Inst::ArrIndex { dst, arr, index } => {
                // Per lane: `Inst::Gather`'s exact index semantics (ties-away `round`; NaN index →
                // NaN, never element 0; `raw <= 0` → 0; `>= last` → last), reading lane `k`'s own
                // array (`buf[k*n + i]`) instead of an element register's lane. Scratch first so
                // the `regs[index]` read can't alias the `regs[dst]` write.
                let buf = &arrs[arr as usize];
                let n = buf.len() / BATCH; // never 0: eval builds no zero-length array node
                let last = n - 1;
                let index = index as usize;
                let mut scratch = [0.0f32; BATCH];
                for k in 0..BATCH {
                    let raw = regs[index][k].round();
                    if raw.is_nan() {
                        scratch[k] = f32::NAN;
                        continue;
                    }
                    let i = if raw <= 0.0 {
                        0
                    } else if raw as usize >= last {
                        last
                    } else {
                        raw as usize
                    };
                    scratch[k] = buf[k * n + i];
                }
                regs[dst as usize].copy_from_slice(&scratch);
            }
        }
    }
}

/// Scalar unary op over a **lane value** (f32 — PLAN-PREGPU Track B). `Not` is logical not over a
/// 0/1 bool column.
///
/// The ops with no pinnable f32 form — `atan`, `round`, `exp` — compute in f64 and round back
/// (never `f32::atan`, i.e. `atanf`). That is the cross-backend contract: the JIT calls a shim
/// that promotes/calls/demotes and the wasm module imports the f64 `Math.*` around an
/// `f64.promote`/`f32.demote` pair, so all three agree bit-for-bit. `sin`/`cos`/`ln` are the
/// shared `approx` f32 polynomials both emitters inline; `sqrt`/`floor`/`ceil` are native f32
/// instructions (correctly rounded, hence identical everywhere).
#[inline]
fn apply_un(op: UnOp, x: f32) -> f32 {
    match op {
        UnOp::Neg => -x,
        UnOp::Not => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
        UnOp::Sin => crate::approx::sin_f32(x),
        UnOp::Cos => crate::approx::cos_f32(x),
        UnOp::Atan => (x as f64).atan() as f32,
        // sign: -1 / 0 / +1 (0 at exactly zero, unlike signum which is ±1 at ±0.0).
        UnOp::Sign => (x > 0.0) as i32 as f32 - (x < 0.0) as i32 as f32,
        UnOp::Round => (x as f64).round() as f32,
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Exp => (x as f64).exp() as f32,
        UnOp::Ln => crate::approx::ln_guarded_f32(x),
        // IEEE sqrt (correctly rounded, so the native sqrt instructions in both code generators
        // are bit-identical to this): sqrt(-0.0) = -0.0, sqrt(x<0) = NaN (incl. -inf).
        UnOp::Sqrt => x.sqrt(),
    }
}

/// Scalar binary op over lane values (f32). Comparisons produce 0.0/1.0 columns (the bool
/// convention from PLAN.md, pre-wiring Phase 3's `P()`).
#[inline]
fn apply_bin(op: BinOp, a: f32, b: f32) -> f32 {
    // The single shared lane kernel (finding F4) — the same `num::` definition both emitters
    // transcribe, so the VM can never drift from them.
    crate::num::fold_binop_f32(op, a, b)
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
        let mut buf: Vec<Box<[f32]>> = (0..prog.n_regs)
            .map(|_| vec![0.0f32; BATCH].into_boxed_slice())
            .collect();
        run_batch(&prog, &mut buf, &mut [], crate::rng::Key::from_seed(0), 0);
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
        let mut buf: Vec<Box<[f32]>> = (0..prog.n_regs)
            .map(|_| vec![0.0f32; BATCH].into_boxed_slice())
            .collect();
        run_batch(&prog, &mut buf, &mut [], crate::rng::Key::from_seed(0), 0);
        assert!(buf[prog.root as usize].iter().all(|&x| x == 20.0));
    }

    /// Build `Permutation(n)` plus its `n` constant-index `ArrIndex` element reads (the shape
    /// `draw_permutation` emits) and return `(graph, arr, element_roots)`.
    fn perm_with_elems(n: usize) -> (RvGraph, RvId, Vec<RvId>) {
        let mut g = RvGraph::default();
        let arr = g.push(RvNode::Permutation { n: n as u32 }, RvKind::Arr(n as u32));
        let roots = (0..n)
            .map(|i| {
                let index = g.push(RvNode::ConstNum(i as f64), RvKind::Num);
                g.push(RvNode::ArrIndex { arr, index }, RvKind::Num)
            })
            .collect();
        (g, arr, roots)
    }

    /// One worker's column file (scalar or array registers alike).
    type RegFile = Vec<Box<[f32]>>;

    /// Allocate the scalar + array register files a program needs (the runner's job in backend.rs).
    fn reg_files(prog: &Program) -> (RegFile, RegFile) {
        let regs = (0..prog.n_regs)
            .map(|_| vec![0.0f32; BATCH].into_boxed_slice())
            .collect();
        let arrs = prog
            .arrays
            .iter()
            .map(|&n| vec![0.0f32; n as usize * BATCH].into_boxed_slice())
            .collect();
        (regs, arrs)
    }

    #[test]
    fn permutation_lanes_are_valid_and_uniform() {
        // Every lane must hold a true permutation of 0..n (all distinct, sum = n(n-1)/2), and the
        // Fisher–Yates must be uniform: over many lanes each value lands at each position with
        // frequency ≈ 1/n (the χ²-style bound below is ~7σ at this sample size, like the
        // `fill_uniform_int_is_uniform_over_range` test).
        const N: usize = 6;
        let (g, _, roots) = perm_with_elems(N);
        let (prog, regs) = compile_roots(&g, &roots);
        let (mut buf, mut arrs) = reg_files(&prog);
        let key = crate::rng::Key::from_seed(7);
        let batches = 100;
        let mut counts = [[0u64; N]; N]; // counts[position][value]
        for b in 0..batches {
            run_batch(&prog, &mut buf, &mut arrs, key, (b * BATCH) as u32);
            for k in 0..BATCH {
                let mut seen = [false; N];
                let mut sum = 0.0f32;
                for (pos, &r) in regs.iter().enumerate() {
                    let v = buf[r as usize][k];
                    assert!(
                        v.fract() == 0.0 && (0.0..N as f32).contains(&v),
                        "non-permutation value {v}"
                    );
                    let vi = v as usize;
                    assert!(!seen[vi], "duplicate value {vi} in one lane");
                    seen[vi] = true;
                    sum += v;
                    counts[pos][vi] += 1;
                }
                assert_eq!(sum, (N * (N - 1) / 2) as f32, "lane is not a permutation");
            }
        }
        let expected = (batches * BATCH) as f64 / N as f64;
        for (pos, row) in counts.iter().enumerate() {
            for (val, &c) in row.iter().enumerate() {
                let dev = (c as f64 - expected).abs() / expected;
                assert!(dev < 0.05, "position {pos} value {val}: deviation {dev:.4}");
            }
        }
    }

    #[test]
    fn arr_index_edge_semantics_match_gather() {
        // ArrIndex must copy Inst::Gather's index semantics exactly: ties-away round (2.5 → 3),
        // raw <= 0 → element 0, past-the-end/+inf → last, NaN → NaN (never element 0). Checked
        // lane-for-lane against the reference constant indices over the SAME permutation draw.
        const N: usize = 4;
        let mut g = RvGraph::default();
        let arr = g.push(RvNode::Permutation { n: N as u32 }, RvKind::Arr(N as u32));
        let read = |g: &mut RvGraph, v: f64| {
            let index = g.push(RvNode::ConstNum(v), RvKind::Num);
            g.push(RvNode::ArrIndex { arr, index }, RvKind::Num)
        };
        let first = read(&mut g, 0.0);
        let last = read(&mut g, 3.0);
        let neg = read(&mut g, -7.3); // <= 0 clamps to 0
        let tie = read(&mut g, 2.5); // ties away → 3
        let huge = read(&mut g, f64::INFINITY); // clamps to last
        let nan = read(&mut g, f64::NAN); // NaN propagates
        let roots = [first, last, neg, tie, huge, nan];
        let (prog, regs) = compile_roots(&g, &roots);
        let (mut buf, mut arrs) = reg_files(&prog);
        run_batch(&prog, &mut buf, &mut arrs, crate::rng::Key::from_seed(11), 0);
        let col = |r: Reg| &buf[r as usize];
        for k in 0..BATCH {
            assert_eq!(col(regs[2])[k], col(regs[0])[k], "negative index → element 0");
            assert_eq!(col(regs[3])[k], col(regs[1])[k], "2.5 rounds away to 3");
            assert_eq!(col(regs[4])[k], col(regs[1])[k], "+inf clamps to last");
            assert!(col(regs[5])[k].is_nan(), "NaN index must yield NaN");
        }
    }

    /// Build `Rotation(d)` plus its `d²` constant-index `ArrIndex` element reads (the shape
    /// `draw_rotation` emits, row-major `k = row·d + col`) and return `(graph, element_roots)`.
    fn rot_with_elems(d: usize) -> (RvGraph, Vec<RvId>) {
        let mut g = RvGraph::default();
        let arr = g.push(
            RvNode::Rotation { d: d as u32 },
            RvKind::Arr((d * d) as u32),
        );
        let roots = (0..d * d)
            .map(|i| {
                let index = g.push(RvNode::ConstNum(i as f64), RvKind::Num);
                g.push(RvNode::ArrIndex { arr, index }, RvKind::Num)
            })
            .collect();
        (g, roots)
    }

    #[test]
    fn rotation_lanes_are_orthonormal() {
        // Every lane's d² element reads must form an orthonormal matrix: each row unit-norm and each
        // row pair orthogonal. The MGS now runs in **f32** (G4b — matching the GPU rather than the
        // old f64 scratch), so orthonormality is limited by f32 *arithmetic* accumulating through an
        // O(d³) elimination — ~2e-5 at d = 5, a hundred-fold looser than the old f64's ~1e-7, and the
        // deliberate cost of keeping the lane type f32 everywhere. The determinant's sign is
        // irrelevant — the draw is Haar over O(d).
        // f32 MGS gives a *typical* residual ~2e-5, but the tail is heavy: a near-singular Gaussian
        // matrix (which occurs at rate ~1/1000s at d=5) is ill-conditioned, and there f32 loses
        // enough that a lane's residual reaches ~1e-3 — f64 absorbed these with its 29 extra bits, f32
        // doesn't. So a hard per-lane max is the wrong assertion; check the RMS residual (robust, and
        // the quantity that governs a Monte Carlo expectation) is tight, plus a generous per-lane cap
        // that still catches a matrix that isn't a rotation at all (residual O(0.1)).
        const D: usize = 5;
        let (g, roots) = rot_with_elems(D);
        let (prog, regs) = compile_roots(&g, &roots);
        let (mut buf, mut arrs) = reg_files(&prog);
        let key = crate::rng::Key::from_seed(13);
        let (mut sumsq, mut count, mut worst) = (0.0f64, 0u64, 0.0f64);
        for b in 0..20u32 {
            run_batch(&prog, &mut buf, &mut arrs, key, b * BATCH as u32);
            for k in 0..BATCH {
                let q = |r: usize, c: usize| buf[regs[r * D + c] as usize][k];
                for r1 in 0..D {
                    for r2 in r1..D {
                        let dot: f64 = (0..D).map(|c| q(r1, c) as f64 * q(r2, c) as f64).sum();
                        let want = if r1 == r2 { 1.0 } else { 0.0 };
                        let err = (dot - want).abs();
                        sumsq += err * err;
                        count += 1;
                        worst = worst.max(err);
                        assert!(err < 1e-2, "lane {k}: rows {r1}·{r2} = {dot}, want {want} — not a rotation");
                    }
                }
            }
        }
        let rms = (sumsq / count as f64).sqrt();
        assert!(rms < 1e-4, "RMS orthonormality residual {rms:e} — f32 MGS should hold ~2e-5 typical");
        assert!(worst < 1e-2, "worst orthonormality residual {worst:e}");
    }

    #[test]
    fn rotation_entries_match_haar_moments() {
        // Haar sanity across lanes: each entry is symmetric about 0 (`E[q00] = 0`) and, the rows
        // being unit with `d` exchangeable columns, `E[q00²] = 1/d` exactly. Bounds are ~5σ for
        // this sample size (q00 ≈ N(0, 1/d) for the mean; Var(q00²) ≲ E[q00⁴] ≈ 3/d² for the
        // second moment), seeded and deterministic like the permutation uniformity test.
        const D: usize = 5;
        let (g, roots) = rot_with_elems(D);
        let (prog, regs) = compile_roots(&g, &roots);
        let (mut buf, mut arrs) = reg_files(&prog);
        let key = crate::rng::Key::from_seed(17);
        let batches = 20;
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for b in 0..batches {
            run_batch(&prog, &mut buf, &mut arrs, key, (b * BATCH) as u32);
            for k in 0..BATCH {
                let q00 = buf[regs[0] as usize][k] as f64;
                sum += q00;
                sum_sq += q00 * q00;
            }
        }
        let n = (batches * BATCH) as f64;
        let (mean, mean_sq) = (sum / n, sum_sq / n);
        let sd_mean = (1.0 / D as f64).sqrt() / n.sqrt();
        assert!(mean.abs() < 5.0 * sd_mean, "E[q00] = {mean}");
        let sd_mean_sq = (3.0f64).sqrt() / D as f64 / n.sqrt();
        assert!(
            (mean_sq - 1.0 / D as f64).abs() < 5.0 * sd_mean_sq,
            "E[q00²] = {mean_sq}, want {}",
            1.0 / D as f64
        );
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
