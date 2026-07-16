//! Drawing recipes into the sample-DAG (user-fn calls, `~`/`~[shape]` draws, the noise draws) and the operator-lifting helpers (`lift_unary`/`lift_binary`/`lift_if`/`operand_to_rv`).
//!
//! Extracted verbatim from the monolithic `eval.rs` (finding F1); an `impl Engine` block
//! whose methods reach the rest of the evaluator through `self` and the shared free
//! helpers/tables that stay in the module root.

use std::collections::HashMap;
use std::rc::Rc;

use super::*;
use crate::dist::{DistArg, Recipe, RvId, RvKind, RvNode, Source, Uniform};
use crate::error::{NoiseError, Result, Span};
use crate::signal::{NoiseKind, NoiseSpec, RealizationId, SigExpr};
use crate::value::Value;

impl Engine {
    pub(super) fn call_user_fn(
        &mut self,
        name: &str,
        f: &UserFn,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value> {
        if args.len() != f.params.len() {
            return Err(NoiseError::runtime(
                format!(
                    "{name} expects {} argument(s), got {}",
                    f.params.len(),
                    args.len()
                ),
                span,
            ));
        }
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(NoiseError::runtime(
                format!(
                    "call stack too deep (limit {MAX_CALL_DEPTH}) calling '{name}' — Noise \
                     unrolls calls at build time, so recursion must terminate"
                ),
                span,
            ));
        }
        self.call_depth += 1;
        // Swap in a fresh frame holding only the parameters; restore on the way out.
        let mut frame = HashMap::with_capacity(f.params.len());
        for (p, a) in f.params.iter().zip(args) {
            frame.insert(p.clone(), a);
        }
        let saved = std::mem::replace(&mut self.vars, frame);
        let result = self.eval(&f.body);
        self.vars = saved;
        self.call_depth -= 1;
        // A stochastic (`~`) function draws on each call (recipe → fresh RV); a deterministic
        // (`=`) function returns its body value verbatim.
        match f.kind {
            BindKind::Sample => self.draw_if_recipe(result?),
            BindKind::Assign => result,
        }
    }

    /// `~` semantics in one place: a recipe is drawn into a fresh RV; an undrawn noise generator
    /// is drawn into ONE lazy **realization** (a `Signal` noise leaf — length still lazy);
    /// anything else (a point mass, an already-drawn RV) binds as-is, since there is nothing new
    /// to draw. Fallible because a structured recipe (`rotation`) builds a whole matrix and could
    /// surface a shape error; scalar recipe and noise draws never fail.
    pub(super) fn draw_if_recipe(&mut self, v: Value) -> Result<Value> {
        match v {
            Value::Recipe(r) => self.draw(r),
            Value::Noise(spec) => Ok(self.draw_noise(spec)),
            other => Ok(other),
        }
    }

    /// Draw one lazy noise **realization** (PLAN-SIGNALS §1.1): allocate a fresh
    /// [`RealizationId`] and wrap it in a signal leaf. The length stays lazy — the realization
    /// pins it at first materialization (see [`Engine::realization`]). The complex generator
    /// splits into two independent real lanes of strength `sigma/√2` (per-quadrature CSCG, so
    /// `E|z|² = sigma²` like `rand::normal_complex`).
    fn draw_noise(&mut self, spec: NoiseSpec) -> Value {
        if let NoiseKind::WhiteComplex = spec.kind {
            let lane = NoiseSpec {
                sigma: spec.sigma / std::f64::consts::SQRT_2,
                kind: NoiseKind::White,
            };
            let re = self.draw_noise(lane);
            let im = self.draw_noise(lane);
            return Value::complex(re, im);
        }
        let id = RealizationId(self.next_realization);
        self.next_realization += 1;
        Value::Signal(Rc::new(SigExpr::Noise { id, spec }))
    }

    /// `~[n] noise` / `~[m, n] noise` — an **eager** realization pinned up front: the last
    /// dimension is the realization length, outer dimensions draw independent realizations. This
    /// is the old `sample(noise_*(…), n)`, now spelled as a draw; it materializes directly to an
    /// ordinary array of RVs (no cache entry needed — the value IS the realization).
    fn draw_noise_shaped(&mut self, dims: &[usize], spec: NoiseSpec) -> Value {
        if let [n] = dims {
            let vals = match spec.kind {
                NoiseKind::WhiteComplex => {
                    let lane = NoiseSpec {
                        sigma: spec.sigma / std::f64::consts::SQRT_2,
                        kind: NoiseKind::White,
                    };
                    let re = self.materialize_noise(lane, *n);
                    let im = self.materialize_noise(lane, *n);
                    re.into_iter()
                        .zip(im)
                        .map(|(a, b)| Value::complex(a, b))
                        .collect()
                }
                _ => self.materialize_noise(spec, *n),
            };
            return Value::Array(Rc::new(vals));
        }
        let (m, rest) = (dims[0], &dims[1..]);
        Value::Array(Rc::new(
            (0..m).map(|_| self.draw_noise_shaped(rest, spec)).collect(),
        ))
    }

    /// The prefix draw operator `~[shape]? body` (LANG.md §2). Evaluate the operand once to a
    /// recipe (or any value), then materialize: a bare `~` draws a single sample; a shape draws a
    /// nested array with an *independent* draw at every leaf. Kept out of the `eval` match so that
    /// arm's locals don't inflate the (deeply recursive) `eval` stack frame.
    pub(super) fn eval_sample(&mut self, shape: &[Spanned], body: &Spanned) -> Result<Value> {
        let v = self.eval(body)?;
        if shape.is_empty() {
            return self.draw_if_recipe(v);
        }
        let mut dims = Vec::with_capacity(shape.len());
        for dim in shape {
            let dv = self.eval(dim)?;
            dims.push(self.count_arg("~", &dv, dim.span)?);
        }
        // Cap the total number of leaves (product of dims) up front, before any allocation, so a
        // `~[1e15]` can't `Vec::with_capacity` an astronomical count and abort (finding A6). The
        // product is computed with saturating arithmetic so it can't itself overflow.
        let leaves = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX);
        if leaves > MAX_DRAW_ELEMS {
            let shape_span = shape.first().map(|s| s.span).unwrap_or(body.span);
            return Err(NoiseError::runtime(
                format!(
                    "draw shape {dims:?} has {leaves} leaves, over the {MAX_DRAW_ELEMS} cap — each \
                     leaf is an independent draw; use a smaller shape"
                ),
                shape_span,
            ));
        }
        // `~[n]` on a noise generator pins ONE realization to length `n` (the shape is the time
        // axis), not `n` independent realizations — so it gets its own arm.
        if let Value::Noise(spec) = v {
            return Ok(self.draw_noise_shaped(&dims, spec));
        }
        self.draw_shaped(&dims, &v)
    }

    /// Build a nested array of the given shape, drawing the recipe independently at every leaf.
    /// Backs the shaped prefix draw `~[n, m, …] recipe`.
    ///
    /// The leaves are still iid, and still draw exactly the stream they always did — but they now
    /// share **one** [`RvNode::ArrDraw`] block per base source instead of pushing `leaves`
    /// independent [`RvNode::Src`] nodes. `push_src` does the redirection; this just opens the
    /// shaped context (with the total leaf count, so `~[3,4]` is a single 12-wide block) and walks
    /// the leaves in row-major order.
    fn draw_shaped(&mut self, dims: &[usize], recipe: &Value) -> Result<Value> {
        let leaves: usize = dims.iter().product(); // `eval_sample` capped this at MAX_DRAW_ELEMS
        let outer = self.shaped.take(); // a shaped draw nested inside one is not a thing, but be total
        self.shaped = Some(ShapedDraw {
            leaves: leaves as u32,
            k: 0,
            blocks: Vec::new(),
            pos: 0,
        });
        let out = self.draw_leaves(dims, recipe);
        self.shaped = outer;
        out
    }

    /// The row-major leaf walk of [`draw_shaped`], inside the open shaped context.
    fn draw_leaves(&mut self, dims: &[usize], recipe: &Value) -> Result<Value> {
        let (n, rest) = (dims[0], &dims[1..]);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if rest.is_empty() {
                // A fresh leaf: base-source position back to 0, so base `j` of this leaf lands in
                // block `j` (the same block leaf 0 created for it).
                if let Some(sh) = self.shaped.as_mut() {
                    sh.pos = 0;
                }
                out.push(self.draw_if_recipe(recipe.clone())?);
                if let Some(sh) = self.shaped.as_mut() {
                    sh.k += 1;
                }
            } else {
                out.push(self.draw_leaves(rest, recipe)?);
            }
        }
        Ok(Value::Array(Rc::new(out)))
    }

    /// Push a recipe's **base source** — the single point where a draw enters the sample-DAG, and
    /// therefore the single point that a shaped draw (`~[n] recipe`) has to redirect.
    ///
    /// Outside a shaped draw this is just `graph.push(Src(src))`, exactly as before. Inside one it
    /// returns `ArrElem { arr, k }` against the shaped draw's `ArrDraw` block instead — allocating
    /// that block on the first leaf and reusing it on the rest (see [`crate::eval::ShapedDraw`]).
    /// The draws are unchanged: element `k` of the block has ordinal `base + k`, which is the
    /// ordinal the `k`-th independent `Src` would have been given.
    pub(super) fn push_src(&mut self, src: Source) -> RvId {
        let Some(sh) = self.shaped.as_ref() else {
            return self.graph.push(RvNode::Src(src), RvKind::Num);
        };
        let (pos, k, leaves) = (sh.pos, sh.k, sh.leaves);
        let arr = match self.shaped.as_ref().unwrap().blocks.get(pos) {
            // Leaves 1..n: the block for this base-source position already exists.
            Some(&arr) => {
                debug_assert_eq!(
                    self.graph.node(arr),
                    &RvNode::ArrDraw { n: leaves, src },
                    "a shaped draw's leaves must push the same base sources in the same order"
                );
                arr
            }
            // Leaf 0: allocate the block. `RvKind::Arr(leaves)` — it is an array-valued source, in
            // the shape of `Permutation`/`Rotation`.
            None => {
                let arr = self
                    .graph
                    .push(RvNode::ArrDraw { n: leaves, src }, RvKind::Arr(leaves));
                self.shaped.as_mut().unwrap().blocks.push(arr);
                arr
            }
        };
        self.shaped.as_mut().unwrap().pos += 1;
        self.graph.push(RvNode::ArrElem { arr, k }, RvKind::Num)
    }

    /// Draw a fresh random variable from a recipe — the *only* place sampling-DAG source nodes
    /// are created (LANG.md §2: `~` is the only thing that draws). Each call instantiates new
    /// source node(s), so two `~` on the same recipe are independent. The scalar recipes return a
    /// `Value::Dist`; the structured `rotation` recipe returns a `Value::Array` (a matrix of RVs).
    pub(super) fn draw(&mut self, r: Recipe) -> Result<Value> {
        // The multivariate recipes: drawing them builds a whole array of correlated draws.
        if let Recipe::Rotation { d } = r {
            return self.draw_rotation(d);
        }
        if let Recipe::Permutation { n } = r {
            return self.draw_permutation(n);
        }
        if let Recipe::Empirical { data } = r {
            return Ok(self.draw_empirical(data));
        }
        if let Recipe::BlockBootstrap { data, block_len } = r {
            return Ok(self.draw_block_bootstrap(data, block_len));
        }
        // A complex draw yields a `Value::Complex` (two independent real channels), not a scalar id.
        if let Recipe::NormalComplex { sigma } = r {
            let s = sigma / std::f64::consts::SQRT_2;
            let re = self.push_src(Source::Normal { mu: 0.0, sigma: s });
            let im = self.push_src(Source::Normal { mu: 0.0, sigma: s });
            return Ok(Value::complex(Value::Dist(re), Value::Dist(im)));
        }
        let id = match r {
            Recipe::Uniform { lo, hi } => self.push_src(Source::Uniform(Uniform { lo, hi })),
            Recipe::UniformInt { lo, hi } => self.push_src(Source::UniformInt { lo, hi }),
            Recipe::Normal { mu, sigma } => self.push_src(Source::Normal { mu, sigma }),
            Recipe::Exp { rate } => self.push_src(Source::Exp { rate }),
            Recipe::Poisson { lambda } => self.push_src(Source::Poisson { lambda }),
            Recipe::Geometric { p } => self.push_src(Source::Geometric { p }),
            // The `_int` family draws a continuous source then rounds each lane to an integer.
            Recipe::NormalInt { mu, sigma } => {
                let z = self.push_src(Source::Normal { mu, sigma });
                self.graph.push(RvNode::Unary(UnOp::Round, z), RvKind::Num)
            }
            Recipe::ExpInt { rate } => {
                let z = self.push_src(Source::Exp { rate });
                self.graph.push(RvNode::Unary(UnOp::Round, z), RvKind::Num)
            }
            Recipe::Bernoulli { p } => {
                // bernoulli(p) ≡ (unif(0,1) < p): a bool-RV that is 1 with probability p.
                let u = self.push_src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 }));
                let c = self.graph.push(RvNode::ConstNum(p), RvKind::Num);
                self.graph
                    .push(RvNode::Binary(BinOp::Lt, u, c), RvKind::Bool)
            }
            // --- distributions with a (possibly) random parameter: lower to a standard base draw +
            //     a deterministic transform, so the VM/RNG never change (LANG.md "Hierarchical
            //     distributions"). A fresh base draw per `~`, the SAME parameter node reused, gives
            //     conditional independence given the parameter (`a ~ bernoulli(p); b ~ bernoulli(p)`
            //     are independent given `p`). The transform nodes simplify/CSE/lower like any other.
            Recipe::UniformDyn { lo, hi } => {
                // lo + (hi − lo)·U,  U ~ unif(0,1).
                let u = self.push_src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 }));
                let (lo, hi) = (self.arg_id(lo), self.arg_id(hi));
                let width = self
                    .graph
                    .push(RvNode::Binary(BinOp::Sub, hi, lo), RvKind::Num);
                let scaled = self
                    .graph
                    .push(RvNode::Binary(BinOp::Mul, width, u), RvKind::Num);
                self.graph
                    .push(RvNode::Binary(BinOp::Add, lo, scaled), RvKind::Num)
            }
            Recipe::UniformIntDyn { lo, hi } => {
                // lo + floor(max(hi − lo + 1, 1)·U),  U ~ unif(0,1) → inclusive integers lo..=hi.
                // The `max(·, 1)` clamp (finding B4) makes an inverted per-lane range well-defined:
                // if `hi < lo` on some lane the raw width `hi − lo + 1` is ≤ 0, which without the
                // clamp yields floored *negative* offsets and thus values *below* `lo` (out of any
                // sensible range). Clamping the width to ≥ 1 degenerates that lane to a point mass
                // at `lo` — the same well-defined behavior the constant path gives for `lo == hi`.
                let u = self.push_src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 }));
                let (lo, hi) = (self.arg_id(lo), self.arg_id(hi));
                let diff = self
                    .graph
                    .push(RvNode::Binary(BinOp::Sub, hi, lo), RvKind::Num);
                let one = self.graph.push(RvNode::ConstNum(1.0), RvKind::Num);
                let width_raw = self
                    .graph
                    .push(RvNode::Binary(BinOp::Add, diff, one), RvKind::Num);
                // width = max(width_raw, 1) via select(width_raw < 1, 1, width_raw).
                let narrow = self
                    .graph
                    .push(RvNode::Binary(BinOp::Lt, width_raw, one), RvKind::Bool);
                let width = self.graph.push(
                    RvNode::Select {
                        cond: narrow,
                        a: one,
                        b: width_raw,
                    },
                    RvKind::Num,
                );
                let scaled = self
                    .graph
                    .push(RvNode::Binary(BinOp::Mul, u, width), RvKind::Num);
                let floored = self
                    .graph
                    .push(RvNode::Unary(UnOp::Floor, scaled), RvKind::Num);
                self.graph
                    .push(RvNode::Binary(BinOp::Add, lo, floored), RvKind::Num)
            }
            Recipe::NormalDyn { mu, sigma, int } => {
                // mu + sigma·Z,  Z ~ N(0,1); `int` rounds each lane (normal_int).
                let z = self.push_src(Source::Normal {
                    mu: 0.0,
                    sigma: 1.0,
                });
                let (mu, sigma) = (self.arg_id(mu), self.arg_id(sigma));
                let scaled = self
                    .graph
                    .push(RvNode::Binary(BinOp::Mul, sigma, z), RvKind::Num);
                let val = self
                    .graph
                    .push(RvNode::Binary(BinOp::Add, mu, scaled), RvKind::Num);
                if int {
                    self.graph
                        .push(RvNode::Unary(UnOp::Round, val), RvKind::Num)
                } else {
                    val
                }
            }
            Recipe::ExpDyn { rate, int } => {
                // E / rate,  E ~ Exp(1) → Exp(rate); `int` rounds each lane (exponential_int).
                let e = self.push_src(Source::Exp { rate: 1.0 });
                let rate = self.arg_id(rate);
                let val = self
                    .graph
                    .push(RvNode::Binary(BinOp::Div, e, rate), RvKind::Num);
                if int {
                    self.graph
                        .push(RvNode::Unary(UnOp::Round, val), RvKind::Num)
                } else {
                    val
                }
            }
            Recipe::BernoulliDyn { p } => {
                // (U < p),  U ~ unif(0,1): a bool-RV true with the lane's probability p.
                let u = self.push_src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 }));
                let p = self.arg_id(p);
                self.graph
                    .push(RvNode::Binary(BinOp::Lt, u, p), RvKind::Bool)
            }
            // Handled above with an early return (they yield arrays/complex, not a scalar `id`).
            Recipe::Rotation { .. } => unreachable!("rotation drawn via draw_rotation"),
            Recipe::Permutation { .. } => unreachable!("permutation drawn via draw_permutation"),
            Recipe::Empirical { .. } => unreachable!("empirical drawn via draw_empirical"),
            Recipe::BlockBootstrap { .. } => {
                unreachable!("block_bootstrap drawn via draw_block_bootstrap")
            }
            Recipe::NormalComplex { .. } => {
                unreachable!("normal_complex drawn via the complex path")
            }
        };
        Ok(Value::Dist(id))
    }

    /// Materialize a (possibly random) distribution parameter as a sample-DAG node: a constant folds
    /// to a `ConstNum`; a random parameter reuses its existing node, so every `~` draw of the recipe
    /// shares the SAME per-lane parameter value (with a fresh base draw) — conditional independence
    /// given the parameter.
    fn arg_id(&mut self, a: DistArg) -> RvId {
        match a {
            DistArg::Const(x) => self.graph.push(RvNode::ConstNum(x), RvKind::Num),
            DistArg::Rv(id) => id,
        }
    }

    /// Lower a symbolic input scalar to sample-DAG nodes: an `Input` leaf per input slot (value read
    /// at run time, never baked), and ordinary `Unary`/`Binary`/`ConstNum` nodes for the arithmetic
    /// around it (PLAN-UNIFORM-INPUTS). The mirror of `SigExpr` materialization, scalar. Every op is
    /// pure and deterministic, so simplify/CSE fold the arithmetic while keeping the `Input` leaves
    /// opaque — the value never re-bakes.
    pub(super) fn lower_sym(&mut self, s: &crate::sym::SymExpr) -> RvId {
        use crate::sym::SymExpr;
        match s {
            SymExpr::Input(idx) => self.graph.push(RvNode::Input { idx: *idx }, RvKind::Num),
            SymExpr::Const(c) => self.graph.push(RvNode::ConstNum(*c), RvKind::Num),
            SymExpr::Unary(op, a) => {
                let a = self.lower_sym(a);
                self.graph.push(RvNode::Unary(*op, a), RvKind::Num)
            }
            SymExpr::Binary(op, a, b) => {
                let (a, b) = (self.lower_sym(a), self.lower_sym(b));
                let kind = if matches!(
                    op,
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne
                ) {
                    RvKind::Bool
                } else {
                    RvKind::Num
                };
                self.graph.push(RvNode::Binary(*op, a, b), kind)
            }
        }
    }

    /// Lift a unary op over a random variable. The operand is a `Value::Dist` (the caller's
    /// pre-check guarantees it). Type-checked by `RvKind` with spanned errors before sampling.
    pub(super) fn lift_unary(&mut self, op: UnOp, v: Value, span: Span) -> Result<Value> {
        let id = match v {
            Value::Dist(id) => id,
            _ => unreachable!("lift_unary only reached with a Dist operand"),
        };
        let kind = self.graph.kind(id);
        let result_kind = match op {
            UnOp::Neg => {
                if kind != RvKind::Num {
                    return Err(NoiseError::type_mismatch(
                        format!("cannot apply Neg to {}", kind.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
            UnOp::Not => {
                if kind != RvKind::Bool {
                    return Err(NoiseError::type_mismatch(
                        format!("cannot apply Not to {}", kind.type_name()),
                        span,
                    ));
                }
                RvKind::Bool
            }
            // Math ufuncs need a numeric RV and yield a numeric RV.
            UnOp::Sin
            | UnOp::Cos
            | UnOp::Atan
            | UnOp::Sign
            | UnOp::Round
            | UnOp::Floor
            | UnOp::Ceil
            | UnOp::Exp
            | UnOp::Ln
            | UnOp::Sqrt => {
                if kind != RvKind::Num {
                    return Err(NoiseError::type_mismatch(
                        format!("cannot apply {} to {}", unop_name(op), kind.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
        };
        Ok(Value::Dist(
            self.graph.push(RvNode::Unary(op, id), result_kind),
        ))
    }

    /// Lift a binary op over random variables. At least one operand is a `Value::Dist`;
    /// deterministic operands are folded into `ConstNum`/`ConstBool` graph nodes. Type rules
    /// mirror the deterministic evaluator, on `RvKind`, with spanned errors before sampling.
    pub(super) fn lift_binary(
        &mut self,
        op: BinOp,
        l: Value,
        r: Value,
        span: Span,
    ) -> Result<Value> {
        use BinOp::*;
        let (lid, lk) = self.operand_to_rv(l, span)?;
        let (rid, rk) = self.operand_to_rv(r, span)?;
        let result_kind = match op {
            Add | Sub | Mul | Div | Mod | Pow => {
                if lk != RvKind::Num || rk != RvKind::Num {
                    return Err(NoiseError::type_mismatch(
                        format!("arithmetic on {} and {}", lk.type_name(), rk.type_name()),
                        span,
                    ));
                }
                RvKind::Num
            }
            Lt | Gt | Le | Ge => {
                if lk != RvKind::Num || rk != RvKind::Num {
                    return Err(NoiseError::type_mismatch(
                        format!("cannot compare {} and {}", lk.type_name(), rk.type_name()),
                        span,
                    ));
                }
                RvKind::Bool
            }
            Eq | Ne => {
                if lk != rk {
                    return Err(NoiseError::type_mismatch(
                        format!("cannot compare {} and {}", lk.type_name(), rk.type_name()),
                        span,
                    ));
                }
                RvKind::Bool
            }
            And | Or => {
                if lk != RvKind::Bool || rk != RvKind::Bool {
                    return Err(NoiseError::type_mismatch(
                        format!(
                            "logical operator needs two bool events, got {} and {}",
                            lk.type_name(),
                            rk.type_name()
                        ),
                        span,
                    ));
                }
                RvKind::Bool
            }
        };
        Ok(Value::Dist(
            self.graph.push(RvNode::Binary(op, lid, rid), result_kind),
        ))
    }

    /// Lift `if cond { then } else { else }` where `cond` is a bool random variable. Builds a
    /// per-lane `Select` RV: `cond ? then : else`. An `else` branch is REQUIRED (every lane
    /// needs a value), and the two branches must have the same kind.
    pub(super) fn lift_if(
        &mut self,
        cond: RvId,
        then_b: &Spanned,
        else_b: Option<&Spanned>,
        span: Span,
    ) -> Result<Value> {
        let else_b = else_b.ok_or_else(|| {
            NoiseError::runtime(
                "an `if` over a random variable needs an `else` branch (every sample needs a value)"
                    .to_string(),
                span,
            )
        })?;
        // Both branches are evaluated: a lifted `if` is a value-select, not control flow.
        let then_v = self.eval(then_b)?;
        let else_v = self.eval(else_b)?;
        let (a, ak) = self.operand_to_rv(then_v, then_b.span)?;
        let (b, bk) = self.operand_to_rv(else_v, else_b.span)?;
        if ak != bk {
            return Err(NoiseError::runtime(
                format!(
                    "`if` branches must have the same type, got {} and {}",
                    ak.type_name(),
                    bk.type_name()
                ),
                span,
            ));
        }
        Ok(Value::Dist(
            self.graph.push(RvNode::Select { cond, a, b }, ak),
        ))
    }

    /// Coerce an operand to an `(RvId, RvKind)`. `Dist` reuses its id (structural sharing);
    /// `Num`/`Bool` fold into a const node; `Str`/`Unit` are spanned errors (preserving the
    /// deterministic type-error contract, e.g. for `X + "a"`).
    pub(super) fn operand_to_rv(&mut self, v: Value, span: Span) -> Result<(RvId, RvKind)> {
        match v {
            Value::Dist(id) => Ok((id, self.graph.kind(id))),
            Value::Num(n) => Ok((
                self.graph.push(RvNode::ConstNum(n), RvKind::Num),
                RvKind::Num,
            )),
            // A symbolic input entering the RV graph as a *value*: lower to `RvNode::Input` uniform
            // leaves (never baked), so the compiled kernel is stable across input values.
            Value::Sym(s) => Ok((self.lower_sym(&s), RvKind::Num)),
            // An estimate folds in as its central value (its error is dropped inside the RV).
            Value::Est { val, .. } => Ok((
                self.graph.push(RvNode::ConstNum(val), RvKind::Num),
                RvKind::Num,
            )),
            Value::Bool(b) => Ok((
                self.graph.push(RvNode::ConstBool(b), RvKind::Bool),
                RvKind::Bool,
            )),
            other => Err(NoiseError::runtime(
                format!(
                    "cannot use {} in a random-variable expression",
                    other.type_name()
                ),
                span,
            )),
        }
    }
}
