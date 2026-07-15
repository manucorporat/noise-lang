//! Per-engine compiled-program cache (PLAN-PERF-2 §4).
//!
//! Every forcing (`P`/`E`/`Var`/`Q`, a `describe`/`hist`/`corr` pass, a playground re-run) used to
//! recompile its cone from scratch — `noise_colors` compiles 14 kernels per run, `kelly` 13, and an
//! introspection pass forces one root several times. The compiled artifact is a pure function of
//! the *simplified cone* and the backend's emit-vs-interpret gate decision, so identical forcings
//! can share one compile.
//!
//! **Per-engine, installed thread-locally (the [`crate::stats`] pattern).** The cache is owned by
//! each [`Engine`](crate::Engine) — a REPL/playground reuses it across `run` calls, and it drops
//! with its engine — and *installed* as the thread's active cache around a forcing region, because
//! the forcing paths (`sampler`/`reduce`) are free functions with no engine parameter. Compilation
//! always happens on the driver thread (before any parallel fan-out), so a thread-local install is
//! sound; a raw `sampler::*` call outside any engine simply compiles uncached. Nesting is correct
//! for the same reason it is in `stats`: the guard saves and restores the previous installation.
//!
//! **The key is the full canonical form, not a hash.** Two lookups may hit only if the compiled
//! artifact would be byte-equivalent, so the key serializes the post-simplify cone exactly — every
//! node in the simplified graph's deterministic (post-order) index order, f64 constants bit-exact
//! via `to_bits` (`RvNode` has f64 fields, so it isn't `Hash`/`Eq`-friendly directly), plus the
//! root list (count and order — column order is baked into a joint kernel) and the gate bucket
//! (see `backend::gate_bucket`). Storing the whole byte string as the `HashMap` key means a hit is
//! exact structural equality — a 64-bit hash collision cannot alias two different cones.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::backend::{JointProgram, Program};
use crate::dist::{RvGraph, RvId, RvKind, RvNode, Source};
use crate::kernel::NodeCost;

/// A cache entry: the shared compiled artifact plus the simplified cone's [`NodeCost`] (computed
/// once at compile; a hit must return the same cost without re-walking the cone).
type Entry<P> = (Arc<P>, NodeCost);

/// The cache proper: one map per seam (the value types differ). Bounded by [`MAX_ENTRIES`].
#[derive(Default)]
pub struct CacheState {
    single: HashMap<Vec<u8>, Entry<dyn Program>>,
    joint: HashMap<Vec<u8>, Entry<dyn JointProgram>>,
}

/// Most entries either map holds. **Eviction is clear-on-full**: inserting into a full map drops
/// the whole map rather than tracking recency/insertion order — the working set of a real program
/// is its forcing count (≤ ~15 in the corpus), so 64 is generous and a clear is a rare, cheap
/// worst case (the next run recompiles once). The remaining backends (interpreter bytecode, emitted
/// wasm) free cleanly on drop, so eviction reclaims what it drops; the cache's job is to make
/// re-compiles of the same cone hit instead of churn.
const MAX_ENTRIES: usize = 64;

/// A per-engine cache cell, shared so it can be *installed* as the thread's active cache around a
/// forcing region (see the module docs). An [`Engine`](crate::Engine) owns one.
pub type Cache = Rc<RefCell<CacheState>>;

/// A fresh empty cache for a new engine.
pub fn new_cache() -> Cache {
    Rc::default()
}

thread_local! {
    /// The cache of the engine currently forcing on this thread, if any. Installed by [`install`]
    /// around a forcing region; `None` (compile uncached) when no engine is forcing.
    static CURRENT: RefCell<Option<Cache>> = const { RefCell::new(None) };
}

/// RAII guard installing a cache as this thread's active cache, restoring the previous
/// installation on drop (nesting-correct, like [`crate::stats::Installed`]).
#[must_use]
pub struct Installed(Option<Cache>);

/// Install `cache` as the thread's active compile cache for as long as the returned guard lives.
pub fn install(cache: &Cache) -> Installed {
    CURRENT.with(|c| Installed(c.borrow_mut().replace(cache.clone())))
}

impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.0.take());
    }
}

/// Look up a single-root program in the installed cache. `None` on miss *or* when no cache is
/// installed — the caller compiles either way.
pub(crate) fn lookup_single(key: &[u8]) -> Option<Entry<dyn Program>> {
    CURRENT.with(|c| {
        let cur = c.borrow();
        let cache = cur.as_ref()?;
        let state = cache.borrow();
        state.single.get(key).cloned()
    })
}

/// Store a freshly compiled single-root program. A no-op when no cache is installed.
pub(crate) fn store_single(key: Vec<u8>, program: &Arc<dyn Program>, cost: NodeCost) {
    CURRENT.with(|c| {
        if let Some(cache) = c.borrow().as_ref() {
            let mut state = cache.borrow_mut();
            if state.single.len() >= MAX_ENTRIES {
                state.single.clear(); // clear-on-full — see MAX_ENTRIES
            }
            state.single.insert(key, (program.clone(), cost));
        }
    });
}

/// Look up a joint (multi-root) program in the installed cache.
pub(crate) fn lookup_joint(key: &[u8]) -> Option<Entry<dyn JointProgram>> {
    CURRENT.with(|c| {
        let cur = c.borrow();
        let cache = cur.as_ref()?;
        let state = cache.borrow();
        state.joint.get(key).cloned()
    })
}

/// Store a freshly compiled joint program. A no-op when no cache is installed.
pub(crate) fn store_joint(key: Vec<u8>, program: &Arc<dyn JointProgram>, cost: NodeCost) {
    CURRENT.with(|c| {
        if let Some(cache) = c.borrow().as_ref() {
            let mut state = cache.borrow_mut();
            if state.joint.len() >= MAX_ENTRIES {
                state.joint.clear(); // clear-on-full — see MAX_ENTRIES
            }
            state.joint.insert(key, (program.clone(), cost));
        }
    });
}

/// Canonical byte form of a (post-simplify cone, roots, gate bucket) triple — the cache key.
///
/// `graph` must be the **simplified** graph: simplify rebuilds the cone into a fresh compact arena
/// in deterministic post-order (children before parents, sources in RNG-consumption order), so
/// serializing nodes `0..len` in index order *is* a canonical form — two forcings whose simplified
/// cones are identical produce identical bytes, and every backend's artifact is a pure function of
/// exactly these bytes (plus the gate bucket). f64 fields go through `to_bits`, so `0.0` and
/// `-0.0` — equal under `==`, different in the artifact — key differently. Node kinds are included
/// (they're an input to lowering, cheap to be exact about); the roots' count and order close the
/// key because a joint kernel bakes its column order.
pub(crate) fn key(graph: &RvGraph, roots: &[RvId], gate: bool) -> Vec<u8> {
    // ~18 bytes/node covers every variant but a wide Gather; one realloc worst case.
    let mut out = Vec::with_capacity(graph.len() * 18 + roots.len() * 4 + 8);
    out.push(gate as u8);
    push_u32(&mut out, graph.len() as u32);
    for i in 0..graph.len() {
        let id = RvId(i as u32);
        match graph.kind(id) {
            RvKind::Num => out.push(0),
            RvKind::Bool => out.push(1),
            RvKind::Arr(n) => {
                out.push(2);
                push_u32(&mut out, n);
            }
        }
        match graph.node(id) {
            RvNode::Src(src) => {
                out.push(0);
                push_source(&mut out, src);
            }
            RvNode::ConstNum(x) => {
                out.push(1);
                push_f64(&mut out, *x);
            }
            RvNode::ConstBool(b) => out.extend_from_slice(&[2, *b as u8]),
            RvNode::Unary(op, a) => {
                out.extend_from_slice(&[3, *op as u8]);
                push_id(&mut out, *a);
            }
            RvNode::Binary(op, a, b) => {
                out.extend_from_slice(&[4, *op as u8]);
                push_id(&mut out, *a);
                push_id(&mut out, *b);
            }
            RvNode::Select { cond, a, b } => {
                out.push(5);
                push_id(&mut out, *cond);
                push_id(&mut out, *a);
                push_id(&mut out, *b);
            }
            RvNode::Gather { elems, index } => {
                out.push(6);
                push_u32(&mut out, elems.len() as u32);
                for &e in elems.iter() {
                    push_id(&mut out, e);
                }
                push_id(&mut out, *index);
            }
            RvNode::Permutation { n } => {
                out.push(7);
                push_u32(&mut out, *n);
            }
            RvNode::Rotation { d } => {
                out.push(8);
                push_u32(&mut out, *d);
            }
            RvNode::ArrIndex { arr, index } => {
                out.push(9);
                push_id(&mut out, *arr);
                push_id(&mut out, *index);
            }
            RvNode::ArrDraw { n, src } => {
                out.push(10);
                push_u32(&mut out, *n);
                push_source(&mut out, src);
            }
            RvNode::ArrElem { arr, k } => {
                out.push(11);
                push_id(&mut out, *arr);
                push_u32(&mut out, *k);
            }
            // A Scan (G4c): single-arena, so its body nodes are already serialized in id order above;
            // here we only need its shape — `trip`, the carried inits / placeholders / nexts, the
            // optional index placeholder, and the kinds. Two loops that differ anywhere key differently.
            RvNode::Scan { body } => {
                out.push(12);
                push_u32(&mut out, body.trip);
                push_u32(&mut out, body.inits.len() as u32);
                for slot in body
                    .inits
                    .iter()
                    .chain(body.placeholders.iter())
                    .chain(body.nexts.iter())
                {
                    push_id(&mut out, *slot);
                }
                match body.index_ph {
                    Some(ph) => {
                        out.push(1);
                        push_id(&mut out, ph);
                    }
                    None => out.push(0),
                }
                for &k in body.kinds.iter() {
                    out.push(match k {
                        RvKind::Num => 0,
                        RvKind::Bool => 1,
                        RvKind::Arr(_) => 2,
                    });
                }
            }
            RvNode::ScanOut { scan, slot } => {
                out.push(13);
                push_id(&mut out, *scan);
                push_u32(&mut out, *slot);
            }
            RvNode::Placeholder { slot } => {
                out.push(14);
                push_u32(&mut out, *slot);
            }
        }
    }
    push_u32(&mut out, roots.len() as u32);
    for &r in roots {
        push_id(&mut out, r);
    }
    out
}

/// Serialize a [`Source`]: tag + parameter f64 bits. Every parameter is significant — two sources
/// differing in any parameter bit compile to different sampling code.
fn push_source(out: &mut Vec<u8>, src: &Source) {
    match src {
        Source::Uniform(u) => {
            out.push(0);
            push_f64(out, u.lo);
            push_f64(out, u.hi);
        }
        Source::UniformInt { lo, hi } => {
            out.push(1);
            push_f64(out, *lo);
            push_f64(out, *hi);
        }
        Source::Normal { mu, sigma } => {
            out.push(2);
            push_f64(out, *mu);
            push_f64(out, *sigma);
        }
        Source::Exp { rate } => {
            out.push(3);
            push_f64(out, *rate);
        }
        Source::Poisson { lambda } => {
            out.push(4);
            push_f64(out, *lambda);
        }
        Source::Geometric { p } => {
            out.push(5);
            push_f64(out, *p);
        }
    }
}

fn push_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn push_id(out: &mut Vec<u8>, id: RvId) {
    push_u32(out, id.0);
}

fn push_f64(out: &mut Vec<u8>, x: f64) {
    out.extend_from_slice(&x.to_bits().to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp;
    use crate::backend::{compile_root, probe};
    use crate::dist::Uniform;
    use crate::eval::Engine;
    use crate::value::Value;

    /// `X * c` over one uniform source — a minimal cone whose canonical form depends on `c`'s
    /// exact bit pattern (`x * 0` is deliberately NOT folded by simplify, so `c` survives).
    fn mul_graph(c: f64) -> (RvGraph, RvId) {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let k = g.push(RvNode::ConstNum(c), RvKind::Num);
        let root = g.push(RvNode::Binary(BinOp::Mul, x, k), RvKind::Num);
        (g, root)
    }

    fn est(v: Value) -> f64 {
        match v {
            Value::Est { val, .. } => val,
            other => panic!("expected an estimate, got {other:?}"),
        }
    }

    /// The headline behavior: forcing the same cone twice through the public engine path compiles
    /// once (each `run` appends fresh graph nodes, but the simplified cone — hence the key — is
    /// identical), and a structurally different cone compiles again.
    #[test]
    fn engine_repeated_forcing_compiles_once() {
        let mut eng = Engine::new();
        eng.run("use rand; X ~ unif(0, 1)").unwrap();
        let c0 = probe::compiles();
        eng.run("P(X > 0.5)").unwrap();
        let c1 = probe::compiles();
        assert!(c1 > c0, "the first forcing must compile");
        eng.run("P(X > 0.5)").unwrap();
        assert_eq!(probe::compiles(), c1, "an identical forcing must hit");
        eng.run("P(X > 0.25)").unwrap();
        assert!(
            probe::compiles() > c1,
            "a different constant is a different cone — no false hit"
        );
    }

    /// Caching must be observationally invisible: a cache hit (second query, same engine) and a
    /// cache miss (fresh engine, same program) return bit-identical estimates — same seed, same
    /// draws, same artifact.
    #[test]
    fn cache_hits_are_observationally_invisible() {
        let prog = "use rand; X ~ normal(0, 1); E(X + X * X)";
        let mut a = Engine::new();
        let first = est(a.run(prog).unwrap());
        let second = est(a.run("E(X + X * X)").unwrap()); // cache hit
        let mut b = Engine::new();
        let fresh = est(b.run(prog).unwrap()); // cache miss (fresh engine)
        assert_eq!(second.to_bits(), first.to_bits(), "hit changed the result");
        assert_eq!(fresh.to_bits(), first.to_bits(), "miss changed the result");
    }

    /// Repeated introspection over one variable (`describe` forces its root several times per
    /// call) stops recompiling on the second pass — the cache's motivating workload.
    #[test]
    fn describe_repeat_adds_no_compiles() {
        let mut eng = Engine::new();
        eng.run("use rand; X ~ normal(0, 1)").unwrap();
        eng.run("describe(X)").unwrap();
        let c1 = probe::compiles();
        eng.run("describe(X)").unwrap();
        assert_eq!(probe::compiles(), c1, "repeat describe must be all hits");
    }

    /// The joint seam: one joint pass compiles once, a repeat hits, and hit results are
    /// bit-identical to an uncached (no cache installed) run at the same seed. The cached artifact is
    /// the multi-root bytecode interpreter on native (or the emitted wasm joint kernel on wasm).
    #[test]
    fn joint_pass_hits_cache_and_matches_uncached() {
        let mut g = RvGraph::default();
        let x = g.push(
            RvNode::Src(Source::Uniform(Uniform { lo: 0.0, hi: 1.0 })),
            RvKind::Num,
        );
        let two = g.push(RvNode::ConstNum(2.0), RvKind::Num);
        let y = g.push(RvNode::Binary(BinOp::Mul, x, two), RvKind::Num);
        let n = 150_000;
        let uncached = crate::sampler::grid_moments(&g, &[x, y], n, 7).unwrap(); // no cache installed

        let cache = new_cache();
        let _i = install(&cache);
        let c0 = probe::compiles();
        let first = crate::sampler::grid_moments(&g, &[x, y], n, 7).unwrap();
        let c1 = probe::compiles();
        assert_eq!(c1 - c0, 1, "one joint compile");
        let second = crate::sampler::grid_moments(&g, &[x, y], n, 7).unwrap();
        assert_eq!(probe::compiles(), c1, "repeat joint pass must hit");
        for ((a, b), u) in first.iter().zip(&second).zip(&uncached) {
            assert_eq!(a.mean.to_bits(), b.mean.to_bits());
            assert_eq!(a.mean.to_bits(), u.mean.to_bits());
            assert_eq!(a.variance.to_bits(), u.variance.to_bits());
        }
    }

    /// A draw count that flips the emit-vs-interpret gate is a different key — the low-draw
    /// interpreter artifact must not be served for a high-draw query (or vice versa). Within a
    /// bucket, different raw counts share the entry.
    #[test]
    fn gate_flip_is_not_a_stale_hit() {
        let (g, root) = mul_graph(3.0);
        let cache = new_cache();
        let _i = install(&cache);
        let c0 = probe::compiles();
        let _ = compile_root(&g, root, 1_000); // below the wasm gate
        let _ = compile_root(&g, root, 200_000); // above it
        // Native has no CPU codegen gate now the JIT is gone (one bucket → the second call hits);
        // only the wasm build gates (`MIN_DRAWS_WASM`), where the two counts flip the bucket. This
        // test runs on native, so `expected` is 1.
        let expected = if cfg!(target_arch = "wasm32") { 2 } else { 1 };
        assert_eq!(probe::compiles() - c0, expected);
        // Different raw counts in the same buckets: all hits.
        let _ = compile_root(&g, root, 2_000);
        let _ = compile_root(&g, root, 300_000);
        assert_eq!(probe::compiles() - c0, expected);
    }

    /// Collision paranoia: two cones differing ONLY in one constant's bit pattern (`0.0` vs
    /// `-0.0`, equal under `==`) must key differently — the canonical form is bit-exact.
    #[test]
    fn zero_and_negative_zero_do_not_collide() {
        let cache = new_cache();
        let _i = install(&cache);
        let c0 = probe::compiles();
        let (g1, r1) = mul_graph(0.0);
        let (g2, r2) = mul_graph(-0.0);
        let _ = compile_root(&g1, r1, 1_000);
        let _ = compile_root(&g2, r2, 1_000);
        assert_eq!(probe::compiles() - c0, 2, "-0.0 must not hit the 0.0 entry");
        // And each repeats as a hit against its own entry.
        let _ = compile_root(&g1, r1, 1_000);
        let _ = compile_root(&g2, r2, 1_000);
        assert_eq!(probe::compiles() - c0, 2);
    }

    /// The size cap holds: pushing past `MAX_ENTRIES` distinct cones clears rather than growing
    /// without bound (a pathological REPL session stays bounded).
    #[test]
    fn cache_is_bounded() {
        let cache = new_cache();
        let _i = install(&cache);
        for k in 0..(MAX_ENTRIES + 8) {
            let (g, root) = mul_graph(k as f64 + 2.0);
            let _ = compile_root(&g, root, 1_000);
        }
        let len = cache.borrow().single.len();
        assert!(len <= MAX_ENTRIES, "cache grew past the cap: {len}");
        assert!(len > 0, "clear-on-full still re-inserts the newest entry");
    }
}
