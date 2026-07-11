//! Run-time counters for the playground's "engine" readout.
//!
//! Every forcing operation (`P`/`E`/`Var`/`Q`, a raw `sample`, or a joint introspection pass —
//! `describe`/`hist`/`corr`/`fan`) draws `n` Monte-Carlo samples over a compiled cone. We don't
//! instrument the hot loop (that would tax every backend and slow the very thing we're measuring);
//! instead we record, per forcing, the *static* per-draw cost of the cone — its distinct-node count
//! (`ops`) and source-node count (`sources`), both computed once by [`crate::kernel::cost`] on the
//! simplified graph — and multiply by `n`. So `ops`/`rng_draws` are exact lane-evaluation totals,
//! independent of which backend (interpreter / JIT / wasm) ran.
//!
//! **Per-engine, not global (finding B8).** The counters are owned by each [`Engine`](crate::Engine)
//! as a shared [`Counters`] cell. Around a forcing region an `Engine` *installs* its cell as the
//! thread's active recorder (see [`install`]); the free-function forcing paths (`sampler`/`reduce`
//! and the joint drivers) then accumulate into *that* engine's counters via [`record`]. This fixes
//! the old thread-local-global design where two engines on one thread (the documented playground
//! sidecar pattern) corrupted each other's counts, and where reading an engine's stats coupled to
//! whichever thread last forced. Recording happens on the *driver* thread (before any parallel
//! fan-out), so worker threads never touch the counters and no synchronization is needed. Reading an
//! engine's stats is just reading its own cell.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// A snapshot of the counters accumulated since the last reset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    /// Number of forcing operations (`P`/`E`/`Var`/`Q`/`sample`/joint pass) executed.
    pub forcings: u64,
    /// Total Monte-Carlo draws, summed across every forcing.
    pub samples: u64,
    /// Total per-lane operations executed: Σ over forcings of `draws × cone-node-count`.
    pub ops: u64,
    /// Total random source draws: Σ over forcings of `draws × source-node-count`.
    pub rng_draws: u64,
}

/// A per-engine counter cell, shared so it can be briefly *installed* as the thread's active
/// recorder around a forcing region (see the module docs). An [`Engine`](crate::Engine) owns one.
pub type Counters = Rc<Cell<RunStats>>;

/// A fresh zeroed counter cell for a new engine.
pub fn new_counters() -> Counters {
    Rc::new(Cell::new(RunStats::default()))
}

thread_local! {
    /// The counters of the engine currently forcing on this thread, if any. Installed by
    /// [`install`] around a forcing region; `None` when no engine is forcing.
    static CURRENT: RefCell<Option<Counters>> = const { RefCell::new(None) };
}

/// Record one forcing into the thread's currently-installed counters: `n` draws over a cone with
/// `ops` distinct nodes and `sources` source nodes. A no-op when nothing is installed (a raw
/// `sampler::*` call outside any engine simply isn't accounted — nothing reads those stats).
pub fn record(n: usize, ops: u64, sources: u64) {
    let n = n as u64;
    CURRENT.with(|c| {
        if let Some(cell) = c.borrow().as_ref() {
            let mut cur = cell.get();
            cur.forcings += 1;
            cur.samples = cur.samples.saturating_add(n);
            cur.ops = cur.ops.saturating_add(ops.saturating_mul(n));
            cur.rng_draws = cur.rng_draws.saturating_add(sources.saturating_mul(n));
            cell.set(cur);
        }
    });
}

/// RAII guard installing `counters` as this thread's active recorder, restoring the previous
/// installation on drop. Nesting is correct: an engine that forces inside another engine's run (the
/// sidecar pattern) saves and restores its parent's installation.
#[must_use]
pub struct Installed(Option<Counters>);

/// Install `counters` as the thread's active recorder for as long as the returned guard lives.
pub fn install(counters: &Counters) -> Installed {
    CURRENT.with(|c| Installed(c.borrow_mut().replace(counters.clone())))
}

impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.0.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_targets_the_installed_engine_only() {
        let a = new_counters();
        let b = new_counters();
        {
            let _g = install(&a);
            record(100, 3, 1);
        }
        {
            let _g = install(&b);
            record(50, 2, 1);
        }
        assert_eq!(a.get().samples, 100);
        assert_eq!(a.get().forcings, 1);
        assert_eq!(b.get().samples, 50);
        // Nothing installed now: a record is dropped, not misattributed.
        record(999, 9, 9);
        assert_eq!(a.get().samples, 100);
        assert_eq!(b.get().samples, 50);
    }

    #[test]
    fn nested_installs_restore_the_parent() {
        let outer = new_counters();
        let inner = new_counters();
        let _o = install(&outer);
        record(10, 1, 1);
        {
            let _i = install(&inner);
            record(20, 1, 1);
        }
        // Back to `outer` after the inner guard drops.
        record(5, 1, 1);
        assert_eq!(inner.get().samples, 20);
        assert_eq!(outer.get().samples, 15);
    }
}
