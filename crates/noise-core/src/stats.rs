//! Run-time counters for the playground's "engine" readout.
//!
//! Every forcing operation (`P`/`E`/`Var`/`Q`, or a raw `sample`) draws `n` Monte-Carlo samples
//! over a compiled cone. We don't instrument the hot loop (that would tax every backend and slow
//! the very thing we're measuring); instead we record, per forcing, the *static* per-draw cost of
//! the cone — its distinct-node count (`ops`) and source-node count (`sources`), both computed once
//! by [`crate::kernel::cost`] on the simplified graph — and multiply by `n`. So `ops`/`rng_draws`
//! are exact lane-evaluation totals, independent of which backend (interpreter / JIT / wasm) ran.
//!
//! The counters live in a thread-local accumulated on the *driver* thread (the one calling
//! `run_reduction`/`for_each_batch`), before any parallel fan-out — so worker threads never touch
//! them and no synchronization is needed. The wasm playground is single-threaded anyway.

use std::cell::Cell;

/// A snapshot of the counters accumulated since the last [`reset`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    /// Number of forcing operations (`P`/`E`/`Var`/`Q`/`sample`) executed.
    pub forcings: u64,
    /// Total Monte-Carlo draws, summed across every forcing.
    pub samples: u64,
    /// Total per-lane operations executed: Σ over forcings of `draws × cone-node-count`.
    pub ops: u64,
    /// Total random source draws: Σ over forcings of `draws × source-node-count`.
    pub rng_draws: u64,
}

thread_local! {
    static STATS: Cell<RunStats> = const { Cell::new(RunStats { forcings: 0, samples: 0, ops: 0, rng_draws: 0 }) };
}

/// Record one forcing: `n` draws over a cone with `ops` distinct nodes and `sources` source nodes.
pub fn record(n: usize, ops: u64, sources: u64) {
    let n = n as u64;
    STATS.with(|s| {
        let mut cur = s.get();
        cur.forcings += 1;
        cur.samples = cur.samples.saturating_add(n);
        cur.ops = cur.ops.saturating_add(ops.saturating_mul(n));
        cur.rng_draws = cur.rng_draws.saturating_add(sources.saturating_mul(n));
        s.set(cur);
    });
}

/// Clear the counters (called at the start of each `Engine::run`).
pub fn reset() {
    STATS.with(|s| s.set(RunStats::default()));
}

/// Snapshot the counters accumulated so far.
pub fn snapshot() -> RunStats {
    STATS.with(Cell::get)
}
