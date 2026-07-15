//! Per-forcing phase timing behind `NOISE_PROFILE=1` (PLAN-DROP-JIT D0).
//!
//! The corpus after gpu-only is ~923 ms and 66% of it is three programs, but we don't know where
//! *inside* a forcing the time goes — eval/graph-build, simplify+lower, gate decision, pipeline
//! compile, dispatch, readback, CPU fold, or the plot/introspect sampler passes. This module answers
//! that: it records, per forcing, the wall time of each named phase and prints a compact block when
//! the forcing finishes. Enable with `NOISE_PROFILE=1`; off, every call here is a branch on a cached
//! bool and nothing else (no `Instant`, no allocation).
//!
//! It mirrors [`crate::stats`]'s shape — a thread-local installed on the *driver* thread, where all
//! compilation and dispatch happen (workers only fold) — rather than adding a second install
//! mechanism. A forcing is one [`Forcing`] guard; the [`span`]/[`note`]/[`record_ms`] calls inside it
//! accumulate into the thread-local, and the guard's `Drop` prints and clears them.
//!
//! Native-only in effect: [`enabled`] is compile-time `false` on wasm32 (no env, no clock), so every
//! entry point is inert there and the timing types are never exercised (`Instant::now` would panic on
//! `wasm32-unknown-unknown`). The API is uniform across targets so the cross-target callers
//! (`reduce`/`sampler`) need no `cfg`.

use std::cell::{Cell, RefCell};
use std::time::Instant;

thread_local! {
    /// Accumulated `(phase, summed_ms, count)` for the forcing currently running on this thread.
    /// Linear-scanned (a forcing touches a handful of phases), summed so repeated dispatches fold
    /// into one row.
    static PHASES: RefCell<Vec<(&'static str, f64, u32)>> = const { RefCell::new(Vec::new()) };
    /// Free-text notes for this forcing (the gate decision + which term failed, cache hit/miss).
    static NOTES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// Ops/draw of the current forcing's cone — filled in by [`set_ops`] once the cost is known
    /// (after compile / the GPU emit), read by the [`Forcing`] header.
    static OPS: Cell<u64> = const { Cell::new(0) };
}

/// Whether `NOISE_PROFILE=1` is set. Read once; compile-time `false` on wasm32 (no clock/env there),
/// so the whole module folds away for the browser build.
#[cfg(not(target_arch = "wasm32"))]
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("NOISE_PROFILE").is_ok_and(|v| v == "1"))
}
#[cfg(target_arch = "wasm32")]
pub fn enabled() -> bool {
    false
}

/// Add `ms` to `phase`'s running total (creating the row on first sight).
fn add(name: &'static str, ms: f64) {
    PHASES.with(|p| {
        let mut v = p.borrow_mut();
        match v.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => v.push((name, ms, 1)),
        }
    });
}

/// A running phase timer. Records its elapsed wall time into the thread-local on `Drop`, so a phase
/// is just `let _s = profile::span("compile");` at the top of a scope. Inert (holds `None`, no clock
/// read) when profiling is off.
#[must_use]
pub struct Span {
    name: &'static str,
    start: Option<Instant>,
}

/// Start timing a phase. Cheap no-op guard when profiling is disabled (no `Instant::now`).
pub fn span(name: &'static str) -> Span {
    Span {
        name,
        // `enabled()` is `false` on wasm32, so `Instant::now()` (which panics there) is never reached.
        start: enabled().then(Instant::now),
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(t) = self.start {
            add(self.name, t.elapsed().as_secs_f64() * 1e3);
        }
    }
}

/// Attach a note to the current forcing — the gate decision and which profitability term failed, a
/// pipeline cache hit/miss, the chosen backend.
pub fn note(msg: impl Into<String>) {
    if enabled() {
        NOTES.with(|n| n.borrow_mut().push(msg.into()));
    }
}

/// Record the current forcing's ops/draw (its simplified cone size), shown in the [`Forcing`] header.
/// Called once the cost is known — after `compile_root` on the CPU path, after the WGSL emit on the
/// GPU path.
pub fn set_ops(ops: u64) {
    if enabled() {
        OPS.with(|o| o.set(ops));
    }
}

/// One forcing's timing scope. Create it at the top of a forcing (`run_reduction`, the sampler batch
/// loops); every [`span`]/[`note`] until it drops belongs to this forcing, and its `Drop` prints the
/// accumulated block and clears the thread-local for the next one.
#[must_use]
pub struct Forcing {
    label: &'static str,
    n: usize,
}

/// Begin a forcing timing scope labelled `label`, over `n` draws. Clears any stragglers so the block
/// reflects only this forcing; fill in ops/draw with [`set_ops`] once the cone is compiled.
pub fn forcing(label: &'static str, n: usize) -> Forcing {
    if enabled() {
        PHASES.with(|p| p.borrow_mut().clear());
        NOTES.with(|n| n.borrow_mut().clear());
        OPS.with(|o| o.set(0));
    }
    Forcing { label, n }
}

impl Drop for Forcing {
    fn drop(&mut self) {
        if !enabled() {
            return;
        }
        let phases = PHASES.with(|p| std::mem::take(&mut *p.borrow_mut()));
        let notes = NOTES.with(|n| std::mem::take(&mut *n.borrow_mut()));
        let ops = OPS.with(Cell::get);
        eprintln!("[profile] {} · n={} ops/draw={ops}", self.label, self.n);
        for (name, ms, count) in &phases {
            if *count > 1 {
                eprintln!("[profile]   {name:<14} {ms:9.3} ms  (×{count})");
            } else {
                eprintln!("[profile]   {name:<14} {ms:9.3} ms");
            }
        }
        for note in &notes {
            eprintln!("[profile]   · {note}");
        }
    }
}
