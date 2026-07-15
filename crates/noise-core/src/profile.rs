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
//! mechanism. A forcing is one [`Forcing`] guard; the [`span`]/[`note`] calls inside it accumulate
//! into per-forcing thread-locals, and the guard's `Drop` merges them into any installed readable
//! accumulator (see below) and/or prints them.
//!
//! **Two ways to enable it** (see [`enabled`]):
//!   * `NOISE_PROFILE=1` — the native CLI path: prints a `[profile]` block to stderr per forcing.
//!   * an installed [`Profile`] accumulator — a host (the playground) wants the timings back *in the
//!     document*, on any target. Installing the cell is what turns timing on; the guard's `Drop`
//!     folds each forcing's phases/notes into it, and the engine snapshots it into `DocResult`.
//!
//! When neither holds, every entry point is inert (no clock read, no allocation). Unlike the old
//! design this is **not** native-only: the clock is [`web_time::Instant`] on wasm32 (which reads
//! `performance.now()` instead of panicking like `std::time::Instant`), so the browser playground can
//! surface per-phase timings too.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

// `std::time::Instant::now()` panics on `wasm32-unknown-unknown` (no clock); `web_time` reads the
// browser's monotonic `performance.now()` there and is a drop-in `Instant` everywhere else.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

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

/// Whether profiling is active on this thread — the master gate every entry point checks. True when
/// either `NOISE_PROFILE=1` (native stderr path) *or* a readable [`Profile`] accumulator has been
/// [`install`]ed (a host wants the timings in the document). Off, nothing here reads a clock.
pub fn enabled() -> bool {
    env_enabled() || CURRENT.with(|c| c.borrow().is_some())
}

/// Whether `NOISE_PROFILE=1` is set. Read once; compile-time `false` on wasm32 (no env there). This
/// only drives the stderr print — the readable accumulator works on every target.
#[cfg(not(target_arch = "wasm32"))]
fn env_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("NOISE_PROFILE").is_ok_and(|v| v == "1"))
}
#[cfg(target_arch = "wasm32")]
fn env_enabled() -> bool {
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
        // Only reads the clock when profiling is on. `Instant` is `web_time`'s on wasm32, so this is
        // safe there too (it reads `performance.now()` rather than panicking).
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
        // Fold this forcing into the installed readable accumulator, if a host wants the timings in
        // the document (the playground). Accumulates across every forcing in the run.
        CURRENT.with(|c| {
            if let Some(cell) = c.borrow().as_ref() {
                cell.borrow_mut().record(&phases, &notes);
            }
        });
        // The stderr block is the `NOISE_PROFILE=1` native path only — unchanged, and never fires on
        // wasm (no stderr there) even when the readable accumulator is capturing.
        if env_enabled() {
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
}

// === readable accumulator (the document / playground path) =======================================
//
// The per-forcing thread-locals above are scratch: cleared each forcing, summarized on `Drop`. To
// hand a *whole run's* timings back to a host we need a readable sink that survives across forcings
// and can be snapshotted. This mirrors `crate::stats` exactly — a per-engine `Rc` cell that the
// engine *installs* as the thread's active accumulator around a run, into which each `Forcing::drop`
// folds its phases and notes.

/// A readable snapshot of the phase timings a run accumulated — the payload a host surfaces
/// (`DocResult.profile`). Phases are summed across every forcing in the run; `notes` are the gate
/// decisions / backend / cache hit-miss lines, in order.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    /// `(phase, summed_ms, count)` — e.g. `("compile", 4.2, 3)` if three forcings each compiled.
    pub phases: Vec<(String, f64, u32)>,
    /// Free-text notes accumulated across the run.
    pub notes: Vec<String>,
}

impl Timings {
    /// Fold one finished forcing's phases/notes in, summing repeated phase rows by name.
    fn record(&mut self, phases: &[(&'static str, f64, u32)], notes: &[String]) {
        for &(name, ms, count) in phases {
            match self.phases.iter_mut().find(|e| e.0 == name) {
                Some(e) => {
                    e.1 += ms;
                    e.2 += count;
                }
                None => self.phases.push((name.to_string(), ms, count)),
            }
        }
        self.notes.extend(notes.iter().cloned());
    }

    /// Nothing was recorded — no phase timed, no note. A host renders no profiling section for this.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.phases.is_empty() && self.notes.is_empty()
    }
}

/// A per-engine readable timing accumulator, shared so it can be briefly *installed* as the thread's
/// active sink around a run (see the module docs). An [`Engine`](crate::Engine) owns one.
pub type Profile = Rc<RefCell<Timings>>;

/// A fresh empty accumulator for a new engine.
pub fn new_profile() -> Profile {
    Rc::default()
}

thread_local! {
    /// The readable accumulator of the run currently profiling on this thread, if any. Installed by
    /// [`install`]; `None` when no host asked for timings (then [`enabled`] falls back to the env).
    static CURRENT: RefCell<Option<Profile>> = const { RefCell::new(None) };
}

/// RAII guard installing `profile` as this thread's active accumulator, restoring the previous one on
/// drop. Nesting is correct (the sidecar pattern), exactly like [`crate::stats::install`].
#[must_use]
pub struct Installed(Option<Profile>);

/// Install `profile` as the thread's active accumulator for as long as the returned guard lives.
/// Installing it also flips [`enabled`] on for this thread, so timing actually happens.
pub fn install(profile: &Profile) -> Installed {
    CURRENT.with(|c| Installed(c.borrow_mut().replace(profile.clone())))
}

impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.0.take());
    }
}
