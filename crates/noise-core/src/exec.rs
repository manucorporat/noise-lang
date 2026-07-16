//! Cooperative cancellation and soft-stop (PLAN-PREGPU Track A, PLAN-PRECISION Track H).
//!
//! A long forcing — `P(event, 10_000_000)` — can run for seconds. This module lets a *caller*
//! ask it to stop, at two strengths:
//!
//! * [`CancelToken::cancel`] — **hard**: the reducer aborts with
//!   [`ErrorKind::Cancelled`](crate::error::ErrorKind::Cancelled) and everything is discarded.
//!   Teardown, page navigation, engine drop. A cancelled reduction has folded only *some* of its
//!   chunks, and that partial answer must never escape as though it were a real estimate —
//!   returning `Err` makes that structural.
//! * [`CancelToken::stop`] — **soft** (Track H): workers stop claiming chunks, the driver folds
//!   the chunks that completed and returns `Ok` with the accumulator's *true* count plus a
//!   [`StopCause`] marker. Statistically legitimate: with counter keying every chunk is an iid
//!   block of draws, and *which* chunks completed depends on timing, never on the drawn values —
//!   so a fold over any subset of completed chunks is an unbiased estimate with an honest se.
//!   The engine treats a tripped soft flag as "finish this query with what you have, then skip
//!   the remaining forcings and return a complete document".
//!
//! The **run deadline** (`max_time`, PLAN-PRECISION) rides the soft path: it is installed
//! thread-locally around a run (like the token itself) and checked at the same per-chunk cadence;
//! a passed deadline trips the token's soft flag with [`StopCause::Time`].
//!
//! **The token is installed, not threaded.** An [`Engine`](crate::Engine) installs its token as the
//! thread's active one around a run (see [`install`]) — the same shape [`crate::stats`] and
//! [`crate::compile_cache`] use — so the ~30 forcing call sites don't each grow a parameter.
//!
//! **Who can set the flags.** Natively, any thread: a CLI Ctrl-C handler, an embedding host, a
//! watchdog. In the browser nothing can set them while Rust is running — except through shared
//! memory, which is exactly how the wasm host does it (a SAB-backed cell the main thread writes and
//! the per-chunk check reads). The token itself is deliberately web-agnostic (no `wasm-bindgen`).
//!
//! **What a cancelled run leaves behind.** The `Engine`'s scope is *partially updated*: bindings
//! that completed before the abort persist. Treat a hard-cancelled engine as stale and rebuild from
//! a fresh one. A soft-stopped engine is different: every completed binding is a real value and the
//! partial estimates are honest (wider se), so its scope remains usable — that is the point.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use std::cell::Cell;
use std::cell::RefCell;

// `web-time` re-exports `std::time` on native and reads `performance.now()` on wasm32-unknown
// (which has no std clock) — the deadline check must work on both.
use web_time::Instant;

/// Why a run was soft-stopped: a host/user asked ([`CancelToken::stop`]), or the run's `max_time`
/// deadline passed. The distinction only affects the warning message — both mean "fold what
/// completed and report it honestly".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCause {
    /// A host called `stop()` (a user's "I've waited long enough").
    User,
    /// The run's `max_time` deadline passed.
    Time,
}

/// The token's shared state: the hard flag plus the soft one (with its cause). Separate atomics —
/// the hard path must stay a single relaxed load per chunk, and the soft cause is only read when
/// the soft flag is up.
#[derive(Debug, Default)]
struct Flags {
    cancel: AtomicBool,
    /// 0 = running, 1 = soft-stopped by a user, 2 = soft-stopped by the deadline.
    stop: AtomicU8,
}

/// A shared "please stop" flag pair. Cloning shares the flags (it is an `Arc` inside), so a host
/// can hand one clone to a watchdog thread and keep another to cancel with.
///
/// `Send + Sync`: the parallel reducer's worker threads each read it, so it cannot be an `Rc` like
/// the per-engine `stats` cell.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<Flags>);

impl CancelToken {
    /// A fresh, un-cancelled token.
    pub fn new() -> Self {
        CancelToken(Arc::new(Flags::default()))
    }

    /// Request **hard** cancellation (discard everything). Idempotent, callable from any thread.
    pub fn cancel(&self) {
        // `Release` pairs with the `Acquire` in `is_cancelled`, but nothing is published alongside
        // the flag — the ordering is immaterial here and `Relaxed` would do. `Release`/`Acquire`
        // costs nothing measurable at one load per 16,384 samples and keeps the pairing obvious.
        self.0.cancel.store(true, Ordering::Release);
    }

    /// Has hard cancellation been requested?
    pub fn is_cancelled(&self) -> bool {
        self.0.cancel.load(Ordering::Acquire)
    }

    /// Request a **soft** stop (Track H): finish folding what completed, report it with an honest
    /// se, skip the remaining forcings. Idempotent; a hard `cancel` still wins over this.
    pub fn stop(&self) {
        self.stop_with(StopCause::User);
    }

    /// Trip the soft flag with an explicit cause (the deadline check uses [`StopCause::Time`]).
    /// First cause wins — a deadline firing after a user stop must not relabel it.
    pub fn stop_with(&self, cause: StopCause) {
        let v = match cause {
            StopCause::User => 1,
            StopCause::Time => 2,
        };
        let _ = self
            .0
            .stop
            .compare_exchange(0, v, Ordering::Release, Ordering::Relaxed);
    }

    /// The soft-stop cause, if the soft flag is up.
    pub fn stop_cause(&self) -> Option<StopCause> {
        match self.0.stop.load(Ordering::Acquire) {
            1 => Some(StopCause::User),
            2 => Some(StopCause::Time),
            _ => None,
        }
    }

    /// Reset the soft flag (the hard flag deliberately has no reset here — see
    /// [`Engine::reset_cancel`](crate::Engine::reset_cancel)). A host that reuses an engine after a
    /// soft stop starts the next run fresh.
    pub fn reset_stop(&self) {
        self.0.stop.store(0, Ordering::Release);
    }
}

/// The **host stop cell** (PLAN-PRECISION Track H, browser half): a process-global soft-stop flag
/// an embedder can flip from *outside* the engine's control flow. It exists for exactly one host —
/// the browser. There the engine worker is **blocked** inside a synchronous forcing (no
/// `postMessage` can reach it), but on the threaded (`wasm-threads`) build the wasm linear memory
/// IS a `SharedArrayBuffer` — so the main thread writes this cell directly (`noise-wasm` exports
/// its address; the JS `stop()` is one `Atomics.store` into it) and the per-chunk stop check reads
/// it beside the token and the deadline. Native hosts have real threads and never need it (they
/// clone the [`CancelToken`]); it stays 0 there and costs one relaxed load per 16,384 samples.
static HOST_STOP: AtomicU32 = AtomicU32::new(0);

/// The stop cell's address, for the wasm surface to hand to JS. (An `AtomicU32` rather than a bool
/// so the JS side can `Atomics.store` through an `Int32Array` view — it is 4-byte aligned.)
pub fn host_stop_cell() -> *const AtomicU32 {
    &HOST_STOP
}

/// Clear the host stop cell — the engine calls this at the start of each run, beside
/// [`CancelToken::reset_stop`]: a click that raced the *previous* run's completion must not stop
/// the next one.
pub fn reset_host_stop() {
    HOST_STOP.store(0, Ordering::Release);
}

fn host_stopped() -> bool {
    HOST_STOP.load(Ordering::Acquire) != 0
}

thread_local! {
    /// The token of the run currently executing on this thread, if any (see [`install`]).
    static CURRENT: RefCell<Option<CancelToken>> = const { RefCell::new(None) };
    /// The wall-clock deadline of the run currently executing on this thread, if any (`max_time`,
    /// PLAN-PRECISION). Checked at the same per-chunk cadence as the token; passing it trips the
    /// token's soft flag with [`StopCause::Time`].
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// RAII guard restoring the previously-installed token on drop (so nested/reentrant runs — an
/// engine forcing inside another engine's run — can't clobber each other).
pub struct Installed(Option<CancelToken>);

impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.0.take());
    }
}

/// Install `token` as this thread's active token for as long as the returned guard lives.
pub fn install(token: &CancelToken) -> Installed {
    CURRENT.with(|c| Installed(c.borrow_mut().replace(token.clone())))
}

/// RAII guard restoring the previously-installed deadline on drop (nested runs, like [`Installed`]).
pub struct InstalledDeadline(Option<Instant>);

impl Drop for InstalledDeadline {
    fn drop(&mut self) {
        DEADLINE.with(|c| c.set(self.0.take()));
    }
}

/// Install an absolute wall-clock `deadline` for this thread's current run (`None` = no deadline).
/// The reducer's per-chunk check compares against it; past it, the run soft-stops with
/// [`StopCause::Time`].
pub fn install_deadline(deadline: Option<Instant>) -> InstalledDeadline {
    DEADLINE.with(|c| InstalledDeadline(c.replace(deadline)))
}

/// This thread's installed deadline, if any. The parallel reducer reads it ONCE on the driver
/// thread (thread-locals are invisible inside `thread::scope`) and hands the `Instant` to workers.
pub fn deadline() -> Option<Instant> {
    DEADLINE.with(Cell::get)
}

/// A clone of this thread's installed token, if any. The parallel reducer calls this ONCE on the
/// driver thread and hands the clone to its workers — a thread-local is invisible inside
/// `thread::scope`, so the `Arc` has to travel explicitly.
pub fn current() -> Option<CancelToken> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Has this thread's installed token been hard-cancelled? `false` when nothing is installed — a
/// raw `sampler::*` call outside any engine is simply never cancelled.
pub fn cancelled() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(CancelToken::is_cancelled))
}

/// The soft-stop state of this thread's run: the token's soft flag, or — checked here, at the same
/// cadence — a passed deadline (which also trips the token so the whole run sees it). `None` when
/// running normally or when nothing is installed.
pub fn stop_cause() -> Option<StopCause> {
    let token = CURRENT.with(|c| c.borrow().clone());
    stop_cause_of(token.as_ref(), deadline())
}

/// Check a `(token, deadline)` pair a worker carried across a scope boundary (the parallel drivers
/// can't reach the thread-locals) — [`stop_cause`] is this over the thread's installed pair. Also
/// reads the [host stop cell](host_stop_cell) at the same cadence. Trips the token's soft flag when
/// the deadline/cell fired, so every other worker sees it as a flag load rather than a clock read.
pub fn stop_cause_of(token: Option<&CancelToken>, deadline: Option<Instant>) -> Option<StopCause> {
    if let Some(cause) = token.and_then(CancelToken::stop_cause) {
        return Some(cause);
    }
    if host_stopped() {
        if let Some(t) = token {
            t.stop_with(StopCause::User);
        }
        return Some(StopCause::User);
    }
    if deadline.is_some_and(|d| Instant::now() >= d) {
        if let Some(t) = token {
            t.stop_with(StopCause::Time);
        }
        return Some(StopCause::Time);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_scoped_and_restores() {
        let a = CancelToken::new();
        assert!(!cancelled(), "nothing installed → never cancelled");
        {
            let _g = install(&a);
            assert!(!cancelled());
            a.cancel();
            assert!(cancelled());
        }
        assert!(!cancelled(), "the guard must restore the previous (none)");
    }

    #[test]
    fn nested_installs_restore_the_outer_token() {
        let outer = CancelToken::new();
        let inner = CancelToken::new();
        let _g = install(&outer);
        {
            let _g2 = install(&inner);
            inner.cancel();
            assert!(cancelled(), "the inner token is active");
        }
        // The inner engine's cancellation must not leak into the outer run.
        assert!(!cancelled(), "outer token is back and un-cancelled");
    }

    #[test]
    fn a_clone_shares_the_flag_across_threads() {
        let t = CancelToken::new();
        let worker = t.clone();
        let h = std::thread::spawn(move || {
            while !worker.is_cancelled() {
                std::hint::spin_loop();
            }
            true
        });
        t.cancel();
        assert!(
            h.join().unwrap(),
            "cancel must be visible on another thread"
        );
    }

    #[test]
    fn soft_stop_is_separate_from_cancel_and_first_cause_wins() {
        let t = CancelToken::new();
        assert_eq!(t.stop_cause(), None);
        t.stop();
        assert_eq!(t.stop_cause(), Some(StopCause::User));
        assert!(!t.is_cancelled(), "soft stop must not hard-cancel");
        // A later deadline must not relabel the user's stop.
        t.stop_with(StopCause::Time);
        assert_eq!(t.stop_cause(), Some(StopCause::User));
        t.reset_stop();
        assert_eq!(t.stop_cause(), None);
    }

    #[test]
    fn a_passed_deadline_trips_the_soft_flag_with_time() {
        let t = CancelToken::new();
        let _g = install(&t);
        let _d = install_deadline(Some(Instant::now() - std::time::Duration::from_millis(1)));
        assert_eq!(stop_cause(), Some(StopCause::Time));
        // …and the token itself was tripped, so workers see it as a flag load.
        assert_eq!(t.stop_cause(), Some(StopCause::Time));
    }

    #[test]
    fn no_deadline_means_no_stop() {
        let t = CancelToken::new();
        let _g = install(&t);
        assert_eq!(stop_cause(), None);
    }
}
