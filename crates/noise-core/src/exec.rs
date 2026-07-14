//! Cooperative cancellation (PLAN-PREGPU Track A).
//!
//! A long forcing — `P(event, 10_000_000)` — can run for seconds. This module lets a *caller*
//! ask it to stop: a [`CancelToken`] is a shared flag; the reducer checks it once per 16,384-sample
//! chunk (a relaxed load, free against the sampling it guards) and a tripped token aborts the whole
//! forcing with [`ErrorKind::Cancelled`](crate::error::ErrorKind::Cancelled).
//!
//! **The token is installed, not threaded.** An [`Engine`](crate::Engine) installs its token as the
//! thread's active one around a run (see [`install`]) — the same shape [`crate::stats`] and
//! [`crate::compile_cache`] use — so the ~30 forcing call sites don't each grow a parameter. What
//! they DO grow is a `Result`: a cancelled reduction has only folded *some* of its chunks, and that
//! partial answer must never escape as if it were a real estimate. Returning `Err` is what makes
//! that structural rather than a discipline everyone has to remember.
//!
//! **Who can set the flag.** Natively, any thread: a CLI Ctrl-C handler, an embedding host, a
//! watchdog. In the browser nothing can set it while Rust is running — `abort()` is JS, and JS only
//! runs when the engine returns to the event loop — so the web path additionally needs the
//! cooperative yield that Track A's async spine introduces (A3). The token itself is ready for both;
//! it is deliberately web-agnostic (no `wasm-bindgen` here).
//!
//! **What a cancelled run leaves behind.** The `Engine`'s scope is *partially updated*: bindings
//! that completed before the abort persist. Treat a cancelled engine as stale and rebuild from a
//! fresh one — stated here rather than discovered (the playground's introspection sidecar relies on
//! scope persisting across runs, so it must NOT reuse an engine it cancelled).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::cell::RefCell;

/// A shared "please stop" flag. Cloning shares the flag (it is an `Arc` inside), so a host can hand
/// one clone to a watchdog thread and keep another to cancel with.
///
/// `Send + Sync`: the parallel reducer's worker threads each read it, so it cannot be an `Rc` like
/// the per-engine `stats` cell.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A fresh, un-cancelled token.
    pub fn new() -> Self {
        CancelToken(Arc::new(AtomicBool::new(false)))
    }

    /// Request cancellation. Idempotent, callable from any thread.
    pub fn cancel(&self) {
        // `Release` pairs with the `Acquire` in `is_cancelled`, but nothing is published alongside
        // the flag — the ordering is immaterial here and `Relaxed` would do. `Release`/`Acquire`
        // costs nothing measurable at one load per 16,384 samples and keeps the pairing obvious.
        self.0.store(true, Ordering::Release);
    }

    /// Has cancellation been requested?
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

thread_local! {
    /// The token of the run currently executing on this thread, if any (see [`install`]).
    static CURRENT: RefCell<Option<CancelToken>> = const { RefCell::new(None) };
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

/// A clone of this thread's installed token, if any. The parallel reducer calls this ONCE on the
/// driver thread and hands the clone to its workers — a thread-local is invisible inside
/// `thread::scope`, so the `Arc` has to travel explicitly.
pub fn current() -> Option<CancelToken> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Has this thread's installed token been tripped? `false` when nothing is installed — a raw
/// `sampler::*` call outside any engine is simply never cancelled.
pub fn cancelled() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(CancelToken::is_cancelled))
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
        assert!(h.join().unwrap(), "cancel must be visible on another thread");
    }
}
