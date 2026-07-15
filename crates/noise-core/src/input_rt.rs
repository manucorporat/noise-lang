//! Runtime input values — the channel from a program's declared `input::real` sliders to the
//! interpreter columns (PLAN-UNIFORM-INPUTS).
//!
//! An `input::real` lowers to an [`RvNode::Input { idx }`](crate::dist::RvNode::Input) whose value is
//! **not** baked into the compiled program (that is what keeps the compiled artifact cache-stable
//! across input *values*). The value is supplied at *run* time instead: this module holds the
//! engine's current input values as **f64** (one per manifest slot — the resolved `input::` value),
//! narrowed to the f32 lane only at the column fill (`Inst::Input`, exactly like `Inst::ConstNum`),
//! installed thread-locally
//! around a forcing region exactly like [`crate::compile_cache`] / [`crate::stats`], so the free
//! forcing functions (`sampler`/`reduce`) can read them without an engine parameter.
//!
//! The forcing driver snapshots [`current`] ONCE on the driver thread into an `Arc<[f64]>` and hands
//! it to each per-worker [`Runner`](crate::backend::Runner); `Send` `Arc` makes the fan-out to
//! `thread::scope`/rayon workers correct (the `CancelToken` precedent). A raw `sampler::*` call
//! outside any engine — or a program with no inputs — snapshots an empty slice.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// The engine-owned cell of current input values, shared so it can be *installed* as the thread's
/// active input values around a forcing region. An [`Engine`](crate::Engine) owns one and pushes to
/// it as `input::` declarations resolve (keeping it parallel to the input manifest).
pub type Inputs = Rc<RefCell<Vec<f64>>>;

/// A fresh empty input-values cell for a new engine.
pub fn new_inputs() -> Inputs {
    Rc::default()
}

thread_local! {
    /// The input values of the engine currently forcing on this thread, if any.
    static CURRENT: RefCell<Option<Inputs>> = const { RefCell::new(None) };
}

/// RAII guard installing an input-values cell as this thread's active values, restoring the previous
/// installation on drop (nesting-correct, like [`crate::compile_cache::Installed`]).
#[must_use]
pub struct Installed(Option<Inputs>);

/// Install `inputs` as the thread's active input values for as long as the returned guard lives.
pub fn install(inputs: &Inputs) -> Installed {
    CURRENT.with(|c| Installed(c.borrow_mut().replace(inputs.clone())))
}

impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.0.take());
    }
}

/// Snapshot the installed input values into an `Arc<[f64]>` for the forcing drivers to hand to
/// workers. An empty slice when no engine is forcing (a raw `sampler::*` call) or the program
/// declared no inputs.
pub fn current() -> Arc<[f64]> {
    CURRENT.with(|c| {
        c.borrow()
            .as_ref()
            .map(|cell| Arc::from(cell.borrow().as_slice()))
            .unwrap_or_else(|| Arc::from(&[] as &[f64]))
    })
}
