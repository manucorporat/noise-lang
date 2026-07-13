//! Defines the `threaded` cfg: "this build has a parallel executor for `reduce`".
//!
//! That is true on every native target, and on wasm32 only with the `wasm-threads` feature (which
//! pulls in rayon over a Web Worker pool). Writing it once here keeps `reduce.rs` saying what it
//! means — `#[cfg(threaded)]` — instead of repeating `any(not(target_arch = "wasm32"), feature =
//! "wasm-threads")` at every site, where the reader has to reconstruct the intent each time.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(threaded)");
    println!("cargo:rerun-if-changed=build.rs");

    let is_wasm = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32");
    let has_wasm_threads = std::env::var_os("CARGO_FEATURE_WASM_THREADS").is_some();
    if !is_wasm || has_wasm_threads {
        println!("cargo::rustc-cfg=threaded");
    }
}
