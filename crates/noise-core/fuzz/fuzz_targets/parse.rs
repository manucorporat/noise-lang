#![no_main]
//! Fuzz target for the Noise parser (PLAN-TECH-DEBT.md Phase 1, finding J / A1).
//!
//! `noise_core::parser::parse` is a pure `&str -> Result<Program, NoiseError>` — the classic
//! hand-written-Pratt fuzz payoff. The invariant under test is simply: **the parser must never
//! panic or abort on ANY input.** Malformed bytes, deeply nested delimiters, and huge unary/`^`
//! chains must all come back as a typed `Err`, never a crash (the depth guard, finding A1, is what
//! keeps a pathological nesting from overflowing the stack). Run with:
//!
//! ```sh
//! cargo +nightly fuzz run parse
//! ```

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(src) = std::str::from_utf8(data) {
        // The only contract: this returns (Ok or Err) without panicking or aborting.
        let _ = noise_core::parser::parse(src);
    }
});
