//! Verify the vendored editor extension is in sync with the canonical copy (finding G3).
//!
//! The CLI bakes `editors/vscode-noise/` into the binary via `include_str!`, but those files live
//! at the repo root — outside this crate — so they vanish from the `cargo publish` tarball. We
//! instead `include_str!` a **committed** vendored copy under `vendor/vscode-noise/` (which *is*
//! packaged).
//!
//! This script used to *copy* the canonical files over the vendored copy on every build — it
//! **mutated the source tree**, which breaks read-only / hermetic builds, and it demoted a failed
//! copy to a warning so a stale grammar could ship silently. It is now inverted: the build only
//! **reads and verifies**. If the canonical and vendored copies have drifted it fails the build
//! loudly with instructions; it never writes into the source tree. Re-vendoring is an explicit,
//! opt-in step — run `crates/noise-cli/vendor/sync.sh` (or the workspace `xtask`) after editing the
//! canonical extension. During `cargo publish` the repo root isn't present, so the canonical copy
//! is simply absent and the committed vendored snapshot is used as-is.

use std::path::Path;

const FILES: &[&str] = &[
    "package.json",
    "language-configuration.json",
    "syntaxes/noise.tmLanguage.json",
];

fn main() {
    // `editors/` is two levels up from this crate (crates/noise-cli/ -> repo root).
    let canonical = Path::new("../../editors/vscode-noise");
    let vendor = Path::new("vendor/vscode-noise");

    if !canonical.is_dir() {
        // Publishing or building from a packaged tarball: the canonical tree isn't here, so there
        // is nothing to verify — the committed vendored snapshot is authoritative.
        return;
    }

    for rel in FILES {
        let src = canonical.join(rel);
        let dst = vendor.join(rel);
        // Re-run the check whenever either copy changes.
        println!("cargo:rerun-if-changed={}", src.display());
        println!("cargo:rerun-if-changed={}", dst.display());

        let canon = std::fs::read_to_string(&src);
        let vend = std::fs::read_to_string(&dst);
        match (canon, vend) {
            (Ok(canon), Ok(vend)) if canon == vend => {} // in sync
            (Ok(_), Ok(_)) => panic!(
                "vendored editor extension is STALE: {} differs from {}.\n\
                 The vendored copy is what ships in the `cargo publish` tarball, so it must match \
                 the canonical source. Re-vendor with `crates/noise-cli/vendor/sync.sh` and commit \
                 the result. (This build intentionally does NOT rewrite the source tree — finding G3.)",
                dst.display(),
                src.display()
            ),
            (Ok(_), Err(e)) => panic!(
                "vendored editor file {} is missing or unreadable ({e}); re-vendor with \
                 `crates/noise-cli/vendor/sync.sh` and commit it.",
                dst.display()
            ),
            (Err(e), _) => panic!("cannot read canonical editor file {}: {e}", src.display()),
        }
    }
}
