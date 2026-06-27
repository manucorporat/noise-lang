//! Keep the vendored editor extension in sync with the canonical copy.
//!
//! The CLI bakes `editors/vscode-noise/` into the binary via `include_str!`, but
//! those files live at the repo root — outside this crate — so they vanish from the
//! `cargo publish` tarball. We instead `include_str!` a vendored copy under
//! `vendor/vscode-noise/` (which *is* packaged), and this script re-copies the
//! canonical files over it on every normal build. During `cargo publish` verification
//! the repo root isn't present, so the canonical copy is simply absent and the
//! committed vendored snapshot is used as-is.

use std::path::Path;

const FILES: &[&str] = &[
    "package.json",
    "language-configuration.json",
    "syntaxes/noise.tmLanguage.json",
];

fn main() {
    // `editors/` is three levels up from this crate (crates/noise-cli/ -> repo root).
    let canonical = Path::new("../../editors/vscode-noise");
    let vendor = Path::new("vendor/vscode-noise");

    if !canonical.is_dir() {
        // Publishing or building from a packaged tarball: nothing to sync.
        return;
    }

    for rel in FILES {
        let src = canonical.join(rel);
        let dst = vendor.join(rel);
        println!("cargo:rerun-if-changed={}", src.display());
        if let Some(parent) = dst.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::copy(&src, &dst) {
            println!(
                "cargo:warning=failed to sync {} -> {}: {e}",
                src.display(),
                dst.display()
            );
        }
    }
}
