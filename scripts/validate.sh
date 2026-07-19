#!/usr/bin/env bash
# Every gate CI enforces, in one command. Keep in sync with .github/workflows/ci.yml.
set -euo pipefail

# rustup's toolchain first: Homebrew's rustc has no wasm32 std, so the wasm build would fail with a
# misleading "can't find crate for `std`" (same reason packages/www's `wasm` npm script does this).
export PATH="$HOME/.cargo/bin:$PATH"

set -x
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test -p noise-core --features gpu
cargo build --target wasm32-unknown-unknown -p noise-wasm
