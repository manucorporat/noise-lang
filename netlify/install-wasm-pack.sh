#!/usr/bin/env bash
# Ensure the Rust + wasm-pack toolchain exists in the Netlify build image.
set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

TOOLCHAIN="${RUST_VERSION:-stable}"

# Netlify ships the rustup shim but with NO default toolchain configured, so
# `cargo`/`rustup target` resolve to nothing. Make sure rustup itself exists,
# then pin and install a real toolchain as the default.
if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup not found — installing Rust toolchain..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$TOOLCHAIN"
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Setting default Rust toolchain to $TOOLCHAIN..."
rustup toolchain install "$TOOLCHAIN" --profile minimal
rustup default "$TOOLCHAIN"

# wasm-pack auto-adds the wasm32 target, but add it explicitly to be safe.
rustup target add wasm32-unknown-unknown

# --- nightly, for the multi-threaded wasm build -------------------------------------------------
#
# `packages/core/scripts/build-wasm.sh` produces TWO artifacts: the portable single-threaded engine
# (stable, above) and a multi-threaded one that fans each Monte-Carlo reduction across a Web Worker
# pool (~6-8x in the browser). The threaded build needs nightly for one specific reason: wasm threads
# require a std compiled with atomics, and the std shipped for wasm32 is not — so it must be rebuilt
# from source with `-Z build-std`, which is nightly-only. `rust-src` is what supplies that source.
#
# If this step is ever removed, the build does not fail loudly: `build-wasm.sh` would error, taking
# the whole site build with it. That is deliberate — a silently single-threaded playground would be a
# 6-8x regression nobody notices.
echo "Installing nightly toolchain (for -Z build-std; the threaded wasm build needs it)..."
rustup toolchain install nightly --profile minimal --component rust-src
rustup target add wasm32-unknown-unknown --toolchain nightly

if command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack already installed: $(wasm-pack --version)"
else
  echo "Installing wasm-pack..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

wasm-pack --version
