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

if command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack already installed: $(wasm-pack --version)"
else
  echo "Installing wasm-pack..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

wasm-pack --version
