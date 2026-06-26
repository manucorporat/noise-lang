#!/usr/bin/env bash
# Ensure the Rust + wasm-pack toolchain exists in the Netlify build image.
set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"

# Netlify ships rustup, but make sure cargo is reachable; if not, install Rust.
if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found — installing Rust toolchain..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# wasm-pack auto-adds the wasm32 target, but add it explicitly to be safe.
rustup target add wasm32-unknown-unknown || true

if command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack already installed: $(wasm-pack --version)"
else
  echo "Installing wasm-pack..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

wasm-pack --version
