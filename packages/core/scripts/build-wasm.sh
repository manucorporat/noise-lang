#!/usr/bin/env bash
# Build the two wasm artifacts @noiselang/core ships.
#
#   wasm/        — the portable, single-threaded engine. Stable Rust, no special headers, works in
#                  any browser, any bundler, and in Node. This is the fallback, and for most pages it
#                  is what actually runs.
#   wasm-mt/     — the same engine with a multi-threaded reducer (rayon over a Web Worker pool).
#
# Why two. Threads in wasm need SharedArrayBuffer, which browsers only hand to a **cross-origin
# isolated** page (COOP/COEP headers). A library cannot impose those headers on the app that
# installs it, so the threaded build can never be the only build — `pool.ts` feature-detects
# `crossOriginIsolated` at runtime and loads whichever one the page can actually use.
#
# The threaded build additionally needs a nightly toolchain and a std rebuilt with atomics: the
# shipped wasm32 std is compiled without them, so `std::thread`/rayon cannot work against it.
# That's `-Z build-std`, which is why this is a script and not a plain `wasm-pack build`.
#
# Both builds produce identical *answers* — the reducer merges chunks in index order, so the result
# is bit-identical for any thread count (see `reduce.rs`). Threads change wall clock, nothing else.
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
export PATH="$HOME/.cargo/bin:$PATH"

echo "==> single-threaded (stable, universal)"
wasm-pack build crates/noise-wasm \
  --target web --out-dir ../../packages/core/wasm --out-name noise --release
rm -f packages/core/wasm/.gitignore

echo "==> multi-threaded (nightly + atomics + build-std)"
# The flag set is not optional folklore — wasm-bindgen's thread transform demands each piece, and
# omitting any one fails in a different, unhelpful way (each of these was hit in turn):
#   +atomics,+bulk-memory,+mutable-globals  the threads proposal itself
#   --shared-memory                         makes linear memory a SharedArrayBuffer. Without it the
#                                           module builds fine but `initThreadPool` dies at runtime
#                                           with "DataCloneError: #<Memory> could not be cloned" —
#                                           you cannot postMessage a non-shared Memory to a worker.
#   --import-memory                         wasm-bindgen asserts `mem.import.is_some()`; workers are
#                                           instantiated *against* the main thread's memory.
#   --max-memory                            a shared memory must be bounded.
#   --export=__heap_base                    wasm-bindgen needs it to inject the per-thread id.
#   --export=__wasm_init_tls,__tls_*        per-thread TLS setup, run on each worker at startup.
# -Z build-std rebuilds std with atomics: the shipped wasm32 std has none, so threads can't work
# against it. That's what forces nightly.
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals \
  -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=2147483648 \
  -C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls \
  -C link-arg=--export=__tls_base -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align" \
  rustup run nightly wasm-pack build crates/noise-wasm \
    --target web --out-dir ../../packages/core/wasm-mt --out-name noise --release \
    -- --features wasm-threads,gpu -Z build-std=std,panic_abort
rm -f packages/core/wasm-mt/.gitignore

echo "==> done: packages/core/wasm (st) + packages/core/wasm-mt (mt)"
