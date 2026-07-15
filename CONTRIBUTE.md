# Contributing

## Releasing

There are two release surfaces, versioned independently:

- **crates.io** — `noise-core` (engine) and `noise-cli` (the `noise` binary). Both share the workspace version in the root `Cargo.toml`.
- **npm** — `@noiselang/core` (the WASM build + TS bindings), versioned in `packages/core/package.json`.

`noise-wasm` is **not** published to crates.io — it only exists to be compiled by `wasm-pack` into the npm package.

### Rust crates (noise-core, noise-cli)

One-time setup: `cargo login` with a crates.io token that has publish rights.

1. Bump the version in **two places**:
   - `[workspace.package] version` in the root `Cargo.toml` (both crates inherit it).
   - The pinned dependency in `crates/noise-cli/Cargo.toml`:
     ```toml
     noise-core = { path = "../noise-core", version = "X.Y.Z" }
     ```
     This must match the new workspace version, or the `noise-cli` publish will fail (crates.io strips `path` and uses `version`).
2. Sanity check:
   ```sh
   cargo test
   cargo publish -p noise-core --dry-run
   ```
3. Publish **core first, then cli** (cli depends on the new core being on the index):
   ```sh
   cargo publish -p noise-core
   cargo publish -p noise-cli
   ```
   If the `noise-cli` publish complains it can't find the new `noise-core`, wait a minute for the index to catch up and retry.
4. Commit the version bump and tag:
   ```sh
   git commit -am "release: vX.Y.Z"
   git tag vX.Y.Z && git push --tags
   ```

Verify with `cargo info noise-cli` or by installing: `cargo install noise-cli`.

**Backend note for the release announcement (since PLAN-DROP-JIT).** The shipped `noise` binary now
enables `noise-core/gpu`, so it runs forcings on the machine's GPU where profitable (a **4.1×**
speedup on the example corpus over the old interpreter-only binary), falling back to the interpreter
on any machine with no usable GPU adapter. Results stay under the engine's **two-tier contract**:
tier-1 quantities (draws, counts, probabilities) are **bit-identical** across machines, while tier-2
f32 arithmetic (means, variances, and other reductions) is **ULP-close** — a user diffing a stat
between a GPU machine and a no-GPU one can see last-bit differences. This was already true under
`--features gpu`; it is now the default, so say it out loud. (The native Cranelift JIT backend was
retired in the same change — it never shipped in the CLI, so no released binary loses anything.)

### npm package (@noiselang/core)

Requirements: a Rust toolchain with the `wasm32-unknown-unknown` target, `wasm-pack` on `PATH` (or in `~/.cargo/bin`), and `npm login` as a user with publish rights to the `@noiselang` scope.

1. Bump `version` in `packages/core/package.json`.
2. Publish from the package directory:
   ```sh
   cd packages/core
   pnpm publish
   ```
   `prepublishOnly` runs the full build automatically (`wasm-pack` on `crates/noise-wasm` → `packages/core/wasm`, then `tsc` → `dist`), and `publishConfig.access: public` is already set, so no extra flags are needed.
3. To check what would ship without publishing: `pnpm publish --dry-run` (should contain `dist/` and `wasm/`, including the `.wasm` binary).
4. Commit the bump (tags are reserved for crate releases; prefix npm tags as `npm-vX.Y.Z` if you want one).

Verify with `npm view @noiselang/core version`.

### Which one to release?

- Changed anything under `crates/noise-core` or `crates/noise-cli` → release the crates.
- Changed the engine (`noise-core`), `crates/noise-wasm`, or `packages/core/src` → also release npm, since the `.wasm` is built from the current checkout, not from crates.io.
- The website (`packages/www`) is deployed via Netlify and needs no version bump.
