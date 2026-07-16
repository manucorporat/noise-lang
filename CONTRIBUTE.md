# Contributing

## Releasing

Noise ships **one version number** across all three artifacts, released together:

- **crates.io** — `noise-core` (engine) and `noise-cli` (the `noise` binary).
- **npm** — `@noiselang/core` (the WASM build + TS bindings).

`noise-wasm` is not published to crates.io — it only exists to be compiled by `wasm-pack` into the
npm package (it still inherits the version, so `cargo` stays consistent).

Releases are driven by [changesets](https://github.com/changesets/changesets) and
`.github/workflows/release.yml`. Nobody edits a version by hand.

### Day to day: add a changeset

Any PR that changes shipped behaviour — Rust or TS, it's all one version — describes itself:

```sh
pnpm changeset
```

Pick `@noiselang/core` (the only listed package: it stands in for the whole release, crates
included), pick patch/minor/major, and write a line aimed at users. That writes a
`.changeset/*.md` file — commit it with the PR. A PR that ships nothing user-visible (docs, tests,
refactors) needs no changeset.

### The release itself

1. Merging a changeset-carrying PR into `master` makes the workflow open (or update) a **📦 Release
   PR**. It bumps `packages/core/package.json`, syncs the Cargo workspace, and writes the changelog.
2. Review that PR — it's the last look at the version and the release notes.
3. Merge it. The workflow then publishes, in this order:
   - `noise-core` → crates.io, then waits for the index (crates.io resolves `noise-cli`'s dependency
     on `noise-core` against the index, not the local path, so the order is load-bearing).
   - `noise-cli` → crates.io.
   - `@noiselang/core` → npm, built fresh from the checkout by `prepublishOnly`.
   - Git tags.

Publishing is idempotent: each registry is asked what already exists, so re-runs and ordinary
pushes to `master` are safe no-ops.

Verify with `cargo info noise-cli` and `npm view @noiselang/core version`.

### How one version stays one version

`packages/core/package.json` is the source of truth. `scripts/sync-cargo-version.mjs` copies it into
the root `Cargo.toml` — both `[workspace.package] version` (which every crate inherits) and the
`noise-core` entry under `[workspace.dependencies]`, whose `version` is what crates.io resolves for
`noise-cli` — then refreshes `Cargo.lock`. Editing `[workspace.package] version` by hand does
nothing: the next release overwrites it.

The website (`packages/www`) is deployed via Netlify, is `private`, and is ignored by changesets — it
never takes a version bump.

### Repo secrets the workflow needs

| Secret | What for |
| --- | --- |
| `CARGO_REGISTRY_TOKEN` | crates.io publish token with rights to both crates |
| `RELEASE_GITHUB_TOKEN` | optional PAT — see below |

**npm needs no secret.** `@noiselang/core` publishes via
[trusted publishing](https://docs.npmjs.com/trusted-publishers): npm is configured to trust this
repo's `release.yml`, and the workflow's `id-token: write` permission mints the short-lived
credential. Set it up once on npmjs.com under the package's **Settings → Trusted publisher**:
GitHub Actions, repo `manucorporat/noise-lang`, workflow filename `release.yml`, no environment.

Three preconditions, all already wired up, each of which silently breaks the publish if disturbed:

- **Never set an `NPM_TOKEN` secret.** The changesets action writes an `.npmrc` auth line whenever it
  sees one and only uses OIDC when it doesn't — a stray token quietly reverts you to token auth.
- **npm >= 11.5.1 on `PATH`.** `changeset publish` detects the pnpm workspace and spawns
  `pnpm publish`, which shells out to plain `npm` off the `PATH` (pnpm bundles no npm of its own),
  and that npm does the OIDC exchange. Node 22 ships npm 10.x, hence the upgrade step.
- **pnpm 10**, pinned via the root `packageManager` field. pnpm 11 regressed OIDC publishing
  ([pnpm/pnpm#11513](https://github.com/pnpm/pnpm/issues/11513)).

Provenance attestation comes free with OIDC on a public repo — no `NPM_CONFIG_PROVENANCE` needed.

#### The optional PAT

Without `RELEASE_GITHUB_TOKEN` everything still works, except the Release PR gets no CI run: pushes
made with the default `GITHUB_TOKEN` don't trigger workflows. A fine-grained PAT scoped to **only
this repo** needs:

| Permission | Access | Why |
| --- | --- | --- |
| Contents | Read and write | push the release branch and tags |
| Pull requests | Read and write | open and update the Release PR |
| Metadata | Read | mandatory, auto-selected |

Nothing else — in particular **not** `Workflows`, which is only needed if a release commit ever edits
`.github/workflows/**` (`ci:version` doesn't). A classic PAT wants the `repo` scope, which is far
broader; prefer fine-grained. Note the PAT's owner shows up as the Release PR's author, and the PAT
expiring is a silent failure mode — the workflow falls back to `GITHUB_TOKEN` and the PR just stops
getting CI.

### Releasing by hand (escape hatch)

If the workflow is broken and something must ship:

```sh
pnpm run ci:version      # applies pending changesets to npm + Cargo
pnpm run ci:publish      # crates.io (in order), then npm, then tags
```

Needs `cargo login`, `npm login`, `wasm-pack` on `PATH`, and a nightly toolchain with `rust-src`
(the threaded wasm build uses `-Z build-std` — see `packages/core/scripts/build-wasm.sh`).

**Backend note for the release announcement (since PLAN-DROP-JIT).** The shipped `noise` binary now
enables `noise-core/gpu`, so it runs forcings on the machine's GPU where profitable (a **4.1×**
speedup on the example corpus over the old interpreter-only binary), falling back to the interpreter
on any machine with no usable GPU adapter. Results stay under the engine's **two-tier contract**:
tier-1 quantities (draws, counts, probabilities) are **bit-identical** across machines, while tier-2
f32 arithmetic (means, variances, and other reductions) is **ULP-close** — a user diffing a stat
between a GPU machine and a no-GPU one can see last-bit differences. This was already true under
`--features gpu`; it is now the default, so say it out loud. (The native Cranelift JIT backend was
retired in the same change — it never shipped in the CLI, so no released binary loses anything.)
