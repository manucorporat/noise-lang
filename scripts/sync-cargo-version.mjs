// Noise ships one version number across three artifacts: the `noise-core` and `noise-cli` crates
// and the `@noiselang/core` npm package. Changesets only understands npm, so `packages/core/
// package.json` is the single source of truth and this script copies its version into the Cargo
// workspace right after `changeset version` runs. Nothing else may edit `[workspace.package]
// version` — a hand-bumped Cargo.toml would be silently overwritten on the next release.
//
// Two places in the root Cargo.toml carry the version: the workspace package version (which all
// three crates inherit) and the `noise-core` entry under `[workspace.dependencies]`, whose
// `version` is what crates.io actually resolves for `noise-cli` (the `path` is stripped on
// publish). They must move together or the `noise-cli` publish fails.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const cargoTomlPath = join(root, "Cargo.toml");

const { version } = JSON.parse(
  readFileSync(join(root, "packages/core/package.json"), "utf8"),
);
if (!/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(version)) {
  throw new Error(`Refusing to sync a version Cargo can't parse: ${version}`);
}

const before = readFileSync(cargoTomlPath, "utf8");
let after = replaceOnce(
  before,
  /^(\[workspace\.package\][\s\S]*?^version = ")[^"]+(")$/m,
  version,
  "[workspace.package] version",
);
after = replaceOnce(
  after,
  /^(noise-core = \{ path = "crates\/noise-core", version = ")[^"]+(")/m,
  version,
  "[workspace.dependencies] noise-core version",
);

if (after !== before) {
  writeFileSync(cargoTomlPath, after);
}

// Cargo.lock records the workspace members' own versions, so it goes stale the moment the manifest
// moves. `--workspace` touches only those entries — no incidental dependency upgrades ride along.
execFileSync("cargo", ["update", "--workspace", "--quiet"], {
  cwd: root,
  stdio: "inherit",
});

console.log(`Cargo workspace synced to ${version}`);

function replaceOnce(text, pattern, value, what) {
  const matches = text.match(new RegExp(pattern, pattern.flags + "g")) ?? [];
  if (matches.length !== 1) {
    throw new Error(
      `Expected exactly one ${what} in Cargo.toml, found ${matches.length}. ` +
        `The manifest layout changed — update scripts/sync-cargo-version.mjs.`,
    );
  }
  return text.replace(pattern, `$1${value}$2`);
}
