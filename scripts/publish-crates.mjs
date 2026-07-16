// Publishes `noise-core` then `noise-cli` to crates.io at the current workspace version.
//
// This runs from the changesets action's `publish` step, which fires on every push to master that
// has no pending changesets — including pushes that release nothing. So the script must be
// idempotent: it asks crates.io what's already there and skips those, the same way
// `changeset publish` skips npm versions that already exist. Without that, an ordinary
// docs-only commit would fail the workflow on "crate version already uploaded".
//
// Order matters and cannot be parallelised: crates.io resolves `noise-cli`'s dependency on
// `noise-core` against the *index*, not the local path, so the new core must be live and indexed
// before the cli upload is even accepted.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as sleep } from "node:timers/promises";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const version = readFileSync(join(root, "Cargo.toml"), "utf8").match(
  /^\[workspace\.package\][\s\S]*?^version = "([^"]+)"$/m,
)?.[1];
if (!version) throw new Error("No [workspace.package] version in Cargo.toml");

for (const crate of ["noise-core", "noise-cli"]) {
  if (await isPublished(crate, version)) {
    console.log(`${crate}@${version} already on crates.io — skipping`);
    continue;
  }
  console.log(`Publishing ${crate}@${version}`);
  execFileSync("cargo", ["publish", "-p", crate], { cwd: root, stdio: "inherit" });
  // The upload returns before the index serves the new version, and `noise-cli` can't resolve
  // until it does. Poll rather than sleep a fixed guess.
  await waitForIndex(crate, version);
}

async function isPublished(crate, wanted) {
  const res = await fetch(`https://crates.io/api/v1/crates/${crate}/${wanted}`, {
    headers: { "user-agent": "noise-lang release script (github.com/manucorporat/noise-lang)" },
  });
  if (res.status === 404) return false;
  if (!res.ok) throw new Error(`crates.io lookup for ${crate}@${wanted}: HTTP ${res.status}`);
  return true;
}

async function waitForIndex(crate, wanted) {
  for (let attempt = 0; attempt < 30; attempt++) {
    if (await isPublished(crate, wanted)) return;
    console.log(`Waiting for ${crate}@${wanted} to appear on the index…`);
    await sleep(10_000);
  }
  throw new Error(`${crate}@${wanted} never appeared on the crates.io index`);
}
