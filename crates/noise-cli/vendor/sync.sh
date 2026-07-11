#!/usr/bin/env sh
# Re-vendor the VS Code / Cursor editor extension into the CLI crate (finding G3).
#
# The CLI `include_str!`s a committed copy of the canonical `editors/vscode-noise/` extension under
# `crates/noise-cli/vendor/vscode-noise/` so those files survive the `cargo publish` tarball. The
# build script only *verifies* these are in sync (it never writes into the source tree); this is the
# explicit step that regenerates them. Run it after editing the canonical extension, then commit the
# result.
#
# Usage:  crates/noise-cli/vendor/sync.sh   (from anywhere in the repo)
set -eu

# Resolve paths relative to this script so it works from any working directory.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
canonical="$script_dir/../../../editors/vscode-noise"
vendor="$script_dir/vscode-noise"

if [ ! -d "$canonical" ]; then
  echo "error: canonical extension not found at $canonical" >&2
  exit 1
fi

for rel in package.json language-configuration.json syntaxes/noise.tmLanguage.json; do
  mkdir -p "$(dirname -- "$vendor/$rel")"
  cp "$canonical/$rel" "$vendor/$rel"
  echo "vendored $rel"
done

echo "done — review and commit crates/noise-cli/vendor/vscode-noise/"
