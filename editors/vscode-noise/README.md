# Noise Language — VS Code / Cursor extension

Syntax highlighting for the [Noise](../../LANG.md) probabilistic language (`.noise` files).

It's a pure [TextMate grammar](./syntaxes/noise.tmLanguage.json) — no build step, no
dependencies, no language server. Token categories mirror the website palette in
`packages/www/src/lib/highlight.ts`: keywords, module namespaces (`rand::`, `vec::`, …),
distributions (`unif`, `normal`, `rotation`, …), queries/builtins (`P`, `E`, `Print`, …),
the stochastic-bind operator `~`, comments (`//` and `/* … */`), the line-1 `#!` shebang,
`---` frontmatter (highlighted as YAML), inline and ``` fenced templates with `${…}` holes
(md/latex bodies get their language's highlighting), numbers, and strings.

## Install

Cursor is VS Code-compatible, so the same extension works in both.

### Quick (local dev — no build)

Symlink (or copy) this folder into your editor's extensions directory, then reload the
window (`Cmd/Ctrl+Shift+P` → "Reload Window"):

```sh
# Cursor
ln -s "$PWD/editors/vscode-noise" ~/.cursor/extensions/noise-lang

# VS Code
ln -s "$PWD/editors/vscode-noise" ~/.vscode/extensions/noise-lang
```

(Run from the repo root. Copy instead of symlink if you prefer.)

### Packaged (optional)

```sh
cd editors/vscode-noise
npx @vscode/vsce package      # produces noise-lang-0.1.0.vsix
```

Then `Cmd/Ctrl+Shift+P` → "Extensions: Install from VSIX…".

## Verify

Open `examples/turboquant.noise`. To check a specific token's scope, run
"Developer: Inspect Editor Tokens and Scopes" and hover.
