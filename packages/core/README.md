# @noiselang/core

The [Noise](https://noise-lang.dev) probabilistic-language engine, compiled to WebAssembly and
wrapped in a small typed API. Parse and run `.noise` programs — including variable introspection —
from any JavaScript/TypeScript project.

The `.wasm` binary ships **inside** this package. The generated glue references it with
`new URL('noise_bg.wasm', import.meta.url)`, so any bundler that understands that pattern
(Vite, Rollup, webpack 5, esbuild) fingerprints the binary and emits it as an asset of **your**
build. No copying files into `public/`, no CDN, no runtime configuration.

## Install

```sh
npm add @noiselang/core   # or: pnpm add / yarn add
```

## Usage

```ts
import { run } from '@noiselang/core';

const result = await run(`
  let coin = flip(0.5)
  P(coin)
`);

console.log(result.value);   // e.g. "0.4997"
console.log(result.stats);   // { forcings, samples, ops, rng_draws }
console.log(result.elapsedMs);
```

`run` never throws — parse/eval failures come back on `result.error` (with a source span).

The module instantiates lazily on first use. To warm it up ahead of time (e.g. on app mount):

```ts
import { load } from '@noiselang/core';
await load();
```

### Variable introspection

Run a program and interrogate its retained scope without editing the source — describe a
variable's distribution, correlate two, or explain what drives one:

```ts
import { runWithIntrospection } from '@noiselang/core';

const r = await runWithIntrospection(src, [
  { vars: ['height'] },                       // describe(height)
  { vars: ['weight'], explain: true },        // explain(weight)
  { vars: ['height', 'weight'] },             // corr(height, weight)
]);

r.bindings;        // live top-level variables (for a picker)
r.introspections;  // one tagged result per request, in order
r.log;             // Print lines + plot::* charts, in source order
```

Passing `[]` still returns `bindings`, so a plain run can populate a variable picker.

## Bundler notes

- **Vite** — works out of the box for app builds. If Vite's dependency optimizer trips on the
  `import.meta.url` asset reference in dev, add the package to `optimizeDeps.exclude`:
  ```js
  // vite.config.js
  export default { optimizeDeps: { exclude: ['@noiselang/core'] } }
  ```
- **Node.js** — the module targets browsers/bundlers (it uses `fetch` to load the `.wasm`). For a
  server runtime, instantiate the raw `wasm-pack` glue under `@noiselang/core/wasm` yourself.

## API

| Export | Description |
| --- | --- |
| `run(src)` | Parse + evaluate a program → `NoiseResult`. |
| `runWithIntrospection(src, requests)` | Run + resolve introspection requests → `NoiseIntrospectResult`. |
| `version()` | The engine (crate) version string. |
| `load()` | Force one-time WASM instantiation (optional warm-up). |

Full types (`NoiseResult`, `NoiseStats`, `Introspection`, `IntrospectRequest`, …) are exported.

## License

MIT
