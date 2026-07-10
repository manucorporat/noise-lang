# Flint demo prototype (see ../PLAN-FLINT.md)

Proof that Noise plot payloads → Flint `ChartAssemblyInput` → Vega-Lite render, with
zero hand-written chart code. Validated 2026-07-10 (headless SVG + live in Chrome).

- `mapper.mjs` — **the prototype of `noise-core/src/flint.rs`**: maps each
  `IntrospectionOut` payload (dist1/dist2/fan) to Flint inputs; `fanToFlint` is the
  layered-fan recipe (2× Range Area + median Line, merged post-compile).
- `gen-data.mjs` — runs a real Noise program (barrier option) through the wasm engine
  in Node and dumps the plot payloads to `noise-log.json` (checked in, so the demo
  runs without rebuilding wasm).
- `probe.mjs` — chart-type compatibility probes against `assembleVegaLite`.
- `render-check.mjs` — headless verification: payload → Flint → Vega-Lite → SVG.
- `main.mjs` — browser entry (payloads → cards with embedded charts).

Rebuild/run:

```sh
npm init -y && npm i flint-chart vega vega-lite vega-embed esbuild
node gen-data.mjs         # optional: refresh payloads from current wasm build
node render-check.mjs     # headless: writes chart-*.svg
npx esbuild main.mjs --bundle --format=iife --minify --outfile=bundle.js
# inline bundle.js into an HTML shell with <div id="charts"> and open it
```
