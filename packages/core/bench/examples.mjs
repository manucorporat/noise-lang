// End-to-end WASM benchmark over the real programs in `examples/`.
//
// This is the benchmark that matters: almost everyone runs Noise in a browser, and until now
// nothing measured that path. The Rust benches (`crates/noise-core/benches/`) time the native
// interpreter and the Cranelift JIT — neither of which a browser user ever executes. What runs in
// the browser is `noise-wasm` (the engine, compiled to wasm32) driving the **WASM emitter**: per
// forcing it emits a kernel, hands it to the JS host to `WebAssembly.instantiate`, and runs it.
//
// So this harness runs each example through the shipped wasm build under V8 and reports wall time.
// Compile it for Node with:
//
//   PATH="$HOME/.cargo/bin:$PATH" wasm-pack build crates/noise-wasm \
//     --target nodejs --out-dir /tmp/nwasm --out-name noise --release
//
// then:  node packages/core/bench/examples.mjs /tmp/nwasm
//
// Pass `--json` to emit machine-readable results (for tracking across commits).

import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const examplesDir = resolve(here, "../../../examples");

const pkgDir = process.argv[2] ?? "/tmp/nwasm";
const asJson = process.argv.includes("--json");
const { run } = await import(resolve(pkgDir, "noise.js"));

// Median of `reps` runs. Monte Carlo work is the bulk of each program, so run-to-run spread is
// small; the median just drops the occasional GC/scheduler outlier.
const REPS = 5;
function timeMs(src, reps = REPS) {
  const ts = [];
  for (let i = 0; i < reps; i++) {
    const t = performance.now();
    run(src, undefined);
    ts.push(performance.now() - t);
  }
  return ts.sort((a, b) => a - b)[Math.floor(reps / 2)];
}

const cases = readdirSync(examplesDir)
  .filter((f) => f.endsWith(".noise"))
  .map((f) => [f.replace(/\.noise$/, ""), readFileSync(join(examplesDir, f), "utf8")]);

const results = [];
for (const [name, src] of cases) {
  // A failed program would time an error path, not the engine — surface it instead of reporting it.
  const probe = JSON.parse(run(src, undefined));
  const error = probe?.result?.error?.message ?? null;
  results.push({ name, ms: error ? null : timeMs(src), error });
}

if (asJson) {
  console.log(JSON.stringify(results, null, 2));
} else {
  console.log(`\n${"EXAMPLE".padEnd(20)}${"WASM (ms)".padStart(12)}`);
  for (const { name, ms, error } of results) {
    const cell = error ? `ERROR: ${error}` : ms.toFixed(1);
    console.log(`${name.padEnd(20)}${String(cell).padStart(12)}`);
  }
  const total = results.reduce((a, r) => a + (r.ms ?? 0), 0);
  console.log(`${"—".repeat(32)}\n${"total".padEnd(20)}${total.toFixed(1).padStart(12)}`);
}
