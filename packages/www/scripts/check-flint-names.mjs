// Guard: every chart type the Noise engine names must exist in the pinned flint-chart registry.
//
// The engine emits `ChartAssemblyInput` specs whose `chartType` is a *string* Flint looks up at
// assemble time (`crates/noise-core/src/flint.rs`). Flint's registry names are not schema'd, so a
// dependency bump that renames or drops one would fail at runtime, in the browser, on a chart the
// user asked for. This check moves that failure to build time.
//
// The Rust side has the other half: `flint::tests::assert_well_formed` asserts every emitted spec
// names a type from this same list. Together they pin both ends of the string.
import { vlAllTemplateDefs } from 'flint-chart';

/** Mirrors `REGISTERED` in `crates/noise-core/src/flint.rs`. Keep the two lists in step. */
const EMITTED = ['Bar Chart', 'Scatter Plot', 'Line Chart', 'Range Area Chart', 'Heatmap', 'Ranged Dot Plot'];

const registry = new Set(vlAllTemplateDefs.map((d) => d.chart));
const missing = EMITTED.filter((name) => !registry.has(name));

if (missing.length > 0) {
  console.error(
    `flint-chart no longer registers: ${missing.map((m) => JSON.stringify(m)).join(', ')}\n` +
      `The engine emits these in crates/noise-core/src/flint.rs. Registered names are:\n  ` +
      [...registry].sort().join(' | '),
  );
  process.exit(1);
}

console.log(`flint-chart: all ${EMITTED.length} emitted chart types are registered.`);
