import { assembleVegaLite, vlAllTemplateDefs } from 'flint-chart';

// 1. What chart types are registered on the Vega-Lite backend?
const defs = typeof vlAllTemplateDefs === 'function' ? vlAllTemplateDefs() : vlAllTemplateDefs;
console.log('== registered vega-lite chart types ==');
console.log((Array.isArray(defs) ? defs : Object.values(defs)).map(d => d.name ?? d.chartType ?? d.id ?? d).join(' | '));

function tryChart(label, input) {
  try {
    const out = assembleVegaLite(input);
    const spec = out.spec ?? out;
    console.log(`\n== ${label}: OK ==`);
    console.log(JSON.stringify(spec).slice(0, 600));
    if (out.warnings?.length) console.log('warnings:', JSON.stringify(out.warnings));
  } catch (e) {
    console.log(`\n== ${label}: FAIL == ${e.message}`);
  }
}

// 2. Histogram from RAW samples (Flint bins) — small n only.
const samples = Array.from({ length: 200 }, (_, i) => ({ x: Math.sin(i * 12.9898) * 43758.5453 % 3 }));
tryChart('Histogram (raw samples, flint bins)', {
  data: { values: samples },
  semantic_types: { x: 'Number' },
  chart_spec: { chartType: 'Histogram', encodings: { x: 'x' }, chartProperties: { binCount: 20 } },
});

// 3. PRE-BINNED histogram as Bar chart (what our engine would emit for n=1e6).
const bins = Array.from({ length: 20 }, (_, i) => ({ bin: -3 + i * 0.3, count: Math.round(1000 * Math.exp(-((i - 10) ** 2) / 20)) }));
tryChart('Bar (pre-binned hist)', {
  data: { values: bins },
  semantic_types: { bin: 'Number', count: 'Count' },
  chart_spec: { chartType: 'Bar Chart', encodings: { x: 'bin', y: 'count' } },
});

// 4. Line chart (a path / expectation curve).
const line = Array.from({ length: 52 }, (_, t) => ({ t, price: 100 * Math.exp(0.001 * t + 0.03 * Math.sin(t)) }));
tryChart('Line (path)', {
  data: { values: line },
  semantic_types: { t: 'Number', price: 'Price' },
  chart_spec: { chartType: 'Line Chart', encodings: { x: 't', y: 'price' } },
});

// 5. Range Area — the fan chart core: q05..q95 bands over time.
const fan = Array.from({ length: 52 }, (_, t) => ({
  t,
  q05: 100 - t * 0.8, q25: 100 - t * 0.3, q50: 100 + t * 0.05, q75: 100 + t * 0.5, q95: 100 + t * 1.1,
}));
tryChart('Range Area (fan band q05-q95)', {
  data: { values: fan },
  semantic_types: { t: 'Number', q05: 'Number', q95: 'Number' },
  chart_spec: { chartType: 'Range Area Chart', encodings: { x: 't', y: 'q05', y2: 'q95' } },
});

// 6. Multi-series line via array-y (fan quantile lines fallback).
tryChart('Multi-series line (quantile lines)', {
  data: { values: fan },
  semantic_types: { t: 'Number', q05: 'Number', q25: 'Number', q50: 'Number', q75: 'Number', q95: 'Number' },
  chart_spec: { chartType: 'Line Chart', encodings: { x: 't', y: ['q05', 'q25', 'q50', 'q75', 'q95'] } },
});

// 7. Scatter (corr/scatter introspection view).
const sc = Array.from({ length: 300 }, (_, i) => ({ a: i % 17, b: (i % 17) * 2 + (i % 7) }));
tryChart('Scatter', {
  data: { values: sc },
  semantic_types: { a: 'Number', b: 'Number' },
  chart_spec: { chartType: 'Scatter Plot', encodings: { x: 'a', y: 'b' } },
});

// 8. ECDF from raw samples.
tryChart('ECDF', {
  data: { values: samples },
  semantic_types: { x: 'Number' },
  chart_spec: { chartType: 'ECDF', encodings: { x: 'x' } },
});
