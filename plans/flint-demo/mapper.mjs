// IntrospectionOut (Noise wasm plot payload) -> Flint ChartAssemblyInput(s).
//
// This is the prototype of the code that would live in the playground (or later
// in Rust): every plot:: payload becomes one or more Flint semantic specs.
// Where a chart needs layers (the fan), we compile each layer as its own
// Flint spec and merge post-compile — the Flint-sanctioned recipe
// ("compile, then edit the backend JSON minimally").
import { assembleVegaLite } from 'flint-chart';

const SIZE = { width: 560, height: 300 };

// dist1: engine already binned the 1e6 draws -> pre-binned Bar chart.
export function dist1ToFlint(p) {
  const { lo, hi, bins } = p.hist;
  const w = (hi - lo) / bins.length;
  const rows = bins.map((count, i) => ({ value: lo + (i + 0.5) * w, count }));
  return {
    title: `hist(${p.label})  n=${p.n}  mean=${p.mean.toFixed(3)}  sd=${p.sd.toFixed(3)}`,
    input: {
      data: { values: rows },
      semantic_types: { value: 'Number', count: 'Count' },
      chart_spec: {
        chartType: 'Bar Chart',
        encodings: { x: 'value', y: 'count' },
        canvasSize: SIZE,
      },
    },
  };
}

// dist2: joint draws -> Scatter Plot.
export function dist2ToFlint(p) {
  const rows = p.points.map(([a, b]) => ({ [p.label]: a, [p.label_b]: b }));
  return {
    title: `scatter(${p.label}, ${p.label_b})  corr=${p.corr.toFixed(3)}`,
    input: {
      data: { values: rows },
      semantic_types: { [p.label]: 'Number', [p.label_b]: 'Number' },
      chart_spec: {
        chartType: 'Scatter Plot',
        encodings: { x: p.label, y: p.label_b },
        chartProperties: { opacity: 0.4 },
        canvasSize: SIZE,
      },
    },
  };
}

// fan: quantile envelope over time -> two Range Areas + median Line,
// each a Flint spec, layered post-compile.
export function fanToFlint(p) {
  const rows = Array.from({ length: p.cols }, (_, t) => ({
    t, q05: p.q05[t], q25: p.q25[t], q50: p.q50[t], q75: p.q75[t], q95: p.q95[t],
  }));
  const band = (lo, hi) => ({
    data: { values: rows },
    semantic_types: { t: 'Number', [lo]: 'Number', [hi]: 'Number' },
    chart_spec: {
      chartType: 'Range Area Chart',
      encodings: { x: 't', y: lo, y2: hi },
      canvasSize: SIZE,
    },
  });
  const median = {
    data: { values: rows },
    semantic_types: { t: 'Number', q50: 'Number' },
    chart_spec: {
      chartType: 'Line Chart',
      encodings: { x: 't', y: 'q50' },
      canvasSize: SIZE,
    },
  };
  return {
    title: `fan(${p.label})  n=${p.n}  q05..q95`,
    layers: [band('q05', 'q95'), band('q25', 'q75'), median],
  };
}

// Compile one payload to a renderable Vega-Lite spec.
export function payloadToVegaLite(p) {
  if (p.type === 'dist1') {
    const { title, input } = dist1ToFlint(p);
    return { title, spec: assembleVegaLite(input) };
  }
  if (p.type === 'dist2') {
    const { title, input } = dist2ToFlint(p);
    return { title, spec: assembleVegaLite(input) };
  }
  if (p.type === 'fan') {
    const { title, layers } = fanToFlint(p);
    const compiled = layers.map(assembleVegaLite);
    // Post-compile merge: shared data + x scale, stacked translucent bands.
    const [outer, inner, mid] = compiled;
    for (const s of [outer, inner]) { s.mark.opacity = 0.25; s.mark.line = false; s.encoding.y.scale.zero = false; }
    outer.encoding.y.title = mid.encoding.y.title = null;
    const layered = {
      width: SIZE.width, height: SIZE.height,
      data: outer.data,
      config: outer.config,
      layer: [
        { mark: outer.mark, encoding: outer.encoding },
        { mark: inner.mark, encoding: inner.encoding },
        { mark: mid.mark, encoding: mid.encoding },
      ],
    };
    return { title: `fan — layered from 3 Flint specs`, spec: layered };
  }
  return null;
}
