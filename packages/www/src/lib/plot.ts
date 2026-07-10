// The plot renderer: Flint chart specs in, rendered charts out. Nothing here knows what a histogram
// or a fan chart *is* — the engine already decided (see `noise-core/src/flint.rs`) and shipped a
// semantic `ChartAssemblyInput` per layer. This module only:
//
//   1. sizes each spec to the card it will live in,
//   2. compiles it with stock `flint-chart` (`assembleVegaLite`),
//   3. layers the compiled specs when a plot has more than one (the fan: two quantile bands + a
//      median line), and
//   4. paints them with the site's palette.
//
// Steps 2–3 are Flint's own recipe for a composite chart: compile each spec, then edit the backend
// JSON minimally. Step 4 is a Vega config overlay, not chart design. `reconcileDomain` is the one
// place we work around a library bug; it is documented at its definition.
//
// The whole module is loaded lazily (`import()` from the playground on its first plot), so a
// text-only run never pays for ~350 kB of chart libraries.
import type { ChartSpec } from './noise';

/** A Vega-Lite spec — opaque; we only touch the handful of keys the merge below names. */
type VlSpec = Record<string, any>;

type Libs = {
  assembleVegaLite: (input: unknown) => VlSpec;
  embed: (el: HTMLElement, spec: VlSpec, opts: Record<string, unknown>) => Promise<unknown>;
};

let libs: Promise<Libs> | null = null;

/** Load `flint-chart` + `vega-embed` once, on first use. */
function load(): Promise<Libs> {
  libs ??= Promise.all([import('flint-chart'), import('vega-embed')]).then(([flint, ve]) => ({
    assembleVegaLite: flint.assembleVegaLite as Libs['assembleVegaLite'],
    embed: ve.default as unknown as Libs['embed'],
  }));
  return libs;
}

/**
 * Render a plot's charts into `mount` at `width` px. One spec renders directly; several are layered
 * back-to-front. Rejects if the specs don't compile — the caller shows the plot's `text` instead.
 */
export async function renderCharts(mount: HTMLElement, charts: ChartSpec[], width: number): Promise<void> {
  const { assembleVegaLite, embed } = await load();
  const compiled = charts.map((c) => compile(assembleVegaLite, c, width));
  await embed(mount, merge(compiled), { actions: false, renderer: 'svg', config: palette() });
}

/**
 * Size a spec to the card, compile it, and clean up the result.
 *
 * `canvasSize` in the emitted spec fixes the chart's *aspect*, not its size — only the browser knows
 * how wide the card is. Flint then derives its own plot-area width from that canvas (leaving room
 * for the axis labels it sized), which is why we hand it the full measured width.
 */
function compile(assemble: Libs['assembleVegaLite'], chart: ChartSpec, width: number): VlSpec {
  const input = structuredClone(chart) as any;
  const size = input.chart_spec?.canvasSize;
  if (size?.width > 0) {
    input.chart_spec.canvasSize = { width, height: Math.round((width * size.height) / size.width) };
  }
  const spec = assemble(input);
  reconcileDomain(spec.encoding);
  return spec;
}

/**
 * Work around a `flint-chart@0.2.0` bug: it pins an explicit `domain` on a scale **and** leaves
 * `zero: true` on it whenever the mark is bar-like. Vega honors both, and `zero` silently widens the
 * domain Flint just chose — so a histogram of prices around 100 gets an axis starting at 0, wasting
 * half the canvas. The explicit domain is the more specific of the two contradictory statements (a
 * histogram's x is a *coordinate*; only a bar's length is measured from zero), so it wins.
 *
 * Root cause, for whoever revisits this: `resolveFieldSemantics` computes an annotation-aware
 * `zeroClass` — its own docstring says a domain starting above 0 makes zero arbitrary "regardless of
 * what the base type says" — but `resolveChannelSemantics` never copies that `zeroClass` onto the
 * channel, so `computeZeroDecision` re-derives it from the bare semantic type and forces
 * `zero: true` for every bar-like mark. `resolveZeroClassFromAnnotation` is therefore dead code on
 * all three backends, and no input (`intrinsicDomain`, `includeZero_x`, an explicit `scale`) can
 * reach the decision. Delete this function once that is fixed upstream.
 *
 * Narrow by construction: it only fires where Flint contradicted itself. A scale with no pinned
 * domain — a bar chart's count axis, a driver chart's share axis — keeps its zero baseline, which is
 * exactly where a baseline belongs.
 */
function reconcileDomain(encoding: VlSpec | undefined): void {
  for (const channel of ['x', 'y'] as const) {
    const scale = encoding?.[channel]?.scale;
    if (scale && scale.domain !== undefined && scale.zero) delete scale.zero;
  }
}

/**
 * One compiled spec renders as-is; several become a Vega-Lite `layer`. The specs share an x field
 * and row set by construction (the emitter guarantees it), so the merged scales agree. Their y
 * fields differ (`q05`, `q25`, `path`), and Vega-Lite would title the shared axis with all of them
 * — the emitter puts the meaningful field last, so every layer borrows that title.
 */
function merge(compiled: VlSpec[]): VlSpec {
  const top = compiled[compiled.length - 1];
  if (compiled.length === 1) return strip(top);
  const yTitle = top.encoding?.y?.title ?? top.encoding?.y?.field;
  for (const c of compiled) {
    if (c.encoding?.y) c.encoding.y.title = yTitle;
  }
  return {
    width: compiled[0]._width,
    height: compiled[0]._height,
    config: compiled[0].config,
    layer: compiled.map((c) => ({ data: c.data, mark: c.mark, encoding: c.encoding })),
  };
}

/** Drop the `_`-prefixed internals Flint attaches to a compiled spec; Vega-Lite doesn't know them. */
function strip(spec: VlSpec): VlSpec {
  for (const key of Object.keys(spec)) {
    if (key.startsWith('_')) delete spec[key];
  }
  return spec;
}

/**
 * A Vega config overlay in the site's palette, read live from the CSS custom properties so a chart
 * never drifts from the page around it. Vega-Lite merges this *under* each spec's own `config`, so
 * Flint's layout decisions (label sizing, view dimensions) still win — we only supply color and type.
 */
function palette(): Record<string, unknown> {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  const ink = v('--ink', '#1b1a17');
  const dim = v('--ink-dim', '#585347');
  const rule = v('--rule', '#ddd7c7');
  const accent = v('--accent', '#8a2d4a');
  const mono = v('--mono', 'monospace');
  const text = { labelColor: dim, titleColor: ink, labelFont: mono, titleFont: mono };
  return {
    background: 'transparent',
    font: mono,
    // Marks with no color *encoding* — bars, lines, the fan's bands — take the site's accent.
    // A scale-driven color (the correlation heatmap's diverging ramp) is Flint's call and untouched.
    mark: { color: accent },
    axis: { ...text, gridColor: rule, domainColor: rule, tickColor: rule },
    legend: text,
    view: { stroke: 'transparent' },
  };
}
