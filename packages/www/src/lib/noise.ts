// The engine lives in @noiselang/core (the Noise WASM package). This module re-exports its typed
// API so the rest of the site keeps importing from `../lib/noise`, and adds the presentation-only
// formatting helpers that are specific to how this website displays a run (HTML, units, etc.).
// The package API is deliberately terse (`run`, `version`, …) since the import path already says
// "noise". Inside the site we alias to domain-qualified names so a `runNoise(src)` call reads
// clearly amid the surrounding DOM code.
export {
  load as loadNoise,
  run as runNoise,
  meta as noiseMeta,
  version as noiseVersion,
  runWithIntrospection as runNoiseWithIntrospection,
} from '@noiselang/core';
export type {
  NoiseStats,
  NoiseResult,
  Binding,
  ChartSpec,
  Plot,
  IntrospectRequest,
  LogItem,
  NoiseIntrospectResult,
  Knob,
  KnobKind,
  KnobValue,
  KnobOverrides,
  RunOpts,
  NoiseMeta,
} from '@noiselang/core';

import type { NoiseResult } from '@noiselang/core';

/** Compact count: 1.2B / 3.4M / 12k / 950. */
export function fmtCompact(n: number): string {
  if (!isFinite(n) || n <= 0) return '0';
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k';
  return String(Math.round(n));
}

/** Wall-clock time, human-friendly: "0.9 ms" / "12 ms" / "1.20 s". */
export function fmtMs(ms: number): string {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
  if (ms >= 10) return ms.toFixed(0) + ' ms';
  return ms.toFixed(1) + ' ms';
}

/**
 * The metrics tail shown after a demo's engine answer: how many samples it drew, how long the
 * run took, and the resulting throughput — so every demo advertises the engine's speed, not just
 * its answer. Returns an HTML string (wrapped in `.metric` for muted styling).
 */
export function engineMetrics(r: NoiseResult): string {
  const secs = r.elapsedMs / 1000;
  const persec = secs > 0 ? r.stats.samples / secs : 0;
  return `<span class="metric">${fmtCompact(r.stats.samples)} samples · ${fmtMs(r.elapsedMs)} · ${fmtCompact(persec)} samples/s</span>`;
}
