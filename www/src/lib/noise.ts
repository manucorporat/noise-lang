// Thin client-side wrapper around the WASM engine. Loads the module once (lazily) and exposes a
// typed `runNoise`. The .wasm is imported as a URL so Vite fingerprints and serves it correctly.
import init, {
  run as wasmRun,
  run_with_introspection as wasmRunIntrospect,
  version as wasmVersion,
} from '../wasm/pkg/noise.js';
import wasmUrl from '../wasm/pkg/noise_bg.wasm?url';

let ready: Promise<void> | null = null;

/** Initialize the WASM module exactly once; subsequent calls await the same promise. */
export function loadNoise(): Promise<void> {
  if (!ready) {
    ready = init({ module_or_path: wasmUrl }).then(() => undefined);
  }
  return ready;
}

/** Run-time counters from the engine — what the program actually computed (see Rust `stats`). */
export interface NoiseStats {
  /** Forcing operations (`P`/`E`/`Var`/`Q`/`sample`) the program ran. */
  forcings: number;
  /** Total Monte-Carlo draws across all forcings. */
  samples: number;
  /** Total per-lane operations executed (Σ draws × cone-node-count). */
  ops: number;
  /** Total random source draws (Σ draws × source-node-count). */
  rng_draws: number;
}

export interface NoiseResult {
  ok: boolean;
  /** Display form of the program's final value, or null for `unit` / on error. */
  value: string | null;
  /** Everything `Print` emitted, in order. */
  output: string;
  /** Error message (with a source span) on failure, else null. */
  error: string | null;
  /** Engine run-time counters (zeroed if a program forced no sampling). */
  stats: NoiseStats;
  /** Wall-clock time of the WASM `run` call, in milliseconds (measured here, not in Rust). */
  elapsedMs: number;
}

const ZERO_STATS: NoiseStats = { forcings: 0, samples: 0, ops: 0, rng_draws: 0 };

/** Parse + evaluate a Noise program. Never throws — failures come back in `error`. */
export async function runNoise(src: string): Promise<NoiseResult> {
  await loadNoise();
  // Time only the engine call (module load is excluded — it's a one-off, not per-run cost).
  const t0 = performance.now();
  const raw = wasmRun(src);
  const elapsedMs = performance.now() - t0;
  const parsed = JSON.parse(raw) as Omit<NoiseResult, 'elapsedMs'>;
  // The defensive serialization-error fallback omits `stats`; default it so callers needn't guard.
  return { ...parsed, stats: parsed.stats ?? ZERO_STATS, elapsedMs };
}

export async function noiseVersion(): Promise<string> {
  await loadNoise();
  return wasmVersion();
}

// --- variable introspection (the "inspect without editing the code" path) --------------------

/** A live top-level binding of the program — what the variable picker lists. */
export interface Binding {
  name: string;
  /** Value kind, e.g. `dist<number>` / `dist<bool>` (introspectable) or `number` / `array`. */
  kind: string;
}

/** A histogram to draw: equal-width buckets `bins` spanning `[lo, hi]`. */
export interface Hist {
  lo: number;
  hi: number;
  bins: number[];
}

/** One introspection result, tagged by `type` (mirrors the Rust `IntrospectionOut`). */
export type Introspection =
  | {
      type: 'dist1';
      label: string;
      n: number;
      mean: number;
      sd: number;
      min: number;
      max: number;
      q05: number;
      q25: number;
      q50: number;
      q75: number;
      q95: number;
      boolean: boolean;
      hist: Hist;
      head: number[];
    }
  | {
      type: 'dist2';
      label: string;
      label_b: string;
      n: number;
      corr: number;
      cov: number;
      mean_a: number;
      mean_b: number;
      sd_a: number;
      sd_b: number;
      points: [number, number][];
    }
  | {
      type: 'explain';
      label: string;
      sd: number;
      drivers: { name: string; corr: number; share: number }[];
    }
  | { type: 'value'; label: string; val: number; se: number }
  | {
      type: 'grid';
      label: string;
      rows: number;
      cols: number;
      /** true → vector (series view); false → matrix (heatmap view). */
      series: boolean;
      mean: number[];
      sd: number[];
    }
  | { type: 'corrmatrix'; label: string; n: number; corr: number[] }
  | { type: 'error'; error: string };

/** A request: one variable (`describe`/`explain`) or two (`corr`), with an optional condition. */
export interface IntrospectRequest {
  /** One or two variable *expressions*, evaluated in the program's scope. */
  vars: string[];
  /** Optional condition expression — `describe(v | given)`. */
  given?: string;
  /** When true, the one-variable request becomes `explain(v)` (driver fan-out). */
  explain?: boolean;
  /** When true, a single (array) variable becomes `corr(v)` (element correlation heatmap). */
  correlate?: boolean;
}

/** One item in the program's output stream: a `Print` line or a `plot::*` chart, in source order. */
export type LogItem = { kind: 'text'; text: string } | { kind: 'plot'; plot: Introspection };

export interface NoiseIntrospectResult extends NoiseResult {
  /** The program's live top-level variables (for the picker). */
  bindings: Binding[];
  /** One result per request, in request order. */
  introspections: Introspection[];
  /** The output stream in source order: `Print` lines and `plot::*` charts, interleaved. */
  log: LogItem[];
}

/**
 * Run a program and, against its retained scope, resolve a list of introspection requests — the
 * sidecar that powers the playground inspector. Passing `[]` requests still returns `bindings`, so a
 * plain Run can populate the variable picker. Never throws; failures surface in `error` (and
 * per-request failures as `{ type: 'error' }` entries).
 */
export async function runNoiseWithIntrospection(
  src: string,
  requests: IntrospectRequest[],
): Promise<NoiseIntrospectResult> {
  await loadNoise();
  const t0 = performance.now();
  const raw = wasmRunIntrospect(src, JSON.stringify(requests));
  const elapsedMs = performance.now() - t0;
  const parsed = JSON.parse(raw) as Omit<NoiseIntrospectResult, 'elapsedMs'>;
  return {
    ...parsed,
    stats: parsed.stats ?? ZERO_STATS,
    bindings: parsed.bindings ?? [],
    introspections: parsed.introspections ?? [],
    log: parsed.log ?? [],
    elapsedMs,
  };
}

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
