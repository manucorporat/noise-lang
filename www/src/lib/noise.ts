// Thin client-side wrapper around the WASM engine. Loads the module once (lazily) and exposes a
// typed `runNoise`. The .wasm is imported as a URL so Vite fingerprints and serves it correctly.
import init, { run as wasmRun, version as wasmVersion } from '../wasm/pkg/noise.js';
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
