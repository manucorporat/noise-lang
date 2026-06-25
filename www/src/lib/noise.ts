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

export interface NoiseResult {
  ok: boolean;
  /** Display form of the program's final value, or null for `unit` / on error. */
  value: string | null;
  /** Everything `Print` emitted, in order. */
  output: string;
  /** Error message (with a source span) on failure, else null. */
  error: string | null;
}

/** Parse + evaluate a Noise program. Never throws — failures come back in `error`. */
export async function runNoise(src: string): Promise<NoiseResult> {
  await loadNoise();
  return JSON.parse(wasmRun(src)) as NoiseResult;
}

export async function noiseVersion(): Promise<string> {
  await loadNoise();
  return wasmVersion();
}
