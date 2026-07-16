// The shared-memory protocol between the engine worker (which runs the synchronous Monte-Carlo
// forcing) and the main thread (which owns the WebGPU device). This is the whole reason browser GPU
// needs anything beyond "call WebGPU": the forcing is synchronous Rust and WebGPU is async-only, so
// the worker cannot await a dispatch. Instead it writes the request into a `SharedArrayBuffer`,
// `postMessage`s the main thread, and blocks on `Atomics.wait`; the main thread runs the async
// dispatch and `Atomics.notify`s the result back. See `gpu-host.ts` (main) and `worker.ts` (worker).
//
// One SAB, laid out as: an Int32 control header, then a UTF-8 region for the WGSL shader text, then
// an f32 region for the readback column(s). Both realms import THIS module so the layout can never
// drift between the writer and the reader.

/** Control words, as indices into an `Int32Array` over the head of the SAB. */
export const Ctrl = {
  /** The futex word. Worker stores 1 and `Atomics.wait`s; main stores 0 and `Atomics.notify`s. */
  SIGNAL: 0,
  /** Request kind: `Op.PREPARE` or `Op.DISPATCH`. */
  OP: 1,
  /** Result: 1 = ok, 0 = failed/declined (main writes it before notifying). */
  STATUS: 2,
  /** Length of the WGSL shader text, in UTF-8 bytes. */
  WGSL_LEN: 3,
  /** Lanes in this dispatch. */
  N: 4,
  /** Columns (roots) read back per lane — 1 for single-root, k for joint, **0 for reduce mode**
   *  (PLAN-PRECISION Track F: the shader folds on-device; one `(Σx, Σx², count)` triple per
   *  workgroup comes back — see `dispatchShape` in `gpu-host.ts`). */
  COLS: 5,
  /** RNG key low word. */
  K0: 6,
  /** RNG key high word. */
  K1: 7,
  /** First lane index of this dispatch. */
  LANE0: 8,
  /** How many f32 the main thread wrote into the result region (`dispatchShape(n, cols).outCount`). */
  OUT_LEN: 9,
} as const;

/** Request kinds written to `Ctrl.OP`. */
export const Op = {
  /** Compile + cache the pipeline for the WGSL; no dispatch. Returns ok/declined in `STATUS`. */
  PREPARE: 1,
  /** Run one lane range and read the result column(s) back into the f32 region. */
  DISPATCH: 2,
} as const;

/** Signal values for the futex word `Ctrl.SIGNAL`. */
export const Signal = { IDLE: 0, PENDING: 1 } as const;

/** Bytes reserved for the Int32 control header (a whole 64-byte line; only ~10 words are used). */
export const HEADER_BYTES = 64;

/** Bytes reserved for the WGSL shader text. The gate caps a shader at ~8000 emitted instructions;
 *  512 KiB is far above any shader that clears it. */
export const WGSL_CAP_BYTES = 512 * 1024;

/** Max f32 read back in one dispatch — the joint driver's ceiling (`GPU_JOINT_ELEMS = 8 << 20` in
 *  `gpu.rs`); the single-root path reads back at most `GPU_DISPATCH = 1 << 20`, well under this. Keep
 *  in sync with those two constants. */
export const RESULT_CAP_F32 = 8 << 20;

export const WGSL_OFFSET = HEADER_BYTES;
export const RESULT_OFFSET = WGSL_OFFSET + WGSL_CAP_BYTES;
export const SAB_BYTES = RESULT_OFFSET + RESULT_CAP_F32 * 4;

/** Allocate the control SAB. Requires a cross-origin-isolated context (SharedArrayBuffer). */
export function makeGpuSab(): SharedArrayBuffer {
  return new SharedArrayBuffer(SAB_BYTES);
}
