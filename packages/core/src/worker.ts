// The worker entry point: this is where every Noise program actually runs.
//
// Nothing in this package executes the engine on the main thread. A Monte-Carlo run is a tight loop
// over millions of draws — on the main thread that is a frozen tab (`turboquant.noise` alone is
// ~6s of solid compute). So `index.ts` never calls the wasm directly; it posts a job here.
//
// One file serves both hosts. The engine is the same wasm either way; only the message plumbing and
// the way the module bytes are located differ between a browser `Worker` and a Node `worker_threads`
// worker, so those two differences are isolated at the top and everything below is shared.

/** The engine's exports, from whichever of the two builds this worker loaded. */
interface Engine {
  run(src: string, optsJson?: string): string;
  run_with_introspection(src: string, requestsJson: string, optsJson?: string): string;
  meta(src: string): string;
  version(): string;
}

/** A job from the pool. `id` correlates the reply; the pool has no other way to match them. */
export interface WorkerRequest {
  id: number;
  op: 'run' | 'runWithIntrospection' | 'meta' | 'version';
  src?: string;
  optsJson?: string;
  requestsJson?: string;
}

/** The one-time GPU handshake (PLAN-WEBGPU G3): the main thread hands the worker the shared control
 *  buffer and whether it acquired a WebGPU device. Sent before any job, only on the threaded path. */
export interface GpuInit {
  type: 'gpu-init';
  sab: SharedArrayBuffer;
  ready: boolean;
}

/** The reply. `raw` is the engine's JSON, left unparsed — see `elapsedMs` below. */
export interface WorkerResponse {
  id: number;
  ok: boolean;
  raw?: string;
  /** Engine time only, measured *inside* the worker. Postmessage latency is not the engine's cost. */
  elapsedMs?: number;
  error?: string;
  /** Execution-environment + GPU-host diagnostics, attached by the pool on the main thread (the
   *  worker itself can't see the device or the isolation state). */
  diag?: RunDiagnostics;
}

/** What actually executed a run, and where — the JS-side complement to the engine's `result.profile`
 *  (which carries phase timings + gate/decline reasons). Surfaced as `result.diagnostics` so a host
 *  can answer "why is this slow / not on the GPU?" from one place: isolation state, thread/worker
 *  count, and the GPU host's real dispatch volume + the exact shader errors Chrome raised. */
export interface RunDiagnostics {
  /** Cross-origin isolated (COOP/COEP). The prerequisite for SharedArrayBuffer, wasm threads, and the
   *  GPU bridge — when false, the engine runs single-threaded on the CPU with no GPU. */
  crossOriginIsolated: boolean;
  /** Running the multi-threaded (`wasm-mt`) engine rather than the single-threaded fallback. */
  threaded: boolean;
  /** Engine workers in the pool: 1 on the threaded path (a run fans across an internal rayon pool),
   *  N single-threaded workers otherwise (concurrent runs, one core each). */
  workers: number;
  gpu: {
    /** `navigator.gpu` exists in this browser. */
    supported: boolean;
    /** A WebGPU device was acquired on the main thread (the bridge is live). */
    deviceAcquired: boolean;
    /** GPU dispatches serviced for THIS run. 0 → every forcing ran on the CPU; see
     *  `result.profile.notes` for the gate/decline reason. */
    dispatches: number;
    /** Lanes (draws) sent to the GPU this run. */
    lanes: number;
    /** f32 values read back from the GPU this run (`Σ cols × n`). */
    elements: number;
    /** Pipelines Chrome's validator rejected this run (a shader Tint won't compile → CPU fallback). */
    shaderFailures: number;
    /** The most recent shader-validation error message (e.g. a Tint-vs-naga divergence), else null. */
    lastShaderError: string | null;
  };
}

// Browser workers get a `self` with `postMessage`; Node workers get a `parentPort` instead.
const isNodeWorker =
  typeof self === 'undefined' &&
  typeof process !== 'undefined' &&
  process.versions?.node != null;

// === GPU bridge (PLAN-WEBGPU G3) =================================================================
//
// The engine's wasm calls three imports — `nz_gpu_available/prepare/dispatch` (see `gpu.rs`) — which
// forward to `globalThis.__noiseGpu`, installed here. Because a forcing is synchronous and WebGPU is
// async, `prepare`/`dispatch` cannot run the GPU themselves: they write the request into the shared
// control buffer, `postMessage` the main thread (which owns the device), and block on `Atomics.wait`
// until it writes the answer back. All the async work happens over there (`gpu-host.ts`).

import {
  Ctrl,
  Op,
  Signal,
  WGSL_OFFSET,
  WGSL_CAP_BYTES,
  RESULT_OFFSET,
} from './gpu-protocol.js';

/** The engine-worker view onto the GPU bridge, installed on `globalThis` once the main thread sends
 *  the handshake. Unset on rayon sub-workers, so their `available()` is 0 and they never dispatch. */
interface NoiseGpu {
  available(): number;
  prepare(wgsl: string): number;
  dispatch(
    wgsl: string,
    out: Float32Array,
    n: number,
    cols: number,
    k0: number,
    k1: number,
    lane0: number,
  ): number;
}

/** Write the WGSL text into the SAB's UTF-8 region and record its byte length. Encodes to a private
 *  buffer first, then copies in — `encodeInto` over shared memory is not universally supported. */
function writeWgsl(sab: SharedArrayBuffer, ctrl: Int32Array, wgsl: string): void {
  const bytes = new TextEncoder().encode(wgsl);
  new Uint8Array(sab, WGSL_OFFSET, WGSL_CAP_BYTES).set(bytes);
  Atomics.store(ctrl, Ctrl.WGSL_LEN, bytes.length);
}

/** Post the request and block until the main thread services it. Returns the STATUS word (1 ok). */
function roundTrip(ctrl: Int32Array): number {
  Atomics.store(ctrl, Ctrl.SIGNAL, Signal.PENDING);
  // `postMessage` queues to the main thread's event loop *before* we block, so it is delivered even
  // though this thread is about to freeze in `Atomics.wait`. The main thread's loop is free.
  (self as unknown as Worker).postMessage({ type: 'gpu' });
  Atomics.wait(ctrl, Ctrl.SIGNAL, Signal.PENDING);
  return Atomics.load(ctrl, Ctrl.STATUS);
}

/** Install the bridge on `globalThis` for the wasm imports to find. */
function installGpuBridge(sab: SharedArrayBuffer, ready: boolean): void {
  const ctrl = new Int32Array(sab, 0, 16);
  const bridge: NoiseGpu = {
    available: () => (ready ? 1 : 0),
    prepare(wgsl) {
      writeWgsl(sab, ctrl, wgsl);
      Atomics.store(ctrl, Ctrl.OP, Op.PREPARE);
      return roundTrip(ctrl);
    },
    dispatch(wgsl, out, n, cols, k0, k1, lane0) {
      writeWgsl(sab, ctrl, wgsl);
      Atomics.store(ctrl, Ctrl.OP, Op.DISPATCH);
      Atomics.store(ctrl, Ctrl.N, n);
      Atomics.store(ctrl, Ctrl.COLS, cols);
      Atomics.store(ctrl, Ctrl.K0, k0 | 0);
      Atomics.store(ctrl, Ctrl.K1, k1 | 0);
      Atomics.store(ctrl, Ctrl.LANE0, lane0 | 0);
      const status = roundTrip(ctrl);
      if (status === 1) {
        const len = Atomics.load(ctrl, Ctrl.OUT_LEN);
        out.set(new Float32Array(sab, RESULT_OFFSET, len));
      }
      return status;
    },
  };
  (globalThis as unknown as { __noiseGpu?: NoiseGpu }).__noiseGpu = bridge;
}

/**
 * Can this worker run the multi-threaded engine?
 *
 * Wasm threads are built on `SharedArrayBuffer`, which browsers only expose to a **cross-origin
 * isolated** page (COOP/COEP headers). A library cannot impose those headers on the app that
 * installs it, so this is a runtime question, not a build-time one — hence two builds and this
 * check. Workers inherit the isolation of the page that spawned them, so reading the flag here is
 * the same answer the main thread would get.
 *
 * Node is excluded deliberately: it has `SharedArrayBuffer` unconditionally, but the threaded
 * build's pool bootstrap is `wasm-bindgen-rayon`'s, which spawns browser `Worker`s. It has no
 * `worker_threads` backend, so the threaded build simply cannot start its pool under Node.
 */
const canUseThreads =
  !isNodeWorker && typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;

/**
 * Load and instantiate the engine — the threaded build when the page allows it, the portable one
 * otherwise. Both compute identical results (the reducer merges chunks in index order, so thread
 * count changes wall clock and nothing else); this only decides how fast.
 *
 * In the browser the generated glue resolves `noise_bg.wasm` from `import.meta.url` and fetches it.
 * Node has no `fetch` for `file:` URLs, so we read the bytes and hand them to the same `init` —
 * wasm-bindgen's `init` accepts bytes or a `WebAssembly.Module`, which is what lets one
 * `--target web` build serve both hosts.
 */
async function initEngine(): Promise<Engine> {
  if (canUseThreads) {
    const mt = await import('../wasm-mt/noise.js');
    await mt.default();
    // Spawn the rayon pool. Wasm cannot create a thread itself — the threads proposal leaves thread
    // *creation* to the embedder — so this is the call that actually does `new Worker()` N times and
    // hands the set to rayon. Until it resolves, the reducer runs single-threaded.
    await mt.initThreadPool(navigator.hardwareConcurrency);
    return mt as unknown as Engine;
  }

  const st = await import('../wasm/noise.js');
  if (isNodeWorker) {
    const { readFile } = await import('node:fs/promises');
    const bytes = await readFile(new URL('../wasm/noise_bg.wasm', import.meta.url));
    await st.default({ module_or_path: bytes });
  } else {
    await st.default();
  }
  return st as unknown as Engine;
}

const ready = initEngine();

async function handle(req: WorkerRequest): Promise<WorkerResponse> {
  const engine = await ready;
  try {
    // Time the engine call itself. The pool adds its own wall-clock if a caller wants round-trip.
    const t0 = performance.now();
    let raw: string;
    switch (req.op) {
      case 'run':
        raw = engine.run(req.src!, req.optsJson);
        break;
      case 'runWithIntrospection':
        raw = engine.run_with_introspection(req.src!, req.requestsJson!, req.optsJson);
        break;
      case 'meta':
        raw = engine.meta(req.src!);
        break;
      case 'version':
        raw = engine.version();
        break;
    }
    return { id: req.id, ok: true, raw, elapsedMs: performance.now() - t0 };
  } catch (e) {
    // The engine catches its own panics and returns them as documents, so reaching here means the
    // instance is broken (an OOM, a trap). Report it; the pool retires the worker.
    return { id: req.id, ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

if (isNodeWorker) {
  const { parentPort } = await import('node:worker_threads');
  parentPort!.on('message', (req: WorkerRequest) => {
    handle(req).then((res) => parentPort!.postMessage(res));
  });
} else {
  self.onmessage = (ev: MessageEvent<WorkerRequest | GpuInit>) => {
    // The one-time GPU handshake sets up the bridge and returns — it is not a job.
    if ((ev.data as GpuInit).type === 'gpu-init') {
      const init = ev.data as GpuInit;
      installGpuBridge(init.sab, init.ready);
      return;
    }
    handle(ev.data as WorkerRequest).then((res) => (self as unknown as Worker).postMessage(res));
  };
}
