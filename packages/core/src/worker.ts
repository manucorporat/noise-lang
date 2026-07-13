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

/** The reply. `raw` is the engine's JSON, left unparsed — see `elapsedMs` below. */
export interface WorkerResponse {
  id: number;
  ok: boolean;
  raw?: string;
  /** Engine time only, measured *inside* the worker. Postmessage latency is not the engine's cost. */
  elapsedMs?: number;
  error?: string;
}

// Browser workers get a `self` with `postMessage`; Node workers get a `parentPort` instead.
const isNodeWorker =
  typeof self === 'undefined' &&
  typeof process !== 'undefined' &&
  process.versions?.node != null;

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
  self.onmessage = (ev: MessageEvent<WorkerRequest>) => {
    handle(ev.data).then((res) => (self as unknown as Worker).postMessage(res));
  };
}
