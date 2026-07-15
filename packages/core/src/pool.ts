// A persistent pool of engine workers. Every Noise run goes through here — the main thread never
// executes the engine (see `worker.ts` for why).
//
// Deliberately NOT SharedArrayBuffer + wasm threads. That route needs a nightly Rust toolchain, a
// rebuilt std, and cross-origin isolation (COOP/COEP) — and the isolation requirement is viral: it
// would force *every app that installs @noiselang/core* to serve those headers. Instead each worker
// owns an ordinary, independent wasm instance and we pass plain JSON. Nothing is shared, so nothing
// needs to be shareable.
//
// Workers are persistent because spawning one and instantiating the module costs tens of
// milliseconds — fine once, absurd per keystroke in a playground.

import type { WorkerRequest, WorkerResponse, RunDiagnostics } from './worker.js';
import {
  acquireGpuHost,
  serviceGpuRequest,
  newGpuHostStats,
  type GpuHost,
  type GpuHostStats,
} from './gpu-host.js';
import { makeGpuSab } from './gpu-protocol.js';

/** A worker plus the id of the job it's running (`null` when idle). */
interface Slot {
  worker: BrowserLikeWorker;
  jobId: number | null;
}

// === browser GPU host (PLAN-WEBGPU G3) ===========================================================
//
// On the cross-origin-isolated (threaded) path the main thread owns the WebGPU device — the engine
// worker can't, since its event loop freezes during a synchronous forcing. Acquired once, lazily, on
// pool start. Each worker gets its own control `SharedArrayBuffer`; when it posts `{ type: 'gpu' }`
// mid-forcing (blocked on `Atomics.wait`), we run the async dispatch and notify it back. Null host
// (no WebGPU / not isolated) → the worker's bridge reports unavailable and every forcing uses wasm.

let gpuHost: GpuHost | null = null;
let gpuHostTried = false;

/** GPU-host stat snapshot taken when a job is posted, so the reply can report the delta THIS run
 *  caused. Keyed by job id (the threaded path serializes on one worker, but keying is robust). */
const jobGpuBaseline = new Map<number, GpuHostStats>();

/** Assemble the run's diagnostics: static environment (isolation, threads, worker count, GPU
 *  availability) plus the GPU-host delta since the job was posted. */
function buildDiag(id: number): RunDiagnostics {
  const base = jobGpuBaseline.get(id) ?? newGpuHostStats();
  jobGpuBaseline.delete(id);
  const s = gpuHost?.stats;
  return {
    crossOriginIsolated:
      typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true,
    threaded: canUseThreads,
    workers: slots?.length ?? 0,
    gpu: {
      supported: typeof navigator !== 'undefined' && !!(navigator as { gpu?: unknown }).gpu,
      deviceAcquired: !!gpuHost,
      dispatches: s ? s.dispatches - base.dispatches : 0,
      lanes: s ? s.lanes - base.lanes : 0,
      elements: s ? s.elements - base.elements : 0,
      shaderFailures: s ? s.shaderFailures - base.shaderFailures : 0,
      lastShaderError: s?.lastShaderError ?? null,
    },
  };
}

/** The subset of the Worker API we use — satisfied by both `Worker` and Node's `worker_threads`. */
interface BrowserLikeWorker {
  postMessage(msg: unknown): void;
  terminate(): void;
}

interface Pending {
  resolve: (res: WorkerResponse) => void;
  reject: (err: Error) => void;
}

const isNode =
  typeof process !== 'undefined' && process.versions?.node != null && typeof Worker === 'undefined';

/**
 * Is the threaded engine available? (Same test the worker makes — see `worker.ts`.)
 *
 * This decides the pool's *shape*, not just its size:
 *
 *  * **Threaded** (cross-origin isolated page): ONE engine worker, which internally fans each
 *    Monte-Carlo reduction across a rayon pool of its own — so a single program already uses every
 *    core. Spawning more engine workers here would just have them fight over the same cores.
 *  * **Not threaded**: N engine workers, each single-threaded. One program still runs on one core
 *    (a single reduction cannot be split — see the note on sharding in `worker.ts`), but concurrent
 *    runs spread out, and the main thread stays free either way.
 */
const canUseThreads =
  !isNode && typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;

/**
 * How many engine workers to run.
 *
 * One per core, minus one so the UI thread keeps a core to stay responsive — the whole point of
 * moving off the main thread is not to starve it. Clamped: past ~8 the Monte-Carlo fan-out is
 * memory-bandwidth-bound, and each worker holds its own wasm instance.
 */
function defaultSize(): number {
  if (canUseThreads) return 1; // the single engine worker already uses every core
  const cores =
    (typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : undefined) ??
    (isNode
      ? ((globalThis as { process?: { availableParallelism?: () => number } }).process
          ?.availableParallelism?.() ?? 4)
      : 4);
  return Math.max(1, Math.min(8, cores - 1));
}

let slots: Slot[] | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();
const queue: { req: WorkerRequest; p: Pending }[] = [];

async function spawn(): Promise<BrowserLikeWorker> {
  const url = new URL('./worker.js', import.meta.url);
  if (isNode) {
    const { Worker: NodeWorker } = await import('node:worker_threads');
    const w = new NodeWorker(url);
    w.on('message', (res: WorkerResponse) => settle(res));
    // A worker that dies takes its in-flight job with it; fail that job rather than hang forever.
    w.on('error', (err: Error) => failAll(err));
    return w as unknown as BrowserLikeWorker;
  }
  const w = new Worker(url, { type: 'module' });
  // When the GPU host is up, give THIS worker its own control buffer and route its mid-forcing GPU
  // requests to the host. The SAB is per-worker so concurrent workers never collide on one (the
  // threaded path runs a single engine worker anyway, but a cancel-replacement spawns a fresh one).
  const sab = gpuHost ? makeGpuSab() : null;
  w.onmessage = (ev: MessageEvent<WorkerResponse | { type: 'gpu' }>) => {
    if (gpuHost && sab && (ev.data as { type?: string }).type === 'gpu') {
      void serviceGpuRequest(gpuHost, sab);
      return;
    }
    settle(ev.data as WorkerResponse);
  };
  w.onerror = (ev) => failAll(new Error(ev.message ?? 'worker error'));
  if (gpuHost && sab) {
    w.postMessage({ type: 'gpu-init', sab, ready: true });
  }
  return w;
}

function settle(res: WorkerResponse): void {
  const p = pending.get(res.id);
  if (!p) return;
  pending.delete(res.id);
  // Free the slot that ran *this* job — matched by id, since replies can land out of order.
  const slot = slots?.find((s) => s.jobId === res.id);
  if (slot) slot.jobId = null;
  // Attach execution diagnostics (environment + this run's GPU-host delta) before resolving.
  res.diag = buildDiag(res.id);
  p.resolve(res);
  drain();
}

function failAll(err: Error): void {
  for (const p of pending.values()) p.reject(err);
  pending.clear();
  for (const q of queue) q.p.reject(err);
  queue.length = 0;
}

function drain(): void {
  if (!slots) return;
  while (queue.length > 0) {
    const slot = slots.find((s) => s.jobId === null);
    if (!slot) return; // every worker busy — the rest stay queued until one replies
    const job = queue.shift()!;
    slot.jobId = job.req.id;
    pending.set(job.req.id, job.p);
    // Snapshot GPU-host stats so the reply can report just THIS run's dispatch delta.
    if (gpuHost) jobGpuBaseline.set(job.req.id, { ...gpuHost.stats });
    slot.worker.postMessage(job.req);
  }
}

let starting: Promise<void> | null = null;

/**
 * Bring the pool up. Idempotent, and safe to call eagerly to hide the spawn + instantiate cost
 * before the user's first run.
 */
export function start(size = defaultSize()): Promise<void> {
  if (!starting) {
    starting = (async () => {
      // Acquire the WebGPU device once, before spawning, so each worker's `spawn()` can hand it a
      // control buffer. Only on the threaded path (the SAB bridge needs cross-origin isolation, and
      // the design needs a blockable worker). A null host is normal — forcing then uses wasm.
      if (canUseThreads && !gpuHostTried) {
        gpuHostTried = true;
        gpuHost = await acquireGpuHost();
      }
      const workers = await Promise.all(Array.from({ length: size }, spawn));
      slots = workers.map((worker) => ({ worker, jobId: null }));
      drain();
    })();
  }
  return starting;
}

/**
 * The `AbortError` the platform throws — same shape `fetch` rejects with, so a caller's existing
 * `err.name === 'AbortError'` check works unchanged. `DOMException` exists in browsers and in Node
 * ≥17; the fallback keeps the name right on anything older.
 */
function abortError(reason: unknown): Error {
  if (reason instanceof Error) return reason;
  if (typeof DOMException === 'function') {
    return new DOMException('The operation was aborted.', 'AbortError');
  }
  const e = new Error('The operation was aborted.');
  e.name = 'AbortError';
  return e;
}

/**
 * Cancel the job `id`: drop it if it hasn't started, otherwise **terminate the worker running it**
 * and put a fresh one in its place.
 *
 * Terminating is not a blunt instrument here — it is the only instrument. A Noise run is a tight
 * Monte-Carlo loop inside wasm; while it runs, that worker's event loop never turns, so a
 * `postMessage` asking it to stop would simply sit in the queue until the run it was meant to
 * cancel had already finished. Without `SharedArrayBuffer` (which this package deliberately does
 * not require — see the header) there is no flag the worker could poll either. So we kill the
 * thread.
 *
 * That is also why the engine's own `CancelToken` (noise-core `exec::CancelToken`) is not what runs
 * this path: it is the *native* mechanism, for embedders that can set a flag from another thread.
 * In the browser the worker boundary already gives us a harder guarantee — the run is gone, not
 * asked to leave.
 *
 * The cost is the replacement worker's spawn + wasm instantiate (tens of ms, off the main thread)
 * and the loss of that worker's engine scope — which is exactly the semantics the core documents
 * for a cancelled engine: **treat it as stale and rebuild**. Other workers, and every other
 * in-flight job, are untouched.
 */
function cancelJob(id: number, reason: unknown): void {
  const p = pending.get(id);
  const queued = queue.findIndex((q) => q.req.id === id);
  if (queued >= 0) {
    // Never dispatched: no worker to kill, just drop it.
    const [job] = queue.splice(queued, 1);
    job.p.reject(abortError(reason));
    return;
  }
  if (!p) return; // already settled — abort arrived too late, which is fine
  pending.delete(id);

  const slot = slots?.find((s) => s.jobId === id);
  if (!slot) {
    p.reject(abortError(reason));
    return;
  }
  // REMOVE the slot before terminating, don't just mark it idle. A slot with `jobId === null` is
  // one `drain()` will post the next job to — and for the moments between `terminate()` and the
  // replacement arriving (`spawn()` is async: a worker spawn + wasm instantiate), that worker is
  // dead. Posting to it drops the message on the floor and the job hangs forever with no error.
  // (Found by the test that runs a program *after* an abort — see `pool.abort.test`.)
  const i = slots?.indexOf(slot) ?? -1;
  if (i >= 0) slots?.splice(i, 1);
  slot.worker.terminate();
  p.reject(abortError(reason));
  // Bring a replacement up, then let anything queued flow into it.
  void spawn().then((w) => {
    slots?.push({ worker: w, jobId: null });
    drain();
  });
}

/**
 * Dispatch one job to a free worker (queueing if all are busy) and resolve with its reply.
 *
 * `signal` follows the platform convention (`fetch`-style): an already-aborted signal rejects
 * immediately; otherwise an `'abort'` listener cancels the job (see [`cancelJob`]) and the promise
 * rejects with `signal.reason` — the standard `AbortError`. The listener is removed when the job
 * settles either way, so a long-lived signal doesn't accumulate listeners.
 */
export async function call(
  req: Omit<WorkerRequest, 'id'>,
  signal?: AbortSignal,
): Promise<WorkerResponse> {
  if (signal?.aborted) throw abortError(signal.reason);
  await start();
  const id = nextId++;
  return new Promise<WorkerResponse>((resolve, reject) => {
    const onAbort = () => cancelJob(id, signal?.reason);
    const done = <T,>(f: (v: T) => void) => (v: T) => {
      signal?.removeEventListener('abort', onAbort);
      f(v);
    };
    signal?.addEventListener('abort', onAbort, { once: true });
    // Aborted between the `await start()` above and here — the listener would never fire.
    if (signal?.aborted) {
      signal.removeEventListener('abort', onAbort);
      reject(abortError(signal.reason));
      return;
    }
    const full: WorkerRequest = { ...req, id };
    queue.push({ req: full, p: { resolve: done(resolve), reject: done(reject) } });
    drain();
  });
}

/** Tear the pool down (tests, HMR, a Node script that wants to exit). */
export function stop(): void {
  for (const s of slots ?? []) s.worker.terminate();
  slots = null;
  starting = null;
  failAll(new Error('pool stopped'));
}
