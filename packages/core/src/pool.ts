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

import type { WorkerRequest, WorkerResponse } from './worker.js';

/** A worker plus the id of the job it's running (`null` when idle). */
interface Slot {
  worker: BrowserLikeWorker;
  jobId: number | null;
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
  w.onmessage = (ev: MessageEvent<WorkerResponse>) => settle(ev.data);
  w.onerror = (ev) => failAll(new Error(ev.message ?? 'worker error'));
  return w;
}

function settle(res: WorkerResponse): void {
  const p = pending.get(res.id);
  if (!p) return;
  pending.delete(res.id);
  // Free the slot that ran *this* job — matched by id, since replies can land out of order.
  const slot = slots?.find((s) => s.jobId === res.id);
  if (slot) slot.jobId = null;
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
    starting = Promise.all(Array.from({ length: size }, spawn)).then((workers) => {
      slots = workers.map((worker) => ({ worker, jobId: null }));
      drain();
    });
  }
  return starting;
}

/** Dispatch one job to a free worker (queueing if all are busy) and resolve with its reply. */
export async function call(req: Omit<WorkerRequest, 'id'>): Promise<WorkerResponse> {
  await start();
  return new Promise<WorkerResponse>((resolve, reject) => {
    const full: WorkerRequest = { ...req, id: nextId++ };
    queue.push({ req: full, p: { resolve, reject } });
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
