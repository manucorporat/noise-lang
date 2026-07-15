// The main-thread half of browser GPU (PLAN-WEBGPU G3). The engine worker can't touch WebGPU — it
// runs a synchronous forcing and would deadlock waiting on an async dispatch — so the device lives
// here, on the main thread, whose event loop stays free. The worker posts `{ type: 'gpu' }` after
// writing a request into the shared control buffer; `serviceGpuRequest` reads it, runs the async
// WebGPU dispatch, writes the result column back, and `Atomics.notify`s the blocked worker.
//
// The buffer layout below mirrors the native `Device::dispatch` in `gpu.rs` byte-for-byte (uniform
// `[k0, k1, lane0, n]`, one `array<f32>` output of `cols × n`, `@workgroup_size(64)`), because both
// consume the exact same WGSL that `wgsl_emit::emit` produces. If one drifts, the answers diverge.

import { Ctrl, Op, Signal, WGSL_OFFSET, RESULT_OFFSET } from './gpu-protocol.js';

// WebGPU types aren't in the default TS lib and we don't want to pull `@webgpu/types` + a tsconfig
// change just for this. The surface we use is tiny; alias it loosely and keep the boundary honest.
type GpuDevice = any; // eslint-disable-line @typescript-eslint/no-explicit-any
type GpuQueue = any; // eslint-disable-line @typescript-eslint/no-explicit-any
type GpuPipeline = any; // eslint-disable-line @typescript-eslint/no-explicit-any

/** Workgroup size — must equal `wgsl_emit::WORKGROUP` (the shaders declare `@workgroup_size(64)`). */
const WORKGROUP = 64;

/** Running tallies of what the GPU host actually did — surfaced in `result.diagnostics.gpu` so the
 *  playground (and any host) can see dispatch volume and, crucially, the exact shader-validation
 *  errors Chrome's Tint raises (which the wasm engine only ever sees as an opaque "declined"). */
export interface GpuHostStats {
  /** Successful dispatches serviced. */
  dispatches: number;
  /** Σ lanes dispatched (draws sent to the GPU). */
  lanes: number;
  /** Σ f32 read back (`cols × n` per dispatch). */
  elements: number;
  /** Pipelines Chrome's validator rejected (shader/Tint incompatibilities → CPU fallback). */
  shaderFailures: number;
  /** The most recent shader-validation error message, if any (e.g. a Tint-vs-naga divergence). */
  lastShaderError: string | null;
}

/** The acquired device plus a content-addressed pipeline cache (keyed on the shader text, which is a
 *  complete description of the artifact — so a cache hit can never serve a stale kernel). */
export interface GpuHost {
  device: GpuDevice;
  queue: GpuQueue;
  pipelines: Map<string, GpuPipeline>;
  stats: GpuHostStats;
}

/** A fresh zeroed GPU-host stats block. */
export function newGpuHostStats(): GpuHostStats {
  return { dispatches: 0, lanes: 0, elements: 0, shaderFailures: 0, lastShaderError: null };
}

/**
 * Acquire a WebGPU device once, for the page's life. Returns `null` when WebGPU is unavailable (no
 * `navigator.gpu`, no adapter) — the caller then simply never advertises GPU to the worker, and every
 * forcing falls back to the wasm kernel. Never throws: a missing device is a normal, silent decline.
 */
export async function acquireGpuHost(): Promise<GpuHost | null> {
  try {
    const gpu = (navigator as unknown as { gpu?: unknown }).gpu as
      | {
          requestAdapter(opts?: unknown): Promise<{ requestDevice(): Promise<GpuDevice> } | null>;
        }
      | undefined;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    if (!device) return null;
    return { device, queue: device.queue, pipelines: new Map(), stats: newGpuHostStats() };
  } catch {
    return null;
  }
}

/** Compile (or reuse) the pipeline for `wgsl`. Returns `null` if the driver rejects the shader — a
 *  decline, matching the native `Device::pipeline` contract (fall back, never crash the program). */
async function pipelineFor(host: GpuHost, wgsl: string): Promise<GpuPipeline | null> {
  const cached = host.pipelines.get(wgsl);
  if (cached) return cached;
  try {
    host.device.pushErrorScope('validation');
    const module = host.device.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo?.();
    const msgs = [...(info?.messages ?? [])]
      .filter((m: { type: string }) => m.type === 'error')
      .map((m: { message: string; lineNum: number; linePos: number }) => ({
        message: m.message,
        line: m.lineNum,
        pos: m.linePos,
      }));
    if (msgs.length) {
      host.stats.shaderFailures++;
      host.stats.lastShaderError = `${msgs[0].message} (line ${msgs[0].line})`;
      console.log(`[noise-gpu][main] shader compile errors: ${JSON.stringify(msgs.slice(0, 4))}`);
      // Dump the offending lines so we can see the exact WGSL Chrome rejects.
      const lines = wgsl.split('\n');
      for (const m of msgs.slice(0, 4)) {
        console.log(`[noise-gpu][main]   L${m.line}: ${lines[m.line - 1]}`);
      }
    }
    // `createComputePipelineAsync` rejects on a validation error, unlike the sync form which returns
    // an invalid pipeline; pair it with the error scope so a bad shader declines cleanly.
    const pipeline = await host.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const err = await host.device.popErrorScope();
    if (err) {
      console.log(`[noise-gpu][main] pipeline validation error: ${err.message}`);
      return null;
    }
    host.pipelines.set(wgsl, pipeline);
    return pipeline;
  } catch (e) {
    console.log(`[noise-gpu][main] createComputePipeline threw: ${e}`);
    return null;
  }
}

/** Read the WGSL shader text out of the control buffer's UTF-8 region. */
function readWgsl(sab: SharedArrayBuffer, ctrl: Int32Array): string {
  const len = Atomics.load(ctrl, Ctrl.WGSL_LEN);
  const bytes = new Uint8Array(sab, WGSL_OFFSET, len);
  // Copy out of shared memory before decoding: `TextDecoder` won't accept a view over a
  // SharedArrayBuffer directly in some engines.
  return new TextDecoder().decode(bytes.slice());
}

/** Run one dispatch: build the buffers (native layout), dispatch, read back `cols × n` f32. Returns
 *  the result column(s) as a plain `Float32Array`, or `null` on failure. */
async function runDispatch(
  host: GpuHost,
  pipeline: GpuPipeline,
  n: number,
  cols: number,
  k0: number,
  k1: number,
  lane0: number,
): Promise<Float32Array | null> {
  try {
    const { device, queue } = host;
    const outCount = cols * n;
    const bytes = outCount * 4;

    // Uniform `Params { key: vec2<u32>, lane0: u32, n: u32 }` — 16 bytes, same order as native.
    const ubuf = device.createBuffer({ size: 16, usage: 0x40 | 0x8 }); // UNIFORM | COPY_DST
    queue.writeBuffer(ubuf, 0, new Uint32Array([k0 >>> 0, k1 >>> 0, lane0 >>> 0, n >>> 0]));

    const out = device.createBuffer({ size: bytes, usage: 0x80 | 0x4 }); // STORAGE | COPY_SRC
    const staging = device.createBuffer({ size: bytes, usage: 0x1 | 0x8 }); // MAP_READ | COPY_DST

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ubuf } },
        { binding: 1, resource: { buffer: out } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP));
    pass.end();
    enc.copyBufferToBuffer(out, 0, staging, 0, bytes);
    queue.submit([enc.finish()]);

    await staging.mapAsync(0x1); // MapMode.READ
    const col = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    // Free per-dispatch buffers; the pipeline (the expensive artifact) stays cached.
    ubuf.destroy?.();
    out.destroy?.();
    staging.destroy?.();
    return col;
  } catch (e) {
    console.log(`[noise-gpu][main] runDispatch threw: ${e}`);
    return null;
  }
}

/**
 * Service one request the worker placed in `sab`, then wake the worker. Called from the pool's
 * `{ type: 'gpu' }` message handler on the main thread. Always resolves (never throws into the
 * message loop): on any failure it writes `STATUS = 0` and still notifies, so the worker unblocks and
 * declines to the CPU rather than hanging.
 */
export async function serviceGpuRequest(host: GpuHost, sab: SharedArrayBuffer): Promise<void> {
  const ctrl = new Int32Array(sab, 0, HEADER_I32);
  let status = 0;
  try {
    const op = Atomics.load(ctrl, Ctrl.OP);
    const wgsl = readWgsl(sab, ctrl);
    const pipeline = await pipelineFor(host, wgsl);
    if (pipeline) {
      if (op === Op.PREPARE) {
        status = 1;
      } else if (op === Op.DISPATCH) {
        const n = Atomics.load(ctrl, Ctrl.N);
        const cols = Atomics.load(ctrl, Ctrl.COLS);
        const k0 = Atomics.load(ctrl, Ctrl.K0);
        const k1 = Atomics.load(ctrl, Ctrl.K1);
        const lane0 = Atomics.load(ctrl, Ctrl.LANE0);
        const col = await runDispatch(host, pipeline, n, cols, k0, k1, lane0);
        if (!col) console.log(`[noise-gpu][main] dispatch failed (n=${n} cols=${cols})`);
        if (col) {
          new Float32Array(sab, RESULT_OFFSET, col.length).set(col);
          Atomics.store(ctrl, Ctrl.OUT_LEN, col.length);
          host.stats.dispatches++;
          host.stats.lanes += n;
          host.stats.elements += col.length;
          status = 1;
        }
      }
    }
  } catch (e) {
    console.log(`[noise-gpu][main] serviceGpuRequest threw: ${e}`);
    status = 0;
  } finally {
    Atomics.store(ctrl, Ctrl.STATUS, status);
    // Release the worker: clear the futex word and wake it.
    Atomics.store(ctrl, Ctrl.SIGNAL, Signal.IDLE);
    Atomics.notify(ctrl, Ctrl.SIGNAL);
  }
}

/** Int32 words spanned by the control header (`HEADER_BYTES / 4`). */
const HEADER_I32 = 16;
