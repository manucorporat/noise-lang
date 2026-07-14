// Cancellation check for `run(src, { signal })` — PLAN-PREGPU Track A.
//
// Self-checking: exits non-zero on failure, so it can gate a release. Run with:
//   node bench/abort.mjs        (from packages/core, after `pnpm build`)
//
// The load-bearing part is the FIRST measurement. A "cancellation was fast" assertion proves
// nothing unless the thing being cancelled is genuinely slow — so this times the query uncancelled
// first and fails if it isn't slow enough for the abort to be meaningful. (An earlier version of
// the Rust-side test was vacuous exactly this way: the engine's default op budget silently clamped
// the query to ~0.3s, so the abort "passed" without doing anything. Hence the raised max_opts.)

import { run, stop } from '../dist/index.js';

const LONG = `use rand;
engine::set_max_opts(1000000000000);
X ~ unif(0,1); Y ~ unif(0,1); Z ~ normal(0,1);
P(X*X + Y*Y + math::sin(Z) * math::cos(Z) < 1.5, 2000000000)`;

/** Abort must beat completion by at least this much, or the check is not measuring anything. */
const MIN_SPEEDUP = 10;

let failures = 0;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? `  — ${detail}` : ''}`);
  if (!ok) failures++;
};

// 1. Baseline. If this is fast, every other assertion below is vacuous.
let t = Date.now();
const base = await run(LONG);
const baseMs = Date.now() - t;
check(
  baseMs > 2000,
  'the query is genuinely slow uncancelled',
  `${baseMs} ms -> ${base.result?.value?.text}`,
);

// 2. Abort mid-run: rejects with AbortError, and does so *promptly*.
const ctl = new AbortController();
setTimeout(() => ctl.abort(), 100);
t = Date.now();
let abortMs = Infinity;
let name = 'none';
try {
  await run(LONG, { signal: ctl.signal });
  check(false, 'aborting mid-run rejects', 'it returned a value instead');
} catch (e) {
  abortMs = Date.now() - t;
  name = e.name;
  check(name === 'AbortError', 'aborting mid-run rejects with AbortError', `${name}, ${abortMs} ms`);
}
check(
  baseMs / abortMs > MIN_SPEEDUP,
  `abort beats completion by >${MIN_SPEEDUP}x`,
  `${baseMs} ms -> ${abortMs} ms (${(baseMs / abortMs).toFixed(0)}x)`,
);

// 3. An already-aborted signal rejects immediately (fetch semantics).
try {
  await run('1 + 1', { signal: AbortSignal.abort() });
  check(false, 'a pre-aborted signal rejects');
} catch (e) {
  check(e.name === 'AbortError', 'a pre-aborted signal rejects with AbortError', e.name);
}

// 4. The pool RECOVERS: the killed worker's replacement takes over. (This is the one that caught
// the real bug — `drain()` was posting to the terminated worker before its replacement existed,
// and the next run hung forever with no error.)
t = Date.now();
const after = await run('use rand; D ~ unif_int(1,6); E(D, 200000)');
const ok = after.result?.value?.text === '3.5';
check(ok, 'the pool still works after an abort', `${Date.now() - t} ms -> E[die] = ${after.result?.value?.text}`);

// 5. An un-aborted signal must not disturb a healthy run.
const idle = new AbortController();
const healthy = await run('use rand; X ~ unif(0,1); E(X, 100000)', { signal: idle.signal });
check(healthy.result?.error == null, 'an un-tripped signal changes nothing', healthy.result?.value?.text);

stop();
console.log(failures === 0 ? '\nall checks passed' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
