---
"@noiselang/core": patch
---

Fix the engine worker failing to load in bundled production builds. The worker URL was hoisted into
a variable, which stopped bundlers from recognizing `new Worker(new URL('./worker.js', import.meta.url))`
as a worker — so `worker.js` was copied verbatim instead of bundled, and its `import './gpu-protocol.js'`
resolved to a file that was never emitted. Every run then hung forever waiting on a worker that had
died on load. Dev servers were unaffected, so this only appeared in production bundles.
