# @noiselang/core

## 0.6.0

### Minor Changes

- b1729c1: Array slicing: indexing with a deterministic array of indices selects into a new array. Since
  `a..b` is already an array, `xs[0..r]` takes the first `r` elements (Rust-style half-open), and
  any index list works — `xs[[2, 0, 0]]` reorders and repeats, `xs[perm]` applies a deterministic
  permutation. Each index obeys the existing scalar rules (constant non-negative integer, bounds
  checked element-wise); random indices inside the array are an error. It is pure build-time sugar
  for `[for i in inds { xs[i] }]`, so nothing changes for codegen. The secretary example's
  observation phase collapses from a fold loop to `bar = max(quality[0..r])`.

## 0.5.1

### Patch Changes

- cd83e72: Fix the engine worker failing to load in bundled production builds. The worker URL was hoisted into
  a variable, which stopped bundlers from recognizing `new Worker(new URL('./worker.js', import.meta.url))`
  as a worker — so `worker.js` was copied verbatim instead of bundled, and its `import './gpu-protocol.js'`
  resolved to a file that was never emitted. Every run then hung forever waiting on a worker that had
  died on load. Dev servers were unaffected, so this only appeared in production bundles.
