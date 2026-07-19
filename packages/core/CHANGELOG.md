# @noiselang/core

## 0.7.0

### Minor Changes

- d38da29: Sliders work everywhere a number does, plus `onehot` and "did you mean?" hints.

  `input::real` evaluates to a symbolic value (`Value::Sym`), a fourth scalar shape beside
  `Num`/`Est`/`Dist`, and most builtins had never grown an arm for it — every `math::` ufunc,
  `norm`/`normsq`/`normalize`/`mse`, `max`/`min`/`cummax`/`cummin`/`quantize`, `count`/`any`/`all`,
  and `poisson`/`geometric`/`normal_complex` all rejected a slider. They now defer symbolically, so
  the value stays an input (no recompile on drag) and gives the same answer as the constant it
  currently holds. A pair of tests locks this in: each builtin is exercised with a slider against the
  literal, and any new `math`/`vec`/`rand` name that is neither probed nor explicitly exempt fails
  the suite.

  New `onehot(v, width)` — the length-`width` indicator row of `v`: 1 at index `v`, 0 elsewhere.
  A random `v` yields a row of indicator RVs; a constant `v` folds to a constant array.

  Unknown module calls now suggest the nearest registered name (`math::sqtr` → "did you mean
  `math::sqrt`?"), with an edit-distance budget that stays quiet for genuinely unrelated names.

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
