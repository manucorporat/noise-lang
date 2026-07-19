---
"@noiselang/core": minor
---

Sliders work everywhere a number does, plus `onehot` and "did you mean?" hints.

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
