---
"@noiselang/core": minor
---

Array slicing: indexing with a deterministic array of indices selects into a new array. Since
`a..b` is already an array, `xs[0..r]` takes the first `r` elements (Rust-style half-open), and
any index list works — `xs[[2, 0, 0]]` reorders and repeats, `xs[perm]` applies a deterministic
permutation. Each index obeys the existing scalar rules (constant non-negative integer, bounds
checked element-wise); random indices inside the array are an error. It is pure build-time sugar
for `[for i in inds { xs[i] }]`, so nothing changes for codegen. The secretary example's
observation phase collapses from a fold loop to `bar = max(quality[0..r])`.
