---
"@noiselang/core": minor
---

Slider values are now true runtime uniforms in every backend (PLAN-UNIFORM-INPUTS P1): the WGSL and WASM emitters lower `input::` cones instead of declining them, and the value rides the dispatch (a GPU params block / a host-written memory region) rather than the compiled artifact. Dragging a slider re-dispatches cached pipelines and cached kernel instances with zero recompiles, and slider-heavy documents GPU-accelerate — `turboquant.noise` drops from ~14 s to under 0.1 s on the GPU CLI.
