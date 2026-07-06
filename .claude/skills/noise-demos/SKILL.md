---
name: noise-demos
description: Build "cool demos" for the Noise website (packages/www/) — scroll-driven, self-animating visualizations that turn a probability idea into a sticky canvas + narrated steps + a real-engine confirmation. Use when creating or editing a demo component in packages/www/src/components or designing a new interactive figure for the landing page.
---

# Building Noise demos

The packages/www/ site teaches probability through **scrollytelling demos**: a sticky visual on the left, a
column of short narrated steps on the right. As you scroll, the visual animates itself and the
matching Noise code reveals line by line — then the *real* WASM engine runs the same program and
confirms the number. No buttons, no "click to play": the story plays as you read.

This skill distills the know-how from the existing demos (`ConceptDemo`, `CircleDemo`,
`BirthdayDemo`, `BellDemo`, `AmFmDemo`, `PrisonersDemo`, `LargeNumbersDemo`) into a repeatable
recipe. **Read one real component alongside this guide** — `CircleDemo.astro` is the cleanest
WebGL example, `LargeNumbersDemo.astro` the cleanest 2D-canvas one.

## What makes a good demo

Every demo follows the same dramatic arc. Pick an idea that fits it:

1. **A surprising or counterintuitive result** — π falls out of random darts; 23 people share a
   birthday more often than not; flat noise summed becomes a bell; FM shrugs off static that wrecks AM.
2. **It's literally a Noise program** — the visual *is* the program running. The code shown on the
   stage is the code the engine runs at the end (`code shown == code run`). Don't visualize something
   Noise can't express.
3. **It converges as samples accumulate** — the payoff is watching an estimate settle onto the truth
   (law of large numbers made visible). Show the running estimate as a hero number.
4. **It reads in 4–5 beats** — one idea per step: *set up the draws → ask the question → watch it
   happen → here's the answer (engine confirms)*. `BellDemo` runs 8 steps as "two acts"; most run 4.

If an idea doesn't have a number that converges and a 4-beat story, it's a static figure, not a demo.

## The anatomy (don't reinvent this)

The layout, sticky behavior, step fading, code-reveal styling, and responsive collapse are **already
solved** in `packages/www/src/styles/global.css` under `--- shared scrollytelling demo scaffold ---`. Reuse
these classes; never re-implement the grid/sticky/observer geometry:

```
.scrolly                     ← root (your demo also gets a unique class, e.g. .pi-demo)
  .scrolly-grid              ← 2-col grid (minmax(0,1fr) / minmax(0,0.6fr)); collapses <860px
    .scrolly-stage           ← sticky, full-height, vertically-centered; pinned on mobile
      .stage-inner.glass     ← the bordered card
        .stage-title
        .stage-canvas <yourclass>  ← holds the <canvas>; size it in your scoped <style>
        <CodePanel ... />    ← the program, with progressive line reveal
        .stage-foot          ← live readouts: .stat, .stat.key (the hero number, maroon)
        .stage-engine        ← the real-engine confirmation line
      figcaption.figure-caption  ← "Figure N. ..." — lives INSIDE the sticky stage
    .scrolly-steps
      .scrolly-step[data-step="0"]  ← h4 + p; .active when centered (JS toggles)
```

Color palette (match these exactly — they're the site's ink/teal/maroon on paper):

| role | 2D canvas (`rgba` string) | WebGL (`vec3` 0–1) | hex |
|------|---------------------------|--------------------|-----|
| canvas background | `#f4f1e8` | `0.957,0.945,0.91` | paper-2 |
| samples / bars (teal) | `31,111,139` | `0.12,0.43,0.55` | `#1f6f8b` |
| answer / match (maroon) | `138,45,74` | `0.54,0.18,0.29` | `#8a2d4a` |
| ink / axis | `27,26,23` | `0.5,0.47,0.43` | `#1b1a17` |
| hairline / faint | `221,215,199` | `0.74,0.68,0.6` | `#ddd7c7` |

## The script skeleton

Every demo's `<script>` (a real Astro module script — TS, bundled, in the `.astro` file) does the
same six things. Here is the canonical shape, distilled:

```ts
import { runNoise, engineMetrics } from '../lib/noise';

const root = document.querySelector('.my-demo') as HTMLElement | null;
const canvas = document.getElementById('my-canvas') as HTMLCanvasElement | null;
if (root && canvas) {
  // readout elements + the code lines (for progressive reveal)
  const piEl = document.getElementById('my-pi')!, engineEl = document.getElementById('my-engine')!;
  const codeBody = root.querySelector('.code-body') as HTMLElement | null;
  const lines = codeBody ? Array.from(codeBody.querySelectorAll<HTMLElement>('.cl')) : [];
  if (codeBody) codeBody.classList.add('progressive');   // dims unrevealed lines

  // 1) STEPS table — one row per narrated step. `show` = last code line revealed,
  //    `add` = lines that flash with the accent rule, `run` = animate during this step.
  const STEPS = [
    { show: 1, add: [0, 1], run: false },
    { show: 3, add: [3],    run: false },
    { show: 5, add: [5],    run: true  },
    { show: 5, add: [],     run: true  },   // last step triggers the engine
  ];
  let engineDone = false, stepRun = false;

  function setStep(i: number) {
    const c = STEPS[i];
    stepRun = c.run;
    lines.forEach((el, k) => { el.classList.toggle('shown', k <= c.show); el.classList.toggle('added', c.add.includes(k)); });
    for (const el of root!.querySelectorAll('.scrolly-step'))
      el.classList.toggle('active', Number((el as HTMLElement).dataset.step) === i);
    if (i === STEPS.length - 1 && !engineDone) { engineDone = true; engineResult(); }
  }

  // 2) rendering — see "Two rendering paths" below. Use a DPR-aware resize + ResizeObserver.
  // 3) the sim loop — accumulate samples only when `inView && stepRun`, then draw():
  function loop() {
    if (inView && stepRun && count < MAX) { /* draw N samples, update readouts */ }
    raf = requestAnimationFrame(loop);
  }

  // 4) replay: a SEPARATE observer (threshold 0.2) tracks whether the stage is on-screen, and
  //    RESETS when it leaves — so scrolling back up replays the animation from scratch.
  function resetReplay() { /* zero counters; */ draw(); setStep(0); }
  new IntersectionObserver((es) => { for (const e of es) { inView = e.isIntersecting; if (!inView) resetReplay(); } }, { threshold: 0.2 }).observe(canvas);

  // 5) the real engine — runs the SAME program shown on the stage, prints value + throughput.
  async function engineResult() {
    engineEl.textContent = 'engine (10⁶ samples)…';
    try {
      const r = await runNoise('X ~ rand::unif(-1,1); Y ~ rand::unif(-1,1); 4 * P(X**2 + Y**2 < 1)');
      engineEl.innerHTML = `engine: π ≈ <span class="ok">${r.value ?? r.output.trim()}</span> — ${engineMetrics(r)}`;
    } catch { engineEl.textContent = ''; }
  }

  // 6) step observer — fires setStep() as each step scrolls through the middle band.
  const io = new IntersectionObserver(
    (es) => { for (const e of es) if (e.isIntersecting && e.intersectionRatio > 0.5) setStep(Number((e.target as HTMLElement).dataset.step)); },
    { threshold: [0.5], rootMargin: '-18% 0px -18% 0px' },
  );
  for (const el of root.querySelectorAll('.scrolly-step')) io.observe(el);
  setStep(0);
}
```

**Two observers, deliberately:** one (`threshold: [0.5]`, `-18%` margins) decides *which step* is
active; another (`threshold: 0.2` on the canvas) decides *whether to animate / reset*. Don't merge them.

## Two rendering paths

**Reach for a 2D canvas first.** It's far less code and handles every line/bar/scatter chart. Use
WebGL only when you're drawing **thousands of moving points** (raining darts, a Galton cascade).

### 2D canvas (default — see `LargeNumbersDemo.astro`)

```ts
const ctx = canvas.getContext('2d')!;
const dpr = Math.min(window.devicePixelRatio || 1, 2);   // cap at 2 — retina without waste
let W = 0, H = 0;
function resize() {
  const r = canvas.getBoundingClientRect();
  W = Math.floor(r.width); H = Math.floor(r.height);
  canvas.width = Math.floor(W * dpr); canvas.height = Math.floor(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);                // now draw in CSS pixels
}
new ResizeObserver(() => { resize(); draw(); }).observe(canvas);
// draw(): clearRect → fill paper-2 bg → axes/target lines → data. Map data→pixels with helpers
// like `const x = i => padL + ...; const y = p => padT + (1-p)*plotH`.
```

### WebGL (thousands of points — see `CircleDemo.astro`, `BellDemo.astro`)

Use the tiny inline-shader helpers the demos share (copy them; don't add a library):

```ts
const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
const sh = (t, s) => { const x = gl.createShader(t)!; gl.shaderSource(x, s); gl.compileShader(x); return x; };
const mk = (v, f) => { const p = gl.createProgram()!; gl.attachShader(p, sh(gl.VERTEX_SHADER, v)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, f)); gl.linkProgram(p); return p; };
```

Patterns to copy: a full-screen background triangle for the paper/figure fill; a **point program**
(`gl_PointSize` + a round-mask in the fragment shader: `vec2 d=gl_PointCoord-0.5; if(dot(d,d)>0.25) discard;`)
for scatter; a **flat-color triangle program** with a `rect()` helper for bars/curves. Preallocate
`Float32Array(MAX*2)` position buffers and `bufferData(..., DYNAMIC_DRAW)` each frame. Clip space is
`[-1,1]`; multiply positions by ~0.94 to leave a margin. **Always guard `if (gl) { ... }`** — WebGL
can be unavailable.

## Performance & correctness rules

- **Cap `dpr` at 2.** `Math.min(window.devicePixelRatio || 1, 2)`.
- **Bulk-sample per frame, not per-rAF-one.** Add tens–thousands of samples each frame
  (`PER_FRAME`), cap total (`MAX` / `count < MAX`), and stop when converged. The screen draws at
  60fps; the sim runs much faster underneath.
- **Animate only when visible.** `if (inView && stepRun)` — never burn CPU on an off-screen canvas.
- **Respect reduced motion.** `const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;`
  When set, skip the animation and fast-forward to the converged state (e.g. `ConceptDemo` runs
  200k samples up front instead of raining them).
- **Cheap estimators beat clearing arrays.** See `BirthdayDemo`'s generation-stamped duplicate table
  (`stamp: Int32Array`, bump `gen` per trial) — no per-trial allocation.
- **The engine call is the proof.** The string passed to `runNoise()` must match the `CodePanel`
  `code`. Show `r.value ?? r.output.trim()` and append `engineMetrics(r)` (samples · ms · samples/s).

## Wiring it up

### On the landing page

1. Create `packages/www/src/components/MyDemo.astro` (frontmatter `code`, the `.scrolly` markup, scoped
   `<style>` for canvas size, the `<script>`). Give the root a unique class and ids.
2. Import and place it in `packages/www/src/pages/index.astro` inside the tour section. Bump the figure
   number in your `figcaption`.

## Build & verify

```sh
cd packages/www
./node_modules/.bin/astro build        # skips the wasm rebuild; catches type errors
```

Then preview and **actually scroll it**: confirm steps fade in/out, code reveals line by line, the
animation runs and resets on scroll-up, the hero number converges, and the engine line shows a real
result. A demo that builds but doesn't animate is the most common failure — check `inView`/`stepRun`
gating and that your ids match between markup and script.

## Checklist for a new demo

- [ ] Idea has a surprising result, a convergent number, and a 4–5 beat story
- [ ] It's a real Noise program; `CodePanel` code == `runNoise()` string
- [ ] Reuses `.scrolly*` scaffold; unique root class + ids; figure number set
- [ ] DPR capped at 2; `ResizeObserver`; animate only when `inView && stepRun`
- [ ] Progressive code reveal wired (`.progressive`, `show`/`add` per step)
- [ ] Replay-on-leave reset; `prefers-reduced-motion` fast-forward
- [ ] `if (gl)` guard (WebGL) or 2D fallback; engine confirmation with `engineMetrics`
- [ ] `astro build` passes; scrolled and visually verified
