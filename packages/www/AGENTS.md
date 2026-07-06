# AGENTS.md — the Noise website (`packages/www/`)

The marketing + learning site for the Noise probabilistic language, and the in-browser
playground. It runs the **real** Noise engine, compiled to WebAssembly, entirely client-side.
This file is the single source of truth for working on `packages/www/`. Read it before editing; update it
when you change architecture or add patterns.

> Sibling context: the repo root has `AGENT.md` (engine state), `LANG.md` (language spec),
> `plans/PLAN.md` (roadmap). This file is **only** about the website.

---

## 1. What it is

A static **Astro 5** site (package manager: **pnpm**). The home page is a single long document in
the style of an academic paper (LaTeX/Computer-Modern aesthetic, paper-white, left-aligned), built
around five **scroll-driven WebGL story demos** and a full Monaco-based playground.

Page order (`src/pages/index.astro`), top to bottom:

1. **Top bar** — sticky nav (`Tour · Playground · Examples · How it works · GitHub`).
2. **Masthead** — title, italic meta line, an `Abstract.` block.
3. **Figure 1** — `ShaderFigure.astro`: an animated, cursor-reactive ink-on-paper fractal-noise
   field (the language's namesake), captioned like a figure.
4. **The tour** (`#tour`) — five scrollytelling demos, simplest → hardest:
   π (`CircleDemo`) → fair die (`DiceDemo`) → CLT (`CltDemo`) → Galton board (`GaltonDemo`) →
   AM vs FM (`AmFmDemo`, the flagship). Figures 2–6.
5. **Playground** (`#playground-section`) — `Playground.astro`: Monaco editor + Run + Share +
   category dropdown + output, running the WASM engine.
6. **Examples** (`#examples`) — `Gallery.astro`: category-grouped cards that load programs into the
   playground.
7. **How it works** (`#ideas`) — three prose "core ideas" with code listings.
8. **Footer**.

---

## 2. Quick start

```sh
cd www
pnpm install
pnpm dev          # http://localhost:4321  (runs `predev` → rebuilds WASM first, ~1 min cold)
pnpm build        # production build to dist/ (runs `prebuild` → rebuilds WASM first)
pnpm preview      # serve the production build
```

**Iterating fast:** `pnpm dev`/`pnpm build` rebuild the WASM every time via `predev`/`prebuild`
(~1 min). If `src/wasm/pkg/` already exists and you're only touching front-end code, skip the
rebuild with **`pnpm exec astro dev`** / **`pnpm exec astro build`** directly. Only re-run
`pnpm run wasm` when you change Rust in `crates/noise-core` or `crates/noise-wasm`.

**Toolchain (already set up on the dev machine; needed for the WASM step):**
- `rustup` lives at `~/.cargo/bin` (a stable toolchain *separate* from Homebrew's `rustc`) with the
  `wasm32-unknown-unknown` target installed.
- `wasm-pack` (installed via Homebrew).
- The `wasm` npm script prepends `$HOME/.cargo/bin` to `PATH` so wasm-pack uses rustup's cargo
  (Homebrew's rustc has no wasm32 std → "can't find crate for `std`" if it's used by mistake).

---

## 3. How the engine reaches the browser

```
crates/noise-core  (the language: lexer→parser→eval, Print buffered via Engine::drain_output)
        │
crates/noise-wasm  (wasm-bindgen bindings)  ──wasm-pack──▶  packages/www/src/wasm/pkg/  (gitignored)
        │                                                         │
        │  exports:  run(src) -> JSON string                      │
        │            version() -> string                          ▼
        └────────────────────────────────────────────▶  src/lib/noise.ts  (loadNoise / runNoise)
                                                                  │
                                                  Playground + every demo's engine readout
```

- **`run(src)` returns a JSON string** parsed into `{ ok, value, output, error }`
  (`NoiseResult` in `src/lib/noise.ts`). `value` is the last statement's display form (or `null`
  for `unit`/error); `output` is everything `Print` emitted (captured, since `wasm32` has no
  stdout); `error` carries a spanned message on failure. **`run` never throws.**
- **`loadNoise()`** initializes the module once (lazily); **`runNoise(src)`** awaits it and returns
  the parsed result. Use it from any client `<script>`.
- The `pkg/` directory (and `crates/noise-wasm/pkg/`) are **build artifacts, gitignored**, rebuilt
  by `pnpm run wasm`.

---

## 4. Directory map

```
packages/www/
  astro.config.mjs        # static site; vite.server.fs.allow:['..'] so we can ?raw-import repo examples
  package.json            # scripts (incl. the wasm build), deps: astro, monaco-editor
  src/
    pages/index.astro     # the whole page: masthead, tour, playground, gallery, ideas, footer
    layouts/Layout.astro  # <head>, scroll-reveal IntersectionObserver, scroll-progress bar
    styles/global.css     # design tokens + ALL shared classes (.glass, .scrolly-*, code tokens, reveal)
    components/
      ShaderFigure.astro  # Figure 1: full-bleed-ish ink fractal-noise figure, cursor-reactive (WebGL)
      CircleDemo.astro    # π — darts in a circle           (scrollytelling, WebGL points)
      DiceDemo.astro      # fair die — histogram → 1/6       (scrollytelling, WebGL bars)
      CltDemo.astro       # CLT — scroll raises N, → bell    (scrollytelling, WebGL bars + curve)
      GaltonDemo.astro    # bean machine → binomial          (scrollytelling, WebGL points + bars)
      AmFmDemo.astro      # AM vs FM phasor explainer        (scrollytelling, WebGL lines/points) ★flagship
      CodePanel.astro     # reusable code listing: filename header + per-line highlighted, addressable
      Playground.astro    # Monaco editor + run/share + grouped example dropdown + output
      Gallery.astro       # category-grouped example cards (loads into the playground)
    data/
      examples.ts         # gallery/playground examples: globbed repo .noise + curated metadata + categories
    lib/
      noise.ts            # WASM loader + runNoise()
      noise-lang.ts       # Monaco language: Monarch tokenizer + 'noise-paper' light theme
      highlight.ts        # static (non-Monaco) syntax highlighter for code listings
      share.ts            # URL-fragment sharing (#x=<id> / #c=<base64url>)
    wasm/pkg/             # generated WASM package (gitignored)
  public/favicon.svg
```

---

## 5. Design system (LaTeX paper aesthetic)

All tokens and shared classes live in `src/styles/global.css`. **Do not re-invent these per
component** — reuse them so the site stays cohesive.

- **Palette:** `--paper #fbfaf6`, `--paper-2 #f4f1e8` (code/listing bg), `--ink #1b1a17`,
  `--ink-dim`, `--rule` (hairlines/borders), `--link` (hyperref blue), `--accent #8a2d4a` (maroon).
- **Fonts:** `--serif` = Computer Modern web fonts (LaTeX look, CDN + serif fallback) for prose;
  `--mono` = **JetBrains Mono** (CDN) for all code. (We moved off Computer Modern Typewriter — its
  `~` looked wrong.)
- **`.glass`** — was a dark glass panel; now repurposed as a **light bordered panel** (paper bg,
  `--rule` border, subtle radius). Used by the playground, gallery cards, demo stages.
- **`.measure`** (≈720px) for prose columns, **`.wide`** (≈1080px) for interactive/figure blocks.
  Wrap interactive components in `<div class="wide">` (see gotcha #2).
- **No rigid article numbering** — the user explicitly wanted "the design, not a real article," so
  section/subsection counters were removed. Keep headings plain.
- Syntax-highlight accent colors are shared between Monaco (`noise-lang.ts`) and the static
  highlighter (`highlight.ts` + `.code-body .{k,d,q,o,c,n}` in global.css): keyword=maroon,
  distribution=teal `#1f6f8b`, query(P/E/Var/Q)=sienna `#b5651d`, operator=`#a23e6a`,
  comment=`#8a8473` italic, module-prefix(`rand::`…)=`#6a6356` italic.

---

## 6. The scrollytelling demo pattern (the core reusable thing)

Every tour demo shares one structure. Markup:

```html
<div class="scrolly <name>-demo">
  <div class="scrolly-grid">                 <!-- sticky stage | scrolling steps -->
    <div class="scrolly-stage">              <!-- full-height sticky, vertically centers its child -->
      <div class="stage-inner glass">        <!-- the actual panel -->
        <h4 class="stage-title">Title</h4>   <!-- persistent title (stays visible while pinned) -->
        <div class="stage-canvas <name>-canvas"><canvas id="<name>-canvas"></canvas></div>
        <CodePanel filename="x.noise" code={code} class="<name>-code" />
        <div class="stage-foot"> <span class="stat">…</span> <span class="stat key">…</span> </div>
        <p class="stage-engine" id="<name>-engine"></p>
      </div>
    </div>
    <div class="scrolly-steps" id="<name>-steps">
      <div class="scrolly-step" data-step="0"><h4>…</h4><p>…</p></div>
      …                                       <!-- one per narrative step -->
    </div>
  </div>
  <figcaption class="figure-caption"><span class="fig-n">Figure N.</span> caption.</figcaption>
</div>
```

Behavior, implemented in each component's client `<script>`:

- An **`IntersectionObserver`** on `.scrolly-step` (threshold ~0.5, `rootMargin: '-18% 0 -18% 0'`)
  calls `setStep(i)` for the step centered in the viewport.
- `setStep(i)` does three things: (a) toggle `.active` on the step, (b) reveal/highlight code lines,
  (c) set the demo's visual phase. The **last step** triggers the engine readout (`runNoise`) once.
- **Story = code first, then animation.** Each step has `{ show, add, run }`: `show` = highest code
  line index visible, `add` = line indices to flash with the accent rule, `run` = whether the
  animation is live. The opening steps walk the code with a *static* canvas scaffold; the animation
  is **gated** (`inView && stepRun`) and only starts at a later step.
- **Auto-runs on scroll, no buttons.** A second `IntersectionObserver` on the canvas sets `inView`;
  the rAF loop runs only while `inView && stepRun`.
- **CLT is the exception**: it has no "run" gate — each step maps to an `N` (`STEPN = [1,2,4,12]`)
  and re-samples on entry, so scrolling literally morphs flat → triangle → bell.

### Progressive code reveal
`CodePanel` renders each line as `<span class="cl" data-ln={i}>`. In the demo script:
`codeBody.classList.add('progressive')` (dims unrevealed lines), then per step toggle `.shown`
(`ln <= show`) and `.added` (`ln in add`). The CSS for these states is in global.css
(`.code-body.progressive .cl`, `.cl.shown`, `.cl.added`).

### Stage centering (important — don't regress this)
The stage centers vertically via a **full-height sticky wrapper**:
`.scrolly-stage { position: sticky; top: 56px; height: calc(100vh - 56px); justify-content: center }`
centering an inner `.stage-inner`. **Do NOT use `top: 50%; transform: translateY(-50%)`** — that was
tried and it shifts the panel up *out of its section*, overlapping the figure above it (gotcha #4).

### No inner code scroll
The user dislikes scroll traps. `.stage-inner` has **no `overflow`/`max-height`** — the full program
is always shown. Size the canvas (and, for long programs, the code font) so the panel fits; e.g.
AM/FM (14 lines) uses a 150px canvas and `:global(.amfm-code .code-body){font-size:.9rem}`.

---

## 7. WebGL conventions (used by all demos + ShaderFigure)

Hand-rolled WebGL1, no libraries. Per demo, two tiny programs are typical:

- **triangle program** (`attribute vec2 p; uniform vec3 u_col`) for bars/lines/rects, and
- **points program** (`gl_PointSize` + a round-mask `discard` in the fragment) for dots.

Helpers you'll see repeated (copy them; they're small):
- `rect(x0,y0,x1,y1)` → 6 verts (2 tris). `drawTris(verts,col)` / `drawPoints(verts,size,col)`.
- `line(pairs, wd, col)` — thick polyline as triangles. **Must aspect-correct the normal** (scale
  by `aspect()` = W/H) or steep/jagged segments balloon on wide canvases (gotcha #5). `wd` is
  half-thickness in clip-Y; keep waveform `wd` ~0.003–0.006.
- Colors: TEAL `[0.12,0.43,0.55]`, MAROON `[0.54,0.18,0.29]`, INK `[0.42,0.40,0.36]`,
  FAINT `[0.78,0.75,0.69]`; clear color = paper `(0.957,0.945,0.91)`.
- `ResizeObserver` on the canvas → set `canvas.width/height = rect * dpr` (cap dpr at ~2),
  `gl.viewport`. Animate with `requestAnimationFrame`; respect `prefers-reduced-motion` in
  `ShaderFigure`.

`ShaderFigure.astro` is the most advanced shader (Ashima simplex `snoise` + fbm + domain warp +
contour shading + cursor `u_mouse` swirl). Reuse its snoise/fbm GLSL if you need procedural noise.

---

## 8. CodePanel + the static highlighter

- **`CodePanel.astro`** props: `filename`, `code`, optional `class`. Renders a filename header tab
  (`› x.noise`) and the code with each line individually addressable (`.cl[data-ln]`) for
  progressive reveal. Token colors are global (`.code-body .k/.d/.q/.o/.c/.n`).
- **`src/lib/highlight.ts`** — regex highlighter (`LISTING_HL`, `LISTING_LINES`, `hlLine`). Runs at
  build time (server-side), no Monaco needed. Update the `MODULES/KEYWORDS/DISTS/QUERIES` regexes
  here if the language gains builtins. Keep it in sync with the Monaco tokenizer in `noise-lang.ts`.

---

## 9. The playground, examples data, and sharing

- **`Playground.astro`** imports Monaco via **`monaco-editor/esm/vs/editor/editor.api`** (NOT the
  full `monaco-editor`, which bundles every language = ~3.3MB; the API-only import is ~600KB gz).
  Workers: `MonacoEnvironment.getWorker` returns the base editor worker only. Language + theme come
  from `registerNoise(monaco)` in `noise-lang.ts` (theme id `noise-paper`).
- **`src/data/examples.ts`** — the example catalogue. Code is **globbed live** from the repo's
  `../../../examples/*.noise` (via `import.meta.glob(..., { query:'?raw' })`) so the site never
  drifts from what the CLI runs; curated metadata (title/blurb/explanation/analytic/category) is
  merged in. `examplesByCategory()` groups them; `categories` is the display order;
  `defaultExampleId = 'pi'`.
- **`src/lib/share.ts`** — `#x=<id>` for a named example, `#c=<base64url>` for arbitrary code.
  `parseHash`, `buildShareUrl`, `encode/decodeCode`. The playground reads the hash on load and on
  `hashchange`, updates it when you pick an example or hit Share, and copies the link to clipboard.
- Gallery cards / the dropdown dispatch / handle a `noise:load-example` CustomEvent to load a
  program into the editor.

---

## 10. Noise code conventions on the site

- **Show == run.** The code displayed in a demo's `CodePanel` must be the same program passed to
  `runNoise` for that demo's engine readout. Keep them in sync (they're separate strings today).
- **Fully-qualified module paths, no `use`.** Hand-written demo code uses `rand::`, `vec::`,
  `math::`, `signal::` on every call (cleaner, self-documenting). `P`, `E`, `Var`, `Q`, `Print`,
  `Len` are in the always-on `builtin` module — **no prefix**. Statements are `;`-separated
  (newlines are not separators in Noise).
- ⚠️ **Known inconsistency / good first task:** the *tour demos* use qualified paths, but the
  *gallery/playground examples* are the repo `examples/*.noise` files, which still use `use rand;`
  etc. (they're shared with the CLI + golden tests). If you convert those for consistency, do it in
  the repo and re-run `cargo test`, or transform-on-display in `examples.ts` — don't silently break
  the tests.

---

## 11. Gotchas (hard-won; re-reading saves hours)

1. **Astro can drop an `id`.** A `<div id="x" class="x">` once rendered with the `id` stripped.
   Select demo containers by **class**, not id, where it matters (the playground chips bug).
2. **Wrap interactive components in `<div class="wide">`.** A demo placed directly in a `.tour-item`
   with no `.wide` wrapper stretches to full page width (this is why AM/FM was "much wider").
3. **Monaco: import `editor.api`, not the package root** (bundle size, see §9).
4. **Sticky centering = full-height wrapper, not `transform`** (see §6 "Stage centering").
5. **`line()` must aspect-correct** or steep segments look chunky on wide canvases (see §7).
6. **No inner scroll in code panels** — size to fit instead (see §6).
7. **wasm-pack `--out-dir` is relative to the crate dir**, not the cwd — the `wasm` script path is
   `../../packages/www/src/wasm/pkg` for that reason.
8. **`Print` is captured, not printed** — relies on the core `Engine::drain_output` change; the
   WASM `run` returns it in `output`.
9. The agent-browser test window defaults to ~577px tall (unusually short). Layout that "tucks under
   the header" there is usually fine on real laptop heights — verify on a normal viewport before
   chasing it.

---

## 12. How to add things

### A new scrollytelling demo
1. Copy the closest existing demo (`DiceDemo` for bars, `CircleDemo`/`GaltonDemo` for point clouds,
   `CltDemo` for step-driven, `AmFmDemo` for multi-phase + line plots).
2. Keep the §6 markup. Write the `code` string (qualified paths, show == run). Define `STEPS`
   (`{show, add, run}` or step→N), narrative `.scrolly-step`s, and the WebGL draw per phase.
3. Wire the engine readout via `runNoise(...)` at the last step.
4. In `index.astro`: import it and add `<div class="tour-item"><div class="wide"><YourDemo /></div></div>`
   inside `#tour`. Give it the next `Figure N` number.
5. Verify: code reveals correctly, animation gates on scroll, engine number matches, panel fits
   (no inner scroll), width == the other demos.

### A new gallery/playground example
Add the `.noise` file under the repo's top-level `examples/` (so the CLI + tests cover it), then add
a metadata entry + a `categoryOf` mapping in `src/data/examples.ts`. It appears in the gallery and
the playground dropdown automatically.

### Verifying changes
Use the `agent-browser` skill to drive the dev server: scroll demos via
`document.querySelector('.x-demo .scrolly-step[data-step="N"]').scrollIntoView({block:'center'})`,
then read `#x-engine` / stats / `console --level error`. Always end with `pnpm exec astro build` to
confirm a clean production build, and stop the dev server + browser when done.

---

## 13. Deploy

Static output in `dist/` (`pnpm build`). Includes the fingerprinted `.wasm`. Any static host works;
ensure `.wasm` is served with `application/wasm`. `site:` in `astro.config.mjs` is set to a
placeholder (`noise-lang.dev`) — update it for real deploys. The GitHub link in the nav is a
placeholder; wire it to the real repo.
