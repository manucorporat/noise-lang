# PLAN-FLINT — plots become Flint specs, rendering leaves the compiler

**Date:** 2026-07-10 · **Status:** **FL1 + FL2 shipped** (2026-07-10). FL3–FL5 not started.

## The decision

Today Noise re-implements charting three times: ~235 lines of ASCII renderers in
`introspect.rs` (sparklines, `█`-bars, heat glyphs), ~235 lines of hand-rolled SVG in
`Playground.astro`, and a bespoke JSON shape (`IntrospectionOut`) in between. Every new
view (fan, heatmap, …) costs three implementations, and they drift (fan renders in the
CLI but the playground mis-routes it — the TS types never learned the `fan` variant).

We replace all of that with one contract: **the Noise runtime always emits a
[Flint](https://github.com/microsoft/flint-chart) `ChartAssemblyInput` spec as the plot
payload.** The web side never converts Noise data to Flint at runtime — it receives a
finished spec and renders it with stock libraries. We do **not** re-implement Flint; we
are a *producer* of its spec, exactly like a pandas user is a producer of a plot call.

```
noise-core (Rust)                      browser (www playground)
introspect compute (KEEP)              flint-chart: assembleVegaLite(spec)
  └─ flint.rs: Payload → spec (NEW)      └─ vega-embed renders (stock)
       └─ Output::Plot = spec JSON     ← the only contract between the two
```

- `plot::flint(spec)` renders *anything* Flint can express (escape hatch).
- `plot::hist/fan/line/scatter/…` stay as shortcuts: they run the computation they run
  today (binning, quantiles, joint sampling) and emit a spec under the hood.
- **The layering guarantee:** every shortcut is exactly `stats:: compute` → spec
  template (`flint.rs`) → the same emit sink `plot::flint` uses. Nothing a shortcut
  does is privileged — `plot::fan(path)` ≡ `plot::flint(fan_spec_of(stats::fan(path)))`,
  and a user could rebuild any shortcut in userland once FL2+FL3 ship.
- Every computation a `plot::` shortcut performs is **also exposed as a raw-data
  builtin** (`stats::histogram(x, bins)` → array), so users can get the numbers, not
  just the picture.
- CLI: the ASCII art renderers get deleted. v1 prints a one-line text card per plot;
  later phases serve the run as HTML (print a link) and render inline images via the
  kitty graphics protocol where the terminal supports it.

## What Flint is (research findings)

- **A chart compiler, not a renderer.** npm `flint-chart@0.2.0` (MIT, Microsoft, zero
  runtime deps). Input: `ChartAssemblyInput { data: {values}, semantic_types,
  chart_spec: {chartType, encodings, chartProperties, canvasSize} }`. Output of
  `assembleVegaLite(input)` (also `assembleECharts`/`assembleChartjs`): a complete
  backend-native spec you hand to vega-embed / echarts / chart.js.
- **Semantic types drive design**: fields are tagged `Number`/`Count`/`Price`/
  `Percentage`/`Correlation`/… and Flint picks formats, zero-baselines, color schemes.
  A `Price` axis formats as currency for free — nice fit for the finance track.
- **40+ chart types** on the Vega-Lite backend. The ones our payloads map onto, all
  verified by probe: `Histogram` (raw values, Flint bins), `Bar Chart` (our pre-binned
  histograms), `Line Chart` (incl. multi-series via `y: [a, b, …]`), `Range Area Chart`
  (`y`/`y2` — the fan band), `Scatter Plot`, `Boxplot`, `Density`, `Heatmap`, `KPI
  Card`. (Registered names are exact; "ECDF" as a name failed — resolve names against
  the registry, not the docs.)
- **Layering is post-compile.** One Flint input = one chart. For the fan (two bands +
  median) we compile three Flint specs and merge them into a Vega-Lite `layer` — the
  Flint-sanctioned recipe ("compile, then edit the backend JSON minimally"). Verified.
- **Data rides inside the spec** (`data.values`, inline rows). Remote URLs are
  disabled. So the engine must keep doing the aggregation: we ship bins/quantiles/
  capped point clouds, never the 10⁶ raw draws. Our existing caps (30 bins, 800
  scatter points, 1024 fan cols) already fit this budget.

## Demo (validated 2026-07-10)

Prototype sources + captured payloads live in **`plans/flint-demo/`** (see its README
to rebuild). One real Noise program (barrier option: 52-week GBM, knock-out at 75) ran
through a freshly rebuilt wasm engine; its actual `IntrospectionOut` payloads were
mapped to Flint inputs and rendered by vega-embed:

- `plot::fan(path)` → 2 × Range Area (q05–q95, q25–q75) + median Line, layered → a
  proper price cone (vs today's stacked sparklines).
- `plot::hist(st)` → pre-binned Bar Chart, n=200 000 in 30 bins, ~2 KB of spec.
- `plot::scatter(st, payoff)` → Scatter Plot; the knock-out hockey stick is instantly
  readable.
- `Print` text interleaves with the charts in source order (the `log` stream already
  guarantees this).

Files: `gen-data.mjs` (runs Noise wasm in Node, dumps payloads → `noise-log.json`,
checked in), `mapper.mjs` (**the prototype of the Rust emitter** — payload →
ChartAssemblyInput, incl. the layered-fan recipe), `probe.mjs` (chart-type
compatibility probes), `render-check.mjs` (headless SVG verification).

Bundle cost for www: flint-chart + vega + vega-lite + vega-embed ≈ 1.1 MB minified
(~350 KB gzip), lazy-loadable on first plot.

## Phases

### FL1 — Rust emitter + playground swap (the core) — ✅ **shipped 2026-07-10**

What landed, and where it departed from the plan below:

- `crates/noise-core/src/flint.rs` — `to_flint(&Summary) -> Plot { title, text, charts }`.
  `noise-core` gained one dependency, `serde_json` (a Flint spec *is* JSON).
- The ASCII renderers are gone (`introspect.rs` lost ~235 lines); `Display for Summary` is now the
  one-line text card, so the CLI, the REPL, and a chartless web card all print the same string.
- `noise-wasm` emits one `PlotOut { title, text, charts, error? }`; the `IntrospectionOut` union and
  the missing-`fan` bug are gone by construction. `packages/core/src/index.ts` exports `Plot` +
  `ChartSpec`.
- `packages/www/src/lib/plot.ts` — the whole web renderer (~130 lines incl. comments), lazy-loaded.
  `Playground.astro` lost ~205 lines of hand-rolled SVG.
- `packages/www/scripts/check-flint-names.mjs` — the registry name-compat guard, run from `prebuild`.
  Its `EMITTED` list mirrors `flint.rs`'s `REGISTERED`; `flint::tests::assert_well_formed` pins the
  other end.

**Decisions taken during the build** (each verified by rendering, not by reading docs):

- **`ValueCard` → `Ranged Dot Plot`, not `KPI Card`.** A KPI Card hard-codes a white fill and dark
  text into its marks — unthemeable. The dot plot draws the actual 95% interval (low · estimate ·
  high). Its row axis is a *blank* category (a field named `" "`) because Flint titles an axis after
  its field and labels it with its values; a one-variable chart has nothing to say there, and the
  naive spec printed the variable name twice with a three-entry legend explaining three dots.
- **`DistGrid` series is banded, like the fan.** When any `sd > 0` it emits Range Area (mean±sd) +
  Line, layered — so nothing is lost vs. the old `seriesSvg` whiskers. A deterministic vector is
  just the line.
- **Layered line charts carry `includeZero_y: false`.** A `Range Area` never anchors at zero and a
  `Line Chart` does; merging the two made Vega-Lite warn and pick one arbitrarily.
- **Heatmap indices are `Rank`, not `Category`.** A nominal axis sorts `"10"` before `"2"`.
- **A `Percentage` needs `intrinsicDomain: [0,1]`** before Flint's percent formatter turns on;
  without it a share axis reads `0.42`, not `42%`.
- **Field names are the source labels.** Flint escapes and `title`s them, so `path[51]` is a legal
  field rather than a Vega-Lite nested lookup. Collisions with a spec's fixed columns
  (`hist(count)`, `scatter(x, x)`) get a `_` suffix.
- **Two post-compile edits in the renderer, both reconciling Flint with itself**, not designing a
  chart: (a) the layer merge, and (b) dropping `scale.zero` where Flint pinned an explicit `domain`
  — see *Open questions* for the upstream issue.

The plan as written:

1. **`crates/noise-core/src/flint.rs` (new).** `pub fn to_flint(&Summary) ->
   serde_json::Value` translating every existing payload:
   | Payload | Flint chartType | Notes |
   |---|---|---|
   | `Dist1` (hist) | Bar Chart | bin midpoints + counts (pre-binned; don't ship draws) |
   | `Dist1` (boolean) | Bar Chart | two bars, `Percentage` semantic type |
   | `Dist2` | Scatter Plot | capped points; corr in title |
   | `DistGrid` series | Line Chart | mean line; later + band like fan |
   | `DistGrid` matrix | Heatmap | |
   | `FanChart` | 2×Range Area + Line, layered | post-compile `layer` merge (see recipe in mapper.mjs) |
   | `ValueCard` | KPI Card | val ± se |
   | `CorrMatrix` | Heatmap | `Correlation` semantic type ([-1,1] diverging for free) |
   | `Explain` | Bar Chart (horizontal) | driver shares |
   The compute structs in `introspect.rs` stay exactly as they are — `flint.rs` is a
   serializer at the output boundary. Port `mapper.mjs` 1:1; snapshot-test the emitted
   JSON per payload type.
2. **Wire the seam — payload shape (decided).** Stay Flint-semantic all the way to the
   renderer: `noise-wasm` emits
   `{kind:"plot", title, text:<fallback line>, charts:[ChartAssemblyInput, …]}`.
   One-chart plots have a single-element `charts`; the fan carries its three inputs
   (band, band, median) and the renderer compiles each with `assembleVegaLite` and
   merges into a Vega-Lite `layer` (recipe prototyped in `mapper.mjs::fanToFlint`).
   We never emit backend (Vega-Lite) JSON from Rust — specs survive regeneration,
   backend JSON doesn't. Replaces the per-type `IntrospectionOut` union; update
   `packages/core/src/index.ts` (union collapses to one variant — this also fixes
   the missing-`fan` bug by construction).
3. **Playground: replace the Output tab's rendering.** The `#pg-output` panel in
   `Playground.astro` already renders the run as an interleaved stream (`.out-text`
   Print lines + `.pg-card` plot cards in source order) — keep that stream shape,
   swap what fills a card:
   - Delete `renderCard` and every hand-rolled builder: `histSvg`, `scatterSvg`,
     `seriesSvg`, `heatmapSvg`, `gridSvg`, `ciSvg`, `barsSvg`, `renderDist1/2`,
     `renderExplain`, `renderValue`, `renderGrid`, `renderCorrMatrix`, the color
     ramps (~lines 429–663).
   - New generic card: title from the payload, then `charts.map(assembleVegaLite)`
     → (length > 1 ? layer-merge : single) → `vega-embed` with `{actions:false}`.
   - Lazy-load `flint-chart` + `vega-embed` on the first plot item (dynamic
     `import()`), so text-only runs don't pay the ~350 KB gzip.
   - Theme: key a vega config overlay (background, axis/label colors) off the site's
     light/dark state; re-render cards on theme toggle.
   - Sizing: `canvasSize` from the card's measured width so charts fit the panel on
     mobile (the panel is a flex pane, see `.pg-output-wrap`).
   - The introspection sidecar (`run_with_introspection` variable cards) goes through
     the same generic card — one renderer for both streams.
4. **CLI.** Delete `fmt_*` + `Display for Summary` ASCII art (introspect.rs:440-675);
   print the fallback text card (`hist(st): n=200000 mean=104.3 sd=18.9 q05=…`). The
   stats are already in the payloads; nothing is lost except the art.

### FL2 — raw data out of the plot jail — ✅ **shipped 2026-07-10**

`stats::histogram(x[, bins])` → `[[midpoints],[counts]]`, `stats::quantiles(x, [qs])`,
`stats::moments(x)` → `[n, mean, sd, min, max]`, `stats::fan(path)` → 6×cols matrix
(`q05,q25,q50,q75,q95,mean`), `stats::corr(a, b)` → number / `stats::corr(v)` → n×n matrix.

Namespace: `stats::` (they force sampling; `math::` never does). **Always qualified**, like
`plot::` — `use stats;` grants nothing unqualified, because `fan`/`corr`/`moments` would shadow far
too much, and a bare `corr(a,b)` must keep meaning the always-on introspection summary. A bare
`fan(path)` now errors with "write `stats::fan(...)`" instead of "unknown function".

**How the layering guarantee is actually enforced.** Not by discipline — by there being one function:

| Shared function | `plot::` caller | `stats::` caller |
|---|---|---|
| `introspect::histogram(draws, boolean, nbins)` | `Dist1::from_draws` (30 bins) | `stats::histogram` (any bins) |
| `introspect::draws(graph, root, …)` | `introspect::dist1` | `stats::{histogram,quantiles,moments}` |
| `introspect::quantile_sorted` | `Dist1::from_draws` | `stats::quantiles` |
| `Eval::fan_chart` | `plot::fan` | `stats::fan` |
| `Eval::corr_matrix_of` | `corr(v)` / `plot::corr(v)` | `stats::corr(v)` |
| `introspect::dist2` | `corr(a,b)` / `plot::scatter` | `stats::corr(a,b)` |

Same budget (`INTROSPECT_N`, 200 000) and same seed as the charts, so the two agree *exactly*, not
approximately. The tests assert bit-equality against the plot payload on the same engine — if anyone
reimplements one of the two, they fail. Consequence to document (done, in LANG.md): `Q(x, 0.5)` and
`stats::quantiles(x, [0.5])` can differ in the final places, because `Q` draws `P`'s budget. `Q` is
the estimator; `stats::quantiles` is what the picture is made of.

Also: `midpoints()` moved onto `Histogram`, so `flint.rs` no longer computes bin centers itself;
check mode returns correctly-*shaped* placeholders (indexing a `stats::` result under
`noise validate` must not raise a phantom error) and draws nothing.

### FL3 — `plot::flint(spec)` escape hatch

Now unblocked on the data side: FL2 gives a user the numbers, so `plot::flint` only needs to accept
a spec. The remaining blocker is language-level: Noise has no record/object literal, and strings have
no escapes (can't even hold JSON's `"`). Two options:
- **v1 (cheap):** raw strings (`'…'` or backticks) + `plot::flint(json_string)` with
  build-time validation against the Flint registry; data composed via string `+` or a
  `{data}` placeholder filled from a Noise array arg: `plot::flint(spec_str, xs)`.
- **v2 (right):** a build-time record literal `{ chartType: "Bar Chart", encodings: {…} }`
  — deterministic values only. Bigger spec, unlocks more than plots.
Recommend shipping v1 behind the same `plot::flint` name so v2 is a pure upgrade.

### FL4 — CLI serves HTML

`noise run --html out.html` (and later `noise serve`): write the run's log (text +
specs) into a self-contained HTML using the same renderer bundle, print the
`file://`/localhost link. The demo's `demo.html` assembly is the template. This is
where the CLI catches back up to the browser without ever owning chart code again.

### FL5 — inline terminal graphics (kitty protocol)

When the terminal supports it, render the actual chart inline instead of the text
card. Detection: kitty graphics protocol query escape (kitty, Ghostty, WezTerm) or the
iTerm2 inline-image variant; fall back to the FL4 link / text card otherwise.
Rendering a Vega-Lite spec needs a JS runtime, so this is opportunistic, layered:

1. spec → SVG: shell out to `node`/`bun` if on PATH, running a small bundled script
   (flint-chart + vega, same bundle as FL4's HTML template, executed headless — the
   demo's `render-check.mjs` already proves this path).
2. SVG → PNG in pure Rust via `resvg` (no native deps).
3. PNG → terminal via kitty escape sequences (`kitty +kitten icat`-style chunked
   base64; `viuer`/`kitty-image` crates handle protocol details).

No JS runtime found → no degradation of correctness, just no inline art. Importantly
this still honors the "no chart code in the compiler" rule: the CLI transports pixels,
it never lays out a chart.

## Open questions

- **Flint forces `zero: true` on a bar chart's x.** `computeZeroDecision` calls `isBarLike` on the
  *mark*, so a pre-binned histogram's x — a coordinate, not a magnitude — is anchored at 0, wasting
  half the canvas for a price around 100. Flint *also* computes the correct explicit `domain` for
  that scale, and Vega lets `zero` silently widen it: the two outputs contradict each other. Neither
  `intrinsicDomain` (which `resolveFieldSemantics` derives a zero-class from, then throws away) nor
  `includeZero_x` (position-mark templates only) reaches it. `plot.ts::reconcileDomain` drops the
  redundant `zero`; **file this upstream** and delete that shim when it lands.
- ~~**Pin flint-chart.**~~ Done: `packages/www/scripts/check-flint-names.mjs` asserts the emitted
  chart-type names against the registry, from `prebuild`.
- **Where does `assemble*` run?** Browser for now (playground). If we ever want
  server/CLI PNG export, `assembleVegaLite` also runs fine in Node (verified headless
  SVG render).
- ~~**describe() cards**~~: a scalar draws its 95% interval (`Ranged Dot Plot`) and states every
  number in its text line. An exact scalar emits no chart at all.
- **`plans/flint-demo/`** is now superseded by `flint.rs` + `plot.ts`. Keep it as the research
  record, or delete it?
- **ECharts/Chart.js backends**: free optionality from the spec; no reason to use them
  now, but don't preclude (emit ChartAssemblyInput, not Vega-Lite).
- The stale-wasm footgun: `packages/core/wasm` was 4 days old and missing F1–F3;
  rebuilt today. Consider a CI check that the committed wasm matches noise-core HEAD.
