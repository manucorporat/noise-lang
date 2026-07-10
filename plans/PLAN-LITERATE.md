# PLAN-LITERATE — frontmatter, knobs, and the .noise file as an interactive document

**Date:** 2026-07-10 · **Status:** LT1–LT4 shipped (frontmatter + knobs; template blocks; the
`Document` contract; the Preview tab). LT5 (hover introspection + full example migration) not
started. Builds on PLAN-FLINT (the `log` stream contract) — FL1/FL2 shipped, so charts already ride
the ordered output stream this plan extends.

**LT1 note:** frontmatter is parsed by `serde_yaml` (YAML is a superset of JSON, so the `{ … }`
escape hatch is free) rather than the hand-rolled YAML subset the plan originally proposed —
simpler, and the crate stays wasm-clean (`unsafe-libyaml` is pure Rust). `serde_json`'s
`preserve_order` feature keeps knobs in source order. `meta()`'s `knobs` is an **ordered array**
of `{ name, type, … }` (not a name-keyed object) so order survives JSON without ceremony.

**LT2 note:** templates emit an engine-internal `Output::Note { text, syntax }`; until LT3's
`Document` lands, the CLI prints notes as text and the wasm layer folds them into the existing
text/`log` stream (a `LogItem::Text`). The distinct `note` block + syntax tag arrives with LT3.

**LT3 note:** shipped as the plan's D5 spec. New `doc.rs` (`segment` + `comment_layer` + `assemble`
+ `Document`/`Block`/`Comment` + JSON); `Engine::run_to_document` is the single entry both hosts
call (they can't drift). `lexer::tokenize_with_trivia` records comment spans via a shared inner fn
(hot-path `tokenize` unchanged). Emissions are span-tagged (`Emission { stmt_span, output }`) with a
`MAX_EMISSIONS = 200` cap → `result.truncated`. WASM `run`/`run_with_introspection` return a
`Document` (the latter as `{ document, bindings, introspections }`); `packages/core` bumped to
0.2.0 and exports the `Document`/`Block`/`Comment` types. The JS `run()` enriches the wire
`Document` with legacy convenience fields (`ok`/`value`/`output`/`error`/`stats`/`log`) so the
landing-page demos keep working; the playground's flat output view now renders over `blocks`. The
Preview tab + comment-layer/knob UI is LT4. Also added: unknown-knob-field rejection (catches the
YAML `min:1`-without-a-space footgun).

## The decision

A `.noise` file stops being "source code that happens to print things" and becomes a
**self-describing interactive document**: metadata + tunable parameters in a frontmatter
block, prose in comments that *attach* to the code they describe, multi-line interpolated
text blocks that emit without `Print`, and a run contract that interleaves **the code
itself** with its outputs. The playground's Output panel becomes a **Preview**: code
blocks inline with their notes and charts; knob sliders that re-run the
simulation; hover a variable to introspect it; hover a code line to reveal its comment.

```
---
title: "Roll a die"
knobs:
  dice_sides:    { type: "int", min: 1, max: 100, step: 1, default: 6 }
  target_number: { type: "int", min: 1, max: 100, step: 1, default: 4 }
---

// This comment block attaches to the next two lines of code
// (a blank line after the code would detach it).
Dice ~ rand::unif_int(1, dice_sides);
p = P(Dice == target_number);  // trailing comments attach to their own line

`
P(rolling a ${target_number}) = ${p}
`

plot::histogram(Dice)
```

(That's the single-backtick template form — D3; use ` ```md … ``` ` when the note
should carry a syntax tag, e.g. render as markdown.)

Everything is **data-level**, and we take the freedom to **break the run contract**: a
run no longer returns `{ ok, value, output, error, stats, log }` — it returns **one
`Document`**: frontmatter meta + one flat, ordered array of typed blocks (code, notes,
plots), emitted in source/run order, plus a **comment layer** — each comment a
`(selfSpan, codeSpan?)` pair annotating anything from a single line to a whole
statement group, or nothing (a detached run) — and span links from each emission back
to its producing statement. Every rendering choice —
hide code, show only plots, only text, comments on hover, sliders vs inputs — is a
frontend-side filter over that one array. The CLI is just another renderer of the same `Document`
(notes as text, one-line chart cards), exactly like the Flint split (runtime
emits specs, hosts render).

### Why now

- `packages/www/src/data/examples.ts` hand-curates `{ id, title, blurb, … }` per example
  and joins it to live-globbed `examples/*.noise` — the code can't drift, but the
  *metadata* can. Frontmatter moves the title (and later blurb/category) into the file.
- Examples hardcode constants (`unif_int(1, 6)`); "what if the die had 20 sides?" is the
  whole point of Monte Carlo, and knobs make every example a toy you can turn.
- The `log` stream (PLAN-FLINT) already interleaves text and charts in source order; the
  missing piece for a literate Preview is interleaving *the source itself*.

## Current state (verified 2026-07-10)

- **Output stream**: `Output { Text(String), Plot(Rc<Summary>) }` in
  `crates/noise-core/src/eval.rs:91`, buffered on `Engine.outputs` in exact source order,
  drained by `take_output()` (`eval.rs:126`). `Print` pushes `Output::Text`
  (`eval.rs:331`), plots push `Output::Plot` (`eval.rs:1468`).
- **WASM surface** (`crates/noise-wasm/src/lib.rs`): `run(src)` → `RunResult { ok, value,
  output, error, stats, log }`; `log: Vec<LogItem>` serde-tagged `{ "kind":
  "text"|"plot", … }` (`lib.rs:19`); `run_with_introspection(src, requests_json)` adds
  `bindings` + per-request `PlotOut`s against the retained scope.
- **Parser**: `parse(src) -> Program { stmts: Vec<Spanned> }` (`parser.rs:17`,
  `ast.rs:165`) — statements are spanned expressions, no separate `Stmt` type. Newlines
  double as separators via per-token line-break flags (`parser.rs:31`).
- **Lexer**: `//` and `#` comments are **discarded** — scanner skips to end of line, no
  token, no text (`lexer.rs:79-84`). Strings are double-quoted `Str` tokens, **no
  escapes, no interpolation** (LANG.md "Strings").
- **No frontmatter/knobs concept anywhere**; `---` at the top of a file lexes as three
  unary-minus tokens and fails to parse, and a backtick is a lex error — both syntaxes
  are free to claim. No valid program breaks.
- **Playground** (`packages/www/src/components/Playground.astro`): one Output panel
  (`#pg-output`), `run()` calls `runNoiseWithIntrospection(src, [])` and renders
  `res.log` as interleaved `.out-text` lines and plot cards.

## Design decisions

### D1 — Frontmatter: `---` fences at the very top, YAML subset *or* JSON

- Recognized **only at byte 0**: line 1 is exactly `---`, block runs to the next line
  that is exactly `---`. Anywhere else, `---` keeps meaning three minuses. Unterminated
  fence = spanned parse error.
- **Content auto-detect**: first non-whitespace char `{` → JSON (`serde_json`, already a
  dep); otherwise a **hand-rolled YAML subset** in a new `frontmatter.rs`. noise-core
  stays dependency-light and wasm-clean, so no `serde_yaml`; the subset parser lowers to
  `serde_json::Value`, then **one** `Value → Frontmatter` path serves both syntaxes.
- YAML subset (documented in LANG.md, errors beyond it point at the JSON escape hatch):
  `key: value` maps, two-space-indented nested maps (two levels is all the schema
  needs), inline `{ k: v, … }` maps, scalars (numbers, `true`/`false`, quoted +
  unquoted strings), `#` comments. No anchors, no multi-doc, no sequences in v1.
- **Lexer treats the block as trivia** (same as a comment): skip it in place so all
  spans/offsets keep pointing into the original source — error messages and the doc
  model (D4) need original coordinates. `frontmatter::parse(src) -> Result<Option<(Frontmatter, Span)>>`
  is a separate entry hosts can call without running.

### D2 — Knobs: typed, host-tunable globals injected before statement 1

```
Frontmatter { title: Option<String>, knobs: IndexMap<String, Knob>, extra: Map }
Knob { kind: Int | Float | Bool | Choice, min, max, step, default, options, label? }
```

- Each knob binds its (validated) value as a plain deterministic global before the first
  statement — a point mass, exactly like `dice_sides = 6`. Programs may shadow the name
  (normal rebind); knob names must be valid identifiers and not reserved words.
- Validation at parse time: `default` within `[min, max]`, `options` non-empty for
  `choice`, types coherent. Validation at run time: host-supplied overrides type-checked
  and clamped/snapped to `step` by the *engine* (one implementation, not per host).
- `extra`: unknown frontmatter keys are preserved as raw JSON and surfaced to hosts —
  forward-compatible (blurb, category, seed… can grow without engine releases).
- **Engine API**: `Engine::run_with_knobs(src, overrides: &[(String, KnobValue)])`;
  plain `run` = no overrides, defaults apply. **CLI**: `noise file.noise --knob
  dice_sides=20 --knob target_number=3`; `noise validate` also validates frontmatter.
  **WASM**: `run(src, opts_json?)` — opts carry knob overrides (and later sample
  budget/seed) — plus a new pure `meta(src) -> { ok, title, knobs, error? }` so hosts
  can build knob UIs without running the program.

### D3 — Template blocks: fenced, interpolated, emit without `Print`

- New syntax, two fence weights (backtick is currently a lex error → free syntax):
  - `` ` … ` `` — a **single backtick** delimits a plain multi-line template. No info
    string allowed; the body cannot contain a backtick.
  - ` ```syntax … ``` ` — the **triple fence** exists to carry a syntax tag (e.g.
    ` ```md `); the body may contain single backticks (handy for markdown code spans).
    A bare ` ``` ` with no tag is just the plain template again — prefer the single
    backtick for that.
- Body semantics are identical in both: raw multi-line text with `${expr}`
  interpolation; the shared leading indentation is stripped; the closing fence sits on
  its own line.
- **Lexing/parsing**: the lexer captures the raw body + span as one token; the parser
  splits `${…}` holes and sub-parses each hole by re-tokenizing that substring *with its
  original byte offset*, so errors inside `${}` point at the real location. AST:
  `Expr::Template(Vec<TemplatePart>)`, `TemplatePart::Lit(String) | Expr(Spanned)`.
- **Hole scanning**: a hole ends at its *matching* `}` — the scanner tracks brace depth
  (holes can contain `if c { a } else { b }`) and skips `}` inside string literals.
  Unterminated hole = spanned error. A backtick inside a hole (nested template) is an
  error in v1.
- **Semantics**: a template evaluates to a `string`; each hole renders via its display
  form — so an `Est` self-rounds to its standard error (`${p}` → `0.166`), exactly like
  `Print`. Holes are deterministic-only, same rule as string `+`.
- **At root statement position** (a top-level statement that is syntactically a
  template), it emits instead of becoming the program value: pushes `Output::Note` and
  yields `unit` — the `Print`-without-`Print`. Nested anywhere else — inside a
  function, as an argument, in an expression — it's just a string value (usable with
  `+`, as a `Print` arg, etc.); emitting from inside a function is `Print`'s job (D5).
- `Output::Note` stays engine-internal; in the `Document` (D5) a template statement
  surfaces as its own `note` block carrying the fence's syntax tag, distinct from
  `code` because hosts render it differently: the CLI prints the raw text either way;
  the Preview renders by tag — `md` as markdown prose, untagged as preformatted text.
  No `${{`-style escaping in v1 — documented limitation, consistent with strings having
  no escapes yet.

### D4 — Comments become data: trivia capture + attachment rules

- Lexer gains a side channel that does **not** disturb the token stream:
  `tokenize_with_trivia(src) -> (Vec<Token>, Vec<Trivia>)`, `Trivia { text, span, line,
  own_line: bool }`. Frontmatter and template bodies are not trivia. The existing
  `tokenize` stays byte-for-byte identical (hot path untouched).
- **Attachment rules** (pure functions of line numbers, applied in the doc model — the
  evaluator never sees comments). Each attached comment becomes a `(selfSpan,
  codeSpan)` pair in the document's comment layer; `codeSpan` is whatever statement
  range the rule yields — one line or many:
  - A contiguous run of own-line comments annotates the statements **from the next
    statement down to the next interruption** (another own-line comment run, a blank
    line, or the end of the group). So one run atop a five-statement group annotates
    the *whole group* (the example above: both `Dice ~ …` and `p = …`), while five
    runs interleaved with five statements each annotate exactly *their* line.
  - A blank line between the run and the code **detaches** it → it stays in the
    comment layer with **no `codeSpan`**: free-standing prose, positioned by its
    `selfSpan`, annotating nothing.
  - A trailing comment (code + comment on one line) annotates that statement only.
  - A comment run *between* two adjacent statements does **not** split the group —
    groups break only on blank lines and template statements.
  - A **statement group** = consecutive top-level statements with no blank line between
    → one `Code` block; it only aggregates, it owns nothing.
  - **Segmentation is span-based, not raw-line-based**: a statement may legally contain
    a blank line (newlines inside an unfinished expression are insignificant), and that
    interior blank line does **not** split the group — only blank lines *between*
    statement spans do. A comment on a continuation line is a trailing comment of the
    enclosing multi-line statement.

### D5 — One contract: a run returns a `Document` (breaking change, on purpose)

No back-compat. `RunResult { ok, value, output, error, stats, log }` and the parallel
text/log shapes are **deleted**; a run produces exactly one structure, assembled in
noise-core so the CLI and wasm can't drift:

```
Document {
  meta:     { title?, knobs: { name: Knob… }, extra },     // from frontmatter (D1/D2)
  blocks:   Block[],                                        // flat, in emission order
  comments: Comment[],                                      // annotation LAYER (D4) —
  result:   { value?, error?: { message, span }, stats },   //   not inside any block
}

Block =                       // ONE flat array — source segments and emissions alike
  | Code { source, span }                         // one statement group (D4), verbatim
  | Note { text, syntax?, stmt_span }             // emitted text: a rendered template
                                                  //   (D3) or a Print call — same thing;
                                                  //   syntax = fence tag ("md", …)
  | Plot { title, text, charts[], stmt_span }     // stmt_span → producing statement

Comment = { text, selfSpan, codeSpan? }  // selfSpan = where the comment text lives in
                                         // the source; codeSpan = the code it annotates
                                         // (one line, a whole group — any statement
                                         // range); ABSENT codeSpan = a detached run,
                                         // free-standing prose placed by selfSpan
```

(Serialized as JSON, kind-tagged: `{ "kind": "code" | "note" | "plot", … }` — the Plot
payload is PLAN-FLINT's, unchanged.)

- **Outputs are root blocks, emitted directly** — a plot isn't "returned by" code, it's
  pushed to the document exactly when the statement runs, as a sibling *after* its code
  block. A group that emits five plots yields five `Plot` blocks in a row. `stmt_span`
  links each emission back to its producing statement, so grouping outputs under code
  (or highlighting the line on hover) is a *frontend* choice, not the wire format.
- **One text-emission kind: `Note`.** `Print(a, b)` emits the same `Note` block a
  root template statement does (untagged, text = space-joined display forms), so the
  engine's internal `Output::Text`/`Output::Note` split collapses; nothing downstream
  distinguishes them. But the two spellings keep distinct jobs in the *language*: a
  template auto-emits **only as a root statement** (anywhere else it's a string value),
  while `Print` is the imperative emitter that works at any depth — inside a function
  body, a loop, a branch. `Print(` ``` … ``` `)` therefore composes naturally: the
  template renders to its text, `Print` emits it as a Note from wherever it runs.
  `stmt_span` on such a Note is the *top-level* statement that led to the emission
  (the outermost call), so it still lands after the right code block. (One v1
  limitation: a template *value* is just a string, so a triple-fence syntax tag doesn't
  survive the trip through `Print` — Print-emitted Notes are always untagged.)
- **Loops and repeated calls just repeat blocks.** The document is an emission *log*,
  and `stmt_span` is many-to-one by design: a root `for` that prints five times yields
  five `Note` blocks with the same `stmt_span`, in iteration order, all placed after
  the loop's code block; a plotting function called from three root statements
  attributes each emission to its **call site's** root statement. No reconciliation
  step exists — order in `blocks` *is* the reconciliation.
- **Emission cap** (the one real hazard of the log model): a run may emit at most
  `MAX_EMISSIONS` text/plot blocks (start ~200, same spirit as the 30-bin/800-point
  plot caps). Past the cap the engine stops recording, keeps running, and sets
  `result.truncated = { dropped, first_dropped_stmt_span }` so hosts render an
  explicit "…N more emissions not shown" card instead of silently swallowing output —
  no silent truncation, and no 10⁴-block payload melting the Preview.
- **All comments live in the layer, including detached ones.** Every comment is
  `{ text, selfSpan, codeSpan? }`: with a `codeSpan` it annotates that code (one line
  or a whole group, per D4); without one it's a detached run — free-standing prose
  whose document position is just its `selfSpan` (blocks carry spans too, so a reading
  view interleaves by source order). No separate block kind needed. `Code.source` is
  the verbatim slice of the group's span; renderers use `selfSpan` to hide, dim, or
  hover-reveal comment text, and trailing-vs-leading placement is derivable from the
  two spans.
- **Every view is a pure filter over one array**: *only plots* =
  `blocks.filter(b => b.kind === "plot")`; *hide code* = drop `code` blocks; *only
  text* = keep `note`; *full literate Preview* = render everything, overlay
  `comments`. No re-run, no second endpoint per view.
- Mechanics: new `doc.rs` — `parse_doc(src) -> { frontmatter, blocks }` (pure
  segmentation per D4, also the seed of a future formatter/doc generator); the engine's
  internal output entries gain the span of the top-level statement that produced them
  (the run loop already iterates `Program.stmts: Vec<Spanned>`); `assemble(doc, outputs,
  result) -> Document` interleaves emissions after the code block whose span contains
  their `stmt_span`.
- A template statement is its own `note` block (it *is* source, but renders as prose);
  it splits the surrounding statement group, which matches how the file reads.
- **`result.value` serialization**: a `Value` crosses the boundary as
  `{ kind: "num" | "est" | "dist" | "array" | …, text }` where `text` is the display
  form (the same string the CLI prints today, incl. `Est` self-rounding) — hosts render
  text, `kind` exists for styling. No structural serialization of arrays/dists in v1;
  introspection is the tool for looking inside a value. Absent when the program ends in
  `unit` (e.g. a `Print`/plot), same rule as the CLI's no-echo today.
- **Errors don't lose the document**: a *runtime* failure still returns all blocks,
  with `result.error` spanned — the Preview renders the doc up to (and pointing at) the
  failing block instead of a bare error string. A *parse/lex* failure returns a
  best-effort `Document`: `meta` if the frontmatter parsed, empty `blocks`/`comments`,
  spanned `result.error` — hosts always receive the same shape, never a second error
  channel.
- **WASM**: `run(src, opts_json?) -> Document` (opts = knob overrides, sample budget…);
  `meta(src)` stays for building knob UIs without running; `run_with_introspection`
  survives as the hover sidecar but returns its per-request `PlotOut`s next to a
  `Document`. `packages/core` exports the `Document`/`Block` types as the API.
- **CLI**: `print_output` becomes a `Document` renderer — notes as text,
  plots as the one-line text card (later kitty graphics, per PLAN-FLINT FL4/FL5). The
  REPL renders single-statement documents the same way.

## Phases

### LT1 — Frontmatter + knobs (the core) ✅ shipped

- `frontmatter.rs` (YAML-subset → `serde_json::Value` → `Frontmatter`, JSON path,
  validation), lexer skips the fence as trivia, `Engine::run_with_knobs`, knob injection
  + override clamping.
- CLI `--knob k=v` (repeatable) + frontmatter errors in `validate`.
- WASM: optional `opts_json` on `run`/`run_with_introspection` (knob overrides), new
  `meta(src)` returning `{ ok, title, knobs, extra, error? }`; `packages/core` types
  (`Frontmatter`, `Knob`).
- Convert `dice.noise` (+ 2–3 more) as proof; `examples.ts` prefers the frontmatter
  `title` when present, keeps its curated fields otherwise.
- Tests: fence edge cases (not-at-top, unterminated, `---` mid-file still minus), both
  syntaxes, override clamping, span correctness of errors after the fence.

### LT2 — Template blocks ✅ shipped

- Lexer token for fenced bodies, `Expr::Template`, offset-correct hole sub-parsing,
  statement-position emit → `Output::Note`; CLI prints the rendered text.
- LANG.md section + one example converted to use a template instead of `Print` chains.

### LT3 — The `Document` contract (the breaking change) ✅ shipped

- `tokenize_with_trivia`, attachment rules, `doc.rs` (`parse_doc` + `assemble`),
  span-tagged engine outputs.
- **Delete** `RunResult`/`IntrospectResult`'s `output`/`log` split in noise-wasm; `run`
  returns a `Document`, `run_with_introspection` returns `Document` + sidecar
  `PlotOut`s. `packages/core` re-exports the new types; version bump, no compat shims.
- CLI `print_output` → `Document` renderer (file runner + REPL); playground
  `renderResult` rewritten over `blocks` (still the flat "everything" view at this
  phase — the point is the site keeps working on the new contract before Preview
  exists).
- Golden tests: fixture `.noise` files → expected `Document` JSON (comment-layer
  `selfSpan`/`codeSpan` pairs incl. run-annotates-whole-group vs interleaved-per-line +
  detach-on-blank-line + mid-group comments don't split + note-splits-group).
- **Snapshot tests for indirect emission** — the cases where attribution can silently
  go wrong. Each fixture snapshots the full `Document` JSON:
  - `Print` inside a function called from a root statement → Note lands after the
    *call site's* code block with the caller's `stmt_span`, not the definition's.
  - `plot::*` inside a function, same assertion; a function that prints *and* plots →
    both emissions in call order under one `stmt_span`.
  - Function called from **two different** root statements → emissions split across
    the two call sites, none attributed to the `f(x) = …` definition block.
  - Root `for` loop calling an emitting function → N Notes share one `stmt_span` in
    iteration order; past `MAX_EMISSIONS` the run keeps going and `result.truncated`
    is set.
  - Nested calls (root → f → g → `Print`) → still the root statement's span.

### LT4 — Preview tab in the playground ✅ shipped (hover-reveal deferred to LT5)

- Output panel grows tabs that are all **filters over the same `Document`** — no
  re-run when switching: **Preview** (default, full literate render), **Output**
  (outputs only: notes/plots, today's view), and cheap extras if wanted (plots
  only). Preview: knob sliders/inputs from `meta` at the top (debounced re-run with
  overrides on change), code blocks via the existing `lib/highlight.ts`, notes as
  markdown, note blocks + Flint chart cards following their code block (grouped
  client-side via `stmt_span`), the comment layer hidden behind a hover affordance
  (gutter dot spanning each comment's `codeSpan` lines; hovering an emission highlights
  its producing line).
- Share links carry knob state: non-default overrides join the existing hash scheme
  (`#x=<id>&k=name:value,…` / alongside `#c=<encoded>`), so a tuned example is
  shareable exactly as tuned.

**LT4 note (shipped):** the whole playground was reworked into a **resizable split** (draggable
divider, keyboard-nudgeable) with the example title/explanation moved to a full-width card *on
top* and a **full-height** output. The extra tabs were dropped — the **Preview is the only view**,
rendered as a **typeset LaTeX-style paper** (the site's Computer Modern `--serif`, warm-paper
page): emitted notes as body prose (markdown for a `md` fence), plots as **numbered figures**
(`Figure N.` + a stats caption, reusing the global `.figure` styles), and code as clean listings
with their **comments stripped from the page and revealed on hover as right-margin notes** (LaTeX
`\marginpar`: an absolutely-positioned floating card in a reserved right gutter on a wide panel,
dropping just below the listing on a narrow one via a container query — never shifts the layout). A
faint "✎ N notes" tag + maroon spine marks a listing with hidden notes. Knob sliders/number-inputs
are generated from `meta()` (kept in sync as you edit) and re-run debounced on change; share links
carry non-default knobs (`&k=name:value`); the cap surfaces as a "…N more not shown" line.
**Implementation gotcha:** the Preview DOM is built in JS, so Astro's *scoped* `<style>` never
reaches it — all Preview styling lives in a `<style is:global>` block scoped by the `#pg-output`
id. **Deferred to LT5:** hover-*introspection* (the variable popover) and hover-highlighting a
producing line. Verified live in-browser (knob re-run, resizer, margin-note reveal on wide + narrow
panels, share hash, no console errors).

### LT5 — Hover introspection + metadata migration

- Hover a bound variable in a Preview code block → popover with
  `run_with_introspection` describe/hist for that name (bindings list already ships;
  the retained-scope sidecar already works — see the variable-introspection notes).
- Migrate remaining examples to frontmatter titles (+ blurb via `extra`), shrinking
  `examples.ts`'s curated `meta` to what genuinely can't live in the file.
- Docs: LANG.md "Literate files" section; examples/README.md update.

## Open questions

- **Choice knobs** (`type: "choice"`, `options: [...]`): strings as values need string
  equality in programs (exists) — include in LT1 or defer? Leaning: defer to LT5,
  int/float/bool cover every current example.
- Should detached comment runs (`codeSpan`-less comments) render in Preview as
  free-standing prose or stay invisible like today? Leaning prose — it's what makes
  the file read as a document.
- `seed`/`samples` as reserved frontmatter keys (run reproducibility per file)? Cheap
  once `extra` exists; decide when a demo needs it.
- Does the Gallery (`Gallery.astro`) also switch to frontmatter-driven cards? Natural
  follow-up to LT5, not blocking.
