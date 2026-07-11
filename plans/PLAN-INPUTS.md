# PLAN-INPUTS — inputs as a language construct, declared inline (replacing frontmatter knobs)

**Date:** 2026-07-11 · **Status:** ✅ Implemented (all phases IN1–IN4). Supersedes the **knobs** half
of PLAN-LITERATE (LT1 §D2). Frontmatter stays for pure metadata (`title`/`abstract`/`tags`/`extra`);
the tunable-parameter job moved out of the frontmatter and into the program body. Knobs are fully
removed (no legacy aliases) per the resolved open questions: `name: value` delimiter, LHS-name
inference shipped, `input::real`, control rendered as its own block after the code group, non-taken
branches accepted + documented.

## The decision

Today a `.noise` file declares its host-tunable parameters in a `knobs:` map inside the frontmatter:

```
---
title: "Roll a die"
knobs:
  dice_sides:    { type: int, min: 1, max: 100, step: 1, default: 6 }
  target_number: { type: int, min: 1, max: 100, step: 1, default: 4 }
---
Dice ~ rand::unif_int(1, dice_sides);
p = P(Dice == target_number);
```

This is the one place where frontmatter stops being metadata and starts declaring **runtime inputs**
— values that flow into the program. That's inconsistent (everything else in frontmatter is inert
description), it forces every input to the *top* of the file regardless of where it's used, and it
means each new input *type* is an out-of-band extension of a YAML schema rather than of the language.

**Replace it with a first-class expression.** An input is a value the host may tune, declared where
it is used — passed with **named arguments** (a general call feature, §2), the input's `name`
inferred from the binding on the left:

```
dice_sides    = input::real(min: 1, max: 100, step: 1, default: 6);
target_number = input::real(min: 1, max: 100, step: 1, default: 4);

Dice ~ rand::unif_int(1, dice_sides);
p = P(Dice == target_number);
```

`input::real(...)` evaluates to the input's **current value** — a deterministic point mass, exactly
what a knob binds today — so downstream code reads it like any number. The host renders a control
**inline in the document, at the point of declaration** (a slider that appears after the code block
where the input lives), not a bar of every input stacked at the top. `input::real` is the first of a
family: `input::int`, `input::bool`, later `input::choice`, `input::color`, … each a new *builtin*,
not a new frontmatter schema.

Net effect: frontmatter returns to being purely descriptive; inputs become ordinary, composable,
extensible language constructs that live next to the code they parameterize.

---

## Current architecture (what we're moving)

Grounded in the code as of this plan:

- **Declaration.** `crates/noise-core/src/frontmatter.rs` — `Frontmatter { knobs: Vec<Knob> }`,
  `Knob { name, kind: KnobKind {Int,Float,Bool}, min, max, step, default: KnobValue, label }`.
  `Knob::resolve(override) -> KnobValue` type-checks, clamps to `[min,max]`, snaps to `step`.
- **Injection.** `eval.rs::inject_knobs_fm` (called before statement 1 by `run_with_knobs` /
  `run_to_document`): every override must name a declared knob (else spanned error); each knob binds
  as a point-mass global `self.vars.insert(name, Value::Num|Bool)`. A program may shadow the name.
- **Wire / hosts.**
  - `crates/noise-wasm/src/lib.rs` — `RunOpts { knobs: Map<String,Value> }`; `meta(src)` parses
    frontmatter *without running* so a host can build the knob UI first.
  - `packages/core/src/index.ts` — `Knob`, `KnobKind`, `KnobValue`, `KnobOverrides`, `RunOptions
    { knobs }`, `NoiseMeta { knobs: Knob[] }`, `meta()`.
  - `packages/www/src/components/Playground.astro` — `refreshKnobs()` calls `noiseMeta(code)`,
    renders `#pg-knobs` (a **top bar** of sliders), keeps `knobValues`, re-runs (debounced) on change.
  - `packages/www/src/lib/share.ts` — `encodeKnobs`/`parseKnobs`, the `&k=name:value` share fragment.
  - `crates/noise-cli/src/main.rs` — `--knob name=value` (repeatable).
- **Namespaced calls.** `eval.rs` `Expr::Call(name, args)` → `split_path(name) -> (module, base)`.
  `plot::*` and `stats::*` are intercepted before generic module resolution: `plot::*` **emits** an
  `Output::Plot` (captured like `Print`) and returns `Unit`; `stats::*` returns a number. `MODULES`
  lists the eight namespaces. This is the exact hook `input::` will use.
- **Document model.** `crates/noise-core/src/doc.rs` — a run returns one `Document { meta, blocks:
  Vec<Block>, comments, result }`. `Block::{Code, Note, Plot}` carries a `stmt_span`; `assemble`
  interleaves emissions after the code group whose span contains their `stmt_span`. This is how a new
  inline **input control** block will thread into the page.
- **Examples with knobs (migration set):** `examples/dice.noise`, `examples/coin_streak.noise`,
  `examples/insurance.noise`.

---

## Design

### 1. The `input::<type>(spec)` primitive

An `input::` call is a namespaced builtin intercepted in `Expr::Call` (like `plot::`/`stats::`). It:

1. **Reads** the current value = `resolve(override_by_name ?? default)` — the same clamp/snap logic
   `Knob::resolve` has today (lift it out of `Knob` into a shared `resolve_input` so hosts and the
   engine agree). Returns `Value::Num` (`real`/`int`) or `Value::Bool` (`bool`).
2. **Registers** the input in the run's **input manifest** (see §3), keyed by `name`.
3. **Emits** an inline control (see §4), stamped with the current `stmt_span`, so the host can render
   a slider where the input was declared.

First evaluation of a given `name` registers + emits; later evaluations of the same `name` return the
same resolved value and do **not** re-emit (dedup by name). A second declaration of the same `name`
with a *different* spec is a spanned error ("input `x` redeclared with a different spec").

Types for the first cut, mirroring today's `KnobKind`:

| call           | value | spec fields                                   |
|----------------|-------|-----------------------------------------------|
| `input::real`  | Num   | `name`, `min?`, `max?`, `step?`, `default`, `label?` |
| `input::int`   | Num   | same; value rounded to an integer             |
| `input::bool`  | Bool  | `name`, `default`, `label?`                    |

(`real` reads better than `float` and matches the math surface; `int`/`bool` keep parity with knobs.
`choice`/`color`/… are later builtins, out of scope here.)

Headless (CLI, tests, landing-page demos): with no override, `input::real` returns `snap(default)`
deterministically — programs run unchanged with no host UI.

### 2. Named arguments — a general call feature

Rather than a record type, add **named arguments** to the call grammar. This is not `input::`-specific
— it works for **every** function (user functions and builtins alike), and `input::real(...)` is then
just an ordinary call.

**The rule:** a call's arguments are **either all positional or all named — never mixed.**

```
f(x, y)                       # positional, in parameter order
f(a: x, b: y)                 # named, any order
f(x, b: y)                    # ERROR: mixed positional + named
input::real(min: 1, max: 100, step: 1, default: 6)
```

- **Grammar.** Keep the hot path unchanged by making the two forms disjoint at the AST level:
  `Expr::Call(name, CallArgs)` with `CallArgs = Positional(Vec<Spanned>) | Named(Vec<(String,
  Spanned)>)`. Existing positional calls stay `Positional` verbatim — zero churn for the many call
  sites. Named uses `name: expr` pairs (a `Colon` token; `crates/noise-core/src/lexer.rs` already has
  `::` for paths — confirm a lone `:` lexes, add the token if not; `:` is otherwise unused in
  expression position — ranges are `..`).
- **Parse.** After `(`, look ahead: if the first argument is `IDENT :`, the whole list is **named**
  (every entry must be `ident: expr`; a bare positional entry among them is a spanned "can't mix
  positional and named arguments" error). Otherwise it's **positional** as today. A duplicate name is
  a spanned error.
- **Resolution (eval).** Named args bind to parameters **by name**:
  - *User functions* — `f(a, b) = …` knows its parameter names; a named call maps `a:`/`b:` onto them,
    fills every parameter exactly once, errors on an unknown name or a missing parameter.
  - *Builtins* (`input::real`, and any builtin that opts in) — each declares its accepted parameter
    names + which are required; resolution is the same. Builtins that don't opt in accept positional
    only (unchanged).
  This keeps positional calls on the exact current code path; named calls are a thin
  name→slot mapping layered before argument evaluation.

Why this over a record literal: it's one feature that pays off everywhere (readable calls to any
multi-arg function), needs no new *value* kind, and avoids the `{…}` block-vs-record ambiguity
entirely. `input::` gains nothing special — it's a builtin with named parameters like any other.

**Ergonomic enhancement (recommended):** make an input's `name` **optional** when `input::…` is the
*direct RHS of a bind* — infer it from the LHS identifier, so `dice_sides = input::real(min: 1, max:
100, default: 6)` names the input `"dice_sides"`. Removes the most repetitive part of the example.
Requires the binder to pass the target name into evaluation of its RHS (small, contained). A
standalone `input::real(...)` with no `name:` and no binding LHS is a spanned error.

### 3. Discovery: the input manifest comes from the run, not from `meta()`

Knobs are discovered by `meta()` **without running**, because the UI had to exist before the first
run. That constraint is gone: the playground now **auto-runs on load and on demo switch** (this
session). So inputs are discovered **dynamically, from the run** — the engine accumulates an input
manifest as `input::` calls evaluate and returns it on the `Document`:

```
Document.result.inputs: [ { name, type, min?, max?, step?, default, label?, value, stmt_span } ]
```

- **Pro:** no separate static analyzer; specs can be computed; positions fall out of emission order
  exactly like notes/plots. One code path (the evaluator) is the source of truth.
- **Con:** an input inside a not-taken branch won't appear until that branch runs. Acceptable — the
  same is already true of a `plot::` inside a branch. Document it.
- `meta()` **drops `knobs` entirely** and keeps returning `title`/`abstract`/`tags`/`extra` for the
  pre-run paper header. Inputs are no longer a pre-run concept.

(If a pre-run input UI is ever needed, add a static AST scan for `input::*` calls with literal specs
as a *pure optimization* that must agree with the run manifest — not part of this plan.)

### 4. Inline rendering

Add `Block::Input { name, spec, value, stmt_span }` (wire kind `"input"`). An `input::` call emits an
`Output::Input`, and `doc.rs::assemble` interleaves it after the code group containing its
`stmt_span` — so the control renders **right after the code block that declares it**, satisfying
"inline in the document, not all at the top". In the playground:

- Delete the `#pg-knobs` top bar and `refreshKnobs`'s knob half.
- `renderPreview` renders each `Block::Input` as a labelled slider/checkbox (reuse `.pg-knob*`
  styles, restyled as an inline control card). Changing it updates `inputValues[name]` and re-runs
  (debounced), exactly like the knob bar does now — but the control sits in the flow of the paper.
- The control can render as a small framed widget between code and its downstream figures; the
  existing plot-fold / margin-note machinery is untouched.

### 5. Overrides, wire, and hosts (rename knob → input)

The override shape is unchanged (name → value); only names change:

- **Engine:** `run_with_knobs(overrides)` → `run_with_inputs(overrides: &[(String, InputValue)])`;
  overrides feed `input::` resolution by name instead of `inject_knobs`. "override for an input the
  program never declares" is a spanned error, reported after the run (the manifest is known only then).
- **wasm:** `RunOpts { knobs }` → `{ inputs }`; drop `meta().knobs`.
- **`packages/core`:** `Knob*` types → `Input*` (`InputSpec`, `InputValue`, `InputOverrides`);
  `RunOptions { inputs }`; `Document.result.inputs`; remove `NoiseMeta.knobs`. Bump minor.
- **Playground:** `knobValues` → `inputValues`; render from `Document.result.inputs`; share/`&k=`
  fragment → `&i=` (keep reading legacy `&k=` for old links, mapping to inputs by name).
- **CLI:** `--knob name=value` → `--input name=value` (keep `--knob` as a hidden alias for one
  release).

### 6. Frontmatter cleanup

- Remove `Knob`, `KnobKind`, `KnobValue`, `knobs` from `Frontmatter`. A `knobs:` key becomes an
  **unknown top-level key** (already an error per the strict schema) — the error message gains a hint:
  "`knobs:` is no longer frontmatter; declare inputs with `input::real(...)` in the body."
- Keep the shared clamp/snap resolver (moved out of `Knob`) as `input`'s validator.

---

## Phases

**IN1 — named arguments (general call feature).** `CallArgs = Positional | Named` in the AST; lexer
`Colon` token if missing; parser lookahead (`IDENT :` ⇒ named) with the "no mixing" + "no duplicate
name" errors; eval name→slot resolution for **user functions** (unknown-name / missing-param errors).
Positional calls unchanged. Tests: `f(a,b)=…; f(b: 2, a: 1)` binds correctly; `f(1, b: 2)` errors;
`f(z: 1)` (unknown param) errors; duplicate name errors. *No `input::` yet.*

**IN2 — the `input::` primitive (engine).** Extend named-arg resolution (IN1) to **builtins** —
`input::{real,int,bool}` each declare their accepted named params (`name?`, `min?`, `max?`, `step?`,
`default`, `label?`); intercept them in `Expr::Call` like `plot::`/`stats::`; lift clamp/snap out of
`Knob` into `resolve_input`; build the run **input manifest** with name dedup + redeclaration error;
`run_with_inputs(overrides)` plumbing + post-run "unknown input override" error; LHS-name inference
(§2). Tests: default resolution, override clamp/snap, bool, dedup, redeclare-conflict, name
inference, headless determinism.

**IN3 — Document + inline rendering.** `Output::Input` + `Block::Input` + `assemble` interleaving +
wire JSON; `Document.result.inputs`. Playground: drop `#pg-knobs`/`refreshKnobs` knob half, render
inline input controls from the run, re-run on change; move overrides to `inputValues`; share `&i=`
(+ legacy `&k=` read). Verify a two-input demo renders both controls inline and re-runs on drag.

**IN4 — remove frontmatter knobs + migrate.** Delete `Knob*` from `frontmatter.rs` (+ `knobs:` hint
error); rename `Knob*`→`Input*` across `packages/core`, `lib/noise.ts`; CLI `--input` (+ `--knob`
alias); update `LANG.md` (move the knob section to an "Inputs" section under Builtins/Templates) and
`packages/www/src/data/examples.ts` comment. Convert `dice.noise`, `coin_streak.noise`,
`insurance.noise` to `input::`. Full check: `cargo test`, all examples `validate` + run, `astro
build`, browser pass on the three migrated demos.

---

## Open questions

1. **Named-arg delimiter (§2):** `name: value` (recommended — reads like the frontmatter/spec it
   replaces) vs `name = value` (but `=` is already bind, so `:` is cleaner and unambiguous).
2. **`name` inference (§2):** ship LHS-name inference in IN2, or require explicit `name:` first and add
   inference later? Recommend shipping it — it's the biggest ergonomic win over knobs.
3. **`real` vs `float`:** name the numeric input `input::real` (recommended) or keep `float` for
   parity with the old `KnobKind`?
4. **Control placement (§4):** render the input control as its own block *after* the code group
   (simple, recommended), or as a margin widget on the exact declaration line (prettier, more work)?
5. **Non-taken branches (§3):** accept that an input in an unexecuted branch is invisible until run,
   or add the optional static scan? Recommend accept + document.
