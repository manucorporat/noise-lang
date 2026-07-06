# The Noise Programming Language

Noise is an expression-based, probabilistic language: variables don't hold exact values, they
hold *random variables* (probability distributions). Operators lift over random variables, and
`P(condition)` estimates a probability by simulation — so propagating uncertainty and running
Monte Carlo experiments reads like ordinary math.

**Scope (be honest about it):** Noise today is a **static random-variable algebra + forward
Monte Carlo** tool — excellent for things like estimating π, summing risks, or propagating
uncertainty through a formula. It also does **conditioning** (`P(A | C)`, Bayes scoped to a query)
and **hierarchical models** (a random parameter, `p ~ unif(0,1); k ~ bernoulli(p)`), which together
give *rejection-based* Bayesian inference — priors, posteriors, and predictives you can write and
query (see `examples/beta_bernoulli.noise`). What's still out of scope: inference that scales to lots
of continuous data (importance/MCMC weighting) and *dynamic* stochastic systems (queues, Markov
chains, random walks), which need sequential/stateful sampling — deliberate future tracks, not
current capabilities. See `GOAL.md`, `plans/PLAN.md`, and `AGENT.md` for the precise state and roadmap.

> **The one rule that surprises everyone:** a name bound with `~` is *one fixed draw* that every
> mention reuses. So `X - X` is exactly `0`, and `Dice + Dice` is `2·Dice` — **not** two dice.
> Independent draws come only from separate `~` bindings (or shaped draws / function calls). See
> "Random variables and sharing" in [`LANG.md`](LANG.md).


## Try it

- **Playground** — [noise-lang.dev/play](https://noise-lang.dev/play) runs the real engine in
  your browser (compiled to WebAssembly). Pick an example or share a program via link.
- **CLI** — install from crates.io, then run a file or open the REPL:

  ```sh
  cargo install noise-cli

  noise examples/pi.noise   # run a program; prints the LAST statement's value
  noise                     # REPL — one line at a time, persistent environment
  ```

  From a clone of this repo, `cargo run -p noise-cli -- examples/pi.noise` works without
  installing anything.


## The language in five minutes

A program is a sequence of `;`-separated statements, and its result is the value of the **last**
statement. Four rules carry the whole mental model:

1. **Everything is a distribution.** A number is just the degenerate case — a *point mass*
   (`mean(5) = 5`, `variance(5) = 0`, `P(5 > 3) = 1`). Operators map distributions to
   distributions uniformly, so the same `+`, `*`, `==`, `if` work on constants and random
   variables alike.

2. **`~` draws; `=` transforms.** `name ~ dist` is a **stochastic node** — it draws a fresh
   random variable, and `~` is the *only* thing that draws. `name = expr` is a **deterministic
   node** — a transform, a constant, or an undrawn *recipe*:

   ```noise
   Die = rand::unif_int(1, 6);   # a recipe — nothing drawn yet
   a ~ Die;                      # draw it
   b = a + 1;                    # transform the draw
   ```

3. **One name = one fixed draw.** Every mention of a name is the *same* draw, exactly like `X`
   in math. `X + X` is `2X`; there is no re-draw on reuse.

4. **Independence is explicit.** Two independent draws come from two `~` declarations, or from
   the shaped draw `~[n] dist` — never from repeating a name:

   ```noise
   A ~ unif_int(1, 6); B ~ unif_int(1, 6)   # two independent dice
   dice ~[2] unif_int(1, 6)                 # same thing, as a length-2 array
   ```

Nothing is sampled until a **query** forces it — `P(event)`, `E(x)`, `Var(x)`, `Q(x, q)` — and
queries return *estimates* that carry a standard error. Everything upstream stays symbolic.

Beyond the core:

- **Modules.** `builtin` is always active (`P`, `Q`, `E`, `Var`, `Print`, `Len`); everything else
  is strict — `use rand;` for distributions (`unif`, `unif_int`, `bernoulli`, `normal`,
  `exponential`, `poisson`, `geometric`, `categorical`, `rotation`, `permutation`, …), `use math;`
  for `sqrt`/`exp`/`log`/trig and complex helpers, `use vec;` for array & linear-algebra helpers
  (`sum`, `mean`, `dot`, `transpose`, `quantize`, …), `use signal;` for lazy waveforms. Or reach
  one item by path with no `use`: `math::sqrt(2)`.
- **Arrays & matrices.** `xs ~[n] dist` draws `n` independent values as an array, `M ~[n, m] dist`
  a matrix; `@` is the matrix product (`*` stays elementwise), indexing is `M[i][j]`, and
  `for i in 0..n { … }` loops over half-open integer ranges.
- **Functions.** `f(args) = body` is a pure transform; `f() ~ dist` draws **fresh on every call**
  — so `roll() + roll()` really is two independent dice (`examples/functions.noise`).
- **Conditioning.** `P(hit | fired)` — rejection-based, scoped to the query.
- **Hierarchical models.** A parameter can itself be random: `p ~ unif(0, 1); k ~ bernoulli(p)`
  gives priors, posteriors, and predictives (`examples/beta_bernoulli.noise`).
- **Complex numbers** are first-class scalars (`math::i`), enough for phasors, FFT-style demos,
  and Shor-period toys (`examples/am_vs_fm.noise`, `examples/shor_period.noise`).
- **Plots & introspection.** `plot::histogram(x)`, `plot::scatter(x, y)`, `plot::value(p)` render
  in the CLI and the playground; `describe`/`explain`/`corr` interrogate a variable's
  distribution and what drives it.

The full specification is [`LANG.md`](LANG.md). The compact "how to write correct Noise" guide is
the agent skill, [`.claude/skills/noise-lang/SKILL.md`](.claude/skills/noise-lang/SKILL.md),
rendered for humans at [noise-lang.dev/skill](https://noise-lang.dev/skill).

### Two examples

Estimate π — points fall uniformly in the 2×2 square; the fraction inside the unit circle is
`π/4` ([`examples/pi.noise`](examples/pi.noise)):

```noise
X ~ rand::unif(-1, 1);
Y ~ rand::unif(-1, 1);
pi = 4 * P(X^2 + Y^2 < 1);
Print("Estimated pi ~", pi)
```

The birthday paradox — give 23 people a random birthday each and ask how often two collide
([`examples/birthday.noise`](examples/birthday.noise)):

```noise
use rand;   # unif_int

n     = 23;
bday  = unif_int(1, 365);
days  ~[n] bday;             # n independent draws, as an array
match = vec::has_duplicates(days);
Print("P(shared birthday among", n, ") =", P(match))   # ≈ 0.507
```

The [`examples/`](examples/) folder has ~30 more self-contained, commented programs — Monty Hall,
the 100-prisoners problem, Buffon's needle, the St. Petersburg paradox, signal dithering,
TurboQuant quantization, and friends.


## Use it from JavaScript / TypeScript

The engine ships on npm as [`@noiselang/core`](packages/core) — the Rust core compiled to
WebAssembly behind a small typed API. The `.wasm` binary lives **inside** the package and is
emitted as an asset of *your* build by any bundler that understands
`new URL(..., import.meta.url)` (Vite, Rollup, webpack 5, esbuild) — no copying files into
`public/`, no CDN, no runtime configuration. It's the exact engine behind the
[playground](https://noise-lang.dev/play).

```sh
npm add @noiselang/core   # or: pnpm add / yarn add
```

```ts
import { run } from '@noiselang/core';

const result = await run(`
  X ~ rand::unif(-1, 1);
  Y ~ rand::unif(-1, 1);
  4 * P(X^2 + Y^2 < 1)
`);

result.value;   // "3.1415…" — the last statement's value
result.output;  // everything Print(...) emitted
result.stats;   // { forcings, samples, ops, rng_draws }
```

`run` never throws — parse/eval failures come back on `result.error` with a source span. For
building an inspector UI there is `runWithIntrospection(src, requests)`, which runs a program and
then interrogates its retained scope — describe a variable's distribution, correlate two, or
explain what drives one, without editing the source. It's what powers the playground's variable
inspector. Full API, types, and bundler notes: [`packages/core/README.md`](packages/core/README.md).


## Use it in your AI agent

This repo ships a **skill** that teaches a coding agent to write correct, idiomatic Noise — the
mental model, the module/builtin surface, the idioms, and the hazards
([`.claude/skills/noise-lang/SKILL.md`](.claude/skills/noise-lang/SKILL.md), also rendered as
human docs at [noise-lang.dev/skill](https://noise-lang.dev/skill)).

Install it into any agent with [`vercel-labs/skills`](https://github.com/vercel-labs/skills) — no
install needed, `npx` runs it:

```sh
# auto-detect the agents on this machine and install the skill (symlinked to one canonical copy)
npx skills add manucorporat/noise-lang

# or target specific agents (claude-code, cursor, codex, github-copilot, windsurf, opencode, …)
npx skills add manucorporat/noise-lang -a claude-code -a cursor -a codex

# install globally (~/) instead of into the current project, or make independent copies
npx skills add manucorporat/noise-lang -g --copy
```

You can also install straight from the skill's **URL** (no clone, any git host):

```sh
npx skills add https://github.com/manucorporat/noise-lang/tree/master/.claude/skills/noise-lang
```

It installs into each agent's conventional location — `.claude/skills/` for Claude Code,
`.agents/skills/` for Cursor / Codex / Copilot, `.windsurf/skills/` for Windsurf, etc. — so the
agent picks it up automatically. (`npx skills list` to see what's installed, `npx skills remove
noise-lang` to undo.)


## Repository layout

| Path | What it is |
| --- | --- |
| [`crates/noise-core`](crates/noise-core) | The language: parser, random-variable graph, samplers, JIT. |
| [`crates/noise-cli`](crates/noise-cli) | The `noise` binary — file runner + REPL (`cargo install noise-cli`). |
| [`crates/noise-wasm`](crates/noise-wasm) | WebAssembly bindings for the engine. |
| [`packages/core`](packages/core) | [`@noiselang/core`](packages/core) — the npm package wrapping the WASM engine. |
| [`examples/`](examples) | Self-contained, commented `.noise` programs. |
| [`packages/www/`](packages/www) | [noise-lang.dev](https://noise-lang.dev) — site, playground. |
| [`LANG.md`](LANG.md) | The language specification. |
