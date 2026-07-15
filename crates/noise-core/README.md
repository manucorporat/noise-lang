# noise-core

The core library of the [Noise](https://github.com/manucorporat/noise-lang) probabilistic
language: lexer, parser, evaluator, and Monte-Carlo sampler.

In Noise, variables don't hold exact values — they hold *random variables* (probability
distributions). Operators lift over random variables, and `P(condition)` estimates a probability
by simulation. This crate implements the language runtime that powers the `noise` CLI.

## Usage

```toml
[dependencies]
noise-core = "0.1"
```

To run Noise programs from the command line instead, install the CLI:

```sh
cargo install noise-cli   # provides the `noise` binary
```

## Features

- `gpu` — enable the native WebGPU backend (off by default; native only). Emits a WGSL kernel and
  dispatches whole lane ranges to the GPU, falling back to the columnar interpreter for any graph it
  can't lower or any forcing that wouldn't pay for itself, so results are unchanged.

The columnar interpreter is always available as the portable floor; in the browser the engine uses
the WASM emitter instead.

## License

MIT
