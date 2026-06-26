# noise-cli

The command-line interface for the [Noise](https://github.com/manucorporat/noise-lang)
probabilistic language — a REPL and a file runner. Installs a binary named `noise`.

In Noise, variables don't hold exact values — they hold *random variables* (probability
distributions). Operators lift over random variables, and `P(condition)` estimates a probability
by simulation, so propagating uncertainty and running Monte-Carlo experiments is built into the
language.

## Install

```sh
cargo install noise-cli
```

## Usage

```sh
noise              # start the REPL
noise program.noise   # run a Noise program
```

## License

MIT
