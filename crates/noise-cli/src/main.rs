//! `noise` CLI: run a file (`noise file.noise`), start a REPL (`noise`), or install
//! the editor integration (`noise ide-integration`).

use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use noise_core::{Engine, InputValue};

/// Process exit codes (BSD `sysexits`-style, the subset a CLI needs): `0` success, `1` a program
/// or I/O failure (parse/runtime error, unreadable file), `2` a *usage* error (bad flags/arguments).
/// Separating usage (2) from runtime (1) is what lets scripts tell "you called me wrong" apart from
/// "the program failed" (finding G2).
const EXIT_RUNTIME: i32 = 1;
const EXIT_USAGE: i32 = 2;

/// The VS Code / Cursor syntax extension, baked into the binary so `ide-integration`
/// is self-contained no matter where `noise` runs from. These are a vendored copy of
/// `editors/vscode-noise/` kept in sync by `build.rs` — vendored (rather than reached
/// via `../../../editors`) so the files survive the `cargo publish` tarball.
const EXT_PKG_JSON: &str = include_str!("../vendor/vscode-noise/package.json");
const EXT_LANG_CONFIG: &str = include_str!("../vendor/vscode-noise/language-configuration.json");
const EXT_TMLANGUAGE: &str = include_str!("../vendor/vscode-noise/syntaxes/noise.tmLanguage.json");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--help`/`-h` and `--version` are honored in ANY position (finding G1) and win over everything.
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return;
    }
    if args.iter().any(|a| a == "--version") {
        print_version();
        return;
    }
    match args.first().map(String::as_str) {
        Some("ide-integration") => install_ide_integration(),
        Some("validate") => run_validate(&args[1..]),
        // Everything else is "run a program": a file path plus `--input` flags, or — when nothing is
        // given and stdin is piped — a program read from stdin.
        _ => run_or_repl(&args),
    }
}

/// Print a usage error to stderr followed by the short usage, and exit with [`EXIT_USAGE`] (2).
/// A usage error is the caller invoking `noise` wrong (bad/unknown flags, missing arguments) — kept
/// distinct from a program failure (exit 1) so scripts can tell them apart (finding G2).
fn usage_error(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!("run `noise --help` for usage.");
    std::process::exit(EXIT_USAGE);
}

/// The two-rung Ctrl-C ladder (PLAN-PRECISION Track H). The **first** Ctrl-C during a run
/// soft-stops it: completed statements keep their values, the in-flight query folds the chunks it
/// drew (an honest partial estimate), and the document renders normally with the "stopped early"
/// warning — the same path the `max_time` deadline takes. The **second** kills the process the
/// classic way (exit 130 = 128 + SIGINT). With no run active (REPL idle, reading stdin), Ctrl-C
/// exits immediately, as it always did.
mod interrupt {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Mutex;

    /// The active run's soft-stop handle, registered for the duration of each `run_to_document`.
    static ACTIVE: Mutex<Option<noise_core::exec::CancelToken>> = Mutex::new(None);
    /// Ctrl-C presses during the current run (the ladder's rung).
    static HITS: AtomicU8 = AtomicU8::new(0);

    /// Install the process-wide handler once, before the first run. The handler runs on `ctrlc`'s
    /// own thread, so locking / printing here is safe (this is not an async-signal context).
    pub fn install() {
        let _ = ctrlc::set_handler(|| {
            let active = ACTIVE.lock().ok().and_then(|g| g.clone());
            match active {
                Some(token) if HITS.fetch_add(1, Ordering::SeqCst) == 0 => {
                    eprintln!(
                        "\nstopping — showing what was sampled so far (Ctrl-C again to kill)"
                    );
                    token.stop();
                }
                _ => std::process::exit(130),
            }
        });
    }

    /// Register `engine`'s token as the run the ladder stops; deregisters on drop. Resets the
    /// rung, so each run gets its own first-soft/second-hard pair.
    pub fn guard(engine: &noise_core::Engine) -> Guard {
        HITS.store(0, Ordering::SeqCst);
        if let Ok(mut g) = ACTIVE.lock() {
            *g = Some(engine.cancel_token());
        }
        Guard
    }

    pub struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            if let Ok(mut g) = ACTIVE.lock() {
                *g = None;
            }
        }
    }
}

/// The CLI's default `max_time` (PLAN-PRECISION: the runaway guard is a runtime deadline now that
/// the op budget is gone). Lenient — every corpus example finishes orders of magnitude under it —
/// and `--max-time 0` turns it off entirely (what corpus goldens pin for bit-reproducibility).
const DEFAULT_MAX_TIME: std::time::Duration = std::time::Duration::from_secs(60);

/// The parsed form of a run/validate invocation: an optional positional file path, the repeatable
/// `--input name=value` overrides, and the PLAN-PRECISION runtime settings (`--precision`,
/// `--max-time`, `--resolution` — each pins its setting over the program's pragma).
/// Pure and testable (finding G6) — it never touches the filesystem or exits the process; the
/// caller decides what to do with the result.
#[derive(Debug, Default, PartialEq)]
struct Args {
    file: Option<String>,
    inputs: Vec<(String, InputValue)>,
    /// `--precision rel[,abs]`: keep drawing until `se <= max(abs, rel*|est|)`.
    precision: Option<(f64, f64)>,
    /// `--max-time <dur>`: the run's wall-clock ceiling. `None` = the default guard; `Some(None)`
    /// is expressed as `max_time: None` + `max_time_off: true` (`--max-time 0`).
    max_time: Option<std::time::Duration>,
    max_time_off: bool,
    /// `--resolution N`: the ambient signal-sampling resolution.
    resolution: Option<usize>,
}

impl Args {
    /// Apply the runtime settings to an engine (each pins its setting — "pragmas declare, `run()`
    /// overrides"). The `max_time` ladder: an explicit `--max-time 0` = no deadline, an explicit
    /// duration = that deadline, nothing = the lenient default guard.
    fn configure(&self, engine: &mut Engine) {
        if let Some(t) = self.precision {
            engine.set_precision(Some(t));
        }
        if let Some(n) = self.resolution {
            engine.set_resolution(n);
        }
        engine.set_max_time(if self.max_time_off {
            None
        } else {
            Some(self.max_time.unwrap_or(DEFAULT_MAX_TIME))
        });
    }
}

/// Parse a `--precision` value: `rel` or `rel,abs`, both finite and ≥ 0, not both 0.
fn parse_precision(v: &str) -> Result<(f64, f64), String> {
    let (rel_s, abs_s) = match v.split_once(',') {
        Some((r, a)) => (r.trim(), Some(a.trim())),
        None => (v.trim(), None),
    };
    let rel: f64 = rel_s
        .parse()
        .map_err(|_| format!("bad --precision {v:?}: {rel_s:?} is not a number"))?;
    let abs: f64 = match abs_s {
        Some(s) => s
            .parse()
            .map_err(|_| format!("bad --precision {v:?}: {s:?} is not a number"))?,
        None => 0.0,
    };
    if !rel.is_finite() || !abs.is_finite() || rel < 0.0 || abs < 0.0 || (rel == 0.0 && abs == 0.0)
    {
        return Err(format!(
            "bad --precision {v:?}: needs rel >= 0 and abs >= 0, not both 0 (e.g. --precision 1e-4)"
        ));
    }
    Ok((rel, abs))
}

/// Parse a `--max-time` value: `0` (off), a bare number of seconds (`2`, `1.5`), or a suffixed
/// duration (`2s`, `500ms`, `3m`).
fn parse_max_time(v: &str) -> Result<Option<std::time::Duration>, String> {
    let v = v.trim();
    let (num, scale) = if let Some(n) = v.strip_suffix("ms") {
        (n, 1e-3)
    } else if let Some(n) = v.strip_suffix('s') {
        (n, 1.0)
    } else if let Some(n) = v.strip_suffix('m') {
        (n, 60.0)
    } else {
        (v, 1.0)
    };
    let secs: f64 = num.trim().parse().map_err(|_| {
        format!("bad --max-time {v:?}: expected 0, seconds, or e.g. 2s / 500ms / 3m")
    })?;
    if !secs.is_finite() || secs < 0.0 {
        return Err(format!("bad --max-time {v:?}: must be >= 0"));
    }
    let secs = secs * scale;
    if secs == 0.0 {
        return Ok(None); // 0 = no deadline
    }
    Ok(Some(std::time::Duration::from_secs_f64(secs)))
}

/// Parse a positive integer setting value (`--resolution`), accepting `3e3` notation.
fn parse_count(flag: &str, v: &str) -> Result<usize, String> {
    let n: f64 = v
        .trim()
        .parse()
        .map_err(|_| format!("bad {flag} {v:?}: not a number"))?;
    if !n.is_finite() || n < 1.0 || n.fract() != 0.0 {
        return Err(format!("bad {flag} {v:?}: needs a whole number >= 1"));
    }
    Ok(n as usize)
}

/// Parse `--input name=value` overrides, the runtime settings flags, and a single positional file
/// path out of `args`. An unknown `-`-prefixed argument is a usage error (rather than being
/// silently treated as a file path — finding G1); a second positional is a usage error.
fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut out = Args::default();
    let mut i = 0;
    // A settings flag's value: `--flag value` or `--flag=value`.
    fn flag_value<'a>(
        args: &'a [String],
        i: &mut usize,
        name: &str,
    ) -> Result<Option<&'a str>, String> {
        let a = &args[*i];
        if a == name {
            let v = args
                .get(*i + 1)
                .ok_or_else(|| format!("`{name}` needs a value"))?;
            *i += 2;
            return Ok(Some(v));
        }
        if let Some(v) = a.strip_prefix(&format!("{name}=")) {
            *i += 1;
            return Ok(Some(v));
        }
        Ok(None)
    }
    while i < args.len() {
        let a = &args[i];
        if a == "--input" || a == "-i" {
            let kv = args
                .get(i + 1)
                .ok_or_else(|| format!("`{a}` needs a `name=value` argument"))?;
            out.inputs.push(parse_input_arg(kv)?);
            i += 2;
        } else if let Some(kv) = a.strip_prefix("--input=") {
            out.inputs.push(parse_input_arg(kv)?);
            i += 1;
        } else if let Some(v) = flag_value(args, &mut i, "--precision")? {
            out.precision = Some(parse_precision(v)?);
        } else if let Some(v) = flag_value(args, &mut i, "--max-time")? {
            match parse_max_time(v)? {
                Some(d) => out.max_time = Some(d),
                None => out.max_time_off = true,
            }
        } else if let Some(v) = flag_value(args, &mut i, "--resolution")? {
            out.resolution = Some(parse_count("--resolution", v)?);
        } else if a.starts_with('-') && a != "-" {
            // An unknown flag is a usage error — not a file path (`noise --bogus`, finding G1).
            // (`-` alone is allowed through as a positional: the conventional "stdin" placeholder.)
            return Err(format!("unknown option {a:?}"));
        } else if out.file.is_none() {
            out.file = Some(a.clone());
            i += 1;
        } else {
            return Err(format!(
                "unexpected argument {a:?} (only one file path is accepted)"
            ));
        }
    }
    Ok(out)
}

/// `noise [file] [--input k=v]...` — run a program. With a file path, run it. With no path: if stdin
/// is piped (not a terminal), read the whole program from stdin and run it; otherwise start the REPL
/// (finding G2 — piped input no longer drops into the REPL and pollutes the stream with `»` prompts).
fn run_or_repl(args: &[String]) {
    let parsed = parse_args(args).unwrap_or_else(|e| usage_error(&e));
    interrupt::install();
    match parsed.file.clone() {
        Some(p) => run_file(&p, &parsed),
        None if !io::stdin().is_terminal() => run_stdin(&parsed),
        None => {
            if !parsed.inputs.is_empty() {
                usage_error("`--input` needs a program (a file, or piped stdin) to run against");
            }
            repl(&parsed)
        }
    }
}

/// `noise validate <file> [--input k=v]...` — parse + build the graph without running. Now honors
/// `--input` overrides (finding G2) so a program's inputs can be validated at specific values.
fn run_validate(args: &[String]) {
    let parsed = parse_args(args).unwrap_or_else(|e| usage_error(&e));
    match parsed.file.clone() {
        Some(p) => validate_file(&p, parsed.inputs),
        None => usage_error("`validate` needs a file path: noise validate <file>"),
    }
}

/// Read a whole program from stdin and run it (the piped, non-interactive path).
fn run_stdin(args: &Args) {
    let mut src = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut src) {
        eprintln!("error: cannot read program from stdin: {e}");
        std::process::exit(EXIT_RUNTIME);
    }
    let mut engine = Engine::new();
    engine.set_input_overrides(args.inputs.clone());
    args.configure(&mut engine);
    let doc = {
        let _stop = interrupt::guard(&engine);
        engine.run_to_document(&src)
    };
    if render_document(&doc, Some((&src, "<stdin>"))) {
        std::process::exit(EXIT_RUNTIME);
    }
}

fn print_version() {
    // The tool is invoked as `noise` (the crate is `noise-cli`), so present the tool name.
    println!("noise {}", env!("CARGO_PKG_VERSION"));
}

/// Parse a `name=value` input override. `true`/`false` become bools; everything else parses as a
/// number (the engine type-checks it against the input's declared kind).
fn parse_input_arg(kv: &str) -> std::result::Result<(String, InputValue), String> {
    let (name, value) = kv
        .split_once('=')
        .ok_or_else(|| format!("bad --input {kv:?}: expected name=value"))?;
    let name = name.trim();
    if name.is_empty() {
        return Err(format!("bad --input {kv:?}: empty input name"));
    }
    let value = value.trim();
    let v = match value {
        "true" => InputValue::Bool(true),
        "false" => InputValue::Bool(false),
        _ => InputValue::Num(
            value
                .parse::<f64>()
                .map_err(|_| format!("bad --input {kv:?}: {value:?} is not a number or bool"))?,
        ),
    };
    Ok((name.to_string(), v))
}

fn print_help() {
    println!(
        "noise — the Noise probabilistic language (v{})",
        env!("CARGO_PKG_VERSION")
    );
    println!("usage:");
    println!("  noise                       start a REPL (or run a program piped on stdin)");
    println!("  noise <file>                run a program file");
    println!("  noise <file> --input k=v    tune an inline input (repeatable; also -i k=v)");
    println!("  noise <file> --precision p  sample until se <= p*|est| (or `rel,abs`); overrides pragmas");
    println!(
        "  noise <file> --max-time t   wall-clock ceiling (e.g. 2s, 500ms; 0 = off, default 60s)"
    );
    println!("  noise <file> --resolution n ambient signal resolution");
    println!("  noise validate <file>       parse and build the graph without producing output");
    println!("  noise ide-integration       install the VS Code / Cursor syntax extension");
    println!("  noise --version             print the version");
    println!("  noise --help, -h            show this help");
}

/// Install the bundled syntax-highlighting extension into every editor we can find.
/// Cursor and VS Code share the same extension format, so one set of files serves both.
fn install_ide_integration() {
    let home = match home_dir() {
        Some(h) => h,
        None => {
            eprintln!("error: cannot locate your home directory (set HOME)");
            std::process::exit(1);
        }
    };

    // (display name, `~/<dir>` that exists when that editor is installed).
    let editors = [
        ("Cursor", ".cursor"),
        ("VS Code", ".vscode"),
        ("VS Code Insiders", ".vscode-insiders"),
    ];

    let mut installed = 0;
    for (name, dir) in editors {
        let base = home.join(dir);
        if !base.exists() {
            continue;
        }
        let dest = base.join("extensions").join("noise-lang");
        // A symlink here means a dev install (e.g. `ln -s` from the README) already
        // points at a source tree — leave it be rather than writing through it.
        if let Ok(meta) = std::fs::symlink_metadata(&dest) {
            if meta.file_type().is_symlink() {
                println!(
                    "• {name}: already linked (dev install) → {}",
                    dest.display()
                );
                installed += 1;
                continue;
            }
        }
        match write_extension(&dest) {
            Ok(()) => {
                println!("✓ {name}: installed → {}", dest.display());
                installed += 1;
            }
            Err(e) => eprintln!("✗ {name}: {e}"),
        }
    }

    if installed == 0 {
        eprintln!(
            "no supported editor found (looked for ~/.cursor, ~/.vscode, ~/.vscode-insiders)."
        );
        std::process::exit(1);
    }
    println!("\nReload the editor window to activate: Cmd/Ctrl+Shift+P → \"Reload Window\".");
}

/// Write the three extension files into `dest`, creating directories as needed.
fn write_extension(dest: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dest.join("syntaxes"))?;
    std::fs::write(dest.join("package.json"), EXT_PKG_JSON)?;
    std::fs::write(dest.join("language-configuration.json"), EXT_LANG_CONFIG)?;
    std::fs::write(
        dest.join("syntaxes").join("noise.tmLanguage.json"),
        EXT_TMLANGUAGE,
    )?;
    Ok(())
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn run_file(path: &str, args: &Args) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(EXIT_RUNTIME);
        }
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(args.inputs.clone());
    args.configure(&mut engine);
    let doc = {
        let _stop = interrupt::guard(&engine);
        engine.run_to_document(&src)
    };
    // Render the one `Document`: notes as text and plots as their one-line text cards, in emission
    // order, then the final value (or the error). The CLI is just another renderer of the same
    // structure the playground uses (PLAN-LITERATE §D5).
    let errored = render_document(&doc, Some((&src, path)));
    if errored {
        std::process::exit(EXIT_RUNTIME);
    }
}

/// Render a `Document` to the terminal: emitted notes/plots in order, then the final value or the
/// error. Code blocks are the input file, so they're not re-printed. Returns whether the run errored.
/// `source`, when present, is `(src, name)` used to render a `name:line:col` header and a caret line
/// under the offending span (finding D1).
fn render_document(doc: &noise_core::doc::Document, source: Option<(&str, &str)>) -> bool {
    use noise_core::doc::Block;
    for block in &doc.blocks {
        match block {
            Block::Code { .. } => {}
            Block::Note { text, .. } => println!("{text}"),
            Block::Plot { text, .. } => println!("{text}"),
            Block::Input { spec, value, .. } => {
                let shown = match value {
                    InputValue::Num(n) => format!("{n}"),
                    InputValue::Bool(b) => format!("{b}"),
                };
                println!("input {} = {shown}", spec.name);
            }
            // `Block` is `#[non_exhaustive]` (E2): ignore any future block kind this CLI predates.
            _ => {}
        }
    }
    if let Some(t) = &doc.result.truncated {
        println!("… {} more emissions not shown (output capped)", t.dropped);
    }
    // Run warnings (PLAN-PRECISION Track C): a query the deadline cut short of its precision
    // target, a soft-stopped run. Stderr, so piped output stays clean.
    for w in &doc.result.warnings {
        eprintln!("warning: {w}");
    }
    match &doc.result.error {
        Some(e) => {
            match source {
                Some((src, name)) => eprintln!("{}", render_error(src, name, e)),
                None => eprintln!("{}", e.message),
            }
            true
        }
        None => {
            // Don't echo a trailing `unit` (e.g. when the program ends in `plot(...)`).
            if let Some(v) = &doc.result.value {
                println!("{}", v.text);
            }
            false
        }
    }
}

/// Render a spanned error the rustc way: a `name:line:col: message` header, then a two-line caret
/// block underlining the offending span in its source line (finding D1). Columns are character-based
/// (UTF-8-safe, coordinating with D4), and a multi-line span points at its start line. Example:
///
/// ```text
/// prog.noise:2:5: runtime error: undefined variable 'foo'
///   |
/// 2 | y = foo + 1
///   |     ^^^
/// ```
fn render_error(src: &str, name: &str, err: &noise_core::doc::DocError) -> String {
    render_span_error(src, name, &err.message, err.span)
}

/// Core caret renderer shared by the run path (a `DocError`) and `validate` (a raw `NoiseError`).
fn render_span_error(
    src: &str,
    name: &str,
    message: &str,
    span: noise_core::error::Span,
) -> String {
    use std::fmt::Write;
    let (line, col) = span.line_col(src);
    // The engine's message ends with a redundant " (at start..end)" byte range; the line:col header
    // and caret replace it, so trim it for the CLI.
    let msg = strip_span_suffix(message);

    let mut out = String::new();
    let _ = writeln!(out, "{name}:{line}:{col}: {msg}");

    let src_line = src.lines().nth(line.saturating_sub(1)).unwrap_or("");
    let gutter = line.to_string();
    let pad = " ".repeat(gutter.len());
    let _ = writeln!(out, "{pad} |");
    let _ = writeln!(out, "{gutter} | {src_line}");

    // Indent the caret to `col`, preserving tabs so it lines up in a tab-using terminal.
    let indent: String = src_line
        .chars()
        .take(col.saturating_sub(1))
        .map(|c| if c == '\t' { '\t' } else { ' ' })
        .collect();
    // Caret width = the span's character count, clamped to what remains on this line (so a
    // multi-line span just underlines to end of the start line), at least one `^`.
    let span_chars = src
        .get(span.start..span.end)
        .map(|s| s.chars().count())
        .unwrap_or(0);
    let line_remaining = src_line
        .chars()
        .count()
        .saturating_sub(col.saturating_sub(1));
    let carets = "^".repeat(span_chars.min(line_remaining).max(1));
    let _ = write!(out, "{pad} | {indent}{carets}");
    out
}

/// Drop a trailing " (at start..end)" byte-range suffix from an engine error message (the CLI shows
/// line:col + a caret instead).
fn strip_span_suffix(msg: &str) -> &str {
    if msg.ends_with(')') {
        if let Some(idx) = msg.rfind(" (at ") {
            return &msg[..idx];
        }
    }
    msg
}

/// Parse `path` and evaluate it to build the sample-DAG, reporting any errors — but without
/// running the Monte Carlo (`P`/`E`/`Var`/`Q` skip sampling) and without printing the program's
/// output or its final value. This catches syntax errors and graph-construction errors (undefined
/// names, type/shape mismatches, etc.) that pure parsing would miss, so it's a fast "does this
/// program hold together?" check that finishes regardless of the program's sample budget.
fn validate_file(path: &str, inputs: Vec<(String, InputValue)>) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(EXIT_RUNTIME);
        }
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(inputs);
    match engine.check(&src) {
        Ok(_) => println!("✓ {path}: valid"),
        Err(e) => {
            eprintln!("{}", render_span_error(&src, path, &e.to_string(), e.span));
            std::process::exit(EXIT_RUNTIME);
        }
    }
}

fn repl(args: &Args) {
    println!("noise REPL — type expressions, Ctrl-D to exit");
    let mut engine = Engine::new();
    args.configure(&mut engine);
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("» ");
        stdout.flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let doc = {
            let _stop = interrupt::guard(&engine);
            engine.run_to_document(line)
        };
        render_document(&doc, Some((line, "<repl>")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise_core::Engine;

    /// Render the caret for the first error a program produces (the run path a user hits).
    fn error_caret(src: &str, name: &str) -> String {
        let doc = Engine::new().run_to_document(src);
        let e = doc.result.error.expect("expected an error");
        render_error(src, name, &e)
    }

    #[test]
    fn strip_span_suffix_drops_only_the_trailing_byte_range() {
        assert_eq!(
            strip_span_suffix("runtime error: undefined variable 'x' (at 4..5)"),
            "runtime error: undefined variable 'x'"
        );
        // no suffix → unchanged
        assert_eq!(strip_span_suffix("boom"), "boom");
        // a parenthesized message with no " (at " is untouched
        assert_eq!(strip_span_suffix("f(x)"), "f(x)");
    }

    #[test]
    fn caret_points_under_a_mid_file_error() {
        // `foo` is undefined; it sits on line 2, column 5 — the caret must land under it.
        let src = "a = 1\ny = foo + 1\n";
        let rendered = error_caret(src, "prog.noise");
        let expected = "\
prog.noise:2:5: runtime error: undefined variable 'foo'
  |
2 | y = foo + 1
  |     ^^^";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn caret_is_utf8_safe_and_does_not_panic() {
        // A stray non-ASCII char (finding D4) must render a caret under the actual glyph without
        // panicking on a non-char-boundary slice — the whole point of the D4 span fix.
        let src = "x = 1\nπ = 3\n";
        let rendered = error_caret(src, "u.noise");
        let expected = "\
u.noise:2:1: unexpected character 'π'
  |
2 | π = 3
  | ^";
        assert_eq!(rendered, expected);
    }

    // === argv parsing (findings G1/G2/G6) ===================================================

    fn s(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn parse_input_arg_numbers_bools_and_errors() {
        assert_eq!(
            parse_input_arg("dice=6").unwrap(),
            ("dice".into(), InputValue::Num(6.0))
        );
        // whitespace around name/value is trimmed
        assert_eq!(
            parse_input_arg(" k = 2.5 ").unwrap(),
            ("k".into(), InputValue::Num(2.5))
        );
        assert_eq!(
            parse_input_arg("flag=true").unwrap(),
            ("flag".into(), InputValue::Bool(true))
        );
        assert_eq!(
            parse_input_arg("flag=false").unwrap(),
            ("flag".into(), InputValue::Bool(false))
        );
        // no `=`, empty name, and non-number/bool value are all errors
        assert!(parse_input_arg("noequals").is_err());
        assert!(parse_input_arg("=5").is_err());
        assert!(parse_input_arg("k=notanum").is_err());
    }

    #[test]
    fn parse_args_splits_file_and_inputs() {
        // file + repeated inputs in both `--input k=v` and `--input=k=v` and `-i` forms
        let got = parse_args(&s(&[
            "prog.noise",
            "--input",
            "a=1",
            "--input=b=2",
            "-i",
            "c=true",
        ]))
        .unwrap();
        assert_eq!(got.file.as_deref(), Some("prog.noise"));
        assert_eq!(
            got.inputs,
            vec![
                ("a".into(), InputValue::Num(1.0)),
                ("b".into(), InputValue::Num(2.0)),
                ("c".into(), InputValue::Bool(true)),
            ]
        );
        // inputs may precede the file
        assert_eq!(
            parse_args(&s(&["-i", "n=3", "p.noise"]))
                .unwrap()
                .file
                .as_deref(),
            Some("p.noise")
        );
        // no args at all → no file, no inputs (the REPL/stdin case)
        assert_eq!(parse_args(&[]).unwrap(), Args::default());
    }

    #[test]
    fn parse_args_rejects_unknown_flags_and_extra_positionals() {
        // an unknown `-`-flag is a usage error, NOT a file path (finding G1)
        assert!(parse_args(&s(&["--version"])).is_err()); // handled earlier in main, never a file
        assert!(parse_args(&s(&["--bogus"])).is_err());
        assert!(parse_args(&s(&["-x"])).is_err());
        // `--input` with no value is a usage error
        assert!(parse_args(&s(&["--input"])).is_err());
        // a second positional file is a usage error
        assert!(parse_args(&s(&["a.noise", "b.noise"])).is_err());
        // a bad input value is surfaced as an error (routed to exit 2 by the caller)
        assert!(parse_args(&s(&["--input", "k=nope"])).is_err());
    }

    #[test]
    fn parse_args_reads_the_precision_settings_flags() {
        let got = parse_args(&s(&[
            "p.noise",
            "--precision",
            "1e-4",
            "--max-time",
            "2s",
            "--resolution=64",
        ]))
        .unwrap();
        assert_eq!(got.precision, Some((1e-4, 0.0)));
        assert_eq!(got.max_time, Some(std::time::Duration::from_secs(2)));
        assert!(!got.max_time_off);
        assert_eq!(got.resolution, Some(64));

        // `rel,abs` form; `--max-time 0` = explicitly off; `500ms` and bare seconds parse.
        let got = parse_args(&s(&["p.noise", "--precision=1e-3,1e-6", "--max-time=0"])).unwrap();
        assert_eq!(got.precision, Some((1e-3, 1e-6)));
        assert!(got.max_time_off);
        assert_eq!(
            parse_max_time("500ms").unwrap(),
            Some(std::time::Duration::from_millis(500))
        );
        assert_eq!(
            parse_max_time("1.5").unwrap(),
            Some(std::time::Duration::from_secs_f64(1.5))
        );
        assert_eq!(
            parse_max_time("3m").unwrap(),
            Some(std::time::Duration::from_secs(180))
        );

        // Validation: both-zero precision, negative durations, fractional counts.
        assert!(parse_args(&s(&["p.noise", "--precision", "0"])).is_err());
        assert!(parse_args(&s(&["p.noise", "--max-time", "-1s"])).is_err());
        assert!(parse_args(&s(&["p.noise", "--resolution", "2.5"])).is_err());
        assert!(parse_args(&s(&["p.noise", "--resolution", "0"])).is_err());
        // A settings flag with no value is a usage error.
        assert!(parse_args(&s(&["p.noise", "--precision"])).is_err());
    }

    #[test]
    fn parse_args_allows_bare_dash_as_positional() {
        // `-` alone is the conventional stdin placeholder, not an unknown flag.
        assert_eq!(parse_args(&s(&["-"])).unwrap().file.as_deref(), Some("-"));
    }

    /// Finding G3: the CLI ships a vendored copy of the editor grammar (baked in via `include_str!`)
    /// so it survives the `cargo publish` tarball. The build script verifies the whole extension is
    /// in sync; this test double-checks the grammar specifically as a CI signal, comparing the
    /// baked-in vendored bytes against the canonical source on disk. Skips when the canonical tree
    /// isn't present (a packaged build), where the vendored snapshot is authoritative.
    #[test]
    fn vendored_grammar_matches_canonical() {
        let canonical = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../editors/vscode-noise/syntaxes/noise.tmLanguage.json"
        );
        let Ok(on_disk) = std::fs::read_to_string(canonical) else {
            eprintln!("canonical grammar absent; skipping (packaged build)");
            return;
        };
        assert_eq!(
            on_disk, EXT_TMLANGUAGE,
            "the vendored TextMate grammar is stale — run crates/noise-cli/vendor/sync.sh and commit"
        );
    }
}
