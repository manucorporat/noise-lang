//! `noise` CLI: run a file (`noise file.noise`), start a REPL (`noise`), or install
//! the editor integration (`noise ide-integration`).

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use noise_core::{Engine, InputValue};

/// The VS Code / Cursor syntax extension, baked into the binary so `ide-integration`
/// is self-contained no matter where `noise` runs from. These are a vendored copy of
/// `editors/vscode-noise/` kept in sync by `build.rs` — vendored (rather than reached
/// via `../../../editors`) so the files survive the `cargo publish` tarball.
const EXT_PKG_JSON: &str = include_str!("../vendor/vscode-noise/package.json");
const EXT_LANG_CONFIG: &str = include_str!("../vendor/vscode-noise/language-configuration.json");
const EXT_TMLANGUAGE: &str = include_str!("../vendor/vscode-noise/syntaxes/noise.tmLanguage.json");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("-h") | Some("--help") => print_help(),
        Some("ide-integration") => install_ide_integration(),
        Some("validate") => match args.get(1) {
            Some(path) => validate_file(path),
            None => {
                eprintln!("error: `validate` needs a file path: noise validate <file>");
                std::process::exit(1);
            }
        },
        Some(_) => run_cli(&args),
        None => repl(),
    }
}

/// Parse the `noise <file> [--input k=v]...` invocation, then run it. Splits `--input name=value`
/// flags (repeatable) from the single positional file path.
fn run_cli(args: &[String]) {
    let mut path: Option<&str> = None;
    let mut inputs: Vec<(String, InputValue)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--input" || a == "-i" {
            match args.get(i + 1) {
                Some(kv) => match parse_input_arg(kv) {
                    Ok(pair) => inputs.push(pair),
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    }
                },
                None => {
                    eprintln!("error: `{a}` needs a `name=value` argument");
                    std::process::exit(1);
                }
            }
            i += 2;
        } else if let Some(kv) = a.strip_prefix("--input=") {
            match parse_input_arg(kv) {
                Ok(pair) => inputs.push(pair),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
            i += 1;
        } else if path.is_none() {
            path = Some(a);
            i += 1;
        } else {
            eprintln!("error: unexpected argument {a:?}");
            std::process::exit(1);
        }
    }
    match path {
        Some(p) => run_file(p, inputs),
        None => {
            eprintln!("error: no file to run");
            std::process::exit(1);
        }
    }
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
    println!("noise — the Noise probabilistic language");
    println!("usage:");
    println!("  noise                    start a REPL");
    println!("  noise <file>             run a program file");
    println!("  noise <file> --input k=v tune an inline input (repeatable)");
    println!("  noise validate <file>    parse and build the graph without producing output");
    println!("  noise ide-integration    install the VS Code / Cursor syntax extension");
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

fn run_file(path: &str, inputs: Vec<(String, InputValue)>) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let mut engine = Engine::new();
    engine.set_input_overrides(inputs);
    let doc = engine.run_to_document(&src);
    // Render the one `Document`: notes as text and plots as their one-line text cards, in emission
    // order, then the final value (or the error). The CLI is just another renderer of the same
    // structure the playground uses (PLAN-LITERATE §D5).
    let errored = render_document(&doc);
    if errored {
        std::process::exit(1);
    }
}

/// Render a `Document` to the terminal: emitted notes/plots in order, then the final value or the
/// error. Code blocks are the input file, so they're not re-printed. Returns whether the run errored.
fn render_document(doc: &noise_core::doc::Document) -> bool {
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
        }
    }
    if let Some(t) = &doc.result.truncated {
        println!("… {} more emissions not shown (output capped)", t.dropped);
    }
    match &doc.result.error {
        Some(e) => {
            eprintln!("{}", e.message);
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

/// Parse `path` and evaluate it to build the sample-DAG, reporting any errors — but without
/// running the Monte Carlo (`P`/`E`/`Var`/`Q` skip sampling) and without printing the program's
/// output or its final value. This catches syntax errors and graph-construction errors (undefined
/// names, type/shape mismatches, etc.) that pure parsing would miss, so it's a fast "does this
/// program hold together?" check that finishes regardless of the program's sample budget.
fn validate_file(path: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let mut engine = Engine::new();
    match engine.check(&src) {
        Ok(_) => println!("✓ {path}: valid"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn repl() {
    println!("noise REPL — type expressions, Ctrl-D to exit");
    let mut engine = Engine::new();
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
        let doc = engine.run_to_document(line);
        render_document(&doc);
    }
}
