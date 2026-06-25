//! `noise` CLI: run a file (`noise file.noise`) or start a REPL (`noise`).

use std::io::{self, BufRead, Write};

use noise_core::Engine;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("-h") | Some("--help") => print_help(),
        Some(path) => run_file(path),
        None => repl(),
    }
}

fn print_help() {
    println!("noise — the Noise probabilistic language");
    println!("usage:");
    println!("  noise            start a REPL");
    println!("  noise <file>     run a program file");
}

fn run_file(path: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let mut engine = Engine::new();
    let result = engine.run(&src);
    // Flush anything `Print` captured during the run (in source order), then the final value.
    print!("{}", engine.drain_output());
    match result {
        // Don't echo a trailing `unit` (e.g. when the program ends in `print(...)`).
        Ok(noise_core::Value::Unit) => {}
        Ok(value) => println!("{value}"),
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
        let result = engine.run(line);
        print!("{}", engine.drain_output());
        match result {
            Ok(noise_core::Value::Unit) => {}
            Ok(value) => println!("{value}"),
            Err(e) => eprintln!("{e}"),
        }
    }
}
