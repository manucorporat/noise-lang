//! Ad-hoc backend comparison harness (not a committed bench). Times `run_to_document` over every
//! `examples/*.noise`, fresh engine per rep, median of N reps. The active backend is whatever the
//! build features select (default interpreter / `--features jit` / `--features gpu`). Prints
//! `label<TAB>median_ms` so three runs can be joined into one table.
use noise_core::Engine;
use std::time::Instant;

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let filter = std::env::var("FILTER").ok();
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "noise"))
        .collect();
    files.sort();

    let mut total = 0.0f64;
    for path in &files {
        let src = std::fs::read_to_string(path).unwrap();
        let label = path.file_stem().unwrap().to_string_lossy().to_string();
        if filter.as_ref().is_some_and(|f| !label.contains(f.as_str())) {
            continue;
        }
        // one warm-up (cold caches / GPU adapter init) that we don't record
        let _ = Engine::new().run_to_document(&src);
        let mut times: Vec<f64> = (0..reps)
            .map(|_| {
                let mut eng = Engine::new();
                let t = Instant::now();
                let _ = std::hint::black_box(eng.run_to_document(std::hint::black_box(&src)));
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[times.len() / 2];
        total += med;
        println!("{label}\t{med:.2}");
    }
    println!("__TOTAL__\t{total:.2}");
}
