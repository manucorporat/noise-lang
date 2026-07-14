//! The kernel shapes G0 measures — each a stand-in for a real program's cone, generated the way an
//! emitter would generate it (a flat post-order `let vN = ...;` chain, one source per draw).

/// A generated kernel: the per-lane statement body, the root expression, and the two numbers the
/// cost model cares about (distinct nodes ≈ statements, RNG sources).
pub struct Shape {
    pub name: &'static str,
    pub body: String,
    pub root: String,
    pub stmts: usize,
    pub sources: usize,
}

/// `pi`: two uniforms and a circle test. The gate is *expected* to decline this on GPU — it is here
/// to price the floor (dispatch latency vs a kernel that costs nothing).
pub fn pi() -> Shape {
    let body = "\
    let v0 = src_unif(key, 0u, lane, -1.0, 2.0);
    let v1 = src_unif(key, 1u, lane, -1.0, 2.0);
    let v2 = v0 * v0 + v1 * v1;
    let v3 = select(0.0, 4.0, v2 < 1.0);"
        .to_string();
    Shape { name: "pi", body, root: "v3".into(), stmts: 4, sources: 2 }
}

/// `barrier_option`-shaped: a `steps`-step GBM path with a knock-out check per step. The plan's
/// single best GPU candidate — many lanes, moderate cone, one `normal` (ln + sincos + sqrt) per
/// step. 100 steps ≈ 400 ops/draw, matching the measured shape of the real example.
pub fn barrier(steps: usize) -> Shape {
    let mut body = String::from("    var s = 100.0;\n    var alive = 1.0;\n");
    for t in 0..steps {
        body.push_str(&format!(
            "    let z{t} = src_normal(key, {t}u, lane, 0.0, 1.0);
    s = s * (1.0 + 0.0003 + 0.02 * z{t});
    alive = alive * select(1.0, 0.0, s > 130.0);\n"
        ));
    }
    body.push_str("    let payoff = alive * max(s - 100.0, 0.0);");
    Shape {
        name: "barrier",
        body,
        root: "payoff".into(),
        stmts: 4 * steps + 3,
        sources: steps,
    }
}

/// `am_vs_fm`-shaped: a trig-dense signal kernel. Few draws, lots of `sin` — the shape where the
/// GPU's hardware transcendentals should be worth the most, and where the CPU pays a polynomial.
pub fn signal(harmonics: usize) -> Shape {
    let mut body = String::from(
        "    let f = src_unif(key, 0u, lane, 200.0, 600.0);
    let a = src_unif(key, 1u, lane, 0.5, 1.0);
    var acc = 0.0;\n",
    );
    for h in 1..=harmonics {
        body.push_str(&format!(
            "    let t{h} = f32({h}u) * 0.001;
    let c{h} = nz_sincos(6.2831855 * f * t{h} + a * nz_sincos(2.0 * t{h}).x);
    acc = acc + c{h}.x * a / f32({h}u);\n"
        ));
    }
    Shape {
        name: "signal",
        body,
        root: "acc".into(),
        stmts: 5 * harmonics + 3,
        sources: 2,
    }
}

/// A synthetic straight-line chain of `n` statements — the compile-time probe. `turboquant` is
/// ~17.6k nodes per draw and `prisoners` ~45k, and the plan's single biggest unknown is whether a
/// shader that size compiles at all, and how long the driver takes. This shape answers that without
/// needing the emitter to exist.
///
/// It is deliberately *not* foldable: each statement depends on the previous one and on a lane
/// value, so neither Naga nor the Metal compiler can collapse the chain.
///
/// `salt` perturbs the coefficients so that each call produces *different shader text*. That is not
/// cosmetic — Metal keeps an on-disk compiled-shader cache, so re-compiling an identical source
/// measures a cache hit, not a compile. (This spike reported a 2.6x-too-fast compile time until the
/// salt was added.) A cold compile is what a first-time playground visitor pays, so it is the number
/// the gate has to be built on; the warm one is what a repeat visitor pays.
pub fn chain(n: usize, salt: usize) -> Shape {
    let mut body = String::from("    let x = src_normal(key, 0u, lane, 0.0, 1.0);\n    var acc = x;\n");
    for i in 0..n {
        // A rotating mix of cheap ops, so the chain isn't a single fused multiply-add pattern.
        let c = 1.0 + ((i + salt) % 7) as f32 * 0.01 + salt as f32 * 0.0001;
        match i % 4 {
            0 => body.push_str(&format!("    acc = acc * {c:.3} + x;\n")),
            1 => body.push_str(&format!("    acc = acc - x * {c:.3};\n")),
            2 => body.push_str(&format!("    acc = max(acc, x * {c:.3});\n")),
            _ => body.push_str(&format!("    acc = acc * 0.5 + {c:.3} * x * x;\n")),
        }
    }
    Shape {
        name: "chain",
        body,
        root: "acc".into(),
        stmts: n + 2,
        sources: 1,
    }
}

/// The head-to-head shape: sum `n` standard normals per lane.
///
/// Chosen because it is expressible *identically* in Noise (`zs ~[n] normal(0,1); vec::sum(zs)`) and
/// in WGSL, so the CPU and GPU can be timed on the same kernel over the same draws — no
/// hand-waving about "comparable" shapes. It is also transcendental-dense in the same proportion as
/// `barrier_option` (one ln + one sincos + one sqrt per draw), which is the demo the plan bets on.
pub fn sum_normals(n: usize) -> Shape {
    let mut body = String::from("    var acc = 0.0;\n");
    for i in 0..n {
        body.push_str(&format!(
            "    acc = acc + src_normal(key, {i}u, lane, 0.0, 1.0);\n"
        ));
    }
    Shape { name: "sum_normals", body, root: "acc".into(), stmts: 2 * n + 1, sources: n }
}

/// [`sum_normals`], but emitted as a **loop** over the source index instead of `n` unrolled draws.
///
/// This is the same computation over the same draws (sources `0..n`, so the counters and therefore
/// every bit are identical) — but the shader contains **one** `squares64` instead of `n` of them.
/// If compile cost really is driven by the inlined hash, this collapses it, and it is the single
/// most important thing G1's emitter can know.
///
/// It is not a general answer: an arbitrary `RvGraph` cone is a DAG, not a loop. But it is the
/// answer for the shape that dominates the demos — an **array draw** (`zs ~[n] normal(0,1)`) folded
/// by a vector op is exactly "n sources with consecutive ids, one body", which is a loop by
/// construction. `barrier_option`, `turboquant` and `am_vs_fm` are all this shape.
pub fn sum_normals_looped(n: usize) -> Shape {
    let body = format!(
        "    var acc = 0.0;
    for (var s = 0u; s < {n}u; s = s + 1u) {{
        acc = acc + src_normal(key, s, lane, 0.0, 1.0);
    }}"
    );
    Shape { name: "sum_normals_loop", body, root: "acc".into(), stmts: 4, sources: n }
}

/// The FMA probe — the single most consequential shape in the spike.
///
/// `a*b + c` may be *contracted* into a fused multiply-add (one rounding instead of two). WGSL
/// explicitly permits this and Metal does it by default. If the GPU contracts, then no amount of
/// care with constants or Horner order can make GPU lane arithmetic bit-identical to the CPU
/// backends — which would move the cross-backend contract's boundary from "everything but the
/// vendor transcendentals" to "the draws, and nothing else".
///
/// Each lane computes `a*b + c` on values chosen so the two roundings visibly disagree; `main.rs`
/// compares against BOTH `a*b + c` and `f32::mul_add(a, b, c)` on the CPU and reports which one the
/// GPU produced.
pub fn fma_probe() -> Shape {
    let body = "\
    let a = 1.0 + f32(lane) * 0.0000001;
    let b = 1.0 + f32(lane) * 0.0000003;
    let v = a * b - 1.0;"
        .to_string();
    Shape { name: "fma_probe", body, root: "v".into(), stmts: 3, sources: 0 }
}

/// The **raw draw** shape: write the lane's 24-bit draw straight out as an f32 integer (every value
/// below 2^24 is exact in f32), so nothing rounds between the hash and the comparison.
///
/// This is what isolates the RNG from the arithmetic. It is the only shape that can be held to a
/// bitwise standard once the FMA probe has spoken, and it is the one that carries the C0
/// certification onto the GPU.
pub fn raw_bits(source: u32, pair: bool) -> Shape {
    let body = if pair {
        format!("    let v = f32(my_half(pair_bits(key, {source}u, lane), lane));")
    } else {
        format!("    let v = f32(lane_bits48(key, {source}u, lane).x);")
    };
    Shape { name: "raw_bits", body, root: "v".into(), stmts: 1, sources: 1 }
}

/// Distribution shapes: one source, written straight out, compared against `noise_core::rng`'s fill.
/// These live in the ULP tier (the arithmetic on top of the draw contracts).
pub fn conformance(kind: &str) -> Shape {
    let body = match kind {
        "unif(0,1)" => "    let v = src_unif(key, 0u, lane, 0.0, 1.0);",
        // A non-trivial location/span: `loc + span*u` is a multiply-add, so this is the shape that
        // shows contraction reaching even a plain uniform (the (0,1) case hides it — 0 + 1*u).
        "unif(2,5)" => "    let v = src_unif(key, 0u, lane, 2.0, 3.0);",
        "normal(0,1)" => "    let v = src_normal(key, 2u, lane, 0.0, 1.0);",
        "exp(1)" => "    let v = src_exp(key, 3u, lane, 1.0);",
        other => panic!("unknown conformance shape {other}"),
    };
    Shape {
        name: "conformance",
        body: body.to_string(),
        root: "v".into(),
        stmts: 1,
        sources: 1,
    }
}
