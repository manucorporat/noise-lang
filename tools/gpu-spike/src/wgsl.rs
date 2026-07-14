//! WGSL source generation for the G0 spike (PLAN-WEBGPU).
//!
//! This is a *throwaway* transcription of what `wgsl_emit.rs` would eventually generate — enough to
//! answer G0's questions honestly, not the emitter itself. What it must get exactly right is the
//! **draw contract**: the same `squares64`, the same counter layout, the same 48-bit consumption,
//! the same pair split. The spike checks that bit-for-bit against `noise_core::rng` (see `main.rs`),
//! because a benchmark of the *wrong* hash would tell us nothing about the real kernel's cost.
//!
//! The load-bearing discovery: **WGSL has no `u64`.** `squares64` is five 64-bit wrapping multiplies,
//! so every one of them is emulated on `vec2<u32>` (lo, hi) — and that emulation is what the plan's
//! "70–90 ALU ops per uniform" risk is about. Two things make it much cheaper than feared:
//!
//!   * `rotate_left(x, 32)` on a u64 is a **half-swap** — `x.yx` in WGSL, free.
//!   * A 64×64 wrapping multiply only needs the *low* 64 bits, so three of the four partial
//!     products collapse: `hi = hi(a.lo·b.lo) + a.lo·b.hi + a.hi·b.lo`. Only one wide 32×32→64
//!     multiply survives per `mul64`.
//!
//! And the consumption contract falls out beautifully: the C0 rule is "the top 24 bits of each u32
//! half", so `draw48`'s two 24-bit halves are just `w.x >> 8` and `w.y >> 8`. The 48-bit value is
//! never assembled at all for the pair-shared sources.

/// Which generator the kernel draws from — the A/B that prices the plan's central RNG risk.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rng {
    /// The real thing: `squares64` with 64-bit arithmetic emulated on `vec2<u32>`.
    Squares,
    /// A GPU-native u32 hash (murmur-ish, ~12 ALU) standing in for the pcg4d family C0 rejected.
    /// Statistically NOT admissible — it exists only to measure what the certified hash costs us.
    Cheap,
    /// No hashing at all: the draw is a trivial function of the lane. Isolates the RNG's share by
    /// subtraction — anything left is the kernel's own arithmetic.
    None,
}

/// Where the transcendentals come from — the other A/B, and the one that prices *bitwise*
/// cross-backend parity (the plan's "cross-device reproducibility" risk row).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Trans {
    /// WGSL's `log`/`sin`/`cos` built-ins: hardware, but vendor-defined precision — so lanes would
    /// be ULP-close to the CPU backends, not bit-identical.
    Native,
    /// `approx.rs`'s f32 polynomials, transcribed. Bit-identical to every CPU backend, at whatever
    /// the polynomial costs over the built-in.
    Poly,
}

/// The u64-on-vec2<u32> layer plus `squares64`. `x` is the low word, `y` the high word.
const U64_EMU: &str = r#"
fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (mid << 16u) | (p00 & 0xffffu);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// Low 64 bits of a 64x64 product: the two high partial products are discarded, so only ONE
// 32x32->64 multiply is wide.
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul_wide(a.x, b.x);
    return vec2<u32>(ll.x, ll.y + a.x * b.y + a.y * b.x);
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

// squares64 (Widynski). `x.yx` IS `rotate_left(x, 32)` on a u64 — the half-swap is free.
fn squares64(ctr: vec2<u32>, key: vec2<u32>) -> vec2<u32> {
    var x = mul64(ctr, key);
    let y = x;
    let z = add64(y, key);
    x = add64(mul64(x, x), y); x = x.yx;
    x = add64(mul64(x, x), z); x = x.yx;
    x = add64(mul64(x, x), y); x = x.yx;
    let t = add64(mul64(x, x), z);
    x = t.yx;
    let f = add64(mul64(x, x), y);
    return vec2<u32>(t.x ^ f.y, t.y);   // t ^ (f >> 32)
}

// The pair-shared draw. Counter is `(source << 36) + (lane >> 1)`, which in words is just
// `(lane >> 1, source << 4)` — the high word is a compile-time constant per source.
//
// C0's consumption contract takes bits 8..31 of each half, so draw48's low 24 bits are `w.x >> 8`
// and its high 24 are `w.y >> 8`. The 48-bit value is never assembled.
fn pair_bits(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    let w = squares64(vec2<u32>(lane >> 1u, source << 4u), key);
    return vec2<u32>(w.x >> 8u, w.y >> 8u);
}

// `unif_int`'s per-lane draw: counter `(source << 36) + lane`, all 48 bits spent on one lane.
fn lane_bits48(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    let w = squares64(vec2<u32>(lane, source << 4u), key);
    return vec2<u32>(w.x >> 8u, w.y >> 8u);   // (lo24, hi24)
}
"#;

/// The GPU-native stand-in. Same signatures, so every kernel shape is unchanged between the two.
const CHEAP_RNG: &str = r#"
fn pair_bits(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    var a = ((lane >> 1u) ^ (source * 0x9e3779b9u)) ^ key.x;
    a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u;
    var b = a ^ 0x85ebca6bu; b *= 0xc2b2ae35u; b ^= b >> 13u;
    return vec2<u32>(a >> 8u, b >> 8u);
}
fn lane_bits48(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    return pair_bits(key, source, lane << 1u);
}
"#;

/// No hash at all — the subtraction baseline.
const NO_RNG: &str = r#"
fn pair_bits(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    let a = (lane >> 1u) + source + key.x;
    return vec2<u32>(a & 0xffffffu, (a * 3u) & 0xffffffu);
}
fn lane_bits48(key: vec2<u32>, source: u32, lane: u32) -> vec2<u32> {
    return pair_bits(key, source, lane << 1u);
}
"#;

/// Spell an f32 as an **exact** WGSL constant, by its bit pattern.
///
/// Not fastidiousness: a decimal literal is a re-rounding, and `approx.rs`'s coefficients are
/// specified as *f32 values* (`1.0 / 3.0` in f32, `std::f32::consts::LN_2`), so writing `0.33333334`
/// in the shader is a different number than the CPU evaluates. Bit-for-bit parity is decided by
/// exactly this kind of detail, so the emitter must never hand-round a constant — it must transcribe
/// the bits. (This is the form `wgsl_emit.rs` should use for every `ConstNum` too.)
fn f32c(x: f32) -> String {
    format!("bitcast<f32>({:#010x}u)", x.to_bits())
}

/// `approx.rs`'s f32 polynomials, transcribed with exact coefficients and the identical Horner
/// order. The subnormal lift is omitted: the draw paths feed `1 - u ∈ [2^-24, 1]`, so it is
/// unreachable there. A real emitter needs it for user-facing `math::log`.
fn poly_trans() -> String {
    let ln2 = f32c(std::f32::consts::LN_2);
    let sqrt2 = f32c(std::f32::consts::SQRT_2);
    let l = |i: i32| f32c(1.0f32 / i as f32);
    let two_pi = f32c(std::f32::consts::FRAC_2_PI);
    let pio2_hi = f32c(1.5703125);
    let pio2_lo = f32c((std::f64::consts::FRAC_PI_2 - 1.5703125) as f32);
    let s = |i: f32| f32c(i);
    format!(
        r#"
fn nz_ln(x: f32) -> f32 {{
    let bits = bitcast<u32>(x);
    let e0 = i32((bits >> 23u) & 0xffu) - 127;
    let m0 = bitcast<f32>((bits & 0x007fffffu) | 0x3f800000u);
    let big = m0 > {sqrt2};
    let m = select(m0, m0 * 0.5, big);
    let e = select(e0, e0 + 1, big);
    let f = (m - 1.0) / (m + 1.0);
    let z = f * f;
    var acc = {c9};
    acc = acc * z + {c7};
    acc = acc * z + {c5};
    acc = acc * z + {c3};
    acc = acc * z + 1.0;
    return 2.0 * f * acc + f32(e) * {ln2};
}}

// (sin(x), cos(x)) — one Cody-Waite reduction feeds both kernels, which is exactly what Box-Muller
// wants. WGSL's `round` is ties-to-even, matching Rust's `round_ties_even`.
fn nz_sincos(x: f32) -> vec2<f32> {{
    let k = round(x * {two_pi});
    let r = (x - k * {pio2_hi}) - k * {pio2_lo};
    let z = r * r;
    var s = {s3};
    s = s * z + {s2};
    s = s * z + {s1};
    s = s * z + {s0};
    let sin_r = r + r * z * s;
    var c = {k3};
    c = c * z + {k2};
    c = c * z + {k1};
    c = c * z + {k0};
    let cos_r = 1.0 - 0.5 * z + z * z * c;
    let kq = i32(k) & 3;
    let sn = select(select(select(sin_r, cos_r, kq == 1), -sin_r, kq == 2), -cos_r, kq == 3);
    let cs = select(select(select(cos_r, -sin_r, kq == 1), -cos_r, kq == 2), sin_r, kq == 3);
    return vec2<f32>(sn, cs);
}}
"#,
        c9 = l(9),
        c7 = l(7),
        c5 = l(5),
        c3 = l(3),
        s3 = s(1.0 / 362_880.0),
        s2 = s(-1.0 / 5040.0),
        s1 = s(1.0 / 120.0),
        s0 = s(-1.0 / 6.0),
        k3 = s(-1.0 / 3_628_800.0),
        k2 = s(1.0 / 40_320.0),
        k1 = s(-1.0 / 720.0),
        k0 = s(1.0 / 24.0),
    )
}

/// The built-in path: same two entry points, so the source lowerings below don't branch on `Trans`.
const NATIVE_TRANS: &str = r#"
fn nz_ln(x: f32) -> f32 { return log(x); }
fn nz_sincos(x: f32) -> vec2<f32> { return vec2<f32>(sin(x), cos(x)); }
"#;

/// The RNG *sources*, spelled once over whatever `pair_bits`/`nz_*` resolve to.
///
/// Each source is one hash **per lane** on the GPU, where the CPU pays one per lane *pair* — the
/// GPU recomputes the pair draw in both lanes and picks its half by parity. That is a deliberate
/// 2x hash cost bought in exchange for bit-identical draws and zero cross-lane communication; the
/// alternative (one invocation producing two lanes) doubles register pressure and halves occupancy.
fn sources() -> String {
    format!(
        "{}",
        SOURCES
            .replace("$SCALE24", &f32c(1.0 / (1u32 << 24) as f32))
            .replace("$TAU", &f32c(std::f32::consts::TAU))
    )
}

const SOURCES: &str = r#"
fn unit24(b: u32) -> f32 { return f32(b) * $SCALE24; }

fn my_half(b: vec2<u32>, lane: u32) -> u32 {
    return select(b.x, b.y, (lane & 1u) == 1u);
}

fn src_unif(key: vec2<u32>, source: u32, lane: u32, loc: f32, span: f32) -> f32 {
    return loc + span * unit24(my_half(pair_bits(key, source, lane), lane));
}

fn src_normal(key: vec2<u32>, source: u32, lane: u32, mu: f32, sigma: f32) -> f32 {
    let b = pair_bits(key, source, lane);
    let r = sqrt(-2.0 * nz_ln(1.0 - unit24(b.x)));
    let sc = nz_sincos($TAU * unit24(b.y));
    // even lane -> cos branch, odd lane -> sin branch (rng::normal_pair).
    let z = select(sc.y, sc.x, (lane & 1u) == 1u);
    return mu + sigma * r * z;
}

fn src_exp(key: vec2<u32>, source: u32, lane: u32, rate: f32) -> f32 {
    let b = my_half(pair_bits(key, source, lane), lane);
    return -nz_ln(1.0 - unit24(b)) / rate;
}

// Lemire multiply-high on 48 bits, done in 32-bit pieces. `bits48 = hi*2^24 + lo`, so
// `(bits48 * count) >> 48 == (count*hi + ((count*lo) >> 24)) >> 24`, and `count*hi` needs the wide
// multiply (count <= 2^24, hi < 2^24 -> up to 48 bits).
fn src_unif_int(key: vec2<u32>, source: u32, lane: u32, loc: f32, count: u32) -> f32 {
    let b = lane_bits48(key, source, lane);
    let t = mul_wide(count, b.x);                    // count * lo24
    let u = mul_wide(count, b.y);                    // count * hi24
    let tsh = (t.x >> 24u) | (t.y << 8u);            // (count*lo) >> 24
    let s = add64(u, vec2<u32>(tsh, 0u));
    let k = (s.x >> 24u) | (s.y << 8u);              // >> 24
    return loc + f32(k);
}
"#;

/// `src_unif_int` needs `mul_wide`/`add64` even under the cheap/none RNG, so those two helpers
/// come along for the ride. (Cost-wise irrelevant: no benchmarked shape draws integers.)
const WIDE_HELPERS_ONLY: &str = r#"
fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((mid << 16u) | (p00 & 0xffffu), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
"#;

/// Lanes per workgroup. 64 = two SIMD groups on Apple silicon; the spike does not sweep this.
pub const WORKGROUP: u32 = 64;

/// Assemble a full compute shader: prelude (RNG + transcendentals + sources), then the per-lane
/// body, then one store. `body` is a sequence of `let vN = ...;` statements and `root` names the
/// value written out — exactly the shape a post-order emitter produces.
pub fn shader(rng: Rng, trans: Trans, body: &str, root: &str) -> String {
    shader_salted(rng, trans, body, root, 0)
}

/// [`shader`] with a **cache-busting salt** — for compile-time measurements only.
///
/// Metal keeps a compiled-shader cache **on disk**, keyed on the source, so re-running this spike
/// re-compiles nothing and every "cold compile" number is a lie. (That is not hypothetical: it made
/// the 100-source row report 1.2 ms instead of ~700 ms, and it is why the first compile table in
/// this spike was ~2.6x too fast.) A non-zero salt emits a branch on a *uniform* the compiler cannot
/// fold away, making the generated MSL text unique per salt — one compare, and a genuinely cold
/// compile. Throughput measurements pass `0`, so they never pay for it.
pub fn shader_salted(rng: Rng, trans: Trans, body: &str, root: &str, salt: u32) -> String {
    let rng_src = match rng {
        Rng::Squares => U64_EMU.to_string(),
        Rng::Cheap => format!("{WIDE_HELPERS_ONLY}{CHEAP_RNG}"),
        Rng::None => format!("{WIDE_HELPERS_ONLY}{NO_RNG}"),
    };
    let trans_src = match trans {
        Trans::Native => NATIVE_TRANS.to_string(),
        Trans::Poly => poly_trans(),
    };
    let sources = sources();
    // `P.n` is the lane count and is never this value, so the store is dead at runtime — but the
    // compiler can't prove it, so it must emit the branch, and the MSL differs per salt.
    let salt_stmt = if salt == 0 {
        String::new()
    } else {
        format!("    if (P.n == {}u) {{ out[i] = {}.0; }}\n", 0xFFFF_0000u32 ^ salt, salt)
    };
    format!(
        r#"
struct Params {{
    key: vec2<u32>,
    lane0: u32,
    n: u32,
}};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
{rng_src}{trans_src}{sources}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= P.n) {{ return; }}
    let lane = P.lane0 + i;
    let key = P.key;
{salt_stmt}{body}
    out[i] = {root};
}}
"#
    )
}
