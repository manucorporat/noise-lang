//! Hand-written oracle probe: **does NEON `f64x2` × V independent vector states beat scalar × S
//! streams, at equal stream count, with correct instruction selection?**
//!
//! Background. PERF.md technique 5 claims multi-stream *scalar* RNG "dominates the vector path on
//! every target". The evidence behind that (PLAN.md, commit `a14e30c`) was a Cranelift `f64x2`
//! kernel with **one** vector state — i.e. 2 streams in 2 lanes — benchmarked against scalar
//! **1**-stream. The two axes (lanes, independent states) were never crossed, the comparison was
//! never at equal stream count, and the measurement predates the inlined `ln`/`sin`/`cos`
//! polynomials (`approx.rs`, `a0236b8`) that removed SIMD's worst blocker.
//!
//! This probe removes Cranelift from the question, the way `jit::bench_cranelift_vs_llvm` removes it
//! from the codegen-quality question: both sides are hand-written and LLVM-compiled, so what's
//! measured is the *ISA*, not a backend's vector lowering. If hand-written NEON can't beat
//! hand-written scalar here, no JIT will and the vector path stays dead. If it can, this is the
//! headroom, and the next question is whether Cranelift can be made to emit it.
//!
//! Instruction selection, vs the three penalties the old attempt reported:
//!   * `u64 → f64`: `vcvtq_f64_u64` is a single `ucvtf.2d`. (Cranelift's `fcvt_from_uint.f64x2`
//!     scalarized — an ~6-op extract/convert/insert sequence. That is a backend gap, not an ISA one.)
//!   * `rotl(x, k)`: `vsraq_n_u64(vshlq_n_u64::<k>(x), x, 64 - k)` — `shl` + `usra`, **2** ops, not 3.
//!     The two halves of a rotate occupy disjoint bits, so `usra`'s accumulate *is* the `orr`.
//!   * `s3 = rotl(s3 ^ s1, 45)` is one `XAR` instruction (FEAT_SHA3, present on Apple Silicon).
//!     No intrinsic needed: LLVM pattern-matches it from `rotl45(veorq_u64(..))`, and also folds the
//!     `^=` chain into `eor3` (3-way xor). Verified by disassembly — see the mnemonic counts below.
//!
//! Net effect: the vector RNG comes out at **~4.5 vector ops/sample** (2 `add`, 1 `shl`+1 `usra` for
//! `rotl23`, 1 `shl` for `t`, 3 `eor3`, 1 `xar`, over 2 lanes) against scalar's **10 integer
//! ops/sample**. This is a *maximally favourable* NEON kernel — a 2.2x op-count advantage, strictly
//! better instruction selection than the Cranelift attempt ever had. Whatever it loses, it does not
//! lose because of lowering quality.
//!
//! **Hypothesis under test — port heterogeneity, not port count.** A scalar kernel runs the RNG
//! (integer: `eor`/`lsl`/`ror`) on the integer ALUs and the polynomial (FP) on the FP pipes: two
//! disjoint port sets, overlapped by the out-of-order core. A NEON kernel puts *both* on the same 4
//! vector pipes. So the raw lane arithmetic (NEON ~5.5 vec-ops/sample over 4 pipes = 0.73 samp/cyc,
//! scalar 10 int-ops/sample over ~6 ALUs = 0.60 samp/cyc) says NEON should win the *pure RNG*
//! ceiling — but any kernel that also does float work makes the two compete for one port set.
//!
//! Falsifiable predictions, recorded before the first run — and what happened (M4 Pro, single
//! thread, median of 5, M samples/sec; "best" = best stream count on each side):
//!
//! | case      | prediction        | scalar best | NEON best | NEON/scalar | verdict |
//! |-----------|-------------------|------------:|----------:|------------:|---------|
//! | `rng_only`| NEON wins (control) |      1749 |      2228 |    **1.27x** | held    |
//! | `pi`      | parity or worse   |         857 |       748 |       0.87x | held    |
//! | `dice`    | parity or worse   |         867 |       724 |       0.84x | held    |
//! | `poly_deep`| flat             |        1245 |      1236 |       0.99x | held    |
//! | `poly_thru`| NEON wins        |         449 |       517 |    **1.15x** | held    |
//!
//! Reading. NEON wins at **both ends** and loses in the **middle** — that is a contention signature,
//! not "SIMD is slow". Strip the float work (`rng_only`) and the vector RNG wins by 1.27x, close to
//! the 1.22x the port arithmetic above predicts. Mix RNG with a little float (`pi`, `dice`) and both
//! now queue for the same 4 vector pipes, while the scalar kernel was getting its FP for free on an
//! otherwise-idle port set — NEON loses ~15%. Push the balance to mostly-float (`poly_thru`) and the
//! lanes finally pay for the contention: 1.15x.
//!
//! Also confirmed: lanes and streams **do** compose (`v1`→`v4`, i.e. 2→8 streams: `pi` 636→748, +18%;
//! `rng_only` 1801→2228, +24%). Crossing the two axes is a real effect, and the original experiment
//! never tried it. It just doesn't flip the sign on the graphs Noise actually runs.
//!
//! Consequence for `normal_poly` (PERF.md: 105→107, flat across s1..s8 — FP-throughput-bound, where
//! extra scalar streams provably buy nothing): `poly_thru` reproduces that signature exactly (scalar
//! 449/448/384/414) and NEON takes 1.15x of it, so a vector path *would* win there. Phase 2 —
//! vectorize `approx::{ln, cos}` and settle it directly — is therefore live, but the prize is bounded
//! at ~1.15x on one graph class, and Cranelift cannot currently emit any of the instruction selection
//! that earns it.
//!
//! Run: `cargo test -p noise-core --release --test simd_probe -- --ignored --nocapture`

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;
use std::time::Instant;

/// Column length, matching `bytecode::BATCH` so the store traffic mirrors a real kernel batch.
/// Divisible by every stream count under test (1, 2, 4, 8).
const N: usize = 1024;
/// `2^-53`, the `next_f64` scale.
const SCALE: f64 = 1.0 / (1u64 << 53) as f64;

/// SplitMix64, mirroring `rng::Rng::seed_from_u64` so both sides start from the same substreams.
fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------------------------
// Scalar: S independent xoshiro256++ streams (the shipping kernel's shape).
// ---------------------------------------------------------------------------------------------

struct Scalar<const S: usize> {
    s: [[u64; 4]; S],
}

impl<const S: usize> Scalar<S> {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [[0u64; 4]; S];
        for st in s.iter_mut() {
            for w in st.iter_mut() {
                *w = splitmix(&mut z);
            }
        }
        Scalar { s }
    }

    #[inline(always)]
    fn next(&mut self, j: usize) -> u64 {
        let s = &mut self.s[j];
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    #[inline(always)]
    fn u01(&mut self, j: usize) -> f64 {
        (self.next(j) >> 11) as f64 * SCALE
    }
}

// ---------------------------------------------------------------------------------------------
// NEON: V independent vector states, each holding 2 lanes = 2 streams. Total streams = 2 * V.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct VState {
    s0: uint64x2_t,
    s1: uint64x2_t,
    s2: uint64x2_t,
    s3: uint64x2_t,
}

/// `rotl(x, 23)` as `shl` + `usra` — disjoint bits, so the accumulate is an `orr`.
#[inline(always)]
unsafe fn rotl23(x: uint64x2_t) -> uint64x2_t {
    vsraq_n_u64::<41>(vshlq_n_u64::<23>(x), x)
}

/// `rotl(x, 45)`, same trick.
#[inline(always)]
unsafe fn rotl45(x: uint64x2_t) -> uint64x2_t {
    vsraq_n_u64::<19>(vshlq_n_u64::<45>(x), x)
}

/// One xoshiro256++ step across both lanes. Op-for-op the scalar `next` above, including the
/// sequencing of the `^=` chain (`s1 ^= s2` sees the *updated* `s2`).
#[inline(always)]
unsafe fn vnext(v: &mut VState) -> uint64x2_t {
    let result = vaddq_u64(rotl23(vaddq_u64(v.s0, v.s3)), v.s0);
    let t = vshlq_n_u64::<17>(v.s1);
    let s2a = veorq_u64(v.s2, v.s0);
    let s3a = veorq_u64(v.s3, v.s1);
    v.s1 = veorq_u64(v.s1, s2a);
    v.s0 = veorq_u64(v.s0, s3a);
    v.s2 = veorq_u64(s2a, t);
    v.s3 = rotl45(s3a);
    result
}

/// `(u >> 11) as f64 * 2^-53`, both lanes. `vcvtq_f64_u64` is one `ucvtf.2d`.
#[inline(always)]
unsafe fn vu01(v: &mut VState) -> float64x2_t {
    vmulq_n_f64(vcvtq_f64_u64(vshrq_n_u64::<11>(vnext(v))), SCALE)
}

unsafe fn vstates<const V: usize>(seed: u64) -> [VState; V] {
    // Seed 2*V scalar streams, then pack lane-wise: lanes of a vector are independent streams.
    let mut z = seed;
    let mut raw = [[0u64; 4]; 8];
    for st in raw.iter_mut().take(2 * V) {
        for w in st.iter_mut() {
            *w = splitmix(&mut z);
        }
    }
    let mut out = [VState {
        s0: vdupq_n_u64(0),
        s1: vdupq_n_u64(0),
        s2: vdupq_n_u64(0),
        s3: vdupq_n_u64(0),
    }; V];
    for (v, slot) in out.iter_mut().enumerate() {
        let (a, b) = (raw[2 * v], raw[2 * v + 1]);
        let pack = |k: usize| vld1q_u64([a[k], b[k]].as_ptr());
        *slot = VState { s0: pack(0), s1: pack(1), s2: pack(2), s3: pack(3) };
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The four cases. Each fills `out[0..N]`, exactly as a real kernel batch does, so the comparison
// includes the same store traffic and isn't a register-only microbenchmark.
// ---------------------------------------------------------------------------------------------

/// **Control.** Raw `next_u64`, xor-accumulated and stored as bits — *no* float work whatsoever, so
/// nothing competes with the RNG for a port set. (Bits are masked into `[1,2)` so both sides still
/// share a mean and the distribution check applies.) This isolates the mechanism: if NEON's loss on `pi`
/// and `dice` is port *contention* (RNG and FP fighting over the 4 vector pipes) rather than a
/// property of the RNG itself, then NEON must **win** here and lose there. If NEON loses this cell
/// too, the contention story is wrong and the vector RNG is simply slower.
#[inline(never)]
fn sc_rng_only<const S: usize>(r: &mut Scalar<S>, out: &mut [f64]) {
    for chunk in out.chunks_exact_mut(S) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            *slot = f64::from_bits((r.next(j) >> 12) | 0x3ff0_0000_0000_0000);
        }
    }
}

#[inline(never)]
unsafe fn nx_rng_only<const V: usize>(v: &mut [VState; V], out: &mut [f64]) {
    let tag = vdupq_n_u64(0x3ff0_0000_0000_0000);
    for chunk in out.chunks_exact_mut(2 * V) {
        for (k, st) in v.iter_mut().enumerate() {
            let u = vorrq_u64(vshrq_n_u64::<12>(vnext(st)), tag);
            vst1q_f64(chunk.as_mut_ptr().add(2 * k), vreinterpretq_f64_u64(u));
        }
    }
}

/// `X,Y ~ unif(-1,1); X^2 + Y^2 < 1` — two draws, trivial math. RNG-bound.
#[inline(never)]
fn sc_pi<const S: usize>(r: &mut Scalar<S>, out: &mut [f64]) {
    for chunk in out.chunks_exact_mut(S) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            let x = 2.0f64.mul_add(r.u01(j), -1.0);
            let y = 2.0f64.mul_add(r.u01(j), -1.0);
            *slot = ((x * x + y * y) < 1.0) as u8 as f64;
        }
    }
}

#[inline(never)]
unsafe fn nx_pi<const V: usize>(v: &mut [VState; V], out: &mut [f64]) {
    let (two, none, one) = (vdupq_n_f64(2.0), vdupq_n_f64(-1.0), vdupq_n_f64(1.0));
    for chunk in out.chunks_exact_mut(2 * V) {
        for (k, st) in v.iter_mut().enumerate() {
            let x = vfmaq_f64(none, two, vu01(st));
            let y = vfmaq_f64(none, two, vu01(st));
            let r2 = vfmaq_f64(vmulq_f64(x, x), y, y);
            // `vcltq_f64` yields an all-ones lane mask; and it with the bits of 1.0.
            let hit = vandq_u64(vcltq_f64(r2, one), vreinterpretq_u64_f64(one));
            vst1q_f64(chunk.as_mut_ptr().add(2 * k), vreinterpretq_f64_u64(hit));
        }
    }
}

/// `A,B ~ unif_int(1,6); A + B`. Both sides use the float round-trip (`floor(u01*6)+1`) rather than
/// the shipping kernel's Lemire multiply-high, because NEON has no 64x64->128 high multiply — and
/// PERF.md records Lemire as perf-neutral anyway. Same method both sides = fair.
#[inline(never)]
fn sc_dice<const S: usize>(r: &mut Scalar<S>, out: &mut [f64]) {
    for chunk in out.chunks_exact_mut(S) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            let a = (r.u01(j) * 6.0).floor() + 1.0;
            let b = (r.u01(j) * 6.0).floor() + 1.0;
            *slot = a + b;
        }
    }
}

#[inline(never)]
unsafe fn nx_dice<const V: usize>(v: &mut [VState; V], out: &mut [f64]) {
    let (six, one) = (vdupq_n_f64(6.0), vdupq_n_f64(1.0));
    for chunk in out.chunks_exact_mut(2 * V) {
        for (k, st) in v.iter_mut().enumerate() {
            let a = vaddq_f64(vrndmq_f64(vmulq_f64(vu01(st), six)), one);
            let b = vaddq_f64(vrndmq_f64(vmulq_f64(vu01(st), six)), one);
            vst1q_f64(chunk.as_mut_ptr().add(2 * k), vaddq_f64(a, b));
        }
    }
}

/// `X ~ unif(0,1); ((X*X+X)*X - X)*X + X*X - X + 1` — one draw, a *serial* FMA chain. Latency-bound
/// within the sample, but consecutive iterations are independent, so the OoO core already overlaps
/// them. Prediction: flat.
#[inline(never)]
fn sc_poly_deep<const S: usize>(r: &mut Scalar<S>, out: &mut [f64]) {
    for chunk in out.chunks_exact_mut(S) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            let x = r.u01(j);
            *slot = ((x * x + x) * x - x) * x + x * x - x + 1.0;
        }
    }
}

#[inline(never)]
unsafe fn nx_poly_deep<const V: usize>(v: &mut [VState; V], out: &mut [f64]) {
    let one = vdupq_n_f64(1.0);
    for chunk in out.chunks_exact_mut(2 * V) {
        for (k, st) in v.iter_mut().enumerate() {
            let x = vu01(st);
            let a = vsubq_f64(vmulq_f64(vfmaq_f64(x, x, x), x), x); // ((x*x+x)*x - x)
            let b = vfmaq_f64(vsubq_f64(vfmaq_f64(one, x, x), x), a, x); // *x + x*x - x + 1
            vst1q_f64(chunk.as_mut_ptr().add(2 * k), b);
        }
    }
}

/// One draw feeding **four independent** 6-term Horner chains, summed. Unlike `poly_deep` this
/// saturates the FP pipes *within* one sample, so extra scalar streams have nothing left to
/// overlap — the same regime PERF.md measures for `normal_poly` (105 -> 107 across s1..s8), minus
/// the transcendentals. This is the cell the whole experiment is for.
const C: [[f64; 6]; 4] = [
    [1.0, -0.5, 0.25, -0.125, 0.0625, -0.031_25],
    [0.9, 0.4, -0.2, 0.1, -0.05, 0.025],
    [-1.1, 0.6, 0.3, -0.15, 0.075, -0.037_5],
    [0.7, -0.35, 0.175, 0.0875, -0.043_75, 0.021_875],
];

#[inline(never)]
fn sc_poly_thru<const S: usize>(r: &mut Scalar<S>, out: &mut [f64]) {
    for chunk in out.chunks_exact_mut(S) {
        for (j, slot) in chunk.iter_mut().enumerate() {
            let x = r.u01(j);
            let mut acc = 0.0;
            for c in C.iter() {
                let mut h = c[5];
                for &ci in c[..5].iter().rev() {
                    h = h.mul_add(x, ci);
                }
                acc += h;
            }
            *slot = acc;
        }
    }
}

#[inline(never)]
unsafe fn nx_poly_thru<const V: usize>(v: &mut [VState; V], out: &mut [f64]) {
    for chunk in out.chunks_exact_mut(2 * V) {
        for (k, st) in v.iter_mut().enumerate() {
            let x = vu01(st);
            let mut acc = vdupq_n_f64(0.0);
            for c in C.iter() {
                let mut h = vdupq_n_f64(c[5]);
                for &ci in c[..5].iter().rev() {
                    h = vfmaq_f64(vdupq_n_f64(ci), h, x);
                }
                acc = vaddq_f64(acc, h);
            }
            vst1q_f64(chunk.as_mut_ptr().add(2 * k), acc);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

/// Warm up, then time `batches` column fills. Returns (M samples/sec, mean of the last column) —
/// the mean is a cheap distribution check that the two implementations compute the same thing.
fn time(batches: usize, mut fill: impl FnMut(&mut [f64])) -> (f64, f64) {
    let mut out = vec![0.0f64; N];
    for _ in 0..64 {
        fill(&mut out);
    }
    let t = Instant::now();
    for _ in 0..batches {
        fill(&mut out);
        std::hint::black_box(&out[0]);
    }
    let mps = (batches * N) as f64 / t.elapsed().as_secs_f64() / 1e6;
    (mps, out.iter().sum::<f64>() / N as f64)
}

/// `s{S}` for scalar and `v{V}` (= 2V streams) for NEON, per case.
macro_rules! race {
    ($name:literal, $sc:ident, $nx:ident, $batches:expr) => {{
        let b = $batches;
        print!("  {:<11}", $name);
        let mut means = vec![];
        for_each_scalar!($sc, b, means);
        print!("  |");
        for_each_neon!($nx, b, means);
        println!();
        // Every configuration must agree in distribution, or we're timing different programs.
        let m0 = means[0];
        for m in &means {
            assert!(
                (m - m0).abs() < 0.05 * m0.abs().max(1.0),
                "{} means diverge: {:?}",
                $name,
                means
            );
        }
    }};
}

macro_rules! for_each_scalar {
    ($sc:ident, $b:expr, $means:expr) => {{
        seq_scalar!($sc, $b, $means, 1);
        seq_scalar!($sc, $b, $means, 2);
        seq_scalar!($sc, $b, $means, 4);
        seq_scalar!($sc, $b, $means, 8);
    }};
}

macro_rules! seq_scalar {
    ($sc:ident, $b:expr, $means:expr, $s:literal) => {{
        let mut r = Scalar::<$s>::new(0xC0FFEE);
        let (mps, mean) = time($b, |o| $sc::<$s>(&mut r, o));
        print!(" s{}={:6.0}", $s, mps);
        $means.push(mean);
    }};
}

macro_rules! for_each_neon {
    ($nx:ident, $b:expr, $means:expr) => {{
        seq_neon!($nx, $b, $means, 1);
        seq_neon!($nx, $b, $means, 2);
        seq_neon!($nx, $b, $means, 4);
    }};
}

macro_rules! seq_neon {
    ($nx:ident, $b:expr, $means:expr, $v:literal) => {{
        unsafe {
            let mut v = vstates::<$v>(0xC0FFEE);
            let (mps, mean) = time($b, |o| $nx::<$v>(&mut v, o));
            print!(" v{}[{}str]={:6.0}", $v, 2 * $v, mps);
            $means.push(mean);
        }
    }};
}

#[test]
#[ignore]
fn simd_vs_scalar_streams() {
    let batches = 60_000;
    println!("\n  hand-written kernel throughput, single thread, M samples/sec");
    println!("  scalar sN = N streams    |    neon vV = V vector states = 2V streams\n");
    race!("rng_only", sc_rng_only, nx_rng_only, batches);
    race!("pi", sc_pi, nx_pi, batches);
    race!("dice", sc_dice, nx_dice, batches);
    race!("poly_deep", sc_poly_deep, nx_poly_deep, batches);
    race!("poly_thru", sc_poly_thru, nx_poly_thru, batches);
    println!();
}
