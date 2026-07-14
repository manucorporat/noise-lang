//! PLAN-PREGPU C0 — certification battery for the counter-RNG swap.
//!
//! Everything here mirrors how the engine will consume the generator (Track C keying,
//! Track B f32 conversion), not how RNG batteries usually consume one. Criteria are
//! frozen in README.md; this binary just measures and gates.

use std::env;
use std::io::{self, Write};
use std::time::Instant;

/// Deterministic run seed ("noise"). The keys below derive from it exactly as the
/// engine will: `key = SplitMix64(seed)` split into two u32s.
const SEED: u64 = 0x6E6F_6973_65;

/// |z| at or above this fails: global two-sided alpha 1e-3 Bonferroni over m = 7 tests.
const Z_CRIT: f64 = 3.81;
/// Chi-square tail-probability gate (either tail): alpha/2 per side of the same budget.
const P_CRIT: f64 = 7.1e-5;
/// Avalanche band around 0.5, per frozen criterion 1.
const AVALANCHE_BAND: f64 = 0.01;
/// Trials per input bit for avalanche.
const AVALANCHE_N: usize = 1 << 17;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

#[inline(always)]
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// pcg4d (Jarzynski–Olano, JCGT 2020). The Track C candidate.
#[inline(always)]
fn pcg4d(mut v: [u32; 4]) -> [u32; 4] {
    for x in v.iter_mut() {
        *x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    v
}

/// squares32 (Widynski, arXiv:2004.06278). The certified in-harness reference and
/// contingency generator. Key from Widynski's key-construction procedure.
#[inline(always)]
fn squares32(ctr: u64, key: u64) -> u32 {
    let mut x = ctr.wrapping_mul(key);
    let y = x;
    let z = y.wrapping_add(key);
    x = x.wrapping_mul(x).wrapping_add(y);
    x = (x >> 32) | (x << 32);
    x = x.wrapping_mul(x).wrapping_add(z);
    x = (x >> 32) | (x << 32);
    x = x.wrapping_mul(x).wrapping_add(y);
    x = (x >> 32) | (x << 32);
    (x.wrapping_mul(x).wrapping_add(z) >> 32) as u32
}

// Constructed per Widynski's key rules (arXiv:2004.06278 key utility): each 8-hex-digit
// half uses distinct non-zero digits, least-significant digit odd. The first harness key
// (0xc58efd154ce32f6d) violated this — nibble `5` repeats in the upper half — and that
// out-of-spec key, not the generator, failed PractRand at 256 GB (RESULTS.md defect #3).
const SQUARES_KEY: u64 = 0xf7c3_b1a9_e6d5_c8b3;

/// A generator under test, consumed the way the engine's kernels will: four u32 words
/// per (lane, source) cell.
trait Gen {
    fn name(&self) -> &'static str;
    /// Output words for one draw cell. Word 0 is what an f32 uniform consumes.
    fn words(&self, lane: u32, source: u32) -> [u32; 4];
    /// Raw 128-bit-in hash for the avalanche test.
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4];
    /// How many of the output words carry entropy (pcg4d: 4; squares32: 1).
    fn out_words(&self) -> usize;
}

/// Track C keying: (key_lo, key_hi, global_lane, source_offset).
struct Pcg {
    k0: u32,
    k1: u32,
}

impl Pcg {
    fn new(seed: u64) -> Self {
        let k = splitmix(seed);
        Pcg { k0: k as u32, k1: (k >> 32) as u32 }
    }
}

impl Gen for Pcg {
    fn name(&self) -> &'static str {
        "pcg4d"
    }
    #[inline(always)]
    fn words(&self, lane: u32, source: u32) -> [u32; 4] {
        pcg4d([self.k0, self.k1, lane, source])
    }
    #[inline(always)]
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4] {
        pcg4d(inp)
    }
    fn out_words(&self) -> usize {
        4
    }
}

/// pcg4d with one extra final xorshift per word (+4 ops): the minimal patch for the
/// carries-only-go-up / single-downward-mix structure the avalanche test exposes.
#[inline(always)]
fn pcg4d_xs(v: [u32; 4]) -> [u32; 4] {
    let mut v = pcg4d(v);
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    v
}

struct PcgXs {
    k0: u32,
    k1: u32,
}

impl PcgXs {
    fn new(seed: u64) -> Self {
        let k = splitmix(seed);
        PcgXs { k0: k as u32, k1: (k >> 32) as u32 }
    }
}

impl Gen for PcgXs {
    fn name(&self) -> &'static str {
        "pcg4d-xs"
    }
    #[inline(always)]
    fn words(&self, lane: u32, source: u32) -> [u32; 4] {
        pcg4d_xs([self.k0, self.k1, lane, source])
    }
    #[inline(always)]
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4] {
        pcg4d_xs(inp)
    }
    fn out_words(&self) -> usize {
        4
    }
}

/// pcg4d-3r with fmix32 applied per output word on top (+20 ops): full cross-word mixing
/// from the three product rounds, then a certified-grade per-word finalizer to scrub the
/// pcg family's weak low bits.
#[inline(always)]
fn pcg4d_3rf(v: [u32; 4]) -> [u32; 4] {
    let v = pcg4d_3r(v);
    [fmix32(v[0]), fmix32(v[1]), fmix32(v[2]), fmix32(v[3])]
}

struct Pcg3rf {
    k0: u32,
    k1: u32,
}

impl Pcg3rf {
    fn new(seed: u64) -> Self {
        let k = splitmix(seed);
        Pcg3rf { k0: k as u32, k1: (k >> 32) as u32 }
    }
}

impl Gen for Pcg3rf {
    fn name(&self) -> &'static str {
        "pcg4d-3rf"
    }
    #[inline(always)]
    fn words(&self, lane: u32, source: u32) -> [u32; 4] {
        pcg4d_3rf([self.k0, self.k1, lane, source])
    }
    #[inline(always)]
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4] {
        pcg4d_3rf(inp)
    }
    fn out_words(&self) -> usize {
        4
    }
}

/// pcg4d with a third mixing round (xorshift + dependent-product round appended, +12 ops):
/// keeps all four output words, gives word 0 a second full round of cross-mixing.
#[inline(always)]
fn pcg4d_3r(v: [u32; 4]) -> [u32; 4] {
    let mut v = pcg4d(v);
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    v
}

struct Pcg3r {
    k0: u32,
    k1: u32,
}

impl Pcg3r {
    fn new(seed: u64) -> Self {
        let k = splitmix(seed);
        Pcg3r { k0: k as u32, k1: (k >> 32) as u32 }
    }
}

impl Gen for Pcg3r {
    fn name(&self) -> &'static str {
        "pcg4d-3r"
    }
    #[inline(always)]
    fn words(&self, lane: u32, source: u32) -> [u32; 4] {
        pcg4d_3r([self.k0, self.k1, lane, source])
    }
    #[inline(always)]
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4] {
        pcg4d_3r(inp)
    }
    fn out_words(&self) -> usize {
        4
    }
}

/// MurmurHash3's 32-bit finalizer: full avalanche for a 32->32 bijection (5 ops).
#[inline(always)]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

/// pcg4d with fmix32 applied per output word (+20 ops): pcg4d's 128-bit cross-mixing
/// carries every input bit into each word, fmix32 then spreads it across the word's bits.
#[inline(always)]
fn pcg4d_f(v: [u32; 4]) -> [u32; 4] {
    let v = pcg4d(v);
    [fmix32(v[0]), fmix32(v[1]), fmix32(v[2]), fmix32(v[3])]
}

struct PcgF {
    k0: u32,
    k1: u32,
}

impl PcgF {
    fn new(seed: u64) -> Self {
        let k = splitmix(seed);
        PcgF { k0: k as u32, k1: (k >> 32) as u32 }
    }
}

impl Gen for PcgF {
    fn name(&self) -> &'static str {
        "pcg4d-f"
    }
    #[inline(always)]
    fn words(&self, lane: u32, source: u32) -> [u32; 4] {
        pcg4d_f([self.k0, self.k1, lane, source])
    }
    #[inline(always)]
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4] {
        pcg4d_f(inp)
    }
    fn out_words(&self) -> usize {
        4
    }
}

/// Squares reference: sequential counter per (lane, source, word), certified key.
struct Sq {
    key: u64,
}

impl Gen for Sq {
    fn name(&self) -> &'static str {
        "squares32"
    }
    #[inline(always)]
    fn words(&self, lane: u32, source: u32) -> [u32; 4] {
        let base = ((source as u64) << 36) | ((lane as u64) << 2);
        [
            squares32(base, self.key),
            squares32(base | 1, self.key),
            squares32(base | 2, self.key),
            squares32(base | 3, self.key),
        ]
    }
    #[inline(always)]
    fn hash_raw(&self, inp: [u32; 4]) -> [u32; 4] {
        let ctr = (inp[0] as u64) | ((inp[1] as u64) << 32);
        let key = (inp[2] as u64) | ((inp[3] as u64) << 32);
        [squares32(ctr, key), 0, 0, 0]
    }
    fn out_words(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// The engine's numeric pipeline (Track B): u32 word -> f32 uniform.
// ---------------------------------------------------------------------------

#[inline(always)]
fn unif_f32(w: u32) -> f32 {
    (w >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

// ---------------------------------------------------------------------------
// Battery
// ---------------------------------------------------------------------------

struct Verdict {
    failures: usize,
}

impl Verdict {
    fn gate(&mut self, label: &str, detail: String, ok: bool) {
        println!("  [{}] {label}: {detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.failures += 1;
        }
    }
}

/// Flip-count grid for the avalanche test: realistic base points (random keys, small
/// lanes/sources so the low-counter-bit region is stressed), all 128 input bits flipped.
/// Returns `cnt[in_bit * out_bits + out_bit]` over `AVALANCHE_N` trials.
fn avalanche_grid(g: &dyn Gen) -> Vec<u32> {
    let out_bits = g.out_words() * 32;
    let mut cnt = vec![0u32; 128 * out_bits];
    let mut s = SEED;
    for trial in 0..AVALANCHE_N {
        s = splitmix(s);
        let k = s;
        // Base input shaped like real keying: random key words, small lane, small source.
        let base = [k as u32, (k >> 32) as u32, (trial & 0xFFF) as u32, (trial & 3) as u32];
        let base_out = g.hash_raw(base);
        for in_bit in 0..128 {
            let mut flipped = base;
            flipped[in_bit / 32] ^= 1 << (in_bit % 32);
            let out = g.hash_raw(flipped);
            let row = &mut cnt[in_bit * out_bits..(in_bit + 1) * out_bits];
            for w in 0..g.out_words() {
                let x = out[w] ^ base_out[w];
                for b in 0..32 {
                    row[w * 32 + b] += (x >> b) & 1;
                }
            }
        }
    }
    cnt
}

/// Criterion 1 — avalanche over the full 128-bit input.
fn avalanche(g: &dyn Gen, v: &mut Verdict) {
    let t = Instant::now();
    let out_bits = g.out_words() * 32;
    let cnt = avalanche_grid(g);
    let mut worst = 0.0f64;
    let mut worst_at = (0, 0);
    let mut outside = 0usize;
    for in_bit in 0..128 {
        for ob in 0..out_bits {
            let p = cnt[in_bit * out_bits + ob] as f64 / AVALANCHE_N as f64;
            let dev = (p - 0.5).abs();
            if dev > worst {
                worst = dev;
                worst_at = (in_bit, ob);
            }
            if dev > AVALANCHE_BAND {
                outside += 1;
            }
        }
    }
    v.gate(
        "avalanche",
        format!(
            "worst |p-0.5| = {worst:.5} at in-bit {} -> out-bit {} ({} of {} cells outside ±{AVALANCHE_BAND}) [{:.1}s]",
            worst_at.0,
            worst_at.1,
            outside,
            128 * out_bits,
            t.elapsed().as_secs_f64()
        ),
        outside == 0,
    );
}

/// Diagnostic (not a gate): where do the out-of-band cells live? Blocks the 128×N grid
/// by (input word × output word), splitting each output word into the bits the engine
/// discards (`w & 0xFF`, out-bit 0..8) vs the bits an f32 uniform consumes (`w >> 8`,
/// out-bit 8..32).
fn avalanche_map(g: &dyn Gen) {
    println!("== {} avalanche map (band ±{AVALANCHE_BAND}, N = {AVALANCHE_N}) ==", g.name());
    let out_bits = g.out_words() * 32;
    let cnt = avalanche_grid(g);
    let in_names = ["key_lo", "key_hi", "lane  ", "source"];
    println!(
        "  {:>8} | per output word: outside-band cells (worst |p-0.5|), discarded bits 0..8 / consumed bits 8..32",
        ""
    );
    for iw in 0..4 {
        let mut line = format!("  {:>8} |", in_names[iw]);
        for ow in 0..g.out_words() {
            let (mut n_lo, mut w_lo, mut n_hi, mut w_hi) = (0usize, 0f64, 0usize, 0f64);
            for ib in iw * 32..(iw + 1) * 32 {
                for ob in ow * 32..(ow + 1) * 32 {
                    let p = cnt[ib * out_bits + ob] as f64 / AVALANCHE_N as f64;
                    let dev = (p - 0.5).abs();
                    let (n, w) = if ob % 32 < 8 { (&mut n_lo, &mut w_lo) } else { (&mut n_hi, &mut w_hi) };
                    if dev > AVALANCHE_BAND {
                        *n += 1;
                    }
                    if dev > *w {
                        *w = dev;
                    }
                }
            }
            line += &format!("  w{ow}: {n_lo:>3} ({w_lo:.3}) / {n_hi:>3} ({w_hi:.3})");
        }
        println!("{line}");
    }
    // The verdict that matters for the engine: restrict to consumed out-bits and to the
    // input bits real runs exercise (all 64 key bits; lane bits 0..27 — 134M draws;
    // source bits 0..8).
    let realistic_in = |ib: usize| ib < 64 || (64..92).contains(&ib) || (96..105).contains(&ib);
    let (mut outside, mut worst, mut worst_at) = (0usize, 0f64, (0, 0));
    let mut total = 0usize;
    for ib in (0..128).filter(|&ib| realistic_in(ib)) {
        for ob in (0..out_bits).filter(|ob| ob % 32 >= 8) {
            total += 1;
            let p = cnt[ib * out_bits + ob] as f64 / AVALANCHE_N as f64;
            let dev = (p - 0.5).abs();
            if dev > AVALANCHE_BAND {
                outside += 1;
            }
            if dev > worst {
                worst = dev;
                worst_at = (ib, ob);
            }
        }
    }
    println!(
        "  usage-restricted (consumed out-bits, realistic in-bits): {outside} of {total} outside, worst {worst:.5} at in-bit {} -> out-bit {}",
        worst_at.0, worst_at.1
    );
}

/// Criteria 2–4 — Pearson correlation between the uniform streams whose independence
/// the engine's joint queries assume.
fn correlations(g: &dyn Gen, v: &mut Verdict) {
    let n = 1usize << 23;
    let mut run = |label: &str, f: &dyn Fn(u32) -> (f32, f32)| {
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for l in 0..n as u32 {
            let (x, y) = f(l);
            let (x, y) = (x as f64, y as f64);
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        let nf = n as f64;
        let cov = sxy / nf - (sx / nf) * (sy / nf);
        let vx = sxx / nf - (sx / nf) * (sx / nf);
        let vy = syy / nf - (sy / nf) * (sy / nf);
        let r = cov / (vx * vy).sqrt();
        let z = r * nf.sqrt();
        v.gate(label, format!("r = {r:+.2e}, z = {z:+.2}"), z.abs() < Z_CRIT);
    };
    run("corr lane vs lane+1", &|l| {
        (unif_f32(g.words(l, 0)[0]), unif_f32(g.words(l + 1, 0)[0]))
    });
    run("corr lane vs lane+1024", &|l| {
        (unif_f32(g.words(l, 0)[0]), unif_f32(g.words(l + 1024, 0)[0]))
    });
    run("corr source 0 vs source 1", &|l| {
        (unif_f32(g.words(l, 0)[0]), unif_f32(g.words(l, 1)[0]))
    });
}

/// Criterion 5 — π/4 through the exact cross-source pairing joint queries use.
fn pi_quarter(g: &dyn Gen, v: &mut Verdict) {
    let t = Instant::now();
    let n = 1usize << 28;
    let mut hits = 0u64;
    for l in 0..n as u32 {
        let u = unif_f32(g.words(l, 0)[0]);
        let w = unif_f32(g.words(l, 1)[0]);
        if u * u + w * w < 1.0 {
            hits += 1;
        }
    }
    let p = std::f64::consts::FRAC_PI_4;
    let phat = hits as f64 / n as f64;
    let z = (phat - p) / (p * (1.0 - p) / n as f64).sqrt();
    v.gate(
        "pi/4 cross-source",
        format!("p̂ = {phat:.7} (π/4 = {p:.7}), z = {z:+.2} [{:.1}s]", t.elapsed().as_secs_f64()),
        z.abs() < Z_CRIT,
    );
}

/// Abramowitz–Stegun 7.1.26 erfc (|ε| ≤ 1.5e-7 — the gate is at 7.1e-5, far above it).
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// Upper-tail probability of chi-square with 5 degrees of freedom (closed form).
fn chi2_df5_tail(x: f64) -> f64 {
    erfc((x / 2.0).sqrt()) + (-x / 2.0).exp() * (2.0 * x / std::f64::consts::PI).sqrt() * (1.0 + x / 3.0)
}

/// Criterion 6 — die faces through the engine's `floor(u * 6)` integer-draw path.
fn die_chi2(g: &dyn Gen, v: &mut Verdict) {
    let t = Instant::now();
    let n = 100_000_000usize;
    let mut counts = [0u64; 6];
    for l in 0..n as u32 {
        let u = unif_f32(g.words(l, 2)[0]);
        counts[(u * 6.0) as usize] += 1;
    }
    let e = n as f64 / 6.0;
    let chi2: f64 = counts.iter().map(|&o| (o as f64 - e).powi(2) / e).sum();
    let q = chi2_df5_tail(chi2);
    v.gate(
        "die chi-square (df=5)",
        format!("chi2 = {chi2:.2}, upper-tail p = {q:.4} [{:.1}s]", t.elapsed().as_secs_f64()),
        q > P_CRIT && (1.0 - q) > P_CRIT,
    );
}

/// Criterion 7 — Box–Muller pair from a single hash (words 0,1), f32 transcendentals,
/// f64 aggregation: skewness and excess kurtosis of the resulting normals.
fn normality(g: &dyn Gen, v: &mut Verdict) {
    let t = Instant::now();
    let n_hash = 1usize << 26;
    let (mut s1, mut s2, mut s3, mut s4) = (0f64, 0f64, 0f64, 0f64);
    for l in 0..n_hash as u32 {
        let w = g.words(l, 3);
        // Offset keeps u1 away from 0 exactly as the engine's fill will.
        let u1 = ((w[0] >> 8) as f32 + 0.5) * (1.0 / (1u32 << 24) as f32);
        let u2 = unif_f32(w[1]);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        for z in [r * theta.cos(), r * theta.sin()] {
            let z = z as f64;
            s1 += z;
            s2 += z * z;
            s3 += z * z * z;
            s4 += z * z * z * z;
        }
    }
    let n = (2 * n_hash) as f64;
    let mu = s1 / n;
    let m2 = s2 / n - mu * mu;
    let m3 = s3 / n - 3.0 * mu * s2 / n + 2.0 * mu.powi(3);
    let m4 = s4 / n - 4.0 * mu * s3 / n + 6.0 * mu * mu * s2 / n - 3.0 * mu.powi(4);
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2) - 3.0;
    let z_skew = skew / (6.0 / n).sqrt();
    let z_kurt = kurt / (24.0 / n).sqrt();
    v.gate(
        "Box-Muller skewness",
        format!("g1 = {skew:+.2e}, z = {z_skew:+.2} [{:.1}s]", t.elapsed().as_secs_f64()),
        z_skew.abs() < Z_CRIT,
    );
    v.gate(
        "Box-Muller excess kurtosis",
        format!("g2 = {kurt:+.2e}, z = {z_kurt:+.2}"),
        z_kurt.abs() < Z_CRIT,
    );
}

fn battery(g: &dyn Gen) -> usize {
    println!("== {} ==", g.name());
    let mut v = Verdict { failures: 0 };
    avalanche(g, &mut v);
    correlations(g, &mut v);
    pi_quarter(g, &mut v);
    die_chi2(g, &mut v);
    normality(g, &mut v);
    println!(
        "  => {}: {}",
        g.name(),
        if v.failures == 0 { "ALL PASS".to_string() } else { format!("{} FAILURE(S)", v.failures) }
    );
    v.failures
}

/// Criterion 8 feed — words in kernel-consumption order: batches of 1024 lanes,
/// 4 sources per batch, word 0 per hash (or all 4 with `words4`). With `consumed`,
/// emit only the 24 bits the engine consumes (`w >> 8`, exactly 3 LE bytes per word) —
/// the stream that matches the actual numeric contract.
fn stream(g: &dyn Gen, all_words: bool, consumed: bool, skip_w0: bool) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());
    // All four cell words are streamable for every generator (for squares32 they are four
    // sequential counters — `out_words()` is only about the raw avalanche hash's width).
    let n_words = if all_words { 4 } else { 1 };
    let first_word = if skip_w0 { 1 } else { 0 };
    // 2^22 batches of 1024 lanes exhaust one source block's 2^32-lane space; then move to
    // the next block of 4 sources. Real usage never reuses a (lane, source) cell, so the
    // stream must not either — the first harness version wrapped `lane0` at 2^32 lanes and
    // repeated the stream verbatim past ~206 GB, voiding every deep verdict beyond it.
    const BATCHES_PER_BLOCK: u64 = 1 << 22;
    let mut batch = 0u64;
    loop {
        let block = (batch / BATCHES_PER_BLOCK) as u32;
        let lane0 = ((batch % BATCHES_PER_BLOCK) * 1024) as u32;
        for s in 0..4u32 {
            let source = block.wrapping_mul(4).wrapping_add(s);
            for l in 0..1024u32 {
                let w = g.words(lane0.wrapping_add(l), source);
                for &word in &w[first_word..n_words.max(first_word + 1)] {
                    if consumed {
                        out.write_all(&(word >> 8).to_le_bytes()[..3])?;
                    } else {
                        out.write_all(&word.to_le_bytes())?;
                    }
                }
            }
        }
        batch += 1;
    }
}

/// Per-input-bit detail for one input word: worst consumed-bit deviation per in-bit.
fn avalanche_row(g: &dyn Gen, in_word: usize) {
    let out_bits = g.out_words() * 32;
    let cnt = avalanche_grid(g);
    println!("== {} in-word {in_word}, worst consumed |p-0.5| per in-bit ==", g.name());
    for ib in in_word * 32..(in_word + 1) * 32 {
        let mut worst = 0.0f64;
        for ob in (0..out_bits).filter(|ob| ob % 32 >= 8) {
            let p = cnt[ib * out_bits + ob] as f64 / AVALANCHE_N as f64;
            worst = worst.max((p - 0.5).abs());
        }
        let flag = if worst > AVALANCHE_BAND { "  <-- outside" } else { "" };
        println!("  bit {:>3}: {worst:.5}{flag}", ib);
    }
}

/// Single-thread throughput per generator, hashing sequential lanes (M words/s).
fn bench(names: &[&str]) {
    const N: u32 = 1 << 26;
    for name in names {
        let g = make_gen(name);
        let t = Instant::now();
        let mut acc = 0u32;
        for l in 0..N {
            let w = g.words(l, 0);
            acc ^= w[0] ^ w[1] ^ w[2] ^ w[3];
        }
        std::hint::black_box(acc);
        let el = t.elapsed().as_secs_f64();
        println!(
            "  {name:<10} {:>7.0} M hashes/s = {:>7.0} M u32 words/s",
            N as f64 / el / 1e6,
            N as f64 * g.out_words() as f64 / el / 1e6
        );
    }
}

fn make_gen(name: &str) -> Box<dyn Gen> {
    match name {
        "pcg4d" => Box::new(Pcg::new(SEED)),
        "pcg4d-xs" => Box::new(PcgXs::new(SEED)),
        "pcg4d-f" => Box::new(PcgF::new(SEED)),
        "pcg4d-3r" => Box::new(Pcg3r::new(SEED)),
        "pcg4d-3rf" => Box::new(Pcg3rf::new(SEED)),
        "squares" | "squares32" => Box::new(Sq { key: SQUARES_KEY }),
        other => {
            eprintln!("unknown generator '{other}' (want pcg4d | pcg4d-xs | pcg4d-f | squares)");
            std::process::exit(2);
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("all") => {
            let gens: Vec<&str> = match args.get(1) {
                Some(g) => vec![g.as_str()],
                None => vec!["pcg4d", "squares"],
            };
            let mut total = 0;
            for name in gens {
                total += battery(&*make_gen(name));
            }
            std::process::exit(if total == 0 { 0 } else { 1 });
        }
        Some("avrow") => {
            let g = make_gen(args.get(1).map(String::as_str).unwrap_or("squares"));
            let iw: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
            avalanche_row(&*g, iw);
            Ok(())
        }
        Some("bench") => {
            bench(&["pcg4d", "pcg4d-xs", "pcg4d-f", "pcg4d-3r", "pcg4d-3rf", "squares"]);
            Ok(())
        }
        Some("avmap") => {
            let names: Vec<&str> = match args.get(1) {
                Some(g) => vec![g.as_str()],
                None => vec!["pcg4d", "pcg4d-xs", "squares"],
            };
            for name in names {
                avalanche_map(&*make_gen(name));
            }
            Ok(())
        }
        Some("stream") => {
            let g = make_gen(args.get(1).map(String::as_str).unwrap_or("pcg4d"));
            let all_words = args.iter().any(|a| a == "words4");
            let consumed = args.iter().any(|a| a == "consumed");
            let skip_w0 = args.iter().any(|a| a == "skipw0");
            // Broken pipe when RNG_test stops reading is a clean exit, not an error.
            match stream(&*g, all_words, consumed, skip_w0) {
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
                r => r,
            }
        }
        _ => {
            eprintln!("usage: rng-cert all [pcg4d|squares] | rng-cert stream <pcg4d|squares> [words4]");
            std::process::exit(2);
        }
    }
}
