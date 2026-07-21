// The example gallery. Code and the engine-first-class metadata (title / abstract / tags) are
// pulled live from the repo's top-level `examples/*.noise` files: each file owns them in a
// `---`-fenced frontmatter block (PLAN-LITERATE §D1), parsed at build time with a real YAML parser
// so the site never drifts from what the file itself declares. The engine's `meta()` is the runtime
// source of truth for the same block; this is its build-time twin for the static gallery.
//
// Gallery-only presentation (blurb / category / analytic) is the website's own concern and lives
// in the `gallery` list below, keeping the .noise files free of site chrome.

import yaml from "js-yaml";

const rawFiles = import.meta.glob("../../../../examples/*.noise", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const codeByName: Record<string, string> = {};
for (const [path, src] of Object.entries(rawFiles)) {
  const name = path
    .split("/")
    .pop()!
    .replace(/\.noise$/, "");
  codeByName[name] = (src as string).trimEnd() + "\n";
}

export interface Example {
  id: string;
  title: string;
  /** One-line summary shown on the gallery card. */
  blurb: string;
  /** A paper-style abstract — the prose the Preview renders under the title. */
  abstract?: string;
  /** The closed-form / analytic answer, for "you can check this" callouts. */
  analytic?: string;
  /** Grouping bucket (keeps the list scalable as examples are added). */
  category: string;
  tags: string[];
  code: string;
}

/** Display order of the gallery / dropdown sections. */
export const categories = [
  "Basics",
  "Probability",
  "Games & risk",
  "Continuous & CLT",
  "Statistics",
  "Signals & DSP",
  "Functions & research",
  "Quantum",
] as const;

/** The website-side card metadata for one gallery entry. */
interface GalleryEntry {
  id: string;
  blurb: string;
  category: (typeof categories)[number];
  analytic?: string;
}

/**
 * The gallery, in pedagogical order. This is the website-side curation: which files appear, in
 * what order, and how each is presented as a card (blurb / category / analytic). Title, abstract,
 * and tags are read from each file's frontmatter. A file not listed here (utility demos) simply
 * doesn't appear in the gallery.
 */
const gallery: GalleryEntry[] = [
  { id: "pi", blurb: "Monte Carlo π from random darts in a square.", category: "Basics", analytic: "3.14159…" },
  { id: "buffon", blurb: "Measure π by dropping a needle on a lined floor.", category: "Basics", analytic: "3.14159…" },
  { id: "dice", blurb: "Why dice need unif_int, not unif.", category: "Basics", analytic: "1/6 ≈ 0.1667" },
  { id: "dice_sum", blurb: "Independence is a separate ~ draw, not a repeated name.", category: "Basics", analytic: "1/6 ≈ 0.1667" },
  { id: "coin_streak", blurb: "Model the event, don’t multiply probabilities by hand.", category: "Probability", analytic: "1/8 = 0.125" },
  { id: "exactly_two_heads", blurb: "A tiny Binomial built from boolean events.", category: "Probability", analytic: "3/8 = 0.375" },
  { id: "birthday", blurb: "How often does a group share a birthday?", category: "Probability", analytic: "23 people → 50.7%" },
  { id: "monty_hall", blurb: "Switching doors wins 2/3 of the time.", category: "Probability", analytic: "2/3 ≈ 0.6667" },
  { id: "conditional_bayes", blurb: "P(A | B) as a ratio of probabilities.", category: "Probability", analytic: "1/3 ≈ 0.3333" },
  { id: "prisoners", blurb: "Follow the cycle and beat impossible odds.", category: "Probability", analytic: "≈ 0.3118" },
  { id: "secretary", blurb: "Reject the first n/e, then grab the next record-breaker.", category: "Probability", analytic: "1/e ≈ 0.3679" },
  { id: "advantage", blurb: "Keep the higher of two d20s.", category: "Games & risk" },
  { id: "max_of_dice", blurb: "max over random variables via a lifted if.", category: "Games & risk", analytic: "11/36 ≈ 0.3056" },
  { id: "dice_bet", blurb: "Build a payoff distribution, then ask about profit.", category: "Games & risk" },
  { id: "st_petersburg", blurb: "A game with infinite expected value that nobody would pay for.", category: "Games & risk", analytic: "E ≈ rounds (unbounded)" },
  { id: "kelly", blurb: "How much of your bankroll to stake on a winning bet.", category: "Games & risk", analytic: "f* = 0.2 → 0.0201/round" },
  { id: "insurance", blurb: "A deductible as a lifted if over a loss.", category: "Games & risk" },
  { id: "reliability", blurb: "Three parallel components, each 90% reliable.", category: "Games & risk", analytic: "0.999" },
  { id: "barrier_option", blurb: "Pricing a knock-out option by brute-force simulation.", category: "Games & risk", analytic: "vanilla = 10.4506 (Black-Scholes)" },
  { id: "irwin_hall", blurb: "The Irwin–Hall distribution by simulation.", category: "Continuous & CLT", analytic: "1/6 ≈ 0.1667" },
  { id: "clt_normal", blurb: "A normal built from twelve uniforms.", category: "Continuous & CLT" },
  { id: "beta_bernoulli", blurb: "A Bayesian update: from a flat prior to a coin's posterior bias.", category: "Statistics", analytic: "E[bias | 7/10] = 0.6667" },
  { id: "bootstrap", blurb: "Model tomorrow by resampling real history, not a bell curve.", category: "Statistics", analytic: "P(-4% day) = 1/24 ≈ 4.2%" },
  { id: "am_vs_fm", blurb: "Why FM survives noise better than AM.", category: "Signals & DSP" },
  { id: "nyquist", blurb: "Aliasing by counterexample.", category: "Signals & DSP" },
  { id: "dithering", blurb: "How noise buys resolution for a 1-bit sensor.", category: "Signals & DSP" },
  { id: "noise_colors", blurb: "White, pink, brown — sorted by how much each sample remembers the last.", category: "Signals & DSP" },
  { id: "functions", blurb: "Deterministic (=) vs stochastic (~) functions.", category: "Functions & research" },
  { id: "qjl_scalar", blurb: "One bit per projection, answering inner products it never saw (TurboQuant building block).", category: "Functions & research" },
  { id: "turboquant", blurb: "TurboQuant's extreme vector compression, rebuilt as an interactive article: the hidden 2/π bias and the sketch that fixes it.", category: "Functions & research", analytic: "bias 2/π ≈ 0.637" },
  { id: "shor_period", blurb: "How a quantum computer factors numbers — and breaks RSA.", category: "Quantum" },
];

interface FrontmatterFields {
  title?: string;
  abstract?: string;
  tags?: string[];
}

/**
 * Parse a file's `---`-fenced frontmatter into its fields. Only a fence at byte 0 counts (matching the
 * engine). YAML is parsed with `js-yaml` — the same superset-of-JSON grammar `serde_yaml` reads on the
 * Rust side, so folded abstracts and `[tag, tag]` lists round-trip identically.
 */
function parseFrontmatter(src: string): FrontmatterFields {
  if (!src.startsWith("---")) return {};
  const end = src.indexOf("\n---", 3);
  if (end === -1) return {};
  const block = src.slice(3, end);
  try {
    const doc = yaml.load(block);
    return (doc && typeof doc === "object" ? doc : {}) as FrontmatterFields;
  } catch {
    return {};
  }
}

export const examples: Example[] = gallery
  .filter((entry) => codeByName[entry.id])
  .map((entry) => {
    const code = codeByName[entry.id];
    const fm = parseFrontmatter(code);
    return {
      id: entry.id,
      title: fm.title ?? entry.id,
      blurb: entry.blurb,
      abstract: fm.abstract,
      analytic: entry.analytic,
      category: entry.category,
      tags: Array.isArray(fm.tags) ? fm.tags : [],
      code,
    };
  });

export const defaultExampleId = "pi";

/** Examples grouped by category, in `categories` order (empty groups dropped). */
export function examplesByCategory(): { category: string; items: Example[] }[] {
  return categories
    .map((category) => ({
      category,
      items: examples.filter((e) => e.category === category),
    }))
    .filter((g) => g.items.length > 0);
}
