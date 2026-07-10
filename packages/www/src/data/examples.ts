// The example gallery. Code AND metadata are pulled live from the repo's top-level `examples/*.noise`
// files: each file OWNS its title / abstract / tags (engine first-class) plus blurb / category /
// analytic (gallery-only, nested under `extra:`) in a `---`-fenced frontmatter block (PLAN-LITERATE
// §D1). We parse that block at build time with a real YAML parser so
// the site never drifts from — and never has to duplicate — what the file itself declares. The engine's
// `meta()` is the runtime source of truth for the same block; this is its build-time twin for the
// static gallery.

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
  "Signals & DSP",
  "Functions & research",
] as const;

/**
 * The gallery, in pedagogical order. This is the *only* website-side curation left: which files
 * appear and in what order. Everything else (title, abstract, tags, and the `extra:` blurb /
 * category / analytic) is read from each file's frontmatter. A file not listed here (utility demos)
 * simply doesn't appear in the gallery.
 */
const galleryOrder = [
  "pi",
  "dice",
  "dice_sum",
  "coin_streak",
  "exactly_two_heads",
  "birthday",
  "monty_hall",
  "conditional_bayes",
  "advantage",
  "max_of_dice",
  "dice_bet",
  "insurance",
  "reliability",
  "irwin_hall",
  "clt_normal",
  "am_vs_fm",
  "nyquist",
  "functions",
  "qjl_scalar",
  "turboquant",
] as const;

interface FrontmatterFields {
  title?: string;
  abstract?: string;
  tags?: string[];
  /** Host-specific metadata lives under `extra:` — only title/abstract/tags/knobs are first-class
   *  in the engine (see crates/noise-core/src/frontmatter.rs). `blurb`/`category`/`analytic` are the
   *  gallery's own fields, so the site reads them from here. */
  extra?: { blurb?: string; category?: string; analytic?: string };
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

export const examples: Example[] = galleryOrder
  .filter((id) => codeByName[id])
  .map((id) => {
    const code = codeByName[id];
    const fm = parseFrontmatter(code);
    const extra = fm.extra ?? {};
    return {
      id,
      title: fm.title ?? id,
      blurb: extra.blurb ?? "",
      abstract: fm.abstract,
      analytic: extra.analytic,
      category: extra.category ?? "Basics",
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
