// Tiny static highlighter for Noise code listings rendered outside Monaco (the explainer demos).
// Runs at build time and returns HTML with <span class> tokens styled by the component. Class
// names match the Monaco-paper palette: k=keyword, d=distribution, q=query, o=operator, c=comment.
const MODULES = /\b(rand|vec|math|signal|builtin)::/g;
const KEYWORDS = /\b(use|if|else|for|in|continue|true|false)\b/g;
const DISTS = /\b(unif_int|unif|normal_int|normal_complex|normal|bernoulli|poisson|geometric|exponential_int|exponential|categorical|rotation|permutation)\b/g;
const QUERIES = /\b(P|E|Var|Q|Print|Len)\b/g;

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Highlight a single line of Noise into token-span HTML. */
export function hlLine(line: string): string {
  const ci = line.indexOf('//');
  const codePart = ci >= 0 ? line.slice(0, ci) : line;
  const comment = ci >= 0 ? line.slice(ci) : '';
  const hi = esc(codePart)
    .replace(MODULES, '<span class="n">$1::</span>')
    .replace(KEYWORDS, '<span class="k">$1</span>')
    .replace(DISTS, '<span class="d">$1</span>')
    .replace(QUERIES, '<span class="q">$1</span>')
    .replace(/(~|@)/g, '<span class="o">$1</span>');
  return hi + (comment ? `<span class="c">${esc(comment)}</span>` : '');
}

/** Highlight a whole program into one HTML string (newline-joined). */
export function LISTING_HL(code: string): string {
  return code.split('\n').map(hlLine).join('\n');
}

/** Highlight a program into an array of per-line HTML strings (for line-addressable panels). */
export function LISTING_LINES(code: string): string[] {
  return code.split('\n').map(hlLine);
}
