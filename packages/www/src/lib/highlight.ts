// Static highlighter for Noise listings rendered outside Monaco — the compiled-document code cards
// in the playground preview, and the explainer panels on the landing page. Returns HTML with
// <span class> tokens styled by `.code-body` in global.css.
//
// A scanner rather than a chain of regex replaces: Noise has strings, backtick templates with
// `${…}` holes and block comments, so token boundaries can only be found by reading left to right.
// A `//` inside a string does not open a comment, and a keyword inside one is not a keyword.
//
// The vocabulary comes from `./noise-lang` — the same lists the Monaco Monarch tokenizer uses, so
// the editor and the rendered page cannot disagree about what a name is.
import {
  BUILTINS, CONSTANTS, DISTRIBUTIONS, KEYWORDS, MODULES, NUMBER_RE, OPERATORS, QUERIES,
} from './noise-lang';

// class names: c=comment n=namespace k=keyword d=distribution q=query f=builtin t=constant
//              o=operator s=string h=template-hole m=number
const CLASS_FOR_WORD = new Map<string, string>([
  ...KEYWORDS.map((w) => [w, 'k'] as [string, string]),
  ...DISTRIBUTIONS.map((w) => [w, 'd'] as [string, string]),
  ...QUERIES.map((w) => [w, 'q'] as [string, string]),
  ...BUILTINS.map((w) => [w, 'f'] as [string, string]),
  ...CONSTANTS.map((w) => [w, 't'] as [string, string]),
]);
const MODULE_SET = new Set(MODULES);
// Longest-first, so `==` wins over `=`, `::` over `:`, `..` over `.`.
const OPS = [...OPERATORS].sort((a, b) => b.length - a.length);
const NUMBER_AT = new RegExp('^(?:' + NUMBER_RE.source + ')');
const IDENT_AT = /^[A-Za-z_]\w*/;

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const span = (cls: string, text: string) => (text ? `<span class="${cls}">${esc(text)}</span>` : '');

/** Carried between lines: block comments, fenced templates and frontmatter all span lines. */
type Mode = 'code' | 'block-comment' | 'fenced' | 'frontmatter';
export interface HlState {
  mode: Mode;
  /** Line index — `---` opens frontmatter and `#!` is a shebang only at index 0. */
  line: number;
}
export const initialState = (): HlState => ({ mode: 'code', line: 0 });

/** Ordinary Noise code, with no line-spanning construct left in it. */
function tokens(src: string): string {
  let out = '';
  let i = 0;
  let afterPath = false; // previous token was `::`, so the next name is a module member
  while (i < src.length) {
    const rest = src.slice(i);

    const ws = /^\s+/.exec(rest);
    if (ws) {
      out += esc(ws[0]);
      i += ws[0].length;
      continue;
    }

    const ident = IDENT_AT.exec(rest);
    if (ident) {
      const word = ident[0];
      const after = rest.slice(word.length);
      const isPath = /^\s*::/.test(after);
      // A named argument (`min: 1`) is a label, not a reference — `min` and `max` are also vec
      // builtins, and colouring them as such here would be a lie about what the name resolves to.
      const isNamedArg = !isPath && /^\s*:/.test(after);
      // `plot::histogram` and `input::real` are module members with no standalone entry — the `::`
      // is what identifies them, so anything following a path separator reads as a builtin.
      const cls = isPath && MODULE_SET.has(word)
        ? 'n'
        : isNamedArg
          ? ''
          : CLASS_FOR_WORD.get(word) ?? (afterPath ? 'f' : '');
      out += cls ? span(cls, word) : esc(word);
      i += word.length;
      afterPath = false;
      continue;
    }

    const num = NUMBER_AT.exec(rest);
    if (num) {
      out += span('m', num[0]);
      i += num[0].length;
      afterPath = false;
      continue;
    }

    const op = OPS.find((o) => rest.startsWith(o));
    if (op) {
      out += span('o', op);
      i += op.length;
      afterPath = op === '::';
      continue;
    }

    out += esc(src[i]);
    i += 1;
    afterPath = false;
  }
  return out;
}

/** Template body: literal text, except `${…}` holes, which are live Noise expressions. */
function template(body: string): string {
  let out = '';
  let i = 0;
  while (i < body.length) {
    const hole = body.indexOf('${', i);
    if (hole < 0) return out + span('s', body.slice(i));
    out += span('s', body.slice(i, hole));
    const end = body.indexOf('}', hole);
    if (end < 0) return out + span('h', body.slice(hole));
    out += span('h', '${') + tokens(body.slice(hole + 2, end)) + span('h', '}');
    i = end + 1;
  }
  return out;
}

/** One line of code, splitting off strings, templates and comments before the rest is tokenized. */
function scan(line: string): { html: string; openBlockComment: boolean } {
  let out = '';
  let i = 0;
  let plain = 0; // start of the pending run of ordinary code
  const flush = (upto: number) => {
    out += tokens(line.slice(plain, upto));
  };

  while (i < line.length) {
    const two = line.slice(i, i + 2);
    if (two === '//') {
      flush(i);
      return { html: out + span('c', line.slice(i)), openBlockComment: false };
    }
    if (two === '/*') {
      flush(i);
      const end = line.indexOf('*/', i + 2);
      if (end < 0) return { html: out + span('c', line.slice(i)), openBlockComment: true };
      out += span('c', line.slice(i, end + 2));
      i = end + 2;
      plain = i;
      continue;
    }
    if (line[i] === '"') {
      flush(i);
      const end = line.indexOf('"', i + 1); // Noise strings have no escape sequences
      const stop = end < 0 ? line.length : end + 1;
      out += span('s', line.slice(i, stop));
      i = stop;
      plain = i;
      continue;
    }
    if (line[i] === '`') {
      flush(i);
      const end = line.indexOf('`', i + 1);
      const body = end < 0 ? line.slice(i + 1) : line.slice(i + 1, end);
      out += span('s', '`') + template(body) + (end < 0 ? '' : span('s', '`'));
      i = end < 0 ? line.length : end + 1;
      plain = i;
      continue;
    }
    i += 1;
  }
  flush(line.length);
  return { html: out, openBlockComment: false };
}

/** Highlight one line, threading the multi-line state; returns the state for the next line. */
export function hlLine(
  line: string,
  state: HlState = initialState(),
): { html: string; state: HlState } {
  const at = state.line;
  const next = (mode: Mode): HlState => ({ mode, line: at + 1 });

  if (state.mode === 'block-comment') {
    const end = line.indexOf('*/');
    if (end < 0) return { html: span('c', line), state: next('block-comment') };
    const tail = scan(line.slice(end + 2));
    return {
      html: span('c', line.slice(0, end + 2)) + tail.html,
      state: next(tail.openBlockComment ? 'block-comment' : 'code'),
    };
  }
  if (state.mode === 'frontmatter') {
    return { html: span('c', line), state: next(/^---\s*$/.test(line) ? 'code' : 'frontmatter') };
  }
  if (state.mode === 'fenced') {
    if (/^\s*```\s*$/.test(line)) return { html: span('s', line), state: next('code') };
    return { html: template(line), state: next('fenced') };
  }

  // `---` opens the YAML-ish frontmatter block, but only as the very first line.
  if (at === 0 && /^---\s*$/.test(line)) return { html: span('c', line), state: next('frontmatter') };
  if (at === 0 && line.startsWith('#!')) return { html: span('c', line), state: next('code') };
  // A ```tag fence opens a template block running until a bare ```.
  if (/^\s*```/.test(line)) return { html: span('s', line), state: next('fenced') };

  const r = scan(line);
  return { html: r.html, state: next(r.openBlockComment ? 'block-comment' : 'code') };
}

/** Highlight a whole program into one HTML string (newline-joined). */
export function LISTING_HL(src: string): string {
  return LISTING_LINES(src).join('\n');
}

/** Highlight a program into an array of per-line HTML strings (for line-addressable panels). */
export function LISTING_LINES(src: string): string[] {
  let state = initialState();
  return src.split('\n').map((line) => {
    const r = hlLine(line, state);
    state = r.state;
    return r.html;
  });
}
