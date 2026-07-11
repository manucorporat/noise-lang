// URL-fragment sharing for the playground. Two forms keep links short for the common case but
// still allow arbitrary edits:
//   #x=<id>                a named gallery example (clean, short)
//   #c=<base64url(utf8)>   an arbitrary program typed by the user

/** Encode a program into a URL-safe base64 fragment value. */
export function encodeCode(code: string): string {
  const b64 = btoa(unescape(encodeURIComponent(code)));
  return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** Decode a `#c=` fragment value back into source. Returns null if malformed. */
export function decodeCode(s: string): string | null {
  try {
    const b64 = s.replace(/-/g, '+').replace(/_/g, '/');
    return decodeURIComponent(escape(atob(b64)));
  } catch {
    return null;
  }
}

/** An input override value carried in a share link — a number or a bool. */
export type InputHashValue = number | boolean;

/** Encode non-default input overrides as `name:value,name2:value2` for the `i=` fragment param. */
export function encodeInputs(inputs: Record<string, InputHashValue>): string {
  return Object.entries(inputs)
    .map(([k, v]) => `${encodeURIComponent(k)}:${v}`)
    .join(',');
}

/** Parse an `i=` fragment value into input overrides. Unparseable pairs are skipped. */
export function parseInputs(s: string): Record<string, InputHashValue> {
  const out: Record<string, InputHashValue> = {};
  for (const pair of s.split(',')) {
    const idx = pair.indexOf(':');
    if (idx < 0) continue;
    const name = decodeURIComponent(pair.slice(0, idx));
    const raw = pair.slice(idx + 1);
    if (!name || raw === '') continue;
    out[name] = raw === 'true' ? true : raw === 'false' ? false : Number(raw);
  }
  return out;
}

export interface ParsedHash {
  exampleId?: string;
  code?: string;
  /** Input overrides carried alongside `x=`/`c=` (a tuned example is shareable as tuned). */
  inputs?: Record<string, InputHashValue>;
}

/** Parse the current location hash into a share target. */
export function parseHash(hash: string): ParsedHash {
  const h = hash.replace(/^#/, '');
  if (!h) return {};
  const params = new URLSearchParams(h);
  const raw = params.get('i');
  const inputs = raw ? parseInputs(raw) : undefined;
  const id = params.get('x');
  if (id) return { exampleId: id, inputs };
  const c = params.get('c');
  if (c) {
    const code = decodeCode(c);
    if (code !== null) return { code, inputs };
  }
  return { inputs };
}

/** Build a shareable absolute URL: a clean `#x=id` when the code is an unmodified example,
 *  otherwise a `#c=` snapshot of the current source. */
export function buildShareUrl(opts: { exampleId?: string; code: string }): string {
  const base = `${location.origin}${location.pathname}`;
  if (opts.exampleId) return `${base}#x=${encodeURIComponent(opts.exampleId)}`;
  return `${base}#c=${encodeCode(opts.code)}`;
}
