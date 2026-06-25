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

export interface ParsedHash {
  exampleId?: string;
  code?: string;
}

/** Parse the current location hash into a share target. */
export function parseHash(hash: string): ParsedHash {
  const h = hash.replace(/^#/, '');
  if (!h) return {};
  const params = new URLSearchParams(h);
  const id = params.get('x');
  if (id) return { exampleId: id };
  const c = params.get('c');
  if (c) {
    const code = decodeCode(c);
    if (code !== null) return { code };
  }
  return {};
}

/** Build a shareable absolute URL: a clean `#x=id` when the code is an unmodified example,
 *  otherwise a `#c=` snapshot of the current source. */
export function buildShareUrl(opts: { exampleId?: string; code: string }): string {
  const base = `${location.origin}${location.pathname}`;
  if (opts.exampleId) return `${base}#x=${encodeURIComponent(opts.exampleId)}`;
  return `${base}#c=${encodeCode(opts.code)}`;
}
