// A tiny static server that sets COOP/COEP, so the page is cross-origin isolated and the browser
// hands it SharedArrayBuffer — the precondition for wasm threads. Without these two headers the
// threaded build cannot run at all (`crossOriginIsolated` is false and `new WebAssembly.Memory({
// shared: true })` is refused), which is exactly why the library also ships a single-threaded build.
//
//   node packages/core/bench/serve-mt.mjs   →   http://localhost:844/bench/mt.html
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url)); // packages/core
const TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
};


/**
 * Resolve a directory request to its package entry, the way a bundler or Node would.
 *
 * `wasm-bindgen-rayon`'s worker bootstrap does `import('../../..')`. From its `workerHelpers.js`
 * (three levels down, under `wasm-mt/snippets/...`) that resolves to the *package directory*
 * `wasm-mt/`, which Vite/webpack resolve through its `package.json`. A plain static file server
 * 404s on it instead, the worker module never loads, and `initThreadPool` hangs forever with no
 * error — so the threaded build looks broken when only the server is.
 */
async function resolveEntry(file) {
  const direct = await stat(file).catch(() => null);
  if (direct?.isDirectory()) {
    const pkg = JSON.parse(await readFile(join(file, 'package.json'), 'utf8'));
    return join(file, pkg.module ?? pkg.main ?? 'index.js');
  }
  return file;
}

createServer(async (req, res) => {
  // The page POSTs its results here: headless Chrome gives us no reliable way to scrape the DOM,
  // and a round-trip through the server is simpler than driving DevTools.
  if (req.method === 'POST' && req.url === '/report') {
    let body = '';
    for await (const c of req) body += c;
    console.log(body);
    res.writeHead(204).end();
    return;
  }
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(
    /^(\.\.[/\\])+/,
    '',
  );
  try {
    const body = await readFile(await resolveEntry(join(root, path)));
    res.writeHead(200, {
      'Content-Type': TYPES[extname(path)] ?? 'text/javascript',
      // The two headers that make the page cross-origin isolated.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}).listen(844, () => console.log('http://localhost:844/bench/mt.html'));
