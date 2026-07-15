// @ts-check
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';

// Reuse the editor's TextMate grammar so the /docs page (which renders the SKILL.md guide)
// highlights ```noise fences with the same rules VS Code uses. Single grammar, two consumers.
const noiseGrammar = {
  ...JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../editors/vscode-noise/syntaxes/noise.tmLanguage.json', import.meta.url)),
      'utf8',
    ),
  ),
  name: 'noise',
};

// Cross-origin isolation for the dev + preview servers. Production sets these in netlify.toml; the
// local servers don't, so without this the browser withholds `SharedArrayBuffer` and the engine runs
// its single-threaded build — and the WebGPU host (which needs the same isolation as the SAB bridge)
// never activates. A tiny dev-only middleware mirrors the two production headers so `pnpm dev`
// exercises the same threaded + GPU path a deploy does. See PLAN-WEBGPU G3.
const crossOriginIsolation = {
  name: 'noise-cross-origin-isolation',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      next();
    });
  },
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      next();
    });
  },
};

// Static site. The playground (Monaco + the WASM engine) runs entirely client-side, so no
// adapter/SSR is needed. `vite.server.fs.allow` lets us `?raw`-import the .noise example files
// that live in the repo's top-level `examples/` directory (outside this site root).
export default defineConfig({
  site: 'https://noise-lang.dev',
  markdown: {
    // /docs renders the agent skill (.claude/skills/noise-lang/SKILL.md) as Astro markdown.
    shikiConfig: { theme: 'github-light', langs: [noiseGrammar] },
  },
  vite: {
    plugins: [crossOriginIsolation],
    server: {
      fs: { allow: ['..'] },
    },
    worker: {
      format: 'es',
    },
    // @noiselang/core ships the engine .wasm and references it via `new URL(..., import.meta.url)`.
    // Vite's dep pre-bundler (esbuild) would rewrite that reference and lose the asset, so exclude
    // the package from optimization — Vite then processes its ESM directly and emits the .wasm as a
    // fingerprinted asset of this build.
    optimizeDeps: {
      exclude: ['@noiselang/core'],
    },
  },
});
