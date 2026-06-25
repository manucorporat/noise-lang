// @ts-check
import { defineConfig } from 'astro/config';

// Static site. The playground (Monaco + the WASM engine) runs entirely client-side, so no
// adapter/SSR is needed. `vite.server.fs.allow` lets us `?raw`-import the .noise example files
// that live in the repo's top-level `examples/` directory (outside this site root).
export default defineConfig({
  site: 'https://noise-lang.dev',
  vite: {
    server: {
      fs: { allow: ['..'] },
    },
    worker: {
      format: 'es',
    },
  },
});
