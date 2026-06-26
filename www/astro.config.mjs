// @ts-check
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';

// Reuse the editor's TextMate grammar so the /docs page (which renders the SKILL.md guide)
// highlights ```noise fences with the same rules VS Code uses. Single grammar, two consumers.
const noiseGrammar = {
  ...JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../editors/vscode-noise/syntaxes/noise.tmLanguage.json', import.meta.url)),
      'utf8',
    ),
  ),
  name: 'noise',
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
    server: {
      fs: { allow: ['..'] },
    },
    worker: {
      format: 'es',
    },
  },
});
