// Monaco language support for Noise: a Monarch tokenizer, a matching dark theme, and an
// auto-closing/config bundle. Registered once via `registerNoise(monaco)`.
import type * as Monaco from 'monaco-editor';

export const LANGUAGE_ID = 'noise';
export const THEME_ID = 'noise-paper';

// Module-scoped builtin names, by role, so the highlighter can color them distinctly.
const DISTRIBUTIONS = [
  'unif', 'unif_int', 'bernoulli', 'normal', 'normal_int', 'normal_complex',
  'exponential', 'exponential_int', 'poisson', 'geometric', 'categorical',
  'rotation', 'permutation',
];
const QUERIES = ['P', 'Q', 'E', 'Var']; // probability / moment / quantile queries
const BUILTINS = [
  'Print', 'Len',
  'sqrt', 'round', 'log', 'log10', 'sin', 'cos', 'atan', 'sign',
  'exp', 'abs', 'arg', 'conj', 're', 'im', 'floor', 'ceil', 'gcd', 'modpow',
  'sum', 'count', 'any', 'all', 'max', 'min', 'mean', 'dot', 'vdot', 'normsq', 'norm',
  'transpose', 'adjoint', 'normalize', 'outer', 'quantize', 'onehot', 'has_duplicate', 'mse',
  'ones', 'zeros', 'iota', 'vsign', 'scale',
  'sine', 'cosine', 'noise_white', 'noise_white_complex', 'noise_brown', 'noise_pink',
  'noise_ou', 'sample',
  'set_precision', 'set_resolution',
];
const CONSTANTS = ['pi', 'e', 'i', 'j'];
const MODULES = ['builtin', 'rand', 'math', 'vec', 'signal', 'engine', 'plot', 'stats'];

let registered = false;

export function registerNoise(monaco: typeof Monaco): void {
  if (registered) return;
  registered = true;

  monaco.languages.register({ id: LANGUAGE_ID });

  monaco.languages.setLanguageConfiguration(LANGUAGE_ID, {
    comments: { lineComment: '//', blockComment: ['/*', '*/'] },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')'],
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: '`', close: '`' },
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: '`', close: '`' },
    ],
  });

  monaco.languages.setMonarchTokensProvider(LANGUAGE_ID, {
    defaultToken: '',
    keywords: ['if', 'else', 'for', 'in', 'continue', 'use', 'true', 'false'],
    distributions: DISTRIBUTIONS,
    queries: QUERIES,
    builtins: BUILTINS,
    constants: CONSTANTS,
    modules: MODULES,
    // longest-first so `**`, `==`, `..`, `::`, `&&` win over their prefixes
    operators: [
      '**', '==', '!=', '<=', '>=', '&&', '||', '..', '::',
      '+', '-', '*', '/', '<', '>', '!', '=', '~', '@', '|',
    ],
    symbols: /[=~!<>+\-*/&|.:@]+/,
    tokenizer: {
      root: [
        // `#!` shebang (only legal on line 1; the lexer skips it as trivia)
        [/^#!.*$/, 'comment.shebang'],
        // `---` frontmatter fence (YAML-ish metadata block at the top of the file)
        [/^---\s*$/, { token: 'meta.frontmatter.delim', next: '@frontmatter' }],
        // comments: `//` line, `/* … */` block (`#` is not a comment in Noise)
        [/\/\/.*$/, 'comment'],
        [/\/\*/, { token: 'comment', next: '@blockComment' }],
        // triple-fenced template with an optional syntax tag: ```latex … ```
        [/```[^`\n]*$/, { token: 'string.template.delim', next: '@fencedTemplate' }],
        // single-backtick template: `text ${expr}`
        [/`/, { token: 'string.template.delim', next: '@inlineTemplate' }],
        // strings (no escapes in Noise yet)
        [/"[^"]*"/, 'string'],
        // numbers (float or int)
        [/\d*\.\d+/, 'number.float'],
        [/\d+\.\d*/, 'number.float'],
        [/\d+/, 'number'],
        // module path qualifier:  rand::unif
        [/[A-Za-z_]\w*(?=\s*::)/, { cases: { '@modules': 'namespace', '@default': 'identifier' } }],
        // identifiers / keywords / builtins
        [
          /[A-Za-z_]\w*/,
          {
            cases: {
              '@keywords': 'keyword',
              '@queries': 'support.function.query',
              '@distributions': 'support.function.dist',
              '@builtins': 'support.function',
              '@constants': 'constant.language',
              '@default': 'identifier',
            },
          },
        ],
        // delimiters & operators
        [/[{}()[\]]/, '@brackets'],
        [/[,;]/, 'delimiter'],
        [
          /@symbols/,
          { cases: { '@operators': 'operator', '@default': '' } },
        ],
        [/\s+/, 'white'],
      ],
      blockComment: [
        [/\*\//, { token: 'comment', next: '@pop' }],
        [/[^*]+/, 'comment'],
        [/./, 'comment'],
      ],
      // YAML-lite coloring for the metadata block; closes on a line that is exactly `---`.
      frontmatter: [
        [/^---\s*$/, { token: 'meta.frontmatter.delim', next: '@pop' }],
        [/^(\s*)([\w-]+)(\s*:)/, ['white', 'meta.frontmatter.key', 'delimiter']],
        [/#.*$/, 'comment'],
        [/.+/, 'meta.frontmatter.value'],
      ],
      fencedTemplate: [
        [/^\s*```\s*$/, { token: 'string.template.delim', next: '@pop' }],
        [/\$\{/, { token: 'string.template.hole', next: '@templateHole' }],
        [/[^$]+/, 'string.template'],
        [/./, 'string.template'],
      ],
      inlineTemplate: [
        [/`/, { token: 'string.template.delim', next: '@pop' }],
        [/\$\{/, { token: 'string.template.hole', next: '@templateHole' }],
        [/[^`$]+/, 'string.template'],
        [/./, 'string.template'],
      ],
      // A `${…}` hole holds a full Noise expression; re-use the root rules and pop on `}`.
      // (A `}` inside the hole — a block expression — pops early; templates rarely hold blocks.)
      templateHole: [
        [/\}/, { token: 'string.template.hole', next: '@pop' }],
        { include: '@root' },
      ],
    },
  } as Monaco.languages.IMonarchLanguage);

  // A light "paper" theme: ink on cream, academic accent colors (maroon/teal/sienna/blue).
  monaco.editor.defineTheme(THEME_ID, {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '8a8473', fontStyle: 'italic' },
      { token: 'comment.shebang', foreground: '8a8473', fontStyle: 'italic' },
      { token: 'meta.frontmatter.delim', foreground: 'bcb6a3' },
      { token: 'meta.frontmatter.key', foreground: '6a6356', fontStyle: 'italic' },
      { token: 'meta.frontmatter.value', foreground: '8a8473' },
      { token: 'string.template', foreground: '4f7a2e' },
      { token: 'string.template.delim', foreground: 'a8a08a' },
      { token: 'string.template.hole', foreground: 'b5651d', fontStyle: 'bold' },
      { token: 'string', foreground: '4f7a2e' },
      { token: 'number', foreground: '9a5b00' },
      { token: 'number.float', foreground: '9a5b00' },
      { token: 'keyword', foreground: '8a2d4a', fontStyle: 'bold' },
      { token: 'operator', foreground: 'a23e6a' },
      { token: 'namespace', foreground: '6a6356', fontStyle: 'italic' },
      { token: 'support.function.query', foreground: 'b5651d', fontStyle: 'bold' },
      { token: 'support.function.dist', foreground: '1f6f8b' },
      { token: 'support.function', foreground: '2a5a9c' },
      { token: 'constant.language', foreground: '9a5b00', fontStyle: 'italic' },
      { token: 'identifier', foreground: '1b1a17' },
      { token: 'delimiter', foreground: '7a7468' },
    ],
    colors: {
      'editor.background': '#f4f1e8',
      'editor.foreground': '#1b1a17',
      'editorLineNumber.foreground': '#bcb6a3',
      'editorLineNumber.activeForeground': '#8a2d4a',
      'editor.selectionBackground': '#dfd6bf',
      'editor.lineHighlightBackground': '#00000008',
      'editorCursor.foreground': '#8a2d4a',
      'editorIndentGuide.background1': '#0000000d',
      'editorWidget.background': '#f4f1e8',
      'editorSuggestWidget.background': '#f4f1e8',
    },
  });
}
