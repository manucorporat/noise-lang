// Monaco language support for Noise: a Monarch tokenizer, a matching dark theme, and an
// auto-closing/config bundle. Registered once via `registerNoise(monaco)`.
import type * as Monaco from 'monaco-editor';

export const LANGUAGE_ID = 'noise';
export const THEME_ID = 'noise-dark';

// Module-scoped builtin names, by role, so the highlighter can color them distinctly.
const DISTRIBUTIONS = [
  'unif', 'unif_int', 'bernoulli', 'normal', 'normal_int', 'exp', 'exp_int',
  'poisson', 'geometric', 'iid', 'iidmat',
];
const QUERIES = ['P', 'Q', 'E', 'Var']; // probability / moment / quantile queries
const BUILTINS = [
  'Print', 'Len',
  'sqrt', 'round', 'log', 'log10', 'sin', 'cos', 'atan', 'sign',
  'sum', 'count', 'any', 'all', 'max', 'min', 'mean', 'dot', 'normsq', 'norm',
  'vadd', 'vsub', 'matvec', 'transpose', 'normalize', 'has_duplicate', 'mse',
  'ones', 'zeros', 'iota', 'vsign', 'scale',
  'sine', 'cosine', 'noise_white', 'sample',
];
const CONSTANTS = ['pi', 'e'];
const MODULES = ['builtin', 'rand', 'math', 'vec', 'signal'];

let registered = false;

export function registerNoise(monaco: typeof Monaco): void {
  if (registered) return;
  registered = true;

  monaco.languages.register({ id: LANGUAGE_ID });

  monaco.languages.setLanguageConfiguration(LANGUAGE_ID, {
    comments: { lineComment: '#' },
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
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
    ],
  });

  monaco.languages.setMonarchTokensProvider(LANGUAGE_ID, {
    defaultToken: '',
    keywords: ['if', 'else', 'for', 'in', 'use', 'true', 'false'],
    distributions: DISTRIBUTIONS,
    queries: QUERIES,
    builtins: BUILTINS,
    constants: CONSTANTS,
    modules: MODULES,
    // longest-first so `**`, `==`, `..`, `::`, `&&` win over their prefixes
    operators: [
      '**', '==', '!=', '<=', '>=', '&&', '||', '..', '::',
      '+', '-', '*', '/', '<', '>', '!', '=', '~',
    ],
    symbols: /[=~!<>+\-*/&|.:]+/,
    tokenizer: {
      root: [
        // comments
        [/#.*$/, 'comment'],
        [/\/\/.*$/, 'comment'],
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
    },
  } as Monaco.languages.IMonarchLanguage);

  // A dark theme tuned to the site palette (cool indigo/teal/amber on near-black).
  monaco.editor.defineTheme(THEME_ID, {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '5b6477', fontStyle: 'italic' },
      { token: 'string', foreground: '8be2c0' },
      { token: 'number', foreground: 'f0b66a' },
      { token: 'number.float', foreground: 'f0b66a' },
      { token: 'keyword', foreground: 'c08cf0', fontStyle: 'bold' },
      { token: 'operator', foreground: 'ff8fb0' },
      { token: 'namespace', foreground: '6fb7ff', fontStyle: 'italic' },
      { token: 'support.function.query', foreground: 'ffd479', fontStyle: 'bold' },
      { token: 'support.function.dist', foreground: '7fd1ff' },
      { token: 'support.function', foreground: '9ad0ff' },
      { token: 'constant.language', foreground: 'f0b66a', fontStyle: 'italic' },
      { token: 'identifier', foreground: 'e6e9f0' },
      { token: 'delimiter', foreground: '8891a8' },
    ],
    colors: {
      'editor.background': '#0c0f1a00', // transparent — the shader shows through the glass panel
      'editor.foreground': '#e6e9f0',
      'editorLineNumber.foreground': '#3a4258',
      'editorLineNumber.activeForeground': '#8aa0d0',
      'editor.selectionBackground': '#3a4a8055',
      'editor.lineHighlightBackground': '#ffffff08',
      'editorCursor.foreground': '#9ad0ff',
      'editorIndentGuide.background1': '#ffffff0c',
      'editorWidget.background': '#121728',
      'editorSuggestWidget.background': '#121728',
    },
  });
}
