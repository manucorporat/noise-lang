// Monaco language support for Noise: a Monarch tokenizer, a matching dark theme, and an
// auto-closing/config bundle. Registered once via `registerNoise(monaco)`.
import type * as Monaco from 'monaco-editor';

export const LANGUAGE_ID = 'noise';
export const THEME_ID = 'noise-paper';

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

  // A light "paper" theme: ink on cream, academic accent colors (maroon/teal/sienna/blue).
  monaco.editor.defineTheme(THEME_ID, {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '8a8473', fontStyle: 'italic' },
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
