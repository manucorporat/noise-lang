//! Hand-written lexer. Produces a flat `Vec<Token>` ending in `Eof`.
//!
//! Token set is intentionally a superset of what Phase 0 evaluates so the Pratt parser
//! and later phases (comparisons, `!`, `if`/`else`, `^`, strings) can grow without
//! re-touching the lexer. See LANG.md for the canonical token table.

use crate::error::{ErrorKind, NoiseError, Result, Span};

#[derive(Debug, Clone, PartialEq)]
pub enum TokKind {
    // literals
    Number(f64),
    Ident(String),
    Str(String),
    True,
    False,
    /// A fenced **template** body (PLAN-LITERATE §D3). The lexer captures the raw text between the
    /// fences as one token; the parser splits `${…}` holes and sub-parses each at its true offset.
    /// `body_offset` is the byte position where `body` begins in the original source. `syntax` is
    /// the triple-fence info tag (`md`), `None` for a single-backtick template.
    Template {
        body: String,
        syntax: Option<String>,
        body_offset: usize,
    },
    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,  // % — floored modulo
    Caret,    // ^ — exponentiation (reads like math)
    At,       // @ — matrix product
    Eq,       // =
    Tilde,    // ~
    EqEq,     // ==
    BangEq,   // !=
    Lt,       // <
    Gt,       // >
    Le,       // <=
    Ge,       // >=
    AmpAmp,   // &&
    PipePipe, // ||
    Pipe,     // | — conditioning bar (`P(A | C)`); only meaningful inside P/E/Var/Q
    Bang,     // !
    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,   // [
    RBracket,   // ]
    ColonColon, // ::
    Colon,      // : — named-argument separator (`f(a: x)`)
    DotDot,     // ..
    Comma,
    Semi,
    // keywords
    If,
    Else,
    For,
    In,
    Continue,
    Use,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokKind,
    pub span: Span,
}

/// Tokenize `src` into a flat `Vec<Token>` ending in `Eof`. The hot path — no trivia recorded.
pub fn tokenize(src: &str) -> Result<Vec<Token>> {
    tokenize_inner(src, None)
}

/// Tokenize *and* collect the spans of every comment (`//` line, `/* … */` block) as a side channel — trivia
/// for the literate doc model (PLAN-LITERATE §D4). Comments inside strings, template bodies, and the
/// frontmatter block are **not** trivia (they never reach the comment branch). The token stream is
/// identical to [`tokenize`]'s.
pub fn tokenize_with_trivia(src: &str) -> Result<(Vec<Token>, Vec<Span>)> {
    let mut comments = Vec::new();
    let tokens = tokenize_inner(src, Some(&mut comments))?;
    Ok((tokens, comments))
}

fn tokenize_inner(src: &str, mut comments: Option<&mut Vec<Span>>) -> Result<Vec<Token>> {
    let bytes = src.as_bytes();
    // File-top trivia is skipped *in place* so every token span keeps pointing into the original
    // source (error messages and the doc model rely on it): a `#!` shebang on line 1, then an
    // optional `---`-fenced frontmatter block. Only a fence at the very top counts; `---` anywhere
    // else stays three unary minuses. See `crate::frontmatter`.
    let mut i =
        crate::frontmatter::block_end(src)?.unwrap_or_else(|| crate::frontmatter::shebang_end(src));
    let mut out = Vec::new();
    let n = bytes.len();

    while i < n {
        let c = bytes[i] as char;

        // whitespace
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // line comments: `//` to end of line
        if c == '/' && i + 1 < n && bytes[i + 1] == b'/' {
            let cstart = i;
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            if let Some(sink) = comments.as_deref_mut() {
                sink.push(Span::new(cstart, i));
            }
            continue;
        }

        // block comments: `/* … */` (C-style, non-nesting)
        if c == '/' && i + 1 < n && bytes[i + 1] == b'*' {
            let cstart = i;
            i += 2;
            loop {
                if i + 1 >= n {
                    return Err(NoiseError {
                        kind: ErrorKind::Parse(
                            "unterminated block comment: `/*` has no closing `*/`".into(),
                        ),
                        span: Span::new(cstart, n),
                    });
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            if let Some(sink) = comments.as_deref_mut() {
                sink.push(Span::new(cstart, i));
            }
            continue;
        }

        // `#` is not a comment marker (comments are `//` / `/* … */`); its only legal use is the
        // `#!` shebang on line 1, which was skipped as trivia above. Catch it with a pointed error
        // instead of the generic unexpected-character one.
        if c == '#' {
            return Err(NoiseError {
                kind: ErrorKind::Parse(
                    "`#` is not a comment — use `//` (or `/* … */`); `#!` is only allowed as a \
                     shebang on line 1"
                        .into(),
                ),
                span: Span::new(i, i + 1),
            });
        }

        let start = i;

        // numbers: integer or decimal. Leading `-` is handled by the parser (unary minus).
        if c.is_ascii_digit() || (c == '.' && i + 1 < n && (bytes[i + 1] as char).is_ascii_digit())
        {
            let mut seen_dot = false;
            while i < n {
                let d = bytes[i] as char;
                if d.is_ascii_digit() {
                    i += 1;
                } else if d == '.' && !seen_dot && !(i + 1 < n && bytes[i + 1] == b'.') {
                    // A `.` followed by another `.` is the range operator (`0..10`), not a decimal
                    // point — stop the number here and let `..` lex as its own token.
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            // Scientific-notation exponent: `1e6`, `1.5e-3`, `2E10` (finding D9). Only consume the
            // `e`/`E` when a valid exponent (optional sign + at least one digit) actually follows —
            // otherwise leave it for identifier lexing so `math::e` and names like `e2e` are
            // unaffected, and a bare `1e` stays `Number(1)` then `Ident("e")`.
            if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
                let mut j = i + 1;
                if j < n && (bytes[j] == b'+' || bytes[j] == b'-') {
                    j += 1;
                }
                if j < n && bytes[j].is_ascii_digit() {
                    j += 1;
                    while j < n && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                }
            }
            let text = &src[start..i];
            let value: f64 = text.parse().map_err(|_| NoiseError {
                kind: ErrorKind::Parse(format!("invalid number {text:?}")),
                span: Span::new(start, i),
            })?;
            // A finite decimal that overflows `f64` parses as `Ok(inf)` — a 300-digit literal or a
            // `1e400`. Silently becoming infinity is a footgun (finding D9): diagnose it instead.
            if !value.is_finite() {
                return Err(NoiseError {
                    kind: ErrorKind::Parse(format!(
                        "number literal {text:?} is too large — it overflows to infinity (the \
                         largest finite f64 is about 1.8e308)"
                    )),
                    span: Span::new(start, i),
                });
            }
            out.push(Token {
                kind: TokKind::Number(value),
                span: Span::new(start, i),
            });
            continue;
        }

        // identifiers / keywords
        if c.is_ascii_alphabetic() || c == '_' {
            while i < n {
                let d = bytes[i] as char;
                if d.is_ascii_alphanumeric() || d == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let text = &src[start..i];
            let kind = match text {
                "if" => TokKind::If,
                "else" => TokKind::Else,
                "for" => TokKind::For,
                "in" => TokKind::In,
                "continue" => TokKind::Continue,
                "true" => TokKind::True,
                "false" => TokKind::False,
                "use" => TokKind::Use,
                _ => TokKind::Ident(text.to_string()),
            };
            out.push(Token {
                kind,
                span: Span::new(start, i),
            });
            continue;
        }

        // templates: backtick-fenced (PLAN-LITERATE §D3). Two weights:
        //   `` `…` ``      — single backtick, plain body; the body cannot contain a backtick.
        //   ` ```tag…``` ` — triple fence carrying an optional syntax tag; body may hold backticks.
        if c == '`' {
            let triple = i + 2 < n && bytes[i + 1] == b'`' && bytes[i + 2] == b'`';
            if triple {
                // Info tag runs from just past the ``` to end of that line.
                let tag_start = i + 3;
                let tag_end = match src[tag_start..].find('\n') {
                    Some(rel) => tag_start + rel,
                    None => {
                        return Err(NoiseError {
                            kind: ErrorKind::Parse(
                                "unterminated template: ``` has no closing ```".into(),
                            ),
                            span: Span::new(start, n),
                        })
                    }
                };
                let tag = src[tag_start..tag_end].trim();
                let syntax = if tag.is_empty() {
                    None
                } else {
                    Some(tag.to_string())
                };
                let body_offset = tag_end + 1; // just past the newline after the info line
                                               // Find a closing line that is exactly ``` (after trimming surrounding whitespace).
                let mut line_start = body_offset;
                let close_body_end;
                let close_fence_end;
                loop {
                    let line_end = match src[line_start..].find('\n') {
                        Some(rel) => line_start + rel,
                        None => bytes.len(),
                    };
                    if src[line_start..line_end].trim() == "```" {
                        close_body_end = line_start;
                        close_fence_end = line_end;
                        break;
                    }
                    if line_end >= bytes.len() {
                        return Err(NoiseError {
                            kind: ErrorKind::Parse(
                                "unterminated template: ``` has no closing ```".into(),
                            ),
                            span: Span::new(start, n),
                        });
                    }
                    line_start = line_end + 1;
                }
                let body = src[body_offset..close_body_end].to_string();
                i = close_fence_end;
                out.push(Token {
                    kind: TokKind::Template {
                        body,
                        syntax,
                        body_offset,
                    },
                    span: Span::new(start, i),
                });
                continue;
            } else {
                let body_offset = i + 1;
                let mut j = body_offset;
                while j < n && bytes[j] != b'`' {
                    j += 1;
                }
                if j >= n {
                    return Err(NoiseError {
                        kind: ErrorKind::Parse(
                            "unterminated template: `` ` `` has no closing backtick".into(),
                        ),
                        span: Span::new(start, n),
                    });
                }
                let body = src[body_offset..j].to_string();
                i = j + 1; // past the closing backtick
                out.push(Token {
                    kind: TokKind::Template {
                        body,
                        syntax: None,
                        body_offset,
                    },
                    span: Span::new(start, i),
                });
                continue;
            }
        }

        // strings: double-quoted, no escapes yet (matches the legacy grammar).
        if c == '"' {
            i += 1; // opening quote
            let body_start = i;
            while i < n && bytes[i] != b'"' {
                i += 1;
            }
            if i >= n {
                return Err(NoiseError {
                    kind: ErrorKind::UnterminatedString,
                    span: Span::new(start, i),
                });
            }
            let text = src[body_start..i].to_string();
            i += 1; // closing quote
            out.push(Token {
                kind: TokKind::Str(text),
                span: Span::new(start, i),
            });
            continue;
        }

        // multi/single-char operators and punctuation
        let two = if i + 1 < n {
            Some(bytes[i + 1] as char)
        } else {
            None
        };
        let (kind, len) = match (c, two) {
            ('=', Some('=')) => (TokKind::EqEq, 2),
            ('!', Some('=')) => (TokKind::BangEq, 2),
            ('<', Some('=')) => (TokKind::Le, 2),
            ('>', Some('=')) => (TokKind::Ge, 2),
            ('&', Some('&')) => (TokKind::AmpAmp, 2),
            ('|', Some('|')) => (TokKind::PipePipe, 2),
            ('|', _) => (TokKind::Pipe, 1),
            (':', Some(':')) => (TokKind::ColonColon, 2),
            (':', _) => (TokKind::Colon, 1),
            ('.', Some('.')) => (TokKind::DotDot, 2),
            ('+', _) => (TokKind::Plus, 1),
            ('-', _) => (TokKind::Minus, 1),
            ('*', _) => (TokKind::Star, 1),
            ('/', _) => (TokKind::Slash, 1),
            ('%', _) => (TokKind::Percent, 1),
            ('^', _) => (TokKind::Caret, 1),
            ('=', _) => (TokKind::Eq, 1),
            ('~', _) => (TokKind::Tilde, 1),
            ('@', _) => (TokKind::At, 1),
            ('<', _) => (TokKind::Lt, 1),
            ('>', _) => (TokKind::Gt, 1),
            ('!', _) => (TokKind::Bang, 1),
            ('(', _) => (TokKind::LParen, 1),
            (')', _) => (TokKind::RParen, 1),
            ('{', _) => (TokKind::LBrace, 1),
            ('}', _) => (TokKind::RBrace, 1),
            ('[', _) => (TokKind::LBracket, 1),
            (']', _) => (TokKind::RBracket, 1),
            (',', _) => (TokKind::Comma, 1),
            (';', _) => (TokKind::Semi, 1),
            _ => {
                // `c` was decoded via `bytes[i] as char`, which mangles any non-ASCII byte (a
                // leading UTF-8 byte like `π`'s `0xCF` becomes `Ï`) and would emit a 1-byte span
                // that is **not a char boundary** — a host slicing `&src[span]` for a caret then
                // panics (finding D4). Decode the real char from the source and span its full
                // width so the reported character is correct and the span is boundary-valid.
                let real = src[start..].chars().next().unwrap_or(c);
                return Err(NoiseError {
                    kind: ErrorKind::UnexpectedChar(real),
                    span: Span::new(start, start + real.len_utf8()),
                });
            }
        };
        i += len;
        out.push(Token {
            kind,
            span: Span::new(start, i),
        });
    }

    out.push(Token {
        kind: TokKind::Eof,
        span: Span::new(n, n),
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokKind> {
        tokenize(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn operators_are_matched_greedily() {
        use TokKind::*;
        assert_eq!(
            kinds("^ == != <= >= && || :: : .. = ~ @ < > ! + - * / % ( ) { } [ ] , ;"),
            vec![
                Caret, EqEq, BangEq, Le, Ge, AmpAmp, PipePipe, ColonColon, Colon, DotDot, Eq,
                Tilde, At, Lt, Gt, Bang, Plus, Minus, Star, Slash, Percent, LParen, RParen, LBrace,
                RBrace, LBracket, RBracket, Comma, Semi, Eof,
            ]
        );
    }

    #[test]
    fn single_pipe_is_distinct_from_double_pipe() {
        use TokKind::*;
        // `|` (conditioning bar) and `||` (logical or) lex as different tokens.
        assert_eq!(
            kinds("a | b || c"),
            vec![
                Ident("a".into()),
                Pipe,
                Ident("b".into()),
                PipePipe,
                Ident("c".into()),
                Eof
            ]
        );
    }

    #[test]
    fn numbers_include_decimals_and_leading_dot() {
        use TokKind::*;
        assert_eq!(
            kinds("3 2.75 .5 10."),
            vec![Number(3.0), Number(2.75), Number(0.5), Number(10.0), Eof]
        );
    }

    #[test]
    fn range_operator_is_not_swallowed_by_number_lexing() {
        use TokKind::*;
        // `0..10` is three tokens, not `0.` `.10`; spacing is irrelevant.
        assert_eq!(kinds("0..10"), vec![Number(0.0), DotDot, Number(10.0), Eof]);
        assert_eq!(
            kinds("i+1 .. n"),
            vec![
                Ident("i".into()),
                Plus,
                Number(1.0),
                DotDot,
                Ident("n".into()),
                Eof,
            ]
        );
        // a trailing-dot number still lexes as a decimal
        assert_eq!(kinds("10."), vec![Number(10.0), Eof]);
    }

    #[test]
    fn line_and_block_comments_are_skipped() {
        use TokKind::*;
        assert_eq!(
            kinds("1 // comment\n2 /* inline */ 3"),
            vec![Number(1.0), Number(2.0), Number(3.0), Eof]
        );
        // Block comments may span lines.
        assert_eq!(
            kinds("1 /* a\nmulti-line\ncomment */ 2"),
            vec![Number(1.0), Number(2.0), Eof]
        );
    }

    #[test]
    fn hash_is_not_a_comment() {
        let err = tokenize("1 # not a comment\n").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("`//`"), "should point at `//`; got: {msg}");
    }

    #[test]
    fn unterminated_block_comment_is_an_error() {
        let err = tokenize("1 /* no close").unwrap_err();
        assert!(format!("{err}").contains("unterminated block comment"));
    }

    #[test]
    fn shebang_on_line_one_is_trivia() {
        use TokKind::*;
        assert_eq!(kinds("#!/usr/bin/env noise\n1"), vec![Number(1.0), Eof]);
        // Shebang then frontmatter: both are skipped.
        assert_eq!(
            kinds("#!/usr/bin/env noise\n---\ntitle: t\n---\n1"),
            vec![Number(1.0), Eof]
        );
        // A shebang-only file lexes to just EOF.
        assert_eq!(kinds("#!/usr/bin/env noise"), vec![Eof]);
    }

    #[test]
    fn keywords_are_distinguished_from_identifiers() {
        use TokKind::*;
        assert_eq!(
            kinds("if else for in continue use iffy _x x1"),
            vec![
                If,
                Else,
                For,
                In,
                Continue,
                Use,
                Ident("iffy".into()),
                Ident("_x".into()),
                Ident("x1".into()),
                Eof,
            ]
        );
    }

    #[test]
    fn templates_lex_single_and_triple_fence() {
        // Single backtick: raw body, no syntax tag, `body_offset` past the backtick.
        let toks = tokenize("`hi ${x}`").unwrap();
        match &toks[0].kind {
            TokKind::Template {
                body,
                syntax,
                body_offset,
            } => {
                assert_eq!(body, "hi ${x}");
                assert_eq!(*syntax, None);
                assert_eq!(*body_offset, 1);
            }
            other => panic!("expected a template token, got {other:?}"),
        }
        // Triple fence: info tag captured, body is the raw text between the fences (the parser
        // normalizes — dedents and trims the fence-adjacent blank lines — later).
        let toks = tokenize("```md\nhello\n```").unwrap();
        match &toks[0].kind {
            TokKind::Template { body, syntax, .. } => {
                assert_eq!(body, "hello\n");
                assert_eq!(syntax.as_deref(), Some("md"));
            }
            other => panic!("expected a template token, got {other:?}"),
        }
    }

    #[test]
    fn unterminated_template_is_an_error() {
        assert!(tokenize("`no close").is_err());
        assert!(tokenize("```md\nno close").is_err());
    }

    #[test]
    fn strings_lex_and_unterminated_is_an_error() {
        assert_eq!(
            kinds("\"hi there\""),
            vec![TokKind::Str("hi there".into()), TokKind::Eof]
        );
        let err = tokenize("\"no close").unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UnterminatedString));
    }

    #[test]
    fn unexpected_char_errors_with_a_span() {
        let err = tokenize("1 ? 2").unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UnexpectedChar('?')));
        assert_eq!(err.span, Span::new(2, 3));
    }

    #[test]
    fn non_ascii_char_reports_the_real_char_and_a_boundary_span() {
        // `π` is two UTF-8 bytes (0xCF 0x80). Pre-fix this reported `'Ï'` with a 1-byte span at a
        // non-char-boundary (finding D4). It must now report the actual char and a span covering
        // the whole char — and, crucially, `&src[span]` must not panic (a caret host slices it).
        let src = "π = 3";
        let err = tokenize(src).unwrap_err();
        assert!(
            matches!(err.kind, ErrorKind::UnexpectedChar('π')),
            "got {:?}",
            err.kind
        );
        assert_eq!(err.span, Span::new(0, 2), "span covers both bytes of π");
        // The span is a valid char boundary — slicing for a caret does not panic.
        assert_eq!(&src[err.span.start..err.span.end], "π");
        // And it round-trips through the line/col mapping used by the caret renderer (col 1).
        assert_eq!(err.span.line_col(src), (1, 1));
    }

    #[test]
    fn scientific_notation_lexes() {
        use TokKind::*;
        assert_eq!(kinds("1e6"), vec![Number(1_000_000.0), Eof]);
        assert_eq!(kinds("1.5e-3"), vec![Number(0.0015), Eof]);
        assert_eq!(kinds("2E10"), vec![Number(2e10), Eof]);
        assert_eq!(kinds("6.022e23"), vec![Number(6.022e23), Eof]);
        // a bare `e` with no exponent digits is not consumed into the number
        assert_eq!(kinds("1e"), vec![Number(1.0), Ident("e".into()), Eof]);
        // `1e+` (sign but no digit) also leaves `e` alone
        assert_eq!(
            kinds("1e+"),
            vec![Number(1.0), Ident("e".into()), Plus, Eof]
        );
    }

    #[test]
    fn overflowing_number_literal_is_diagnosed_not_infinity() {
        // Both a huge exponent and a 300-digit integer overflow f64 to inf; must be an error, not a
        // silent `Number(inf)` (finding D9).
        let err = tokenize("1e400").unwrap_err();
        assert!(
            matches!(err.kind, ErrorKind::Parse(_)),
            "got {:?}",
            err.kind
        );
        let huge = "9".repeat(400);
        assert!(matches!(
            tokenize(&huge).unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
    }

    #[test]
    fn token_spans_point_at_the_source() {
        let toks = tokenize("12 + 3").unwrap();
        assert_eq!(toks[0].span, Span::new(0, 2)); // 12
        assert_eq!(toks[1].span, Span::new(3, 4)); // +
        assert_eq!(toks[2].span, Span::new(5, 6)); // 3
        assert_eq!(toks[3].kind, TokKind::Eof);
        assert_eq!(toks[3].span, Span::new(6, 6)); // Eof at end
    }
}
