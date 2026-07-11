//! Error and source-span types. Every failure is a typed `NoiseError` with a span — no
//! panics in the language pipeline (see PLAN.md, Phase 1 "real error handling").

use std::fmt;

/// A half-open byte range `[start, end)` into the source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    /// 1-based `(line, column)` of this span's **start** in `src`. Columns count **characters**, not
    /// bytes, so the value is correct for UTF-8 source (a caret placed at `col` lands under the right
    /// glyph — see finding D4). Boundary-safe: if `start` happens to fall inside a multi-byte char
    /// it resolves at the next boundary, and a `start` past the end of `src` clamps to the final
    /// position. A multi-line span reports its start (per finding D1).
    pub fn line_col(&self, src: &str) -> (usize, usize) {
        let target = self.start;
        let mut line = 1usize;
        let mut col = 1usize;
        for (i, ch) in src.char_indices() {
            if i >= target {
                return (line, col);
            }
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// Lexer hit a byte it can't start a token with.
    UnexpectedChar(char),
    /// Unterminated string literal.
    UnterminatedString,
    /// Parser expected something else (message describes what).
    Parse(String),
    /// A name (variable / function) that isn't bound in scope.
    UndefinedName { name: String },
    /// A value had the wrong type for the operation (arithmetic on a string, `!` on a number, a
    /// non-numeric distribution parameter, …). `message` is the human-readable detail.
    TypeMismatch { message: String },
    /// A distribution/recipe/noise value used where a drawn value is required — it must be drawn
    /// with `~` first. `message` is the human-readable detail.
    NotDrawn { message: String },
    /// A call got the wrong number of arguments. `message` is the human-readable detail.
    ArityMismatch { message: String },
    /// Evaluation-time failure that hasn't (yet) been given a dedicated variant — the catch-all that
    /// keeps the [`ErrorKind`] migration incremental (finding D2).
    Runtime(String),
}

impl ErrorKind {
    /// A stable, machine-readable code for this error category. Hosts can branch on this instead of
    /// substring-matching the message (finding D2). The strings are part of the wire contract.
    pub fn code(&self) -> &'static str {
        match self {
            ErrorKind::UnexpectedChar(_) => "unexpected_char",
            ErrorKind::UnterminatedString => "unterminated_string",
            ErrorKind::Parse(_) => "parse_error",
            ErrorKind::UndefinedName { .. } => "undefined_name",
            ErrorKind::TypeMismatch { .. } => "type_mismatch",
            ErrorKind::NotDrawn { .. } => "not_drawn",
            ErrorKind::ArityMismatch { .. } => "arity_mismatch",
            ErrorKind::Runtime(_) => "runtime_error",
        }
    }

    /// True for the evaluation-time error families (everything except lexer/parser errors). Handy in
    /// tests and hosts that want "a spanned semantic error" regardless of which structured variant it
    /// landed in after the D2 migration.
    pub fn is_runtime(&self) -> bool {
        !matches!(
            self,
            ErrorKind::UnexpectedChar(_) | ErrorKind::UnterminatedString | ErrorKind::Parse(_)
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseError {
    pub kind: ErrorKind,
    pub span: Span,
}

impl NoiseError {
    pub fn parse(msg: impl Into<String>, span: Span) -> Self {
        NoiseError {
            kind: ErrorKind::Parse(msg.into()),
            span,
        }
    }
    pub fn runtime(msg: impl Into<String>, span: Span) -> Self {
        NoiseError {
            kind: ErrorKind::Runtime(msg.into()),
            span,
        }
    }
    /// An undefined variable/function reference. Carries the `name` structurally (finding D2); the
    /// rendered message is preserved verbatim from the pre-migration text.
    pub fn undefined_name(name: impl Into<String>, span: Span) -> Self {
        NoiseError {
            kind: ErrorKind::UndefinedName { name: name.into() },
            span,
        }
    }
    /// A wrong-type error (arithmetic on a string, a non-numeric distribution parameter, …).
    pub fn type_mismatch(msg: impl Into<String>, span: Span) -> Self {
        NoiseError {
            kind: ErrorKind::TypeMismatch {
                message: msg.into(),
            },
            span,
        }
    }
    /// An undrawn distribution/recipe/noise used where a drawn value is required.
    pub fn not_drawn(msg: impl Into<String>, span: Span) -> Self {
        NoiseError {
            kind: ErrorKind::NotDrawn {
                message: msg.into(),
            },
            span,
        }
    }
    /// A wrong-argument-count error.
    pub fn arity(msg: impl Into<String>, span: Span) -> Self {
        NoiseError {
            kind: ErrorKind::ArityMismatch {
                message: msg.into(),
            },
            span,
        }
    }
    /// This error's stable category code (see [`ErrorKind::code`]).
    pub fn code(&self) -> &'static str {
        self.kind.code()
    }
}

impl fmt::Display for NoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The migrated variants (Undefined/Type/NotDrawn/Arity) all render under the same
        // `runtime error: …` prefix as the `Runtime` catch-all they were split out of, so the
        // rendered text is byte-identical to the pre-D2 output (message substrings that tests and
        // hosts depend on are preserved; the new structure rides alongside via `code()`).
        let what = match &self.kind {
            ErrorKind::UnexpectedChar(c) => format!("unexpected character {:?}", c),
            ErrorKind::UnterminatedString => "unterminated string literal".to_string(),
            ErrorKind::Parse(m) => format!("parse error: {m}"),
            ErrorKind::UndefinedName { name } => {
                format!("runtime error: undefined variable '{name}'")
            }
            ErrorKind::TypeMismatch { message }
            | ErrorKind::NotDrawn { message }
            | ErrorKind::ArityMismatch { message } => format!("runtime error: {message}"),
            ErrorKind::Runtime(m) => format!("runtime error: {m}"),
        };
        write!(f, "{what} (at {}..{})", self.span.start, self.span.end)
    }
}

impl std::error::Error for NoiseError {}

pub type Result<T> = std::result::Result<T, NoiseError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_is_one_based_and_finds_the_right_line() {
        let src = "a = 1\nb = 22\nc = 333\n";
        // start of file
        assert_eq!(Span::new(0, 1).line_col(src), (1, 1));
        // `22` sits on line 2 at column 5 (1-based: `b`, ` `, `=`, ` `, `2`)
        let at = src.find("22").unwrap();
        assert_eq!(Span::new(at, at + 2).line_col(src), (2, 5));
        // `333` on line 3, column 5
        let at3 = src.find("333").unwrap();
        assert_eq!(Span::new(at3, at3 + 3).line_col(src), (3, 5));
    }

    #[test]
    fn line_col_counts_characters_not_bytes_for_utf8() {
        // Two `π` (2 bytes each) before the caret: the column must be char-based (D4 coordination).
        let src = "ππx";
        let at = src.find('x').unwrap(); // byte offset 4
        assert_eq!(Span::new(at, at + 1).line_col(src), (1, 3));
    }

    #[test]
    fn line_col_past_end_clamps() {
        let src = "abc";
        assert_eq!(Span::new(99, 100).line_col(src), (1, 4));
    }

    #[test]
    fn structured_kinds_expose_stable_codes() {
        assert_eq!(
            NoiseError::undefined_name("foo", Span::default()).code(),
            "undefined_name"
        );
        assert_eq!(
            NoiseError::type_mismatch("x", Span::default()).code(),
            "type_mismatch"
        );
        assert_eq!(
            NoiseError::not_drawn("x", Span::default()).code(),
            "not_drawn"
        );
        assert_eq!(
            NoiseError::arity("x", Span::default()).code(),
            "arity_mismatch"
        );
        assert_eq!(
            NoiseError::runtime("x", Span::default()).code(),
            "runtime_error"
        );
    }

    #[test]
    fn undefined_name_carries_the_name_and_renders_verbatim() {
        let e = NoiseError::undefined_name("ghost", Span::new(3, 8));
        match &e.kind {
            ErrorKind::UndefinedName { name } => assert_eq!(name, "ghost"),
            other => panic!("expected UndefinedName, got {other:?}"),
        }
        assert_eq!(
            e.to_string(),
            "runtime error: undefined variable 'ghost' (at 3..8)"
        );
    }
}
