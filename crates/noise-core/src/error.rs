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
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// Lexer hit a byte it can't start a token with.
    UnexpectedChar(char),
    /// Unterminated string literal.
    UnterminatedString,
    /// Parser expected something else (message describes what).
    Parse(String),
    /// Evaluation-time failure (type error, undefined variable, division, ...).
    Runtime(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseError {
    pub kind: ErrorKind,
    pub span: Span,
}

impl NoiseError {
    pub fn parse(msg: impl Into<String>, span: Span) -> Self {
        NoiseError { kind: ErrorKind::Parse(msg.into()), span }
    }
    pub fn runtime(msg: impl Into<String>, span: Span) -> Self {
        NoiseError { kind: ErrorKind::Runtime(msg.into()), span }
    }
}

impl fmt::Display for NoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match &self.kind {
            ErrorKind::UnexpectedChar(c) => format!("unexpected character {:?}", c),
            ErrorKind::UnterminatedString => "unterminated string literal".to_string(),
            ErrorKind::Parse(m) => format!("parse error: {m}"),
            ErrorKind::Runtime(m) => format!("runtime error: {m}"),
        };
        write!(f, "{what} (at {}..{})", self.span.start, self.span.end)
    }
}

impl std::error::Error for NoiseError {}

pub type Result<T> = std::result::Result<T, NoiseError>;
