//! Hand-written Pratt (precedence-climbing) parser. Turns tokens into a `Program`.
//!
//! Precedence (low → high), all left-associative except `**` and prefix unary:
//!   assignment `=` `~` (right-assoc, lowest)
//!   comparison `== != < > <= >=`
//!   additive   `+ -`
//!   multiplicative `* /`
//!   power `**` (right-assoc)
//!   prefix `- !`
//!   call / primary
//! See LANG.md for the canonical table.

use crate::ast::*;
use crate::error::{NoiseError, Result, Span};
use crate::lexer::{tokenize, TokKind, Token};

pub fn parse(src: &str) -> Result<Program> {
    let tokens = tokenize(src)?;
    let mut p = Parser { tokens, pos: 0 };
    let stmts = p.parse_stmts(&[TokKind::Eof])?;
    p.expect(TokKind::Eof)?;
    Ok(Program { stmts })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &TokKind {
        &self.tokens[self.pos].kind
    }
    fn peek_at(&self, ahead: usize) -> &TokKind {
        let i = (self.pos + ahead).min(self.tokens.len() - 1);
        &self.tokens[i].kind
    }
    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }
    fn bump(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, k: &TokKind) -> bool {
        if self.peek() == k {
            self.bump();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, k: TokKind) -> Result<Token> {
        if *self.peek() == k {
            Ok(self.bump())
        } else {
            Err(NoiseError::parse(
                format!("expected {:?}, found {:?}", k, self.peek()),
                self.span(),
            ))
        }
    }

    /// Parse statements until one of `terminators` is the next token. Statements are
    /// separated/terminated by `;`; a trailing `;` is optional (unlike the legacy grammar).
    fn parse_stmts(&mut self, terminators: &[TokKind]) -> Result<Vec<Spanned>> {
        let mut stmts = Vec::new();
        loop {
            while *self.peek() == TokKind::Semi {
                self.bump();
            }
            if terminators.contains(self.peek()) {
                break;
            }
            let s = self.parse_expr()?;
            stmts.push(s);
            // consume separators (or stop at a terminator)
            if *self.peek() == TokKind::Semi {
                while *self.peek() == TokKind::Semi {
                    self.bump();
                }
            } else if !terminators.contains(self.peek()) {
                return Err(NoiseError::parse(
                    format!("expected ';' or end of block, found {:?}", self.peek()),
                    self.span(),
                ));
            }
        }
        Ok(stmts)
    }

    /// Expression entry point. Handles function definitions and assignment/sample bindings
    /// (lowest precedence).
    fn parse_expr(&mut self) -> Result<Spanned> {
        if let TokKind::Ident(name) = self.peek().clone() {
            // Function definition: `name(params) = body` or `name(params) ~ body`. Disambiguated
            // from a call expression by looking past the matching `)` for a `=` / `~`.
            if *self.peek_at(1) == TokKind::LParen {
                if let Some(after) = self.matching_paren_after(self.pos + 1) {
                    if matches!(
                        self.tokens.get(after).map(|t| &t.kind),
                        Some(TokKind::Eq) | Some(TokKind::Tilde)
                    ) {
                        return self.parse_fn_def(name);
                    }
                }
            }
            // `Ident = ...` or `Ident ~ ...` → binding, right-associative.
            let bind = match self.peek_at(1) {
                TokKind::Eq => Some(BindKind::Assign),
                TokKind::Tilde => Some(BindKind::Sample),
                _ => None,
            };
            if let Some(kind) = bind {
                let start = self.span().start;
                self.bump(); // ident
                self.bump(); // = or ~
                let rhs = self.parse_expr()?;
                let span = Span::new(start, rhs.span.end);
                return Ok(Spanned::new(Expr::Bind(kind, name, Box::new(rhs)), span));
            }
        }
        self.parse_bp(0)
    }

    /// Given the index of a `LParen`, return the index just past its matching `RParen`, or
    /// `None` if unbalanced. Used to look past a parameter list when deciding whether an
    /// `Ident ( … )` is a function *definition* (a `=`/`~` follows) or a call expression.
    fn matching_paren_after(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        let mut i = open;
        while let Some(tok) = self.tokens.get(i) {
            match tok.kind {
                TokKind::LParen => depth += 1,
                TokKind::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                TokKind::Eof => return None,
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Parse `name(params) = body` / `name(params) ~ body`. The leading `name` and the `(` are
    /// the current tokens. Parameters are bare identifiers; the body is a full expression.
    fn parse_fn_def(&mut self, name: String) -> Result<Spanned> {
        let start = self.span().start;
        self.bump(); // name
        self.expect(TokKind::LParen)?;
        let mut params = Vec::new();
        if *self.peek() != TokKind::RParen {
            loop {
                let t = self.bump();
                match t.kind {
                    TokKind::Ident(p) => params.push(p),
                    other => {
                        return Err(NoiseError::parse(
                            format!("expected a parameter name, found {other:?}"),
                            t.span,
                        ))
                    }
                }
                if !self.eat(&TokKind::Comma) {
                    break;
                }
            }
        }
        self.expect(TokKind::RParen)?;
        let kind = if self.eat(&TokKind::Eq) {
            BindKind::Assign
        } else if self.eat(&TokKind::Tilde) {
            BindKind::Sample
        } else {
            return Err(NoiseError::parse(
                format!("expected `=` or `~` in function definition, found {:?}", self.peek()),
                self.span(),
            ));
        };
        let body = self.parse_expr()?;
        let span = Span::new(start, body.span.end);
        Ok(Spanned::new(Expr::FnDef { kind, name, params, body: Box::new(body) }, span))
    }

    /// Precedence-climbing core for infix operators.
    fn parse_bp(&mut self, min_bp: u8) -> Result<Spanned> {
        let mut lhs = self.parse_prefix()?;
        loop {
            // `..` is the lowest-binding infix form (Rust-style ranges sit below every operator,
            // so `i + 1 .. len(xs)` is `(i+1)..(len(xs))`). It builds `Expr::Range`, not `Binary`.
            if *self.peek() == TokKind::DotDot {
                if RANGE_LBP < min_bp {
                    break;
                }
                self.bump(); // ..
                let rhs = self.parse_bp(RANGE_RBP)?;
                let span = Span::new(lhs.span.start, rhs.span.end);
                lhs = Spanned::new(Expr::Range(Box::new(lhs), Box::new(rhs)), span);
                continue;
            }
            let Some((op, l_bp, r_bp)) = infix_op(self.peek()) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.bump(); // operator
            let rhs = self.parse_bp(r_bp)?;
            let span = Span::new(lhs.span.start, rhs.span.end);
            lhs = Spanned::new(Expr::Binary(op, Box::new(lhs), Box::new(rhs)), span);
        }
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Spanned> {
        let start = self.span().start;
        match self.peek() {
            TokKind::Minus => {
                self.bump();
                let rhs = self.parse_bp(PREFIX_BP)?;
                let span = Span::new(start, rhs.span.end);
                Ok(Spanned::new(Expr::Unary(UnOp::Neg, Box::new(rhs)), span))
            }
            TokKind::Bang => {
                self.bump();
                let rhs = self.parse_bp(PREFIX_BP)?;
                let span = Span::new(start, rhs.span.end);
                Ok(Spanned::new(Expr::Unary(UnOp::Not, Box::new(rhs)), span))
            }
            _ => self.parse_postfix(),
        }
    }

    /// Postfix layer: a primary optionally followed by `[index]` subscripts, repeatable so
    /// `M[i][j]` and `f(x)[i]` both work. Binds tighter than every infix/prefix operator (like a
    /// call), looser than nothing — it wraps the primary it just parsed.
    fn parse_postfix(&mut self) -> Result<Spanned> {
        let mut e = self.parse_primary()?;
        while *self.peek() == TokKind::LBracket {
            self.bump(); // [
            let index = self.parse_expr()?;
            let close = self.expect(TokKind::RBracket)?;
            let span = Span::new(e.span.start, close.span.end);
            e = Spanned::new(Expr::Index(Box::new(e), Box::new(index)), span);
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> Result<Spanned> {
        let tok = self.tokens[self.pos].clone();
        match tok.kind {
            TokKind::Number(n) => {
                self.bump();
                Ok(Spanned::new(Expr::Number(n), tok.span))
            }
            TokKind::Str(s) => {
                self.bump();
                Ok(Spanned::new(Expr::Str(s), tok.span))
            }
            TokKind::True => {
                self.bump();
                Ok(Spanned::new(Expr::Bool(true), tok.span))
            }
            TokKind::False => {
                self.bump();
                Ok(Spanned::new(Expr::Bool(false), tok.span))
            }
            TokKind::Ident(name) => {
                self.bump();
                // A `module::name` path (Rust-style, single level for now). The qualifier rides
                // inside the name string with a `::` separator; eval splits it back out.
                let mut name = name;
                let mut end = tok.span.end;
                while *self.peek() == TokKind::ColonColon {
                    self.bump(); // ::
                    let seg = self.bump();
                    match seg.kind {
                        TokKind::Ident(s) => {
                            name = format!("{name}::{s}");
                            end = seg.span.end;
                        }
                        other => {
                            return Err(NoiseError::parse(
                                format!("expected an identifier after `::`, found {other:?}"),
                                seg.span,
                            ))
                        }
                    }
                }
                if *self.peek() == TokKind::LParen {
                    let (args, call_end) = self.parse_call_args()?;
                    let span = Span::new(tok.span.start, call_end);
                    Ok(Spanned::new(Expr::Call(name, args), span))
                } else {
                    Ok(Spanned::new(Expr::Ident(name), Span::new(tok.span.start, end)))
                }
            }
            TokKind::Use => {
                self.bump();
                let seg = self.bump();
                match seg.kind {
                    TokKind::Ident(module) => {
                        let span = Span::new(tok.span.start, seg.span.end);
                        Ok(Spanned::new(Expr::Use(module), span))
                    }
                    other => Err(NoiseError::parse(
                        format!("expected a module name after `use`, found {other:?}"),
                        seg.span,
                    )),
                }
            }
            TokKind::LParen => {
                self.bump();
                let inner = self.parse_expr()?;
                self.expect(TokKind::RParen)?;
                Ok(inner)
            }
            TokKind::LBrace => self.parse_block(),
            TokKind::If => self.parse_if(),
            TokKind::For => self.parse_for(),
            TokKind::LBracket => self.parse_array(),
            other => Err(NoiseError::parse(
                format!("unexpected token {:?}", other),
                tok.span,
            )),
        }
    }

    fn parse_call_args(&mut self) -> Result<(Vec<Spanned>, usize)> {
        self.expect(TokKind::LParen)?;
        let mut args = Vec::new();
        if *self.peek() != TokKind::RParen {
            loop {
                args.push(self.parse_expr()?);
                if !self.eat(&TokKind::Comma) {
                    break;
                }
            }
        }
        let close = self.expect(TokKind::RParen)?;
        Ok((args, close.span.end))
    }

    /// Array literal `[a, b, …]` (or empty `[]`). A trailing comma is not allowed (matches the
    /// call-argument rule).
    fn parse_array(&mut self) -> Result<Spanned> {
        let open = self.expect(TokKind::LBracket)?;
        let mut elems = Vec::new();
        if *self.peek() != TokKind::RBracket {
            loop {
                elems.push(self.parse_expr()?);
                if !self.eat(&TokKind::Comma) {
                    break;
                }
            }
        }
        let close = self.expect(TokKind::RBracket)?;
        let span = Span::new(open.span.start, close.span.end);
        Ok(Spanned::new(Expr::Array(elems), span))
    }

    /// `for IDENT in EXPR { body }` — parsed like `if`. The iterable is a bare `opexpr` (no
    /// binding) so the `{` of the body isn't swallowed as a block-primary.
    fn parse_for(&mut self) -> Result<Spanned> {
        let kw = self.expect(TokKind::For)?;
        let var = match self.bump().kind {
            TokKind::Ident(name) => name,
            other => {
                return Err(NoiseError::parse(
                    format!("expected a loop variable name after `for`, found {other:?}"),
                    self.span(),
                ))
            }
        };
        self.expect(TokKind::In)?;
        let iter = self.parse_bp(0)?;
        let body = self.parse_block()?;
        let span = Span::new(kw.span.start, body.span.end);
        Ok(Spanned::new(
            Expr::For { var, iter: Box::new(iter), body: Box::new(body) },
            span,
        ))
    }

    fn parse_block(&mut self) -> Result<Spanned> {
        let open = self.expect(TokKind::LBrace)?;
        let stmts = self.parse_stmts(&[TokKind::RBrace])?;
        let close = self.expect(TokKind::RBrace)?;
        let span = Span::new(open.span.start, close.span.end);
        Ok(Spanned::new(Expr::Block(stmts), span))
    }

    fn parse_if(&mut self) -> Result<Spanned> {
        let kw = self.expect(TokKind::If)?;
        let cond = self.parse_bp(0)?;
        let then_block = self.parse_block()?;
        let (else_branch, end) = if self.eat(&TokKind::Else) {
            let eb = if *self.peek() == TokKind::If {
                self.parse_if()?
            } else {
                self.parse_block()?
            };
            let end = eb.span.end;
            (Some(Box::new(eb)), end)
        } else {
            (None, then_block.span.end)
        };
        let span = Span::new(kw.span.start, end);
        Ok(Spanned::new(
            Expr::If(Box::new(cond), Box::new(then_block), else_branch),
            span,
        ))
    }
}

/// Prefix operators bind tighter than everything except `**` (so `-2 ** 2 == -(2**2)`,
/// matching common math/Python convention) and looser than `**`.
const PREFIX_BP: u8 = 13;

/// `..` (range) binding powers — the lowest of any infix form, so the bounds are full operator
/// expressions (`i + 1 .. len(xs)` parses as `(i+1)..(len(xs))`). `r_bp > l_bp` keeps it
/// left-leaning, though chained `a..b..c` is a runtime error (a range isn't a number).
const RANGE_LBP: u8 = 1;
const RANGE_RBP: u8 = 2;

/// Returns `(op, left_bp, right_bp)` for an infix token. Left-assoc ops have
/// `left_bp < right_bp`; `**` is right-assoc (`left_bp > right_bp`).
/// Precedence low→high: `..` < `||` < `&&` < comparison < `+ -` < `* /` < prefix < `**`.
fn infix_op(k: &TokKind) -> Option<(BinOp, u8, u8)> {
    use BinOp::*;
    Some(match k {
        TokKind::PipePipe => (Or, 3, 4),
        TokKind::AmpAmp => (And, 5, 6),
        TokKind::EqEq => (Eq, 7, 8),
        TokKind::BangEq => (Ne, 7, 8),
        TokKind::Lt => (Lt, 7, 8),
        TokKind::Gt => (Gt, 7, 8),
        TokKind::Le => (Le, 7, 8),
        TokKind::Ge => (Ge, 7, 8),
        TokKind::Plus => (Add, 9, 10),
        TokKind::Minus => (Sub, 9, 10),
        TokKind::Star => (Mul, 11, 12),
        TokKind::Slash => (Div, 11, 12),
        TokKind::StarStar => (Pow, 14, 13),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    fn parse_one(src: &str) -> Expr {
        let mut prog = parse(src).unwrap();
        assert_eq!(prog.stmts.len(), 1, "expected exactly one statement in {src:?}");
        prog.stmts.pop().unwrap().expr
    }

    #[test]
    fn power_is_right_associative() {
        // 2 ** 3 ** 2  ==>  2 ** (3 ** 2)
        match parse_one("2 ** 3 ** 2") {
            Expr::Binary(BinOp::Pow, l, r) => {
                assert!(matches!(l.expr, Expr::Number(n) if n == 2.0));
                assert!(matches!(r.expr, Expr::Binary(BinOp::Pow, _, _)));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        // 1 + 2 * 3  ==>  1 + (2 * 3)
        match parse_one("1 + 2 * 3") {
            Expr::Binary(BinOp::Add, _, r) => {
                assert!(matches!(r.expr, Expr::Binary(BinOp::Mul, _, _)))
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn binding_is_right_associative() {
        // a = b = 3  ==>  a = (b = 3)
        match parse_one("a = b = 3") {
            Expr::Bind(BindKind::Assign, name, rhs) => {
                assert_eq!(name, "a");
                assert!(matches!(rhs.expr, Expr::Bind(BindKind::Assign, _, _)));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn function_definition_is_distinguished_from_a_call() {
        assert!(matches!(parse_one("f(x) = x"), Expr::FnDef { kind: BindKind::Assign, .. }));
        assert!(matches!(parse_one("g() ~ unif(0,1)"), Expr::FnDef { kind: BindKind::Sample, .. }));
        // a bare `f(x)` (no following `=`/`~`) is a call expression, not a definition
        assert!(matches!(parse_one("f(x)"), Expr::Call(_, _)));
    }

    #[test]
    fn parse_errors_are_typed_and_dont_panic() {
        for src in ["3 +", "(1 + 2", "f(x = 3", "1 2", "f(1,) = 1", "[1, 2", "for in xs {}"] {
            let err = parse(src).unwrap_err();
            assert!(matches!(err.kind, ErrorKind::Parse(_)), "{src:?} -> {:?}", err.kind);
        }
    }

    #[test]
    fn array_literal_and_indexing_parse() {
        assert!(matches!(parse_one("[1, 2, 3]"), Expr::Array(v) if v.len() == 3));
        assert!(matches!(parse_one("[]"), Expr::Array(v) if v.is_empty()));
        // indexing binds tighter than operators: `xs[0] + 1` is `(xs[0]) + 1`.
        match parse_one("xs[0] + 1") {
            Expr::Binary(BinOp::Add, l, _) => assert!(matches!(l.expr, Expr::Index(_, _))),
            other => panic!("got {other:?}"),
        }
        // chained indexing: `M[i][j]` is `(M[i])[j]`.
        match parse_one("M[i][j]") {
            Expr::Index(inner, _) => assert!(matches!(inner.expr, Expr::Index(_, _))),
            other => panic!("got {other:?}"),
        }
        // a call can be indexed: `f(x)[0]`.
        match parse_one("f(x)[0]") {
            Expr::Index(inner, _) => assert!(matches!(inner.expr, Expr::Call(_, _))),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn range_parses_below_arithmetic() {
        // `0..10` is a range of two number bounds
        match parse_one("0..10") {
            Expr::Range(lo, hi) => {
                assert!(matches!(lo.expr, Expr::Number(n) if n == 0.0));
                assert!(matches!(hi.expr, Expr::Number(n) if n == 10.0));
            }
            other => panic!("got {other:?}"),
        }
        // `..` binds looser than `+`: `i + 1 .. len(xs)` is `(i+1)..(len(xs))`
        match parse_one("i + 1 .. Len(xs)") {
            Expr::Range(lo, hi) => {
                assert!(matches!(lo.expr, Expr::Binary(BinOp::Add, _, _)));
                assert!(matches!(hi.expr, Expr::Call(n, _) if n == "Len"));
            }
            other => panic!("got {other:?}"),
        }
        // a range drives a `for`
        match parse_one("for i in 0..n { i }") {
            Expr::For { iter, .. } => assert!(matches!(iter.expr, Expr::Range(_, _))),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn for_loop_parses() {
        match parse_one("for x in xs { x }") {
            Expr::For { var, iter, body } => {
                assert_eq!(var, "x");
                assert!(matches!(iter.expr, Expr::Ident(_)));
                assert!(matches!(body.expr, Expr::Block(_)));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn boolean_literals_parse() {
        assert!(matches!(parse_one("true"), Expr::Bool(true)));
        assert!(matches!(parse_one("false"), Expr::Bool(false)));
    }

    #[test]
    fn module_paths_and_use_parse() {
        // a qualified call carries the path inside the name string
        assert!(matches!(parse_one("rand::unif(0, 1)"), Expr::Call(n, _) if n == "rand::unif"));
        // a qualified constant is a (path-bearing) ident
        assert!(matches!(parse_one("math::pi"), Expr::Ident(n) if n == "math::pi"));
        // `use module;`
        assert!(matches!(parse_one("use rand"), Expr::Use(m) if m == "rand"));
        // a path can be indexed/called like any primary: `vec::range(0, 3)[1]`
        assert!(matches!(parse_one("rand::iid(d, 3)[0]"), Expr::Index(_, _)));
        // errors: dangling `::`, bad `use`
        assert!(matches!(parse("rand::").unwrap_err().kind, ErrorKind::Parse(_)));
        assert!(matches!(parse("use ;").unwrap_err().kind, ErrorKind::Parse(_)));
    }
}
