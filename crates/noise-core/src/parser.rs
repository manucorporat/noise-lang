//! Hand-written Pratt (precedence-climbing) parser. Turns tokens into a `Program`.
//!
//! Precedence (low → high), all left-associative except `^` and prefix unary:
//!   assignment `=` `~` (right-assoc, lowest)
//!   comparison `== != < > <= >=`
//!   additive   `+ -`
//!   multiplicative `* /`
//!   power `^` (right-assoc)
//!   prefix `- !`
//!   call / primary
//! See LANG.md for the canonical table.

use crate::ast::*;
use crate::error::{NoiseError, Result, Span};
use crate::lexer::{tokenize, TokKind, Token};

/// Maximum recursive-descent nesting depth. Deeply nested source — `((((…))))`, a long unary
/// `-`/`~`/`!` chain, a right-leaning `^` tower, or nested `[`/`{`/`if` — recurses the parser one
/// frame per level; without a ceiling a few thousand characters of input blow the call stack and
/// **abort the process** (worse than a panic; the wasm playground's stack is far smaller than the
/// native 8 MiB). Past this limit the parser returns a spanned error instead. Kept well below what
/// even a small (1–2 MiB) stack can hold, and far above any hand-written expression's nesting.
const MAX_PARSE_DEPTH: usize = 256;

pub fn parse(src: &str) -> Result<Program> {
    let tokens = tokenize(src)?;
    let newlines = newline_flags(src, &tokens);
    let mut p = Parser {
        tokens,
        newlines,
        pos: 0,
        depth: 0,
    };
    let stmts = p.parse_stmts(&[TokKind::Eof])?;
    p.expect(TokKind::Eof)?;
    Ok(Program { stmts })
}

/// `flags[i]` is true when a line break appears in the source between token `i-1` and token
/// `i`. The Pratt parser ignores these entirely; `parse_stmts` consults them so a newline can
/// act as an implicit statement separator (making `;` optional). Computed from token spans so
/// it costs nothing in the hot path and needs no extra token kind.
fn newline_flags(src: &str, tokens: &[Token]) -> Vec<bool> {
    let mut flags = vec![false; tokens.len()];
    for i in 1..tokens.len() {
        let gap = &src[tokens[i - 1].span.end..tokens[i].span.start];
        flags[i] = gap.contains('\n');
    }
    flags
}

struct Parser {
    tokens: Vec<Token>,
    /// Per-token "is preceded by a line break" flags; see `newline_flags`.
    newlines: Vec<bool>,
    pos: usize,
    /// Current recursive-descent nesting depth, guarded by [`MAX_PARSE_DEPTH`]. Incremented on
    /// entry to each recursive core (`parse_bp`, `parse_if`) and decremented on exit, so it tracks
    /// live stack depth rather than total nodes.
    depth: usize,
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
    /// Is the current token the start of a new source line? Used to treat a line break as an
    /// implicit statement separator.
    fn newline_before(&self) -> bool {
        self.newlines[self.pos]
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

    /// Parse statements until one of `terminators` is the next token. Statements are separated by
    /// `;` *or* a line break — whichever comes first — so `;` is only needed to put several
    /// statements on one line. A trailing separator is optional.
    ///
    /// The line break is detected *after* a complete expression: the Pratt parser ignores newlines
    /// while it is mid-expression, so an expression that genuinely continues onto the next line
    /// (e.g. a binary operator leading the line, as in `examples/turboquant.noise`) is still parsed
    /// as one statement before we reach this check.
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
            // A statement ends at `;`, a line break, or a terminator. Consume any explicit `;`;
            // a line break (or terminator) is an implicit separator and needs nothing consumed.
            if *self.peek() == TokKind::Semi {
                while *self.peek() == TokKind::Semi {
                    self.bump();
                }
            } else if !terminators.contains(self.peek()) && !self.newline_before() {
                return Err(NoiseError::parse(
                    format!(
                        "expected `;`, a line break, or end of block, found {:?}",
                        self.peek()
                    ),
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
            // `Ident = rhs` → assignment, right-associative.
            if *self.peek_at(1) == TokKind::Eq {
                let start = self.span().start;
                self.bump(); // ident
                self.bump(); // =
                let rhs = self.parse_expr()?;
                let span = Span::new(start, rhs.span.end);
                return Ok(Spanned::new(
                    Expr::Bind(BindKind::Assign, name, Box::new(rhs)),
                    span,
                ));
            }
            // `Ident ~[shape]? rhs` → a sample binding. This is sugar for `Ident = ~[shape]? rhs`:
            // we leave the `~` in place so the prefix parser builds the draw, then bind the result
            // with `=`. So the binding and the inline prefix `~` share one code path.
            if *self.peek_at(1) == TokKind::Tilde {
                let start = self.span().start;
                self.bump(); // ident — cursor now sits on `~`
                let rhs = self.parse_bp(0)?;
                let span = Span::new(start, rhs.span.end);
                return Ok(Spanned::new(
                    Expr::Bind(BindKind::Assign, name, Box::new(rhs)),
                    span,
                ));
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
                format!(
                    "expected `=` or `~` in function definition, found {:?}",
                    self.peek()
                ),
                self.span(),
            ));
        };
        let body = self.parse_expr()?;
        let span = Span::new(start, body.span.end);
        Ok(Spanned::new(
            Expr::FnDef {
                kind,
                name,
                params,
                body: Box::new(body),
            },
            span,
        ))
    }

    /// Increment the nesting depth, erroring past [`MAX_PARSE_DEPTH`]. Every recursive cycle in the
    /// descent passes through `parse_bp` (and `if … else if …` chains through `parse_if`), so
    /// guarding those two entry points bounds total stack depth. On error the depth is left
    /// incremented — harmless, since a parse error aborts the whole parse.
    fn enter(&mut self, span: Span) -> Result<()> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(NoiseError::parse(
                format!(
                    "expression nests too deeply (over {MAX_PARSE_DEPTH} levels) — deeply nested \
                     parentheses, unary operators, or `if`/`[`/`{{` blocks; simplify or split it"
                ),
                span,
            ));
        }
        Ok(())
    }

    /// Precedence-climbing core for infix operators. Depth-guarded (see [`Self::enter`]).
    fn parse_bp(&mut self, min_bp: u8) -> Result<Spanned> {
        let span = self.span();
        self.enter(span)?;
        let r = self.parse_bp_inner(min_bp);
        self.depth -= 1;
        r
    }

    fn parse_bp_inner(&mut self, min_bp: u8) -> Result<Spanned> {
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
            // `|` (conditioning bar) builds a `Cond` node — `event | given`. It binds looser than
            // every other operator (below `||`), so `A && B | C || D` is `(A && B) | (C || D)`: the
            // whole event on the left, the whole condition on the right. Only valid inside a query
            // (eval errors elsewhere); `r_bp > l_bp` keeps it left-leaning so a stray `A | B | C`
            // nests and is rejected at eval rather than silently re-conditioning.
            if *self.peek() == TokKind::Pipe {
                if COND_LBP < min_bp {
                    break;
                }
                self.bump(); // |
                let rhs = self.parse_bp(COND_RBP)?;
                let span = Span::new(lhs.span.start, rhs.span.end);
                lhs = Spanned::new(
                    Expr::Cond {
                        event: Box::new(lhs),
                        given: Box::new(rhs),
                    },
                    span,
                );
                continue;
            }
            // `@` (matrix product) builds its own `MatMul` node, not a `Binary`. It binds like
            // `*` (same precedence as in Python), left-associative.
            if *self.peek() == TokKind::At {
                if MATMUL_LBP < min_bp {
                    break;
                }
                self.bump(); // @
                let rhs = self.parse_bp(MATMUL_RBP)?;
                let span = Span::new(lhs.span.start, rhs.span.end);
                lhs = Spanned::new(Expr::MatMul(Box::new(lhs), Box::new(rhs)), span);
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
            // Prefix `~` / `~[n, m, …]` — the draw operator. `~recipe` draws one sample; a shape
            // draws a nested array of independent samples (subsumes the old `iid`/`iidmat`).
            TokKind::Tilde => {
                self.bump(); // ~
                let shape = if *self.peek() == TokKind::LBracket {
                    self.parse_shape()?
                } else {
                    Vec::new()
                };
                let body = self.parse_bp(PREFIX_BP)?;
                let span = Span::new(start, body.span.end);
                Ok(Spanned::new(
                    Expr::Sample {
                        shape,
                        body: Box::new(body),
                    },
                    span,
                ))
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
            TokKind::Template {
                body,
                syntax,
                body_offset,
            } => {
                self.bump();
                let parts = parse_template_parts(&body, body_offset)?;
                Ok(Spanned::new(Expr::Template { parts, syntax }, tok.span))
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
                    Ok(Spanned::new(
                        Expr::Ident(name),
                        Span::new(tok.span.start, end),
                    ))
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
            TokKind::Continue => {
                let tok = self.bump();
                Ok(Spanned::new(Expr::Continue, tok.span))
            }
            TokKind::LBracket => self.parse_array(),
            other => Err(NoiseError::parse(
                format!("unexpected token {:?}", other),
                tok.span,
            )),
        }
    }

    /// Parse a call's parenthesized argument list into [`CallArgs`]. A call is **either all
    /// positional or all named — never mixed** (PLAN-INPUTS §2). The two forms are told apart by a
    /// one-token lookahead: an argument that begins `IDENT :` is a *named* entry, and the moment one
    /// appears the whole list must be named. A duplicate name, or a positional entry mixed in among
    /// named ones (or vice-versa), is a spanned error.
    fn parse_call_args(&mut self) -> Result<(CallArgs, usize)> {
        self.expect(TokKind::LParen)?;
        // Empty argument list is positional (`f()`).
        if *self.peek() == TokKind::RParen {
            let close = self.bump();
            return Ok((CallArgs::Positional(Vec::new()), close.span.end));
        }

        let mut positional: Vec<Spanned> = Vec::new();
        let mut named: Vec<(String, Spanned)> = Vec::new();
        loop {
            // A named entry is `IDENT :` — but *not* `IDENT ::` (a module path). Peek two tokens.
            let is_named = matches!(&self.tokens[self.pos].kind, TokKind::Ident(_))
                && self.tokens[self.pos + 1].kind == TokKind::Colon;
            if is_named {
                let name_tok = self.bump();
                let name = match name_tok.kind {
                    TokKind::Ident(s) => s,
                    _ => unreachable!("guarded by is_named"),
                };
                self.expect(TokKind::Colon)?;
                let value = self.parse_expr()?;
                if !positional.is_empty() {
                    return Err(NoiseError::parse(
                        "cannot mix positional and named arguments in one call — use all positional \
                         or all named"
                            .to_string(),
                        Span::new(name_tok.span.start, value.span.end),
                    ));
                }
                if named.iter().any(|(n, _)| *n == name) {
                    return Err(NoiseError::parse(
                        format!("duplicate named argument `{name}`"),
                        name_tok.span,
                    ));
                }
                named.push((name, value));
            } else {
                let value = self.parse_expr()?;
                if !named.is_empty() {
                    return Err(NoiseError::parse(
                        "cannot mix positional and named arguments in one call — use all positional \
                         or all named"
                            .to_string(),
                        value.span,
                    ));
                }
                positional.push(value);
            }
            if !self.eat(&TokKind::Comma) {
                break;
            }
        }
        let close = self.expect(TokKind::RParen)?;
        let args = if named.is_empty() {
            CallArgs::Positional(positional)
        } else {
            CallArgs::Named(named)
        };
        Ok((args, close.span.end))
    }

    /// Shape list `[n, m, …]` after a prefix `~`. Like an array literal but kept as a bare
    /// `Vec<Spanned>` of dimension expressions, and required to be non-empty (`~[]` is rejected —
    /// use a bare `~` for a scalar draw).
    fn parse_shape(&mut self) -> Result<Vec<Spanned>> {
        let open = self.expect(TokKind::LBracket)?;
        let mut dims = Vec::new();
        if *self.peek() != TokKind::RBracket {
            loop {
                dims.push(self.parse_expr()?);
                if !self.eat(&TokKind::Comma) {
                    break;
                }
            }
        }
        let close = self.expect(TokKind::RBracket)?;
        if dims.is_empty() {
            return Err(NoiseError::parse(
                "empty draw shape `~[]` — use a bare `~` for a single sample".to_string(),
                Span::new(open.span.start, close.span.end),
            ));
        }
        Ok(dims)
    }

    /// Array literal `[a, b, …]` (or empty `[]`). A trailing comma is not allowed (matches the
    /// call-argument rule).
    fn parse_array(&mut self) -> Result<Spanned> {
        let open = self.expect(TokKind::LBracket)?;
        if *self.peek() == TokKind::RBracket {
            let close = self.expect(TokKind::RBracket)?;
            return Ok(Spanned::new(
                Expr::Array(Vec::new()),
                Span::new(open.span.start, close.span.end),
            ));
        }
        // `[for x in iter (if cond) { body }]` — a comprehension, detected by a leading `for`. It
        // mirrors the `for x in xs { body }` loop statement exactly, just wrapped in `[ ]` so the
        // body values are collected into an array (rather than discarded).
        if *self.peek() == TokKind::For {
            return self.parse_comprehension(open.span.start);
        }
        let mut elems = vec![self.parse_expr()?];
        while self.eat(&TokKind::Comma) {
            elems.push(self.parse_expr()?);
        }
        let close = self.expect(TokKind::RBracket)?;
        let span = Span::new(open.span.start, close.span.end);
        Ok(Spanned::new(Expr::Array(elems), span))
    }

    /// Parse a comprehension `for IDENT in opexpr { body }`, with the leading `[` already consumed
    /// (`start` is its offset) and the trailing `]` consumed here. The iterable is a bare `opexpr`
    /// (binding power 0), so it doesn't swallow the `{` that opens the body block. It is exactly the
    /// `for` loop statement wrapped in `[ ]` — a pure 1-to-1 map yielding `Len(iter)` elements.
    fn parse_comprehension(&mut self, start: usize) -> Result<Spanned> {
        self.expect(TokKind::For)?;
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
        let close = self.expect(TokKind::RBracket)?;
        let span = Span::new(start, close.span.end);
        Ok(Spanned::new(
            Expr::Comprehension {
                body: Box::new(body),
                var,
                iter: Box::new(iter),
            },
            span,
        ))
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
            Expr::For {
                var,
                iter: Box::new(iter),
                body: Box::new(body),
            },
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

    /// Depth-guarded (see [`Self::enter`]) so an `if … else if … else if …` chain — which recurses
    /// `parse_if → parse_if` without passing back through `parse_bp` between levels — can't overflow.
    fn parse_if(&mut self) -> Result<Spanned> {
        let span = self.span();
        self.enter(span)?;
        let r = self.parse_if_inner();
        self.depth -= 1;
        r
    }

    fn parse_if_inner(&mut self) -> Result<Spanned> {
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

/// Prefix operators bind tighter than everything except `^` (so `-2 ^ 2 == -(2^2)`,
/// matching common math/Python convention) and looser than `^`.
const PREFIX_BP: u8 = 13;

/// `..` (range) binding powers — the lowest of any infix form, so the bounds are full operator
/// expressions (`i + 1 .. len(xs)` parses as `(i+1)..(len(xs))`). `r_bp > l_bp` keeps it
/// left-leaning, though chained `a..b..c` is a runtime error (a range isn't a number).
const RANGE_LBP: u8 = 1;
const RANGE_RBP: u8 = 2;

/// `@` (matrix product) binding powers — same precedence as `* /`, left-associative.
const MATMUL_LBP: u8 = 11;
const MATMUL_RBP: u8 = 12;

/// `|` (conditioning bar) binding powers — the loosest infix form (below `||`), so a query's event
/// and condition are each a full operator expression. `r_bp > l_bp` keeps it left-leaning.
const COND_LBP: u8 = 1;
const COND_RBP: u8 = 2;

/// Returns `(op, left_bp, right_bp)` for an infix token. Left-assoc ops have
/// `left_bp < right_bp`; `^` is right-assoc (`left_bp > right_bp`).
/// Precedence low→high: `..` < `||` < `&&` < comparison < `+ -` < `* /` < prefix < `^`.
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
        TokKind::Percent => (Mod, 11, 12),
        TokKind::Caret => (Pow, 14, 13),
        _ => return None,
    })
}

// === Template parsing (PLAN-LITERATE §D3) ========================================================

/// Split a template `body` into literal/hole parts. `body_offset` is where `body` begins in the
/// original source, so each hole's sub-parsed expression carries true byte spans. Literal text is
/// normalized: the shared leading indentation is stripped and a single blank opening/closing line
/// (the ones adjacent to the fences) is removed.
fn parse_template_parts(body: &str, body_offset: usize) -> Result<Vec<TemplatePart>> {
    let bytes = body.as_bytes();
    // 1. Segment into literal ranges and hole (expression) ranges, both byte ranges within `body`.
    enum Seg {
        Lit(usize, usize),
        Hole(usize, usize),
    }
    let mut segs: Vec<Seg> = Vec::new();
    let mut lit_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            segs.push(Seg::Lit(lit_start, i));
            let expr_start = i + 2;
            // Find the matching `}` — track brace depth and skip `}` inside string literals.
            let mut depth = 1i32;
            let mut j = expr_start;
            let mut in_str = false;
            while j < bytes.len() {
                let b = bytes[j];
                if in_str {
                    if b == b'"' {
                        in_str = false;
                    }
                } else {
                    match b {
                        b'"' => in_str = true,
                        b'`' => {
                            return Err(NoiseError::parse(
                                "a nested template inside a `${…}` hole is not supported in v1",
                                Span::new(body_offset + j, body_offset + j + 1),
                            ))
                        }
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                j += 1;
            }
            if j >= bytes.len() {
                return Err(NoiseError::parse(
                    "unterminated `${…}` interpolation in template",
                    Span::new(body_offset + i, body_offset + bytes.len()),
                ));
            }
            segs.push(Seg::Hole(expr_start, j));
            i = j + 1;
            lit_start = i;
        } else {
            i += 1;
        }
    }
    segs.push(Seg::Lit(lit_start, bytes.len()));

    // 2. Extract literal strings + parse holes.
    let indent = common_indent(body);
    let lit_count = segs.iter().filter(|s| matches!(s, Seg::Lit(..))).count();
    let mut lits: Vec<String> = Vec::new();
    let mut parts: Vec<TemplatePart> = Vec::new();
    // Placeholder markers let us fill dedented literals back in the right slots after computing
    // which literal is first/last (for the blank-line trim).
    enum Slot {
        LitIdx(usize),
        Hole(Spanned),
    }
    let mut slots: Vec<Slot> = Vec::new();
    for seg in &segs {
        match *seg {
            Seg::Lit(a, b) => {
                slots.push(Slot::LitIdx(lits.len()));
                lits.push(body[a..b].to_string());
            }
            Seg::Hole(a, b) => {
                let expr = parse_expr_str(&body[a..b], body_offset + a)?;
                slots.push(Slot::Hole(expr));
            }
        }
    }

    // 3. Normalize literals: trim the opening/closing blank line, then dedent.
    if let Some(first) = lits.first_mut() {
        *first = trim_leading_blank_line(first);
    }
    if let Some(last) = lits.last_mut() {
        *last = trim_trailing_blank_line(last);
    }
    for (idx, lit) in lits.iter_mut().enumerate() {
        *lit = dedent(lit, indent, idx == 0);
    }
    let _ = lit_count;

    // 4. Reassemble in source order, dropping empty literals (they render to nothing).
    for slot in slots {
        match slot {
            Slot::LitIdx(k) => {
                if !lits[k].is_empty() {
                    parts.push(TemplatePart::Lit(std::mem::take(&mut lits[k])));
                }
            }
            Slot::Hole(e) => parts.push(TemplatePart::Hole(e)),
        }
    }
    Ok(parts)
}

/// The shared leading-whitespace width across `body`'s non-blank physical lines (0 if none).
fn common_indent(body: &str) -> usize {
    let mut min: Option<usize> = None;
    for line in body.split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        let w = line.len() - line.trim_start_matches([' ', '\t']).len();
        min = Some(min.map_or(w, |m| m.min(w)));
    }
    min.unwrap_or(0)
}

/// Remove up to `d` leading whitespace chars after each newline (and, when `at_start`, at the very
/// beginning) — the dedent that strips a template's shared indentation.
fn dedent(s: &str, d: usize, at_start: bool) -> String {
    if d == 0 {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut skip = if at_start { d } else { 0 };
    for ch in s.chars() {
        if ch == '\n' {
            out.push(ch);
            skip = d;
        } else if skip > 0 && (ch == ' ' || ch == '\t') {
            skip -= 1;
        } else {
            skip = 0;
            out.push(ch);
        }
    }
    out
}

/// If everything before the first newline is whitespace, drop through that newline (the blank
/// opening line next to the fence). Otherwise leave the string unchanged.
fn trim_leading_blank_line(s: &str) -> String {
    if let Some(nl) = s.find('\n') {
        if s[..nl].trim().is_empty() {
            return s[nl + 1..].to_string();
        }
    }
    s.to_string()
}

/// If everything after the last newline is whitespace, drop from that newline (the blank closing
/// line next to the fence). Otherwise leave the string unchanged.
fn trim_trailing_blank_line(s: &str) -> String {
    if let Some(nl) = s.rfind('\n') {
        if s[nl + 1..].trim().is_empty() {
            return s[..nl].to_string();
        }
    }
    s.to_string()
}

/// Parse a single expression from `src`, shifting all spans by `base_offset` so diagnostics point at
/// the original source (used for `${…}` template holes). The whole slice must be one expression.
fn parse_expr_str(src: &str, base_offset: usize) -> Result<Spanned> {
    let mut tokens = tokenize(src)?;
    let newlines = newline_flags(src, &tokens);
    for t in &mut tokens {
        t.span = Span::new(t.span.start + base_offset, t.span.end + base_offset);
    }
    let mut p = Parser {
        tokens,
        newlines,
        pos: 0,
        depth: 0,
    };
    let e = p.parse_expr()?;
    p.expect(TokKind::Eof)?;
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    fn parse_one(src: &str) -> Expr {
        let mut prog = parse(src).unwrap();
        assert_eq!(
            prog.stmts.len(),
            1,
            "expected exactly one statement in {src:?}"
        );
        prog.stmts.pop().unwrap().expr
    }

    #[test]
    fn power_is_right_associative() {
        // 2 ^ 3 ^ 2  ==>  2 ^ (3 ^ 2)
        match parse_one("2 ^ 3 ^ 2") {
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
    fn modulo_binds_like_multiplication() {
        // 1 + 7 % 3  ==>  1 + (7 % 3)
        match parse_one("1 + 7 % 3") {
            Expr::Binary(BinOp::Add, _, r) => {
                assert!(matches!(r.expr, Expr::Binary(BinOp::Mod, _, _)))
            }
            other => panic!("got {other:?}"),
        }
        // 2 ^ 3 % 5  ==>  (2 ^ 3) % 5  (^ binds tighter than %)
        match parse_one("2 ^ 3 % 5") {
            Expr::Binary(BinOp::Mod, l, _) => {
                assert!(matches!(l.expr, Expr::Binary(BinOp::Pow, _, _)))
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn comprehensions_parse() {
        // `[for x in iter { body }]` is a Comprehension, not an Array.
        match parse_one("[for x in 0..5 { x*x }]") {
            Expr::Comprehension { var, iter, body } => {
                assert_eq!(var, "x");
                assert!(matches!(iter.expr, Expr::Range(_, _)));
                assert!(matches!(body.expr, Expr::Block(_)));
            }
            other => panic!("got {other:?}"),
        }
        // a comma-separated list is still a plain array (not a comprehension)
        assert!(matches!(parse_one("[1, 2, 3]"), Expr::Array(v) if v.len() == 3));
        // a single-element array (no `for`) is still an array
        assert!(matches!(parse_one("[42]"), Expr::Array(v) if v.len() == 1));
        // `continue` is a primary expression
        assert!(matches!(parse_one("continue"), Expr::Continue));
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
        assert!(matches!(
            parse_one("f(x) = x"),
            Expr::FnDef {
                kind: BindKind::Assign,
                ..
            }
        ));
        assert!(matches!(
            parse_one("g() ~ unif(0,1)"),
            Expr::FnDef {
                kind: BindKind::Sample,
                ..
            }
        ));
        // a bare `f(x)` (no following `=`/`~`) is a call expression, not a definition
        assert!(matches!(parse_one("f(x)"), Expr::Call(_, _)));
    }

    #[test]
    fn call_args_are_positional_or_named() {
        // all-positional → Positional
        match parse_one("f(1, 2)") {
            Expr::Call(_, CallArgs::Positional(a)) => assert_eq!(a.len(), 2),
            other => panic!("got {other:?}"),
        }
        // empty → Positional
        assert!(matches!(parse_one("f()"), Expr::Call(_, CallArgs::Positional(a)) if a.is_empty()));
        // all-named → Named, keeps source order and names
        match parse_one("f(b: 2, a: 1)") {
            Expr::Call(_, CallArgs::Named(a)) => {
                assert_eq!(
                    a.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                    ["b", "a"]
                );
            }
            other => panic!("got {other:?}"),
        }
        // a `::` path in argument position is NOT mistaken for a named entry
        assert!(matches!(
            parse_one("f(rand::pi)"),
            Expr::Call(_, CallArgs::Positional(_))
        ));
    }

    #[test]
    fn mixed_and_duplicate_named_args_are_errors() {
        // positional then named
        assert!(matches!(
            parse("f(1, b: 2)").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
        // named then positional
        assert!(matches!(
            parse("f(a: 1, 2)").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
        // duplicate name
        assert!(matches!(
            parse("f(a: 1, a: 2)").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
    }

    #[test]
    fn parse_errors_are_typed_and_dont_panic() {
        for src in [
            "3 +",
            "(1 + 2",
            "f(x = 3",
            "1 2",
            "f(1,) = 1",
            "[1, 2",
            "for in xs {}",
        ] {
            let err = parse(src).unwrap_err();
            assert!(
                matches!(err.kind, ErrorKind::Parse(_)),
                "{src:?} -> {:?}",
                err.kind
            );
        }
    }

    #[test]
    fn deep_nesting_errors_instead_of_overflowing_the_stack() {
        // A few thousand nested parens used to recurse the parser one frame per level and abort the
        // process (SIGABRT / stack overflow). The depth guard now turns each of these into a clean
        // parse error. We run on a generously-sized thread: *unoptimized* (debug) parser frames are
        // large, so reaching the ~256-level guard needs several MiB — more than the 2 MiB default
        // test-harness thread. The guard value is production-safe (it survives a 512 KiB release
        // stack, well within the wasm playground's ~1 MiB). A *regression* (guard removed) either
        // overflows even this thread (aborting the binary) or parses the balanced input to `Ok`
        // (failing `unwrap_err`), so the guard is still genuinely exercised.
        let body = || {
            let deep_parens = format!("{}1{}", "(".repeat(2000), ")".repeat(2000));
            assert!(matches!(
                parse(&deep_parens).unwrap_err().kind,
                ErrorKind::Parse(_)
            ));
            // Deep unary chains (`-`, `~`, `!`) recurse through the same `parse_bp` core.
            assert!(matches!(
                parse(&format!("{}1", "-".repeat(4000))).unwrap_err().kind,
                ErrorKind::Parse(_)
            ));
            assert!(matches!(
                parse(&format!("{}1", "!".repeat(4000))).unwrap_err().kind,
                ErrorKind::Parse(_)
            ));
            // A deep `if … else if …` chain recurses `parse_if`; must return, never overflow.
            let deep_if = format!("{}{{}}", "if 1 {} else ".repeat(4000));
            let _ = parse(&deep_if);
            // A moderate, realistic nesting still parses fine.
            assert!(parse(&format!("{}1{}", "(".repeat(100), ")".repeat(100))).is_ok());
        };
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(body)
            .unwrap()
            .join()
            .unwrap();
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
    fn prefix_sample_and_shape_parse() {
        // bare `~e` is a shapeless draw
        match parse_one("~normal(0, 1)") {
            Expr::Sample { shape, body } => {
                assert!(shape.is_empty());
                assert!(matches!(body.expr, Expr::Call(_, _)));
            }
            other => panic!("got {other:?}"),
        }
        // `~[n, m] e` carries a shape list
        match parse_one("~[a, 3] normal(0, 1)") {
            Expr::Sample { shape, .. } => assert_eq!(shape.len(), 2),
            other => panic!("got {other:?}"),
        }
        // the statement form `x ~[3] e` is sugar for `x = ~[3] e`
        match parse_one("x ~[3] unif(0, 1)") {
            Expr::Bind(BindKind::Assign, name, rhs) => {
                assert_eq!(name, "x");
                assert!(matches!(rhs.expr, Expr::Sample { .. }));
            }
            other => panic!("got {other:?}"),
        }
        // `~[]` (empty shape) is a parse error
        assert!(matches!(
            parse("xs ~[] unif(0, 1)").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
    }

    #[test]
    fn matmul_operator_parses_like_star() {
        // `a @ b` is a MatMul node, not a Binary
        assert!(matches!(parse_one("a @ b"), Expr::MatMul(_, _)));
        // `@` binds like `*` (tighter than `+`): `1 + a @ b` is `1 + (a @ b)`
        match parse_one("1 + a @ b") {
            Expr::Binary(BinOp::Add, _, r) => assert!(matches!(r.expr, Expr::MatMul(_, _))),
            other => panic!("got {other:?}"),
        }
        // left-associative: `a @ b @ c` is `(a @ b) @ c`
        match parse_one("a @ b @ c") {
            Expr::MatMul(l, _) => assert!(matches!(l.expr, Expr::MatMul(_, _))),
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
    fn conditioning_bar_parses_below_or() {
        // `a | b` builds a `Cond`, not a `Binary`.
        assert!(matches!(parse_one("a | b"), Expr::Cond { .. }));
        // `|` binds looser than `||`: `a || b | c || d` is `(a||b) | (c||d)`.
        match parse_one("a || b | c || d") {
            Expr::Cond { event, given } => {
                assert!(matches!(event.expr, Expr::Binary(BinOp::Or, _, _)));
                assert!(matches!(given.expr, Expr::Binary(BinOp::Or, _, _)));
            }
            other => panic!("got {other:?}"),
        }
        // inside a query call it is a single argument: `P(D == 6 | D > 3)`.
        match parse_one("P(D == 6 | D > 3)") {
            Expr::Call(n, CallArgs::Positional(args)) => {
                assert_eq!(n, "P");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0].expr, Expr::Cond { .. }));
            }
            other => panic!("got {other:?}"),
        }
        // a binding can hold a conditioned value: `a = X | C`.
        match parse_one("a = D == 6 | D > 3") {
            Expr::Bind(BindKind::Assign, name, rhs) => {
                assert_eq!(name, "a");
                assert!(matches!(rhs.expr, Expr::Cond { .. }));
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
    fn newlines_separate_statements_so_semicolons_are_optional() {
        // a line break ends a statement just like `;`
        let prog = parse("a = 1\nb = 2\na + b").unwrap();
        assert_eq!(prog.stmts.len(), 3);
        // `;` still works, including several statements on one line
        let prog = parse("a = 1; b = 2\nc = 3").unwrap();
        assert_eq!(prog.stmts.len(), 3);
        // blank lines and comment-only lines don't create empty statements
        let prog = parse("a = 1\n\n# note\nb = 2\n").unwrap();
        assert_eq!(prog.stmts.len(), 2);
        // line breaks separate statements inside a block, too
        match parse_one("{ a = 1\n b = 2\n a + b }") {
            Expr::Block(stmts) => assert_eq!(stmts.len(), 3),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn a_leading_operator_continues_the_previous_line() {
        // the Pratt parser ignores newlines mid-expression, so an operator at the start of the
        // next line continues the statement rather than starting a new one (turboquant style).
        let prog = parse("total = a\n + b\n * c").unwrap();
        assert_eq!(prog.stmts.len(), 1);
        match &prog.stmts[0].expr {
            Expr::Bind(BindKind::Assign, name, rhs) => {
                assert_eq!(name, "total");
                assert!(matches!(rhs.expr, Expr::Binary(BinOp::Add, _, _)));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn two_values_on_one_line_without_a_separator_is_still_an_error() {
        // `1 2` (same line, no `;`) must stay an error — the newline rule shouldn't mask it.
        assert!(matches!(
            parse("1 2").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
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
        assert!(matches!(
            parse_one("rand::normal(0, 1)[0]"),
            Expr::Index(_, _)
        ));
        // errors: dangling `::`, bad `use`
        assert!(matches!(
            parse("rand::").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
        assert!(matches!(
            parse("use ;").unwrap_err().kind,
            ErrorKind::Parse(_)
        ));
    }
}
