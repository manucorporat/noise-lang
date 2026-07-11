//! The literate **document model** (PLAN-LITERATE §D4/§D5). A run no longer returns a bag of
//! `{ value, output, log }`; it returns exactly one [`Document`]: frontmatter meta + one flat,
//! ordered array of typed [`Block`]s (code / notes / plots) in emission order, plus a **comment
//! layer** (each comment a `(self_span, code_span?)` pair) and the run [`DocResult`]. Every host
//! (CLI, wasm) renders *this*, and every view (only-plots, hide-code, full literate) is a pure
//! filter over the one array — so the CLI and the playground can't drift.
//!
//! This module is pure segmentation + interleaving: [`segment`] and [`comment_layer`] read the
//! parsed statements and the source (no evaluation); [`assemble`] threads the engine's span-tagged
//! emissions in after the code block whose span contains their `stmt_span`. The evaluator never sees
//! a comment.

use crate::ast::{Expr, Spanned};
use crate::error::{NoiseError, Result, Span};
use crate::eval::{Emission, Output};
use crate::frontmatter::Frontmatter;
use crate::input::{InputSpec, InputValue, ResolvedInput};
use crate::stats::RunStats;
use crate::value::Value;

/// The one structure a run produces (PLAN-LITERATE §D5).
#[derive(Debug, Clone)]
pub struct Document {
    pub meta: Option<Frontmatter>,
    /// One flat array — source segments (code) and emissions (notes, plots) alike, in the order a
    /// reader meets them.
    pub blocks: Vec<Block>,
    /// The annotation **layer** — not inside any block.
    pub comments: Vec<Comment>,
    pub result: DocResult,
}

/// A block in the flat array. `Code` is a verbatim source segment; `Note`/`Plot` are emissions,
/// each carrying the `stmt_span` of the statement that produced it (so a host can group them under
/// their code block or highlight the producing line on hover).
#[derive(Debug, Clone)]
pub enum Block {
    Code { source: String, span: Span },
    Note { text: String, syntax: Option<String>, stmt_span: Span },
    Plot { title: String, text: String, charts: Vec<serde_json::Value>, stmt_span: Span },
    /// An inline **input control** (PLAN-INPUTS §4): a host-tunable parameter declared with
    /// `input::…`, rendered as a slider/checkbox right after the code group that declares it.
    Input { spec: InputSpec, value: InputValue, stmt_span: Span },
}

/// One comment in the layer. `self_span` is where the comment text lives in the source; `code_span`
/// is the code it annotates (one line, a whole group — any statement range). An **absent**
/// `code_span` is a detached run: free-standing prose positioned by `self_span`.
#[derive(Debug, Clone)]
pub struct Comment {
    pub text: String,
    pub self_span: Span,
    pub code_span: Option<Span>,
}

#[derive(Debug, Clone)]
pub struct DocResult {
    /// The program's final value as `{ kind, text }`; absent when the program ends in `unit`.
    pub value: Option<DocValue>,
    /// A lex/parse/runtime failure, spanned. The blocks up to the failure are still present.
    pub error: Option<DocError>,
    pub stats: RunStats,
    /// Set when the run hit the emission cap: how many blocks were dropped and where it first hit.
    pub truncated: Option<Truncated>,
    /// The run's **input manifest** (PLAN-INPUTS §3): every `input::` declared, resolved, in
    /// declaration order. A host reads this to render controls and prune stale overrides. Empty
    /// when the program declares no inputs.
    pub inputs: Vec<ResolvedInput>,
}

#[derive(Debug, Clone)]
pub struct DocValue {
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct DocError {
    pub message: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Truncated {
    pub dropped: usize,
    pub first_dropped_stmt_span: Span,
}

/// A source segment produced by [`segment`]: a code group, or a lone root template statement (which
/// splits the surrounding group and renders as its own note block).
#[derive(Debug, Clone, Copy)]
pub enum Seg {
    /// A statement group — consecutive top-level statements with no blank line between.
    Code(Span),
    /// A root template statement (its text comes from the matching emission at assemble time).
    Template(Span),
}

impl Document {
    /// A best-effort document for a lex/parse failure before evaluation: meta (if it parsed), no
    /// blocks or comments, and the spanned error.
    pub fn error_only(meta: Option<Frontmatter>, error: NoiseError, stats: RunStats) -> Document {
        Document {
            meta,
            blocks: Vec::new(),
            comments: Vec::new(),
            result: DocResult {
                value: None,
                error: Some(DocError { message: error.to_string(), span: error.span }),
                stats,
                truncated: None,
                inputs: Vec::new(),
            },
        }
    }
}

// === segmentation (D4) ===========================================================================

/// Partition top-level statements into code groups and lone template statements. A group breaks
/// only on a **blank line between statement spans** or a **template statement** (per §D4 —
/// comments never split a group). Span-based, not raw-line-based: a blank line *inside* a statement
/// (an unfinished expression) does not split it.
pub fn segment(src: &str, stmts: &[Spanned]) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut group: Option<(usize, usize)> = None; // (start, end) byte offsets
    let mut prev_end: Option<usize> = None;
    for stmt in stmts {
        let is_template = matches!(stmt.expr, Expr::Template { .. });
        let breaks = match prev_end {
            Some(pe) => gap_has_blank_line(&src[pe..stmt.span.start]),
            None => false,
        };
        if is_template {
            if let Some((s, e)) = group.take() {
                segs.push(Seg::Code(Span::new(s, e)));
            }
            segs.push(Seg::Template(stmt.span));
        } else {
            match &mut group {
                Some((_s, e)) if !breaks => *e = stmt.span.end,
                _ => {
                    if let Some((s, e)) = group.take() {
                        segs.push(Seg::Code(Span::new(s, e)));
                    }
                    group = Some((stmt.span.start, stmt.span.end));
                }
            }
        }
        prev_end = Some(stmt.span.end);
    }
    if let Some((s, e)) = group.take() {
        segs.push(Seg::Code(Span::new(s, e)));
    }
    segs
}

/// Does the gap between two statement spans contain a **blank line** — an interior line that is
/// entirely whitespace? A comment-only line is *not* blank (comments don't split groups), so we do
/// **not** strip comments here: only genuinely empty interior lines count.
fn gap_has_blank_line(gap: &str) -> bool {
    let segs: Vec<&str> = gap.split('\n').collect();
    if segs.len() < 3 {
        return false; // need ≥2 newlines to bracket a full interior line
    }
    segs[1..segs.len() - 1].iter().any(|s| s.trim().is_empty())
}

// === comment attachment (D4) =====================================================================

/// A byte-offset → line-number index (0-based), for the line-based attachment rules.
struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    fn new(src: &str) -> LineIndex {
        let mut starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        LineIndex { starts }
    }
    fn line(&self, offset: usize) -> usize {
        match self.starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }
    fn line_start(&self, line: usize) -> usize {
        self.starts[line]
    }
}

/// Build the comment layer (PLAN-LITERATE §D4). Re-lexes `src` for comment spans (trivia), then
/// applies the attachment rules as pure functions of line numbers:
/// - a **trailing** comment (code before it on its line) annotates that statement;
/// - a contiguous run of **own-line** comments annotates the statements from the next statement down
///   to the next interruption (a blank line, a template, or another own-line comment run);
/// - a blank line between the run and the code **detaches** it (no `code_span`).
pub fn comment_layer(src: &str, stmts: &[Spanned]) -> Result<Vec<Comment>> {
    let (_toks, comment_spans) = crate::lexer::tokenize_with_trivia(src)?;
    let li = LineIndex::new(src);

    // Statement line ranges, in source order.
    let stmt_lines: Vec<(usize, usize)> =
        stmts.iter().map(|s| (li.line(s.span.start), li.line(s.span.end))).collect();
    let is_template: Vec<bool> =
        stmts.iter().map(|s| matches!(s.expr, Expr::Template { .. })).collect();

    // Classify comments.
    struct C {
        span: Span,
        line: usize,
        own_line: bool,
    }
    let cs: Vec<C> = comment_spans
        .iter()
        .map(|&span| {
            let line = li.line(span.start);
            let before = &src[li.line_start(line)..span.start];
            C { span, line, own_line: before.trim().is_empty() }
        })
        .collect();

    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = &cs[i];
        if !c.own_line {
            // Trailing: annotate the statement whose line range covers this comment's line.
            let code_span = stmts
                .iter()
                .zip(&stmt_lines)
                .find(|(_, (a, b))| c.line >= *a && c.line <= *b)
                .map(|(s, _)| s.span);
            out.push(Comment {
                text: src[c.span.start..c.span.end].to_string(),
                self_span: c.span,
                code_span,
            });
            i += 1;
            continue;
        }
        // Own-line: gather the contiguous run (comments on consecutive lines).
        let run_start = i;
        let mut last_line = c.line;
        let mut j = i + 1;
        while j < cs.len() && cs[j].own_line && cs[j].line == last_line + 1 {
            last_line = cs[j].line;
            j += 1;
        }
        // Reach: the run attaches to statements starting just below its last line.
        let code_span = run_reach(last_line, stmts, &stmt_lines, &is_template);
        for c in &cs[run_start..j] {
            out.push(Comment {
                text: src[c.span.start..c.span.end].to_string(),
                self_span: c.span,
                code_span,
            });
        }
        i = j;
    }
    Ok(out)
}

/// The code span a leading own-line comment run (last comment on `last_line`) annotates, or `None`
/// when a blank line detaches it. The run reaches from the next statement through the following
/// statements while they stay contiguous — no blank line, no template, no intervening own-line
/// comment run (approximated here by the *statement* start lines, which is what those interruptions
/// move).
fn run_reach(
    last_line: usize,
    stmts: &[Spanned],
    stmt_lines: &[(usize, usize)],
    is_template: &[bool],
) -> Option<Span> {
    // The first statement starting below the run.
    let first = stmt_lines.iter().position(|(a, _)| *a > last_line)?;
    // Detached: the next statement does not immediately follow (a blank line sits between).
    if stmt_lines[first].0 != last_line + 1 {
        return None;
    }
    if is_template[first] {
        // A template splits the group; a run directly above one annotates just it.
        return Some(stmts[first].span);
    }
    // Extend through following statements while each starts on the line right after the previous
    // one ends (contiguous, no blank line / comment run / template between).
    let mut end_idx = first;
    while end_idx + 1 < stmts.len() {
        let next = end_idx + 1;
        let contiguous = stmt_lines[next].0 == stmt_lines[end_idx].1 + 1;
        if !contiguous || is_template[next] {
            break;
        }
        end_idx = next;
    }
    Some(Span::new(stmts[first].span.start, stmts[end_idx].span.end))
}

// === assembly (D5) ===============================================================================

/// Interleave the engine's span-tagged emissions into the source segments: each code group becomes
/// a `Code` block followed by the notes/plots whose `stmt_span` falls inside it; each root template
/// statement becomes its own `Note` block (from its matching emission). Order in `blocks` *is* the
/// reconciliation — no matching step beyond span containment.
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    src: &str,
    meta: Option<Frontmatter>,
    segs: Vec<Seg>,
    comments: Vec<Comment>,
    emissions: Vec<Emission>,
    inputs: Vec<ResolvedInput>,
    last: Value,
    error: Option<NoiseError>,
    stats: RunStats,
    truncated: Option<(usize, Span)>,
) -> Document {
    // Emissions in order; each consumed by the first segment whose span contains its stmt_span.
    let mut used = vec![false; emissions.len()];
    let mut blocks = Vec::new();

    let push_emission = |blocks: &mut Vec<Block>, em: &Emission| match &em.output {
        Output::Text(t) => {
            blocks.push(Block::Note { text: t.clone(), syntax: None, stmt_span: em.stmt_span })
        }
        Output::Note { text, syntax } => blocks.push(Block::Note {
            text: text.clone(),
            syntax: syntax.clone(),
            stmt_span: em.stmt_span,
        }),
        Output::Plot(s) => {
            let p = crate::flint::to_flint(s);
            blocks.push(Block::Plot {
                title: p.title,
                text: p.text,
                charts: p.charts,
                stmt_span: em.stmt_span,
            });
        }
        Output::Input { spec, value } => blocks.push(Block::Input {
            spec: spec.clone(),
            value: *value,
            stmt_span: em.stmt_span,
        }),
    };

    for seg in &segs {
        match *seg {
            Seg::Code(span) => {
                blocks.push(Block::Code { source: src[span.start..span.end].to_string(), span });
                for (k, em) in emissions.iter().enumerate() {
                    if !used[k] && contains(span, em.stmt_span) {
                        used[k] = true;
                        push_emission(&mut blocks, em);
                    }
                }
            }
            Seg::Template(span) => {
                // The template's own note (and any plot it somehow produced) — no Code block.
                for (k, em) in emissions.iter().enumerate() {
                    if !used[k] && contains(span, em.stmt_span) {
                        used[k] = true;
                        push_emission(&mut blocks, em);
                    }
                }
            }
        }
    }
    // Any stragglers (defensive — every emission's stmt_span should sit in some segment).
    for (k, em) in emissions.iter().enumerate() {
        if !used[k] {
            push_emission(&mut blocks, em);
        }
    }

    let value = match (&error, &last) {
        (Some(_), _) => None,
        (None, Value::Unit) => None,
        (None, v) => Some(DocValue { kind: value_kind(v).to_string(), text: v.to_string() }),
    };
    let doc_error =
        error.map(|e| DocError { message: e.to_string(), span: e.span });
    let truncated = truncated.map(|(dropped, span)| Truncated {
        dropped,
        first_dropped_stmt_span: span,
    });

    Document {
        meta,
        blocks,
        comments,
        result: DocResult { value, error: doc_error, stats, truncated, inputs },
    }
}

/// Whole-containment of `inner` within `outer` (byte spans).
fn contains(outer: Span, inner: Span) -> bool {
    inner.start >= outer.start && inner.end <= outer.end
}

/// A styling tag for `result.value` — the same buckets a UI would color by.
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Num(_) => "num",
        Value::Est { .. } => "est",
        Value::Bool(_) => "bool",
        Value::Str(_) => "str",
        Value::Dist(_) => "dist",
        Value::Array(_) => "array",
        _ => v.type_name(),
    }
}

// === JSON wire format ============================================================================

impl Document {
    /// Serialize to the kind-tagged JSON the hosts consume: `{ meta, blocks, comments, result }`
    /// where each block is `{ "kind": "code" | "note" | "plot" | "input", … }`.
    pub fn to_json(&self) -> serde_json::Value {
        let meta = match &self.meta {
            Some(fm) => fm.to_json(),
            None => serde_json::json!({ "tags": [], "extra": {} }),
        };
        let blocks: Vec<serde_json::Value> = self.blocks.iter().map(Block::to_json).collect();
        let comments: Vec<serde_json::Value> = self.comments.iter().map(Comment::to_json).collect();
        serde_json::json!({
            "meta": meta,
            "blocks": blocks,
            "comments": comments,
            "result": self.result.to_json(),
        })
    }
}

fn span_json(s: Span) -> serde_json::Value {
    serde_json::json!({ "start": s.start, "end": s.end })
}

impl Block {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Block::Code { source, span } => serde_json::json!({
                "kind": "code",
                "source": source,
                "span": span_json(*span),
            }),
            Block::Note { text, syntax, stmt_span } => serde_json::json!({
                "kind": "note",
                "text": text,
                "syntax": syntax,
                "stmt_span": span_json(*stmt_span),
            }),
            Block::Plot { title, text, charts, stmt_span } => serde_json::json!({
                "kind": "plot",
                "title": title,
                "text": text,
                "charts": charts,
                "stmt_span": span_json(*stmt_span),
            }),
            Block::Input { spec, value, stmt_span } => {
                let mut m = match spec.to_json_entry(*value, *stmt_span) {
                    serde_json::Value::Object(m) => m,
                    _ => serde_json::Map::new(),
                };
                m.insert("kind".into(), serde_json::json!("input"));
                serde_json::Value::Object(m)
            }
        }
    }
}

impl Comment {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "text": self.text,
            "self_span": span_json(self.self_span),
            "code_span": self.code_span.map(span_json),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;

    fn doc(src: &str) -> Document {
        Engine::new().run_to_document(src)
    }
    fn kinds(d: &Document) -> Vec<&'static str> {
        d.blocks
            .iter()
            .map(|b| match b {
                Block::Code { .. } => "code",
                Block::Note { .. } => "note",
                Block::Plot { .. } => "plot",
                Block::Input { .. } => "input",
            })
            .collect()
    }
    fn note_texts(d: &Document) -> Vec<String> {
        d.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Note { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn interleaves_code_note_plot_in_source_order() {
        let d = doc("X ~ rand::unif_int(1, 6)\nPrint(\"hi\")\nplot::histogram(X)");
        assert_eq!(kinds(&d), vec!["code", "note", "plot"]);
        assert_eq!(note_texts(&d), vec!["hi"]);
    }

    #[test]
    fn input_declaration_emits_an_inline_control_block() {
        // `input::` renders as its own block right after the code group that declares it, and shows
        // up in the run manifest (PLAN-INPUTS §3/§4).
        let d = doc("n = input::real(min: 1, max: 10, default: 4)\nx = n + 1");
        assert_eq!(kinds(&d), vec!["code", "input"]);
        match &d.blocks[1] {
            Block::Input { spec, value, .. } => {
                assert_eq!(spec.name, "n");
                assert_eq!(*value, crate::input::InputValue::Num(4.0));
            }
            other => panic!("expected an input block, got {other:?}"),
        }
        assert_eq!(d.result.inputs.len(), 1);
        assert_eq!(d.result.inputs[0].spec.name, "n");
    }

    #[test]
    fn template_statement_splits_the_group_into_its_own_note() {
        let d = doc("x = 42\n`answer ${x}`\ny = 1");
        assert_eq!(kinds(&d), vec!["code", "note", "code"]);
        assert_eq!(note_texts(&d), vec!["answer 42"]);
    }

    // --- comment attachment (D4) ---

    #[test]
    fn leading_run_annotates_whole_group() {
        let d = doc("# doc\na = 1\nb = 2");
        assert_eq!(d.comments.len(), 1);
        let cs = d.comments[0].code_span.unwrap();
        // Spans the whole group: from `a` to the end of `b`.
        assert_eq!(&"# doc\na = 1\nb = 2"[cs.start..cs.end], "a = 1\nb = 2");
    }

    #[test]
    fn interleaved_runs_annotate_their_own_line() {
        let src = "# c1\na = 1\n# c2\nb = 2";
        let d = doc(src);
        assert_eq!(d.comments.len(), 2);
        assert_eq!(&src[d.comments[0].code_span.unwrap().start..d.comments[0].code_span.unwrap().end], "a = 1");
        assert_eq!(&src[d.comments[1].code_span.unwrap().start..d.comments[1].code_span.unwrap().end], "b = 2");
    }

    #[test]
    fn blank_line_detaches_a_comment_run() {
        let d = doc("# note\n\na = 1");
        assert_eq!(d.comments.len(), 1);
        assert_eq!(d.comments[0].code_span, None, "a blank line detaches");
        assert_eq!(d.comments[0].text, "# note");
    }

    #[test]
    fn trailing_comment_annotates_its_own_statement() {
        let src = "a = 1 # trailing";
        let d = doc(src);
        assert_eq!(d.comments.len(), 1);
        let cs = d.comments[0].code_span.unwrap();
        assert_eq!(&src[cs.start..cs.end], "a = 1");
    }

    #[test]
    fn mid_group_comment_does_not_split_the_code_block() {
        // One code block (only blank lines / templates split groups); the comment still annotates b.
        let d = doc("a = 1\n# mid\nb = 2");
        assert_eq!(kinds(&d), vec!["code"]);
        let src = "a = 1\n# mid\nb = 2";
        let cs = d.comments[0].code_span.unwrap();
        assert_eq!(&src[cs.start..cs.end], "b = 2");
    }

    // --- indirect emission (D5): attribution to the call site's root statement ---

    #[test]
    fn print_in_a_function_attributes_to_the_call_site() {
        let src = "f(z) = Print(z)\nf(7)";
        let d = doc(src);
        assert_eq!(note_texts(&d), vec!["7"]);
        // The note's stmt_span is the call site `f(7)`, not the definition.
        let call = src.find("f(7)").unwrap();
        match &d.blocks[1] {
            Block::Note { stmt_span, .. } => assert_eq!(stmt_span.start, call),
            other => panic!("expected a note, got {other:?}"),
        }
    }

    #[test]
    fn emissions_split_across_two_call_sites() {
        // Blank lines put each call in its own group; each note lands after its own call block.
        let d = doc("f(z) = Print(z)\n\nf(1)\n\nf(2)");
        assert_eq!(kinds(&d), vec!["code", "code", "note", "code", "note"]);
        assert_eq!(note_texts(&d), vec!["1", "2"]);
    }

    #[test]
    fn root_loop_repeats_notes_sharing_one_stmt_span() {
        let d = doc("for i in 0..3 { Print(i) }");
        assert_eq!(note_texts(&d), vec!["0", "1", "2"]);
        let spans: Vec<Span> = d
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Note { stmt_span, .. } => Some(*stmt_span),
                _ => None,
            })
            .collect();
        assert!(spans.windows(2).all(|w| w[0] == w[1]), "all share one stmt_span");
    }

    #[test]
    fn emission_cap_truncates_but_keeps_running() {
        let d = doc("for i in 0..250 { Print(i) }");
        assert_eq!(note_texts(&d).len(), crate::eval::MAX_EMISSIONS);
        let t = d.result.truncated.expect("should be truncated");
        assert_eq!(t.dropped, 250 - crate::eval::MAX_EMISSIONS);
    }

    #[test]
    fn nested_calls_attribute_to_the_root_statement() {
        let src = "g(z) = Print(z)\nf(z) = g(z)\nf(5)";
        let d = doc(src);
        assert_eq!(note_texts(&d), vec!["5"]);
        let root = src.find("f(5)").unwrap();
        match d.blocks.iter().find(|b| matches!(b, Block::Note { .. })).unwrap() {
            Block::Note { stmt_span, .. } => assert_eq!(stmt_span.start, root),
            _ => unreachable!(),
        }
    }

    // --- errors keep the document ---

    #[test]
    fn runtime_error_keeps_prior_blocks_and_spans_the_failure() {
        let src = "Print(\"a\")\nundefined_thing\nPrint(\"b\")";
        let d = doc(src);
        assert_eq!(note_texts(&d), vec!["a"], "only emissions before the failure survive");
        let e = d.result.error.expect("a runtime error");
        assert_eq!(e.span.start, src.find("undefined_thing").unwrap());
    }

    #[test]
    fn parse_error_yields_best_effort_document_with_meta() {
        let d = doc("---\ntitle: T\n---\n1 +");
        assert!(d.meta.is_some(), "frontmatter still parsed");
        assert_eq!(d.meta.as_ref().unwrap().title.as_deref(), Some("T"));
        assert!(d.blocks.is_empty());
        assert!(d.result.error.is_some());
    }

    #[test]
    fn result_value_carries_kind_and_display_text() {
        let d = doc("2 + 3");
        let v = d.result.value.expect("a value");
        assert_eq!(v.kind, "num");
        assert_eq!(v.text, "5");
    }

    #[test]
    fn unit_result_has_no_value() {
        // Ends in a plot (unit) → no echoed value, same as the CLI's no-echo rule.
        let d = doc("X ~ rand::unif_int(1,6)\nplot::histogram(X)");
        assert!(d.result.value.is_none());
    }
}

impl DocResult {
    fn to_json(&self) -> serde_json::Value {
        let value = self.value.as_ref().map(|v| serde_json::json!({ "kind": v.kind, "text": v.text }));
        let error = self
            .error
            .as_ref()
            .map(|e| serde_json::json!({ "message": e.message, "span": span_json(e.span) }));
        let truncated = self.truncated.as_ref().map(|t| {
            serde_json::json!({
                "dropped": t.dropped,
                "first_dropped_stmt_span": span_json(t.first_dropped_stmt_span),
            })
        });
        let inputs: Vec<serde_json::Value> = self.inputs.iter().map(ResolvedInput::to_json).collect();
        serde_json::json!({
            "value": value,
            "error": error,
            "stats": {
                "forcings": self.stats.forcings,
                "samples": self.stats.samples,
                "ops": self.stats.ops,
                "rng_draws": self.stats.rng_draws,
            },
            "truncated": truncated,
            "inputs": inputs,
        })
    }
}
