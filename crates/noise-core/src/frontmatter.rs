//! Frontmatter (PLAN-LITERATE §D1). A `.noise` file may open with a `---`-fenced metadata block that
//! turns it into a self-describing document: a `title`, an `abstract`, `tags`, and an `extra:` map
//! for host-specific metadata. Frontmatter is **purely descriptive** — host-tunable parameters are
//! no longer declared here; they are inline `input::…` calls in the program body (PLAN-INPUTS).
//!
//! The fence is recognized **only at byte 0** (line 1 is exactly `---`); anywhere else `---` keeps
//! meaning three unary minuses, so no existing program breaks. The lexer treats the whole block as
//! trivia — it skips it *in place* (see [`block_end`]) so every downstream span keeps pointing into
//! the original source. This module is the separate entry a host calls to read the metadata without
//! running the program (`meta(src)` in wasm, `validate` in the CLI).
//!
//! Content is parsed by `serde_norway` (the maintained `serde_yaml` fork; wasm-clean, pure-Rust
//! `unsafe-libyaml-norway`). YAML is a superset of
//! JSON, so the same pass handles both the YAML `key: value` form and the `{ … }` JSON escape hatch;
//! it lowers to a `serde_json::Value`, and one `Value -> Frontmatter` path ([`from_value`]) validates
//! it (§D2).

use crate::error::{NoiseError, Result, Span};

/// A parsed frontmatter block. `extra` is the file's explicit `extra:` map — raw JSON the engine
/// passes through untouched, so host-specific metadata (blurb, category, seed…) can grow without an
/// engine release. Only `title`/`abstract`/`tags`/`extra` are recognized at the top level; any
/// other key is a validation error (§D2). Tunable parameters are inline `input::…` calls, not
/// frontmatter (PLAN-INPUTS).
#[derive(Debug, Clone, PartialEq)]
pub struct Frontmatter {
    pub title: Option<String>,
    /// A paper-style abstract — the prose a literate host renders under the title. (Named
    /// `abstract_` because `abstract` is a reserved Rust keyword; the JSON key stays `abstract`.)
    pub abstract_: Option<String>,
    /// Free-form keyword tags (a paper's "Keywords:" line).
    pub tags: Vec<String>,
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Frontmatter {
    /// Serialize to the JSON shape hosts consume (`meta(src)`): `{ title?, abstract?, tags, extra }`.
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if let Some(t) = &self.title {
            m.insert("title".into(), serde_json::json!(t));
        }
        if let Some(a) = &self.abstract_ {
            m.insert("abstract".into(), serde_json::json!(a));
        }
        m.insert(
            "tags".into(),
            serde_json::Value::Array(self.tags.iter().map(|t| serde_json::json!(t)).collect()),
        );
        m.insert(
            "extra".into(),
            serde_json::Value::Object(self.extra.clone()),
        );
        serde_json::Value::Object(m)
    }
}

/// The byte offset where real source begins: the end of the `---`-fenced block (just past the
/// newline after the closing `---`), or `0` when the file has no frontmatter.
///
/// `Ok(None)` — no opening fence (line 1 is not exactly `---`). `Ok(Some(end))` — a well-formed
/// fence; the lexer sets its cursor to `end` and never emits a token for the block. `Err` — an
/// opening fence with no matching close (a spanned parse error). This is the cheap scan the lexer
/// runs; [`parse`] runs the same scan and then parses the content between the fences.
pub fn block_end(src: &str) -> Result<Option<usize>> {
    match fence_content(src)? {
        Some((_content, _content_span, end)) => Ok(Some(end)),
        None => Ok(None),
    }
}

/// Locate the fenced content. Returns `(content_str, content_span, block_end)` where `content_span`
/// is the byte range of the text between the fences (for offset-correct error spans) and
/// `block_end` is where the lexer resumes. `Ok(None)` when line 1 is not exactly `---`.
fn fence_content(src: &str) -> Result<Option<(&str, Span, usize)>> {
    // The opening fence must be the very first line, exactly `---` (optionally `\r`-terminated).
    let bytes = src.as_bytes();
    if !starts_with_fence_line(bytes) {
        return Ok(None);
    }
    // Advance past the opening `---` line.
    let mut i = 3;
    if i < bytes.len() && bytes[i] == b'\r' {
        i += 1;
    }
    // Line 1 is `---` with nothing else, so the next byte (if any) is `\n`.
    debug_assert!(i >= bytes.len() || bytes[i] == b'\n');
    if i < bytes.len() {
        i += 1; // consume the '\n'
    }
    let content_start = i;
    // Scan line by line for a closing fence line that is exactly `---`.
    let mut line_start = i;
    while line_start <= bytes.len() {
        let line_end = match src[line_start..].find('\n') {
            Some(rel) => line_start + rel,
            None => bytes.len(),
        };
        let line = src[line_start..line_end].trim_end_matches('\r');
        if line == "---" {
            let content_end = line_start; // content is everything before this closing line
            let block_end = if line_end < bytes.len() {
                line_end + 1
            } else {
                line_end
            };
            let content = &src[content_start..content_end];
            return Ok(Some((
                content,
                Span::new(content_start, content_end),
                block_end,
            )));
        }
        if line_end >= bytes.len() {
            break;
        }
        line_start = line_end + 1;
    }
    Err(NoiseError::parse(
        "unterminated frontmatter block: opening `---` has no matching closing `---`",
        Span::new(0, content_start.min(bytes.len())),
    ))
}

fn starts_with_fence_line(bytes: &[u8]) -> bool {
    // Exactly `---` then a line break (or EOF).
    bytes.len() >= 3
        && &bytes[0..3] == b"---"
        && match bytes.get(3) {
            None => true,
            Some(b'\n') => true,
            Some(b'\r') => bytes.get(4) == Some(&b'\n') || bytes.get(4).is_none(),
            _ => false,
        }
}

/// Parse the frontmatter of `src`. `Ok(None)` when the file has no fence; `Ok(Some((fm, span)))`
/// with the block's content span otherwise. A malformed fence or invalid metadata is a spanned
/// `Err`. Pure — hosts call this to read the paper header without running the program.
pub fn parse(src: &str) -> Result<Option<(Frontmatter, Span)>> {
    let (content, span, _end) = match fence_content(src)? {
        Some(x) => x,
        None => return Ok(None),
    };
    let value = parse_content(content, span)?;
    let fm = from_value(value, span)?;
    Ok(Some((fm, span)))
}

/// Parse the fenced content into a `serde_json::Value`. YAML is a superset of JSON, so one
/// `serde_norway` pass handles both the YAML `key: value` form and the `{ … }` JSON escape hatch
/// (§D1). The result is lowered to a validated [`Frontmatter`] by [`from_value`].
fn parse_content(content: &str, span: Span) -> Result<serde_json::Value> {
    if content.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    serde_norway::from_str::<serde_json::Value>(content)
        .map_err(|e| NoiseError::parse(format!("invalid frontmatter: {e}"), span))
}

// === Value -> Frontmatter ========================================================================

/// Lower a `serde_json::Value` (from either syntax) into a validated [`Frontmatter`]. Validation
/// per §D2: `default` within `[min, max]`, coherent types, `min <= max`.
fn from_value(value: serde_json::Value, span: Span) -> Result<Frontmatter> {
    let mut obj = match value {
        serde_json::Value::Object(m) => m,
        _ => return Err(NoiseError::parse("frontmatter must be a map of keys", span)),
    };

    let title = match obj.remove("title") {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s),
        Some(_) => {
            return Err(NoiseError::parse(
                "frontmatter `title` must be a string",
                span,
            ))
        }
    };

    let abstract_ = match obj.remove("abstract") {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s),
        Some(_) => {
            return Err(NoiseError::parse(
                "frontmatter `abstract` must be a string",
                span,
            ))
        }
    };

    let tags = match obj.remove("tags") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => Ok(s),
                _ => Err(NoiseError::parse(
                    "frontmatter `tags` must be a list of strings",
                    span,
                )),
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => {
            return Err(NoiseError::parse(
                "frontmatter `tags` must be a list of strings",
                span,
            ))
        }
    };

    // `knobs:` is retired (PLAN-INPUTS): tunable parameters are now inline `input::…` calls in the
    // program body. Catch the old key with a migration hint instead of the generic unknown-key error.
    if obj.contains_key("knobs") {
        return Err(NoiseError::parse(
            "frontmatter `knobs:` is no longer supported — declare tunable parameters inline with \
             `input::real(min: …, max: …, default: …)` in the program body (PLAN-INPUTS)",
            span,
        ));
    }

    // `extra` is the explicit escape hatch for host-specific metadata (blurb, category, seed…): a
    // free-form map the engine passes through untouched. It must be spelled out — everything left at
    // the top level has to be a recognized key, so an unknown key is a typo or a field that belongs
    // under `extra:` rather than something silently accepted.
    let extra = match obj.remove("extra") {
        None => serde_json::Map::new(),
        Some(serde_json::Value::Object(m)) => m,
        Some(_) => return Err(NoiseError::parse("frontmatter `extra` must be a map", span)),
    };
    if let Some(key) = obj.keys().next() {
        return Err(NoiseError::parse(
            format!(
                "frontmatter has unknown key `{key}` \
                 (allowed: title, abstract, tags, extra — put custom metadata under `extra:`)"
            ),
            span,
        ));
    }

    Ok(Frontmatter {
        title,
        abstract_,
        tags,
        extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_fence_is_none() {
        assert_eq!(block_end("x = 1\n").unwrap(), None);
        assert!(parse("x = 1\n").unwrap().is_none());
    }

    #[test]
    fn dashes_mid_file_stay_minus() {
        // `---` not at byte 0 is not a fence.
        assert_eq!(block_end("y = 1\n---\n").unwrap(), None);
    }

    #[test]
    fn unterminated_fence_errors() {
        let err = parse("---\ntitle: hi\n").unwrap_err();
        assert!(format!("{err}").contains("unterminated"));
    }

    #[test]
    fn yaml_title_and_tags() {
        let src = "---\ntitle: \"Roll a die\"\ntags: [games, basics]\n---\nDice ~ unif_int(1, 6)\n";
        let (fm, _span) = parse(src).unwrap().unwrap();
        assert_eq!(fm.title.as_deref(), Some("Roll a die"));
        assert_eq!(fm.tags, vec!["games".to_string(), "basics".to_string()]);
        // block_end points past the closing fence.
        let end = block_end(src).unwrap().unwrap();
        assert_eq!(&src[end..end + 4], "Dice");
    }

    #[test]
    fn json_frontmatter() {
        let src = "---\n{ \"title\": \"J\", \"tags\": [\"a\"] }\n---\nx = 1\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        assert_eq!(fm.title.as_deref(), Some("J"));
        assert_eq!(fm.tags, vec!["a".to_string()]);
    }

    #[test]
    fn knobs_key_is_retired_with_a_migration_hint() {
        // The old `knobs:` block is gone (PLAN-INPUTS); the error points at `input::`.
        let err = parse("---\nknobs:\n  n: { type: int, default: 6 }\n---\n").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no longer supported"), "got: {msg}");
        assert!(
            msg.contains("input::real"),
            "should point at input::; got: {msg}"
        );
    }

    #[test]
    fn extra_map_is_passthrough() {
        let src = "---\ntitle: t\nextra:\n  blurb: hello there\n  category: basics\n---\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        assert_eq!(fm.extra.get("blurb").unwrap().as_str(), Some("hello there"));
        assert_eq!(fm.extra.get("category").unwrap().as_str(), Some("basics"));
    }

    #[test]
    fn unknown_top_level_key_errors() {
        // A field that isn't first-class must live under `extra:`; at the top level it's a typo.
        let err = parse("---\ntitle: t\nblurb: hi\n---\n").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown key `blurb`"), "got: {msg}");
        assert!(
            msg.contains("extra"),
            "should point the user at `extra:`; got: {msg}"
        );
    }

    #[test]
    fn extra_must_be_a_map() {
        let err = parse("---\ntitle: t\nextra: nope\n---\n").unwrap_err();
        assert!(format!("{err}").contains("`extra` must be a map"));
    }

    #[test]
    fn abstract_and_tags_are_native() {
        let src = "---\ntitle: T\nabstract: >\n  A short paper-style summary\n  spanning two lines.\ntags: [monte carlo, basics]\n---\nx = 1\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        assert_eq!(
            fm.abstract_.as_deref(),
            Some("A short paper-style summary spanning two lines.\n")
        );
        assert_eq!(
            fm.tags,
            vec!["monte carlo".to_string(), "basics".to_string()]
        );
        // Native fields don't leak into `extra`.
        assert!(!fm.extra.contains_key("abstract"));
        assert!(!fm.extra.contains_key("tags"));
        let j = fm.to_json();
        assert_eq!(j["tags"], serde_json::json!(["monte carlo", "basics"]));
        assert_eq!(
            j["abstract"],
            serde_json::json!("A short paper-style summary spanning two lines.\n")
        );
    }

    #[test]
    fn tags_must_be_strings() {
        let err = parse("---\ntags: [1, 2]\n---\n").unwrap_err();
        assert!(format!("{err}").contains("list of strings"));
    }

    #[test]
    fn to_json_shape() {
        let src = "---\ntitle: T\ntags: [a, b]\nextra:\n  blurb: hi\n---\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        let j = fm.to_json();
        assert_eq!(j["title"], serde_json::json!("T"));
        assert_eq!(j["tags"], serde_json::json!(["a", "b"]));
        assert_eq!(j["extra"]["blurb"], serde_json::json!("hi"));
        // `knobs` is no longer part of the meta payload.
        assert!(j.get("knobs").is_none());
    }
}
