//! Frontmatter + knobs (PLAN-LITERATE §D1/D2). A `.noise` file may open with a `---`-fenced
//! metadata block that turns it into a self-describing document: a `title`, and a set of typed
//! **knobs** — host-tunable globals injected as point masses before statement 1.
//!
//! The fence is recognized **only at byte 0** (line 1 is exactly `---`); anywhere else `---` keeps
//! meaning three unary minuses, so no existing program breaks. The lexer treats the whole block as
//! trivia — it skips it *in place* (see [`block_end`]) so every downstream span keeps pointing into
//! the original source. This module is the separate entry a host calls to read the metadata without
//! running the program (`meta(src)` in wasm, `--knob`/`validate` in the CLI).
//!
//! Content is parsed by `serde_yaml` (wasm-clean, pure-Rust `unsafe-libyaml`). YAML is a superset of
//! JSON, so the same pass handles both the YAML `key: value` form and the `{ … }` JSON escape hatch;
//! it lowers to a `serde_json::Value`, and one `Value -> Frontmatter` path ([`from_value`]) validates
//! it (§D2).

use crate::error::{NoiseError, Result, Span};

/// A parsed frontmatter block. `knobs` keeps **source order** (a `Vec`, not a map) because that is
/// the order a host renders the knob UI in; `extra` preserves every unknown key as raw JSON so new
/// metadata (blurb, category, seed…) can grow without an engine release.
#[derive(Debug, Clone, PartialEq)]
pub struct Frontmatter {
    pub title: Option<String>,
    pub knobs: Vec<Knob>,
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A typed, host-tunable global. Bound as a plain deterministic point mass before statement 1
/// (exactly like `dice_sides = 6`); a program may shadow the name with a normal rebind.
#[derive(Debug, Clone, PartialEq)]
pub struct Knob {
    pub name: String,
    pub kind: KnobKind,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub default: KnobValue,
    /// Optional human label for the UI (falls back to the name).
    pub label: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnobKind {
    Int,
    Float,
    Bool,
    // Choice deferred to LT5 (int/float/bool cover every current example).
}

impl KnobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            KnobKind::Int => "int",
            KnobKind::Float => "float",
            KnobKind::Bool => "bool",
        }
    }
}

/// A concrete knob value — a number (int/float) or a bool. Injected into the engine as a
/// [`crate::Value`] point mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KnobValue {
    Num(f64),
    Bool(bool),
}

impl KnobValue {
    fn to_json(self) -> serde_json::Value {
        match self {
            KnobValue::Num(n) => serde_json::json!(n),
            KnobValue::Bool(b) => serde_json::json!(b),
        }
    }
}

impl Knob {
    fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("name".into(), serde_json::json!(self.name));
        m.insert("type".into(), serde_json::json!(self.kind.as_str()));
        if let Some(v) = self.min {
            m.insert("min".into(), serde_json::json!(v));
        }
        if let Some(v) = self.max {
            m.insert("max".into(), serde_json::json!(v));
        }
        if let Some(v) = self.step {
            m.insert("step".into(), serde_json::json!(v));
        }
        m.insert("default".into(), self.default.to_json());
        if let Some(l) = &self.label {
            m.insert("label".into(), serde_json::json!(l));
        }
        serde_json::Value::Object(m)
    }

    /// Validate + resolve a host override (or, with `override_ = None`, the default) into the
    /// concrete value to bind. Type-checks against the knob kind, then clamps to `[min, max]` and
    /// snaps to `step` — one implementation, so every host clamps identically (PLAN-LITERATE §D2).
    pub fn resolve(&self, override_: Option<KnobValue>) -> Result<KnobValue> {
        let raw = override_.unwrap_or(self.default);
        match (self.kind, raw) {
            (KnobKind::Bool, KnobValue::Bool(b)) => Ok(KnobValue::Bool(b)),
            (KnobKind::Bool, KnobValue::Num(_)) => Err(NoiseError::runtime(
                format!("knob `{}` is a bool; got a number", self.name),
                Span::new(0, 0),
            )),
            (KnobKind::Int | KnobKind::Float, KnobValue::Bool(_)) => Err(NoiseError::runtime(
                format!("knob `{}` is a number; got a bool", self.name),
                Span::new(0, 0),
            )),
            (kind, KnobValue::Num(n)) => {
                let mut v = n;
                if let Some(min) = self.min {
                    v = v.max(min);
                }
                if let Some(max) = self.max {
                    v = v.min(max);
                }
                if let Some(step) = self.step {
                    if step > 0.0 {
                        let base = self.min.unwrap_or(0.0);
                        v = base + ((v - base) / step).round() * step;
                    }
                }
                if kind == KnobKind::Int {
                    v = v.round();
                }
                // A snap can push us a hair past a bound; re-clamp.
                if let Some(min) = self.min {
                    v = v.max(min);
                }
                if let Some(max) = self.max {
                    v = v.min(max);
                }
                Ok(KnobValue::Num(v))
            }
        }
    }
}

impl Frontmatter {
    /// Serialize to the JSON shape hosts consume (`meta(src)`): `{ title?, knobs: [...], extra }`.
    /// `knobs` is an **array** (not an object) so source order survives without a `preserve_order`
    /// serde feature; a host renders sliders top-to-bottom in that order.
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if let Some(t) = &self.title {
            m.insert("title".into(), serde_json::json!(t));
        }
        m.insert(
            "knobs".into(),
            serde_json::Value::Array(self.knobs.iter().map(Knob::to_json).collect()),
        );
        m.insert("extra".into(), serde_json::Value::Object(self.extra.clone()));
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
            let block_end = if line_end < bytes.len() { line_end + 1 } else { line_end };
            let content = &src[content_start..content_end];
            return Ok(Some((content, Span::new(content_start, content_end), block_end)));
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
/// `Err`. Pure — hosts call this to build knob UIs without running the program.
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
/// `serde_yaml` pass handles both the YAML `key: value` form and the `{ … }` JSON escape hatch
/// (§D1). The result is lowered to a validated [`Frontmatter`] by [`from_value`].
fn parse_content(content: &str, span: Span) -> Result<serde_json::Value> {
    if content.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    serde_yaml::from_str::<serde_json::Value>(content)
        .map_err(|e| NoiseError::parse(format!("invalid frontmatter: {e}"), span))
}

// === Value -> Frontmatter ========================================================================

/// Lower a `serde_json::Value` (from either syntax) into a validated [`Frontmatter`]. Validation
/// per §D2: `default` within `[min, max]`, coherent types, `min <= max`.
fn from_value(value: serde_json::Value, span: Span) -> Result<Frontmatter> {
    let mut obj = match value {
        serde_json::Value::Object(m) => m,
        _ => {
            return Err(NoiseError::parse(
                "frontmatter must be a map of keys",
                span,
            ))
        }
    };

    let title = match obj.remove("title") {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s),
        Some(_) => return Err(NoiseError::parse("frontmatter `title` must be a string", span)),
    };

    let mut knobs = Vec::new();
    if let Some(knobs_val) = obj.remove("knobs") {
        let knob_map = match knobs_val {
            serde_json::Value::Object(m) => m,
            _ => return Err(NoiseError::parse("frontmatter `knobs` must be a map", span)),
        };
        for (name, spec) in knob_map {
            knobs.push(parse_knob(&name, spec, span)?);
        }
    }

    Ok(Frontmatter { title, knobs, extra: obj })
}

fn parse_knob(name: &str, spec: serde_json::Value, span: Span) -> Result<Knob> {
    if !is_ident(name) {
        return Err(NoiseError::parse(
            format!("knob name `{name}` is not a valid identifier"),
            span,
        ));
    }
    let map = match spec {
        serde_json::Value::Object(m) => m,
        _ => {
            return Err(NoiseError::parse(
                format!("knob `{name}` must be a map with a `type`"),
                span,
            ))
        }
    };
    let type_str = map
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| NoiseError::parse(format!("knob `{name}` needs a `type`"), span))?;
    let kind = match type_str {
        "int" => KnobKind::Int,
        "float" => KnobKind::Float,
        "bool" => KnobKind::Bool,
        other => {
            return Err(NoiseError::parse(
                format!("knob `{name}` has unknown type `{other}` (want int/float/bool)"),
                span,
            ))
        }
    };
    let num = |key: &str| -> Result<Option<f64>> {
        match map.get(key) {
            None => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or_else(|| {
                NoiseError::parse(format!("knob `{name}` field `{key}` must be a number"), span)
            }),
        }
    };
    let (min, max, step) = (num("min")?, num("max")?, num("step")?);
    if let (Some(lo), Some(hi)) = (min, max) {
        if lo > hi {
            return Err(NoiseError::parse(
                format!("knob `{name}` has min {lo} > max {hi}"),
                span,
            ));
        }
    }
    let default = match map.get("default") {
        Some(serde_json::Value::Bool(b)) => KnobValue::Bool(*b),
        Some(v) => match v.as_f64() {
            Some(n) => KnobValue::Num(n),
            None => {
                return Err(NoiseError::parse(
                    format!("knob `{name}` default must be a number or bool"),
                    span,
                ))
            }
        },
        None => {
            return Err(NoiseError::parse(
                format!("knob `{name}` needs a `default`"),
                span,
            ))
        }
    };
    // Type coherence + default within range.
    match (kind, default) {
        (KnobKind::Bool, KnobValue::Num(_)) => {
            return Err(NoiseError::parse(
                format!("knob `{name}` is a bool but its default is a number"),
                span,
            ))
        }
        (KnobKind::Int | KnobKind::Float, KnobValue::Bool(_)) => {
            return Err(NoiseError::parse(
                format!("knob `{name}` is numeric but its default is a bool"),
                span,
            ))
        }
        (_, KnobValue::Num(n)) => {
            if let Some(lo) = min {
                if n < lo {
                    return Err(NoiseError::parse(
                        format!("knob `{name}` default {n} is below min {lo}"),
                        span,
                    ));
                }
            }
            if let Some(hi) = max {
                if n > hi {
                    return Err(NoiseError::parse(
                        format!("knob `{name}` default {n} is above max {hi}"),
                        span,
                    ));
                }
            }
        }
        _ => {}
    }
    let label = map.get("label").and_then(|v| v.as_str()).map(str::to_string);
    Ok(Knob { name: name.to_string(), kind, min, max, step, default, label })
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
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
    fn yaml_title_and_knobs() {
        let src = "---\ntitle: \"Roll a die\"\nknobs:\n  dice_sides: { type: int, min: 1, max: 100, step: 1, default: 6 }\n  target: { type: int, min: 1, max: 100, default: 4 }\n---\nDice ~ unif_int(1, dice_sides)\n";
        let (fm, _span) = parse(src).unwrap().unwrap();
        assert_eq!(fm.title.as_deref(), Some("Roll a die"));
        assert_eq!(fm.knobs.len(), 2);
        assert_eq!(fm.knobs[0].name, "dice_sides");
        assert_eq!(fm.knobs[0].kind, KnobKind::Int);
        assert_eq!(fm.knobs[0].min, Some(1.0));
        assert_eq!(fm.knobs[0].default, KnobValue::Num(6.0));
        assert_eq!(fm.knobs[1].name, "target");
        // block_end points past the closing fence.
        let end = block_end(src).unwrap().unwrap();
        assert_eq!(&src[end..end + 4], "Dice");
    }

    #[test]
    fn knobs_keep_source_order_not_alphabetical() {
        // `zebra` before `apple` in source must stay that way (preserve_order), so knob UIs render
        // top-to-bottom as written.
        let src = "---\nknobs:\n  zebra: { type: int, default: 1 }\n  apple: { type: int, default: 2 }\n---\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        assert_eq!(fm.knobs[0].name, "zebra");
        assert_eq!(fm.knobs[1].name, "apple");
    }

    #[test]
    fn json_frontmatter() {
        let src = "---\n{ \"title\": \"J\", \"knobs\": { \"n\": { \"type\": \"float\", \"default\": 0.5 } } }\n---\nx = 1\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        assert_eq!(fm.title.as_deref(), Some("J"));
        assert_eq!(fm.knobs[0].kind, KnobKind::Float);
    }

    #[test]
    fn default_out_of_range_errors() {
        let src = "---\nknobs:\n  n: { type: int, min: 1, max: 6, default: 20 }\n---\n";
        let err = parse(src).unwrap_err();
        assert!(format!("{err}").contains("above max"));
    }

    #[test]
    fn resolve_clamps_and_snaps() {
        let k = Knob {
            name: "n".into(),
            kind: KnobKind::Int,
            min: Some(1.0),
            max: Some(10.0),
            step: Some(2.0),
            default: KnobValue::Num(1.0),
            label: None,
        };
        // 20 clamps hard to the max (10); 6.4 snaps to the step grid (1,3,5,7,9) → 7.
        assert_eq!(k.resolve(Some(KnobValue::Num(20.0))).unwrap(), KnobValue::Num(10.0));
        assert_eq!(k.resolve(Some(KnobValue::Num(6.4))).unwrap(), KnobValue::Num(7.0));
        // wrong type
        assert!(k.resolve(Some(KnobValue::Bool(true))).is_err());
    }

    #[test]
    fn extra_keys_preserved() {
        let src = "---\ntitle: t\nblurb: hello there\ncategory: basics\n---\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        assert_eq!(fm.extra.get("blurb").unwrap().as_str(), Some("hello there"));
        assert_eq!(fm.extra.get("category").unwrap().as_str(), Some("basics"));
    }

    #[test]
    fn to_json_shape() {
        let src = "---\ntitle: T\nknobs:\n  a: { type: int, min: 1, max: 6, default: 3 }\n---\n";
        let (fm, _) = parse(src).unwrap().unwrap();
        let j = fm.to_json();
        assert_eq!(j["title"], serde_json::json!("T"));
        assert_eq!(j["knobs"][0]["name"], serde_json::json!("a"));
        assert_eq!(j["knobs"][0]["type"], serde_json::json!("int"));
        assert_eq!(j["knobs"][0]["default"], serde_json::json!(3.0));
    }
}
