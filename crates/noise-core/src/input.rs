//! Inputs — host-tunable parameters declared **inline in the program body** with `input::real(…)`
//! (PLAN-INPUTS). Instead of a YAML schema at the top of the file, a tunable value is an ordinary
//! namespaced call written where it is used. It evaluates to its **current value** — a deterministic
//! point mass — so downstream code reads it like any number, while a literate host renders a control
//! at the point of declaration.
//!
//! The engine intercepts `input::{real,int,bool}` in `Expr::Call` (like `plot::`/`stats::`),
//! resolves the current value here (clamped to `[min, max]` and snapped to `step`), records the
//! input in the run's **manifest** (for host discovery), and emits an inline control block. This
//! module owns the value types and the resolve/validate/serialize logic; `eval.rs` owns the call
//! interception.

use crate::error::{NoiseError, Result, Span};

/// The type of an input. `Real` is the continuous slider; `Int` snaps to whole numbers; `Bool` is a
/// checkbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Real,
    Int,
    Bool,
}

impl InputKind {
    pub fn as_str(self) -> &'static str {
        match self {
            InputKind::Real => "real",
            InputKind::Int => "int",
            InputKind::Bool => "bool",
        }
    }

    /// Parse the `input::<base>` call base into a kind. `None` for an unknown base.
    pub fn from_base(base: &str) -> Option<InputKind> {
        match base {
            "real" => Some(InputKind::Real),
            "int" => Some(InputKind::Int),
            "bool" => Some(InputKind::Bool),
            _ => None,
        }
    }
}

/// A concrete input value — a number (`real`/`int`) or a bool. Bound into the engine as a
/// [`crate::Value`] point mass, and the shape a host override arrives in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputValue {
    Num(f64),
    Bool(bool),
}

impl InputValue {
    fn to_json(self) -> serde_json::Value {
        match self {
            InputValue::Num(n) => serde_json::json!(n),
            InputValue::Bool(b) => serde_json::json!(b),
        }
    }
}

/// A declared input: its name (the manifest / override key), type, optional bounds/step, default,
/// and optional UI label. Built by `eval.rs` from the named arguments of an `input::` call, then
/// validated by [`InputSpec::validate`].
#[derive(Debug, Clone, PartialEq)]
pub struct InputSpec {
    pub name: String,
    pub kind: InputKind,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub default: InputValue,
    pub label: Option<String>,
}

impl InputSpec {
    /// Validate the spec at declaration: coherent bounds (`min <= max`), a type-coherent default,
    /// and the default within `[min, max]`. `span` locates the `input::` call for the error.
    pub fn validate(&self, span: Span) -> Result<()> {
        if let (Some(lo), Some(hi)) = (self.min, self.max) {
            if lo > hi {
                return Err(NoiseError::runtime(
                    format!("input `{}` has min {lo} > max {hi}", self.name),
                    span,
                ));
            }
        }
        match (self.kind, self.default) {
            (InputKind::Bool, InputValue::Num(_)) => {
                return Err(NoiseError::runtime(
                    format!("input `{}` is a bool but its default is a number", self.name),
                    span,
                ))
            }
            (InputKind::Real | InputKind::Int, InputValue::Bool(_)) => {
                return Err(NoiseError::runtime(
                    format!("input `{}` is numeric but its default is a bool", self.name),
                    span,
                ))
            }
            (_, InputValue::Num(n)) => {
                if let Some(lo) = self.min {
                    if n < lo {
                        return Err(NoiseError::runtime(
                            format!("input `{}` default {n} is below min {lo}", self.name),
                            span,
                        ));
                    }
                }
                if let Some(hi) = self.max {
                    if n > hi {
                        return Err(NoiseError::runtime(
                            format!("input `{}` default {n} is above max {hi}", self.name),
                            span,
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Validate + resolve a host override (or, with `override_ = None`, the default) into the
    /// concrete value to bind. Type-checks against the kind, clamps to `[min, max]`, snaps to
    /// `step` — one implementation, so every host agrees.
    pub fn resolve(&self, override_: Option<InputValue>) -> Result<InputValue> {
        let raw = override_.unwrap_or(self.default);
        match (self.kind, raw) {
            (InputKind::Bool, InputValue::Bool(b)) => Ok(InputValue::Bool(b)),
            (InputKind::Bool, InputValue::Num(_)) => Err(NoiseError::runtime(
                format!("input `{}` is a bool; got a number", self.name),
                Span::new(0, 0),
            )),
            (InputKind::Real | InputKind::Int, InputValue::Bool(_)) => Err(NoiseError::runtime(
                format!("input `{}` is a number; got a bool", self.name),
                Span::new(0, 0),
            )),
            (kind, InputValue::Num(n)) => {
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
                if kind == InputKind::Int {
                    v = v.round();
                }
                // A snap can push us a hair past a bound; re-clamp.
                if let Some(min) = self.min {
                    v = v.max(min);
                }
                if let Some(max) = self.max {
                    v = v.min(max);
                }
                Ok(InputValue::Num(v))
            }
        }
    }

    /// Serialize one manifest entry — the spec plus its resolved `value` and the `stmt_span` of the
    /// declaring statement — to the JSON a host consumes (`Document.result.inputs[]`).
    pub fn to_json_entry(&self, value: InputValue, stmt_span: Span) -> serde_json::Value {
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
        m.insert("value".into(), value.to_json());
        if let Some(l) = &self.label {
            m.insert("label".into(), serde_json::json!(l));
        }
        m.insert(
            "stmt_span".into(),
            serde_json::json!({ "start": stmt_span.start, "end": stmt_span.end }),
        );
        serde_json::Value::Object(m)
    }
}

/// One resolved input in the run manifest: the spec, its current value, and the span of the
/// statement that declared it (so a host can place the control inline). The manifest is accumulated
/// as `input::` calls evaluate and returned on `Document.result.inputs` (PLAN-INPUTS §3).
#[derive(Debug, Clone)]
pub struct ResolvedInput {
    pub spec: InputSpec,
    pub value: InputValue,
    pub stmt_span: Span,
}

impl ResolvedInput {
    pub fn to_json(&self) -> serde_json::Value {
        self.spec.to_json_entry(self.value, self.stmt_span)
    }
}

/// Is `s` a valid input name (a Noise identifier)? Explicit `name: "…"` must satisfy this; an
/// inferred name comes from a binding identifier and always does.
pub fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
