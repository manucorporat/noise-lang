//! The output boundary: an introspection [`Summary`] becomes chart specs a host can render.
//!
//! Noise does not draw charts. It emits [Flint](https://github.com/microsoft/flint-chart)
//! `ChartAssemblyInput` specs — *semantic* descriptions of a chart (which type, which fields, what
//! those fields mean) — and a host compiles them to a backend (Vega-Lite, ECharts, Chart.js) with
//! stock libraries. So there is exactly one place in this repo that knows what a histogram looks
//! like, and it is not this file either: this file only says "a histogram is a pre-binned Bar Chart
//! whose x is a `Number` and whose y is a `Count`".
//!
//! Two things ride along with every plot:
//!
//! * `charts` — zero or more `ChartAssemblyInput` values. Zero means "the text says it all" (an
//!   exact scalar); more than one means the host layers them (the fan's two bands + median line),
//!   which is Flint's sanctioned recipe for a composite: compile each spec, then merge the backend
//!   JSON minimally. We never emit backend JSON from Rust — specs survive regeneration, and a
//!   Vega-Lite blob would pin us to one renderer forever.
//! * `text` — the one-line fallback card. It is what the CLI prints and what any host shows when it
//!   cannot draw. Every number a chart encodes is also in this line, so nothing is lost by not
//!   rendering, only the picture.
//!
//! The compute lives in [`crate::introspect`] and is untouched by any of this — `flint.rs` is a
//! serializer, not a second implementation. Data rides *inside* each spec (`data.values`, inline
//! rows), which is why the engine must keep aggregating: we ship 30 bins and 800 scatter points,
//! never the 200 000 draws behind them.

use serde_json::{json, Value as J};

use crate::introspect::{
    CorrMatrix, Dist1, Dist2, DistGrid, Explain, FanChart, Payload, Summary, ValueCard, View,
};

/// The width, in px, every spec is authored against. A host that knows its real pixel width
/// rescales `canvasSize` and keeps the aspect ratio — so this constant fixes *shape*, not size.
const CANVAS_W: f64 = 640.0;
/// The default height for the time-series-ish charts (histogram, scatter, line, fan).
const CANVAS_H: f64 = 360.0;

/// A plot, ready for a host to render: a heading, a text fallback, and the chart specs.
///
/// `charts.len() > 1` means *layer them* — the specs share an x scale by construction (same x
/// field, same rows), so a host merges the compiled marks into one plot.
#[derive(Debug, Clone, PartialEq)]
pub struct Plot {
    /// The card heading, e.g. `hist(st)` or `fan(path)`.
    pub title: String,
    /// A one-line summary carrying every number the charts encode — the no-renderer fallback.
    pub text: String,
    /// Flint `ChartAssemblyInput` specs, to be compiled by the host and layered if more than one.
    pub charts: Vec<J>,
}

/// Translate a computed summary into its renderable form. Total: every payload maps to some title
/// and text, and to zero or more charts.
pub fn to_flint(s: &Summary) -> Plot {
    Plot {
        title: title(s),
        text: text_card(s),
        charts: charts(s),
    }
}

// --- titles ------------------------------------------------------------------------------------

/// The card heading: the operation the program asked for, applied to the source names it named.
/// `describe` is the anonymous view — its heading is just the variable.
fn title(s: &Summary) -> String {
    let a = &s.label;
    let b = s.label_b.as_deref().unwrap_or("?");
    match s.view {
        View::Hist => format!("hist({a})"),
        View::Samples => format!("samples({a})"),
        View::Scatter => format!("scatter({a}, {b})"),
        View::Corr => match &s.payload {
            Payload::CorrMatrix(_) => format!("corr({a})"),
            _ => format!("corr({a}, {b})"),
        },
        View::Explain => format!("explain({a})"),
        View::Fan => format!("fan({a})"),
        View::CorrMatrix => format!("corr({a})"),
        View::Describe | View::Value | View::Grid => a.clone(),
    }
}

// --- the text card -----------------------------------------------------------------------------

/// The one-line fallback. Terse but complete: whatever a chart would show, this line states. Always
/// `"{title}: {detail}"`, so a host with a heading of its own can drop the redundant prefix. This is
/// also `Summary`'s `Display`, so a REPL `describe(x)` reads the same as a CLI `plot::hist(x)`.
pub fn text_card(s: &Summary) -> String {
    let head = title(s);
    match (&s.payload, s.view) {
        (Payload::One(d), View::Samples) => {
            let body = d
                .head
                .iter()
                .map(|x| fmt_n(*x))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{head}: [{body}]")
        }
        (Payload::One(d), _) if d.boolean => {
            format!("{head}: n={} P(true)={}", d.n, fmt_n(d.mean))
        }
        (Payload::One(d), _) => format!(
            "{head}: n={} mean={} sd={} min={} max={} q05={} q25={} med={} q75={} q95={}",
            d.n,
            fmt_n(d.mean),
            fmt_n(d.sd),
            fmt_n(d.min),
            fmt_n(d.max),
            fmt_n(d.q05),
            fmt_n(d.q25),
            fmt_n(d.q50),
            fmt_n(d.q75),
            fmt_n(d.q95),
        ),
        (Payload::Two(d), _) => {
            format!(
                "{head}: n={} corr={} cov={}",
                d.n,
                fmt_n(d.corr),
                fmt_n(d.cov)
            )
        }
        (Payload::Explain(e), _) => {
            if e.drivers.is_empty() {
                return format!("{head}: sd={} (no named upstream variables)", fmt_n(e.sd));
            }
            let drivers = e
                .drivers
                .iter()
                .map(|d| format!("{} {:.0}%", d.name, d.share * 100.0))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{head}: sd={} drivers: {drivers}", fmt_n(e.sd))
        }
        (Payload::Value(v), _) => {
            if v.se > 0.0 {
                let half = 1.96 * v.se;
                format!(
                    "{head}: {} ± {} (95% CI {} … {})",
                    fmt_n(v.val),
                    fmt_n(v.se),
                    fmt_n(v.val - half),
                    fmt_n(v.val + half)
                )
            } else {
                format!("{head}: {} (exact)", fmt_n(v.val))
            }
        }
        (Payload::Grid(g), _) => {
            if g.mean.is_empty() {
                return format!("{head}: (empty)");
            }
            let (lo, hi) = min_max(&g.mean);
            let kind = if g.is_series() { "vector" } else { "matrix" };
            format!(
                "{head}: {kind} {}×{} mean {} … {}",
                g.rows,
                g.cols,
                fmt_n(lo),
                fmt_n(hi)
            )
        }
        (Payload::CorrMatrix(c), _) => {
            // The diagonal is 1 by construction, so the informative number is the strongest
            // *off-diagonal* dependence — "are these elements independent?" answered in one number.
            let peak = (0..c.n)
                .flat_map(|i| (0..c.n).map(move |j| (i, j)))
                .filter(|(i, j)| i != j)
                .map(|(i, j)| c.corr[i * c.n + j].abs())
                .fold(0.0f64, f64::max);
            format!(
                "{head}: {n}×{n} max |corr| off-diagonal = {}",
                fmt_n(peak),
                n = c.n
            )
        }
        (Payload::Fan(c), _) => {
            if c.q50.is_empty() {
                return format!("{head}: (empty)");
            }
            let last = c.cols - 1;
            format!(
                "{head}: {} steps n={} final q05={} med={} q95={}",
                c.cols,
                c.n,
                fmt_n(c.q05[last]),
                fmt_n(c.q50[last]),
                fmt_n(c.q95[last]),
            )
        }
    }
}

/// Trim float dust for compact chart labels — the shared dust-trimmer (finding F5) at 4 decimal
/// places (compact), where `value::format_num` uses 12 (full value precision).
fn fmt_n(x: f64) -> String {
    crate::num::trim_float(x, 4)
}

fn min_max(xs: &[f64]) -> (f64, f64) {
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

// --- the specs ---------------------------------------------------------------------------------

/// Assemble one `ChartAssemblyInput`. `semantic_types` is what makes Flint's design decisions for
/// us: a `Count` axis gets integer ticks, a `Correlation` gets a diverging ramp pinned to ±1.
fn chart(
    rows: Vec<J>,
    semantic: J,
    chart_type: &str,
    encodings: J,
    properties: J,
    w: f64,
    h: f64,
) -> J {
    let mut spec = json!({
        "chartType": chart_type,
        "encodings": encodings,
        "canvasSize": { "width": w, "height": h },
    });
    if !properties.is_null() {
        spec["chartProperties"] = properties;
    }
    json!({ "data": { "values": rows }, "semantic_types": semantic, "chart_spec": spec })
}

/// A field name derived from a source label, kept distinct from the fixed names a spec also uses.
/// (`hist(count)` would otherwise collide its value column with its `count` column.) Flint escapes
/// and titles the raw name for us, so `path[51]` is a legal field, not a Vega-Lite nested lookup.
fn field(label: &str, reserved: &[&str]) -> String {
    let mut name = if label.is_empty() {
        "value".to_string()
    } else {
        label.to_string()
    };
    while reserved.contains(&name.as_str()) {
        name.push('_');
    }
    name
}

/// A `Percentage` that Flint should format as `42%` rather than `0.42` — the `intrinsicDomain`
/// annotation is what flips its percent formatter on.
fn percentage() -> J {
    json!({ "semanticType": "Percentage", "intrinsicDomain": [0.0, 1.0] })
}

/// Line-chart properties for a line that will be **layered over a band**. A lone Line Chart lets
/// Flint anchor y at zero; a banded one must not, both because a price cone dragged down to 0 is
/// unreadable, and because a `Range Area` never anchors at zero — layering the two with different
/// `zero` settings makes Vega-Lite warn and pick one arbitrarily.
fn no_zero_baseline() -> J {
    json!({ "includeZero_y": false })
}

fn charts(s: &Summary) -> Vec<J> {
    match &s.payload {
        Payload::One(d) => dist1_chart(&s.label, d).into_iter().collect(),
        Payload::Two(d) => dist2_chart(&s.label, s.label_b.as_deref().unwrap_or("y"), d)
            .into_iter()
            .collect(),
        Payload::Explain(e) => explain_chart(e).into_iter().collect(),
        Payload::Value(v) => value_chart(&s.label, v).into_iter().collect(),
        Payload::Grid(g) => grid_charts(&s.label, g),
        Payload::CorrMatrix(c) => corr_chart(c).into_iter().collect(),
        Payload::Fan(c) => fan_charts(&s.label, c),
    }
}

/// A distribution → a **pre-binned** Bar Chart (never raw draws: the engine already binned 200 000
/// of them into 30 buckets, and a spec carries its data inline). A boolean quantity is instead two
/// bars of a `Percentage` — for an event, `P(true)` is the whole story and quantiles are noise.
fn dist1_chart(label: &str, d: &Dist1) -> Option<J> {
    if d.boolean {
        let p = d.mean;
        let rows = vec![
            json!({ "outcome": "false", "share": 1.0 - p }),
            json!({ "outcome": "true", "share": p }),
        ];
        return Some(chart(
            rows,
            json!({ "outcome": "Category", "share": percentage() }),
            "Bar Chart",
            json!({ "x": "outcome", "y": "share" }),
            J::Null,
            CANVAS_W,
            CANVAS_H * 0.6,
        ));
    }
    let bins = &d.hist.bins;
    let (lo, hi) = (d.hist.lo, d.hist.hi);
    if bins.is_empty() || !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    // Bars carry each bin's SHARE of the draws, not its raw count: a count axis leaks the sample
    // budget (an implementation detail), while a 0–1 share reads the same at any budget.
    let value = field(label, &["share"]);
    let total = bins.iter().sum::<u64>().max(1) as f64;
    let rows = d
        .hist
        .midpoints(false)
        .into_iter()
        .zip(bins)
        .map(|(mid, &count)| json!({ &value: mid, "share": count as f64 / total }))
        .collect();
    Some(chart(
        rows,
        json!({ &value: "Number", "share": percentage() }),
        "Bar Chart",
        json!({ "x": &value, "y": "share" }),
        J::Null,
        CANVAS_W,
        CANVAS_H,
    ))
}

/// A relationship → a Scatter Plot of the subsampled point cloud (the *statistics* used every draw;
/// only the dots were thinned). Translucent, because 800 points overplot.
fn dist2_chart(label_a: &str, label_b: &str, d: &Dist2) -> Option<J> {
    if d.points.is_empty() {
        return None;
    }
    let a = field(label_a, &[]);
    let b = field(label_b, &[&a]);
    let rows = d
        .points
        .iter()
        .map(|&(x, y)| json!({ &a: x, &b: y }))
        .collect();
    Some(chart(
        rows,
        json!({ &a: "Number", &b: "Number" }),
        "Scatter Plot",
        json!({ "x": &a, "y": &b }),
        json!({ "opacity": 0.4 }),
        CANVAS_W,
        CANVAS_H,
    ))
}

/// Driver attribution → a horizontal Bar Chart of shares, in the order the ranking produced (Flint
/// leaves a nominal axis unsorted, so "strongest first" survives the round trip).
fn explain_chart(e: &Explain) -> Option<J> {
    if e.drivers.is_empty() {
        return None;
    }
    let rows = e
        .drivers
        .iter()
        .map(|d| json!({ "driver": d.name, "share": d.share }))
        .collect();
    let height = (90.0 + 30.0 * e.drivers.len() as f64).clamp(140.0, 480.0);
    Some(chart(
        rows,
        json!({ "driver": "Category", "share": percentage() }),
        "Bar Chart",
        json!({ "x": "share", "y": "driver" }),
        J::Null,
        CANVAS_W,
        height,
    ))
}

/// An array of random variables → a per-index mean line, banded by ±1 sd when there *is* spread
/// (two specs, layered by the host — the same recipe as the fan). A matrix has no index to walk, so
/// it becomes a Heatmap of the cell means.
fn grid_charts(label: &str, g: &DistGrid) -> Vec<J> {
    if g.mean.is_empty() {
        return vec![];
    }
    if !g.is_series() {
        let value = field(label, &["row", "column"]);
        let rows = (0..g.rows)
            .flat_map(|r| (0..g.cols).map(move |c| (r, c)))
            .map(|(r, c)| json!({ "row": r, "column": c, &value: g.mean[r * g.cols + c] }))
            .collect();
        // Square-ish cells: the canvas follows the matrix's own aspect, within reason.
        let height = (CANVAS_W * g.rows as f64 / g.cols as f64).clamp(240.0, CANVAS_W);
        return vec![chart(
            rows,
            json!({ "row": "Rank", "column": "Rank", &value: "Number" }),
            "Heatmap",
            json!({ "x": "column", "y": "row", "color": &value }),
            J::Null,
            CANVAS_W,
            height,
        )];
    }
    let mean = field(label, &["index", "lo", "hi"]);
    let banded = g.sd.iter().any(|&s| s > 0.0);
    let line = chart(
        (0..g.mean.len())
            .map(|i| json!({ "index": i, &mean: g.mean[i] }))
            .collect(),
        json!({ "index": "Number", &mean: "Number" }),
        "Line Chart",
        json!({ "x": "index", "y": &mean }),
        if banded { no_zero_baseline() } else { J::Null },
        CANVAS_W,
        CANVAS_H,
    );
    if !banded {
        return vec![line];
    }
    let band = chart(
        (0..g.mean.len())
            .map(|i| json!({ "index": i, "lo": g.mean[i] - g.sd[i], "hi": g.mean[i] + g.sd[i] }))
            .collect(),
        json!({ "index": "Number", "lo": "Number", "hi": "Number" }),
        "Range Area Chart",
        json!({ "x": "index", "y": "lo", "y2": "hi" }),
        json!({ "opacity": 0.25 }),
        CANVAS_W,
        CANVAS_H,
    );
    vec![band, line] // the mean line rides on top of its own spread
}

/// A scalar with uncertainty → its 95% interval as a Ranged Dot Plot: low · estimate · high on one
/// row. An *exact* scalar has no interval to draw, so it gets no chart — its text card is the whole
/// truth.
///
/// The single row is a blank category (a field named `" "`, whose one value is `" "`), because Flint
/// titles an axis after its field and labels it with its values — and a chart of one variable has
/// nothing to say on that axis. The value axis carries the variable's name instead, so the card
/// reads `pi ├──•──┤` and not `pi` twice with a legend explaining three dots.
fn value_chart(label: &str, v: &ValueCard) -> Option<J> {
    if !v.val.is_finite() || !v.se.is_finite() || v.se <= 0.0 {
        return None;
    }
    let half = 1.96 * v.se;
    let name = field(label, &[BLANK]);
    let point = |x: f64| json!({ BLANK: BLANK, &name: x });
    Some(chart(
        vec![point(v.val - half), point(v.val), point(v.val + half)],
        json!({ BLANK: "Category", &name: "Number" }),
        "Ranged Dot Plot",
        json!({ "x": &name, "y": BLANK }),
        J::Null,
        CANVAS_W,
        120.0,
    ))
}

/// The nameless axis: see [`value_chart`].
const BLANK: &str = " ";

/// The element×element dependence matrix → a Heatmap whose `Correlation` semantic type buys the
/// diverging ±1 ramp for free (blue ← 0 → red), which is the *only* honest scale for a correlation.
fn corr_chart(c: &CorrMatrix) -> Option<J> {
    if c.n == 0 {
        return None;
    }
    let rows = (0..c.n)
        .flat_map(|i| (0..c.n).map(move |j| (i, j)))
        .map(|(i, j)| json!({ "row": i, "column": j, "corr": c.corr[i * c.n + j] }))
        .collect();
    Some(chart(
        rows,
        json!({ "row": "Rank", "column": "Rank", "corr": "Correlation" }),
        "Heatmap",
        json!({ "x": "column", "y": "row", "color": "corr" }),
        J::Null,
        CANVAS_W,
        CANVAS_W,
    ))
}

/// A path's quantile envelope → the cone: two translucent Range Areas (q05–q95 outer, q25–q75
/// inner) with the median Line on top. Three specs, one per layer, in back-to-front order — the
/// host compiles each and merges the marks. They share an x field and identical rows, so the merged
/// scales agree by construction.
fn fan_charts(label: &str, c: &FanChart) -> Vec<J> {
    if c.q50.is_empty() {
        return vec![];
    }
    let median = field(label, &["index", "q05", "q25", "q75", "q95"]);
    let band = |lo_name: &str, hi_name: &str, lo: &[f64], hi: &[f64]| {
        chart(
            (0..c.cols)
                .map(|t| json!({ "index": t, lo_name: lo[t], hi_name: hi[t] }))
                .collect(),
            json!({ "index": "Number", lo_name: "Number", hi_name: "Number" }),
            "Range Area Chart",
            json!({ "x": "index", "y": lo_name, "y2": hi_name }),
            json!({ "opacity": 0.25 }),
            CANVAS_W,
            CANVAS_H,
        )
    };
    vec![
        band("q05", "q95", &c.q05, &c.q95),
        band("q25", "q75", &c.q25, &c.q75),
        chart(
            (0..c.cols)
                .map(|t| json!({ "index": t, &median: c.q50[t] }))
                .collect(),
            json!({ "index": "Number", &median: "Number" }),
            "Line Chart",
            json!({ "x": "index", "y": &median }),
            no_zero_baseline(),
            CANVAS_W,
            CANVAS_H,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::introspect::{Driver, Histogram, ValueCard};

    fn summary(view: View, label: &str, payload: Payload) -> Summary {
        Summary {
            view,
            label: label.into(),
            label_b: None,
            payload,
        }
    }

    /// Every emitted spec must be a well-formed `ChartAssemblyInput`: inline data, a semantic type
    /// per encoded field, and a `chartType` Flint actually registers.
    fn assert_well_formed(spec: &J) {
        let rows = spec["data"]["values"]
            .as_array()
            .expect("data.values must be an array");
        assert!(!rows.is_empty(), "a spec must carry its data inline");
        let types = spec["semantic_types"]
            .as_object()
            .expect("semantic_types must be an object");
        let encodings = spec["chart_spec"]["encodings"]
            .as_object()
            .expect("encodings");
        for (channel, f) in encodings {
            let f = f
                .as_str()
                .unwrap_or_else(|| panic!("encoding {channel} must name a field"));
            assert!(
                types.contains_key(f),
                "field {f:?} is encoded but has no semantic type"
            );
            assert!(
                rows[0].get(f).is_some(),
                "field {f:?} is encoded but absent from the rows"
            );
        }
        // Exactly the chart types this repo's renderer contract promises Flint knows.
        let ct = spec["chart_spec"]["chartType"].as_str().unwrap();
        const REGISTERED: [&str; 6] = [
            "Bar Chart",
            "Scatter Plot",
            "Line Chart",
            "Range Area Chart",
            "Heatmap",
            "Ranged Dot Plot",
        ];
        assert!(REGISTERED.contains(&ct), "unregistered chartType {ct:?}");
        assert!(
            spec["chart_spec"]["canvasSize"]["width"].is_number(),
            "canvasSize"
        );
    }

    fn dist1(boolean: bool) -> Dist1 {
        Dist1 {
            n: 1000,
            mean: if boolean { 0.42 } else { 2.0 },
            sd: 1.0,
            min: 0.0,
            max: 4.0,
            q05: 0.5,
            q25: 1.0,
            q50: 2.0,
            q75: 3.0,
            q95: 3.5,
            hist: if boolean {
                Histogram {
                    lo: 0.0,
                    hi: 1.0,
                    bins: vec![580, 420],
                }
            } else {
                Histogram {
                    lo: 0.0,
                    hi: 4.0,
                    bins: vec![100, 400, 400, 100],
                }
            },
            head: vec![1.0, 2.0, 3.0],
            boolean,
        }
    }

    #[test]
    fn a_histogram_is_a_pre_binned_bar_chart_of_bin_midpoints() {
        let p = to_flint(&summary(View::Hist, "st", Payload::One(dist1(false))));
        assert_eq!(p.title, "hist(st)");
        assert_eq!(p.charts.len(), 1);
        assert_well_formed(&p.charts[0]);
        let c = &p.charts[0];
        assert_eq!(c["chart_spec"]["chartType"], "Bar Chart");
        // Bars encode each bin's share of the draws (0–1), not raw counts — a count axis would
        // leak the sample budget.
        assert_eq!(c["semantic_types"]["share"]["semanticType"], "Percentage");
        // 4 bins over [0, 4] ⇒ midpoints 0.5, 1.5, 2.5, 3.5 — never the bin edges.
        let rows = c["data"]["values"].as_array().unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0]["st"], 0.5);
        assert_eq!(rows[3]["st"], 3.5);
        assert_eq!(rows[0]["share"], 0.1); // 100 of 1000 draws
                                           // The text card carries every number the bars encode.
        assert!(
            p.text.contains("n=1000") && p.text.contains("mean=2") && p.text.contains("q95=3.5"),
            "{}",
            p.text
        );
    }

    /// An event's chart is two `Percentage` bars, not a 30-bin histogram of 0s and 1s.
    #[test]
    fn a_boolean_distribution_is_two_percentage_bars() {
        let p = to_flint(&summary(View::Hist, "knocked", Payload::One(dist1(true))));
        let c = &p.charts[0];
        assert_well_formed(c);
        let rows = c["data"]["values"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["outcome"], "true");
        assert_eq!(rows[1]["share"], 0.42);
        assert_eq!(c["semantic_types"]["share"]["semanticType"], "Percentage");
        assert!(p.text.contains("P(true)=0.42"), "{}", p.text);
    }

    /// A fan is three specs in back-to-front order (outer band, inner band, median) sharing one x
    /// field — that is what makes the host's post-compile layer merge sound.
    #[test]
    fn a_fan_is_three_layerable_specs_sharing_an_x_field() {
        let c = FanChart {
            cols: 3,
            n: 100,
            q05: vec![0.0, -1.0, -2.0],
            q25: vec![0.0, -0.5, -1.0],
            q50: vec![0.0, 0.0, 0.0],
            q75: vec![0.0, 0.5, 1.0],
            q95: vec![0.0, 1.0, 2.0],
            mean: vec![0.0, 0.0, 0.0],
        };
        let p = to_flint(&summary(View::Fan, "path", Payload::Fan(c)));
        assert_eq!(p.title, "fan(path)");
        assert_eq!(p.charts.len(), 3);
        for c in &p.charts {
            assert_well_formed(c);
            assert_eq!(
                c["chart_spec"]["encodings"]["x"], "index",
                "layers must share an x field"
            );
        }
        assert_eq!(p.charts[0]["chart_spec"]["chartType"], "Range Area Chart");
        assert_eq!(
            p.charts[0]["chart_spec"]["encodings"]["y2"], "q95",
            "outer band first"
        );
        assert_eq!(
            p.charts[1]["chart_spec"]["encodings"]["y2"], "q75",
            "inner band second"
        );
        assert_eq!(p.charts[2]["chart_spec"]["chartType"], "Line Chart");
        // The median's field is the variable's own name, so the merged y axis is titled `path`.
        assert_eq!(p.charts[2]["chart_spec"]["encodings"]["y"], "path");
        // A layered line must not anchor y at zero, or it fights the band's scale on merge.
        assert_eq!(
            p.charts[2]["chart_spec"]["chartProperties"]["includeZero_y"],
            false
        );
        assert!(
            p.text.contains("final q05=-2") && p.text.contains("q95=2"),
            "{}",
            p.text
        );
    }

    /// A check-mode fan carries no draws. It must degrade to a text card, not panic on `q50[0]`.
    #[test]
    fn an_empty_fan_emits_no_charts() {
        let c = FanChart {
            cols: 3,
            n: 0,
            q05: vec![],
            q25: vec![],
            q50: vec![],
            q75: vec![],
            q95: vec![],
            mean: vec![],
        };
        let p = to_flint(&summary(View::Fan, "path", Payload::Fan(c)));
        assert!(p.charts.is_empty());
        assert_eq!(p.text, "fan(path): (empty)");
    }

    #[test]
    fn a_correlation_matrix_is_a_heatmap_with_the_correlation_semantic_type() {
        let c = CorrMatrix {
            n: 2,
            corr: vec![1.0, -0.6, -0.6, 1.0],
        };
        let p = to_flint(&summary(View::CorrMatrix, "v", Payload::CorrMatrix(c)));
        assert_eq!(p.title, "corr(v)");
        assert_well_formed(&p.charts[0]);
        assert_eq!(p.charts[0]["chart_spec"]["chartType"], "Heatmap");
        // `Correlation` is what buys the diverging ±1 ramp — a plain `Number` would not.
        assert_eq!(p.charts[0]["semantic_types"]["corr"], "Correlation");
        // Indices are `Rank`, not `Category`: element 10 must sort after element 9, not after 1.
        assert_eq!(p.charts[0]["semantic_types"]["row"], "Rank");
        assert_eq!(p.charts[0]["data"]["values"].as_array().unwrap().len(), 4);
        assert!(
            p.text.contains("2×2 max |corr| off-diagonal = 0.6"),
            "{}",
            p.text
        );
    }

    /// A vector with spread layers a ±sd band under its mean line; a deterministic one is just the
    /// line (there is no band to draw, and an invisible zero-width area would only add ink).
    #[test]
    fn a_series_bands_its_mean_only_when_it_has_spread() {
        let spread = DistGrid {
            rows: 1,
            cols: 3,
            mean: vec![1.0, 2.0, 3.0],
            sd: vec![0.1, 0.2, 0.3],
        };
        let p = to_flint(&summary(View::Grid, "path", Payload::Grid(spread)));
        assert_eq!(p.charts.len(), 2);
        assert_eq!(p.charts[0]["chart_spec"]["chartType"], "Range Area Chart");
        assert_eq!(p.charts[1]["chart_spec"]["chartType"], "Line Chart");
        assert_eq!(p.charts[0]["data"]["values"][0]["lo"], 0.9);
        assert_eq!(
            p.charts[1]["chart_spec"]["chartProperties"]["includeZero_y"],
            false
        );

        let flat = DistGrid {
            rows: 1,
            cols: 3,
            mean: vec![1.0, 2.0, 3.0],
            sd: vec![0.0, 0.0, 0.0],
        };
        let p = to_flint(&summary(View::Grid, "xs", Payload::Grid(flat)));
        assert_eq!(p.charts.len(), 1);
        assert_eq!(p.charts[0]["chart_spec"]["chartType"], "Line Chart");
        // Unlayered, Flint's own zero-baseline judgment stands.
        assert!(p.charts[0]["chart_spec"]["chartProperties"].is_null());
    }

    #[test]
    fn a_matrix_is_a_heatmap_of_cell_means() {
        let g = DistGrid {
            rows: 2,
            cols: 3,
            mean: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            sd: vec![0.0; 6],
        };
        let p = to_flint(&summary(View::Grid, "M", Payload::Grid(g)));
        assert_eq!(p.charts.len(), 1);
        assert_well_formed(&p.charts[0]);
        assert_eq!(p.charts[0]["chart_spec"]["chartType"], "Heatmap");
        let rows = p.charts[0]["data"]["values"].as_array().unwrap();
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[5], json!({ "row": 1, "column": 2, "M": 6.0 }));
        assert!(p.text.contains("matrix 2×3"), "{}", p.text);
    }

    /// An estimate draws its 95% interval; an exact number has none, so it is text only.
    #[test]
    fn a_value_card_draws_an_interval_only_when_it_is_uncertain() {
        let p = to_flint(&summary(
            View::Value,
            "mu",
            Payload::Value(ValueCard { val: 2.5, se: 0.01 }),
        ));
        assert_eq!(p.charts.len(), 1);
        assert_well_formed(&p.charts[0]);
        assert_eq!(p.charts[0]["chart_spec"]["chartType"], "Ranged Dot Plot");
        // The value axis is the variable; the row axis is blank (no title, no tick, no legend).
        assert_eq!(
            p.charts[0]["chart_spec"]["encodings"],
            json!({ "x": "mu", "y": " " })
        );
        let rows = p.charts[0]["data"]["values"].as_array().unwrap();
        assert_eq!(rows.len(), 3, "low · estimate · high");
        assert_eq!(rows[0]["mu"], 2.5 - 1.96 * 0.01);
        assert_eq!(rows[1]["mu"], 2.5);
        assert_eq!(p.text, "mu: 2.5 ± 0.01 (95% CI 2.4804 … 2.5196)");

        let exact = to_flint(&summary(
            View::Value,
            "k",
            Payload::Value(ValueCard {
                val: 100.0,
                se: 0.0,
            }),
        ));
        assert!(
            exact.charts.is_empty(),
            "an exact scalar has no interval to draw"
        );
        assert_eq!(exact.text, "k: 100 (exact)");
    }

    #[test]
    fn a_scatter_names_its_axes_after_the_source_variables() {
        let d = Dist2 {
            n: 10,
            corr: 0.9,
            cov: 1.5,
            mean_a: 0.0,
            mean_b: 0.0,
            sd_a: 1.0,
            sd_b: 1.0,
            points: vec![(1.0, 2.0), (3.0, 4.0)],
        };
        let s = Summary {
            view: View::Scatter,
            label: "st".into(),
            label_b: Some("payoff".into()),
            payload: Payload::Two(d),
        };
        let p = to_flint(&s);
        assert_eq!(p.title, "scatter(st, payoff)");
        assert_well_formed(&p.charts[0]);
        assert_eq!(
            p.charts[0]["chart_spec"]["encodings"],
            json!({ "x": "st", "y": "payoff" })
        );
        assert_eq!(
            p.charts[0]["data"]["values"][0],
            json!({ "st": 1.0, "payoff": 2.0 })
        );
    }

    /// `scatter(x, x)` must not collapse both axes onto one column and silently drop a series.
    #[test]
    fn a_field_name_never_collides_with_a_fixed_column() {
        let d = Dist2 {
            n: 2,
            corr: 1.0,
            cov: 1.0,
            mean_a: 0.0,
            mean_b: 0.0,
            sd_a: 1.0,
            sd_b: 1.0,
            points: vec![(1.0, 1.0)],
        };
        let s = Summary {
            view: View::Scatter,
            label: "x".into(),
            label_b: Some("x".into()),
            payload: Payload::Two(d),
        };
        let p = to_flint(&s);
        assert_eq!(
            p.charts[0]["chart_spec"]["encodings"],
            json!({ "x": "x", "y": "x_" })
        );

        // A variable literally named `share` must not overwrite the histogram's share column.
        let p = to_flint(&summary(View::Hist, "share", Payload::One(dist1(false))));
        assert_eq!(
            p.charts[0]["chart_spec"]["encodings"],
            json!({ "x": "share_", "y": "share" })
        );
        assert_well_formed(&p.charts[0]);
    }

    #[test]
    fn explain_ranks_drivers_as_horizontal_percentage_bars() {
        let e = Explain {
            sd: 2.0,
            drivers: vec![
                Driver {
                    name: "vol".into(),
                    corr: -0.9,
                    share: 0.7,
                },
                Driver {
                    name: "drift".into(),
                    corr: 0.4,
                    share: 0.3,
                },
            ],
        };
        let p = to_flint(&summary(View::Explain, "payoff", Payload::Explain(e)));
        assert_eq!(p.title, "explain(payoff)");
        assert_well_formed(&p.charts[0]);
        // x is the share, y the name ⇒ horizontal bars, strongest driver first (data order).
        assert_eq!(
            p.charts[0]["chart_spec"]["encodings"],
            json!({ "x": "share", "y": "driver" })
        );
        assert_eq!(p.charts[0]["data"]["values"][0]["driver"], "vol");
        assert!(p.text.contains("drivers: vol 70%, drift 30%"), "{}", p.text);

        let none = to_flint(&summary(
            View::Explain,
            "y",
            Payload::Explain(Explain {
                sd: 1.0,
                drivers: vec![],
            }),
        ));
        assert!(none.charts.is_empty());
        assert!(none.text.contains("no named upstream variables"));
    }
}
