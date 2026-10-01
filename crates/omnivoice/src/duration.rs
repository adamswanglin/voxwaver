//! Rule-based text duration estimator (port of `duration.py`
//! `RuleDurationEstimator`): per-character phonetic weights across Unicode
//! scripts, calibrated against a reference text/duration pair.
//!
//! Category mapping approximates Python's `unicodedata.category` with Rust's
//! `char` predicates: `is_alphabetic` covers the letter classes that reach the
//! script-range table, `is_numeric`/`is_whitespace` map to N/Z. Combining
//! marks (Python category `M`, weight 0) are not distinguishable here and fall
//! through to the punctuation weight — a negligible bias for duration
//! estimation.

/// Script weight table: relative speaking time per character vs a Latin letter
/// (1.0 = one Latin character, ~40-50 ms).
fn script_weight(name: &str) -> f64 {
    match name {
        "cjk" => 3.0,
        "hangul" => 2.5,
        "kana" => 2.2,
        "ethiopic" => 3.0,
        "yi" => 3.0,
        "indic" => 1.8,
        "thai_lao" => 1.5,
        "khmer_myanmar" => 1.8,
        "arabic" => 1.5,
        "hebrew" => 1.5,
        "latin" => 1.0,
        "cyrillic" => 1.0,
        "greek" => 1.0,
        "armenian" => 1.0,
        "georgian" => 1.0,
        "punctuation" => 0.5,
        "space" => 0.2,
        "digit" => 3.5,
        "mark" => 0.0,
        _ => 1.0, // "default"
    }
}

/// Unicode block ranges `(end_codepoint, script)` for binary search, copied
/// verbatim from the Python estimator.
const RANGES: &[(u32, &str)] = &[
    (0x02AF, "latin"),
    (0x03FF, "greek"),
    (0x052F, "cyrillic"),
    (0x058F, "armenian"),
    (0x05FF, "hebrew"),
    (0x077F, "arabic"),
    (0x089F, "arabic"),
    (0x08FF, "arabic"),
    (0x097F, "indic"),
    (0x09FF, "indic"),
    (0x0A7F, "indic"),
    (0x0AFF, "indic"),
    (0x0B7F, "indic"),
    (0x0BFF, "indic"),
    (0x0C7F, "indic"),
    (0x0CFF, "indic"),
    (0x0D7F, "indic"),
    (0x0DFF, "indic"),
    (0x0EFF, "thai_lao"),
    (0x0FFF, "indic"),
    (0x109F, "khmer_myanmar"),
    (0x10FF, "georgian"),
    (0x11FF, "hangul"),
    (0x137F, "ethiopic"),
    (0x139F, "ethiopic"),
    (0x13FF, "default"),
    (0x167F, "default"),
    (0x169F, "default"),
    (0x16FF, "default"),
    (0x171F, "default"),
    (0x173F, "default"),
    (0x175F, "default"),
    (0x177F, "default"),
    (0x17FF, "khmer_myanmar"),
    (0x18AF, "default"),
    (0x18FF, "default"),
    (0x194F, "indic"),
    (0x19DF, "indic"),
    (0x19FF, "khmer_myanmar"),
    (0x1A1F, "indic"),
    (0x1AAF, "indic"),
    (0x1B7F, "indic"),
    (0x1BBF, "indic"),
    (0x1BFF, "indic"),
    (0x1C4F, "indic"),
    (0x1C7F, "indic"),
    (0x1C8F, "cyrillic"),
    (0x1CBF, "georgian"),
    (0x1CCF, "indic"),
    (0x1CFF, "indic"),
    (0x1D7F, "latin"),
    (0x1DBF, "latin"),
    (0x1DFF, "default"),
    (0x1EFF, "latin"),
    (0x309F, "kana"),
    (0x30FF, "kana"),
    (0x312F, "cjk"),
    (0x318F, "hangul"),
    (0x9FFF, "cjk"),
    (0xA4CF, "yi"),
    (0xA4FF, "default"),
    (0xA63F, "default"),
    (0xA69F, "cyrillic"),
    (0xA6FF, "default"),
    (0xA7FF, "latin"),
    (0xA82F, "indic"),
    (0xA87F, "default"),
    (0xA8DF, "indic"),
    (0xA8FF, "indic"),
    (0xA92F, "indic"),
    (0xA95F, "indic"),
    (0xA97F, "hangul"),
    (0xA9DF, "indic"),
    (0xA9FF, "khmer_myanmar"),
    (0xAA5F, "indic"),
    (0xAA7F, "khmer_myanmar"),
    (0xAADF, "indic"),
    (0xAAFF, "indic"),
    (0xAB2F, "ethiopic"),
    (0xAB6F, "latin"),
    (0xABBF, "default"),
    (0xABFF, "indic"),
    (0xD7AF, "hangul"),
    (0xFAFF, "cjk"),
    (0xFDFF, "arabic"),
    (0xFE6F, "default"),
    (0xFEFF, "arabic"),
    (0xFFEF, "latin"),
];

/// Weight of a single character (Python `_get_char_weight`).
pub(crate) fn char_weight(c: char) -> f64 {
    let code = c as u32;
    if (65..=90).contains(&code) || (97..=122).contains(&code) {
        return script_weight("latin");
    }
    if code == 32 {
        return script_weight("space");
    }
    // Ignore Arabic Tatweel.
    if code == 0x0640 {
        return script_weight("mark");
    }
    // Python checks category M -> mark here; see the module comment.
    if c.is_alphabetic() {
        // Binary search for the Unicode block.
        let idx = RANGES.partition_point(|&(end, _)| end < code);
        if idx < RANGES.len() {
            return script_weight(RANGES[idx].1);
        }
        // Upper planes (CJK Ext B/C/D, historic scripts).
        if code > 0x20000 {
            return script_weight("cjk");
        }
        return script_weight("default");
    }
    if c.is_numeric() {
        return script_weight("digit");
    }
    if c.is_whitespace() {
        return script_weight("space");
    }
    // Punctuation and symbols (and, by approximation, marks).
    script_weight("punctuation")
}

fn calculate_total_weight(text: &str) -> f64 {
    text.chars().map(char_weight).sum()
}

/// Total phonetic weight of `text` — the estimator's raw measure before the
/// frame conversion (used for chunk packing).
pub fn total_weight(text: &str) -> f64 {
    calculate_total_weight(text)
}

/// Estimated duration (in frames, matching the reference duration's unit) of
/// `target_text`, calibrated on (`ref_text`, `ref_duration`).
///
/// Durations below `low_threshold` get a power-curve boost: the model tends to
/// rush very short texts, so they are pushed toward the threshold with
/// `(est/threshold)^(1/boost_strength)`.
#[allow(clippy::too_many_arguments)]
pub fn estimate_duration(
    target_text: &str,
    ref_text: &str,
    ref_duration: f64,
    low_threshold: f64,
    boost_strength: f64,
) -> f64 {
    if ref_duration <= 0.0 || ref_text.is_empty() {
        return 0.0;
    }
    let ref_weight = calculate_total_weight(ref_text);
    if ref_weight == 0.0 {
        return 0.0;
    }
    let speed_factor = ref_weight / ref_duration;
    let target_weight = calculate_total_weight(target_text);
    let estimated = target_weight / speed_factor;
    if estimated < low_threshold {
        let alpha = 1.0 / boost_strength;
        low_threshold * (estimated / low_threshold).powf(alpha)
    } else {
        estimated
    }
}

/// Pipeline defaults: `estimate_duration(text, "Nice to meet you.", 25)` with
/// `low_threshold=50`, `boost_strength=3`.
pub fn estimate_duration_frames(target_text: &str) -> f64 {
    estimate_duration(
        target_text,
        crate::config::DURATION_REF_TEXT,
        crate::config::DURATION_REF_FRAMES,
        50.0,
        3.0,
    )
}
