//! Long-form text chunking (port of `omnivoice/utils/text.py`):
//! `chunk_text_punctuation` splits text at sentence punctuation into
//! model-friendly chunks with abbreviation awareness, and `add_punctuation`
//! appends missing end punctuation to reference transcripts.

/// Characters after which a sentence may be split (`SPLIT_PUNCTUATION`).
const SPLIT_PUNCTUATION: &[char] = &[
    '.', ',', ';', ':', '!', '?', '。', '，', '；', '：', '！', '？',
];

/// Sentence-closing quotes/brackets: when they lead a sentence they belong to
/// the previous one (`CLOSING_MARKS`).
const CLOSING_MARKS: &[char] = &[
    '"', '\'', '“', '”', '‘', '’', '）', ']', '》', '>', '」', '】',
];

/// Characters that count as end punctuation (`END_PUNCTUATION`; the Python
/// set's only multi-char member `"……"` can never match a final single char).
const END_PUNCTUATION: &[char] = &[
    ';', ':', ',', '.', '!', '?', '…', ')', ']', '}', '"', '\'', '“', '”', '‘', '’', '；', '：',
    '，', '。', '！', '？', '、', '）', '】',
];

/// Titles/months/etc. whose trailing `.` must not end a sentence
/// (`ABBREVIATIONS`, verbatim).
const ABBREVIATIONS: &[&str] = &[
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.",
    "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.",
    "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.",
    "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
    "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
];

/// Split `text` into chunks of at most `chunk_len` characters at punctuation
/// boundaries, avoiding splits on common abbreviations (e.g. `Mr.`, `No.`).
///
/// A sentence longer than `chunk_len` becomes its own (over-length) chunk —
/// the reference never splits mid-sentence. When `min_chunk_len` is given,
/// shorter chunks are merged into their neighbour (the second chunk merges
/// into the first when the first is the short one).
pub fn chunk_text_punctuation(text: &str, chunk_len: usize, min_chunk_len: Option<usize>) -> Vec<String> {
    // 1. Split into sentences at punctuation.
    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut current_sentence: Vec<char> = Vec::new();
    for token in text.chars() {
        // If the first token of the current sentence is punctuation or a
        // closing mark, append it to the end of the previous sentence.
        if current_sentence.is_empty()
            && !sentences.is_empty()
            && (SPLIT_PUNCTUATION.contains(&token) || CLOSING_MARKS.contains(&token))
        {
            sentences.last_mut().expect("checked non-empty").push(token);
        } else {
            current_sentence.push(token);
            if SPLIT_PUNCTUATION.contains(&token) {
                let is_abbreviation = token == '.' && {
                    let joined: String = current_sentence.iter().collect();
                    let trimmed = joined.trim();
                    !trimmed.is_empty()
                        && ABBREVIATIONS.contains(&trimmed.split_whitespace().next_back().expect("non-empty"))
                };
                if !is_abbreviation {
                    sentences.push(std::mem::take(&mut current_sentence));
                }
            }
        }
    }
    if !current_sentence.is_empty() {
        sentences.push(current_sentence);
    }

    // 2. Greedily merge sentences into chunks (character counts).
    let mut merged_chunks: Vec<Vec<char>> = Vec::new();
    let mut current_chunk: Vec<char> = Vec::new();
    for sentence in sentences {
        if current_chunk.len() + sentence.len() <= chunk_len {
            current_chunk.extend(sentence);
        } else {
            if !current_chunk.is_empty() {
                merged_chunks.push(std::mem::take(&mut current_chunk));
            }
            current_chunk = sentence;
        }
    }
    if !current_chunk.is_empty() {
        merged_chunks.push(current_chunk);
    }

    // 3. Merge undersized chunks into their neighbour.
    let final_chunks = match min_chunk_len {
        Some(min_len) => {
            let first_chunk_short = !merged_chunks.is_empty() && merged_chunks[0].len() < min_len;
            let mut out: Vec<Vec<char>> = Vec::new();
            for (i, chunk) in merged_chunks.into_iter().enumerate() {
                if i == 1 && first_chunk_short {
                    out.last_mut().expect("chunk 0 was appended").extend(chunk);
                } else if chunk.len() >= min_len || out.is_empty() {
                    out.push(chunk);
                } else {
                    out.last_mut().expect("non-empty by construction").extend(chunk);
                }
            }
            out
        }
        None => merged_chunks,
    };

    final_chunks
        .into_iter()
        .map(|chunk| chunk.into_iter().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Append end punctuation when the text has none: `。` when it contains any
/// CJK character, `.` otherwise.
pub fn add_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return text.to_string();
    }
    let last = text.chars().next_back().expect("non-empty");
    if END_PUNCTUATION.contains(&last) {
        return text.to_string();
    }
    let is_chinese = text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    format!("{text}{}", if is_chinese { '。' } else { '.' })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text_punctuation("", 10, Some(3)).is_empty());
        assert!(chunk_text_punctuation("   ", 10, Some(3)).is_empty());
    }

    #[test]
    fn short_text_stays_whole() {
        assert_eq!(chunk_text_punctuation("你好，世界。", 50, Some(3)), vec!["你好，世界。"]);
    }

    #[test]
    fn splits_on_sentence_boundaries_and_packs_to_budget() {
        let text = "这是一句用来测试自动切分的完整句子。".repeat(12);
        // 15 s budget calibrated like the engine: 375 frames / ~18 chars.
        let chunks = chunk_text_punctuation(&text, 18, Some(3));
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.ends_with('。'), "chunk does not end on a boundary: {c}");
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn overlong_sentence_becomes_single_chunk() {
        let text = "词".repeat(300);
        // No punctuation: one sentence, one (over-budget) chunk.
        assert_eq!(chunk_text_punctuation(&text, 70, Some(3)), vec![text.clone()]);
    }

    #[test]
    fn abbreviation_not_split() {
        // "Mr." must not split, "arrived." must.
        let chunks = chunk_text_punctuation("Mr. Smith arrived. He left.", 4, Some(3));
        assert_eq!(chunks, vec!["Mr. Smith arrived.", "He left."]);
    }

    #[test]
    fn eg_and_ie_trace_matches_python() {
        // The guard checks the last whitespace-word of the current sentence,
        // so "e." splits before "g." can join it — the dotted abbreviations
        // rely on min-chunk merging instead (verified against the Python).
        let chunks = chunk_text_punctuation("e.g. apples, i.e. fruit.", 4, Some(3));
        assert_eq!(chunks, vec!["e.g.", "apples,", "i.e.", "fruit."]);
    }

    #[test]
    fn closing_mark_attaches_to_previous() {
        // American quoting: the period is inside the quotes, so the closing
        // ” leads the next sentence and is appended to the previous one.
        let chunks = chunk_text_punctuation("He said “hello.” Then he left.", 4, Some(3));
        assert_eq!(chunks, vec!["He said “hello.”", "Then he left."]);
    }

    #[test]
    fn min_chunk_len_merges_short_chunks() {
        // First chunk short -> second merges into it.
        let chunks = chunk_text_punctuation("好. 很长很长的第二句话在这里继续着.", 1, Some(3));
        assert_eq!(chunks.len(), 1);
        // A short middle chunk merges into the previous one.
        let chunks = chunk_text_punctuation("第一句话.好.第三句话继续在这里延伸.", 5, Some(3));
        assert_eq!(chunks, vec!["第一句话.好.", "第三句话继续在这里延伸."]);
    }

    #[test]
    fn add_punctuation_cjk_gets_full_stop() {
        assert_eq!(add_punctuation("你好"), "你好。");
        assert_eq!(add_punctuation(" hello world "), "hello world.");
    }

    #[test]
    fn add_punctuation_keeps_existing() {
        assert_eq!(add_punctuation("ok?"), "ok?");
        assert_eq!(add_punctuation("好。"), "好。");
        assert_eq!(add_punctuation("wait…"), "wait…");
        assert_eq!(add_punctuation(""), "");
    }
}
