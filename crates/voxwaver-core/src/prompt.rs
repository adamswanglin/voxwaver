use crate::tokenizer::Tokenizer;
use anyhow::{ensure, Result};

/// The `[num_codebooks + 1, T]` "values" tensor content from
/// `ContentSequence::encode_for_inference`: row 0 holds token ids, rows 1..=10
/// hold the codebook codes at semantic-token positions (0 elsewhere).
pub struct Prompt {
    /// length T
    pub len: usize,
    pub tokens: Vec<u32>,
    /// codes[i] is codebook i (0 = semantic) at each position; 0 at non-semantic positions
    pub codes: Vec<Vec<u32>>, // [10][T]
}

impl Prompt {
    pub fn semantic_positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.tokens
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t >= 151_658 && t <= 155_753)
            .map(|(i, _)| i)
    }
}

/// A prior speech turn fed back into the prompt (iterative prompting):
/// transcript text + DAC codes `[10][T]`. The reference-audio clone turn and
/// previously generated chunks are both expressed this way.
pub struct Turn<'a> {
    pub text: &'a str,
    pub codes: &'a [Vec<u32>],
}

/// Build the prompt exactly as the s1-mini `ContentSequence(modality="interleave")`
/// inference path does (`fish_speech/models/text2semantic/inference.py`):
///
/// ```text
/// <|interleave|>[<|speaker:0|>{TURN_TEXT}{semantic tokens}<|im_end|>]...<|speaker:0|>{TEXT}
/// ```
/// Each completed turn (reference audio or a previously generated chunk)
/// contributes a `text + semantic tokens + <|im_end|>` segment; the model
/// generates semantic tokens directly after the final target text.
/// There are no `<|im_start|>`/system/`<|voice|>` wrappers; the model generates
/// semantic tokens directly after the target text. `<|speaker:0|>` is not a
/// special token — it is BPE'd as plain text, exactly like upstream.
pub fn build(
    tok: &mut Tokenizer,
    text: &str,
    history: &[Turn<'_>],
    max_seq_len: usize,
    num_codebooks: usize,
) -> Result<Prompt> {
    let mut tokens = Vec::new();
    let mut semantic_codes: Vec<(u32 /*offset*/, Vec<u32> /*10 codes*/)> = Vec::new();

    let push_text = |tok: &mut Tokenizer, tokens: &mut Vec<u32>, s: &str| {
        tokens.extend(tok.encode(s));
    };

    push_text(tok, &mut tokens, "<|interleave|>");
    for turn in history {
        ensure!(
            turn.codes.len() == num_codebooks,
            "turn codes must have {} rows, got {}",
            num_codebooks,
            turn.codes.len()
        );
        push_text(tok, &mut tokens, "<|speaker:0|>");
        push_text(tok, &mut tokens, &clean_text(turn.text));
        for (i, &code) in turn.codes[0].iter().enumerate() {
            tokens.push(tok.semantic_token_id(code));
            semantic_codes.push((tokens.len() as u32 - 1, collect_column(turn.codes, i)));
        }
        push_text(tok, &mut tokens, "<|im_end|>");
    }
    push_text(tok, &mut tokens, "<|speaker:0|>");
    push_text(tok, &mut tokens, &clean_text(text));

    ensure!(
        tokens.len() <= max_seq_len - 2048,
        "prompt too long: {} > {}",
        tokens.len(),
        max_seq_len - 2048
    );

    let mut codes = vec![vec![0u32; tokens.len()]; num_codebooks];
    for (pos, col) in semantic_codes {
        for (cb, &v) in col.iter().enumerate() {
            codes[cb][pos as usize] = v;
        }
    }

    Ok(Prompt {
        len: tokens.len(),
        tokens,
        codes,
    })
}

/// Port of `fish_speech/text/clean.py::clean_text`: trim, map curly single
/// quotes, strip emoji, collapse repeated commas.
pub fn clean_text(text: &str) -> String {
    let text = text.trim();
    let text = text.replace('‘', "'").replace('’', "'");
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let c = ch as u32;
        let emoji = (0x1F600..=0x1F64F).contains(&c)
            || (0x1F300..=0x1F5FF).contains(&c)
            || (0x1F680..=0x1F6FF).contains(&c)
            || (0x1F1E0..=0x1F1FF).contains(&c);
        if !emoji {
            out.push(ch);
        }
    }
    // re.sub(r"[,]{2,}", ...) — collapse runs of commas to one
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_comma = false;
    for ch in out.chars() {
        if ch == ',' {
            if !prev_comma {
                collapsed.push(ch);
            }
            prev_comma = true;
        } else {
            collapsed.push(ch);
            prev_comma = false;
        }
    }
    collapsed
}

fn collect_column(codes: &[Vec<u32>], i: usize) -> Vec<u32> {
    codes.iter().map(|row| row[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_layout() {
        let mut tok = Tokenizer::load(std::path::Path::new(
            "/Users/wanglin/app/projects/github/s1-mini",
        ))
        .unwrap();
        let p = build(&mut tok, "你好世界", &[], 8192, 10).unwrap();
        assert_eq!(p.tokens[0], tok.special_id("<|interleave|>").unwrap());
        assert_eq!(&p.tokens[1..8], &[27, 91, 64476, 25, 15, 91, 29]);
        // no semantic positions
        assert_eq!(p.semantic_positions().count(), 0);
    }

    #[test]
    fn reference_layout() {
        let mut tok = Tokenizer::load(std::path::Path::new(
            "/Users/wanglin/app/projects/github/s1-mini",
        ))
        .unwrap();
        let codes: Vec<Vec<u32>> = (0..10)
            .map(|cb| (0..5).map(|t| (cb * 100 + t) as u32).collect())
            .collect();
        let p = build(
            &mut tok,
            "hello",
            &[Turn {
                text: "hi there",
                codes: &codes,
            }],
            8192,
            10,
        )
        .unwrap();
        let sems: Vec<usize> = p.semantic_positions().collect();
        assert_eq!(sems.len(), 5);
        for (i, &pos) in sems.iter().enumerate() {
            assert_eq!(p.tokens[pos], tok.semantic_token_id(i as u32));
            for cb in 0..10 {
                assert_eq!(p.codes[cb][pos], codes[cb][i]);
            }
        }
        // <|im_end|> after the reference, then speaker + target text
        assert_eq!(p.tokens[sems[4] + 1], tok.im_end);
        assert_eq!(&p.tokens[sems[4] + 2..sems[4] + 9], &[27, 91, 64476, 25, 15, 91, 29]);
    }

    /// Full parity with the PyTorch ContentSequence build over the real
    /// reference (/tmp/prompt_ref.json was dumped from the working harness).
    #[test]
    fn parity_with_torch_prompt() {
        let data = match std::fs::read_to_string("/tmp/prompt_ref.json") {
            Ok(d) => d,
            Err(_) => return, // reference dump not present on this machine
        };
        let v: serde_json::Value = serde_json::from_str(&data).unwrap();
        let want: Vec<u32> = v["tokens"].as_array().unwrap().iter()
            .map(|x| x.as_u64().unwrap() as u32).collect();
        let ref_codes: Vec<Vec<u32>> = v["ref_codes"].as_array().unwrap().iter()
            .map(|row| row.as_array().unwrap().iter()
                .map(|x| x.as_u64().unwrap() as u32).collect())
            .collect();
        let mut tok = Tokenizer::load(std::path::Path::new(
            "/Users/wanglin/app/projects/github/s1-mini",
        ))
        .unwrap();
        let p = build(
            &mut tok,
            "希望你以后越来越好，天天开心。",
            &[Turn {
                text: "这是一段用于测试的参考音频，希望一切顺利。",
                codes: &ref_codes,
            }],
            8192,
            10,
        )
        .unwrap();
        assert_eq!(p.tokens, want);
    }

    #[test]
    fn clean() {
        assert_eq!(clean_text("  hi  "), "hi");
        assert_eq!(clean_text("don‘t"), "don't");
        assert_eq!(clean_text("a,,,b"), "a,b");
        assert_eq!(clean_text("a,,b,"), "a,b,");
        assert_eq!(clean_text("no emoji \u{1F600}\u{1F44D} here"), "no emoji  here");
    }
}
