//! Qwen2 byte-level BPE tokenizer loaded from the unified HF
//! `tokenizer.json` (the OmniVoice repo ships no vocab.json/merges.txt pair).
//!
//! The file is a serialized `tokenizers` pipeline: NFC normalizer + the GPT-2
//! pre-tokenize regex + byte-level BPE, plus 33 `added_tokens` that include
//! the seven `<|denoise|>…<|text_end|>` control tokens (151669..151675).
//! The encode path: NFC, added-token splitting
//! (longest match wins), regex split, then rank-ordered BPE merges.

use anyhow::{Context, Result};
use fancy_regex::Regex;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct TokJson {
    model: ModelJson,
    #[serde(default)]
    added_tokens: Vec<AddedJson>,
}

#[derive(Deserialize)]
struct ModelJson {
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct AddedJson {
    id: u32,
    content: String,
}

/// GPT-2 bytes-to-unicode table: printable bytes map to themselves, the rest
/// to codepoints 256.., so every byte round-trips through vocab strings.
fn bytes_to_unicode() -> Vec<char> {
    let mut out = Vec::with_capacity(256);
    let mut n = 256u32;
    for b in 0u32..256 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if printable {
            out.push(char::from_u32(b).unwrap());
        } else {
            out.push(char::from_u32(n).unwrap());
            n += 1;
        }
    }
    out
}

enum Segment<'a> {
    Added(u32),
    Text(&'a str),
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    added_by_len: Vec<(String, u32)>, // longest-first for scanning
    bpe_cache: RefCell<HashMap<String, Vec<u32>>>,
    pat: Regex,
    byte_encoder: Vec<char>,
}

impl Tokenizer {
    /// Load from a HF `tokenizer.json` file.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let tok: TokJson =
            serde_json::from_reader(file).with_context(|| format!("parse {}", path.display()))?;
        let vocab = tok.model.vocab;
        let mut merges = HashMap::with_capacity(tok.model.merges.len());
        for (i, pair) in tok.model.merges.iter().enumerate() {
            if pair.len() != 2 {
                anyhow::bail!("bad merge #{}: {pair:?}", i);
            }
            merges.insert((pair[0].clone(), pair[1].clone()), i);
        }
        let mut added_by_len: Vec<(String, u32)> = tok
            .added_tokens
            .into_iter()
            .map(|t| (t.content, t.id))
            .collect();
        added_by_len.sort_by_key(|t| std::cmp::Reverse(t.0.len()));
        // Pre-tokenize regex matching tokenizer.json's Split pre-tokenizer.
        let pat = Regex::new(
            r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        )?;
        Ok(Self {
            vocab,
            merges,
            added_by_len,
            bpe_cache: RefCell::new(HashMap::new()),
            pat,
            byte_encoder: bytes_to_unicode(),
        })
    }

    /// HF-compatible encode: NFC-normalize, added tokens pass through
    /// verbatim, plain text is regex-split into words and BPE-encoded per word.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let text: String = text.nfc().collect();
        let mut ids = Vec::new();
        for seg in self.split_added(&text) {
            match seg {
                Segment::Added(id) => ids.push(id),
                Segment::Text(t) => {
                    for m in self.pat.find_iter(t) {
                        let word = m?.as_str();
                        if word.is_empty() {
                            continue;
                        }
                        ids.extend(self.bpe_word(word)?);
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Scan `text` once; at every char boundary the longest added token wins.
    fn split_added<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut out = Vec::new();
        let mut plain_start = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let mut hit: Option<&(String, u32)> = None;
            for a in &self.added_by_len {
                if text[i..].starts_with(a.0.as_str()) {
                    hit = Some(a);
                    break;
                }
            }
            if let Some((tok, id)) = hit {
                if plain_start < i {
                    out.push(Segment::Text(&text[plain_start..i]));
                }
                out.push(Segment::Added(*id));
                i += tok.len();
                plain_start = i;
            } else {
                i += 1;
            }
        }
        if plain_start < text.len() {
            out.push(Segment::Text(&text[plain_start..]));
        }
        out
    }

    /// Standard GPT-2 BPE: map bytes to unicode chars, then repeatedly merge
    /// the lowest-ranked adjacent pair.
    fn bpe_word(&self, word: &str) -> Result<Vec<u32>> {
        if let Some(ids) = self.bpe_cache.borrow().get(word) {
            return Ok(ids.clone());
        }
        let mapped: String = word.bytes().map(|b| self.byte_encoder[b as usize]).collect();
        let mut symbols: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
        while symbols.len() >= 2 {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..symbols.len() - 1 {
                if let Some(&r) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            symbols[i] = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.remove(i + 1);
        }
        let mut ids = Vec::with_capacity(symbols.len());
        for s in &symbols {
            let id = self
                .vocab
                .get(s)
                .with_context(|| format!("token {s:?} not in vocab"))?;
            ids.push(*id);
        }
        self.bpe_cache.borrow_mut().insert(word.to_string(), ids.clone());
        Ok(ids)
    }
}
