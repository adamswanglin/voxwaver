use anyhow::{bail, Context, Result};
use base64::Engine;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

/// Minimal tiktoken BPE tokenizer (Qwen2.5 base + fish-speech special tokens),
/// equivalent to `AutoTokenizer(...).encode(text, add_special_tokens=False,
/// allowed_special="all")`.
pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    id_to_bytes: HashMap<u32, Vec<u8>>,
    specials: HashMap<String, u32>,
    special_to_id: HashMap<String, u32>,
    pattern: fancy_regex::Regex,
    cache: HashMap<String, Vec<u32>>,
    pub vocab_size: u32,

    pub im_start: u32,
    pub im_end: u32,
    pub voice: u32,
    pub semantic_begin: u32,
    pub semantic_end: u32,
}

/// fish-speech pre-tokenization pattern (FISH_TIKTOKEN_PATTERN): differs from
/// the stock GPT-4o pattern in that punctuation is its own piece (`\p{P}`) and
/// digits split one at a time (`\p{N}`); `\s+(\?!\S)` escapes the "?" so it
/// matches a literal question mark.
const PAT: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|\p{P}",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(\?!\S)",
    r"|\s+",
);

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let tiktoken_path = dir.join("tokenizer.tiktoken");
        let f = std::fs::File::open(&tiktoken_path)
            .with_context(|| format!("open {}", tiktoken_path.display()))?;
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut id_to_bytes: HashMap<u32, Vec<u8>> = HashMap::new();
        for line in std::io::BufReader::new(f).lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let (b64, rank) = line
                .split_once(' ')
                .with_context(|| format!("bad tiktoken line: {line:?}"))?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64.trim())
                .with_context(|| format!("bad base64 {b64:?}"))?;
            let rank: u32 = rank.trim().parse()?;
            ranks.insert(bytes.clone(), rank);
            id_to_bytes.insert(rank, bytes);
        }

        let specials_path = dir.join("special_tokens.json");
        let specials: HashMap<String, u32> =
            serde_json::from_str(&std::fs::read_to_string(&specials_path)?)
                .with_context(|| format!("parse {}", specials_path.display()))?;

        let special_to_id: HashMap<String, u32> = specials.clone();
        let semantic_ids: Vec<u32> = (0..4096)
            .filter_map(|i| special_to_id.get(&format!("<|semantic:{i}|>")).copied())
            .collect();
        if semantic_ids.is_empty() {
            bail!("no <|semantic:N|> tokens in {}", specials_path.display());
        }
        let semantic_begin = *semantic_ids.iter().min().unwrap();
        let semantic_end = *semantic_ids.iter().max().unwrap();

        let get = |name: &str| -> Result<u32> {
            special_to_id
                .get(name)
                .copied()
                .with_context(|| format!("special token {name} missing"))
        };
        let (im_start, im_end, voice) = (
            get("<|im_start|>")?,
            get("<|im_end|>")?,
            get("<|voice|>")?,
        );

        Ok(Self {
            vocab_size: ranks.len() as u32 + specials.len() as u32,
            ranks,
            id_to_bytes,
            specials,
            special_to_id,
            pattern: fancy_regex::Regex::new(PAT)?,
            cache: HashMap::new(),
            im_start,
            im_end,
            voice,
            semantic_begin,
            semantic_end,
        })
    }

    pub fn semantic_token_id(&self, code: u32) -> u32 {
        self.semantic_begin + code
    }

    /// Token id -> code for a semantic token (unclamped).
    pub fn semantic_code(&self, token_id: u32) -> u32 {
        token_id.saturating_sub(self.semantic_begin)
    }

    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.special_to_id.get(name).copied()
    }

    /// Encode text with inline special tokens parsed (allowed_special="all").
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            // try special token match (all specials start with "<|" and end "|>")
            if bytes[pos] == b'<' && pos + 1 < bytes.len() && bytes[pos + 1] == b'|' {
                if let Some((id, len)) = self.match_special(&bytes[pos..]) {
                    out.push(id);
                    pos += len;
                    continue;
                }
            }
            // find next special start; encode the text run in between
            let mut end = pos;
            while end < bytes.len() {
                if bytes[end] == b'<' && end + 1 < bytes.len() && bytes[end + 1] == b'|' {
                    if self.match_special(&bytes[end..]).is_some() {
                        break;
                    }
                }
                end += 1;
            }
            let run = &text[pos..end];
            self.encode_bpe_run(run, &mut out);
            pos = end;
        }
        out
    }

    fn match_special(&self, bytes: &[u8]) -> Option<(u32, usize)> {
        let max_scan = bytes.len().min(48);
        for i in (2..max_scan).rev() {
            if bytes[i] == b'>' && i >= 3 && bytes[i - 1] == b'|' {
                if let Ok(s) = std::str::from_utf8(&bytes[..=i]) {
                    if let Some(id) = self.specials.get(s) {
                        return Some((*id, i + 1));
                    }
                }
            }
        }
        None
    }

    fn encode_bpe_run(&mut self, run: &str, out: &mut Vec<u32>) {
        if run.is_empty() {
            return;
        }
        if let Some(ids) = self.cache.get(run) {
            out.extend_from_slice(ids);
            return;
        }
        let mut ids = Vec::new();
        for piece in self.pattern.find_iter(run) {
            let piece = match piece {
                Ok(m) => m.as_str(),
                Err(_) => continue,
            };
            self.bpe(piece.as_bytes(), &mut ids);
        }
        self.cache.insert(run.to_string(), ids.clone());
        out.extend_from_slice(&ids);
    }

    /// Standard tiktoken byte-pair merge over ranks.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        // start from single-byte tokens
        let mut parts: Vec<(Vec<u8>, u32)> = Vec::with_capacity(piece.len());
        for &b in piece {
            let id = match self.ranks.get(&vec![b]) {
                Some(id) => *id,
                // single byte missing from ranks cannot happen for tiktoken bases
                None => continue,
            };
            parts.push((vec![b], id));
        }
        while parts.len() > 1 {
            // find the adjacent pair with the lowest merge rank
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                let mut merged = parts[i].0.clone();
                merged.extend_from_slice(&parts[i + 1].0);
                if let Some(rank) = self.ranks.get(&merged) {
                    let better = match best {
                        Some((r, _)) => *rank < r,
                        None => true,
                    };
                    if better {
                        best = Some((*rank, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let (mut bytes, id0) = parts[i].clone();
                    bytes.extend_from_slice(&parts[i + 1].0);
                    let id = self.ranks.get(&bytes).copied().unwrap_or(id0);
                    parts[i] = (bytes, id);
                    parts.remove(i + 1);
                }
                None => break,
            }
        }
        for (_, id) in parts {
            out.push(id);
        }
    }

    /// Lossy decode for debug output.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(b);
            } else if let Some(s) = self
                .specials
                .iter()
                .find(|(_, v)| **v == id)
                .map(|(k, _)| k.clone().into_bytes())
            {
                bytes.extend_from_slice(&s);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        Tokenizer::load(Path::new(
            "/Users/wanglin/app/projects/github/s1-mini",
        ))
        .unwrap()
    }

    #[test]
    fn special_ids() {
        let t = tok();
        assert_eq!(t.im_start, 151646);
        assert_eq!(t.im_end, 151647);
        assert_eq!(t.voice, 151653);
        assert_eq!(t.semantic_begin, 151658);
        assert_eq!(t.semantic_end, 155753);
        assert_eq!(t.semantic_token_id(1234), 151658 + 1234);
    }

    #[test]
    fn encode_roundtrip() {
        let mut t = tok();
        for s in [
            "hello world",
            "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
            "Elmo's 12345 test!!!\n\r\n  multiple   spaces\ttab",
            "mixed 中文 and English text with 標點, «quotes» and—dashes…",
        ] {
            let ids = t.encode(s);
            assert!(!ids.is_empty(), "{s}");
            let decoded = t.decode(&ids);
            assert_eq!(decoded, s, "roundtrip failed for {s:?}");
        }
    }

    #[test]
    fn encode_inline_specials() {
        let mut t = tok();
        let ids = t.encode("<|im_start|>assistant\n<|voice|>hello<|semantic:7|><|im_end|>\n");
        assert_eq!(ids[0], 151646);
        assert!(ids.contains(&151653));
        assert!(ids.contains(&(151658 + 7)));
        // <|im_end|> is parsed as a special, then "\n" is BPE'd separately
        let last = *ids.last().unwrap();
        assert_eq!(t.decode(&[last]), "\n");
        assert_eq!(ids[ids.len() - 2], 151647);
    }


    /// Reference vectors produced by tiktoken with FISH_TIKTOKEN_PATTERN and
    /// the s1-mini ranks/specials, allowed_special="all". Note punctuation
    /// becomes its own token (id 0 for "!") and digits split individually.
    #[test]
    fn matches_tiktoken_reference() {
        let mut t = tok();
        let cases: &[(&str, &[u32])] = &[
            // "<|speaker:0|>" is NOT a special token: plain BPE, like upstream
            ("<|speaker:0|>", &[27, 91, 64476, 25, 15, 91, 29]),
            (
                "这是一段用于测试的参考音频，希望一切顺利。",
                &[43288, 99639, 37474, 100751, 81705, 9370, 101275, 111268, 3837, 99880, 101109, 102088, 1773],
            ),
            (
                "希望你以后越来越好，天天开心。",
                &[99880, 56568, 103934, 115360, 3837, 104837, 102313, 1773],
            ),
            ("Hello world.", &[9707, 1879, 13]),
            ("你好，世界。", &[108386, 3837, 99489, 1773]),
            (
                "Elmo's 12345 test!!!\n\r\n  multiple   spaces\ttab",
                &[6582, 6355, 594, 220, 16, 17, 18, 19, 20, 1273, 0, 0, 0, 198, 319, 256, 35673, 262, 44285, 58149],
            ),
            (
                "mixed 中文 and English text with 標點, «quotes» and—dashes…",
                &[56685, 72858, 16744, 323, 6364, 1467, 448, 6567, 101, 247, 100758, 11, 12486, 53282, 12992, 323, 2293, 67, 14051, 1940],
            ),
            (
                "<|interleave|><|speaker:0|>hi there<|im_end|><|speaker:0|>",
                &[151654, 27, 91, 64476, 25, 15, 91, 29, 6023, 1052, 151647, 27, 91, 64476, 25, 15, 91, 29],
            ),
            (
                "It costs $5.50, ok? 100%!",
                &[2132, 7049, 400, 20, 13, 20, 15, 11, 5394, 30, 220, 16, 15, 15, 4, 0],
            ),
        ];
        for (s, want) in cases {
            assert_eq!(&t.encode(s), want, "mismatch for {s:?}");
        }
    }

    #[test]
    fn no_false_special_split() {
        let mut t = tok();
        // "<|bogus|>" is not a special token: must be BPE'd and roundtrip
        let s = "text <|bogus|> more";
        let ids = t.encode(s);
        assert_eq!(t.decode(&ids), s);
        // none of the produced ids should be a special id
        for id in ids {
            assert!(!t.specials.values().any(|v| *v == id), "id {id}");
        }
    }
}
