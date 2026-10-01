//! Engine: prompt construction (text cleanup + duration estimation +
//! tokenization) and the two-stage generate/decode pipeline
//! (port of `pipeline_omnivoice.py`'s request preparation).

use anyhow::{bail, Result};
use candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tts_common::{CancelFlag, NullSink, Progress, ProgressSink};

use crate::config as cfg;
use crate::dac::Dac;
use crate::duration;
use crate::encoder::RefEncoder;
use crate::generator::{GenParams, Generator};
use crate::tokenizer::Tokenizer;
use crate::{audio, split_chunks};

fn is_cjk(c: char) -> bool {
    (0x4e00..=0x9fff).contains(&(c as u32))
}

/// `_combine_text`: with a non-empty `ref_text` (clone mode) the two are
/// joined as `ref_text.strip() + " " + text.strip()` first; then strip, drop
/// CR/LF, normalize fullwidth parens, collapse runs of spaces/tabs, and
/// remove whitespace adjacent to CJK characters.
pub fn combine_text(text: &str, ref_text: Option<&str>) -> String {
    let joined = match ref_text {
        Some(rt) if !rt.is_empty() => format!("{} {}", rt.trim(), text.trim()),
        _ => text.trim().to_string(),
    };
    let s: String = joined.chars().filter(|&c| c != '\r' && c != '\n').collect();
    // Replace Chinese parentheses with English ones.
    let s = s.replace('\u{ff08}', "(").replace('\u{ff09}', ")");
    // Collapse consecutive spaces / tabs into a single space.
    let mut collapsed = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            if !in_ws {
                collapsed.push(' ');
                in_ws = true;
            }
        } else {
            collapsed.push(c);
            in_ws = false;
        }
    }
    // Remove whitespace around Chinese characters (Python's lookaround
    // regex over `\s+`; after the collapse only single spaces remain, but
    // other Unicode whitespace still matches).
    let chars: Vec<char> = collapsed.chars().collect();
    let mut out = String::with_capacity(collapsed.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            let prev_cjk = out.chars().last().is_some_and(is_cjk);
            let next_cjk = chars[i + 1..]
                .iter()
                .find(|&&x| !x.is_whitespace())
                .copied()
                .is_some_and(is_cjk);
            if !(prev_cjk || next_cjk) {
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `_NONVERBAL_PATTERN`: tags tokenized standalone so their ids stay
/// consistent regardless of the surrounding language.
const NONVERBAL_TAGS: &[&str] = &[
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
];

/// `_tokenize_with_nonverbal_tags`: split the text on `[tag]` occurrences,
/// tokenizing each segment and each tag independently.
fn tokenize_with_nonverbal_tags(text: &str, tok: &Tokenizer) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    let mut last_end = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' && text.is_char_boundary(i) {
            let rest = &text[i..];
            let tag = NONVERBAL_TAGS
                .iter()
                .find(|t| rest.starts_with(&format!("[{t}]")));
            if let Some(t) = tag {
                let end = i + 2 + t.len(); // '[' + tag + ']'
                if last_end < i {
                    ids.extend(tok.encode(&text[last_end..i])?);
                }
                ids.extend(tok.encode(&text[i..end])?);
                last_end = end;
                i = end;
                continue;
            }
        }
        i += 1;
    }
    if last_end < text.len() {
        ids.extend(tok.encode(&text[last_end..])?);
    }
    Ok(ids)
}

pub struct Engine {
    tokenizer: Tokenizer,
    generator: Generator,
    dac: Dac,
    ref_encoder: RefEncoder,
    /// Kept for drop-time Metal buffer pool release.
    device: Device,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Hand idle pooled GPU buffers back to the OS before the weights
        // go away (mirrors voxwaver-core's `unload`).
        let _ = self.device.clear_metal_pool();
    }
}

impl Engine {
    pub fn load(model_dir: &std::path::Path, force_cpu: bool) -> Result<Self> {
        let device = crate::default_device(force_cpu)?;
        Self::load_on_with(model_dir, &device, &NullSink)
    }

    /// Load onto a caller-resolved device (e.g. picked via the app's device
    /// selector) with stage progress reporting.
    pub fn load_on_with(
        model_dir: &std::path::Path,
        device: &Device,
        sink: &dyn ProgressSink,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        sink.progress(Progress::LoadingLm);
        let tokenizer = Tokenizer::load(&model_dir.join("tokenizer.json"))?;
        let generator = Generator::load(model_dir, device)?;
        sink.progress(Progress::LoadingCodec);
        let dac = Dac::load(model_dir, device)?;
        let ref_encoder = RefEncoder::load(model_dir, device)?;
        sink.log(&format!(
            "model loaded on {:?} in {:.1}s",
            device,
            started.elapsed().as_secs_f32()
        ));
        Ok(Self {
            tokenizer,
            generator,
            dac,
            ref_encoder,
            device: device.clone(),
        })
    }

    /// Voice-clone reference encoding (any sample rate): 24 kHz resample ->
    /// HuBERT semantic + DAC acoustic -> RVQ -> `[8][t_ref]` tokens.
    pub fn encode_ref(&self, wav: &[f32], sample_rate: u32) -> Result<Vec<Vec<u32>>> {
        self.ref_encoder.encode(wav, sample_rate)
    }

    /// Text-to-speech: returns a mono 24 kHz waveform. When `ref_audio`
    /// (waveform + sample rate) is given, its encoded tokens are inserted into
    /// the conditional half (`[text | ref | target]`, voice cloning).
    pub fn tts(
        &self,
        text: &str,
        lang: Option<&str>,
        instruct: Option<&str>,
        seed: Option<u64>,
        params: &GenParams,
        ref_audio: Option<(&[f32], u32)>,
        ref_text: Option<&str>,
    ) -> Result<Vec<f32>> {
        let ref_codes = match ref_audio {
            Some((wav, sr)) => {
                let codes = self.encode_ref(wav, sr)?;
                eprintln!("[omnivoice] reference audio encoded to 8 x {} tokens", codes[0].len());
                Some(codes)
            }
            None => None,
        };
        self.tts_with(
            text,
            lang,
            instruct,
            seed,
            params,
            ref_codes.as_deref(),
            ref_text,
            &CancelFlag::new(),
            &NullSink,
        )
    }

    /// As `tts`, but takes pre-encoded reference codes (the app caches them
    /// per voice) and supports cancellation + progress reporting.
    ///
    /// Estimated durations above `cfg::CHUNK_THRESHOLD_FRAMES` are split into
    /// `cfg::CHUNK_FRAMES`-sized pieces on sentence boundaries, generated
    /// independently and stitched with a short cross-fade (mirrors the Python
    /// pipeline's long-form chunking and voxwaver-core's sentence batching).
    pub fn tts_with(
        &self,
        text: &str,
        lang: Option<&str>,
        instruct: Option<&str>,
        seed: Option<u64>,
        params: &GenParams,
        ref_codes: Option<&[Vec<u32>]>,
        ref_text: Option<&str>,
        cancel: &CancelFlag,
        sink: &dyn ProgressSink,
    ) -> Result<Vec<f32>> {
        let lang = lang.unwrap_or("None");
        let instruct = instruct.unwrap_or("None");

        // Long-form chunking: keep every generation pass inside the trained
        // duration regime; the RoPE capacity guard then only catches
        // pathological references.
        let est = duration::estimate_duration_frames(text);
        let total_frames = ((est.floor() as i64).max(1)) as usize;
        let chunk_texts: Vec<String> = if total_frames > cfg::CHUNK_THRESHOLD_FRAMES {
            let chunks = split_chunks(text, cfg::CHUNK_FRAMES as f64);
            sink.progress(Progress::Chunks { total: chunks.len() });
            sink.log(&format!(
                "[omnivoice] ~{:.1}s estimated: text split into {} chunks (target {:.0}s each)",
                total_frames as f32 / cfg::FRAME_RATE as f32,
                chunks.len(),
                cfg::CHUNK_FRAMES as f32 / cfg::FRAME_RATE as f32,
            ));
            chunks
        } else {
            vec![text.to_string()]
        };

        let style_text =
            format!("<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>");
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut waves: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        let mut frames_done = 0usize;
        for (ci, chunk) in chunk_texts.iter().enumerate() {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            let full_text = combine_text(chunk, ref_text);
            anyhow::ensure!(!full_text.is_empty(), "empty text prompt");

            // Duration estimation on the raw chunk text (pipeline semantics).
            let est = duration::estimate_duration_frames(chunk);
            let target_len = ((est.floor() as i64).max(1)) as usize;

            let wrapped = format!("<|text_start|>{full_text}<|text_end|>");
            let mut text_ids = self.tokenizer.encode(&style_text)?;
            text_ids.extend(tokenize_with_nonverbal_tags(&wrapped, &self.tokenizer)?);
            anyhow::ensure!(!text_ids.is_empty(), "text tokenized to nothing");

            sink.log(&format!(
                "[omnivoice] chunk {}/{}: {} text tokens, target {} frames (~{:.1}s)",
                ci + 1,
                chunk_texts.len(),
                text_ids.len(),
                target_len,
                target_len as f32 / cfg::FRAME_RATE as f32
            ));

            let tokens = self.generator.generate(
                &text_ids,
                ref_codes,
                target_len,
                params,
                &mut rng,
                cancel,
                sink,
                ci + 1,
                chunk_texts.len(),
            )?;
            sink.progress(Progress::Decoding {
                chunk: ci + 1,
                total: chunk_texts.len(),
                frames_total: target_len,
            });
            waves.push(self.dac.decode(&tokens)?);
            frames_done += target_len;
        }

        let wave = audio::cross_fade_chunks(&waves);
        let seconds = wave.len() as f32 / cfg::SAMPLE_RATE as f32;
        sink.progress(Progress::Done { frames: frames_done, seconds });
        Ok(wave)
    }
}
