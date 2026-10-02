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
use crate::text;
use crate::tokenizer::Tokenizer;
use crate::audio;

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

/// Long-form generation + output post-processing options (ports the
/// pipeline-level fields of the Python `OmniVoiceGenerationConfig`).
#[derive(Clone)]
pub struct SpeakOptions {
    /// Speaking-speed factor; estimates and chunk sizing divide by it.
    pub speed: f64,
    /// Fixed output duration in seconds; when set (> 0) it overrides `speed`
    /// and disables long-form chunking (the reference `duration` takes
    /// priority over `speed`).
    pub duration: Option<f64>,
    /// Prepend the `<|denoise|>` tag to the style prompt.
    pub denoise: bool,
    /// Target chunk duration in seconds; 0 disables chunking entirely.
    pub audio_chunk_duration: f64,
    /// Estimated duration (seconds) above which chunking is activated.
    pub audio_chunk_threshold: f64,
    /// Remove long silences from the output before volume/padding.
    pub postprocess_output: bool,
    /// Silence padding per edge (seconds).
    pub pad_duration: f64,
    /// Edge fade length (seconds).
    pub fade_duration: f64,
    /// RMS of the 24 kHz mono reference waveform (pre-normalization) used
    /// for output volume matching; `None` falls back to peak normalization.
    pub ref_rms: Option<f64>,
}

impl Default for SpeakOptions {
    fn default() -> Self {
        Self {
            speed: 1.0,
            duration: None,
            denoise: true,
            audio_chunk_duration: cfg::AUDIO_CHUNK_DURATION,
            audio_chunk_threshold: cfg::AUDIO_CHUNK_THRESHOLD,
            postprocess_output: true,
            pad_duration: cfg::PAD_DURATION,
            fade_duration: cfg::FADE_DURATION,
            ref_rms: None,
        }
    }
}

/// `create_voice_clone_prompt`'s `ref_rms`: RMS of the reference waveform
/// resampled to 24 kHz mono, computed before any normalization.
pub fn ref_rms(wav: &[f32], sample_rate: u32) -> f64 {
    let x24 = crate::resample::resample(wav, sample_rate, cfg::SAMPLE_RATE);
    let n = x24.len().max(1);
    (x24.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / n as f64).sqrt()
}

/// Long-form chunk plan (port of `GenerationTask.get_indices` +
/// `_generate_chunked`'s text splitting): returns the overall target length
/// and, when it exceeds the chunk threshold, the punctuation-split chunk
/// texts sized so each holds ~`audio_chunk_duration` seconds.
pub fn plan_chunks(
    text: &str,
    ref_text: Option<&str>,
    ref_frames: Option<usize>,
    opts: &SpeakOptions,
) -> (usize, Vec<String>) {
    // A fixed duration overrides the estimate and keeps the text whole.
    if let Some(d) = opts.duration.filter(|d| *d > 0.0) {
        let target_len = (d * cfg::FRAME_RATE as f64).max(1.0) as usize;
        return (target_len, vec![text.to_string()]);
    }
    let target_len = duration::estimate_target_tokens(text, ref_text, ref_frames, opts.speed);
    let threshold = (opts.audio_chunk_threshold * cfg::FRAME_RATE as f64) as usize;
    if opts.audio_chunk_duration <= 0.0 || target_len <= threshold {
        return (target_len, vec![text.to_string()]);
    }
    let n_chars = text.chars().count().max(1);
    let avg_tokens_per_char = target_len as f64 / n_chars as f64;
    let chunk_len = ((opts.audio_chunk_duration * cfg::FRAME_RATE as f64 / avg_tokens_per_char)
        .max(1.0)) as usize;
    (target_len, text::chunk_text_punctuation(text, chunk_len, Some(3)))
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
        // go away (mirrors the upstream `unload`).
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
    #[allow(clippy::too_many_arguments)]
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
        let opts = match ref_audio {
            Some((wav, sr)) => {
                let rms = ref_rms(wav, sr);
                SpeakOptions { ref_rms: Some(rms), ..Default::default() }
            }
            None => SpeakOptions::default(),
        };
        self.tts_with(
            text,
            lang,
            instruct,
            seed,
            params,
            ref_codes.as_deref(),
            ref_text,
            &opts,
            &CancelFlag::new(),
            &NullSink,
        )
    }

    /// As `tts`, but takes pre-encoded reference codes (the app caches them
    /// per voice) and supports cancellation + progress reporting.
    ///
    /// Long-form texts (estimated duration above
    /// `opts.audio_chunk_threshold`) are split by
    /// [`crate::text::chunk_text_punctuation`] into `opts.audio_chunk_duration`
    /// -sized chunks. Without reference audio, chunk 0's generated tokens
    /// become the reference for the remaining chunks (voice consistency, as
    /// in the Python `_generate_chunked`). The stitched output is then
    /// post-processed like `_post_process_audio`: silence removal, volume
    /// matching and edge fade/pad.
    #[allow(clippy::too_many_arguments)]
    pub fn tts_with(
        &self,
        text: &str,
        lang: Option<&str>,
        instruct: Option<&str>,
        seed: Option<u64>,
        params: &GenParams,
        ref_codes: Option<&[Vec<u32>]>,
        ref_text: Option<&str>,
        opts: &SpeakOptions,
        cancel: &CancelFlag,
        sink: &dyn ProgressSink,
    ) -> Result<Vec<f32>> {
        let lang = lang.unwrap_or("None");
        let instruct = instruct.unwrap_or("None");

        // `create_voice_clone_prompt` normalizes the reference transcript
        // with end punctuation; the target text is used as-is.
        let ref_text_norm = ref_text.map(text::add_punctuation);

        // Long-form chunking: keep every generation pass inside the trained
        // duration regime; the RoPE capacity guard then only catches
        // pathological references.
        let (total_frames, chunk_texts) =
            plan_chunks(text, ref_text_norm.as_deref(), ref_codes.map(|c| c[0].len()), opts);
        if chunk_texts.len() > 1 {
            sink.progress(Progress::Chunks { total: chunk_texts.len() });
            sink.log(&format!(
                "[omnivoice] ~{:.1}s estimated: text split into {} chunks (target {:.0}s each)",
                total_frames as f32 / cfg::FRAME_RATE as f32,
                chunk_texts.len(),
                opts.audio_chunk_duration,
            ));
        }

        let denoise_tag = if opts.denoise { "<|denoise|>" } else { "" };
        let style_text =
            format!("{denoise_tag}<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>");
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Without a voice-clone reference, chunk 0's generated tokens (and
        // its text) become the reference for chunks 1.. (`_generate_chunked`).
        let mut first_tokens: Option<Vec<Vec<u32>>> = None;

        let mut waves: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
        let mut frames_done = 0usize;
        for (ci, chunk) in chunk_texts.iter().enumerate() {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            let (chunk_ref_codes, chunk_ref_text) = match (ref_codes, first_tokens.as_ref()) {
                (Some(codes), _) => (Some(codes), ref_text_norm.as_deref()),
                (None, Some(tokens)) if ci > 0 => {
                    (Some(tokens.as_slice()), Some(chunk_texts[0].as_str()))
                }
                _ => (None, None),
            };
            let full_text = combine_text(chunk, chunk_ref_text);
            anyhow::ensure!(!full_text.is_empty(), "empty text prompt");

            // Per-chunk duration estimation on the raw chunk text
            // (`_run_batch` re-estimates every chunk individually); a fixed
            // duration pins the (single) chunk instead.
            let target_len = match opts.duration.filter(|d| *d > 0.0) {
                Some(d) => (d * cfg::FRAME_RATE as f64).max(1.0) as usize,
                None => duration::estimate_target_tokens(
                    chunk,
                    chunk_ref_text,
                    chunk_ref_codes.map(|c| c[0].len()),
                    opts.speed,
                ),
            };

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
                chunk_ref_codes,
                target_len,
                params,
                &mut rng,
                cancel,
                sink,
                ci + 1,
                chunk_texts.len(),
            )?;
            if ci == 0 && ref_codes.is_none() && chunk_texts.len() > 1 {
                first_tokens = Some(tokens.clone());
            }
            sink.progress(Progress::Decoding {
                chunk: ci + 1,
                total: chunk_texts.len(),
                frames_total: target_len,
            });
            waves.push(self.dac.decode(&tokens)?);
            frames_done += target_len;
        }

        let mut wave = audio::cross_fade_chunks(&waves);
        if opts.postprocess_output {
            wave = audio::remove_silence(
                &wave,
                cfg::SAMPLE_RATE,
                500,
                100,
                100,
            );
        }
        audio::normalize_volume(&mut wave, opts.ref_rms);
        let wave = audio::fade_and_pad(&wave, opts.pad_duration, opts.fade_duration, cfg::SAMPLE_RATE);
        let seconds = wave.len() as f32 / cfg::SAMPLE_RATE as f32;
        sink.progress(Progress::Done { frames: frames_done, seconds });
        Ok(wave)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_chunks_below_threshold_is_single_chunk() {
        let opts = SpeakOptions::default();
        let (len, chunks) = plan_chunks("你好，世界。", None, None, &opts);
        assert_eq!(chunks, vec!["你好，世界。"]);
        assert!(len <= (opts.audio_chunk_threshold * cfg::FRAME_RATE as f64) as usize);
    }

    #[test]
    fn plan_chunks_splits_long_text() {
        let opts = SpeakOptions::default();
        let text = "这是一句用来测试自动切分的完整句子。".repeat(30);
        let (len, chunks) = plan_chunks(&text, None, None, &opts);
        assert!(len > (opts.audio_chunk_threshold * cfg::FRAME_RATE as f64) as usize);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
        // Every chunk stays within the target duration budget.
        for c in &chunks {
            let est = duration::estimate_target_tokens(c, None, None, opts.speed);
            assert!(
                est as f64 <= opts.audio_chunk_duration * cfg::FRAME_RATE as f64 * 1.5,
                "chunk wildly over budget ({est} frames): {c}"
            );
        }
    }

    #[test]
    fn plan_chunks_zero_duration_disables_chunking() {
        let opts = SpeakOptions { audio_chunk_duration: 0.0, ..Default::default() };
        let text = "这是一句用来测试自动切分的完整句子。".repeat(30);
        let (_, chunks) = plan_chunks(&text, None, None, &opts);
        assert_eq!(chunks, vec![text.clone()]);
    }

    #[test]
    fn plan_chunks_fixed_duration_overrides_and_disables_chunking() {
        let text = "这是一句用来测试自动切分的完整句子。".repeat(30);
        let opts = SpeakOptions { duration: Some(5.0), ..Default::default() };
        let (len, chunks) = plan_chunks(&text, None, None, &opts);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
        assert_eq!(len, (5.0 * cfg::FRAME_RATE as f64) as usize);
        // 0 is treated as unset: chunking proceeds on the estimate.
        let opts = SpeakOptions { duration: Some(0.0), ..Default::default() };
        let (len2, chunks2) = plan_chunks(&text, None, None, &opts);
        assert!(chunks2.len() > 1);
        assert_ne!(len2, len);
    }

    #[test]
    fn plan_chunks_ref_calibration_scales_char_budget() {
        // A slow reference (many frames per char) shrinks the char budget.
        let opts = SpeakOptions::default();
        let text = "Hello there, this is a longer sentence. ".repeat(10);
        let (_, plain) = plan_chunks(&text, None, None, &opts);
        let (_, slow) = plan_chunks(
            &text,
            Some("Nice to meet you."),
            Some(200),
            &opts,
        );
        assert!(slow.len() >= plain.len());
    }
}
