//! OmniVoice TTS inference engine on candle.
//!
//! Two-stage pipeline mirroring the vllm-omni PyTorch implementation:
//! 1. `generator` — Qwen3 bidirectional backbone + 32-step iterative
//!    unmasking producing 8-codebook audio tokens (25 fps)
//! 2. `dac`       — HiggsAudioV2 RVQ + DAC decoder -> 24 kHz audio
//!
//! Weights are read directly from the HuggingFace-format OmniVoice repo:
//! `model.safetensors` (generator) and `audio_tokenizer/model.safetensors`
//! (decoder); the tokenizer loads from the unified `tokenizer.json`.

pub mod audio;
pub mod config;
pub mod dac;
pub mod duration;
pub mod encoder;
pub mod engine;
pub mod generator;
pub mod hubert;
pub mod qwen3;
pub mod resample;
pub mod tokenizer;
pub mod wavio;

/// Bias-free linear layer: `x @ w.T`. The fork's matmul requires equal ranks
/// (no `(b, m, k) @ (k, n)` broadcast): flatten all leading dims into the
/// rows and restore afterwards.
pub(crate) fn linear_nobias(x: &candle_core::Tensor, w: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor> {
    let rank = x.rank();
    let k = x.dim(rank - 1)?;
    let rows: usize = x.dims()[..rank - 1].iter().product();
    let y = x.reshape((rows, k))?.matmul(&w.transpose(0, 1)?)?;
    let mut shape = x.dims().to_vec();
    shape[rank - 1] = w.dim(0)?;
    Ok(y.reshape(shape)?)
}

/// Linear layer with bias: `x @ w.T + b` (same flatten-rows trick).
pub(crate) fn linear(
    x: &candle_core::Tensor,
    w: &candle_core::Tensor,
    b: &candle_core::Tensor,
) -> anyhow::Result<candle_core::Tensor> {
    let y = linear_nobias(x, w)?;
    let c = b.elem_count();
    Ok(y.broadcast_add(&b.reshape((c,))?)?)
}

/// Split `text` into chunks whose estimated duration is at most `max_frames`
/// each, preferring sentence boundaries; the duration estimator's char
/// weights make CJK and Latin text pack to comparable audio lengths.
pub fn split_chunks(text: &str, max_frames: f64) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    if max_frames <= 0.0 {
        return vec![text.to_string()];
    }
    let max_weight = max_frames * duration::total_weight(config::DURATION_REF_TEXT)
        / config::DURATION_REF_FRAMES;

    // Sentence boundaries.
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(
            ch,
            '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?' | ';' | ':'
        ) {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }

    // Hard-split overlong sentences on char boundaries.
    let mut pieces: Vec<(String, f64)> = Vec::new();
    for s in sentences {
        let ws = duration::total_weight(&s);
        if ws <= max_weight {
            pieces.push((s, ws));
            continue;
        }
        let mut cur = String::new();
        let mut w = 0.0;
        for ch in s.chars() {
            let cw = duration::char_weight(ch);
            if !cur.is_empty() && w + cw > max_weight {
                pieces.push((std::mem::take(&mut cur), w));
                w = 0.0;
            }
            cur.push(ch);
            w += cw;
        }
        if !cur.is_empty() {
            pieces.push((cur, w));
        }
    }

    // Greedy packing.
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut w = 0.0;
    for (piece, pw) in pieces {
        if !cur.is_empty() && w + pw > max_weight {
            chunks.push(std::mem::take(&mut cur));
            w = 0.0;
        }
        cur.push_str(&piece);
        w += pw;
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Backend device selection: metal/cuda feature builds pick the accelerator
/// unless `cpu` is forced; otherwise fall back to CPU.
pub fn default_device(force_cpu: bool) -> anyhow::Result<candle_core::Device> {
    use candle_core::Device;
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        let dev = Device::new_metal(0)?;
        return Ok(dev);
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

/// Resolve a device selector (cpu|cuda|metal|auto) into a candle `Device`.
pub fn select_device(sel: &str) -> anyhow::Result<candle_core::Device> {
    use candle_core::Device;
    match sel {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::new_cuda(0).map_err(|e| anyhow::anyhow!("CUDA device requested but unavailable: {e}")),
        "metal" => Device::new_metal(0).map_err(|e| anyhow::anyhow!("Metal device requested but unavailable: {e}")),
        "auto" => {
            #[cfg(feature = "metal")]
            {
                if let Ok(d) = Device::new_metal(0) {
                    return Ok(d);
                }
            }
            #[cfg(feature = "cuda")]
            {
                if let Ok(d) = Device::new_cuda(0) {
                    return Ok(d);
                }
            }
            Ok(Device::Cpu)
        }
        other => anyhow::bail!("unknown device {other:?} (cpu|cuda|metal|auto)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 15 s budget (375 frames ≈ 211 weight units ≈ 70 CJK chars).
    const BUDGET: f64 = 375.0;

    fn frames(text: &str) -> f64 {
        duration::estimate_duration_frames(text)
    }

    #[test]
    fn short_text_stays_in_one_chunk() {
        assert_eq!(split_chunks("你好，世界。", BUDGET), vec!["你好，世界。"]);
        assert!(split_chunks("", BUDGET).is_empty());
    }

    #[test]
    fn long_text_splits_on_sentence_boundaries() {
        let text = "这是一句用来测试自动切分的完整句子。".repeat(12);
        let chunks = split_chunks(&text, BUDGET);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.ends_with('。'), "chunk does not end on a boundary: {c}");
            assert!(frames(c) <= BUDGET + 1e-6, "chunk over budget: {c}");
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn overlong_sentence_is_hard_split() {
        let text = "词".repeat(300);
        let chunks = split_chunks(&text, BUDGET);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| frames(c) <= BUDGET + 1e-6));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn latin_packs_more_chars_than_cjk() {
        let text = "This is a sentence for testing. ".repeat(20);
        let chunks = split_chunks(&text, BUDGET);
        let cjk = split_chunks(&"词".repeat(300), BUDGET);
        assert!(chunks.len() > 1);
        assert!(chunks[0].chars().count() > cjk[0].chars().count() * 2);
        assert_eq!(chunks.concat(), text);
    }
}
