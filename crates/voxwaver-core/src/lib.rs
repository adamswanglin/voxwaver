//! voxwaver-core: fishaudio/s1-mini TTS inference engine (dual-AR LM +
//! modded-DAC codec) on candle. Library facade for the CLI and the Tauri app.
pub mod config;
pub mod dac;
pub mod dual_ar;
pub mod engine;
pub mod prof;
pub mod prompt;
pub mod sampling;
pub mod tokenizer;
pub mod wavio;

pub use anyhow::Result;
use anyhow::{bail, Context};
use candle_core::{Device, DType, Tensor};

/// Repetition-penalty window: the last REP_WIN_SIZE drawn tokens, zero-padded
/// on the right while fewer exist (upstream reads `previous_tokens[:, :16]`
/// out of a zero-initialized buffer). `None`-equivalent is expressed by the
/// caller not passing a window at all (the prefill sample).
pub fn rep_window(hist: &[u32]) -> Option<Vec<u32>> {
    let n = hist.len();
    let mut w = vec![0u32; sampling::REP_WIN_SIZE];
    let take = n.min(sampling::REP_WIN_SIZE);
    let src: &[u32] = if n <= sampling::REP_WIN_SIZE {
        &hist[..take]
    } else {
        &hist[n - take..]
    };
    w[..take].copy_from_slice(src);
    Some(w)
}

/// `rep_window` uploaded to the device as a `[REP_WIN_SIZE]` u32 tensor,
/// for the on-device sampling chain (a 64-byte H2D copy per row per frame).
pub fn rep_window_t(hist: &[u32], dev: &Device) -> Result<Tensor> {
    let w = rep_window(hist).unwrap_or_else(|| vec![0u32; sampling::REP_WIN_SIZE]);
    Ok(Tensor::from_vec(w, sampling::REP_WIN_SIZE, dev)?)
}

/// Split text into chunks of at most `max_bytes` UTF-8 bytes, preferring
/// sentence boundaries (mirrors upstream `group_turns_into_batches`'s
/// ~300-byte batching in `generate_long`).
pub fn split_chunks(text: &str, max_bytes: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?' | ';' | ':') {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    // hard-split overlong sentences on char boundaries
    let mut pieces: Vec<String> = Vec::new();
    for s in sentences {
        if s.len() <= max_bytes {
            pieces.push(s);
            continue;
        }
        let mut start = 0;
        while start < s.len() {
            let mut end = (start + max_bytes).min(s.len());
            while end > start && !s.is_char_boundary(end) {
                end -= 1;
            }
            pieces.push(s[start..end].to_string());
            start = end;
        }
    }
    // greedy packing
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for piece in pieces {
        if !cur.is_empty() && cur.len() + piece.len() > max_bytes {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(&piece);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Resolve a device selector (cpu|cuda|metal|auto) into a candle `Device`.
pub fn select_device(sel: &str) -> Result<Device> {
    match sel {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::new_cuda(0).context("CUDA device requested but unavailable"),
        "metal" => Device::new_metal(0).context("Metal device requested but unavailable"),
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
        other => bail!("unknown device {other:?} (cpu|cuda|metal|auto)"),
    }
}

/// Resolve a dtype selector (auto|bf16|f16|f32) against the target device.
pub fn select_dtype(sel: &str, dev: &Device) -> Result<DType> {
    match sel {
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        "auto" => Ok(match dev {
            // f32 is the fast path on Metal and CPU
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        }),
        other => bail!("unknown dtype {other:?} (auto|bf16|f16|f32)"),
    }
}
