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
pub mod text;
pub mod tokenizer;
pub mod wavio;

#[cfg(feature = "cuda")]
mod cuda_probe;

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

/// Create the CUDA device at ordinal 0, or an error explaining why not.
///
/// cudarc (built with `dynamic-loading`) dlopens the CUDA libraries on first
/// use and panics when one is missing, so a bare `Device::new_cuda` would
/// abort the process on machines without a CUDA stack. The dlopen pre-probe
/// in `cuda_probe` catches that case; `catch_unwind` covers the gap between
/// "library loads" and "device initializes" (a missing symbol, no driver,
/// no device). Call this instead of `Device::new_cuda` in cuda builds.
#[cfg(feature = "cuda")]
pub fn try_new_cuda() -> anyhow::Result<candle_core::Device> {
    if let Some(missing) = cuda_probe::cuda_stack_missing() {
        anyhow::bail!("CUDA libraries not loadable: {missing}");
    }
    let created =
        std::panic::catch_unwind(|| candle_core::Device::new_cuda(0)).map_err(|payload| {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            anyhow::anyhow!("CUDA init failed: {msg}")
        })?;
    created.map_err(|e| anyhow::anyhow!("CUDA device unavailable: {e}"))
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
        return try_new_cuda();
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

/// Resolve a device selector (cpu|cuda|metal|auto) into a candle `Device`.
pub fn select_device(sel: &str) -> anyhow::Result<candle_core::Device> {
    use candle_core::Device;
    match sel {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => try_new_cuda(),
        #[cfg(not(feature = "cuda"))]
        "cuda" => anyhow::bail!("CUDA support not compiled in"),
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
                if let Ok(d) = try_new_cuda() {
                    return Ok(d);
                }
            }
            Ok(Device::Cpu)
        }
        other => anyhow::bail!("unknown device {other:?} (cpu|cuda|metal|auto)"),
    }
}
