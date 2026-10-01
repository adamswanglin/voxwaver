//! Stage-1 decoder: the HiggsAudioV2 audio tokenizer's decode path
//! (`modeling_higgs_audio_v2_tokenizer.py` + `modeling_dac.py`).
//!
//! codes [8][T] -> RVQ (8 residual codebooks, each `embed` lookup +
//! `project_out`, summed) -> fc2 (1024 -> 256) -> acoustic DAC decoder ->
//! 24 kHz waveform. Two HiggsAudioV2-specific adjustments over the HF DAC:
//! every decoder `ConvTranspose1d` gets `output_padding = stride % 2`, and the
//! final `tanh` is removed.
//!
//! The transposed convolutions are implemented as unpadded
//! `conv_transpose1d` (which hits candle's fast col2im path) plus an
//! explicit crop: PyTorch's `padding=ceil(s/2), output_padding=s%2` with
//! `kernel=2s` reduces to `y[i] = full[i + ceil(s/2)]` for `i` in
//! `[0, L*s)` — `output_padding` extends the kept window rather than
//! appending zeros, so the single slice `[ceil(s/2), ceil(s/2)+L*s)`
//! reproduces the reference for odd and even strides alike.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_core::safetensors::MmapedSafetensors;

use crate::config as cfg;

/// Snake1d: `x + sin^2(alpha * x) / (alpha + 1e-9)`, alpha of shape (1, C, 1).
pub struct Snake {
    alpha: Tensor, // [1, C, 1]
    /// `1 / (alpha + 1e-9)`, folded at load time: the hot path then uses a
    /// broadcast multiply instead of a divide, matching the Python reference
    /// (which also forms the reciprocal first).
    inv: Tensor, // [1, C, 1]
}

impl Snake {
    pub fn new(alpha: Tensor) -> Result<Self> {
        let alpha = alpha.reshape((1, alpha.elem_count(), 1))?;
        let inv = alpha.affine(1.0, 1e-9)?.recip()?;
        Ok(Self { alpha, inv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        // Single sin pass; `sqr()` is a unary op, which keeps candle's Metal
        // buffer cache from stalling on a two-input `s.mul(&s)`.
        let s2 = ax.sin()?.sqr()?;
        Ok(x.add(&s2.broadcast_mul(&self.inv)?)?)
    }
}

/// Symmetric zero-padded Conv1d with bias (the HF DAC's plain convs).
pub struct ConvPad {
    w: Tensor, // [out, in, k]
    b: Tensor, // [out]
    k: usize,
    dil: usize,
}

impl ConvPad {
    pub fn new(w: Tensor, b: Tensor, k: usize, dil: usize) -> Self {
        Self { w, b, k, dil }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = (self.k - 1) * self.dil / 2;
        let x = if p > 0 {
            x.pad_with_zeros(2, p, p)?
        } else {
            x.clone()
        };
        let y = x.conv1d(&self.w, 0, 1, self.dil, 1)?;
        let b = self.b.reshape((1, self.b.elem_count(), 1))?;
        Ok(y.broadcast_add(&b)?)
    }
}

/// ConvTranspose1d(k=2*stride, stride, padding=ceil(stride/2),
/// output_padding=stride%2) — the HiggsAudioV2-adjusted DAC upsample.
pub struct TConv {
    w: Tensor, // [in, out, k]
    b: Tensor, // [out]
    stride: usize,
}

impl TConv {
    pub fn new(w: Tensor, b: Tensor, stride: usize) -> Self {
        Self { w, b, stride }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = self.stride;
        let l = x.dim(2)?;
        let y = x.conv_transpose1d(&self.w, 0, 0, s, 1, 1)?;
        // Unpadded length: (L-1)*s + k = (L+1)*s with k = 2s.
        debug_assert_eq!(y.dim(2)?, (l + 1) * s, "conv_transpose1d length mismatch");
        let b = self.b.reshape((1, self.b.elem_count(), 1))?;
        let y = y.broadcast_add(&b)?;
        // PyTorch semantics: y[i] = full[i + ceil(s/2)] for i in [0, L*s).
        // output_padding=s%2 extends the kept window instead of appending
        // zeros, so one unified crop matches the reference exactly:
        // L*s = (L-1)*s - 2*ceil(s/2) + 2*s + s%2 holds for odd and even s.
        let p = s.div_ceil(2);
        let y = y.narrow(2, p, l * s)?;
        Ok(y)
    }
}

/// ResidualUnit: Snake -> dilated Conv(k=7) -> Snake -> Conv(k=1), plain
/// residual (the HF padding-crop branch is dead code for these kernels:
/// both convs preserve the length).
pub struct ResUnit {
    s1: Snake,
    c1: ConvPad,
    s2: Snake,
    c2: ConvPad,
}

impl ResUnit {
    pub fn new(s1: Snake, c1: ConvPad, s2: Snake, c2: ConvPad) -> Self {
        Self { s1, c1, s2, c2 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.c1.forward(&self.s1.forward(x)?)?;
        let y = self.c2.forward(&self.s2.forward(&y)?)?;
        Ok(x.add(&y)?)
    }
}

/// DecoderBlock: Snake -> upsample transposed conv -> 3 ResidualUnits
/// (dilations 1/3/9).
pub struct DecBlock {
    snake: Snake,
    tconv: TConv,
    rus: Vec<ResUnit>,
}

impl DecBlock {
    pub fn new(snake: Snake, tconv: TConv, rus: Vec<ResUnit>) -> Self {
        Self { snake, tconv, rus }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.tconv.forward(&self.snake.forward(x)?)?;
        for ru in &self.rus {
            x = ru.forward(&x)?;
        }
        Ok(x)
    }
}

/// Acoustic decoder: [1, 256, T] -> [1, 1, ~960*T], no final tanh.
pub struct DacDecoder {
    conv1: ConvPad,
    blocks: Vec<DecBlock>,
    snake_out: Snake,
    conv2: ConvPad,
}

impl DacDecoder {
    pub fn new(conv1: ConvPad, blocks: Vec<DecBlock>, snake_out: Snake, conv2: ConvPad) -> Self {
        Self {
            conv1,
            blocks,
            snake_out,
            conv2,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1.forward(x)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
            // Recycle pooled Metal buffers between blocks: each block's
            // im2col intermediates are dead by the next block, but the pool
            // pins one buffer per size class until a sync. candle's
            // synchronize() drops unused pooled buffers as a side effect.
            if x.device().is_metal() {
                x.device().synchronize()?;
            }
        }
        self.conv2.forward(&self.snake_out.forward(&x)?)
    }
}

struct Quantizer {
    embed: Tensor, // [codebook_size, codebook_dim]
    out_w: Tensor, // [rvq_hidden, codebook_dim] (project_out.weight)
    out_b: Tensor, // [rvq_hidden]
}

/// The full stage-1 decoder (RVQ + fc2 + acoustic DAC decoder).
pub struct Dac {
    quantizers: Vec<Quantizer>,
    fc2_w: Tensor, // [256, 1024]
    fc2_b: Tensor, // [256]
    decoder: DacDecoder,
    device: Device,
}

impl Dac {
    pub fn load(model_dir: &std::path::Path, device: &Device) -> Result<Self> {
        let path = model_dir.join("audio_tokenizer").join("model.safetensors");
        // SAFETY: the weight file is a read-only input of this process.
        let st = unsafe { MmapedSafetensors::new(&path) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let vs = |name: &str| st.load(name, device);

        let mut quantizers = Vec::with_capacity(cfg::RVQ_QUANTIZERS);
        for i in 0..cfg::RVQ_QUANTIZERS {
            let q = format!("quantizer.quantizers.{i}");
            quantizers.push(Quantizer {
                embed: vs(&format!("{q}.codebook.embed"))?,
                out_w: vs(&format!("{q}.project_out.weight"))?,
                out_b: vs(&format!("{q}.project_out.bias"))?,
            });
        }
        let fc2_w = vs("fc2.weight")?;
        let fc2_b = vs("fc2.bias")?;

        let conv1 = ConvPad::new(
            vs("acoustic_decoder.conv1.weight")?,
            vs("acoustic_decoder.conv1.bias")?,
            7,
            1,
        );
        // Channel ladder 1024 -> 512 -> 256 -> 128 -> 64 -> 32.
        const C_IN: [usize; 5] = [1024, 512, 256, 128, 64];
        const C_OUT: [usize; 5] = [512, 256, 128, 64, 32];
        let mut blocks = Vec::with_capacity(cfg::UPSAMPLE_RATES.len());
        for i in 0..cfg::UPSAMPLE_RATES.len() {
            let s = cfg::UPSAMPLE_RATES[i];
            let b = format!("acoustic_decoder.block.{i}");
            let snake = Snake::new(vs(&format!("{b}.snake1.alpha"))?)?;
            let tconv = TConv::new(
                vs(&format!("{b}.conv_t1.weight"))?,
                vs(&format!("{b}.conv_t1.bias"))?,
                s,
            );
            let mut rus = Vec::with_capacity(3);
            for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
                let ru = format!("{b}.res_unit{}", j + 1);
                let s1 = Snake::new(vs(&format!("{ru}.snake1.alpha"))?)?;
                let c1 = ConvPad::new(
                    vs(&format!("{ru}.conv1.weight"))?,
                    vs(&format!("{ru}.conv1.bias"))?,
                    7,
                    dil,
                );
                let s2 = Snake::new(vs(&format!("{ru}.snake2.alpha"))?)?;
                let c2 = ConvPad::new(
                    vs(&format!("{ru}.conv2.weight"))?,
                    vs(&format!("{ru}.conv2.bias"))?,
                    1,
                    1,
                );
                rus.push(ResUnit::new(s1, c1, s2, c2));
            }
            blocks.push(DecBlock::new(snake, tconv, rus));
            let _ = (C_IN[i], C_OUT[i]); // documented above; shapes come from the checkpoint
        }
        let snake_out = Snake::new(vs("acoustic_decoder.snake1.alpha")?)?;
        let conv2 = ConvPad::new(
            vs("acoustic_decoder.conv2.weight")?,
            vs("acoustic_decoder.conv2.bias")?,
            7,
            1,
        );
        Ok(Self {
            quantizers,
            fc2_w,
            fc2_b,
            decoder: DacDecoder::new(conv1, blocks, snake_out, conv2),
            device: device.clone(),
        })
    }

    /// Decode `[8][T]` audio codes into a mono 24 kHz waveform.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let t = codes.first().map(|c| c.len()).unwrap_or(0);
        anyhow::ensure!(!codes.is_empty() && t > 0, "empty audio codes");
        // RVQ: sum of per-codebook (embed lookup + project_out).
        let mut quantized: Option<Tensor> = None; // [T, 1024]
        for (i, q) in self.quantizers.iter().enumerate() {
            let idx = Tensor::from_vec(codes[i].clone(), (t,), &self.device)?;
            let e = q.embed.index_select(&idx, 0)?; // [T, 64]
            let out = e.matmul(&q.out_w.t()?)?; // [T, 1024]
            let out = out.broadcast_add(&q.out_b.reshape((1, cfg::RVQ_HIDDEN))?)?;
            quantized = Some(match quantized {
                None => out,
                Some(acc) => acc.add(&out)?,
            });
        }
        let quantized = quantized.expect("non-empty quantizers");
        // fc2: [T, 1024] -> [T, 256], then to [1, 256, T].
        let x = quantized
            .matmul(&self.fc2_w.t()?)?
            .broadcast_add(&self.fc2_b.reshape((1, cfg::DAC_IN))?)?;
        let x = x.t()?.unsqueeze(0)?; // [1, 256, T]
        // Acoustic decoder -> [1, 1, L].
        let wave = self.decoder.forward(&x)?;
        Ok(wave.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?)
    }
}
