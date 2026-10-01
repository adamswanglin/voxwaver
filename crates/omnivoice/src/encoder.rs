//! Voice-clone reference encoder: the HiggsAudioV2 audio tokenizer's encode
//! path (`HiggsAudioV2TokenizerModel.encode`).
//!
//! Reference wav -> 24 kHz -> two parallel branches:
//! - semantic: 24 kHz -> 16 kHz resample, zero-pad 160, HuBERT (13 hidden
//!   states averaged, `::2` frame downsample), `SemanticEncoder` conv stack
//! - acoustic: `DacEncoder` (5 strided conv blocks with Snake activations)
//! The 256 + 768 channel outputs are concatenated and projected by `fc`
//! (1024 -> 1024), then quantized by 8 residual VQ codebooks (bandwidth 2.0)
//! into the 8 x T clone-prompt tokens.
//!
//! The acoustic branch is padded by `hop_length // 2` (= 480) on both sides
//! exactly when its unpadded frame count would differ from the semantic
//! branch's, mirroring the `_get_conv1d_output_lengths` check in `encode`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::safetensors::MmapedSafetensors;

use crate::config as cfg;
use crate::dac::{ConvPad, ResUnit, Snake};
use crate::hubert::Hubert;

/// `HiggsAudioV2TokenizerResidualUnit`: ELU -> conv(k3, pad=dil, no bias)
/// -> ELU -> conv(k1, no bias) -> residual.
struct SemResUnit {
    c1_w: Tensor, // [C, C, 3]
    c2_w: Tensor, // [C, C, 1]
}

impl SemResUnit {
    fn forward(&self, x: &Tensor, dil: usize) -> Result<Tensor> {
        let y = x.elu(1.0)?.conv1d(&self.c1_w, dil, 1, dil, 1)?;
        let y = y.elu(1.0)?.conv1d(&self.c2_w, 0, 1, 1, 1)?;
        Ok(x.add(&y)?)
    }
}

/// `HiggsAudioV2TokenizerSemanticEncoderBlock` (stride 1 special case:
/// kernel 3, pad 1).
struct SemBlock {
    units: Vec<SemResUnit>,
    c_w: Tensor, // [C, C, 3]
    c_b: Tensor, // [C]
}

impl SemBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (u, &dil) in self.units.iter().zip(cfg::SEM_DILATIONS.iter()) {
            x = u.forward(&x, dil)?;
        }
        let y = x.conv1d(&self.c_w, 1, 1, 1, 1)?;
        let c = self.c_b.elem_count();
        Ok(y.broadcast_add(&self.c_b.reshape((1, c, 1))?)?)
    }
}

/// `SemanticEncoder`: bias-free conv + 2 blocks (all strides 1, so the frame
/// count is unchanged).
struct SemanticEncoder {
    conv_w: Tensor, // [768, 768, 3], no bias
    blocks: Vec<SemBlock>,
}

impl SemanticEncoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.conv1d(&self.conv_w, 1, 1, 1, 1)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        Ok(x)
    }
}

/// `DacEncoderBlock`: res(dil 1) -> res(dil 3) -> res(dil 9) -> Snake
/// -> strided conv (kernel 2s, padding ceil(s/2)).
struct EncBlock {
    ru1: ResUnit,
    ru2: ResUnit,
    ru3: ResUnit,
    snake: Snake,
    c_w: Tensor, // [2C, C, 2s]
    c_b: Tensor, // [2C]
    stride: usize,
}

impl EncBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ru1.forward(x)?;
        let x = self.ru2.forward(&x)?;
        let x = self.snake.forward(&self.ru3.forward(&x)?)?;
        let p = self.stride.div_ceil(2);
        let x = x.pad_with_zeros(2, p, p)?;
        let y = x.conv1d(&self.c_w, 0, self.stride, 1, 1)?;
        let c = self.c_b.elem_count();
        Ok(y.broadcast_add(&self.c_b.reshape((1, c, 1))?)?)
    }
}

/// `DacEncoder`: [1, 1, L] -> [1, 256, T] (hop 960).
struct AcousticEncoder {
    conv1: ConvPad, // 1 -> 64, k7 p3
    blocks: Vec<EncBlock>,
    snake_out: Snake, // 2048
    conv2: ConvPad,   // 2048 -> 256, k3 p1
}

impl AcousticEncoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1.forward(x)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        self.conv2.forward(&self.snake_out.forward(&x)?)
    }
}

/// One RVQ stage: `project_in` (1024 -> 64), codebook nearest-neighbour,
/// `project_out` (64 -> 1024).
struct EncQuantizer {
    in_w: Tensor,  // [64, 1024]
    in_b: Tensor,  // [64]
    embed: Tensor, // [1024, 64]
    out_w: Tensor, // [1024, 64]
    out_b: Tensor, // [1024]
}

/// Codebook indices for `proj` [T, 64]: argmin of
/// `||x||^2 - 2 x e + ||e||^2` (the negation of the reference's `dist.max`).
/// Computed as tensors, then scanned on the CPU so ties resolve to the
/// first minimum exactly like `torch.max`.
fn nearest(proj: &Tensor, embed: &Tensor) -> Result<Vec<u32>> {
    let dot = proj.matmul(&embed.t()?.contiguous()?)?; // [T, 1024]
    let xn = proj.sqr()?.sum_keepdim(1)?; // [T, 1]
    let en = embed.sqr()?.sum_keepdim(1)?.t()?; // [1, 1024]
    let score = xn
        .broadcast_sub(&dot.affine(2.0, 0.0)?)?
        .broadcast_add(&en)?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    let rows: Vec<Vec<f32>> = score.to_vec2()?;
    let indices: Vec<u32> = rows
        .iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_v = f32::INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v < best_v {
                    best_v = v;
                    best = j;
                }
            }
            best as u32
        })
        .collect();
    Ok(indices)
}

/// The full encode path (HuBERT + semantic/acoustic conv stacks + RVQ).
pub struct RefEncoder {
    hubert: Hubert,
    semantic: SemanticEncoder,
    acoustic: AcousticEncoder,
    fc_w: Tensor, // [1024, 1024]
    fc_b: Tensor, // [1024]
    quants: Vec<EncQuantizer>,
    device: Device,
}

impl RefEncoder {
    pub fn load(model_dir: &std::path::Path, device: &Device) -> Result<Self> {
        let path = model_dir.join("audio_tokenizer").join("model.safetensors");
        // SAFETY: the weight file is a read-only input of this process.
        let st = unsafe { MmapedSafetensors::new(&path) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let vs = |name: &str| st.load(name, device);

        let hubert = Hubert::load(&|n: &str| vs(&format!("semantic_model.{n}")), device)?;

        let mut sem_blocks = Vec::with_capacity(cfg::SEM_STRIDES.len());
        for i in 0..cfg::SEM_STRIDES.len() {
            let b = format!("encoder_semantic.conv_blocks.{i}");
            let mut units = Vec::with_capacity(cfg::SEM_DILATIONS.len());
            for j in 0..cfg::SEM_DILATIONS.len() {
                let u = format!("{b}.res_units.{j}");
                units.push(SemResUnit {
                    c1_w: vs(&format!("{u}.conv1.weight"))?,
                    c2_w: vs(&format!("{u}.conv2.weight"))?,
                });
            }
            sem_blocks.push(SemBlock {
                units,
                c_w: vs(&format!("{b}.conv.weight"))?,
                c_b: vs(&format!("{b}.conv.bias"))?,
            });
        }
        let semantic = SemanticEncoder {
            conv_w: vs("encoder_semantic.conv.weight")?,
            blocks: sem_blocks,
        };

        let conv1 = ConvPad::new(
            vs("acoustic_encoder.conv1.weight")?,
            vs("acoustic_encoder.conv1.bias")?,
            7,
            1,
        );
        let mut ac_blocks = Vec::with_capacity(cfg::ENC_STRIDES.len());
        for (i, &stride) in cfg::ENC_STRIDES.iter().enumerate() {
            let b = format!("acoustic_encoder.block.{i}");
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
            let [ru1, ru2, ru3]: [ResUnit; 3] = rus
                .try_into()
                .map_err(|_| anyhow::anyhow!("expected 3 residual units"))?;
            ac_blocks.push(EncBlock {
                ru1,
                ru2,
                ru3,
                snake: Snake::new(vs(&format!("{b}.snake1.alpha"))?)?,
                c_w: vs(&format!("{b}.conv1.weight"))?,
                c_b: vs(&format!("{b}.conv1.bias"))?,
                stride,
            });
        }
        let acoustic = AcousticEncoder {
            conv1,
            blocks: ac_blocks,
            snake_out: Snake::new(vs("acoustic_encoder.snake1.alpha")?)?,
            conv2: ConvPad::new(
                vs("acoustic_encoder.conv2.weight")?,
                vs("acoustic_encoder.conv2.bias")?,
                3,
                1,
            ),
        };

        let mut quants = Vec::with_capacity(cfg::RVQ_QUANTIZERS);
        for i in 0..cfg::RVQ_QUANTIZERS {
            let q = format!("quantizer.quantizers.{i}");
            quants.push(EncQuantizer {
                in_w: vs(&format!("{q}.project_in.weight"))?,
                in_b: vs(&format!("{q}.project_in.bias"))?,
                embed: vs(&format!("{q}.codebook.embed"))?,
                out_w: vs(&format!("{q}.project_out.weight"))?,
                out_b: vs(&format!("{q}.project_out.bias"))?,
            });
        }

        Ok(Self {
            hubert,
            semantic,
            acoustic,
            fc_w: vs("fc.weight")?,
            fc_b: vs("fc.bias")?,
            quants,
            device: device.clone(),
        })
    }

    /// Full reference-audio encode: any sample rate -> `[8][T]` tokens.
    pub fn encode(&self, wav: &[f32], sample_rate: u32) -> Result<Vec<Vec<u32>>> {
        let x24 = if sample_rate == cfg::SAMPLE_RATE {
            wav.to_vec()
        } else {
            crate::resample::resample(wav, sample_rate, cfg::SAMPLE_RATE)
        };
        let e_sem = self.semantic_branch(&x24)?; // [1, 768, T_sem]
        let t_sem = e_sem.dim(2)?;
        let use_pad = acoustic_frames(x24.len()) != t_sem;
        let e_ac = self.acoustic_branch(&x24, use_pad)?; // [1, 256, T_sem]
        let emb = self.fc_quantize(&e_ac, &e_sem)?; // [1, 1024, T_sem]
        self.rvq_encode(&emb)
    }

    /// Semantic features of a 24 kHz waveform -> `[1, 768, T_sem]`.
    pub fn semantic_branch(&self, x24: &[f32]) -> Result<Tensor> {
        let x16 = crate::resample::resample(x24, cfg::SAMPLE_RATE, cfg::SEM_SAMPLE_RATE);
        self.semantic_features(&x16)
    }

    /// Unpadded 16 kHz waveform -> semantic-encoder output `[1, 768, T_sem]`
    /// (zero-pad 160 -> HuBERT -> mean over 13 states -> `::2` -> conv stack).
    pub fn semantic_features(&self, x16: &[f32]) -> Result<Tensor> {
        let mut padded = vec![0f32; x16.len() + 2 * cfg::SEM_PAD];
        padded[cfg::SEM_PAD..cfg::SEM_PAD + x16.len()].copy_from_slice(x16);
        let hidden = self.hubert.forward(&padded)?; // [13, T, 768]
        let sem = hidden.mean(0)?; // [T, 768]
        let t = sem.dim(0)?;
        let n = t.div_ceil(cfg::SEM_DOWNSAMPLE);
        let idx: Vec<u32> = (0..n).map(|i| (i * cfg::SEM_DOWNSAMPLE) as u32).collect();
        let idx = Tensor::from_vec(idx, (n,), &self.device)?;
        let sem = sem.index_select(&idx, 0)?; // [T_sem, 768]
        // [1, 768, T_sem] and through the conv stack (frame count unchanged).
        let sem = sem.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        self.semantic.forward(&sem)
    }

    /// HuBERT hidden states `[13, T, 768]` of an already-padded 16 kHz wav.
    pub fn hubert_hidden(&self, x16_padded: &[f32]) -> Result<Tensor> {
        self.hubert.forward(x16_padded)
    }

    /// The semantic model (per-layer parity helpers).
    pub fn hubert(&self) -> &Hubert {
        &self.hubert
    }

    /// Acoustic branch of a 24 kHz waveform -> `[1, 256, T]`.
    pub fn acoustic_branch(&self, x24: &[f32], use_pad: bool) -> Result<Tensor> {
        let x = if use_pad {
            let mut padded = vec![0f32; x24.len() + 2 * cfg::ENC_PAD];
            padded[cfg::ENC_PAD..cfg::ENC_PAD + x24.len()].copy_from_slice(x24);
            padded
        } else {
            x24.to_vec()
        };
        let x = Tensor::from_vec(x.clone(), (1, 1, x.len()), &self.device)?;
        self.acoustic.forward(&x)
    }

    /// `fc` over the concatenated branches -> `[1, 1024, T]`.
    pub fn fc_quantize(&self, e_ac: &Tensor, e_sem: &Tensor) -> Result<Tensor> {
        let emb = Tensor::cat(&[e_ac, e_sem], 1)?; // [1, 1024, T]
        let emb = crate::linear(
            &emb.transpose(1, 2)?.contiguous()?,
            &self.fc_w,
            &self.fc_b,
        )?;
        Ok(emb.transpose(1, 2)?.contiguous()?)
    }

    /// Residual-VQ encode of `[1, 1024, T]` -> `[8][T]` codebook indices.
    pub fn rvq_encode(&self, emb: &Tensor) -> Result<Vec<Vec<u32>>> {
        let mut residual = emb.squeeze(0)?.transpose(0, 1)?.contiguous()?; // [T, 1024]
        let mut codes = Vec::with_capacity(self.quants.len());
        for q in &self.quants {
            let proj = crate::linear(&residual, &q.in_w, &q.in_b)?; // [T, 64]
            let t = proj.dim(0)?;
            let idx = nearest(&proj, &q.embed)?;
            let idx_t = Tensor::from_vec(idx.clone(), (t,), &self.device)?;
            let e = q.embed.index_select(&idx_t, 0)?; // [T, 64]
            let quantized = crate::linear(&e, &q.out_w, &q.out_b)?; // [T, 1024]
            residual = residual.sub(&quantized)?;
            codes.push(idx);
        }
        Ok(codes)
    }
}

/// `_get_conv1d_output_lengths` over the acoustic encoder: one k7/p3 conv,
/// 5 strided block convs (k=2s, p=ceil(s/2)) and the final k3/p1 conv. The
/// residual units preserve the length (odd kernel + symmetric padding).
pub fn acoustic_frames(l: usize) -> usize {
    let out = |l: usize, k: usize, s: usize, p: usize| (l + 2 * p - (k - 1) - 1) / s + 1;
    let mut l = out(l, 7, 1, 3);
    for &s in cfg::ENC_STRIDES.iter() {
        l = out(l, 2 * s, s, s.div_ceil(2));
    }
    out(l, 3, 1, 1)
}
