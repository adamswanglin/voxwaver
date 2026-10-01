//! The modded-DAC codec shipped as s1-mini `codec.pth`: weight loading with
//! weight-norm folding, audio -> codes (for voice cloning) and codes -> audio.
pub mod layers;
pub mod prof;
pub mod rvq;
pub mod transformer;

use crate::config::CodecConfig;
use anyhow::{bail, ensure, Context, Result};
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use layers::*;
use rvq::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use transformer::*;

/// Left-overlap context (in codebook frames) for chunked decoding: the codec
/// is fully causal and the post-module attention window is 128 frames, but the
/// conv stack's receptive field extends a few frames beyond a window-sized
/// context (verified empirically: with 128 the first ~127 frames of each
/// chunk differ slightly from a full decode). 128 + 64 covers it.
pub const DECODE_CONTEXT: usize = 192;

const ROPE_BASE: f64 = 10000.0;

struct Loader {
    tensors: HashMap<String, Tensor>,
    prefix: String,
    device: Device,
}

impl Loader {
    fn open(path: &Path, device: &Device) -> Result<Self> {
        let pth = PthTensors::new(path, None)
            .with_context(|| format!("open {}", path.display()))?;
        let mut tensors = HashMap::new();
        for (name, _info) in pth.tensor_infos() {
            if let Some(t) = pth.get(name)? {
                tensors.insert(name.clone(), t.to_dtype(DType::F32)?);
            }
        }
        ensure!(!tensors.is_empty(), "no tensors in {}", path.display());
        let prefix = ["", "codec.", "generator.", "state_dict.", "model."]
            .into_iter()
            .find(|p| tensors.contains_key(&format!("{p}encoder.block.0.conv.bias")))
            .with_context(|| {
                let mut keys: Vec<_> = tensors.keys().take(8).collect();
                keys.sort();
                format!(
                    "no encoder.block.0.conv.* in {} (sample keys: {keys:?})",
                    path.display()
                )
            })?
            .to_string();
        Ok(Self { tensors, prefix, device: device.clone() })
    }

    fn key(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(&self.key(name))
    }

    fn get(&self, name: &str) -> Result<Tensor> {
        let k = self.key(name);
        self.tensors
            .get(&k)
            .cloned()
            .with_context(|| format!("missing tensor {k}"))
    }

    fn dev(&self, name: &str) -> Result<Tensor> {
        Ok(self.get(name)?.to_device(&self.device)?)
    }

    /// Weight-norm folded weight for the module at `base`: `weight_g`/`weight_v`
    /// where `v` is stored as `weight_orig` (old-style `weight_norm`) or
    /// `parametrizations.weight.original` (new-style parametrization).
    /// Norm over all dims except 0, dim=0 for both conv types.
    fn conv_weight(&self, base: &str) -> Result<Tensor> {
        let plain = self.key(&format!("{base}.weight"));
        if let Some(w) = self.tensors.get(&plain) {
            return Ok(w.clone());
        }
        let (gk, vk) = if self
            .tensors
            .contains_key(&self.key(&format!("{base}.weight_v")))
        {
            // descript-style weight_norm saves weight_g / weight_v directly
            (
                self.key(&format!("{base}.weight_g")),
                self.key(&format!("{base}.weight_v")),
            )
        } else if self.tensors.contains_key(&self.key(&format!("{base}.weight_orig"))) {
            (
                self.key(&format!("{base}.weight_g")),
                self.key(&format!("{base}.weight_orig")),
            )
        } else if self
            .tensors
            .contains_key(&self.key(&format!("{base}.parametrizations.weight.original")))
        {
            (
                self.key(&format!("{base}.parametrizations.weight.0.weight_g")),
                self.key(&format!("{base}.parametrizations.weight.original")),
            )
        } else if self
            .tensors
            .contains_key(&self.key(&format!("{base}.parametrizations.weight.original1")))
        {
            // torch >= 2.1 weight_norm parametrization: original0 = weight_g,
            // original1 = weight_v
            (
                self.key(&format!("{base}.parametrizations.weight.original0")),
                self.key(&format!("{base}.parametrizations.weight.original1")),
            )
        } else {
            bail!("no weight found for {base} (tried .weight, .weight_orig, parametrizations)");
        };
        let (g, v) = (
            self.tensors.get(&gk).cloned().with_context(|| format!("missing {gk}"))?,
            self.tensors.get(&vk).cloned().with_context(|| format!("missing {vk}"))?,
        );
        // `weight_g` may be shape [out_c, 1, 1] (torch >= 2.1 parametrization)
        // or full weight shape; broadcast handles both.
        let out_c = v.dim(0)?;
        let norm = v
            .powf(2.0)?
            .reshape((out_c, v.elem_count() / out_c))?
            .sum(candle_core::D::Minus1)?
            .sqrt()?
            .reshape((out_c, 1, 1))?;
        Ok(g.broadcast_mul(&v)?.broadcast_div(&norm)?)
    }

    /// `module_base` points at the conv module itself: CausalConvNet /
    /// CausalTransConvNet keys carry a trailing `.conv`, descript `WNConv1d`
    /// (the VQ projections) do not. `groups == 0` means depthwise (derive
    /// groups from the folded weight shape).
    fn conv(
        &self,
        module_base: &str,
        k: usize,
        stride: usize,
        dil: usize,
        groups: usize,
    ) -> Result<ConvW> {
        let w = self.conv_weight(module_base)?;
        let out_c = w.dim(0)?;
        let groups = if groups == 0 {
            ensure!(w.dim(1)? == 1, "depthwise weight must have in/groups == 1");
            out_c
        } else {
            groups
        };
        Ok(ConvW::new(
            w.to_device(&self.device)?,
            self.dev(&format!("{module_base}.bias"))?,
            k,
            stride,
            dil,
            groups,
        ))
    }

    fn tconv(&self, module_base: &str, k: usize, stride: usize) -> Result<TransConvW> {
        Ok(TransConvW::new(
            self.conv_weight(module_base)?.to_device(&self.device)?,
            self.dev(&format!("{module_base}.bias"))?,
            k,
            stride,
        ))
    }

    fn snake(&self, base: &str) -> Result<Snake> {
        Snake::new(self.dev(&format!("{base}.alpha"))?)
    }

    /// WindowLimitedTransformer at `base`; n_head/head_dim/intermediate/eps are
    /// identical (16/64/3072/1e-5) for the encoder and quantizer modules.
    fn wlt(&self, base: &str, cfg: &CodecConfig, n_layer: usize, window: usize) -> Result<Wlt> {
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("{base}.layers.{i}");
            blocks.push(WltBlock {
                wqkv: self.dev(&format!("{p}.attention.wqkv.weight"))?,
                wo: self.dev(&format!("{p}.attention.wo.weight"))?,
                attn_norm: self.dev(&format!("{p}.attention_norm.weight"))?,
                ffn_norm: self.dev(&format!("{p}.ffn_norm.weight"))?,
                w1: self.dev(&format!("{p}.feed_forward.w1.weight"))?,
                w3: self.dev(&format!("{p}.feed_forward.w3.weight"))?,
                w2: self.dev(&format!("{p}.feed_forward.w2.weight"))?,
                attn_ls: self.dev(&format!("{p}.attention_layer_scale.gamma"))?,
                ffn_ls: self.dev(&format!("{p}.ffn_layer_scale.gamma"))?,
            });
        }
        Ok(Wlt::new(
            blocks,
            self.dev(&format!("{base}.norm.weight"))?,
            window,
            cfg.q_t_n_head,
            cfg.q_t_head_dim,
            ROPE_BASE,
            cfg.q_t_norm_eps as f32,
            self.device.clone(),
        ))
    }

    /// descript `VectorQuantize` at `base` (e.g. `quantizer.quantizers.3`).
    fn vq(&self, base: &str) -> Result<Vq> {
        Ok(Vq::new(
            // WNConv1d k=1: non-causal, but k=1 stride=1 is padding-free anyway
            self.conv(&format!("{base}.in_proj"), 1, 1, 1, 1)?,
            self.conv(&format!("{base}.out_proj"), 1, 1, 1, 1)?,
            self.dev(&format!("{base}.codebook.weight"))?,
        ))
    }

    fn convnext(&self, base: &str) -> Result<ConvNeXt> {
        Ok(ConvNeXt::new(
            self.conv(&format!("{base}.dwconv.conv"), 7, 1, 1, 0)?, // groups derived (depthwise)
            self.dev(&format!("{base}.norm.weight"))?,
            self.dev(&format!("{base}.norm.bias"))?,
            self.dev(&format!("{base}.pwconv1.weight"))?,
            self.dev(&format!("{base}.pwconv1.bias"))?,
            self.dev(&format!("{base}.pwconv2.weight"))?,
            self.dev(&format!("{base}.pwconv2.bias"))?,
            self.dev(&format!("{base}.gamma"))?,
        ))
    }

    /// ResidualUnit at `base` (`base.block.0..3` inside).
    fn residual_unit(&self, base: &str, dil: usize) -> Result<ResidualUnit> {
        Ok(ResidualUnit::new(
            self.snake(&format!("{base}.block.0"))?,
            self.conv(&format!("{base}.block.1.conv"), 7, 1, dil, 1)?,
            self.snake(&format!("{base}.block.2"))?,
            self.conv(&format!("{base}.block.3.conv"), 1, 1, 1, 1)?,
        ))
    }
}

/// The encode-only half of the codec: waveform -> quantizer-ready latent.
/// Deliberately kept out of [`Dac`] so a decode-only session never
/// materializes these ~1.1 GB of weights; [`Dac::encode`] loads it on
/// demand and drops it again (vllm-omni prunes encoder / pre_module /
/// downsample on its decode worker along the same lines).
struct EncodeHalf {
    encoder: Encoder,
    down: Vec<(ConvW, ConvNeXt)>,
    pre_module: Wlt,
}

impl EncodeHalf {
    /// x [1, 1, samples] -> quantizer input [1, 1024, T].
    fn forward_to_latent(&self, x: &Tensor) -> Result<Tensor> {
        let mut z = self.encoder.forward(x)?;
        for (conv, cn) in &self.down {
            z = cn.forward(&conv.forward(&z)?)?;
        }
        self.pre_module.forward(&z)
    }
}

pub struct Dac {
    cfg: CodecConfig,
    /// Source checkpoint; [`Dac::encode`] re-opens it to materialize the
    /// encode-only half on demand.
    path: PathBuf,
    quantizer: DownRvq,
    decoder: Decoder,
    device: Device,
}

impl Dac {
    /// Load the decode-essential half of the codec: the quantizer decode
    /// side (codebooks, post_module, upsample) and the decoder (~0.75 GB).
    /// The encode-only weights (~1.1 GB: conv encoder, quantizer downsample
    /// + pre_module) are materialized on demand by [`Dac::encode`] instead —
    /// a decode-only session never pays for them.
    pub fn load(path: &Path, cfg: &CodecConfig, device: &Device) -> Result<Self> {
        let ld = Loader::open(path, device)?;

        // ---- quantizer (decode side) ----
        let semantic = Rvq {
            quantizers: vec![ld.vq("quantizer.semantic_quantizer.quantizers.0")?],
        };
        // residual RVQ prefix differs between checkpoints: newer fish-speech
        // uses `quantizer.quantizers.N`, s1-mini's codec.pth uses
        // `quantizer.quantizer.quantizers.N`
        let rvq_base = if ld.contains("quantizer.quantizers.0.in_proj") {
            "quantizer.quantizers"
        } else {
            "quantizer.quantizer.quantizers"
        };
        let residual = Rvq {
            quantizers: (0..cfg.n_residual_codebooks)
                .map(|i| ld.vq(&format!("{rvq_base}.{i}")))
                .collect::<Result<Vec<_>>>()?,
        };
        let post_module = ld.wlt("quantizer.post_module", cfg, cfg.q_t_n_layer, cfg.q_t_window)?;
        let mut up = Vec::new();
        // PyTorch builds `upsample` over `reversed(enumerate(downsample_factor))`:
        // checkpoint key `upsample.0` holds the LAST factor's weights and is
        // applied first, so load keys in application order.
        for (j, &f) in cfg.downsample_factor.iter().rev().enumerate() {
            up.push((
                ld.tconv(&format!("quantizer.upsample.{j}.0.conv"), f, f)?,
                ld.convnext(&format!("quantizer.upsample.{j}.1"))?,
            ));
        }
        let quantizer = DownRvq::new(
            semantic,
            residual,
            post_module,
            up,
            cfg.semantic_codebook_size,
            cfg.codebook_size,
        );

        // ---- decoder: conv_in, 4 DecoderBlocks, snake + conv_out + tanh ----
        let dconv_in = ld.conv("decoder.model.0.conv", 7, 1, 1, 1)?;
        let mut dblocks = Vec::with_capacity(cfg.decoder_rates.len());
        for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
            let base = format!("decoder.model.{}", i + 1);
            let mut rus = Vec::with_capacity(3);
            for (j, dil) in [1usize, 3, 9].iter().enumerate() {
                rus.push(ld.residual_unit(&format!("{base}.block.{}", j + 2), *dil)?);
            }
            dblocks.push(DecoderBlock::new(
                ld.snake(&format!("{base}.block.0"))?,
                ld.tconv(&format!("{base}.block.1.conv"), 2 * stride, stride)?,
                rus,
            ));
        }
        let decoder = Decoder::new(
            dconv_in,
            dblocks,
            ld.snake("decoder.model.5")?,
            ld.conv("decoder.model.6.conv", 7, 1, 1, 1)?,
        );

        Ok(Self {
            cfg: *cfg,
            path: path.to_path_buf(),
            quantizer,
            decoder,
            device: device.clone(),
        })
    }

    /// Materialize the encode-only half (conv encoder + quantizer downsample
    /// + pre_module, ~1.1 GB) from the checkpoint. [`Dac::encode`] drops the
    /// result as soon as it is done with it, so encode weights are resident
    /// only for the duration of an encode call.
    fn load_encode_half(&self) -> Result<EncodeHalf> {
        let ld = Loader::open(&self.path, &self.device)?;
        let cfg = &self.cfg;
        let e = cfg.encoder_rates;

        // ---- encoder: conv_in, 4 EncoderBlocks, snake + conv_out ----
        let conv_in = ld.conv("encoder.block.0.conv", 7, 1, 1, 1)?;
        let mut blocks = Vec::with_capacity(e.len());
        for (i, &stride) in e.iter().enumerate() {
            let mut rus = Vec::with_capacity(3);
            for (j, dil) in [1usize, 3, 9].iter().enumerate() {
                rus.push(ld.residual_unit(&format!("encoder.block.{}.block.{j}", i + 1), *dil)?);
            }
            let snake = ld.snake(&format!("encoder.block.{}.block.3", i + 1))?;
            let conv =
                ld.conv(&format!("encoder.block.{}.block.4.conv", i + 1), 2 * stride, stride, 1, 1)?;
            let transformer = if cfg.enc_transformer_layers[i] > 0 {
                Some(ld.wlt(
                    &format!("encoder.block.{}.block.5", i + 1),
                    cfg,
                    cfg.enc_transformer_layers[i],
                    cfg.enc_t_window,
                )?)
            } else {
                None
            };
            blocks.push(EncoderBlock::new(rus, snake, conv, transformer));
        }
        let encoder = Encoder::new(
            conv_in,
            blocks,
            ld.snake("encoder.block.5")?,
            ld.conv("encoder.block.6.conv", 3, 1, 1, 1)?,
        );

        // ---- quantizer encode side: downsample + pre_module ----
        let mut down = Vec::new();
        for (i, &f) in cfg.downsample_factor.iter().enumerate() {
            down.push((
                ld.conv(&format!("quantizer.downsample.{i}.0.conv"), f, f, 1, 1)?,
                ld.convnext(&format!("quantizer.downsample.{i}.1"))?,
            ));
        }
        let pre_module = ld.wlt("quantizer.pre_module", cfg, cfg.q_t_n_layer, cfg.q_t_window)?;
        Ok(EncodeHalf { encoder, down, pre_module })
    }

    /// Audio samples (mono, 44.1 kHz) -> 10 codebook rows.
    ///
    /// Encode-only weights are loaded for this call and released before it
    /// returns; each call re-pays the load (a reference is normally encoded
    /// once per session, and decode-only sessions pay nothing).
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<u32>>> {
        let fl = self.cfg.frame_length();
        let right_pad = samples.len().div_ceil(fl) * fl - samples.len();
        let mut buf = samples.to_vec();
        buf.extend(std::iter::repeat_n(0.0, right_pad));
        let total = buf.len();
        let x = Tensor::from_vec(buf, (1, 1, total), &self.device)?;
        let enc = self.load_encode_half()?;
        let z = enc.forward_to_latent(&x)?;
        let codes = self.quantizer.encode_latent(&z)?;
        drop(enc);
        ensure!(
            codes.len() == self.cfg.n_residual_codebooks + 1,
            "expected {} codebook rows, got {}",
            self.cfg.n_residual_codebooks + 1,
            codes.len()
        );
        Ok(codes)
    }

    /// 10 codebook rows -> audio samples (length 2048 * frames).
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let z = prof::phase(prof::Q_TOTAL, &self.device, || {
            self.quantizer.decode(codes, &self.device)
        })?;
        // Drop the quantizer's pooled Metal buffers before the decoder's own
        // (much larger) intermediates pile on top of them.
        if self.device.is_metal() {
            self.device.synchronize()?;
        }
        let audio = prof::phase(prof::D_TOTAL, &self.device, || self.decoder.forward(&z))?;
        let audio = prof::phase(prof::D_READBACK, &self.device, || {
            let a = audio.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            Ok(a.to_vec1::<f32>()?)
        })?;
        prof::print_and_reset();
        Ok(audio)
    }

    /// Chunked variant of `decode_codes`: processes at most `chunk_frames`
    /// frames at a time, carrying `DECODE_CONTEXT` frames of left context so
    /// the result matches a full decode. `chunk_frames == 0` decodes at once.
    /// When `cancel` is set, the chunk loop aborts with "cancelled".
    pub fn decode_codes_chunked(
        &self,
        codes: &[Vec<u32>],
        chunk_frames: usize,
        cancel: Option<&crate::engine::CancelFlag>,
    ) -> Result<Vec<f32>> {
        let total = codes.first().map(|r| r.len()).unwrap_or(0);
        if total == 0 {
            return Ok(Vec::new());
        }
        if chunk_frames == 0 || total <= chunk_frames + DECODE_CONTEXT {
            return self.decode_codes(codes);
        }
        let fl = self.cfg.frame_length();
        let mut out: Vec<f32> = Vec::new();
        let mut start = 0usize;
        while start < total {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    bail!("cancelled");
                }
            }
            let end = (start + chunk_frames).min(total);
            let ctx_start = start.saturating_sub(DECODE_CONTEXT);
            let chunk: Vec<Vec<u32>> = codes
                .iter()
                .map(|row| row[ctx_start..end].to_vec())
                .collect();
            let audio = self.decode_codes(&chunk)?;
            let skip = (start - ctx_start) * fl;
            out.extend_from_slice(&audio[skip..]);
            start = end;
        }
        Ok(out)
    }
}
