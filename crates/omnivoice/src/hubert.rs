//! HuBERT semantic encoder (`modeling_hubert.py`) used by the
//! HiggsAudioV2 audio tokenizer's encode path.
//!
//! 7-layer valid conv feature extractor (group-norm + GELU on layer 0),
//! LayerNorm + Linear feature projection (512 -> 768), grouped positional
//! conv (weight-norm merged at load time) with a SamePad crop, and 12
//! bidirectional transformer layers (12 heads x 64 dim, scaling 0.125).
//!
//! `forward` returns the 13 hidden states (pre-layer-0 input + each layer's
//! output) stacked as `[13, T, 768]`; the tokenizer averages over dim 0.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;

/// Conv feature-extractor kernel sizes / strides (`conv_kernel`/`conv_stride`).
const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
const NUM_LAYERS: usize = 12;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const HIDDEN: usize = 768;
const INTERMEDIATE: usize = 3072;
const LN_EPS: f64 = 1e-5;

/// LayerNorm over the last dimension (biased variance).
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let c = x.dim(x.rank() - 1)?;
    let mean = x.mean_keepdim(x.rank() - 1)?;
    let d = x.broadcast_sub(&mean)?;
    let var = d.sqr()?.mean_keepdim(x.rank() - 1)?;
    let y = d.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    Ok(y.broadcast_mul(&w.reshape((c,))?)?
        .broadcast_add(&b.reshape((c,))?)?)
}

/// GroupNorm with `groups == channels`: per-channel normalization over time
/// (biased variance), followed by the affine transform.
fn group_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let c = x.dim(1)?;
    let mean = x.mean_keepdim(2)?;
    let d = x.broadcast_sub(&mean)?;
    let var = d.sqr()?.mean_keepdim(2)?;
    let y = d.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    Ok(y.broadcast_mul(&w.reshape((1, c, 1))?)?
        .broadcast_add(&b.reshape((1, c, 1))?)?)
}

struct Layer {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
    o_b: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    f1_w: Tensor,
    f1_b: Tensor,
    f2_w: Tensor,
    f2_b: Tensor,
    fln_w: Tensor,
    fln_b: Tensor,
}

impl Layer {
    fn load(vs: &dyn Fn(&str) -> candle_core::Result<Tensor>, i: usize) -> Result<Self> {
        let p = |n: &str| vs(&format!("encoder.layers.{i}.{n}"));
        Ok(Self {
            q_w: p("attention.q_proj.weight")?,
            q_b: p("attention.q_proj.bias")?,
            k_w: p("attention.k_proj.weight")?,
            k_b: p("attention.k_proj.bias")?,
            v_w: p("attention.v_proj.weight")?,
            v_b: p("attention.v_proj.bias")?,
            o_w: p("attention.out_proj.weight")?,
            o_b: p("attention.out_proj.bias")?,
            ln_w: p("layer_norm.weight")?,
            ln_b: p("layer_norm.bias")?,
            f1_w: p("feed_forward.intermediate_dense.weight")?,
            f1_b: p("feed_forward.intermediate_dense.bias")?,
            f2_w: p("feed_forward.output_dense.weight")?,
            f2_b: p("feed_forward.output_dense.bias")?,
            fln_w: p("final_layer_norm.weight")?,
            fln_b: p("final_layer_norm.bias")?,
        })
    }

    /// `x` [1, T, 768] -> [1, T, 768]: attention (no mask, full context)
    /// -> residual -> LayerNorm -> FFN -> residual -> LayerNorm.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        debug_assert_eq!(c, HIDDEN);
        let heads = |w: &Tensor, bias: &Tensor| -> Result<Tensor> {
            Ok(crate::linear(x, w, bias)?
                .reshape((b, t, HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = heads(&self.q_w, &self.q_b)?; // [1, 12, T, 64]
        let k = heads(&self.k_w, &self.k_b)?;
        let v = heads(&self.v_w, &self.v_b)?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?; // [1,12,T,T]
        let attn = softmax(&scores.affine(0.125, 0.0)?, candle_core::D::Minus1)?;
        let o = attn.matmul(&v)?; // [1, 12, T, 64]
        let o = o
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;
        let o = crate::linear(&o, &self.o_w, &self.o_b)?;
        let x = x.add(&o)?;
        let x = layer_norm(&x, &self.ln_w, &self.ln_b, LN_EPS)?;
        let ff = crate::linear(&x, &self.f1_w, &self.f1_b)?.gelu_erf()?;
        debug_assert_eq!(ff.dim(2)?, INTERMEDIATE);
        let ff = crate::linear(&ff, &self.f2_w, &self.f2_b)?;
        let x = x.add(&ff)?;
        layer_norm(&x, &self.fln_w, &self.fln_b, LN_EPS)
    }
}

pub struct Hubert {
    convs: Vec<Tensor>,
    gn_w: Tensor,
    gn_b: Tensor,
    fp_ln_w: Tensor,
    fp_ln_b: Tensor,
    fp_w: Tensor,
    fp_b: Tensor,
    pos_w: Tensor,
    pos_b: Tensor,
    enc_ln_w: Tensor,
    enc_ln_b: Tensor,
    layers: Vec<Layer>,
    device: Device,
}

impl Hubert {
    /// `vs` maps a suffix (e.g. `encoder.pos_conv_embed.conv.bias`) to a
    /// weight tensor; the caller supplies the `semantic_model.` prefix.
    pub fn load(vs: &dyn Fn(&str) -> candle_core::Result<Tensor>, device: &Device) -> Result<Self> {
        let mut convs = Vec::with_capacity(7);
        for i in 0..7 {
            convs.push(vs(&format!("feature_extractor.conv_layers.{i}.conv.weight"))?);
        }
        // Positional conv weight norm: weight = g * v / ||v|| over all dims
        // except dim 2 (`torch.nn.utils.parametrizations.weight_norm(dim=2)`).
        let g = vs("encoder.pos_conv_embed.conv.parametrizations.weight.original0")?;
        let v = vs("encoder.pos_conv_embed.conv.parametrizations.weight.original1")?;
        let norm = v
            .sqr()?
            .sum_keepdim(0)? // [768, 48, 128] -> [1, 48, 128]
            .sum_keepdim(1)? // -> [1, 1, 128]
            .sqrt()?;
        let pos_w = g.broadcast_mul(&v)?.broadcast_div(&norm)?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(Layer::load(vs, i)?);
        }
        Ok(Self {
            convs,
            gn_w: vs("feature_extractor.conv_layers.0.layer_norm.weight")?,
            gn_b: vs("feature_extractor.conv_layers.0.layer_norm.bias")?,
            fp_ln_w: vs("feature_projection.layer_norm.weight")?,
            fp_ln_b: vs("feature_projection.layer_norm.bias")?,
            fp_w: vs("feature_projection.projection.weight")?,
            fp_b: vs("feature_projection.projection.bias")?,
            pos_w,
            pos_b: vs("encoder.pos_conv_embed.conv.bias")?,
            enc_ln_w: vs("encoder.layer_norm.weight")?,
            enc_ln_b: vs("encoder.layer_norm.bias")?,
            layers,
            device: device.clone(),
        })
    }

    /// `wav` is the padded 16 kHz waveform. Returns hidden states
    /// `[13, T, 768]` (pre-layer-0 input + 12 layer outputs).
    pub fn forward(&self, wav: &[f32]) -> Result<Tensor> {
        let x = self.feature_extract(wav)?;
        let x = self.feature_project(&x)?;
        let pos = self.pos_conv(&x)?;
        let mut x = self.encoder_norm(&x.add(&pos)?)?;

        let mut states: Vec<Tensor> = Vec::with_capacity(NUM_LAYERS + 1);
        states.push(x.squeeze(0)?);
        for layer in &self.layers {
            x = layer.forward(&x)?;
            states.push(x.squeeze(0)?);
        }
        Ok(Tensor::stack(&states, 0)?) // [13, T, 768]
    }

    /// Feature extractor: 7 valid convs (GroupNorm + GELU on layer 0),
    /// transposed to `[1, T, 512]`.
    pub fn feature_extract(&self, wav: &[f32]) -> Result<Tensor> {
        let mut x = Tensor::from_vec(wav.to_vec(), (1, 1, wav.len()), &self.device)?;
        for (i, w) in self.convs.iter().enumerate() {
            let mut y = x.conv1d(w, 0, CONV_STRIDE[i], 1, 1)?;
            if i == 0 {
                y = group_norm(&y, &self.gn_w, &self.gn_b, LN_EPS)?;
            }
            x = y.gelu_erf()?;
        }
        Ok(x.transpose(1, 2)?.contiguous()?)
    }

    /// LayerNorm + Linear projection: `[1, T, 512]` -> `[1, T, 768]`.
    pub fn feature_project(&self, x: &Tensor) -> Result<Tensor> {
        let x = layer_norm(x, &self.fp_ln_w, &self.fp_ln_b, LN_EPS)?;
        crate::linear(&x, &self.fp_w, &self.fp_b)
    }

    /// Grouped positional conv (k=128, pad=64) + bias + SamePad crop + GELU:
    /// `[1, T, 768]` -> `[1, T, 768]`.
    pub fn pos_conv(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let pos = self.pos_conv_raw(x)?;
        let pos = pos.narrow(2, 0, t)?.gelu_erf()?;
        Ok(pos.transpose(1, 2)?.contiguous()?)
    }

    /// Debug helper: positional conv + bias before the SamePad crop / GELU.
    /// Input `[1, T, 768]`, output `[1, 768, T+1]` (conv layout).
    pub fn pos_conv_raw(&self, x: &Tensor) -> Result<Tensor> {
        let xc = x.transpose(1, 2)?.contiguous()?; // [1, 768, T]
        let pos = xc.conv1d(&self.pos_w, 64, 1, 1, 16)?;
        Ok(pos.broadcast_add(&self.pos_b.reshape((1, HIDDEN, 1))?)?)
    }

    /// Debug helper: the merged positional conv weight and its bias.
    pub fn pos_weight(&self) -> (&Tensor, &Tensor) {
        (&self.pos_w, &self.pos_b)
    }

    /// Pre-layer-0 LayerNorm (`hidden_states[0]` is its output).
    pub fn encoder_norm(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, &self.enc_ln_w, &self.enc_ln_b, LN_EPS)
    }

    /// One transformer layer (0-based).
    pub fn encoder_layer(&self, i: usize, x: &Tensor) -> Result<Tensor> {
        self.layers[i].forward(x)
    }
}

/// Feature-extractor output length for an input of `l` samples (chain of
/// valid convs, `floor((l - k) / s) + 1`).
pub fn conv_stack_output_len(l: usize, kernels: &[usize], strides: &[usize]) -> usize {
    let mut l = l;
    for (&k, &s) in kernels.iter().zip(strides.iter()) {
        l = (l - k) / s + 1;
    }
    l
}

/// Convenience for callers: the HuBERT frame count of a padded 16 kHz signal.
pub fn frame_count(l16_padded: usize) -> usize {
    conv_stack_output_len(l16_padded, &CONV_KERNEL, &CONV_STRIDE)
}
