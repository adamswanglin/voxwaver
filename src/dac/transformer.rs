//! `WindowLimitedTransformer` from fish-speech `modded_dac.py`: a causal
//! transformer with band-limited (windowed) attention, RoPE and LayerScale.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::{rms_norm, silu};
use candle_nn::rotary_emb::rope_i;

pub struct WltBlock {
    pub wqkv: Tensor, // [dim, 3*dim]
    pub wo: Tensor,   // [dim, dim]
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub w1: Tensor,
    pub w3: Tensor,
    pub w2: Tensor,
    pub attn_ls: Tensor, // LayerScale gamma
    pub ffn_ls: Tensor,
}

pub struct Wlt {
    layers: Vec<WltBlock>,
    norm: Tensor,
    window: usize,
    n_head: usize,
    head_dim: usize,
    rope_base: f64,
    eps: f32,
    device: Device,
}

impl Wlt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: Vec<WltBlock>,
        norm: Tensor,
        window: usize,
        n_head: usize,
        head_dim: usize,
        rope_base: f64,
        eps: f32,
        device: Device,
    ) -> Self {
        Self {
            layers,
            norm,
            window,
            n_head,
            head_dim,
            rope_base,
            eps,
            device,
        }
    }

    /// bf16-rounded interleaved RoPE tables for positions 0..t (the reference
    /// precomputes `freqs_cis` in bf16), shape [t, head_dim/2].
    fn rope_tables(&self, t: usize) -> Result<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let mut freqs = vec![0f32; half];
        for i in 0..half {
            freqs[i] = self
                .rope_base
                .powf(-(2.0 * i as f64) / self.head_dim as f64) as f32;
        }
        let mut cos_v = vec![0f32; t * half];
        let mut sin_v = vec![0f32; t * half];
        for p in 0..t {
            for i in 0..half {
                let (s, c) = ((p as f32) * freqs[i]).sin_cos();
                cos_v[p * half + i] = c;
                sin_v[p * half + i] = s;
            }
        }
        let round = |v: Vec<f32>| -> Result<Tensor> {
            let t = Tensor::from_vec(v, (t, half), &Device::Cpu)?;
            Ok(t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.to_device(&self.device)?)
        };
        Ok((round(cos_v)?, round(sin_v)?))
    }

    /// Additive mask for causal window-limited attention: allowed when
    /// `col <= row` and `col >= row - window + 1`. Shape [1, 1, t, t].
    fn window_mask(&self, t: usize) -> Result<Tensor> {
        let mut m = vec![0f32; t * t];
        for row in 0..t {
            let lo = row.saturating_sub(self.window - 1);
            for col in (row + 1)..t {
                m[row * t + col] = f32::NEG_INFINITY;
            }
            for col in 0..lo {
                m[row * t + col] = f32::NEG_INFINITY;
            }
        }
        // Metal sdpa requires the explicit head dim on the mask
        Ok(Tensor::from_vec(m, (1, 1, t, t), &self.device)?
            .expand((1, self.n_head, t, t))?)
    }

    /// x: [1, C, T] (channels first) -> [1, C, T].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(2)?;
        let mut xt = x.transpose(1, 2)?.contiguous()?; // [1, T, C]
        let dim = xt.dim(2)?;
        let (cos, sin) = self.rope_tables(t)?;
        let mask = self.window_mask(t)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        for layer in &self.layers {
            // attention
            let h = rms_norm(&xt, &layer.attn_norm, self.eps)?;
            let qkv = h.matmul(&layer.wqkv.t()?.unsqueeze(0)?)?; // [1, T, 3*dim]
            let q = qkv.narrow(2, 0, dim)?;
            let k = qkv.narrow(2, dim, dim)?;
            let v = qkv.narrow(2, 2 * dim, dim)?;
            let split = |x: &Tensor| -> Result<Tensor> {
                Ok(x.reshape((1, t, self.n_head, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(DType::F32)?)
            };
            let q = rope_i(&split(&q)?, &cos, &sin)?;
            let k = rope_i(&split(&k)?, &cos, &sin)?;
            let v = split(&v)?;
            let y = crate::dual_ar::attention(&q, &k, &v, Some(&mask), false, scale)?;
            let y = y
                .to_dtype(xt.dtype())?
                .transpose(1, 2)?
                .reshape((1, t, dim))?
                .matmul(&layer.wo.t()?.unsqueeze(0)?)?
                .broadcast_mul(&layer.attn_ls)?;
            let h = xt.add(&y)?;

            // ffn
            let h2 = rms_norm(&h, &layer.ffn_norm, self.eps)?;
            let a = h2.matmul(&layer.w1.t()?.unsqueeze(0)?)?;
            let b = h2.matmul(&layer.w3.t()?.unsqueeze(0)?)?;
            let f = silu(&a)?.mul(&b)?.matmul(&layer.w2.t()?.unsqueeze(0)?)?.broadcast_mul(&layer.ffn_ls)?;
            xt = h.add(&f)?;
        }
        let xt = rms_norm(&xt, &self.norm, self.eps)?;
        Ok(xt.transpose(1, 2)?.contiguous()?)
    }
}
