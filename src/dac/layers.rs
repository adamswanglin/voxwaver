//! Conv building blocks of the modded-DAC codec: causal weight-normed convs
//! (already folded at load time), Snake activation, ResidualUnit, encoder /
//! decoder blocks and the ConvNeXt block used by the RVQ down/upsampler.
use super::transformer::Wlt;
use anyhow::Result;
use candle_core::Tensor;

/// `get_extra_padding_for_conv1d`: right-pad so the output length covers a
/// whole number of frames.
fn extra_padding(length: usize, kernel: usize, stride: usize, padding_total: usize) -> usize {
    let n_frames = (length as f64 - kernel as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal = ((n_frames.ceil() as i64 - 1) * stride as i64 + (kernel - padding_total) as i64) as usize;
    ideal.saturating_sub(length)
}

/// CausalConvNet: left pad `k_eff - stride`, right pad `extra`, then a plain
/// (dilated, strided) conv. The `padding` constructor argument of the Python
/// code is ignored there too.
pub struct ConvW {
    w: Tensor, // [out, in/groups, k]
    b: Tensor, // [out]
    k_eff: usize,
    stride: usize,
    dil: usize,
    groups: usize,
}

impl ConvW {
    pub fn new(w: Tensor, b: Tensor, kernel_size: usize, stride: usize, dilation: usize, groups: usize) -> Self {
        Self {
            w,
            b,
            k_eff: (kernel_size - 1) * dilation + 1,
            stride,
            dil: dilation,
            groups,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let l = x.dim(2)?;
        let pad_left = self.k_eff - self.stride;
        let extra = extra_padding(l, self.k_eff, self.stride, pad_left);
        let x = x.pad_with_zeros(2, pad_left, extra)?;
        let y = x.conv1d(&self.w, 0, self.stride, self.dil, self.groups)?;
        let b = self.b.reshape((1, self.b.elem_count(), 1))?;
        Ok(y.broadcast_add(&b)?)
    }
}

/// CausalTransConvNet: ConvTranspose1d (no padding) then crop
/// `ceil(k - s)` from the right.
pub struct TransConvW {
    w: Tensor, // [in, out, k]
    b: Tensor, // [out]
    kernel_size: usize,
    stride: usize,
}

impl TransConvW {
    pub fn new(w: Tensor, b: Tensor, kernel_size: usize, stride: usize) -> Self {
        Self { w, b, kernel_size, stride }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.w, 0, 0, self.stride, 1, 1)?;
        let b = self.b.reshape((1, self.b.elem_count(), 1))?;
        let y = y.broadcast_add(&b)?;
        let crop = self.kernel_size - self.stride;
        let l = y.dim(2)?;
        Ok(y.narrow(2, 0, l - crop)?)
    }
}

/// Snake1d: `x + sin^2(alpha * x) / (alpha + 1e-9)`, alpha of shape (1, C, 1).
pub struct Snake {
    alpha: Tensor, // [C]
}

impl Snake {
    pub fn new(alpha: Tensor) -> Self {
        Self { alpha }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.alpha.reshape((1, self.alpha.elem_count(), 1))?;
        let ax = x.broadcast_mul(&a)?;
        let s2 = ax.sin()?.mul(&ax.sin()?)?;
        let denom = a.affine(1.0, 1e-9)?;
        Ok(x.add(&s2.broadcast_div(&denom)?)?)
    }
}

/// ResidualUnit: Snake -> CausalConv(k=7, dilation) -> Snake -> CausalConv(k=1),
/// residual input cropped from the right if the block shortened it.
pub struct ResidualUnit {
    s1: Snake,
    c1: ConvW,
    s2: Snake,
    c2: ConvW,
}

impl ResidualUnit {
    pub fn new(s1: Snake, c1: ConvW, s2: Snake, c2: ConvW) -> Self {
        Self { s1, c1, s2, c2 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s1 = self.s1.forward(x)?;
        let c1 = self.c1.forward(&s1)?;
        let s2 = self.s2.forward(&c1)?;
        let y = self.c2.forward(&s2)?;
        let pad = x.dim(2)? as i64 - y.dim(2)? as i64;
        if pad > 0 {
            let x = x.narrow(2, 0, x.dim(2)? - pad as usize)?;
            Ok(x.add(&y)?)
        } else {
            Ok(x.add(&y)?)
        }
    }
}

/// EncoderBlock: 3 ResidualUnits (dilation 1/3/9) -> Snake -> downsample conv
/// -> optional WindowLimitedTransformer.
pub struct EncoderBlock {
    rus: Vec<ResidualUnit>,
    snake: Snake,
    conv: ConvW,
    transformer: Option<Wlt>,
}

impl EncoderBlock {
    pub fn new(rus: Vec<ResidualUnit>, snake: Snake, conv: ConvW, transformer: Option<Wlt>) -> Self {
        Self { rus, snake, conv, transformer }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for ru in &self.rus {
            x = ru.forward(&x)?;
        }
        let x = self.conv.forward(&self.snake.forward(&x)?)?;
        match &self.transformer {
            Some(t) => t.forward(&x),
            None => Ok(x),
        }
    }
}

/// DecoderBlock: Snake -> upsample transposed conv -> 3 ResidualUnits.
/// (The `transformer_module` in the Python DecoderBlock is dead code.)
pub struct DecoderBlock {
    snake: Snake,
    conv: TransConvW,
    rus: Vec<ResidualUnit>,
}

impl DecoderBlock {
    pub fn new(snake: Snake, conv: TransConvW, rus: Vec<ResidualUnit>) -> Self {
        Self { snake, conv, rus }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv.forward(&self.snake.forward(x)?)?;
        for ru in &self.rus {
            x = ru.forward(&x)?;
        }
        Ok(x)
    }
}

/// Encoder: [1, 1, L] -> [1, latent, L / hop].
pub struct Encoder {
    conv_in: ConvW,
    blocks: Vec<EncoderBlock>,
    snake_out: Snake,
    conv_out: ConvW,
}

impl Encoder {
    pub fn new(conv_in: ConvW, blocks: Vec<EncoderBlock>, snake_out: Snake, conv_out: ConvW) -> Self {
        Self { conv_in, blocks, snake_out, conv_out }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        self.conv_out.forward(&self.snake_out.forward(&x)?)
    }
}

/// Decoder: [1, latent, T] -> [1, 1, T * hop], tanh output.
pub struct Decoder {
    conv_in: ConvW,
    blocks: Vec<DecoderBlock>,
    snake_out: Snake,
    conv_out: ConvW,
}

impl Decoder {
    pub fn new(conv_in: ConvW, blocks: Vec<DecoderBlock>, snake_out: Snake, conv_out: ConvW) -> Self {
        Self { conv_in, blocks, snake_out, conv_out }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
            // Recycle pooled Metal buffers between blocks: each block's im2col
            // intermediates are dead by the next block, but the pool pins one
            // buffer per size class until a sync, so a large decode otherwise
            // holds the sum of all blocks' classes (~GBs). candle's
            // synchronize() drops unused pooled buffers as a side effect.
            if x.device().is_metal() {
                x.device().synchronize()?;
            }
        }
        let x = self.conv_out.forward(&self.snake_out.forward(&x)?)?;
        Ok(x.tanh()?)
    }
}

/// ConvNeXt block over channels: depthwise causal conv -> LayerNorm(1e-6) over
/// channels -> Linear -> GELU(erf) -> Linear -> gamma(1e-6), with residual.
pub struct ConvNeXt {
    dwconv: ConvW,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1_w: Tensor,
    pw1_b: Tensor,
    pw2_w: Tensor,
    pw2_b: Tensor,
    gamma: Tensor,
}

impl ConvNeXt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dwconv: ConvW,
        ln_w: Tensor,
        ln_b: Tensor,
        pw1_w: Tensor,
        pw1_b: Tensor,
        pw2_w: Tensor,
        pw2_b: Tensor,
        gamma: Tensor,
    ) -> Self {
        Self { dwconv, ln_w, ln_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma }
    }

    /// x: [1, C, T]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.dwconv.forward(x)?; // [1, C, T]
        let yt = y.transpose(1, 2)?.contiguous()?; // [1, T, C]
        // LayerNorm over the channel dim, eps 1e-6
        let mean = yt.mean_keepdim(candle_core::D::Minus1)?;
        let centered = yt.broadcast_sub(&mean)?;
        let var = centered.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = centered
            .broadcast_div(&(var + 1e-6)?.sqrt()?)?
            .broadcast_mul(&self.ln_w)?
            .broadcast_add(&self.ln_b)?;
        let h = normed
            .matmul(&self.pw1_w.t()?.unsqueeze(0)?)?
            .broadcast_add(&self.pw1_b.reshape((1, 1, self.pw1_b.elem_count()))?)?
            .gelu_erf()?
            .matmul(&self.pw2_w.t()?.unsqueeze(0)?)?
            .broadcast_add(&self.pw2_b.reshape((1, 1, self.pw2_b.elem_count()))?)?
            .broadcast_mul(&self.gamma)?;
        let h = h.transpose(1, 2)?.contiguous()?;
        Ok(x.add(&h)?)
    }
}
