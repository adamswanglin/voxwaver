//! Qwen3 bidirectional backbone (28 layers, GQA 16Q/8KV heads, head_dim 128,
//! per-head QK-norm, RoPE theta 1e6).
//!
//! The generator attends over the full sequence in both directions — no
//! causal mask, no KV cache. Qwen3 uses the NeoX ("half-split") rotary
//! convention, which is candle's `candle_nn::rotary_emb::rope`. All
//! projections are bias-free; the MLP is SwiGLU.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::ops::{rms_norm, silu};
use candle_nn::rotary_emb::rope;

use crate::config as cfg;

pub struct Qwen3 {
    blocks: Vec<Block>,
    norm: Tensor,
    rope_cos: Tensor, // [max_pos, head_dim/2] f32
    rope_sin: Tensor,
}

struct Block {
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    q_norm: Tensor, // [head_dim] per-head RMSNorm
    k_norm: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    attn_norm: Tensor,
    ffn_norm: Tensor,
}

/// Scaled dot-product attention, `dual_ar` dispatch: fused SDPA on Metal for
/// non-causal shapes (this backbone is always non-causal, so the fused path is
/// always safe there); manual matmul/softmax otherwise. The manual path
/// requires k/v repeated to the query head count.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    if q.device().is_metal() {
        return Ok(candle_nn::ops::sdpa(q, k, v, None, false, scale as f32, 1.0)?);
    }
    let att = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?; // [1, h, t, s]
    let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
    Ok(att.matmul(v)?)
}

/// Precompute cos/sin tables [max_pos, head_dim/2] for the NeoX rotary.
fn rope_tables(theta: f64, head_dim: usize, max_pos: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f64 / head_dim as f64);
        for p in 0..max_pos {
            let a = p as f64 * inv;
            cos[p * half + j] = a.cos() as f32;
            sin[p * half + j] = a.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_pos, half), device)?,
        Tensor::from_vec(sin, (max_pos, half), device)?,
    ))
}

impl Qwen3 {
    /// Weight names are relative to the `llm.` prefix, e.g.
    /// `layers.0.self_attn.q_proj.weight` — the caller strips the prefix.
    pub fn load(vs: &dyn Fn(&str) -> candle_core::Result<Tensor>) -> Result<Self> {
        let device = vs("norm.weight")?.device().clone();
        let mut blocks = Vec::with_capacity(cfg::LLM_LAYERS);
        for i in 0..cfg::LLM_LAYERS {
            let p = move |name: &str| vs(&format!("layers.{i}.{name}"));
            blocks.push(Block {
                q_w: p("self_attn.q_proj.weight")?,
                k_w: p("self_attn.k_proj.weight")?,
                v_w: p("self_attn.v_proj.weight")?,
                o_w: p("self_attn.o_proj.weight")?,
                q_norm: p("self_attn.q_norm.weight")?,
                k_norm: p("self_attn.k_norm.weight")?,
                gate_w: p("mlp.gate_proj.weight")?,
                up_w: p("mlp.up_proj.weight")?,
                down_w: p("mlp.down_proj.weight")?,
                attn_norm: p("input_layernorm.weight")?,
                ffn_norm: p("post_attention_layernorm.weight")?,
            });
        }
        let (rope_cos, rope_sin) =
            rope_tables(cfg::LLM_ROPE_THETA, cfg::LLM_HEAD_DIM, cfg::MAX_POS, &device)?;
        Ok(Self {
            blocks,
            norm: vs("norm.weight")?,
            rope_cos,
            rope_sin,
        })
    }

    /// Full bidirectional forward: `x` [1, T, hidden] -> post-final-norm
    /// hidden states [1, T, hidden].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, t, _h) = x.dims3()?;
        anyhow::ensure!(
            t <= cfg::MAX_POS,
            "sequence length {t} exceeds RoPE table capacity {}",
            cfg::MAX_POS
        );
        let cos = self.rope_cos.narrow(0, 0, t)?.contiguous()?;
        let sin = self.rope_sin.narrow(0, 0, t)?.contiguous()?;
        let mut x = x.clone();
        let scale = 1.0 / (cfg::LLM_HEAD_DIM as f64).sqrt();
        let rep = cfg::LLM_ATTN_HEADS / cfg::LLM_KV_HEADS;
        for blk in &self.blocks {
            let xn = rms_norm(&x, &blk.attn_norm, cfg::LLM_RMS_EPS as f32)?;
            let q = crate::linear_nobias(&xn, &blk.q_w)?
                .reshape((1, t, cfg::LLM_ATTN_HEADS, cfg::LLM_HEAD_DIM))?
                .transpose(1, 2)? // [1, 16, T, 128]
                .contiguous()?;
            let k = crate::linear_nobias(&xn, &blk.k_w)?
                .reshape((1, t, cfg::LLM_KV_HEADS, cfg::LLM_HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = crate::linear_nobias(&xn, &blk.v_w)?
                .reshape((1, t, cfg::LLM_KV_HEADS, cfg::LLM_HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?;

            // Qwen3 per-head QK-norm before RoPE.
            let q = rms_norm(&q, &blk.q_norm, cfg::LLM_RMS_EPS as f32)?;
            let k = rms_norm(&k, &blk.k_norm, cfg::LLM_RMS_EPS as f32)?;
            let q = rope(&q, &cos, &sin)?;
            let k = rope(&k, &cos, &sin)?;

            // Expand k/v to the query head count with PyTorch's
            // repeat_interleave semantics (query head i reads kv head i / rep),
            // matching `_eager_qkv_norm_rope` and the Triton kernel's
            // `kv_head * kv_repeat + rep` broadcast. A block `repeat` would map
            // query head i to kv head i % rep instead and scramble attention.
            let k = k
                .unsqueeze(2)?
                .expand((1, cfg::LLM_KV_HEADS, rep, t, cfg::LLM_HEAD_DIM))?
                .reshape((1, cfg::LLM_ATTN_HEADS, t, cfg::LLM_HEAD_DIM))?;
            let v = v
                .unsqueeze(2)?
                .expand((1, cfg::LLM_KV_HEADS, rep, t, cfg::LLM_HEAD_DIM))?
                .reshape((1, cfg::LLM_ATTN_HEADS, t, cfg::LLM_HEAD_DIM))?;
            let attn = attention(&q, &k, &v, scale)?
                .transpose(1, 2)?
                .reshape((1, t, cfg::LLM_ATTN_HEADS * cfg::LLM_HEAD_DIM))?;
            let x_new = crate::linear_nobias(&attn, &blk.o_w)?;
            x = (x + x_new)?;

            let xn = rms_norm(&x, &blk.ffn_norm, cfg::LLM_RMS_EPS as f32)?;
            let h = (silu(&crate::linear_nobias(&xn, &blk.gate_w)?)? * crate::linear_nobias(&xn, &blk.up_w)?)?;
            let x_new = crate::linear_nobias(&h, &blk.down_w)?;
            x = (x + x_new)?;
        }
        let out = rms_norm(&x, &self.norm, cfg::LLM_RMS_EPS as f32)?;
        Ok(out)
    }
}
