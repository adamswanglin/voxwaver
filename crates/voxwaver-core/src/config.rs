use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Mirror of fish-speech `DualARModelArgs` (config.json, model_type "dual_ar").
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DualArConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub dim: usize,
    pub intermediate_size: Option<usize>,
    pub n_local_heads: Option<usize>,
    pub head_dim: usize,
    pub rope_base: f64,
    pub norm_eps: f64,
    pub max_seq_len: usize,
    pub dropout: f64,
    pub tie_word_embeddings: bool,
    pub attention_qkv_bias: bool,
    pub attention_o_bias: bool,
    pub attention_qk_norm: bool,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub scale_codebook_embeddings: bool,
    // fast transformer
    pub n_fast_layer: usize,
    pub fast_dim: Option<usize>,
    pub fast_n_head: Option<usize>,
    pub fast_n_local_heads: Option<usize>,
    pub fast_head_dim: Option<usize>,
    pub fast_intermediate_size: Option<usize>,
    pub fast_attention_qkv_bias: Option<bool>,
    pub fast_attention_qk_norm: Option<bool>,
    pub fast_attention_o_bias: Option<bool>,
}

impl Default for DualArConfig {
    fn default() -> Self {
        // Defaults from fish-speech BaseModelArgs / DualARModelArgs.
        Self {
            model_type: "dual_ar".into(),
            vocab_size: 32000,
            n_layer: 32,
            n_head: 32,
            dim: 4096,
            intermediate_size: None,
            n_local_heads: None,
            head_dim: 64,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            max_seq_len: 2048,
            dropout: 0.0,
            tie_word_embeddings: true,
            attention_qkv_bias: false,
            attention_o_bias: false,
            attention_qk_norm: false,
            codebook_size: 160,
            num_codebooks: 4,
            scale_codebook_embeddings: false,
            n_fast_layer: 4,
            fast_dim: None,
            fast_n_head: None,
            fast_n_local_heads: None,
            fast_head_dim: None,
            fast_intermediate_size: None,
            fast_attention_qkv_bias: None,
            fast_attention_qk_norm: None,
            fast_attention_o_bias: None,
        }
    }
}

impl DualArConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let cfg: DualArConfig =
            serde_json::from_str(&data).with_context(|| format!("parse {}", path.display()))?;
        anyhow::ensure!(
            cfg.model_type == "dual_ar",
            "unsupported model_type {} (expected dual_ar)",
            cfg.model_type
        );
        Ok(cfg)
    }

    /// Effective values after DualARModelArgs::__post_init__ fallbacks.
    pub fn n_local_heads(&self) -> usize {
        self.n_local_heads.unwrap_or(self.n_head)
    }
    pub fn fast_dim(&self) -> usize {
        self.fast_dim.unwrap_or(self.dim)
    }
    pub fn fast_n_head(&self) -> usize {
        self.fast_n_head.unwrap_or(self.n_head)
    }
    pub fn fast_n_local_heads(&self) -> usize {
        self.fast_n_local_heads.unwrap_or(self.n_local_heads())
    }
    pub fn fast_head_dim(&self) -> usize {
        self.fast_head_dim.unwrap_or(self.head_dim)
    }
    pub fn fast_intermediate_size(&self) -> usize {
        self.fast_intermediate_size
            .or(self.intermediate_size)
            .unwrap_or(4 * self.dim)
    }
    pub fn fast_attention_qk_norm(&self) -> bool {
        self.fast_attention_qk_norm.unwrap_or(self.attention_qk_norm)
    }
}

/// Hardcoded hyper-parameters of the codec shipped as s1-mini `codec.pth`
/// (fish_speech/configs/modded_dac_vq.yaml + ModelArgs defaults).
#[derive(Debug, Clone, Copy)]
pub struct CodecConfig {
    pub sample_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: [usize; 4],
    pub decoder_dim: usize,
    pub decoder_rates: [usize; 4],
    pub latent_dim: usize,
    pub enc_transformer_layers: [usize; 4],
    pub dec_transformer_layers: [usize; 4],
    pub quantizer_input_dim: usize,
    pub n_residual_codebooks: usize,
    pub codebook_size: usize,
    pub semantic_codebook_size: usize,
    pub codebook_dim: usize,
    pub downsample_factor: [usize; 2],
    // transformer_general_config (ModelArgs) for the encoder transformer
    pub enc_t_n_head: usize,
    pub enc_t_head_dim: usize,
    pub enc_t_intermediate: usize,
    pub enc_t_window: usize,
    // pre/post module transformer (ModelArgs from yaml)
    pub q_t_n_layer: usize,
    pub q_t_n_head: usize,
    pub q_t_dim: usize,
    pub q_t_intermediate: usize,
    pub q_t_head_dim: usize,
    pub q_t_window: usize,
    pub q_t_norm_eps: f64,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            encoder_dim: 64,
            encoder_rates: [2, 4, 8, 8],
            decoder_dim: 1536,
            decoder_rates: [8, 8, 4, 2],
            latent_dim: 1024,
            enc_transformer_layers: [0, 0, 0, 4],
            dec_transformer_layers: [4, 0, 0, 0], // dead code in reference; ignored
            quantizer_input_dim: 1024,
            n_residual_codebooks: 9,
            codebook_size: 1024,
            semantic_codebook_size: 4096,
            codebook_dim: 8,
            downsample_factor: [2, 2],
            enc_t_n_head: 16, // dim(1024) // head_dim(64)
            enc_t_head_dim: 64,
            enc_t_intermediate: 3072, // dim * 3
            enc_t_window: 512,        // ModelArgs default window_size
            q_t_n_layer: 8,
            q_t_n_head: 16,
            q_t_dim: 1024,
            q_t_intermediate: 3072,
            q_t_head_dim: 64,
            q_t_window: 128,
            q_t_norm_eps: 1e-5,
        }
    }
}

impl CodecConfig {
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }
    /// One codec frame = `hop_length * prod(downsample_factor)` input samples.
    pub fn frame_length(&self) -> usize {
        self.hop_length() * self.downsample_factor.iter().product::<usize>()
    }
}
