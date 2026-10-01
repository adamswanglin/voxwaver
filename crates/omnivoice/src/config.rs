//! Model hyper-parameters (mirrors `OmniVoice/config.json`).

/// Qwen3 backbone (`llm_config`).
pub const LLM_HIDDEN: usize = 1024;
pub const LLM_LAYERS: usize = 28;
pub const LLM_ATTN_HEADS: usize = 16;
pub const LLM_KV_HEADS: usize = 8;
pub const LLM_HEAD_DIM: usize = 128;
pub const LLM_INTERMEDIATE: usize = 3072;
pub const LLM_RMS_EPS: f64 = 1e-6;
pub const LLM_ROPE_THETA: f64 = 1_000_000.0;
pub const TEXT_VOCAB: usize = 151_676;
/// RoPE table capacity; bidirectional sequences are a few hundred tokens, so
/// this generous bound also covers long instruct/prompt texts.
pub const MAX_POS: usize = 4096;

/// Audio token layout.
pub const NUM_CODEBOOK: usize = 8;
/// Real audio tokens per codebook.
pub const AUDIO_VOCAB: usize = 1025;
/// Audio embedding table rows: `NUM_CODEBOOK * AUDIO_VOCAB` (offsets 1025 apart).
pub const AUDIO_EMBED_VOCAB: usize = NUM_CODEBOOK * AUDIO_VOCAB;
/// Mask (padding) audio token id.
pub const MASK_ID: u32 = 1024;

/// Default generation parameters (`configs/omnivoice.py`).
pub const NUM_STEP: usize = 32;
pub const GUIDANCE_SCALE: f64 = 2.0;
pub const T_SHIFT: f64 = 0.1;
pub const LAYER_PENALTY_FACTOR: f64 = 5.0;
pub const POSITION_TEMPERATURE: f64 = 5.0;
pub const CLASS_TEMPERATURE: f64 = 0.0;

/// Long-form generation (mirrors the Python pipeline's `audio_chunk_threshold`
/// / `audio_chunk_duration`): an estimated duration above the threshold is
/// generated chunk by chunk instead of one oversized pass.
pub const CHUNK_THRESHOLD_FRAMES: usize = 30 * FRAME_RATE;
pub const CHUNK_FRAMES: usize = 15 * FRAME_RATE;
/// Chunk stitching: 0.1 s fade-out + 0.1 s silence gap + 0.1 s fade-in per
/// boundary (the reference `cross_fade_chunks` splits its 0.3 s into thirds).
pub const CHUNK_FADE_SAMPLES: usize = SAMPLE_RATE as usize / 10;
pub const CHUNK_GAP_SAMPLES: usize = SAMPLE_RATE as usize / 10;

/// HiggsAudioV2 audio tokenizer (decode path only).
pub const RVQ_QUANTIZERS: usize = 8;
pub const CODEBOOK_SIZE: usize = 1024;
pub const CODEBOOK_DIM: usize = 64;
/// RVQ output / DAC input hidden size.
pub const RVQ_HIDDEN: usize = 1024;
/// fc2 output channels fed to the acoustic decoder.
pub const DAC_IN: usize = 256;
pub const DECODER_HIDDEN: usize = 1024;
pub const UPSAMPLE_RATES: [usize; 5] = [8, 5, 4, 2, 3];

/// Audio tokenizer encode path (voice clone).
pub const SEM_SAMPLE_RATE: u32 = 16_000;
pub const SEM_HIDDEN: usize = 768;
/// `semantic_downsample_factor` = hop_length / (24000/16000) / 320 = 2.
pub const SEM_DOWNSAMPLE: usize = 2;
/// Hard-coded pad around the 16 kHz waveform before HuBERT.
pub const SEM_PAD: usize = 160;
/// Semantic-encoder strides / residual dilations / channel ratios.
pub const SEM_STRIDES: [usize; 2] = [1, 1];
pub const SEM_DILATIONS: [usize; 2] = [1, 1];
pub const SEM_CHANNEL_RATIOS: [usize; 2] = [1, 1];
/// DAC encoder first-conv channels (`encoder_hidden_size`).
pub const ENC_CHANNELS: usize = 64;
/// DAC encoder block downsampling ratios (product = hop_length 960).
pub const ENC_STRIDES: [usize; 5] = [8, 5, 4, 2, 3];
/// Pad around the 24 kHz waveform when the acoustic branch is one frame
/// short (`hop_length // 2`).
pub const ENC_PAD: usize = 480;

pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_RATE: usize = 25;

/// Special tokens (tokenizer.json `added_tokens`).
pub const TOK_DENOISE: u32 = 151_669;
pub const TOK_LANG_START: u32 = 151_670;
pub const TOK_LANG_END: u32 = 151_671;
pub const TOK_INSTRUCT_START: u32 = 151_672;
pub const TOK_INSTRUCT_END: u32 = 151_673;
pub const TOK_TEXT_START: u32 = 151_674;
pub const TOK_TEXT_END: u32 = 151_675;

/// Duration estimator reference: "Nice to meet you." spoken in 25 frames
/// (1 s at the 25 fps code rate), matching the Python pipeline.
pub const DURATION_REF_TEXT: &str = "Nice to meet you.";
pub const DURATION_REF_FRAMES: f64 = 25.0;
