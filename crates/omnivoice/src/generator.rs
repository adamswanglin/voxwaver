//! Stage-0 generator: Qwen3 bidirectional backbone + text/audio dual
//! embeddings + the 8-codebook prediction head, wrapped in the 32-step
//! iterative unmasking loop (port of `omnivoice_generator.py`).
//!
//! Inputs are built as `[text tokens | ref audio tokens | target]` for the
//! conditional half (clone mode; `[text | target]` without a reference) and
//! `[target]` for the unconditional half; each step runs both through the
//! backbone, combines the target-region logits with classifier-free guidance
//! `log_softmax((1+s)*c - s*u)`, predicts tokens greedily (or with Gumbel
//! noise at `class_temperature`), ranks positions by confidence minus a
//! per-codebook layer penalty plus Gumbel noise, unmasks the top-k, and
//! mirrors the new tokens into both halves for the next step.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::safetensors::MmapedSafetensors;
use rand::rngs::StdRng;
use rand::Rng;
use tts_common::{CancelFlag, Progress, ProgressSink};

use crate::config as cfg;
use crate::qwen3::Qwen3;

/// Unmasking hyper-parameters (defaults from `configs/omnivoice.py`).
#[derive(Clone)]
pub struct GenParams {
    pub num_step: usize,
    pub guidance_scale: f64,
    pub t_shift: f64,
    pub layer_penalty_factor: f64,
    pub position_temperature: f64,
    pub class_temperature: f64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            num_step: cfg::NUM_STEP,
            guidance_scale: cfg::GUIDANCE_SCALE,
            t_shift: cfg::T_SHIFT,
            layer_penalty_factor: cfg::LAYER_PENALTY_FACTOR,
            position_temperature: cfg::POSITION_TEMPERATURE,
            class_temperature: cfg::CLASS_TEMPERATURE,
        }
    }
}

pub struct Generator {
    text_emb: Tensor,      // [TEXT_VOCAB, hidden]
    audio_emb: Tensor,     // [NUM_CODEBOOK*AUDIO_VOCAB, hidden]
    audio_heads_w: Tensor, // [NUM_CODEBOOK*AUDIO_VOCAB, hidden]
    qwen3: Qwen3,
    device: Device,
}

impl Generator {
    pub fn load(model_dir: &std::path::Path, device: &Device) -> Result<Self> {
        let path = model_dir.join("model.safetensors");
        // SAFETY: the weight file is a read-only input of this process.
        let st = unsafe { MmapedSafetensors::new(&path) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let vs = |name: &str| st.load(name, device);
        let text_emb = vs("llm.embed_tokens.weight")?;
        let audio_emb = vs("audio_embeddings.weight")?;
        let audio_heads_w = vs("audio_heads.weight")?;
        let qwen3 = Qwen3::load(&|name: &str| vs(&format!("llm.{name}")))?;
        Ok(Self {
            text_emb,
            audio_emb,
            audio_heads_w,
            qwen3,
            device: device.clone(),
        })
    }

    /// Mixed text+audio embeddings for one packed sequence (`_prepare_embeddings`).
    ///
    /// `ids` [T, 8] U32: text tokens replicated across codebooks, audio
    /// positions hold per-codebook token ids. `audio_mask` [T] U32: 1 for
    /// audio positions, 0 for text.
    fn prepare_embeddings(&self, ids: &Tensor, audio_mask: &Tensor) -> Result<Tensor> {
        let t = ids.dim(0)?;
        // Text-position embeddings from the first codebook row.
        let first = ids.narrow(1, 0, 1)?.reshape((t,))?;
        let text_embeds = self.text_emb.index_select(&first, 0)?; // [T, hidden]
        // Audio embeddings: per-codebook offset (arange(8) * 1025), lookup,
        // sum across codebooks.
        let offsets: Vec<u32> = (0..cfg::NUM_CODEBOOK).map(|i| (i * cfg::AUDIO_VOCAB) as u32).collect();
        let offsets = Tensor::from_vec(offsets, (1, cfg::NUM_CODEBOOK), &self.device)?;
        let shifted = ids
            .broadcast_mul(&audio_mask.unsqueeze(1)?)?
            .broadcast_add(&offsets)?; // [T, 8]
        let flat = shifted.reshape((t * cfg::NUM_CODEBOOK,))?;
        let audio_embeds = self
            .audio_emb
            .index_select(&flat, 0)?
            .reshape((t, cfg::NUM_CODEBOOK, cfg::LLM_HIDDEN))?
            .sum_keepdim(1)?
            .squeeze(1)?; // [T, hidden]
        // Merge: audio where audio_mask=1, text elsewhere.
        let m = audio_mask.to_dtype(DType::F32)?.unsqueeze(1)?; // [T, 1]
        let inv = m.affine(-1.0, 1.0)?;
        Ok(audio_embeds
            .broadcast_mul(&m)?
            .broadcast_add(&text_embeds.broadcast_mul(&inv)?)?)
    }

    /// Project hidden states to per-codebook logits (`_get_logits`):
    /// [T, hidden] -> [8, T, AUDIO_VOCAB].
    fn get_logits(&self, h: &Tensor) -> Result<Tensor> {
        let t = h.dim(0)?;
        let flat = crate::linear_nobias(h, &self.audio_heads_w)?; // [T, 8*vocab]
        Ok(flat
            .reshape((t, cfg::NUM_CODEBOOK, cfg::AUDIO_VOCAB))?
            .permute((1, 0, 2))?
            .contiguous()?)
    }

    /// One backbone pass over a packed sequence -> logits [8, T, 1025].
    fn step_logits(&self, ids_flat: &[u32], t: usize, mask: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(ids_flat.to_vec(), (t, cfg::NUM_CODEBOOK), &self.device)?;
        let mask = Tensor::from_vec(mask.to_vec(), (t,), &self.device)?;
        let emb = self.prepare_embeddings(&ids, &mask)?; // [T, hidden]
        let h = self.qwen3.forward(&emb.unsqueeze(0)?)?; // [1, T, hidden]
        let logits = self.get_logits(&h.squeeze(0)?)?;
        Ok(logits)
    }

    /// Run the full iterative unmasking generation for one request.
    ///
    /// `ref_codes` (optional) are the voice-clone reference audio tokens as
    /// `[8][t_ref]`; they extend the conditional half to
    /// `[text | ref | target]` (vllm-omni `_prepare_request_input`).
    ///
    /// Returns the audio tokens as `[8][target_len]` (values < 1024).
    pub fn generate(
        &self,
        text_ids: &[u32],
        ref_codes: Option<&[Vec<u32>]>,
        target_len: usize,
        params: &GenParams,
        rng: &mut StdRng,
        cancel: &CancelFlag,
        sink: &dyn ProgressSink,
        chunk: usize,
        chunk_total: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let t_text = text_ids.len();
        let t_ref = ref_codes.map_or(0, |r| r[0].len());
        let t_cond = t_text + t_ref + target_len;
        anyhow::ensure!(t_cond <= cfg::MAX_POS, "cond sequence {t_cond} exceeds RoPE capacity");

        // CPU-side token sheets, row-major [T][8].
        let mut cond_ids = vec![cfg::MASK_ID; t_cond * cfg::NUM_CODEBOOK];
        for (pos, &tok) in text_ids.iter().enumerate() {
            for cb in 0..cfg::NUM_CODEBOOK {
                cond_ids[pos * cfg::NUM_CODEBOOK + cb] = tok;
            }
        }
        // Reference-audio region: real audio tokens per codebook.
        if let Some(rc) = ref_codes {
            for (cb, row) in rc.iter().enumerate() {
                for (pos, &tok) in row.iter().enumerate() {
                    cond_ids[(t_text + pos) * cfg::NUM_CODEBOOK + cb] = tok;
                }
            }
        }
        let mut uncond_ids = vec![cfg::MASK_ID; target_len * cfg::NUM_CODEBOOK];
        let cond_mask: Vec<u32> = (0..t_cond).map(|p| (p >= t_text) as u32).collect();
        let uncond_mask: Vec<u32> = vec![1; target_len];

        // Unmasking schedule (`_get_time_steps` + per-step counts).
        let timesteps = get_time_steps(0.0, 1.0, params.num_step + 1, params.t_shift);
        let total_mask = target_len * cfg::NUM_CODEBOOK;
        let mut rem = total_mask;
        let mut sched = Vec::with_capacity(params.num_step);
        for step in 0..params.num_step {
            let num = if step == params.num_step - 1 {
                rem
            } else {
                ((total_mask as f64 * (timesteps[step + 1] - timesteps[step])).ceil() as usize)
                    .min(rem)
            };
            sched.push(num);
            rem -= num;
        }

        let mut tokens = vec![vec![cfg::MASK_ID; target_len]; cfg::NUM_CODEBOOK];
        for (step, &k) in sched.iter().enumerate() {
            sink.progress(Progress::Unmasking {
                chunk,
                total: chunk_total,
                step,
                total_steps: params.num_step,
            });
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            if k == 0 {
                continue;
            }
            let logits_c = self.step_logits(&cond_ids, t_cond, &cond_mask)?;
            let logits_u = self.step_logits(&uncond_ids, target_len, &uncond_mask)?;
            // Target region only: cond holds it at the sequence tail.
            let c: Vec<Vec<Vec<f32>>> = logits_c
                .narrow(1, t_cond - target_len, target_len)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec3()?;
            let u: Vec<Vec<Vec<f32>>> = logits_u
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec3()?;
            unmask_step(&c, &u, &mut tokens, k, params, rng);
            // Mirror the new tokens into both halves for the next step.
            for cb in 0..cfg::NUM_CODEBOOK {
                for pos in 0..target_len {
                    let tok = tokens[cb][pos];
                    cond_ids[(t_text + t_ref + pos) * cfg::NUM_CODEBOOK + cb] = tok;
                    uncond_ids[pos * cfg::NUM_CODEBOOK + cb] = tok;
                }
            }
        }
        Ok(tokens)
    }
}

/// Unmasking schedule with time shift: `r_n = t_shift*s / (1 + (t_shift-1)*s)`
/// over `linspace(t_start, t_end, num_step)`.
fn get_time_steps(t_start: f64, t_end: f64, num_step: usize, t_shift: f64) -> Vec<f64> {
    let step = (t_end - t_start) / (num_step as f64 - 1.0);
    (0..num_step)
        .map(|i| {
            let s = t_start + i as f64 * step;
            t_shift * s / (1.0 + (t_shift - 1.0) * s)
        })
        .collect()
}

/// Gumbel noise `-ln(-ln(u))` with the same clamp as the Python reference.
fn gumbel(rng: &mut StdRng) -> f64 {
    let u = (rng.gen::<f32>() as f64).max(1e-8);
    -(-u.ln()).ln()
}

/// One sampling/unmasking step over the target region, entirely on the CPU
/// (`_unmask_one_request`). `c`/`u` are [8][t][1025] FP32 logits.
fn unmask_step(
    c: &[Vec<Vec<f32>>],
    u: &[Vec<Vec<f32>>],
    tokens: &mut [Vec<u32>],
    k: usize,
    p: &GenParams,
    rng: &mut StdRng,
) {
    let n_cb = c.len();
    let t = c[0].len();
    let vocab = c[0][0].len();
    let mask_id = cfg::MASK_ID as usize;
    let mut pred = vec![vec![0u32; t]; n_cb];
    let mut scores = vec![f64::NEG_INFINITY; n_cb * t];
    let s = p.guidance_scale;

    for cb in 0..n_cb {
        for pos in 0..t {
            let cr = &c[cb][pos];
            let ur = &u[cb][pos];
            // Guided logits -> log_softmax in f64.
            let mut m = f64::NEG_INFINITY;
            let mut g = vec![0f64; vocab];
            for (j, v) in g.iter_mut().enumerate() {
                let x = if s != 0.0 {
                    (1.0 + s) * cr[j] as f64 - s * ur[j] as f64
                } else {
                    cr[j] as f64
                };
                *v = x;
                if x > m {
                    m = x;
                }
            }
            let mut sum = 0f64;
            for v in &g {
                sum += (v - m).exp();
            }
            let lse = m + sum.ln();
            // Token prediction: greedy argmax of the log-probs (mask column
            // excluded), or Gumbel-argmax when class_temperature > 0.
            let mut best_j = 0usize;
            if p.class_temperature > 0.0 {
                let mut best = f64::NEG_INFINITY;
                for (j, v) in g.iter().enumerate() {
                    if j == mask_id {
                        continue;
                    }
                    let sc = (v - lse) / p.class_temperature + gumbel(rng);
                    if sc > best {
                        best = sc;
                        best_j = j;
                    }
                }
            } else {
                let mut best = f64::NEG_INFINITY;
                for (j, v) in g.iter().enumerate() {
                    if j == mask_id {
                        continue;
                    }
                    let lp = v - lse;
                    if lp > best {
                        best = lp;
                        best_j = j;
                    }
                }
            }
            pred[cb][pos] = best_j as u32;
            // Confidence minus the layer penalty, plus Gumbel position noise.
            let mut score = g[best_j] - lse - cb as f64 * p.layer_penalty_factor;
            if p.position_temperature > 0.0 {
                score = score / p.position_temperature + gumbel(rng);
            }
            if tokens[cb][pos] != cfg::MASK_ID {
                score = f64::NEG_INFINITY;
            }
            scores[cb * t + pos] = score;
        }
    }

    // Top-k positions by score; already-unmasked positions sit at -inf and
    // never win (k never exceeds the remaining mask count).
    let mut order: Vec<usize> = (0..n_cb * t).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    for &flat in order.iter().take(k) {
        let cb = flat / t;
        let pos = flat % t;
        tokens[cb][pos] = pred[cb][pos];
    }
}
