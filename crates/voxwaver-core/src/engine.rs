//! Reusable generation engine: the `generate` CLI pipeline (chunked
//! iterative prompting -> dual-AR sampling -> codec decode) exposed as a
//! resident model holder with progress callbacks and cancellation.

use anyhow::{bail, ensure, Context, Result};
use candle_core::{Device, DType, Tensor};
use std::path::{Path, PathBuf};
use std::time::Instant;

// Shared cancellation / progress types; re-exported so existing import paths
// (`voxwaver_core::engine::CancelFlag` etc.) keep working.
pub use tts_common::{CancelFlag, NullSink, Progress, ProgressSink};

use crate::config::{CodecConfig, DualArConfig};
use crate::dac::Dac;
use crate::dual_ar::DualArModel;
use crate::prof;
use crate::prompt;
use crate::sampling::{SampleParams, Sampler};
use crate::tokenizer;
use crate::wavio;
use crate::{rep_window_t, split_chunks};

#[derive(Clone)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    /// Selector strings resolved by the caller via [`crate::select_device`] /
    /// [`crate::select_dtype`] are preferred; the resolved values live here.
    pub device: Device,
    pub dtype: DType,
    /// Codec decode chunk size in frames (0 = decode at once).
    pub chunk_frames: usize,
    /// Per-chunk cap on generated frames (upstream `generate_long` passes
    /// `max_new_tokens` to `generate()` per batch). 0 = unlimited; each chunk
    /// is still capped by the context window.
    pub max_new_tokens: usize,
    /// Prefill slice size (replaces the CLI's PREFILL_CHUNK env var).
    pub prefill_chunk: usize,
    /// Keep the codec resident between runs (faster clone/decode at the
    /// cost of ~0.8 GB extra memory). When false the codec is dropped
    /// after each decode. Only the decode half is ever resident; the
    /// encode-only weights are loaded per `encode_reference` call.
    pub keep_codec_loaded: bool,
}

impl EngineConfig {
    pub fn new(model_dir: impl Into<PathBuf>, device: Device, dtype: DType) -> Self {
        Self {
            model_dir: model_dir.into(),
            device,
            dtype,
            chunk_frames: 256,
            max_new_tokens: 1024,
            prefill_chunk: 512,
            keep_codec_loaded: true,
        }
    }
}

/// A cloning reference: transcript + pre-encoded codec codes (`[10][T]`).
#[derive(Clone)]
pub struct RefTurn {
    pub text: String,
    pub codes: Vec<Vec<u32>>,
}

pub struct GenerateRequest {
    pub text: String,
    /// None = zero-shot default timbre.
    pub ref_turn: Option<RefTurn>,
    pub params: SampleParams,
    pub seed: u64,
}

pub struct GenerateOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The 10 codebook rows generated (same format as `generate --dump-codes`).
    pub codes: Vec<Vec<u32>>,
}

/// Resident holder for the tokenizer/config plus lazily loaded LM and codec.
///
/// The LM (model.pth, ~2 GB f32) stays resident across generations: reloading
/// takes seconds and would dominate short-form usage. The codec (codec.pth)
/// is loaded on demand — for reference encoding and for the final decode —
/// and optionally kept resident.
pub struct Engine {
    cfg: EngineConfig,
    codec_cfg: CodecConfig,
    tok: tokenizer::Tokenizer,
    cfg_ar: DualArConfig,
    lm: Option<DualArModel>,
    codec: Option<Dac>,
}

impl Engine {
    /// Load only the tokenizer + AR config (milliseconds). Weights are
    /// loaded lazily on first use.
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let tok = tokenizer::Tokenizer::load(&cfg.model_dir)?;
        let cfg_ar = DualArConfig::load(&cfg.model_dir.join("config.json"))?;
        Ok(Self {
            cfg,
            codec_cfg: CodecConfig::default(),
            tok,
            cfg_ar,
            lm: None,
            codec: None,
        })
    }

    /// Configuration fingerprint: any change unloads the resident weights.
    fn fingerprint(cfg: &EngineConfig) -> (PathBuf, String, DType) {
        (
            cfg.model_dir.clone(),
            format!("{:?}", cfg.device.location()),
            cfg.dtype,
        )
    }

    /// Swap configuration (device/precision/model dir). Returns true if the
    /// resident weights were invalidated and will reload on next use.
    pub fn reconfigure(&mut self, cfg: EngineConfig) -> bool {
        let changed = Self::fingerprint(&self.cfg) != Self::fingerprint(&cfg);
        if changed {
            self.unload();
        }
        let fp_changed = changed;
        self.cfg = cfg;
        fp_changed
    }

    /// Drop all resident weights (engine stays usable, reloads on demand).
    pub fn unload(&mut self) {
        self.lm = None;
        self.codec = None;
        let _ = self.cfg.device.synchronize();
    }

    /// Eagerly load the LM (and codec when `keep_codec_loaded`) so the first
    /// generation doesn't pay the load cost. Idempotent.
    pub fn warmup(&mut self) {
        let _ = self.ensure_lm(&NullSink);
        if self.cfg.keep_codec_loaded {
            let _ = self.ensure_codec(&NullSink);
        }
    }

    pub fn is_lm_loaded(&self) -> bool {
        self.lm.is_some()
    }

    pub fn is_codec_loaded(&self) -> bool {
        self.codec.is_some()
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn sample_rate(&self) -> u32 {
        self.codec_cfg.sample_rate
    }

    fn ensure_lm(&mut self, sink: &dyn ProgressSink) -> Result<&mut DualArModel> {
        if self.lm.is_none() {
            sink.progress(Progress::LoadingLm);
            let t0 = Instant::now();
            let lm = DualArModel::load(
                &self.cfg.model_dir.join("model.pth"),
                &self.cfg_ar,
                self.tok.semantic_begin,
                self.tok.semantic_end,
                self.cfg.dtype,
                &self.cfg.device,
            )?;
            sink.log(&format!("model loaded in {:.1}s", t0.elapsed().as_secs_f64()));
            self.lm = Some(lm);
        }
        Ok(self.lm.as_mut().unwrap())
    }

    fn ensure_codec(&mut self, sink: &dyn ProgressSink) -> Result<&mut Dac> {
        if self.codec.is_none() {
            sink.progress(Progress::LoadingCodec);
            let t0 = Instant::now();
            let codec = Dac::load(
                &self.cfg.model_dir.join("codec.pth"),
                &self.codec_cfg,
                &self.cfg.device,
            )?;
            sink.log(&format!("codec loaded in {:.1}s", t0.elapsed().as_secs_f64()));
            self.codec = Some(codec);
        }
        Ok(self.codec.as_mut().unwrap())
    }

    /// Encode a reference WAV (any common sample rate) into codec codes for
    /// voice cloning. Only touches the codec, not the LM.
    pub fn encode_reference(&mut self, wav_path: &Path, sink: &dyn ProgressSink) -> Result<Vec<Vec<u32>>> {
        let (samples, sr) = wavio::read_wav_mono(wav_path)?;
        let samples = wavio::resample(&samples, sr, self.codec_cfg.sample_rate);
        sink.progress(Progress::EncodingRef {
            seconds: samples.len() as f32 / self.codec_cfg.sample_rate as f32,
        });
        let t0 = Instant::now();
        let codec = self.ensure_codec(sink)?;
        let codes = codec.encode(&samples).context("encoding reference audio")?;
        sink.log(&format!(
            "encoded reference: {} frames (encode {:.1}s)",
            codes[0].len(),
            t0.elapsed().as_secs_f64()
        ));
        Ok(codes)
    }

    /// The full generation pipeline. Synchronous and blocking — the caller
    /// is responsible for running it off the async runtime / UI thread.
    pub fn generate(
        &mut self,
        req: &GenerateRequest,
        cancel: &CancelFlag,
        sink: &dyn ProgressSink,
    ) -> Result<GenerateOutput> {
        let dev = self.cfg.device.clone();
        let dtype = self.cfg.dtype;
        // clone the small configs: `ensure_lm` takes &mut self below, so a
        // long-lived borrow of self.cfg_ar would not borrow-check
        let codec_cfg = self.codec_cfg;
        let cfg = self.cfg_ar.clone();
        sink.log(&format!("device={:?} dtype={:?}", dev.location(), dtype));

        // ---- text chunking ----
        // Upstream generate_long never feeds a whole article as one prompt:
        // text is split into ~300-byte batches and each batch is generated
        // to <|im_end|>, with the generated codes fed back as prior turns
        // (iterative prompting). Long single prompts measurably degrade
        // s1-mini's output.
        let chunks = split_chunks(&req.text, 300);
        sink.progress(Progress::Chunks { total: chunks.len() });
        sink.log(&format!("text split into {} chunks", chunks.len()));
        let mut history: Vec<(String, Vec<Vec<u32>>)> = match &req.ref_turn {
            Some(r) => vec![(r.text.clone(), r.codes.clone())],
            None => Vec::new(),
        };

        // ---- model ----
        let model = self.ensure_lm(sink)?;
        let _ = model; // borrow ends here; re-take per chunk below

        let params = req.params;
        // seed the device RNG (Metal tausworthe/lcg) used by the GPU sampling
        // chain so runs are reproducible on the same device
        dev.set_seed(req.seed)?;
        let sampler = Sampler::new(req.seed, dtype == DType::BF16);

        let prefill_chunk = self.cfg.prefill_chunk.max(1);
        let rows = cfg.num_codebooks + 1;
        let t_total = Instant::now();
        let mut gen_codes: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_codebooks];
        // global generated-frame counter (across text chunks) for prof reports
        let mut prof_step: u64 = 0;

        for (ci, chunk_text) in chunks.iter().enumerate() {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            // rebuild the prompt with the generated turns fed back; drop the
            // oldest generated turns if the conversation outgrows the context
            let (p, prompt_len) = loop {
                let hist: Vec<prompt::Turn> = history
                    .iter()
                    .map(|(t, c)| prompt::Turn { text: t, codes: c })
                    .collect();
                match prompt::build(
                    &mut self.tok,
                    chunk_text,
                    &hist,
                    cfg.max_seq_len,
                    cfg.num_codebooks,
                ) {
                    Ok(p) => {
                        let len = p.len;
                        break (p, len);
                    }
                    Err(e) => {
                        // keep the reference turn (index 0) if present
                        let droppable = history.len() > usize::from(req.ref_turn.is_some());
                        if droppable {
                            let dropped =
                                history.remove(usize::from(req.ref_turn.is_some()));
                            sink.log(&format!(
                                "context full: dropping oldest generated turn ({} frames)",
                                dropped.1[0].len()
                            ));
                        } else {
                            return Err(e);
                        }
                    }
                }
            };
            let ctx_room = cfg.max_seq_len.saturating_sub(prompt_len + 1);
            let max_new = match self.cfg.max_new_tokens {
                0 => ctx_room,
                cap => cap.min(ctx_room),
            };
            ensure!(max_new > 0, "prompt fills the context window");
            self.ensure_lm(sink)?.setup_caches(prompt_len, max_new)?;

            // flat values, row-major [11, T]: row 0 tokens, rows 1.. codes
            let mut values: Vec<u32> = Vec::with_capacity(prompt_len * rows);
            values.extend_from_slice(&p.tokens);
            for row in &p.codes {
                values.extend_from_slice(row);
            }

            let t0 = Instant::now();
            let (mut logits, mut hidden) = (None, None);
            for pos0 in (0..prompt_len).step_by(prefill_chunk) {
                if cancel.is_cancelled() {
                    bail!("cancelled");
                }
                let len = prefill_chunk.min(prompt_len - pos0);
                let slice: Vec<u32> = (0..rows)
                    .flat_map(|row| {
                        let base = row * prompt_len + pos0;
                        values[base..base + len].iter().copied()
                    })
                    .collect();
                let slice = Tensor::from_vec(slice, (rows, len), &dev)?;
                let model = self.ensure_lm(sink)?;
                let (l, h) = model.forward(&slice, pos0)?;
                logits = Some(l);
                hidden = Some(h);
                dev.synchronize()?;
            }
            let (mut logits, mut hidden) = (logits.unwrap(), hidden.unwrap());
            sink.log(&format!(
                "chunk {}/{}: prefill {} tokens in {:.1}s",
                ci + 1,
                chunks.len(),
                prompt_len,
                t0.elapsed().as_secs_f64()
            ));
            sink.progress(Progress::Prefill {
                chunk: ci + 1,
                total: chunks.len(),
                tokens: prompt_len,
            });

            // ---- generation loop ----
            let mut pos = prompt_len;
            let mut chunk_codes: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_codebooks];
            // per-row repetition-penalty history (main token row + fast
            // codebook rows), mirroring `previous_tokens` in `decode_n_tokens`
            let mut main_hist: Vec<u32> = Vec::new();
            let mut cb_hist: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_codebooks];
            let t0 = Instant::now();

            for step in 0..max_new {
                // GPU sampling chain: submit the sampler kernels, then a single
                // blocking scalar read for the token id (the im_end control
                // flow and the history bookkeeping need it on the CPU)
                let window = rep_window_t(&main_hist, &dev)?;
                let token_t = {
                    let _g = prof::scope(prof::MAIN_SAMPLE);
                    sampler.sample_gpu(&logits, &params, Some(&window))?
                };
                let token = {
                    let _g = prof::scope(prof::MAIN_READ);
                    token_t.to_vec1::<u32>()?[0]
                };
                if token == self.tok.im_end {
                    sink.log(&format!(
                        "chunk {} done (<|im_end|>) after {step} frames",
                        ci + 1
                    ));
                    break;
                }
                if step > 0 {
                    // upstream `previous_tokens` only records tokens drawn
                    // inside decode_n_tokens; the prefill sample (step 0)
                    // never enters it
                    main_hist.push(token);
                }
                let sem_code = token - self.tok.semantic_begin;

                let cb_windows: Vec<Option<Tensor>> = (1..cfg.num_codebooks)
                    .map(|k| rep_window_t(&cb_hist[k], &dev).map(Some))
                    .collect::<Result<_>>()?;
                let model = self.ensure_lm(sink)?;
                let codes = model.fast_frame(&hidden, sem_code, &sampler, &params, &cb_windows)?;
                for ((cb_gen, cb_win), &c) in chunk_codes
                    .iter_mut()
                    .zip(cb_hist.iter_mut())
                    .zip(&codes)
                {
                    cb_gen.push(c);
                    if step > 0 {
                        cb_win.push(c);
                    }
                }

                if step + 1 == max_new {
                    break;
                }
                if cancel.is_cancelled() {
                    bail!("cancelled");
                }
                // feed [token, codes...] back
                let mut col = Vec::with_capacity(cfg.num_codebooks + 1);
                col.push(token);
                col.extend_from_slice(&codes);
                let col = Tensor::from_vec(col, (cfg.num_codebooks + 1, 1), &dev)?;
                let (new_logits, new_hidden) = {
                    let _g = prof::scope(prof::MAIN_FWD);
                    let model = self.ensure_lm(sink)?;
                    model.forward(&col, pos)?
                };
                logits = new_logits;
                hidden = new_hidden;
                pos += 1;

                prof_step += 1;
                prof::maybe_report(prof_step, false);
                // Periodic full synchronize: the Metal cross-encoder fence map
                // (untracked-hazard ordering) and the buffer pool only get
                // cleaned on `synchronize()`; without this, command encoding
                // slows down monotonically over the decode loop (every new
                // encoder waits on more stale fences). The GPU is already
                // drained by the per-frame readbacks, so the extra wait is
                // ~free.
                if dev.is_metal() && prof_step % 25 == 0 {
                    dev.synchronize()?;
                }
                if step % 25 == 0 {
                    let fps = (step + 1) as f32 / t0.elapsed().as_secs_f32().max(1e-9);
                    sink.log(&format!(
                        "chunk {}/{} frame {step}/{max_new} ({fps:.1} frames/s)",
                        ci + 1,
                        chunks.len()
                    ));
                    sink.progress(Progress::Generating {
                        chunk: ci + 1,
                        total: chunks.len(),
                        frames: step + 1,
                        max_frames: max_new,
                        fps,
                    });
                }
                if pos >= cfg.max_seq_len {
                    sink.log("context limit reached");
                    break;
                }
            }

            prof::maybe_report(prof_step, true);

            let chunk_frames = chunk_codes[0].len();
            ensure!(chunk_frames > 0, "no frames generated for chunk {}", ci + 1);
            sink.log(&format!(
                "chunk {}/{}: {} frames ({:.1}s audio) in {:.1}s",
                ci + 1,
                chunks.len(),
                chunk_frames,
                chunk_frames as f64 * codec_cfg.frame_length() as f64
                    / codec_cfg.sample_rate as f64,
                t0.elapsed().as_secs_f64()
            ));
            for (g, c) in gen_codes.iter_mut().zip(chunk_codes.iter()) {
                g.extend_from_slice(c);
            }
            history.push((chunk_text.clone(), chunk_codes));
        }

        let frames = gen_codes[0].len();
        ensure!(frames > 0, "no frames generated");
        sink.log(&format!(
            "generated {} frames ({:.1}s audio) in {:.1}s",
            frames,
            frames as f64 * codec_cfg.frame_length() as f64 / codec_cfg.sample_rate as f64,
            t_total.elapsed().as_secs_f64()
        ));

        // ---- codec decode ----
        let t0 = Instant::now();
        // Free the KV caches before the memory-hungry codec decode: they are
        // dead weight here and re-allocated by the next generation. The
        // synchronize also sweeps the Metal buffer pool of generation
        // intermediates.
        if let Some(lm) = self.lm.as_mut() {
            lm.release_caches();
        }
        self.cfg.device.synchronize()?;
        let chunk_frames = self.cfg.chunk_frames;
        let audio = {
            let codec = self.ensure_codec(sink)?;
            codec.decode_codes_chunked(&gen_codes, chunk_frames, Some(cancel))?
        };
        if !self.cfg.keep_codec_loaded {
            self.codec = None;
        }
        // Hand every idle pooled buffer back to the OS so a long-running app
        // doesn't sit on decode-peak memory between generations.
        self.cfg.device.clear_metal_pool()?;
        sink.log(&format!(
            "decoded {} samples in {:.1}s",
            audio.len(),
            t0.elapsed().as_secs_f64()
        ));
        sink.progress(Progress::Decoding { chunk: 1, total: 1, frames_total: frames });

        Ok(GenerateOutput {
            samples: audio,
            sample_rate: self.codec_cfg.sample_rate,
            codes: gen_codes,
        })
    }
}
