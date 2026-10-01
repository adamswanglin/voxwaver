//! App-level engine facade: dispatches to the s1-mini (voxwaver-core) or
//! OmniVoice (omnivoice crate) engine behind one enum, so the rest of the
//! app never touches a concrete engine type.

use anyhow::{Context, Result};
use std::path::Path;
use tts_common::{CancelFlag, Progress, ProgressSink};
use voxwaver_core::engine::{Engine as S1Engine, EngineConfig, GenerateRequest, RefTurn};
use voxwaver_core::sampling::SampleParams;
use voxwaver_core::{select_device, select_dtype};

use crate::models::{self, ModelKind};
use crate::store::Settings;

/// s1-mini codec config is a compile-time constant.
const S1_SAMPLE_RATE: u32 = 44_100;

/// One voice-cloning reference: transcript + model-specific codes
/// (`[10][T]` for s1-mini, `[8][T]` for OmniVoice).
pub struct RefMat {
    pub transcript: String,
    pub codes: Vec<Vec<u32>>,
}

/// A generation request, model-agnostic.
pub struct GenJob {
    pub text: String,
    pub instruct: Option<String>,
    /// s1-mini sampling params.
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
    /// OmniVoice class temperature (0 = greedy).
    pub omni_temperature: f64,
    /// Language tag for OmniVoice (e.g. "zh", "en").
    pub lang: String,
    pub seed: u64,
}

pub struct GenAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// OmniVoice resident engine plus the device selector it was built with
/// (the engine itself doesn't track its config; the app needs to know when
/// a rebuild is required).
pub struct OmniResident {
    pub engine: omnivoice::engine::Engine,
    pub device: String,
}

pub enum AnyEngine {
    S1(S1Engine),
    Omni(OmniResident),
}

impl AnyEngine {
    pub fn kind(&self) -> ModelKind {
        match self {
            AnyEngine::S1(_) => ModelKind::S1Mini,
            AnyEngine::Omni(_) => ModelKind::OmniVoice,
        }
    }

    pub fn is_model_loaded(&self) -> bool {
        match self {
            AnyEngine::S1(e) => e.is_lm_loaded(),
            AnyEngine::Omni(_) => true, // eager-loaded at build
        }
    }

    /// Build the engine for the active model in `settings`. Cheap for s1-mini
    /// (weights load lazily on first use); loads everything for OmniVoice —
    /// run on a blocking thread with a progress sink.
    pub fn build(settings: &Settings, sink: &dyn ProgressSink) -> Result<Self> {
        match settings.kind() {
            ModelKind::S1Mini => Ok(AnyEngine::S1(S1Engine::new(s1_config(settings)?)?)),
            ModelKind::OmniVoice => {
                let dir = settings
                    .model_dir_of(ModelKind::OmniVoice)
                    .map(std::path::PathBuf::from)
                    .context("no OmniVoice model directory configured")?;
                let missing = models::check_model_dir(ModelKind::OmniVoice, &dir)?;
                anyhow::ensure!(
                    missing.is_empty(),
                    "OmniVoice model directory missing files: {}",
                    missing.join(", ")
                );
                let device = select_device(&settings.device)?;
                let engine = omnivoice::engine::Engine::load_on_with(&dir, &device, sink)?;
                Ok(AnyEngine::Omni(OmniResident {
                    engine,
                    device: settings.device.clone(),
                }))
            }
        }
    }

    /// Eagerly load weights so the first generation doesn't pay the cost.
    pub fn warmup(&mut self) {
        if let AnyEngine::S1(e) = self {
            e.warmup();
        }
    }

    /// The full pipeline. Synchronous and blocking.
    pub fn generate(
        &mut self,
        job: &GenJob,
        ref_mat: Option<&RefMat>,
        cancel: &CancelFlag,
        sink: &dyn ProgressSink,
    ) -> Result<GenAudio> {
        match self {
            AnyEngine::S1(e) => {
                let req = GenerateRequest {
                    text: job.text.clone(),
                    ref_turn: ref_mat.map(|r| RefTurn {
                        text: r.transcript.clone(),
                        codes: r.codes.clone(),
                    }),
                    params: SampleParams {
                        temperature: job.temperature,
                        top_p: job.top_p,
                        repetition_penalty: job.repetition_penalty,
                    },
                    seed: job.seed,
                };
                let out = e.generate(&req, cancel, sink)?;
                Ok(GenAudio {
                    samples: out.samples,
                    sample_rate: out.sample_rate,
                })
            }
            AnyEngine::Omni(r) => {
                let mut params = omnivoice::generator::GenParams::default();
                params.class_temperature = job.omni_temperature;
                let (ref_codes, ref_text) = match ref_mat {
                    Some(r) => (Some(r.codes.as_slice()), Some(r.transcript.as_str())),
                    None => (None, None),
                };
                let wave = r.engine.tts_with(
                    &job.text,
                    Some(&job.lang),
                    job.instruct.as_deref(),
                    Some(job.seed),
                    &params,
                    ref_codes,
                    ref_text,
                    cancel,
                    sink,
                )?;
                Ok(GenAudio {
                    samples: wave,
                    sample_rate: omnivoice::config::SAMPLE_RATE,
                })
            }
        }
    }

    /// Encode a reference WAV (any common sample rate) into codes for this
    /// engine's model. `sink` gets an `EncodingRef` progress event.
    pub fn encode_reference(
        &mut self,
        wav_path: &Path,
        sink: &dyn ProgressSink,
    ) -> Result<Vec<Vec<u32>>> {
        match self {
            AnyEngine::S1(e) => e.encode_reference(wav_path, sink),
            AnyEngine::Omni(r) => {
                let (samples, sr) = omnivoice::wavio::read_wav(wav_path)?;
                sink.progress(Progress::EncodingRef {
                    seconds: samples.len() as f32 / sr as f32,
                });
                r.engine.encode_ref(&samples, sr)
            }
        }
    }

    /// This engine's output sample rate.
    pub fn sample_rate(&self) -> u32 {
        match self {
            AnyEngine::S1(_) => S1_SAMPLE_RATE,
            AnyEngine::Omni(_) => omnivoice::config::SAMPLE_RATE,
        }
    }
}

/// Resolve settings into an s1-mini engine config (device/dtype/model dir).
pub fn s1_config(settings: &Settings) -> Result<EngineConfig> {
    let dir = settings
        .model_dir_of(ModelKind::S1Mini)
        .map(std::path::PathBuf::from)
        .context("no s1-mini model directory configured")?;
    let missing = models::check_model_dir(ModelKind::S1Mini, &dir)?;
    anyhow::ensure!(
        missing.is_empty(),
        "model directory missing files: {}",
        missing.join(", ")
    );
    let dev = select_device(&settings.device)?;
    let dtype = select_dtype(&settings.dtype, &dev)?;
    let mut cfg = EngineConfig::new(&dir, dev, dtype);
    cfg.keep_codec_loaded = settings.keep_codec_loaded;
    Ok(cfg)
}
