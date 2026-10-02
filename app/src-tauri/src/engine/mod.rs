//! App-level engine facade over the OmniVoice (omnivoice crate) engine,
//! so the rest of the app never touches the concrete engine type.

use anyhow::{Context, Result};
use std::path::Path;
use tts_common::{CancelFlag, Progress, ProgressSink};

use crate::models;
use crate::store::Settings;

/// One voice-cloning reference: transcript + `[8][T]` audio codes + the
/// reference waveform's RMS (for output volume matching).
pub struct RefMat {
    pub transcript: String,
    pub codes: Vec<Vec<u32>>,
    pub rms: Option<f64>,
}

/// A generation request.
pub struct GenJob {
    pub text: String,
    pub instruct: Option<String>,
    /// Language tag for OmniVoice (e.g. "zh", "en").
    pub lang: String,
    pub seed: u64,
    /// Per-request generation-parameter overrides; `None` fields keep the
    /// engine defaults.
    pub overrides: GenOverrides,
}

/// Optional overrides for one generation (mirror of the Python
/// `OmniVoiceGenerationConfig` fields the UI exposes). Every field is
/// `None` = engine default.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GenOverrides {
    // Duration & speed
    /// Speaking-speed factor (> 1 = faster).
    pub speed: Option<f64>,
    /// Fixed output duration in seconds; overrides `speed` when set.
    pub duration: Option<f64>,
    // Decoding
    pub num_step: Option<usize>,
    pub guidance_scale: Option<f64>,
    pub t_shift: Option<f64>,
    /// Prepend the `<|denoise|>` tag.
    pub denoise: Option<bool>,
    // Sampling
    pub position_temperature: Option<f64>,
    pub class_temperature: Option<f64>,
    pub layer_penalty_factor: Option<f64>,
    // Pre/post processing
    pub postprocess_output: Option<bool>,
    pub pad_duration: Option<f64>,
    pub fade_duration: Option<f64>,
    // Long-form generation
    pub audio_chunk_duration: Option<f64>,
    pub audio_chunk_threshold: Option<f64>,
}

pub struct GenAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Resident OmniVoice engine plus the device selector it was built with
/// (the engine itself doesn't track its config; the app needs to know when
/// a rebuild is required).
pub struct Engine {
    pub engine: omnivoice::engine::Engine,
    pub device: String,
}

impl Engine {
    /// Load the OmniVoice engine for the active model in `settings`.
    /// Loads everything eagerly — run on a blocking thread with a progress
    /// sink.
    pub fn build(settings: &Settings, sink: &dyn ProgressSink) -> Result<Self> {
        let dir = settings
            .model_dir_of(crate::models::ModelKind::OmniVoice)
            .map(std::path::PathBuf::from)
            .context("no OmniVoice model directory configured")?;
        let missing = models::check_model_dir(crate::models::ModelKind::OmniVoice, &dir)?;
        anyhow::ensure!(
            missing.is_empty(),
            "OmniVoice model directory missing files: {}",
            missing.join(", ")
        );
        let device = omnivoice::select_device(&settings.device)?;
        let engine = omnivoice::engine::Engine::load_on_with(&dir, &device, sink)?;
        Ok(Engine { engine, device: settings.device.clone() })
    }

    /// The full pipeline. Synchronous and blocking.
    pub fn generate(
        &mut self,
        job: &GenJob,
        ref_mat: Option<&RefMat>,
        cancel: &CancelFlag,
        sink: &dyn ProgressSink,
    ) -> Result<GenAudio> {
        // Unmasking hyper-parameters start from the engine defaults
        // (greedy decode) and take the request's overrides on top.
        let ovr = &job.overrides;
        let params = {
            let mut p = omnivoice::generator::GenParams::default();
            if let Some(v) = ovr.num_step {
                // `get_time_steps` divides by num_step - 1; keep >= 2.
                p.num_step = v.max(2);
            }
            if let Some(v) = ovr.guidance_scale {
                p.guidance_scale = v.max(0.0);
            }
            if let Some(v) = ovr.t_shift {
                // t_shift = 0 hits 0/0 at s = 1 in the shifted schedule; keep
                // a small positive epsilon.
                p.t_shift = v.clamp(1e-3, 1.0);
            }
            if let Some(v) = ovr.position_temperature {
                p.position_temperature = v.max(0.0);
            }
            if let Some(v) = ovr.class_temperature {
                p.class_temperature = v.max(0.0);
            }
            if let Some(v) = ovr.layer_penalty_factor {
                p.layer_penalty_factor = v.max(0.0);
            }
            p
        };
        let (ref_codes, ref_text) = match ref_mat {
            Some(r) => (Some(r.codes.as_slice()), Some(r.transcript.as_str())),
            None => (None, None),
        };
        let mut opts = omnivoice::engine::SpeakOptions {
            ref_rms: ref_mat.and_then(|r| r.rms),
            ..Default::default()
        };
        if let Some(s) = ovr.speed {
            opts.speed = s.max(0.1);
        }
        if let Some(d) = ovr.duration {
            opts.duration = Some(d.max(0.0));
        }
        if let Some(d) = ovr.denoise {
            opts.denoise = d;
        }
        if let Some(v) = ovr.postprocess_output {
            opts.postprocess_output = v;
        }
        if let Some(v) = ovr.pad_duration {
            opts.pad_duration = v.max(0.0);
        }
        if let Some(v) = ovr.fade_duration {
            opts.fade_duration = v.max(0.0);
        }
        if let Some(v) = ovr.audio_chunk_duration {
            opts.audio_chunk_duration = v.max(0.0);
        }
        if let Some(v) = ovr.audio_chunk_threshold {
            opts.audio_chunk_threshold = v.max(0.0);
        }
        let wave = self.engine.tts_with(
            &job.text,
            Some(&job.lang),
            job.instruct.as_deref(),
            Some(job.seed),
            &params,
            ref_codes,
            ref_text,
            &opts,
            cancel,
            sink,
        )?;
        Ok(GenAudio { samples: wave, sample_rate: omnivoice::config::SAMPLE_RATE })
    }

    /// Encode a reference WAV (any common sample rate) into audio codes,
    /// together with the waveform's RMS for volume matching.
    /// `sink` gets an `EncodingRef` progress event.
    pub fn encode_reference(
        &mut self,
        wav_path: &Path,
        sink: &dyn ProgressSink,
    ) -> Result<(Vec<Vec<u32>>, f64)> {
        let (samples, sr) = omnivoice::wavio::read_wav(wav_path)?;
        sink.progress(Progress::EncodingRef {
            seconds: samples.len() as f32 / sr as f32,
        });
        let rms = omnivoice::engine::ref_rms(&samples, sr);
        let codes = self.engine.encode_ref(&samples, sr)?;
        Ok((codes, rms))
    }

    /// This engine's output sample rate.
    pub fn sample_rate(&self) -> u32 {
        omnivoice::config::SAMPLE_RATE
    }
}
