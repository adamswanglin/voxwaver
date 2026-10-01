//! App-level engine facade over the OmniVoice (omnivoice crate) engine,
//! so the rest of the app never touches the concrete engine type.

use anyhow::{Context, Result};
use std::path::Path;
use tts_common::{CancelFlag, Progress, ProgressSink};

use crate::models;
use crate::store::Settings;

/// One voice-cloning reference: transcript + `[8][T]` audio codes.
pub struct RefMat {
    pub transcript: String,
    pub codes: Vec<Vec<u32>>,
}

/// A generation request.
pub struct GenJob {
    pub text: String,
    pub instruct: Option<String>,
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
        let mut params = omnivoice::generator::GenParams::default();
        params.class_temperature = job.omni_temperature;
        let (ref_codes, ref_text) = match ref_mat {
            Some(r) => (Some(r.codes.as_slice()), Some(r.transcript.as_str())),
            None => (None, None),
        };
        let wave = self.engine.tts_with(
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
        Ok(GenAudio { samples: wave, sample_rate: omnivoice::config::SAMPLE_RATE })
    }

    /// Encode a reference WAV (any common sample rate) into audio codes.
    /// `sink` gets an `EncodingRef` progress event.
    pub fn encode_reference(
        &mut self,
        wav_path: &Path,
        sink: &dyn ProgressSink,
    ) -> Result<Vec<Vec<u32>>> {
        let (samples, sr) = omnivoice::wavio::read_wav(wav_path)?;
        sink.progress(Progress::EncodingRef {
            seconds: samples.len() as f32 / sr as f32,
        });
        self.engine.encode_ref(&samples, sr)
    }

    /// This engine's output sample rate.
    pub fn sample_rate(&self) -> u32 {
        omnivoice::config::SAMPLE_RATE
    }
}
