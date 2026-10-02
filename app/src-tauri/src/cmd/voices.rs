//! Voice library: list / create (encode reference) / delete.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager, State};

use tts_common::wavio;

use crate::cmd::tts::{now_ms, ref_cache_file, voice_name, ChannelSink, Fwd, RefFile};
use crate::state::{AppCtx, AppState};
use crate::store::{self, Voice};

/// Transcript lives inside the code cache; legacy voices (pre-OmniVoice
/// builds) kept it in `ref.json`, so read that as a fallback.
fn read_transcript(dir: &std::path::Path) -> Option<String> {
    ["ref.omni.json", "ref.json"]
        .iter()
        .find_map(|f| store::read_json::<RefFile>(&dir.join(f)).ok())
        .map(|rf| rf.transcript)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceView {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    /// optional emoji icon (null = name-letter fallback on the frontend)
    pub icon: Option<String>,
    pub created_at: i64,
    /// cloned voice — the library only stores clones (no built-in timbres)
    pub is_clone: bool,
    /// playable reference sample, if any
    pub ref_wav: Option<String>,
    /// reference transcript, cloned voices only
    pub transcript: Option<String>,
    /// reference sample duration in seconds
    pub sample_seconds: Option<f64>,
}

/// Trim a client-supplied icon; blank/absent means "no icon".
fn clean_icon(icon: Option<&str>) -> Option<String> {
    let icon = icon?.trim();
    (!icon.is_empty()).then(|| icon.to_string())
}

impl Voice {
    fn into_view(self, ctx: &AppCtx) -> VoiceView {
        let dir = ctx.dirs().voice_dir(&self.id);
        let transcript = read_transcript(&dir);
        VoiceView {
            ref_wav: dir.join("ref.wav").to_str().map(|s| s.to_string()),
            transcript,
            sample_seconds: Some(self.sample_seconds),
            is_clone: true,
            id: self.id,
            name: self.name,
            tags: self.tags,
            icon: self.icon,
            created_at: self.created_at,
        }
    }
}

#[tauri::command]
pub fn list_voices(app: AppHandle) -> Vec<VoiceView> {
    let ctx = app.state::<AppCtx>();
    let idx = ctx.dirs().voices_index();
    let mut views = Vec::new();
    for v in store::load_voices(&idx) {
        let dir = ctx.dirs().voice_dir(&v.id);
        let ref_wav = dir.join("ref.wav").to_str().map(|s| s.to_string());
        let transcript = read_transcript(&dir);
        views.push(VoiceView {
            id: v.id,
            name: v.name,
            tags: v.tags,
            icon: v.icon,
            created_at: v.created_at,
            is_clone: true,
            ref_wav,
            transcript,
            sample_seconds: Some(v.sample_seconds),
        });
    }
    views
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateVoiceReq {
    pub name: String,
    /// optional emoji icon; empty string = none
    #[serde(default)]
    pub icon: Option<String>,
    /// absolute path to the sample WAV (chosen via dialog on the frontend)
    pub sample_path: String,
    pub transcript: String,
}

/// Encode `sample` as reference codes and persist codes + transcript +
/// audition copy into `dir`, streaming encoder logs to the UI. Returns the
/// sample duration in seconds.
fn encode_ref_into(
    app: &AppHandle,
    dir: &std::path::Path,
    sample: &std::path::Path,
    transcript: &str,
) -> anyhow::Result<f64> {
    let (tx, rx) = std::sync::mpsc::channel::<Fwd>();
    let app_fwd = app.clone();
    let fwd = std::thread::spawn(move || {
        while let Ok(f) = rx.recv() {
            if let Fwd::Log(m) = f {
                let _ = app_fwd.emit("voice://encoding", m);
            }
        }
    });
    let res = (|| -> anyhow::Result<f64> {
        let sink = ChannelSink { tx };
        let state: State<'_, AppState> = app.state();
        // engine must exist (it encodes the reference)
        crate::cmd::tts::ensure_engine(&state, &sink)?;
        let (codes, _) = {
            let mut eng = state.engine.lock().unwrap();
            let eng = eng.as_mut().expect("engine ensured above");
            eng.encode_reference(sample, &sink)?
        };
        std::fs::create_dir_all(dir)?;
        // keep a 24k pcm16 copy of the reference for auditioning (also the
        // lazy-encoding source when the cache is missing)
        let rate = omnivoice::config::SAMPLE_RATE;
        let (samples, sr) = wavio::read_wav_mono(sample)?;
        let samples = wavio::resample(&samples, sr, rate);
        wavio::write_wav(&dir.join("ref.wav"), &samples, rate, true)?;
        let rms = (samples.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
            / samples.len().max(1) as f64)
            .sqrt();
        store::write_json(
            &dir.join(ref_cache_file()),
            &RefFile {
                transcript: transcript.to_string(),
                codes,
                rms: Some(rms),
            },
        )?;
        Ok(samples.len() as f64 / rate as f64)
    })();
    // dropping `tx` closed the channel; drain remaining events
    let _ = fwd.join();
    res
}

#[tauri::command]
pub async fn create_voice(
    app: AppHandle,
    req: CreateVoiceReq,
) -> Result<VoiceView, String> {
    if req.name.trim().is_empty() {
        return Err(rust_i18n::t!("nameEmpty").into());
    }
    if req.transcript.trim().is_empty() {
        return Err(rust_i18n::t!("transcriptEmpty").into());
    }
    let sample = PathBuf::from(&req.sample_path);
    if !sample.is_file() {
        return Err(rust_i18n::t!("sampleMissing").into());
    }

    let app_blk = app.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<VoiceView> {
        let ctx = app_blk.state::<AppCtx>();
        let id = uuid::Uuid::new_v4().to_string();
        let dir = ctx.dirs().voice_dir(&id);
        let secs = encode_ref_into(&app_blk, &dir, &sample, req.transcript.trim())?;
        let voice = Voice {
            id: id.clone(),
            name: req.name.trim().to_string(),
            icon: clean_icon(req.icon.as_deref()),
            tags: vec!["clone".into()],
            created_at: now_ms(),
            sample_seconds: secs,
        };
        let mut voices = store::load_voices(&ctx.dirs().voices_index());
        voices.push(voice.clone());
        store::save_voices(&ctx.dirs().voices_index(), &voices)?;
        Ok(voice.into_view(&ctx))
    })
    .await
    .map_err(|e| format!("encoding task panicked: {e}"))?
    .map_err(|e| format!("{e:#}"))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateVoiceReq {
    pub id: String,
    pub name: String,
    /// optional emoji icon; empty string = clear (name-letter fallback)
    #[serde(default)]
    pub icon: Option<String>,
    pub transcript: String,
    /// new reference sample; absent = keep the existing one (transcript only)
    pub sample_path: Option<String>,
}

/// Edit an existing cloned voice: metadata + transcript always, reference
/// audio re-encoded only when a new sample is given.
#[tauri::command]
pub async fn update_voice(
    app: AppHandle,
    req: UpdateVoiceReq,
) -> Result<VoiceView, String> {
    if req.name.trim().is_empty() {
        return Err(rust_i18n::t!("nameEmpty").into());
    }
    if req.transcript.trim().is_empty() {
        return Err(rust_i18n::t!("transcriptEmpty").into());
    }
    let sample = match &req.sample_path {
        Some(p) if !p.trim().is_empty() => {
            let p = PathBuf::from(p);
            if !p.is_file() {
                return Err(rust_i18n::t!("sampleMissing").into());
            }
            Some(p)
        }
        _ => None,
    };

    let app_blk = app.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<VoiceView> {
        let ctx = app_blk.state::<AppCtx>();
        let idx = ctx.dirs().voices_index();
        let mut voices = store::load_voices(&idx);
        let voice = voices
            .iter_mut()
            .find(|v| v.id == req.id)
            .ok_or_else(|| anyhow::anyhow!("{}", rust_i18n::t!("voiceNotFound")))?;
        voice.name = req.name.trim().to_string();
        voice.icon = clean_icon(req.icon.as_deref());
        let dir = ctx.dirs().voice_dir(&req.id);
        if let Some(sample) = &sample {
            voice.sample_seconds =
                encode_ref_into(&app_blk, &dir, sample, req.transcript.trim())?;
        } else {
            // transcript may still have been edited: rewrite it into the
            // code cache (and the legacy file, when present)
            for f in ["ref.omni.json", "ref.json"] {
                let path = dir.join(f);
                if let Ok(mut rf) = store::read_json::<RefFile>(&path) {
                    rf.transcript = req.transcript.trim().to_string();
                    store::write_json(&path, &rf)?;
                }
            }
        }
        let view = voice.clone().into_view(&ctx);
        store::save_voices(&idx, &voices)?;
        Ok(view)
    })
    .await
    .map_err(|e| format!("encoding task panicked: {e}"))?
    .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn delete_voice(app: AppHandle, id: String) -> Result<(), String> {
    let ctx = app.state::<AppCtx>();
    let idx = ctx.dirs().voices_index();
    let mut voices = store::load_voices(&idx);
    let before = voices.len();
    voices.retain(|v| v.id != id);
    if voices.len() == before {
        return Err(rust_i18n::t!("voiceNotFound").into());
    }
    store::save_voices(&idx, &voices).map_err(|e| e.to_string())?;
    let dir = ctx.dirs().voice_dir(&id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Persist mono PCM from the webview (mic recording or decoded MP3) as a
/// 24 kHz PCM16 WAV and return its path, usable as a clone sample.
#[tauri::command]
pub fn save_recorded_sample(
    app: AppHandle,
    sample_rate: u32,
    samples: Vec<f32>,
) -> Result<String, String> {
    if samples.is_empty() {
        return Err(rust_i18n::t!("emptyRecording").into());
    }
    let ctx = app.state::<AppCtx>();
    let dir = ctx.dirs().samples_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.wav", uuid::Uuid::new_v4()));
    let rate = omnivoice::config::SAMPLE_RATE;
    let samples = wavio::resample(&samples, sample_rate, rate);
    wavio::write_wav(&path, &samples, rate, true).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Display name helper reused by the frontend history table.
#[tauri::command]
pub fn resolve_voice_name(app: AppHandle, id: Option<String>) -> String {
    let ctx = app.state::<AppCtx>();
    voice_name(&ctx, id.as_deref())
}
