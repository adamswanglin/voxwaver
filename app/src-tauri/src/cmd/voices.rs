//! Voice library: list / create (encode reference) / delete.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::cmd::tts::{now_ms, ref_cache_file, voice_name, ChannelSink, Fwd, RefFile};
use crate::models::ModelKind;
use crate::state::{AppCtx, AppState};
use crate::store::{self, Voice};

/// Transcript lives inside each model's code cache; read the s1-mini copy
/// first, then the OmniVoice one.
fn read_transcript(dir: &std::path::Path) -> Option<String> {
    ["ref.json", "ref.omni.json"]
        .iter()
        .find_map(|f| store::read_json::<RefFile>(&dir.join(f)).ok())
        .map(|rf| rf.transcript)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceView {
    pub id: String,
    pub name: String,
    pub gender: String,
    pub age: String,
    pub style: String,
    pub language: String,
    pub tags: Vec<String>,
    pub created_at: i64,
    /// built-in (default timbre) or cloned
    pub is_clone: bool,
    /// playable reference sample, if any
    pub ref_wav: Option<String>,
    /// reference transcript, cloned voices only
    pub transcript: Option<String>,
    /// reference sample duration in seconds
    pub sample_seconds: Option<f64>,
}

fn default_voice() -> VoiceView {
    VoiceView {
        id: "default".into(),
        name: "默认音色".into(),
        gender: "中性".into(),
        age: "—".into(),
        style: "zero-shot 默认音色（无参考音频）".into(),
        language: "多语言".into(),
        tags: vec!["预置".into()],
        created_at: 0,
        is_clone: false,
        ref_wav: None,
        transcript: None,
        sample_seconds: None,
    }
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
            gender: self.gender,
            age: self.age,
            style: self.style,
            language: self.language,
            tags: self.tags,
            created_at: self.created_at,
        }
    }
}

#[tauri::command]
pub fn list_voices(app: AppHandle) -> Vec<VoiceView> {
    let ctx = app.state::<AppCtx>();
    let idx = ctx.dirs().voices_index();
    let mut views = vec![default_voice()];
    for v in store::load_voices(&idx) {
        let dir = ctx.dirs().voice_dir(&v.id);
        let ref_wav = dir.join("ref.wav").to_str().map(|s| s.to_string());
        let transcript = read_transcript(&dir);
        views.push(VoiceView {
            id: v.id,
            name: v.name,
            gender: v.gender,
            age: v.age,
            style: v.style,
            language: v.language,
            tags: v.tags,
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
    pub gender: String,
    pub age: String,
    pub style: String,
    pub language: String,
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
        // engine must exist (the active model encodes the reference)
        crate::cmd::tts::ensure_engine(&state, &sink)?;
        let kind = state.settings.read().unwrap().kind();
        let codes = {
            let mut eng = state.engine.lock().unwrap();
            let eng = eng.as_mut().expect("engine ensured above");
            eng.encode_reference(sample, &sink)?
        };
        std::fs::create_dir_all(dir)?;
        // keep a 44.1k pcm16 copy of the reference for auditioning (also the
        // lazy-encoding source for the other model's cache)
        let (samples, sr) = voxwaver_core::wavio::read_wav_mono(sample)?;
        let samples = voxwaver_core::wavio::resample(&samples, sr, 44100);
        voxwaver_core::wavio::write_wav(&dir.join("ref.wav"), &samples, 44100, true)?;
        store::write_json(
            &dir.join(ref_cache_file(kind)),
            &RefFile {
                transcript: transcript.to_string(),
                codes,
            },
        )?;
        Ok(samples.len() as f64 / 44100.0)
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
        return Err("声音名称不能为空".into());
    }
    if req.transcript.trim().is_empty() {
        return Err("请填写样本的转写文本".into());
    }
    let sample = PathBuf::from(&req.sample_path);
    if !sample.is_file() {
        return Err("样本文件不存在".into());
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
            gender: req.gender,
            age: req.age,
            style: req.style,
            language: req.language,
            tags: vec!["克隆".into()],
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
    pub gender: String,
    pub age: String,
    pub style: String,
    pub language: String,
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
    if req.id == "default" {
        return Err("不能编辑默认音色".into());
    }
    if req.name.trim().is_empty() {
        return Err("声音名称不能为空".into());
    }
    if req.transcript.trim().is_empty() {
        return Err("请填写样本的转写文本".into());
    }
    let sample = match &req.sample_path {
        Some(p) if !p.trim().is_empty() => {
            let p = PathBuf::from(p);
            if !p.is_file() {
                return Err("样本文件不存在".into());
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
            .ok_or_else(|| anyhow::anyhow!("声音不存在"))?;
        voice.name = req.name.trim().to_string();
        voice.gender = req.gender;
        voice.age = req.age;
        voice.style = req.style;
        voice.language = req.language;
        let dir = ctx.dirs().voice_dir(&req.id);
        if let Some(sample) = &sample {
            voice.sample_seconds =
                encode_ref_into(&app_blk, &dir, sample, req.transcript.trim())?;
        } else {
            // transcript may still have been edited: rewrite it into every
            // per-model cache present on disk
            for kind in [ModelKind::S1Mini, ModelKind::OmniVoice] {
                let path = dir.join(ref_cache_file(kind));
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
    if id == "default" {
        return Err("不能删除默认音色".into());
    }
    let ctx = app.state::<AppCtx>();
    let idx = ctx.dirs().voices_index();
    let mut voices = store::load_voices(&idx);
    let before = voices.len();
    voices.retain(|v| v.id != id);
    if voices.len() == before {
        return Err("声音不存在".into());
    }
    store::save_voices(&idx, &voices).map_err(|e| e.to_string())?;
    let dir = ctx.dirs().voice_dir(&id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Persist mono PCM from the webview (mic recording or decoded MP3) as a
/// 44.1 kHz PCM16 WAV and return its path, usable as a clone sample.
#[tauri::command]
pub fn save_recorded_sample(
    app: AppHandle,
    sample_rate: u32,
    samples: Vec<f32>,
) -> Result<String, String> {
    if samples.is_empty() {
        return Err("录音数据为空".into());
    }
    let ctx = app.state::<AppCtx>();
    let dir = ctx.dirs().samples_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.wav", uuid::Uuid::new_v4()));
    let samples = voxwaver_core::wavio::resample(&samples, sample_rate, 44100);
    voxwaver_core::wavio::write_wav(&path, &samples, 44100, true).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Display name helper reused by the frontend history table.
#[tauri::command]
pub fn resolve_voice_name(app: AppHandle, id: Option<String>) -> String {
    let ctx = app.state::<AppCtx>();
    voice_name(&ctx, id.as_deref())
}
