//! Speech generation: engine lifecycle, progress events, cancellation.

use anyhow::Context as _;
use serde::Serialize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};
use tts_common::{Progress, ProgressSink};
use voxwaver_core::wavio;

use crate::engine::{AnyEngine, GenJob, RefMat};
use crate::models::ModelKind;
use crate::state::{AppCtx, AppState};
use crate::store::{self, GenParams, HistoryEntry};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    pub ready: bool,
    pub lm_loaded: bool,
    pub codec_loaded: bool,
    pub model: String,
    pub model_dir: Option<String>,
    pub device: String,
    pub dtype: String,
}

#[tauri::command]
pub fn get_engine_status(state: State<'_, AppState>) -> EngineStatus {
    let settings = state.settings.read().unwrap();
    let eng = state.engine.lock().unwrap();
    let loaded = eng.as_ref().is_some_and(|e| e.is_model_loaded());
    EngineStatus {
        ready: eng.is_some(),
        lm_loaded: loaded,
        codec_loaded: loaded,
        model: settings.model.clone(),
        model_dir: settings.model_dir_of(settings.kind()),
        device: settings.device.clone(),
        dtype: settings.dtype.clone(),
    }
}

/// Load (or rebuild) the engine to match current settings. Cheap when the
/// engine already exists with the same config. When the model kind or its
/// config changed, the old engine is dropped first — both engines are
/// GB-scale, so the replacement must never coexist with the old one.
pub(crate) fn ensure_engine(state: &AppState, sink: &dyn ProgressSink) -> anyhow::Result<()> {
    let settings = state.settings.read().unwrap().clone();
    let mut eng = state.engine.lock().unwrap();
    match eng.as_mut() {
        // s1-mini stays resident and hot-reconfigures (lazy weight reload).
        Some(AnyEngine::S1(se)) if settings.kind() == ModelKind::S1Mini => {
            if let Ok(cfg) = crate::engine::s1_config(&settings) {
                se.reconfigure(cfg);
                return Ok(());
            }
            // invalid config (e.g. model removed): fall through to drop
        }
        // OmniVoice is eager-loaded; reuse only when still the active model
        // on the same device.
        Some(AnyEngine::Omni(r))
            if settings.kind() == ModelKind::OmniVoice && r.device == settings.device =>
        {
            return Ok(());
        }
        _ => {}
    }
    // Drop the old engine (releasing its weights) before building the new one.
    *eng = None;
    *eng = Some(AnyEngine::build(&settings, sink)?);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateReq {
    pub text: String,
    pub voice_id: Option<String>,
    /// OmniVoice style instruction (e.g. "用愉快的语气").
    #[serde(default)]
    pub instruct: Option<String>,
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub seed: u64,
}

/// Forwarder channel payload: engine progress or a log line.
pub(crate) enum Fwd {
    Progress(Progress),
    Log(String),
}

pub(crate) struct ChannelSink {
    pub(crate) tx: mpsc::Sender<Fwd>,
}

impl ProgressSink for ChannelSink {
    fn progress(&self, p: Progress) {
        let _ = self.tx.send(Fwd::Progress(p));
    }
    fn log(&self, msg: &str) {
        let _ = self.tx.send(Fwd::Log(msg.to_string()));
    }
}

#[tauri::command]
pub async fn generate(
    app: AppHandle,
    state: State<'_, AppState>,
    req: GenerateReq,
) -> Result<HistoryEntry, String> {
    if state
        .generating
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("已有生成任务进行中".into());
    }
    let result = run_generate(&app, req).await;
    state.generating.store(false, Ordering::SeqCst);
    result
}

async fn run_generate(app: &AppHandle, req: GenerateReq) -> Result<HistoryEntry, String> {
    if req.text.trim().is_empty() {
        return Err("文本不能为空".into());
    }
    let state: State<'_, AppState> = app.state();
    let cancel = state.gen_cancel.clone();
    cancel.clear();

    // progress forwarder: mpsc -> throttled window emit
    let (tx, rx) = mpsc::channel::<Fwd>();
    let app_fwd = app.clone();
    let fwd = std::thread::spawn(move || forward_events(app_fwd, rx));

    let voice_id = req.voice_id.clone();
    let text = req.text.clone();
    let params = GenParams {
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
        seed: req.seed,
    };
    let app_gen = app.clone();
    let app_blk = app.clone();
    let inner = tokio::task::spawn_blocking(move || -> anyhow::Result<HistoryEntry> {
        let state: State<'_, AppState> = app_blk.state();
        let ctx = app_blk.state::<AppCtx>();
        let sink = ChannelSink { tx: tx.clone() };
        ensure_engine(&state, &sink)?;
        let settings = state.settings.read().unwrap().clone();
        let kind = settings.kind();
        let ref_mat = voice_id
            .as_deref()
            .filter(|id| !id.is_empty() && *id != "default")
            .map(|id| load_or_encode_ref(&ctx, &state, kind, id, &sink))
            .transpose()?;
        let job = GenJob {
            text: text.clone(),
            instruct: req.instruct.clone().filter(|s| !s.trim().is_empty()),
            temperature: settings.temperature,
            top_p: settings.top_p,
            repetition_penalty: settings.repetition_penalty,
            omni_temperature: settings.omni_temperature,
            lang: settings.language.clone(),
            seed: params.seed,
        };
        let out = {
            let mut eng = state.engine.lock().unwrap();
            let eng = eng.as_mut().expect("engine ensured above");
            eng.generate(&job, ref_mat.as_ref(), &cancel, &sink)?
        };
        // ---- persist audio + history entry ----
        sink.progress(Progress::WritingWav);
        let id = uuid::Uuid::new_v4().to_string();
        let wav_path = ctx.dirs().audio_path(&id);
        wavio::write_wav(&wav_path, &out.samples, out.sample_rate, true)?;
        let duration_sec = out.samples.len() as f64 / out.sample_rate as f64;
        let entry = HistoryEntry {
            id: id.clone(),
            text: job.text.clone(),
            voice_id: voice_id.clone(),
            voice_name: voice_name(&ctx, voice_id.as_deref()),
            model: settings.model.clone(),
            duration_sec,
            file_size: std::fs::metadata(&wav_path).map(|m| m.len()).unwrap_or(0),
            wav_rel: format!("audio/{id}.wav"),
            created_at: now_ms(),
            params,
            wav_abs: wav_path.to_string_lossy().into_owned(),
        };
        store::write_json(&ctx.dirs().history_path(&id), &entry)?;
        Ok(entry)
    })
    .await
    .map_err(|e| format!("generation task panicked: {e}"))?;

    // the closure's drop of `tx` closed the channel; drain remaining events
    let _ = fwd.join();

    match inner {
        Ok(entry) => {
            let _ = app_gen.emit("gen://done", &entry);
            Ok(entry)
        }
        Err(e) => {
            let msg = format!("{e:#}");
            let cancelled = msg.contains("cancelled");
            let _ = app_gen.emit(
                "gen://error",
                serde_json::json!({ "message": msg, "cancelled": cancelled }),
            );
            Err(msg)
        }
    }
}

/// Drain the mpsc channel and emit `gen://progress` events, throttled to
/// ~4 Hz for the chatty `generating`/log variants.
fn forward_events(app: AppHandle, rx: mpsc::Receiver<Fwd>) {
    let mut last_emit = std::time::Instant::now() - std::time::Duration::from_secs(1);
    while let Ok(fwd) = rx.recv() {
        let chatty = matches!(fwd, Fwd::Log(_) | Fwd::Progress(Progress::Generating { .. }) | Fwd::Progress(Progress::Unmasking { .. }));
        if chatty && last_emit.elapsed() < std::time::Duration::from_millis(250) {
            continue;
        }
        last_emit = std::time::Instant::now();
        match fwd {
            Fwd::Progress(p) => {
                let _ = app.emit("gen://progress", &p);
            }
            Fwd::Log(m) => {
                let _ = app.emit("gen://log", m);
            }
        }
    }
}

#[tauri::command]
pub fn cancel_generate(state: State<'_, AppState>) {
    state.gen_cancel.cancel();
}

/// Preload the LM (and codec when configured) so the first generation
/// doesn't pay the load cost.
#[tauri::command]
pub async fn warmup_engine(app: AppHandle) -> Result<(), String> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let state: State<'_, AppState> = app.state();
        let sink = tts_common::NullSink;
        ensure_engine(&state, &sink)?;
        let mut eng = state.engine.lock().unwrap();
        if let Some(e) = eng.as_mut() {
            e.warmup();
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| format!("{e:#}"))
}

// ---- voice helpers (shared with voices.rs) ----

/// Persisted next to the encoded codes in `<voice_dir>/ref.json` (s1-mini)
/// or `ref.omni.json` (OmniVoice).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct RefFile {
    pub transcript: String,
    pub codes: Vec<Vec<u32>>,
}

/// Cache file name for a model kind's encoded reference codes.
pub(crate) fn ref_cache_file(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::S1Mini => "ref.json",
        ModelKind::OmniVoice => "ref.omni.json",
    }
}

/// Load a voice's reference for `kind`; when the per-model cache is missing
/// (e.g. a voice created under the other model), lazily encode the stored
/// `ref.wav` through the resident engine and persist the cache.
pub(crate) fn load_or_encode_ref(
    ctx: &AppCtx,
    state: &AppState,
    kind: ModelKind,
    voice_id: &str,
    sink: &dyn ProgressSink,
) -> anyhow::Result<RefMat> {
    let dir = ctx.dirs().voice_dir(voice_id);
    let cache = dir.join(ref_cache_file(kind));
    if let Ok(rf) = store::read_json::<RefFile>(&cache) {
        return Ok(RefMat {
            transcript: rf.transcript,
            codes: rf.codes,
        });
    }
    // Fall back for the transcript if only the other model's cache exists.
    let transcript = store::read_json::<RefFile>(&dir.join("ref.json"))
        .map(|rf| rf.transcript)
        .or_else(|_| {
            store::read_json::<RefFile>(&dir.join("ref.omni.json")).map(|rf| rf.transcript)
        })
        .unwrap_or_default();
    let wav = dir.join("ref.wav");
    anyhow::ensure!(wav.is_file(), "voice {voice_id} has no reference sample");
    let codes = {
        let mut eng = state.engine.lock().unwrap();
        let eng = eng.as_mut().context("engine not loaded")?;
        eng.encode_reference(&wav, sink)?
    };
    let mat = RefMat {
        transcript,
        codes: codes.clone(),
    };
    let _ = store::write_json(&cache, &RefFile { transcript: mat.transcript.clone(), codes });
    Ok(mat)
}

pub(crate) fn voice_name(ctx: &AppCtx, id: Option<&str>) -> String {
    match id {
        Some(id) if !id.is_empty() && id != "default" => {
            store::load_voices(&ctx.dirs().voices_index())
                .into_iter()
                .find(|v| v.id == id)
                .map(|v| v.name)
                .unwrap_or_else(|| "未知声音".into())
        }
        _ => "默认音色".into(),
    }
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
