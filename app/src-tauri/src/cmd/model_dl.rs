//! Model download from HuggingFace / ModelScope with progress events.

use futures_util::StreamExt;
use serde::Serialize;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::models::{self, ModelSpec};
use crate::state::{AppCtx, AppState};

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DownloadEvent {
    /// model id the event belongs to
    pub model: String,
    /// current file name
    pub file: String,
    pub file_index: usize,
    pub file_count: usize,
    /// bytes of the current file
    pub downloaded: u64,
    /// total bytes of the current file (0 if unknown)
    pub total: u64,
    pub speed_bps: f64,
    pub done: bool,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model: String,
    source: String,
) -> Result<(), String> {
    if state
        .downloading
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(rust_i18n::t!("dlBusy").into());
    }
    let result = run_download(&app, &state, &model, &source).await;
    state.downloading.store(false, Ordering::SeqCst);
    result
}

async fn run_download(
    app: &AppHandle,
    state: &State<'_, AppState>,
    model: &str,
    source: &str,
) -> Result<(), String> {
    let ctx = app.state::<AppCtx>();
    let kind = models::kind_from_id(model).ok_or_else(|| format!("unknown model {model:?}"))?;
    let spec: &ModelSpec = models::spec(kind);
    let base = match source {
        "hf" => format!("https://huggingface.co/{}/resolve/{}/", spec.hf_repo, spec.hf_rev),
        "modelscope" => {
            let ms = spec
                .ms_repo
                .ok_or_else(|| rust_i18n::t!("msMirror", model = spec.display_name).to_string())?;
            format!("https://modelscope.cn/models/{ms}/resolve/master/")
        }
        other => return Err(format!("unknown source {other:?} (hf|modelscope)")),
    };
    let dir = ctx.dirs().model_download_dir(kind);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let cancel = state.dl_cancel.clone();
    cancel.clear();

    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| e.to_string())?;

    for (i, file) in spec.download_files.iter().enumerate() {
        let file_name = (*file).to_string();
        let dest = dir.join(file);
        // subdirectory support (e.g. audio_tokenizer/model.safetensors)
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let done_evt = |error: Option<String>, done: bool| DownloadEvent {
            model: model.to_string(),
            file: file_name.clone(),
            file_index: i,
            file_count: spec.download_files.len(),
            downloaded: 0,
            total: 0,
            speed_bps: 0.0,
            done,
            error,
        };
        if cancel.is_cancelled() {
            let _ = app.emit("model://download", done_evt(Some("cancelled".into()), false));
            // Locale-independent sentinel; the frontend greps for "cancelled"
            return Err("cancelled".into());
        }
        // resume support: skip files already fully downloaded
        if dest.is_file() {
            let _ = app.emit("model://download", done_evt(None, true));
            continue;
        }
        let url = format!("{base}{file}");
        let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(rust_i18n::t!("dlFileFail", file = file, status = resp.status()).to_string());
        }
        let total = resp.content_length().unwrap_or(0);
        let part = dir.join(format!("{file}.part"));
        let mut out = std::fs::File::create(&part).map_err(|e| e.to_string())?;
        let mut downloaded: u64 = 0;
        let t0 = Instant::now();
        let mut last_emit = Instant::now() - std::time::Duration::from_secs(1);
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                let _ = std::fs::remove_file(&part);
                return Err("cancelled".into());
            }
            let chunk = chunk.map_err(|e| e.to_string())?;
            use std::io::Write;
            out.write_all(&chunk).map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;
            if last_emit.elapsed() >= std::time::Duration::from_millis(300) {
                last_emit = Instant::now();
                let _ = app.emit(
                    "model://download",
                    DownloadEvent {
                        model: model.to_string(),
                        file: file_name.clone(),
                        file_index: i,
                        file_count: spec.download_files.len(),
                        downloaded,
                        total,
                        speed_bps: downloaded as f64 / t0.elapsed().as_secs_f64().max(1e-9),
                        done: false,
                        error: None,
                    },
                );
            }
        }
        drop(out);
        std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
        let _ = app.emit(
            "model://download",
            DownloadEvent {
                model: model.to_string(),
                file: file_name.clone(),
                file_index: i,
                file_count: spec.download_files.len(),
                downloaded,
                total,
                speed_bps: downloaded as f64 / t0.elapsed().as_secs_f64().max(1e-9),
                done: true,
                error: None,
            },
        );
    }

    // point settings at the freshly downloaded copy
    state.settings.write().unwrap().model_dirs.insert(
        model.to_string(),
        dir.to_string_lossy().into_owned(),
    );
    let settings = state.settings.read().unwrap().clone();
    crate::cmd::settings::set_settings(app.clone(), state.clone(), settings)?;
    Ok(())
}

#[tauri::command]
pub fn cancel_download(state: State<'_, AppState>) {
    state.dl_cancel.cancel();
}
