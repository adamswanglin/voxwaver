//! Settings, device probing, and model install status commands.

use serde::Serialize;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, State};

use crate::i18n;
use crate::models::{self, ModelKind, REGISTRY};
use crate::state::{AppCtx, AppState};
use crate::store::{self, Settings};

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.read().unwrap().clone()
}

/// Persist new settings; drops the resident engine when the engine-config
/// fingerprint changed (it rebuilds lazily on the next generation).
/// Returns whether the resident weights were invalidated (will reload).
#[tauri::command]
pub fn set_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: Settings,
) -> Result<bool, String> {
    let ctx = app.state::<AppCtx>();
    store::write_json(&ctx.dirs().settings_path(), &settings).map_err(|e| e.to_string())?;
    let needs_reload = {
        let mut guard = state.settings.write().unwrap();
        let changed = guard.engine_fields() != settings.engine_fields();
        *guard = settings.clone();
        changed
    };
    i18n::set_ui_lang(&settings.language);
    if needs_reload {
        // either engine kind may be affected: drop and rebuild on demand
        *state.engine.lock().unwrap() = None;
    }
    Ok(needs_reload)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceProbe {
    pub metal: bool,
    pub cuda: bool,
}

#[tauri::command]
pub fn probe_devices() -> DeviceProbe {
    DeviceProbe {
        metal: omnivoice::select_device("metal").is_ok(),
        cuda: omnivoice::select_device("cuda").is_ok(),
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub model: String,
    pub display_name: String,
    pub installed: bool,
    pub path: Option<String>,
    pub missing_files: Vec<String>,
    /// A download into app-data is in progress.
    pub downloading: bool,
    /// True when this is the active engine.
    pub active: bool,
}

fn status_for(ctx: &AppCtx, state: &AppState, kind: ModelKind) -> ModelStatus {
    let s = models::spec(kind);
    let settings = state.settings.read().unwrap();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = &settings.model_dir_of(kind) {
        candidates.push(PathBuf::from(p));
    }
    let dl = ctx.dirs().model_download_dir(kind);
    candidates.push(dl.clone());
    for dir in candidates {
        let missing = models::check_model_dir(kind, &dir).unwrap_or_default();
        if missing.is_empty() {
            return ModelStatus {
                model: s.id.into(),
                display_name: s.display_name.into(),
                installed: true,
                path: Some(dir.to_string_lossy().into_owned()),
                missing_files: Vec::new(),
                downloading: state.downloading.load(std::sync::atomic::Ordering::Relaxed),
                active: settings.kind() == kind,
            };
        }
    }
    let missing = models::check_model_dir(kind, &dl).unwrap_or_default();
    ModelStatus {
        model: s.id.into(),
        display_name: s.display_name.into(),
        installed: false,
        path: None,
        missing_files: missing,
        downloading: state.downloading.load(std::sync::atomic::Ordering::Relaxed),
        active: settings.kind() == kind,
    }
}

/// Install status for one model (`None` = the active model; kept for the
/// pre-multi-model frontend call shape).
#[tauri::command]
pub fn get_model_status(
    app: AppHandle,
    state: State<'_, AppState>,
    model: Option<String>,
) -> ModelStatus {
    let ctx = app.state::<AppCtx>();
    let kind = model
        .as_deref()
        .and_then(models::kind_from_id)
        .unwrap_or_else(|| state.settings.read().unwrap().kind());
    status_for(&ctx, &state, kind)
}

/// Install status for every model in the registry.
#[tauri::command]
pub fn list_model_status(app: AppHandle, state: State<'_, AppState>) -> Vec<ModelStatus> {
    let ctx = app.state::<AppCtx>();
    REGISTRY.iter().map(|s| status_for(&ctx, &state, s.kind)).collect()
}

/// Pick a local model folder via the native dialog; validates and saves it.
#[tauri::command]
pub async fn import_local_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model: String,
) -> Result<Option<String>, String> {
    let kind = models::kind_from_id(&model).ok_or_else(|| format!("unknown model {model:?}"))?;
    let picked = {
        let app = app.clone();
        tokio::task::spawn_blocking(move || {
            use tauri_plugin_dialog::DialogExt;
            app.dialog().file().blocking_pick_folder()
        })
        .await
        .map_err(|e| e.to_string())?
    };
    let Some(path) = picked else {
        return Ok(None);
    };
    let dir = path.into_path().map_err(|e| e.to_string())?;
    let missing = models::check_model_dir(kind, &dir).map_err(|e| e.to_string())?;
    if !missing.is_empty() {
        return Err(rust_i18n::t!("missingFiles", files = missing.join(", ")).to_string());
    }
    let path_str = dir.to_string_lossy().into_owned();
    state.settings.write().unwrap().model_dirs.insert(model.clone(), path_str.clone());
    // persist + invalidate engine if this is the active model
    let settings = state.settings.read().unwrap().clone();
    set_settings(app, state, settings)?;
    Ok(Some(path_str))
}

/// Remove the model copy from app data and drop the settings reference.
/// Imported folders are only de-referenced — their files are never touched.
#[tauri::command]
pub fn delete_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model: String,
) -> Result<(), String> {
    let ctx = app.state::<AppCtx>();
    let kind = models::kind_from_id(&model).ok_or_else(|| format!("unknown model {model:?}"))?;
    let dl = ctx.dirs().model_download_dir(kind);
    // Only the app-data download copy is physically removed.
    if dl.exists() {
        std::fs::remove_dir_all(&dl).map_err(|e| e.to_string())?;
    }
    // De-reference the model in any case (downloaded or imported).
    state.settings.write().unwrap().model_dirs.remove(&model);
    let settings = state.settings.read().unwrap().clone();
    set_settings(app, state, settings)?;
    Ok(())
}
