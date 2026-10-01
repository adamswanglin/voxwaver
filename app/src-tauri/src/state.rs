//! Shared application state managed by Tauri.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use tts_common::CancelFlag;

use crate::engine::AnyEngine;
use crate::models::{self, ModelKind};
use crate::store::Settings;

pub struct AppState {
    /// Lazily created once a valid model dir is configured; holds whichever
    /// engine the active model resolves to.
    pub engine: Mutex<Option<AnyEngine>>,
    pub settings: RwLock<Settings>,
    /// Current generation (engine lock serializes anyway; this flag gives a
    /// clean "busy" error instead of queueing).
    pub generating: AtomicBool,
    pub gen_cancel: CancelFlag,
    pub dl_cancel: CancelFlag,
    /// Cancelled/done marker for the running download, for UI state.
    pub downloading: AtomicBool,
}

impl AppState {
    pub fn new(settings: Settings) -> Self {
        Self {
            engine: Mutex::new(None),
            settings: RwLock::new(settings),
            generating: AtomicBool::new(false),
            gen_cancel: CancelFlag::new(),
            dl_cancel: CancelFlag::new(),
            downloading: AtomicBool::new(false),
        }
    }
}

/// Directory layout under Tauri's app_data_dir, created on startup.
pub struct Dirs {
    pub root: PathBuf,
}

impl Dirs {
    pub fn init(app_data: &PathBuf) -> std::io::Result<Self> {
        let d = Dirs { root: app_data.clone() };
        for sub in ["voices", "history", "audio", "models", "samples"] {
            std::fs::create_dir_all(d.root.join(sub))?;
        }
        Ok(d)
    }
    pub fn settings_path(&self) -> PathBuf {
        self.root.join("settings.json")
    }
    pub fn voices_index(&self) -> PathBuf {
        self.root.join("voices.json")
    }
    pub fn voice_dir(&self, id: &str) -> PathBuf {
        self.root.join("voices").join(id)
    }
    pub fn history_dir(&self) -> PathBuf {
        self.root.join("history")
    }
    pub fn audio_dir(&self) -> PathBuf {
        self.root.join("audio")
    }
    pub fn audio_path(&self, id: &str) -> PathBuf {
        self.audio_dir().join(format!("{id}.wav"))
    }
    pub fn history_path(&self, id: &str) -> PathBuf {
        self.history_dir().join(format!("{id}.json"))
    }
    pub fn model_download_dir(&self, kind: ModelKind) -> PathBuf {
        self.root.join("models").join(models::download_dir_name(kind))
    }
    /// Uploaded/recorded samples converted to WAV by the frontend.
    pub fn samples_dir(&self) -> PathBuf {
        self.root.join("samples")
    }
}

/// Cheap handle kept in Tauri state alongside AppState.
#[derive(Clone)]
pub struct AppCtx {
    pub dirs: Arc<Dirs>,
}

impl AppCtx {
    pub fn dirs(&self) -> &Dirs {
        &self.dirs
    }
}
