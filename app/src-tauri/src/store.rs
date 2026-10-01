//! JSON persistence: settings, voice index, history entries.
//! Per-item files with tmp+rename atomic writes.

use crate::models::ModelKind;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// auto | cpu | metal | cuda
    pub device: String,
    /// auto | bf16 | f16 | f32 (s1-mini only; OmniVoice always runs F32)
    pub dtype: String,
    /// Active engine: s1-mini | omnivoice (see `models::REGISTRY`).
    #[serde(default = "default_model")]
    pub model: String,
    /// Model directory per model id (imported or downloaded).
    #[serde(default)]
    pub model_dirs: BTreeMap<String, String>,
    /// Keep codec.pth resident between runs (~1.5 GB extra memory).
    pub keep_codec_loaded: bool,
    /// Interface language: en | zh | ja | de | fr | es | ko | ar | ru | nl | it | pl | pt.
    /// Also passed to OmniVoice as the `lang` tag.
    pub language: String,
    /// Sampling params for s1-mini (CLI defaults).
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,
    /// OmniVoice token sampling temperature (class_temperature);
    /// 0 = greedy decoding, matching the upstream CLI default.
    #[serde(default)]
    pub omni_temperature: f64,
    /// Deterministic seed; reroll from the UI for a new take.
    #[serde(default = "rand_seed")]
    pub seed: u64,
}

/// Fields that configure the resident engine; changing only sampling params
/// or the UI language must not trigger a weights reload.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineFingerprint {
    pub kind: ModelKind,
    pub device: String,
    pub dtype: String,
    pub model_dir: Option<String>,
    pub keep_codec_loaded: bool,
}

impl Settings {
    pub fn kind(&self) -> ModelKind {
        crate::models::kind_from_id(&self.model).unwrap_or(ModelKind::S1Mini)
    }
    pub fn model_dir_of(&self, kind: ModelKind) -> Option<String> {
        self.model_dirs
            .get(crate::models::spec(kind).id)
            .cloned()
    }
    pub fn engine_fields(&self) -> EngineFingerprint {
        EngineFingerprint {
            kind: self.kind(),
            device: self.device.clone(),
            dtype: self.dtype.clone(),
            model_dir: self.model_dir_of(self.kind()),
            keep_codec_loaded: self.keep_codec_loaded,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            device: "auto".into(),
            dtype: "auto".into(),
            model: default_model(),
            model_dirs: BTreeMap::new(),
            keep_codec_loaded: true,
            language: "zh".into(),
            temperature: 0.7,
            top_p: 0.7,
            repetition_penalty: 1.5,
            omni_temperature: 0.0,
            seed: rand_seed(),
        }
    }
}

fn default_model() -> String {
    "s1-mini".into()
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.7
}

fn default_repetition_penalty() -> f64 {
    1.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Voice {
    pub id: String,
    pub name: String,
    pub gender: String,
    pub age: String,
    pub style: String,
    pub language: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// epoch ms
    pub created_at: i64,
    /// Sample duration in seconds (approx, pre-resample).
    pub sample_seconds: f64,
}

impl Voice {
    pub fn is_preset(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenParams {
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryEntry {
    pub id: String,
    pub text: String,
    pub voice_id: Option<String>,
    pub voice_name: String,
    pub model: String,
    pub duration_sec: f64,
    pub file_size: u64,
    /// path relative to app data ("audio/<id>.wav")
    pub wav_rel: String,
    /// epoch ms
    pub created_at: i64,
    pub params: GenParams,
    /// absolute wav path, filled in when serving to the UI (not persisted —
    /// serialized so the frontend receives it, skipped when reading back)
    #[serde(skip_deserializing)]
    pub wav_abs: String,
}

/// Atomic write: serialize to `<path>.tmp` then rename over the target.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let data = serde_json::to_string_pretty(value).context("serialize json")?;
    std::fs::write(&tmp, data).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename {}", path.display()))?;
    Ok(())
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn load_settings(path: &Path) -> Settings {
    let mut s = read_json::<Settings>(path).unwrap_or_default();
    // Migrate the legacy single-model `modelDir` field (pre multi-model
    // builds) into `modelDirs["s1-mini"]`.
    if s.model_dirs.is_empty() {
        if let Ok(v) = read_json::<serde_json::Value>(path) {
            if let Some(old) = v.get("modelDir").and_then(|x| x.as_str()) {
                s.model_dirs.insert("s1-mini".into(), old.to_string());
            }
        }
    }
    // Drop dirs referencing unknown model ids.
    s.model_dirs.retain(|k, _| crate::models::kind_from_id(k).is_some());
    s
}

pub fn load_voices(path: &Path) -> Vec<Voice> {
    read_json::<Vec<Voice>>(path).unwrap_or_default()
}

pub fn save_voices(path: &Path, voices: &[Voice]) -> Result<()> {
    write_json(path, &voices)
}

pub fn load_history(dir: &Path) -> Vec<HistoryEntry> {
    load_history_in(dir, dir.parent().unwrap_or(dir))
}

/// Load history entries from `dir`, resolving each wav's absolute path
/// against the app-data `root`.
pub fn load_history_in(dir: &Path, root: &Path) -> Vec<HistoryEntry> {
    let mut entries: Vec<HistoryEntry> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| read_json::<HistoryEntry>(&e.path()).ok())
                .map(|mut e| {
                    e.wav_abs = root.join(&e.wav_rel).to_string_lossy().into_owned();
                    e
                })
                .collect()
        })
        .unwrap_or_default();
    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    entries
}
