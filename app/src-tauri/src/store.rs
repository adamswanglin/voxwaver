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
    /// Active engine (see `models::REGISTRY`).
    #[serde(default = "default_model")]
    pub model: String,
    /// Model directory per model id (imported or downloaded).
    #[serde(default)]
    pub model_dirs: BTreeMap<String, String>,
    /// Interface language: en | zh | ja | de | fr | es | ko | ar | ru | nl | it | pl | pt.
    pub language: String,
    /// TTS output language tag passed to OmniVoice ('' = auto, not passed).
    #[serde(default)]
    pub output_language: String,
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
    pub model_dir: Option<String>,
}

impl Settings {
    pub fn kind(&self) -> ModelKind {
        crate::models::kind_from_id(&self.model).unwrap_or(ModelKind::OmniVoice)
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
            model_dir: self.model_dir_of(self.kind()),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            device: "auto".into(),
            model: default_model(),
            model_dirs: BTreeMap::new(),
            language: "zh".into(),
            output_language: String::new(),
            seed: rand_seed(),
        }
    }
}

fn default_model() -> String {
    "omnivoice".into()
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Voice {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// optional emoji icon shown in place of the name-letter fallback
    #[serde(default)]
    pub icon: Option<String>,
    /// epoch ms
    pub created_at: i64,
    /// Sample duration in seconds (approx, pre-resample).
    pub sample_seconds: f64,
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
    // Settings from pre-OmniVoice builds may name a removed engine; fall
    // back to the only one left.
    if crate::models::kind_from_id(&s.model).is_none() {
        s.model = default_model();
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

/// Search history by grepping the raw JSON text of each entry file —
/// case-insensitive substring match, no JSON parsing on the miss path.
pub fn search_history_in(dir: &Path, root: &Path, query: &str) -> Vec<HistoryEntry> {
    let needle = query.to_lowercase();
    let mut entries: Vec<HistoryEntry> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| {
                    let raw = std::fs::read_to_string(e.path()).ok()?;
                    if !raw.to_lowercase().contains(&needle) {
                        return None;
                    }
                    let mut entry = read_json::<HistoryEntry>(&e.path()).ok()?;
                    entry.wav_abs = root.join(&entry.wav_rel).to_string_lossy().into_owned();
                    Some(entry)
                })
                .collect()
        })
        .unwrap_or_default();
    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    entries
}
