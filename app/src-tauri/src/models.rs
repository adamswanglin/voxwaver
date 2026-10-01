//! Model registry: the engines the app can drive, their required files,
//! download sources, and directory validation.

use anyhow::Result;
use std::path::Path;

/// The engines the app can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// k2-fsa OmniVoice via the omnivoice crate (Qwen3 unmasking + HiggsAudioV2, 24 kHz).
    OmniVoice,
}

pub struct ModelSpec {
    pub kind: ModelKind,
    /// Stable id used in settings / history entries.
    pub id: &'static str,
    pub display_name: &'static str,
    /// Files that must exist for the directory to be loadable.
    pub required_files: &'static [&'static str],
    /// Files fetched on download (may include subdirectories).
    pub download_files: &'static [&'static str],
    pub hf_repo: &'static str,
    pub hf_rev: &'static str,
    /// ModelScope mirror, when one exists.
    pub ms_repo: Option<&'static str>,
    pub size_hint: &'static str,
}

pub const REGISTRY: [ModelSpec; 1] = [
    ModelSpec {
        kind: ModelKind::OmniVoice,
        id: "omnivoice",
        display_name: "OmniVoice",
        required_files: &[
            "model.safetensors",
            "tokenizer.json",
            "audio_tokenizer/model.safetensors",
        ],
        download_files: &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "model.safetensors",
            "audio_tokenizer/config.json",
            "audio_tokenizer/model.safetensors",
            "audio_tokenizer/preprocessor_config.json",
        ],
        hf_repo: "k2-fsa/OmniVoice",
        hf_rev: "main",
        ms_repo: None,
        size_hint: "~3.3 GB",
    },
];

pub fn spec(kind: ModelKind) -> &'static ModelSpec {
    REGISTRY.iter().find(|s| s.kind == kind).expect("kind in registry")
}

pub fn kind_from_id(id: &str) -> Option<ModelKind> {
    REGISTRY.iter().find(|s| s.id == id).map(|s| s.kind)
}

/// Validate that `dir` contains a loadable copy of `kind`; returns the
/// missing file names (empty when complete).
pub fn check_model_dir(kind: ModelKind, dir: &Path) -> Result<Vec<String>> {
    let s = spec(kind);
    let mut missing = Vec::new();
    for f in s.required_files {
        if !dir.join(f).is_file() {
            missing.push((*f).to_string());
        }
    }
    Ok(missing)
}

/// Directory name under `<app-data>/models/` for each model's downloads.
pub fn download_dir_name(kind: ModelKind) -> &'static str {
    spec(kind).id
}
