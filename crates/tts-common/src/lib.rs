//! Types shared by the TTS engines (omnivoice) and the app: cooperative
//! cancellation, stage progress reporting, the sink trait, and WAV I/O.

pub mod wavio;

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Cooperative cancellation flag shared with the engine's caller.
#[derive(Clone)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    /// Reset this flag in place, preserving the shared Arc — the flag held
    /// by the caller (app state) and the one passed to the worker stay the
    /// same, so a later `cancel()` is observed (start of a new run).
    pub fn clear(&self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// Stage-by-stage progress report (serde-tagged for Tauri events).
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Progress {
    LoadingLm,
    LoadingCodec,
    EncodingRef { seconds: f32 },
    Chunks { total: usize },
    Prefill { chunk: usize, total: usize, tokens: usize },
    Generating { chunk: usize, total: usize, frames: usize, max_frames: usize, fps: f32 },
    /// OmniVoice iterative unmasking: step `step` of `total_steps` within
    /// chunk `chunk` (1-based) of `total` chunks.
    Unmasking { chunk: usize, total: usize, step: usize, total_steps: usize },
    /// Codec (DAC) decode of chunk `chunk` (1-based) of `total`.
    Decoding { chunk: usize, total: usize, frames_total: usize },
    WritingWav,
    Done { frames: usize, seconds: f32 },
}

/// Receiver for progress/log events. The engine calls these synchronously;
/// throttling and forwarding (e.g. to Tauri emits) is the caller's concern.
pub trait ProgressSink: Send + Sync {
    fn progress(&self, _p: Progress) {}
    fn log(&self, _msg: &str) {}
}

/// A sink that drops everything.
pub struct NullSink;
impl ProgressSink for NullSink {}
