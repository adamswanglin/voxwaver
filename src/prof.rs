//! Env-gated micro-profiler (`VOX_PROFILE=1`): buckets the decode-loop wall
//! clock so the per-frame cost split (kernel submit vs pipeline-drain readback
//! vs CPU sampling) is visible without an external profiler. Zero cost when
//! the env var is unset.
//!
//! On Metal the CPU-side timing of a GPU op only covers command encoding and
//! submission; the actual execution lands in the *next* blocking readback
//! (`to_vec1`/`to_scalar`), which is why submit and readback are separate
//! buckets rather than one "forward" bucket.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const MAIN_FWD: usize = 0;
pub const MAIN_READ: usize = 1;
pub const MAIN_SAMPLE: usize = 2;
pub const FAST_SUBMIT: usize = 3;
pub const FAST_READ: usize = 4;
pub const FAST_SAMPLE: usize = 5;
pub const N: usize = 6;

const NAMES: [&str; N] = [
    "main_fwd",
    "main_read",
    "main_sample",
    "fast_submit",
    "fast_read",
    "fast_sample",
];

static NANOS: [AtomicU64; N] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VOX_PROFILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Timing scope; accumulates the elapsed nanos into `bucket` on drop. `None`
/// when profiling is off.
pub fn scope(bucket: usize) -> Option<Scope> {
    enabled().then(|| Scope { bucket, t: Instant::now() })
}

pub struct Scope {
    bucket: usize,
    t: Instant,
}

impl Drop for Scope {
    fn drop(&mut self) {
        NANOS[self.bucket].fetch_add(self.t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

struct ReportState {
    step: u64,
    t: Instant,
    nanos: [u64; N],
}

/// Print per-frame bucket means over the window since the last report. Called
/// every decode step; reports when `step % 50 == 0`, or immediately when
/// `force` (end of a text chunk). `step` must be a monotonic frame counter
/// across chunks, not the per-chunk loop variable.
pub fn maybe_report(step: u64, force: bool) {
    if !enabled() {
        return;
    }
    static LAST: OnceLock<Mutex<Option<ReportState>>> = OnceLock::new();
    let mut last = LAST.get_or_init(|| Mutex::new(None)).lock().unwrap();
    let Some(state) = last.as_mut() else {
        // first call: open the window, report nothing
        *last = Some(ReportState { step, t: Instant::now(), nanos: [0; N] });
        return;
    };
    let frames = step.saturating_sub(state.step);
    if frames == 0 || (!force && step % 50 != 0) {
        return;
    }
    let secs = state.t.elapsed().as_secs_f64();
    let mut line = format!(
        "[prof] +{frames}f @{:.1}f/s | ms/frame:",
        frames as f64 / secs.max(1e-9)
    );
    let mut sum_ms = 0f64;
    for i in 0..N {
        let now = NANOS[i].load(Ordering::Relaxed);
        let d = now.saturating_sub(state.nanos[i]);
        let ms = d as f64 / 1e6 / frames as f64;
        sum_ms += ms;
        line.push_str(&format!(" {}={ms:.2}", NAMES[i]));
    }
    let total_ms = secs * 1e3 / frames as f64;
    line.push_str(&format!(
        " other={:.2} total={total_ms:.2}",
        total_ms - sum_ms
    ));
    eprintln!("{line}");
    state.step = step;
    state.t = Instant::now();
    for i in 0..N {
        state.nanos[i] = NANOS[i].load(Ordering::Relaxed);
    }
}
