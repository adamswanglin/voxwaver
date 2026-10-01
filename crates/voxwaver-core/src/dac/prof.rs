//! Env-gated phase profiler for codec decode (`VOX_DAC_PROF=1|2`).
//!
//! Level 1 times module boundaries (quantizer stages, decoder conv_in, each
//! DecoderBlock, tail and the CPU readback, plus the block-internal
//! snake/tconv/RU split). Level 2 (`VOX_DAC_PROF=2`) additionally times the
//! inside of every ResidualUnit. Each timing point synchronizes the Metal
//! device before and after the measured closure, so the recorded wall time is
//! the true GPU time of that segment; the price is queue serialization and
//! extra pooled-buffer recycling, i.e. the sum of all segments is a small
//! over-estimate of an uninstrumented decode. With the variable unset the
//! closures run directly and the overhead is nil.
//!
//! `Dac::decode_codes` prints (and resets) the accumulated breakdown after
//! every decode, so a profiled run gets one line per decode call.

use anyhow::Result;
use candle_core::Device;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const Q_FROM_CODES: usize = 0;
pub const Q_POST: usize = 1;
pub const Q_UP: usize = 2;
pub const Q_UP0_TCONV: usize = 3;
pub const Q_UP0_CN: usize = 4;
pub const Q_UP1_TCONV: usize = 5;
pub const Q_UP1_CN: usize = 6;
pub const Q_TOTAL: usize = 7;
pub const D_CONV_IN: usize = 8;
pub const D_B0: usize = 9;
pub const D_B1: usize = 10;
pub const D_B2: usize = 11;
pub const D_B3: usize = 12;
pub const D_TAIL: usize = 13;
pub const D_TOTAL: usize = 14;
pub const D_READBACK: usize = 15;
pub const BLK_SNAKE: usize = 16;
pub const BLK_TCONV: usize = 17;
pub const BLK_RU0: usize = 18;
pub const BLK_RU1: usize = 19;
pub const BLK_RU2: usize = 20;
pub const RU_S1: usize = 21;
pub const RU_C1: usize = 22;
pub const RU_S2: usize = 23;
pub const RU_C2: usize = 24;
pub const N: usize = 25;

const NAMES: [&str; N] = [
    "q_from_codes",
    "q_post",
    "q_up",
    "q_up0_tconv",
    "q_up0_cn",
    "q_up1_tconv",
    "q_up1_cn",
    "q_total",
    "d_conv_in",
    "d_b0",
    "d_b1",
    "d_b2",
    "d_b3",
    "d_tail",
    "d_total",
    "d_readback",
    "blk_snake",
    "blk_tconv",
    "blk_ru0",
    "blk_ru1",
    "blk_ru2",
    "ru_s1",
    "ru_c1",
    "ru_s2",
    "ru_c2",
];

#[rustfmt::skip]
static NANOS: [AtomicU64; N] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0),
];

#[rustfmt::skip]
static COUNT: [AtomicU64; N] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0),
];

fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VOX_DAC_PROF") {
        Ok(v) if v == "2" || v.eq_ignore_ascii_case("deep") => 2,
        Ok(v) if !v.is_empty() && v != "0" => 1,
        _ => 0,
    })
}

pub fn enabled() -> bool {
    level() >= 1
}

pub fn deep_enabled() -> bool {
    level() >= 2
}

/// Level-1 timing point (`VOX_DAC_PROF>=1`): no-op when profiling is off.
pub fn phase<T>(bucket: usize, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled() {
        return f();
    }
    timed(bucket, dev, f)
}

/// Level-2 timing point (`VOX_DAC_PROF=2`): no-op otherwise.
pub fn phase_deep<T>(bucket: usize, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !deep_enabled() {
        return f();
    }
    timed(bucket, dev, f)
}

fn timed<T>(bucket: usize, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if dev.is_metal() {
        let _ = dev.synchronize();
    }
    let t = Instant::now();
    let out = f()?;
    if dev.is_metal() {
        let _ = dev.synchronize();
    }
    NANOS[bucket].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    COUNT[bucket].fetch_add(1, Ordering::Relaxed);
    Ok(out)
}

/// Print `name=totalMs/calls` for every bucket touched since the last report,
/// then reset all counters.
pub fn print_and_reset() {
    if !enabled() {
        return;
    }
    let mut line = String::from("[dac-prof]");
    let mut total_ns = 0u64;
    for i in 0..N {
        let ns = NANOS[i].swap(0, Ordering::Relaxed);
        let calls = COUNT[i].swap(0, Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        total_ns += ns;
        line.push_str(&format!(" {}={:.1}ms/{calls}", NAMES[i], ns as f64 / 1e6));
    }
    line.push_str(&format!(" all_phases={:.1}ms", total_ns as f64 / 1e6));
    eprintln!("{line}");
}
