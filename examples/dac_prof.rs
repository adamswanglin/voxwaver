//! Standalone codec-decode profiler: loads codec.pth, synthesizes `frames`
//! codebook frames, warms up and then times `Dac::decode_codes` twice.
//! With `VOX_DAC_PROF=1|2` each decode also prints the per-phase breakdown
//! (see `dac::prof`).
//!
//! Run:
//!   VOX_DAC_PROF=2 cargo run --release --features metal --example dac_prof -- \
//!       [frames] [codec.pth path]
//!
//! Defaults: 356 frames (matches the s1-mini reference run),
//! /Users/wanglin/app/projects/github/s1-mini/codec.pth.
use candle_core::Device;
use std::path::Path;
use std::time::Instant;
use voxwaver_core::config::CodecConfig;
use voxwaver_core::dac::Dac;

/// Deterministic pseudo-random codes: row 0 semantic (<4096), rows 1..10
/// residual (<1024), mirroring a real 10-codebook decode input.
fn synth_codes(frames: usize) -> Vec<Vec<u32>> {
    let mut state = 0x1234_5678u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    (0..10)
        .map(|row| {
            let m = if row == 0 { 4096 } else { 1024 };
            (0..frames).map(|_| next() % m).collect()
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(356);
    let path = args.next().unwrap_or_else(|| {
        "/Users/wanglin/app/projects/github/s1-mini/codec.pth".to_string()
    });

    let dev = Device::new_metal(0)?;
    let t0 = Instant::now();
    let dac = Dac::load(Path::new(&path), &CodecConfig::default(), &dev)?;
    println!(
        "codec loaded in {:.1}s (frames={frames}, prof={:?})",
        t0.elapsed().as_secs_f64(),
        std::env::var("VOX_DAC_PROF").unwrap_or_else(|_| "off".into())
    );

    let t0 = Instant::now();
    let audio = dac.decode_codes(&synth_codes(32))?;
    dev.synchronize()?;
    println!(
        "warmup 32f: {:.0}ms ({} samples)",
        t0.elapsed().as_secs_f64() * 1e3,
        audio.len()
    );

    let codes = synth_codes(frames);
    for run in 1..=2 {
        dev.synchronize()?;
        let t0 = Instant::now();
        let audio = dac.decode_codes(&codes)?;
        dev.synchronize()?;
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "run{run}: {:.0}ms ({:.1}ms/frame) out {} samples",
            dt * 1e3,
            dt * 1e3 / frames as f64,
            audio.len()
        );
    }
    Ok(())
}
