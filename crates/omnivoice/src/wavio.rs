//! WAV I/O (hound), mirroring cosyvoice's helper.

use anyhow::{Context, Result};

/// Read a wav file as mono f32 in [-1, 1] (any bit depth hound supports).
pub fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .context("float samples")?,
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .collect::<std::result::Result<Vec<i16>, _>>()
            .context("i16 samples")?
            .into_iter()
            .map(|s| s as f32 / 32768.0)
            .collect(),
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .collect::<std::result::Result<Vec<i32>, _>>()
            .context("i24 samples")?
            .into_iter()
            .map(|s| s as f32 / 8388608.0)
            .collect(),
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .collect::<std::result::Result<Vec<i32>, _>>()
            .context("i32 samples")?
            .into_iter()
            .map(|s| s as f32 / 2147483648.0)
            .collect(),
        (hound::SampleFormat::Int, 8) => reader
            .samples::<i8>()
            .collect::<std::result::Result<Vec<i8>, _>>()
            .context("i8 samples")?
            .into_iter()
            .map(|s| s as f32 / 128.0)
            .collect(),
        (sf, bits) => anyhow::bail!("unsupported wav format: {sf:?} {bits}-bit"),
    };
    let mut samples = samples;
    if channels > 1 {
        // average down-mix
        let frames = samples.len() / channels;
        samples = (0..frames)
            .map(|i| samples[i * channels..(i + 1) * channels].iter().sum::<f32>() / channels as f32)
            .collect();
    }
    Ok((samples, spec.sample_rate))
}

/// Write mono f32 samples as 16-bit PCM.
pub fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {}", path.display()))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}
