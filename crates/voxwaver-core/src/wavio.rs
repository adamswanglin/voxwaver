//! Minimal WAV I/O: mono mixdown reading (PCM 8/16/24/32-bit and float32),
//! Kaiser-windowed sinc resampling to the codec rate, and WAV writing.
use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Read a WAV file and mix all channels down to mono f32 in [-1, 1].
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader = WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    if channels != 1 && channels != 2 {
        bail!("unsupported channel count {} in {}", channels, path.display());
    }
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let scale = 2f32.powi(bits as i32 - 1);
            match bits {
                8 => reader
                    .into_samples::<i8>()
                    .map(|s| s.unwrap_or(0) as f32 / scale)
                    .collect(),
                16 => reader
                    .into_samples::<i16>()
                    .map(|s| s.unwrap_or(0) as f32 / scale)
                    .collect(),
                24 | 32 => reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / scale)
                    .collect(),
                b => bail!("unsupported PCM bit depth {b} in {}", path.display()),
            }
        }
    };
    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks(2)
            .map(|c| (c[0] + c[1]) * 0.5)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Modified Bessel function of the first kind, order 0 (series).
fn i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x * x / 4.0;
    for k in 1..32 {
        term *= half / (k as f64 * k as f64);
        sum += term;
        if term < 1e-14 * sum {
            break;
        }
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Resample with a Kaiser-windowed sinc low-pass (~90 dB stopband).
/// `cutoff_hz` is in cycles per source sample.
pub fn resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = (input.len() as f64 * ratio).floor() as usize;
    // anti-alias cutoff at the smaller Nyquist, with a 5% transition guard
    let cutoff = 0.475 * ratio.min(1.0); // cycles per source sample
    let beta = 8.6; // ~ -80 dB sidelobes
    let half_width = 32.0 / (2.0 * cutoff); // source samples on each side
    let i0_beta = i0(beta);

    let tap = |offset: f64| -> f64 {
        // offset in source samples from the interpolation point
        let u = offset / half_width;
        let window = i0(beta * (1.0 - u * u).sqrt()) / i0_beta;
        2.0 * cutoff * sinc(2.0 * cutoff * offset) * window
    };

    let n_in = input.len();
    let mut out = Vec::with_capacity(out_len);
    for n in 0..out_len {
        let center = n as f64 / ratio;
        let lo = (center - half_width).ceil() as i64;
        let hi = (center + half_width).floor() as i64;
        let mut acc = 0f64;
        for k in lo..=hi {
            let idx = k.clamp(0, n_in as i64 - 1) as usize;
            acc += input[idx] as f64 * tap(k as f64 - center);
        }
        out.push(acc as f32);
    }
    out
}

/// Write mono samples as a WAV file: float32, or 16-bit PCM when `pcm16`.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, pcm16: bool) -> Result<()> {
    let spec = if pcm16 {
        WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        }
    } else {
        WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    if pcm16 {
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            writer.write_sample(v)?;
        }
    } else {
        for &s in samples {
            writer.write_sample(s)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity_and_length() {
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.05).sin()).collect();
        let y = resample(&x, 44100, 44100);
        assert_eq!(y.len(), 1000);
        let z = resample(&x, 44100, 22050);
        assert_eq!(z.len(), 500);
        let w = resample(&x, 22050, 44100);
        assert_eq!(w.len(), 2000);
    }

    #[test]
    fn resample_preserves_dc_and_tone() {
        // constant signal stays constant
        let x = vec![0.5f32; 4410];
        let y = resample(&x, 48000, 44100);
        let mid = &y[100..y.len() - 100];
        let err: f32 = mid.iter().map(|v| (v - 0.5).abs()).sum::<f32>() / mid.len() as f32;
        assert!(err < 1e-3, "dc error {err}");
        // a low-frequency sine survives downsampling; step so that the output
        // index n * 44100/48000 lands exactly on a sample (gcd = 300)
        let t: Vec<f32> = (0..9600).map(|i| (i as f32 * 2.0 * 3.14159 * 100.0 / 48000.0).sin()).collect();
        let y = resample(&t, 48000, 44100);
        let mut max_err = 0f32;
        for n in (160..t.len() - 160).step_by(160) {
            let expected = (n as f32 * 2.0 * 3.14159 * 100.0 / 48000.0).sin();
            let got = y[n * 44100 / 48000];
            max_err = max_err.max((expected - got).abs());
        }
        assert!(max_err < 1e-3, "tone error {max_err}");
    }

    #[test]
    fn wav_roundtrip() {
        let dir = std::env::temp_dir().join("voxwaver-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rt.wav");
        let x: Vec<f32> = (0..256).map(|i| (i as f32 / 64.0).sin() * 0.5).collect();
        write_wav(&p, &x, 44100, false).unwrap();
        let (y, sr) = read_wav_mono(&p).unwrap();
        assert_eq!(sr, 44100);
        assert_eq!(y.len(), 256);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-6);
        }
        let p16 = dir.join("rt16.wav");
        write_wav(&p16, &x, 44100, true).unwrap();
        let (y16, _) = read_wav_mono(&p16).unwrap();
        assert_eq!(y16.len(), 256);
    }
}
