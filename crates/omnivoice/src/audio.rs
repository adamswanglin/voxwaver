//! Output post-processing (port of `omnivoice/utils/audio.py`):
//! `cross_fade_chunks` concatenates decoded chunk waveforms with a fade-out,
//! a short silence gap and a fade-in at every boundary; `remove_silence`,
//! `normalize_volume` and `fade_and_pad` mirror the reference pipeline's
//! `_post_process_audio` stages. The silence detection is a close
//! reimplementation of the pydub calls the Python code makes (int16 dBFS
//! windows, 10 ms seek step), not a bit-exact port.

use crate::config as cfg;

/// A window is silent when its int16 RMS is at/below -50 dBFS, i.e.
/// `32768 * 10^(-50/20)` ≈ 103.7 (pydub's `silence_thresh=-50`).
const SILENCE_RMS_I16: f64 = 103.626;

/// Concatenate decoded chunk waveforms with a fade-out / silence gap /
/// fade-in at each boundary (linear ramps of `CHUNK_FADE_SAMPLES`, gap of
/// `CHUNK_GAP_SAMPLES`; the reference splits its 0.3 s `silence_duration`
/// into three equal parts).
pub fn cross_fade_chunks(chunks: &[Vec<f32>]) -> Vec<f32> {
    match chunks {
        [] => Vec::new(),
        [only] => only.clone(),
        _ => {
            let gap = cfg::CHUNK_GAP_SAMPLES;
            let total: usize =
                chunks.iter().map(Vec::len).sum::<usize>() + gap * (chunks.len() - 1);
            let mut out: Vec<f32> = Vec::with_capacity(total);
            out.extend_from_slice(&chunks[0]);
            for chunk in &chunks[1..] {
                // Fade the tail of everything merged so far.
                let fout = cfg::CHUNK_FADE_SAMPLES.min(out.len());
                if fout > 1 {
                    let base = out.len() - fout;
                    for (i, s) in out[base..].iter_mut().enumerate() {
                        *s *= 1.0 - i as f32 / (fout - 1) as f32;
                    }
                }
                out.resize(out.len() + gap, 0.0);
                // Fade the head of the next chunk.
                let fin = cfg::CHUNK_FADE_SAMPLES.min(chunk.len());
                if fin > 1 {
                    out.extend(
                        chunk[..fin]
                            .iter()
                            .enumerate()
                            .map(|(i, &s)| s * i as f32 / (fin - 1) as f32),
                    );
                } else {
                    out.extend_from_slice(&chunk[..fin]);
                }
                out.extend_from_slice(&chunk[fin..]);
            }
            out
        }
    }
}

/// Quantize to the int16 magnitudes pydub operates on.
fn quantize_i16(wav: &[f32]) -> Vec<f32> {
    wav.iter().map(|&s| (s * 32768.0).clamp(-32768.0, 32767.0)).collect()
}

fn window_is_silent(q: &[f32]) -> bool {
    if q.is_empty() {
        return true;
    }
    let sum: f64 = q.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / q.len() as f64).sqrt() <= SILENCE_RMS_I16
}

/// `pydub.silence.detect_nonsilent` approximation: returns non-silent sample
/// ranges, scanning silence in `min_sil_ms` windows stepped by 10 ms.
fn nonsilent_ranges(q: &[f32], sample_rate: u32, min_sil_ms: u32) -> Vec<(usize, usize)> {
    let win = min_sil_ms as usize * sample_rate as usize / 1000;
    let step = 10usize * sample_rate as usize / 1000;
    if q.len() < win || step == 0 {
        // Too short to hold a silence window: everything is non-silent.
        return if q.is_empty() { Vec::new() } else { vec![(0, q.len())] };
    }

    // Silence ranges [start, start + win) for every silent window start.
    let mut silences: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start + win <= q.len() {
        if window_is_silent(&q[start..start + win]) {
            match silences.last_mut() {
                Some(last) if start <= last.1 => last.1 = start + win,
                _ => silences.push((start, start + win)),
            }
        }
        start += step;
    }

    // Complement into non-silent ranges.
    let mut out = Vec::new();
    let mut pos = 0usize;
    for (s, e) in silences {
        if s > pos {
            out.push((pos, s));
        }
        pos = e;
    }
    if pos < q.len() {
        out.push((pos, q.len()));
    }
    out
}

/// `pydub.silence.detect_leading_silence` with `chunk_size=1` ms.
fn leading_silence_samples(q: &[f32], sample_rate: u32) -> usize {
    let chunk = sample_rate as usize / 1000;
    let mut trim = 0usize;
    while trim + chunk <= q.len() && window_is_silent(&q[trim..trim + chunk]) {
        trim += chunk;
    }
    trim
}

/// Port of `remove_silence`: drop middle silences longer than `mid_sil_ms`
/// (keeping up to `mid_sil_ms` of silence around each kept segment) and trim
/// edge silences down to `lead_sil_ms` / `trail_sil_ms`.
pub fn remove_silence(
    wav: &[f32],
    sample_rate: u32,
    mid_sil_ms: u32,
    lead_sil_ms: u32,
    trail_sil_ms: u32,
) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let q = quantize_i16(wav);
    let mut out: Vec<f32> = Vec::with_capacity(wav.len());
    if mid_sil_ms > 0 {
        // Each non-silent range keeps `keep` of silence on both sides
        // (clamped, merged when the expansions overlap).
        let keep = mid_sil_ms as usize * sample_rate as usize / 1000;
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (s, e) in nonsilent_ranges(&q, sample_rate, mid_sil_ms) {
            let s2 = s.saturating_sub(keep);
            let e2 = (e + keep).min(wav.len());
            match ranges.last_mut() {
                Some(last) if s2 <= last.1 => last.1 = last.1.max(e2),
                _ => ranges.push((s2, e2)),
            }
        }
        for (s, e) in ranges {
            out.extend_from_slice(&wav[s..e]);
        }
    } else {
        out.extend_from_slice(wav);
    }

    // Edge trim on the surviving audio.
    let q2 = quantize_i16(&out);
    let lead = leading_silence_samples(&q2, sample_rate);
    let keep_lead = lead.saturating_sub(lead_sil_ms as usize * sample_rate as usize / 1000);
    let q2_rev: Vec<f32> = q2.iter().rev().copied().collect();
    let trail = leading_silence_samples(&q2_rev, sample_rate);
    let keep_trail = trail.saturating_sub(trail_sil_ms as usize * sample_rate as usize / 1000);
    let end = out.len().saturating_sub(keep_trail);
    out[keep_lead..end].to_vec()
}

/// Port of `fade_and_pad_audio`: linear fade-in/out of `fade_duration` per
/// edge, then `pad_duration` of silence per edge. An empty input is returned
/// unchanged.
pub fn fade_and_pad(wav: &[f32], pad_duration: f64, fade_duration: f64, sample_rate: u32) -> Vec<f32> {
    if wav.is_empty() {
        return Vec::new();
    }
    let fade_samples = (fade_duration * sample_rate as f64) as usize;
    let pad_samples = (pad_duration * sample_rate as f64) as usize;
    let mut out = wav.to_vec();
    if fade_samples > 0 {
        let k = fade_samples.min(out.len() / 2);
        if k > 1 {
            for i in 0..k {
                // np.linspace(0, 1, k) / linspace(1, 0, k) over each edge.
                out[i] *= i as f32 / (k - 1) as f32;
                let n = out.len() - 1 - i;
                out[n] *= i as f32 / (k - 1) as f32;
            }
        }
    }
    if pad_samples > 0 {
        let mut padded = Vec::with_capacity(out.len() + 2 * pad_samples);
        padded.extend(std::iter::repeat_n(0.0, pad_samples));
        padded.extend_from_slice(&out);
        padded.extend(std::iter::repeat_n(0.0, pad_samples));
        out = padded;
    }
    out
}

/// `_post_process_audio`'s volume stage: with a reference RMS below 0.1 the
/// output is scaled to match it; without a reference the output is
/// peak-normalized to 0.5; otherwise it is left untouched.
pub fn normalize_volume(wav: &mut [f32], ref_rms: Option<f64>) {
    match ref_rms {
        Some(r) if r > 0.0 && r < 0.1 => {
            let gain = (r / 0.1) as f32;
            for s in wav.iter_mut() {
                *s *= gain;
            }
        }
        Some(_) => {}
        None => {
            let peak = wav.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            if peak > 1e-6 {
                let gain = 0.5 / peak;
                for s in wav.iter_mut() {
                    *s *= gain;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_chunk_is_unchanged() {
        let c = vec![vec![1.0f32, -1.0, 0.5]];
        assert_eq!(cross_fade_chunks(&c), c[0]);
    }

    #[test]
    fn boundaries_fade_and_gap() {
        let a = vec![1.0f32; cfg::CHUNK_FADE_SAMPLES];
        let b = vec![1.0f32; cfg::CHUNK_FADE_SAMPLES];
        let out = cross_fade_chunks(&[a, b]);
        assert_eq!(out.len(), 2 * cfg::CHUNK_FADE_SAMPLES + cfg::CHUNK_GAP_SAMPLES);
        // tail of the first chunk ramps down to 0
        assert!(out[cfg::CHUNK_FADE_SAMPLES - 1].abs() < 1e-6);
        // the gap is silence
        assert!(out[cfg::CHUNK_FADE_SAMPLES..cfg::CHUNK_FADE_SAMPLES + cfg::CHUNK_GAP_SAMPLES]
            .iter()
            .all(|&s| s == 0.0));
        // head of the second chunk starts at 0 and ramps up
        assert!(out[cfg::CHUNK_FADE_SAMPLES + cfg::CHUNK_GAP_SAMPLES].abs() < 1e-6);
        assert!((out[out.len() - 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn short_chunks_still_stitch() {
        let out = cross_fade_chunks(&[vec![1.0f32], vec![1.0f32]]);
        assert_eq!(out.len(), 2 + cfg::CHUNK_GAP_SAMPLES);
    }

    // ---- fade_and_pad ----

    #[test]
    fn fade_and_pad_adds_edges() {
        let wav = vec![0.5f32; cfg::SAMPLE_RATE as usize];
        let out = fade_and_pad(&wav, 0.1, 0.1, cfg::SAMPLE_RATE);
        let pad = 2400usize;
        let fade = 2400usize;
        assert_eq!(out.len(), wav.len() + 2 * pad);
        assert_eq!(out[..pad].iter().fold(0.0f32, |m, &s| m.max(s.abs())), 0.0);
        assert!(out[pad].abs() < 1e-6, "fade-in must start at 0");
        assert!((out[pad + fade] - 0.5).abs() < 1e-4, "fade-in must reach full");
        // The last sample before the trailing pad is faded out to ~0; full
        // amplitude holds until one fade-length before it.
        assert!(out[out.len() - 1 - pad].abs() < 1e-3, "fade-out must end at 0");
        assert!((out[out.len() - pad - fade - 1] - 0.5).abs() < 1e-4);
        assert_eq!(out[out.len() - pad..].iter().fold(0.0f32, |m, &s| m.max(s.abs())), 0.0);
    }

    // ---- normalize_volume ----

    #[test]
    fn normalize_volume_small_ref_rms_scales_down() {
        let mut wav = vec![0.5f32; 100];
        normalize_volume(&mut wav, Some(0.05));
        assert!((wav[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn normalize_volume_none_peak_normalizes() {
        let mut wav = vec![0.25f32, -0.5, 0.1];
        normalize_volume(&mut wav, None);
        assert!((wav.iter().fold(0.0f32, |m, &s| m.max(s.abs())) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalize_volume_large_ref_rms_untouched() {
        let mut wav = vec![0.25f32, -0.5, 0.1];
        let before = wav.clone();
        normalize_volume(&mut wav, Some(0.3));
        assert_eq!(wav, before);
    }

    // ---- remove_silence ----

    /// A tone well above the -50 dBFS window threshold.
    fn tone(samples: usize) -> Vec<f32> {
        vec![0.5f32; samples]
    }

    #[test]
    fn remove_silence_noop_on_nonsilent() {
        let wav = tone(cfg::SAMPLE_RATE as usize);
        let out = remove_silence(&wav, cfg::SAMPLE_RATE, 500, 100, 100);
        assert_eq!(out, wav);
    }

    #[test]
    fn remove_silence_cuts_long_mid_gap() {
        let sr = cfg::SAMPLE_RATE as usize;
        // 1 s tone + 2 s silence + 1 s tone; keep_silence is 500 ms.
        let mut wav = tone(sr);
        wav.extend(std::iter::repeat_n(0.0, 2 * sr));
        wav.extend(tone(sr));
        let out = remove_silence(&wav, cfg::SAMPLE_RATE, 500, 100, 100);
        // 2 s gap shrinks to ~1 s (500 ms kept on each side of the cut).
        let expected = 4 * sr - sr;
        assert!((out.len() as i64 - expected as i64).abs() < (sr / 10) as i64,
            "len {} vs expected {expected}", out.len());
    }

    #[test]
    fn remove_silence_keeps_short_gaps() {
        let sr = cfg::SAMPLE_RATE as usize;
        // 1 s tone + 200 ms silence + 1 s tone: below the 500 ms threshold.
        let mut wav = tone(sr);
        wav.extend(std::iter::repeat_n(0.0, sr / 5));
        wav.extend(tone(sr));
        let out = remove_silence(&wav, cfg::SAMPLE_RATE, 500, 100, 100);
        assert_eq!(out.len(), wav.len());
    }

    #[test]
    fn remove_silence_trims_edges() {
        let sr = cfg::SAMPLE_RATE as usize;
        // 1 s silence + 1 s tone + 1 s silence.
        let mut wav: Vec<f32> = std::iter::repeat_n(0.0, sr).collect();
        wav.extend(tone(sr));
        wav.extend(std::iter::repeat_n(0.0, sr));
        let out = remove_silence(&wav, cfg::SAMPLE_RATE, 500, 100, 100);
        // Edge silences shrink to 100 ms each.
        let expected = sr + 2 * (sr / 10);
        assert!((out.len() as i64 - expected as i64).abs() < (sr / 10) as i64,
            "len {} vs expected {expected}", out.len());
    }
}
