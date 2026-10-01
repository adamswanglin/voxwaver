//! Chunk-stitching helper (port of `omnivoice/utils/audio.py`'s
//! `cross_fade_chunks`): decoded chunk waveforms are concatenated with a
//! fade-out, a short silence gap and a fade-in at every boundary, so
//! independently generated/decoded chunks don't click at the seams.

use crate::config as cfg;

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
}
