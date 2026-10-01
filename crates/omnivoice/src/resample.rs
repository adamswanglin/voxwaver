//! CPU resampler replicating `torchaudio.functional.resample`'s default
//! `sinc_interp_hann` path (`lowpass_filter_width=6`, `rolloff=0.99`).
//!
//! Used twice in the clone path: reference wav -> 24 kHz (like the vllm-omni
//! pipeline) and 24 kHz -> 16 kHz (inside `HiggsAudioV2TokenizerModel.encode`).
//! The kernel is evaluated in f32 like torch does (waveform dtype), with the
//! window/scale constants derived in f64 first.

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Build the polyphase sinc kernel: `[new_frac, 2*width + orig_frac]` plus
/// `width` (mirrors `_get_sinc_resample_kernel`).
pub fn sinc_kernel(orig_freq: u32, new_freq: u32) -> (Vec<f32>, usize, usize, usize) {
    let g = gcd(orig_freq, new_freq);
    let orig = (orig_freq / g) as usize;
    let new = (new_freq / g) as usize;
    let lowpass_width = 6f64;
    let rolloff = 0.99f64;
    let base_freq = orig.min(new) as f64 * rolloff;
    let width = ((lowpass_width * orig as f64) / base_freq).ceil() as usize;
    let klen = 2 * width + orig;
    let base_f32 = base_freq as f32;
    let scale = (base_freq / orig as f64) as f32;

    let mut kernel = vec![0f32; new * klen];
    for p in 0..new {
        for j in 0..klen {
            // arange(-width, width + orig, f32) / orig
            let idx = (j as i64 - width as i64) as f32 / orig as f32;
            // arange(0, -new, -1, f32) / new + idx
            let t = (-(p as f32) / new as f32) + idx;
            let t = (t * base_f32).clamp(-(lowpass_width as f32), lowpass_width as f32);
            // hann window evaluated at the kernel positions:
            // cos(t * pi / lowpass_filter_width / 2) ** 2
            let win = ((t * std::f32::consts::PI) / lowpass_width as f32 / 2.0).cos();
            let tpi = t * std::f32::consts::PI;
            let sinc = if tpi == 0.0 { 1.0 } else { tpi.sin() / tpi };
            kernel[p * klen + j] = sinc * (win * win * scale);
        }
    }
    (kernel, width, orig, new)
}

/// Resample `x` from `orig_freq` to `new_freq` (mono f32).
pub fn resample(x: &[f32], orig_freq: u32, new_freq: u32) -> Vec<f32> {
    if orig_freq == new_freq || x.is_empty() {
        return x.to_vec();
    }
    let (kernel, width, orig, new) = sinc_kernel(orig_freq, new_freq);
    let klen = kernel.len() / new;
    let l = x.len();
    // F.pad(x, (width, width + orig))
    let mut padded = vec![0f32; l + 2 * width + orig];
    padded[width..width + l].copy_from_slice(x);
    // target_length = ceil(new_freq * length / orig_freq)
    let target = (new as f64 * l as f64 / orig as f64).ceil() as usize;
    let mut out = Vec::with_capacity(target);
    for k in 0..target {
        let m = k / new; // conv output frame
        let p = k % new; // polyphase branch
        let row = &kernel[p * klen..(p + 1) * klen];
        let start = m * orig;
        let mut acc = 0f32;
        for (j, &kv) in row.iter().enumerate() {
            acc += padded[start + j] * kv;
        }
        out.push(acc);
    }
    out
}
