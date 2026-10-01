//! Layer-by-layer parity check for the voice-clone encoder against the
//! Python reference dumps produced by `scripts/omni_clone_dump.py`
//! (default dir `/tmp/omni_clone_ref`).
//!
//!     cargo run -p omnivoice --features metal --example clone_parity -- \
//!         --model-dir ../OmniVoice --ref-dir /tmp/omni_clone_ref
//!
//! Checks (in order): 44.1k -> 24k resample, the 24k -> 16k sinc kernel,
//! 16k resample + semantic pad, HuBERT hidden states, the 13-state mean and
//! `::2` downsample, the semantic conv stack, acoustic branch frame count +
//! pad decision, `fc`, and the RVQ codes (must match exactly).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

use omnivoice::encoder::{acoustic_frames, RefEncoder};
use omnivoice::resample;

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i32(path: &Path) -> Result<Vec<i32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Max absolute difference between a tensor (any device/dtype) and a raw f32
/// dump; also returns the index of the largest deviation.
fn diff(got: &Tensor, want: &[f32]) -> Result<(f32, usize)> {
    let flat = got
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    anyhow::ensure!(
        flat.len() == want.len(),
        "length mismatch: got {} want {}",
        flat.len(),
        want.len()
    );
    let mut best = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in flat.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > best {
            best = d;
            at = i;
        }
    }
    Ok((best, at))
}

fn report(name: &str, d: f32, at: usize, tol: f32) {
    let status = if d <= tol { "ok  " } else { "FAIL" };
    println!("[{status}] {name:<28} max|d| = {d:.3e} (at {at}, tol {tol:.1e})");
}

fn main() -> Result<()> {
    let mut model_dir = PathBuf::from("../OmniVoice");
    let mut ref_dir = PathBuf::from("/tmp/omni_clone_ref");
    let mut force_cpu = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => model_dir = PathBuf::from(args.next().context("--model-dir value")?),
            "--ref-dir" => ref_dir = PathBuf::from(args.next().context("--ref-dir value")?),
            "--cpu" => force_cpu = true,
            other => anyhow::bail!("unknown arg {other}"),
        }
    }

    let device = omnivoice::default_device(force_cpu)?;
    println!("device: {device:?}");
    let enc = RefEncoder::load(&model_dir, &device)?;

    let (wav, sr) = omnivoice::wavio::read_wav(&ref_dir.join("ref_in.wav"))?;
    println!("ref wav: {} samples @ {sr} Hz", wav.len());
    anyhow::ensure!(sr == 44_100, "expected 44.1 kHz reference wav");

    // --- resample 44.1k -> 24k -------------------------------------------
    let x24 = resample::resample(&wav, sr, 24_000);
    let want = read_f32(&ref_dir.join("ref24k.f32"))?;
    anyhow::ensure!(x24.len() == want.len(), "24k length {} vs {}", x24.len(), want.len());
    let (d, at) = diff(&Tensor::from_vec(x24.clone(), x24.len(), &Device::Cpu)?, &want)?;
    report("resample 44.1k->24k", d, at, 1e-4);

    // --- sinc kernel 24k -> 16k ------------------------------------------
    let (kernel, _w, _o, _n) = resample::sinc_kernel(24_000, 16_000);
    let want = read_f32(&ref_dir.join("rsk.f32"))?;
    anyhow::ensure!(kernel.len() == want.len(), "kernel length");
    let (d, at) = diff(&Tensor::from_vec(kernel.clone(), kernel.len(), &Device::Cpu)?, &want)?;
    report("sinc kernel 24k->16k", d, at, 1e-6);

    // --- resample 24k -> 16k ---------------------------------------------
    let x16 = resample::resample(&x24, 24_000, 16_000);
    let want = read_f32(&ref_dir.join("ref16k.f32"))?;
    anyhow::ensure!(x16.len() == want.len(), "16k length {} vs {}", x16.len(), want.len());
    let (d, at) = diff(&Tensor::from_vec(x16.clone(), x16.len(), &Device::Cpu)?, &want)?;
    report("resample 24k->16k", d, at, 1e-4);

    // --- semantic pad 160 --------------------------------------------------
    let mut x16p = vec![0f32; x16.len() + 320];
    x16p[160..160 + x16.len()].copy_from_slice(&x16);
    let want = read_f32(&ref_dir.join("x16_pad.f32"))?;
    let (d, at) = diff(&Tensor::from_vec(x16p.clone(), x16p.len(), &Device::Cpu)?, &want)?;
    report("semantic pad 160", d, at, 1e-6);

    // --- HuBERT per-layer trace (hub_* dumps from omni_hubert_dump.py) ----
    let hubert = enc.hubert();
    let mut step_ok = true;
    let conv = hubert.feature_extract(&x16p)?; // [1, T, 512]
    let conv_t = conv.transpose(1, 2)?.contiguous()?; // [1, 512, T]
    let (d, at) = diff(&conv_t, &read_f32(&ref_dir.join("hub_conv6.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert conv6", d, at, 1e-3);
    let proj = hubert.feature_project(&conv)?;
    let (d, at) = diff(&proj, &read_f32(&ref_dir.join("hub_feat_proj.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert feat_proj", d, at, 1e-3);
    let raw = hubert.pos_conv_raw(&proj)?;
    let (d, at) = diff(&raw, &read_f32(&ref_dir.join("pos_raw.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert pos_conv raw", d, at, 1e-3);

    // --- debug: merged pos weight + grouped-conv variants ------------------
    {
        let (pw, pb) = hubert.pos_weight();
        let (d, at) = diff(pw, &read_f32(&ref_dir.join("pos_w.f32"))?)?;
        report("pos_w merged", d, at, 1e-5);
        let (d, at) = diff(pb, &read_f32(&ref_dir.join("pos_b.f32"))?)?;
        report("pos_b", d, at, 1e-6);

        // raw g/v straight from the safetensors file + step-by-step merge
        {
            use candle_core::safetensors::MmapedSafetensors;
            let path = model_dir.join("audio_tokenizer").join("model.safetensors");
            let st = unsafe { MmapedSafetensors::new(&path) }?;
            let g = st.load(
                "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
                &Device::Cpu,
            )?;
            let vv = st.load(
                "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
                &Device::Cpu,
            )?;
            println!("[info] g {:?} v {:?}", g.dims(), vv.dims());
            let (d, at) = diff(&g, &read_f32(&ref_dir.join("pos_g.f32"))?)?;
            report("pos_g from file", d, at, 0.0);
            let (d, at) = diff(&vv, &read_f32(&ref_dir.join("pos_v.f32"))?)?;
            report("pos_v from file", d, at, 0.0);
            let norm = vv.sqr()?.sum_keepdim(0)?.sum_keepdim(1)?.sqrt()?;
            let (d, at) = diff(&norm, &read_f32(&ref_dir.join("pos_norm.f32"))?)?;
            report("pos_norm cpu", d, at, 1e-5);
            let merged = g.broadcast_mul(&vv)?.broadcast_div(&norm)?;
            let (d, at) = diff(&merged, &read_f32(&ref_dir.join("pos_w.f32"))?)?;
            report("pos_w cpu from file", d, at, 1e-5);
        }

        let xc = proj.transpose(1, 2)?.contiguous()?; // [1, 768, 167]
        let bias = pb.reshape((1, 768, 1))?;
        let want = read_f32(&ref_dir.join("pos_raw.f32"))?;
        let variant = |name: &str, y: candle_core::Result<Tensor>| -> Result<()> {
            let y = y?.broadcast_add(&bias)?;
            let (d, at) = diff(&y, &want)?;
            report(name, d, at, 1e-3);
            Ok(())
        };
        variant("posraw A lib groups", xc.conv1d(pw, 64, 1, 1, 16))?;
        let blockconv = |contig: bool| -> candle_core::Result<Tensor> {
            let blocks = xc.chunk(16, 1)?;
            let kblocks = pw.chunk(16, 0)?;
            let mut ys = Vec::new();
            for (blk, kb) in blocks.iter().zip(kblocks.iter()) {
                let blk = if contig { blk.contiguous()? } else { blk.clone() };
                ys.push(blk.conv1d(kb, 64, 1, 1, 1)?);
            }
            Tensor::cat(&ys, 1)
        };
        variant("posraw B manual contig", blockconv(true))?;
        variant("posraw C manual view", blockconv(false))?;
    }
    let pos = hubert.pos_conv(&proj)?;
    let (d, at) = diff(&pos, &read_f32(&ref_dir.join("hub_pos_conv.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert pos_conv", d, at, 1e-3);
    let mut x = hubert.encoder_norm(&proj.add(&pos)?)?;
    let (d, at) = diff(&x, &read_f32(&ref_dir.join("hub_enc_ln.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert enc_ln (=hs0)", d, at, 1e-3);
    for i in 0..6 {
        x = hubert.encoder_layer(i, &x)?;
        if i == 0 {
            let (d, at) = diff(&x, &read_f32(&ref_dir.join("hub_layer0.f32"))?)?;
            step_ok &= d < 1e-3;
            report("hubert layer0", d, at, 1e-3);
        }
    }
    let (d, at) = diff(&x, &read_f32(&ref_dir.join("hub_layer5.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert layer5", d, at, 1e-3);
    for i in 6..12 {
        x = hubert.encoder_layer(i, &x)?;
    }
    let (d, at) = diff(&x, &read_f32(&ref_dir.join("hub_layer11.f32"))?)?;
    step_ok &= d < 1e-3;
    report("hubert layer11", d, at, 1e-3);
    println!("[info] hubert layer trace clean = {step_ok}");

    // --- HuBERT hidden states [13, 167, 768] ------------------------------
    let hidden = enc.hubert_hidden(&x16p)?;
    let want = read_f32(&ref_dir.join("hub_hidden.f32"))?;
    let (d, at) = diff(&hidden, &want)?;
    report("hubert hidden states", d, at, 1e-3);

    // --- mean over 13 + `::2` ---------------------------------------------
    let mean = hidden.mean(0)?; // [T, 768]
    let want = read_f32(&ref_dir.join("sem_mean.f32"))?;
    let (d, at) = diff(&mean, &want)?;
    report("mean of hidden states", d, at, 1e-3);

    let t = mean.dim(0)?;
    let n = t.div_ceil(2);
    let idx: Vec<u32> = (0..n).map(|i| (i * 2) as u32).collect();
    let idx = Tensor::from_vec(idx, n, &device)?;
    let feat = mean.index_select(&idx, 0)?; // [T_sem, 768]
    let want = read_f32(&ref_dir.join("sem_feat.f32"))?;
    let (d, at) = diff(&feat, &want)?;
    report("semantic ::2 downsample", d, at, 1e-3);

    // --- semantic encoder --------------------------------------------------
    let e_sem = enc.semantic_features(&x16)?; // [1, 768, T_sem]
    let want = read_f32(&ref_dir.join("e_semantic.f32"))?;
    let (d, at) = diff(&e_sem, &want)?;
    report("semantic encoder", d, at, 1e-3);

    // --- acoustic branch + pad decision ------------------------------------
    let t_sem = e_sem.dim(2)?;
    let frames = acoustic_frames(x24.len());
    let use_pad = frames != t_sem;
    println!("[info] acoustic frames {frames} vs semantic {t_sem} -> pad {use_pad}");
    let e_ac = enc.acoustic_branch(&x24, use_pad)?; // [1, 256, T]
    let want = read_f32(&ref_dir.join("e_acoustic.f32"))?;
    let (d, at) = diff(&e_ac, &want)?;
    report("acoustic encoder", d, at, 1e-3);

    // --- fc ----------------------------------------------------------------
    let emb = enc.fc_quantize(&e_ac, &e_sem)?; // [1, 1024, T]
    let want = read_f32(&ref_dir.join("emb_fc.f32"))?;
    let (d, at) = diff(&emb, &want)?;
    report("fc", d, at, 1e-3);

    // --- RVQ codes (exact) -------------------------------------------------
    let codes = enc.rvq_encode(&emb)?;
    let want = read_i32(&ref_dir.join("codes.i32"))?;
    let tt = codes[0].len();
    anyhow::ensure!(want.len() == 8 * tt, "codes length {} vs {}", want.len(), 8 * tt);
    let mut mismatches = 0usize;
    let mut per_cb = [0usize; 8];
    for cb in 0..8 {
        for i in 0..tt {
            if codes[cb][i] != want[cb * tt + i] as u32 {
                mismatches += 1;
                per_cb[cb] += 1;
            }
        }
    }
    println!("[{}] RVQ codes {mismatches} mismatches over 8 x {tt} (per codebook {per_cb:?})",
        if mismatches == 0 { "ok  " } else { "FAIL" });

    // --- full encode() end to end ------------------------------------------
    let codes2 = enc.encode(&wav, sr)?;
    anyhow::ensure!(
        codes2.len() == codes.len() && codes2.iter().zip(&codes).all(|(a, b)| a == b),
        "encode() != step-by-step pipeline"
    );
    println!("[ok  ] encode() end-to-end identical");
    Ok(())
}
