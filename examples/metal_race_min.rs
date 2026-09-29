//! Self-contained repro of the candle 0.11 Metal corruption on long DAC-style
//! decodes. candle_core ONLY -- no voxwaver code.
//!
//! Findings that shaped this repro:
//!   * a single large conv1d, a chain of them, an isolated conv_transpose1d, and
//!     even a full single decoder block are ALL bit-exact at 391 frames;
//!   * the corruption only appears once the WHOLE decoder stack runs, i.e. when
//!     the accumulated >2 GiB buffers of dec0..dec3 push the Metal buffer pool
//!     into recycling buffers that a still-in-flight command buffer references.
//!   * inserting a device->host sync between ops (commit + waitUntilCompleted)
//!     drains the queue and HIDES it -- the signature of a synchronization race.
//!
//! This builds a random-weight 4-block decoder (rates [8,8,4,2], channels
//! 1536->768->384->192->96) mirroring the modded-DAC decoder, and compares
//! CPU vs Metal(no-sync) vs Metal(with-sync) at 390 vs 391 frames.
//!
//! Run:  cargo run --release --features metal --example metal_race_min [frames]
use candle_core::{Device, Tensor};

const DILS: [usize; 3] = [1, 3, 9];

fn stats_corr(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    let av = a.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let bv = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let n = av.len().min(bv.len());
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for i in 0..n {
        sa += av[i] as f64;
        sb += bv[i] as f64;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut cov, mut va, mut vb) = (0.0f64, 0.0, 0.0);
    for i in 0..n {
        let x = av[i] as f64 - ma;
        let y = bv[i] as f64 - mb;
        cov += x * y;
        va += x * x;
        vb += y * y;
    }
    Ok(cov / (va.sqrt() * vb.sqrt()).max(1e-30))
}

fn extra_padding(length: usize, kernel: usize, stride: usize, pad_total: usize) -> usize {
    let n_frames = (length as f64 - kernel as f64 + pad_total as f64) / stride as f64 + 1.0;
    let ideal = ((n_frames.ceil() as i64 - 1) * stride as i64 + (kernel - pad_total) as i64) as usize;
    ideal.saturating_sub(length)
}

/// CausalConvNet: left-pad (k_eff-stride), right-pad extra, dilated conv1d, bias.
/// out_c is derived from the weight so non-square convs (conv_in/out) work too.
fn convw(x: &Tensor, w: &Tensor, b: &Tensor, k: usize, dil: usize) -> anyhow::Result<Tensor> {
    let out_c = w.dim(0)?;
    let k_eff = (k - 1) * dil + 1;
    let l = x.dim(2)?;
    let pad_left = k_eff - 1; // stride 1
    let extra = extra_padding(l, k_eff, 1, pad_left);
    let xp = x.pad_with_zeros(2, pad_left, extra)?;
    let y = xp.conv1d(w, 0, 1, dil, 1)?;
    Ok(y.broadcast_add(&b.reshape((1, out_c, 1))?)?)
}

/// CausalTransConvNet: conv_transpose1d (upsample by stride) + bias + right crop.
fn transconv(x: &Tensor, w: &Tensor, b: &Tensor, k: usize, stride: usize) -> anyhow::Result<Tensor> {
    let out_c = w.dim(1)?;
    let y = x.conv_transpose1d(w, 0, 0, stride, 1, 1)?;
    let y = y.broadcast_add(&b.reshape((1, out_c, 1))?)?;
    let crop = k - stride;
    let l = y.dim(2)?;
    Ok(y.narrow(2, 0, l - crop)?)
}

fn snake(x: &Tensor, alpha: &Tensor, c: usize) -> anyhow::Result<Tensor> {
    let a = alpha.reshape((1, c, 1))?;
    let ax = x.broadcast_mul(&a)?;
    let s2 = ax.sin()?.mul(&ax.sin()?)?;
    let denom = a.affine(1.0, 1e-9)?;
    Ok(x.add(&s2.broadcast_div(&denom)?)?)
}

// ---- random decoder weights (built once on CPU, moved per device) ----
struct RuW {
    sa: Tensor,
    w1: Tensor,
    b1: Tensor,
    sb: Tensor,
    w2: Tensor,
    b2: Tensor,
}
struct BlockW {
    snake_a: Tensor,
    tw: Tensor,
    tb: Tensor,
    stride: usize,
    rus: Vec<RuW>,
}
struct DecW {
    conv_in_w: Tensor,
    conv_in_b: Tensor,
    blocks: Vec<BlockW>,
    snake_out: Tensor,
    conv_out_w: Tensor,
    conv_out_b: Tensor,
}

fn rnd(cpu: &Device, std: f32, shape: &[usize]) -> candle_core::Result<Tensor> {
    Tensor::randn(0f32, std, shape, cpu)
}

fn build_weights(cpu: &Device, latent: usize, dim: usize) -> anyhow::Result<DecW> {
    let rates = [8usize, 8, 4, 2];
    let conv_in_w = rnd(cpu, 0.02, &[dim, latent, 7])?;
    let conv_in_b = rnd(cpu, 0.02, &[dim])?;
    let mut ch = dim; // 1536
    let mut blocks = Vec::new();
    for &s in rates.iter() {
        let out = ch / 2;
        let snake_a = rnd(cpu, 0.1, &[ch])?;
        let tw = rnd(cpu, 0.02, &[ch, out, 2 * s])?;
        let tb = rnd(cpu, 0.02, &[out])?;
        let mut rus = Vec::new();
        for _ in 0..3 {
            rus.push(RuW {
                sa: rnd(cpu, 0.1, &[out])?,
                w1: rnd(cpu, 0.02, &[out, out, 7])?,
                b1: rnd(cpu, 0.02, &[out])?,
                sb: rnd(cpu, 0.1, &[out])?,
                w2: rnd(cpu, 0.02, &[out, out, 1])?,
                b2: rnd(cpu, 0.02, &[out])?,
            });
        }
        blocks.push(BlockW { snake_a, tw, tb, stride: s, rus });
        ch = out;
    }
    let snake_out = rnd(cpu, 0.1, &[ch])?;
    let conv_out_w = rnd(cpu, 0.02, &[1, ch, 7])?;
    let conv_out_b = rnd(cpu, 0.02, &[1])?;
    Ok(DecW { conv_in_w, conv_in_b, blocks, snake_out, conv_out_w, conv_out_b })
}

fn mv(t: &Tensor, dev: &Device) -> candle_core::Result<Tensor> {
    t.to_device(dev)
}

/// Run the full decoder on `dev`. When `sync` is true, force a device->host read
/// after every heavy op (this is what the model's write_npy dumps did).
fn run_decoder(x: &Tensor, w: &DecW, dev: &Device, dim: usize, sync: bool) -> anyhow::Result<Tensor> {
    let s = |t: &Tensor| {
        if sync {
            let _ = t.to_device(&Device::Cpu);
        }
    };
    let mut y = convw(&mv(x, dev)?, &mv(&w.conv_in_w, dev)?, &mv(&w.conv_in_b, dev)?, 7, 1)?;
    s(&y);
    let mut ch = dim;
    for b in &w.blocks {
        y = snake(&y, &mv(&b.snake_a, dev)?, ch)?;
        y = transconv(&y, &mv(&b.tw, dev)?, &mv(&b.tb, dev)?, 2 * b.stride, b.stride)?;
        s(&y);
        ch /= 2;
        for (i, ru) in b.rus.iter().enumerate() {
            let h = convw(&snake(&y, &mv(&ru.sa, dev)?, ch)?, &mv(&ru.w1, dev)?, &mv(&ru.b1, dev)?, 7, DILS[i])?;
            s(&h);
            let h = convw(&snake(&h, &mv(&ru.sb, dev)?, ch)?, &mv(&ru.w2, dev)?, &mv(&ru.b2, dev)?, 1, 1)?;
            s(&h);
            let pad = y.dim(2)? as i64 - h.dim(2)? as i64;
            y = if pad > 0 {
                y.narrow(2, 0, y.dim(2)? - pad as usize)?.add(&h)?
            } else {
                y.add(&h)?
            };
        }
    }
    y = convw(&snake(&y, &mv(&w.snake_out, dev)?, ch)?, &mv(&w.conv_out_w, dev)?, &mv(&w.conv_out_b, dev)?, 7, 1)?;
    let y = y.tanh()?;
    Ok(y)
}

fn main() -> anyhow::Result<()> {
    let cpu = Device::Cpu;
    let mtl = Device::new_metal(0)?;
    let (latent, dim) = (1024usize, 1536usize);
    let frames: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(391);
    let w = build_weights(&cpu, latent, dim)?;
    let l = frames * 4; // decoder input length (post-quantizer-upsample)
    let x = Tensor::randn(0f32, 0.5f32, (1, latent, l), &cpu)?;

    let y_cpu = run_decoder(&x, &w, &cpu, dim, false)?;
    let y_sync = run_decoder(&x, &w, &mtl, dim, true)?;
    let y_nosync = run_decoder(&x, &w, &mtl, dim, false)?;

    let c_ns = stats_corr(&y_cpu, &y_nosync)?;
    let c_s = stats_corr(&y_cpu, &y_sync)?;
    println!("full-decoder frames={frames} (out len {}):", y_cpu.dim(2)?);
    println!("  Metal no-sync   corr={c_ns:.4}  [{}]", if c_ns < 0.99 { "BROKEN" } else { "ok" });
    println!("  Metal with-sync corr={c_s:.4}  [{}]", if c_s < 0.99 { "BROKEN" } else { "ok" });
    Ok(())
}
