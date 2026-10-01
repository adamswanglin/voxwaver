//! Isolated per-op throughput bench on Metal for the exact tensor shapes the
//! DAC decoder produces (356 code frames). Steady state: 2 warmup iterations,
//! then 5 back-to-back iterations amortized (pool warm, no sync churn).
//!
//! Run: cargo run --release --features metal --example conv1d_bench
use candle_core::{Device, Tensor};
use std::time::Instant;

fn bench(
    label: &str,
    dev: &Device,
    gflop: f64,
    iters: usize,
    mut f: impl FnMut() -> anyhow::Result<Tensor>,
) -> anyhow::Result<()> {
    for _ in 0..2 {
        f()?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let rate = if gflop > 0.0 {
        format!("  ({:.0} GFLOPS)", gflop / dt)
    } else {
        String::new()
    };
    println!("  {label}: {:.1} ms{rate}", dt * 1e3);
    Ok(())
}

/// Snake matching `dac::layers::Snake`: single sin pass, unary `sqr()` and
/// a load-folded reciprocal. The binary `s.mul(&s)` form is avoided on
/// Metal (it cost +0.28 GB peak footprint); see that type for details.
fn snake(x: &Tensor) -> anyhow::Result<Tensor> {
    let c = x.dim(1)?;
    let a = Tensor::full(0.5f32, (1, c, 1), x.device())?;
    let inv = a.affine(1.0, 1e-9)?.recip()?;
    let ax = x.broadcast_mul(&a)?;
    let s2 = ax.sin()?.sqr()?;
    Ok(x.add(&s2.broadcast_mul(&inv)?)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    println!("=== candle-metal isolated op bench (f32, 356-frame DAC shapes) ===");

    // ---- dense k7 convs (im2col + matmul path) with dilation 9 ----
    let cases: [(&str, usize, usize); 4] = [
        ("B1 RU k7d9 [1,768,11392]", 768, 11392),
        ("B2 RU k7d9 [1,384,91136]", 384, 91136),
        ("B3 RU k7d9 [1,192,364544]", 192, 364544),
        ("B4 RU k7d9 [1,96,729088]", 96, 729088),
    ];
    for (label, c, t) in cases {
        let x = Tensor::randn(0f32, 1.0, (1, c, t), &dev)?;
        let w = Tensor::randn(0f32, 0.1, (c, c, 7), &dev)?;
        let gflop = 2.0 * (t as f64) * (c as f64) * (c as f64) * 7.0 / 1e9;
        bench(label, &dev, gflop, 5, || Ok(x.conv1d(&w, 0, 1, 9, 1)?))?;
    }

    // ---- k1 convs (c2 in ResidualUnit) ----
    let x = Tensor::randn(0f32, 1.0, (1, 192, 364544), &dev)?;
    let w1 = Tensor::randn(0f32, 0.1, (192, 192, 1), &dev)?;
    let g = 2.0 * 364544.0 * 192.0 * 192.0 / 1e9;
    bench("B3 RU k1 [1,192,364544]", &dev, g, 5, || Ok(x.conv1d(&w1, 0, 1, 1, 1)?))?;

    // ---- conv_out k7 96->1 (N=1 GEMM) + tail snake ----
    let x96 = Tensor::randn(0f32, 1.0, (1, 96, 729088), &dev)?;
    let wout = Tensor::randn(0f32, 0.1, (1, 96, 7), &dev)?;
    let gout = 2.0 * 729088.0 * 96.0 * 7.0 / 1e9;
    bench("conv_out k7 [1,96,729088]->[1,1]", &dev, gout, 5, || {
        Ok(x96.conv1d(&wout, 0, 1, 1, 1)?)
    })?;
    bench("snake [1,96,729088] (tail)", &dev, 0.0, 5, || snake(&x96))?;

    // ---- raw matmul rates for the im2col GEMM shapes ----
    let a = Tensor::randn(0f32, 1.0, (364544, 1344), &dev)?;
    let b192 = Tensor::randn(0f32, 1.0, (1344, 192), &dev)?;
    let b768 = Tensor::randn(0f32, 1.0, (1344, 768), &dev)?;
    let gm = 2.0 * 364544.0 * 1344.0 * 192.0 / 1e9;
    bench("[mm] [364544,1344]x[1344,192]", &dev, gm, 5, || Ok(a.matmul(&b192)?))?;
    let gm2 = 2.0 * 364544.0 * 1344.0 * 768.0 / 1e9;
    bench("[mm] [364544,1344]x[1344,768]", &dev, gm2, 5, || Ok(a.matmul(&b768)?))?;
    let a2 = Tensor::randn(0f32, 1.0, (91136, 2688), &dev)?;
    let b2 = Tensor::randn(0f32, 1.0, (2688, 384), &dev)?;
    let gm3 = 2.0 * 91136.0 * 2688.0 * 384.0 / 1e9;
    bench("[mm] [91136,2688]x[2688,384]", &dev, gm3, 5, || Ok(a2.matmul(&b2)?))?;

    // ---- snake at other scales + single elementwise ops ----
    let x3 = Tensor::randn(0f32, 1.0, (1, 192, 364544), &dev)?;
    bench("snake [1,192,364544]", &dev, 0.0, 5, || snake(&x3))?;
    bench("sin [1,192,364544]", &dev, 0.0, 10, || Ok(x3.sin()?))?;
    bench("mul [1,192,364544]", &dev, 0.0, 10, || Ok(x3.mul(&x3)?))?;
    bench("add [1,192,364544]", &dev, 0.0, 10, || Ok(x3.add(&x3)?))?;
    let a1 = Tensor::full(0.5f32, (1, 192, 1), &dev)?;
    bench("broadcast_mul [1,192,T]x[1,192,1]", &dev, 0.0, 10, || {
        Ok(x3.broadcast_mul(&a1)?)
    })?;
    bench("broadcast_div by [1,192,1]", &dev, 0.0, 10, || {
        Ok(x3.broadcast_div(&a1)?)
    })?;
    let a_full = Tensor::full(0.5f32, (1, 192, 364544), &dev)?;
    bench("mul by full [1,192,T]", &dev, 0.0, 10, || Ok(x3.mul(&a_full)?))?;
    let x2 = Tensor::randn(0f32, 1.0, (1, 384, 91136), &dev)?;
    bench("snake [1,384,91136]", &dev, 0.0, 5, || snake(&x2))?;
    let x1 = Tensor::randn(0f32, 1.0, (1, 768, 11392), &dev)?;
    bench("snake [1,768,11392]", &dev, 0.0, 5, || snake(&x1))?;

    // ---- transposed convs (col2im + matmul path) ----
    let xt = Tensor::randn(0f32, 1.0, (1, 384, 91136), &dev)?;
    let wt = Tensor::randn(0f32, 0.1, (384, 192, 8), &dev)?;
    let gt = 2.0 * 91136.0 * 384.0 * 192.0 * 8.0 / 1e9;
    bench("tconv [1,384,91136] k8 s4 (B3)", &dev, gt, 5, || {
        Ok(xt.conv_transpose1d(&wt, 0, 0, 4, 1, 1)?)
    })?;
    let xt1 = Tensor::randn(0f32, 1.0, (1, 1536, 1424), &dev)?;
    let wt1 = Tensor::randn(0f32, 0.1, (1536, 768, 16), &dev)?;
    let gt1 = 2.0 * 1424.0 * 1536.0 * 768.0 * 16.0 / 1e9;
    bench("tconv [1,1536,1424] k16 s8 (B1)", &dev, gt1, 5, || {
        Ok(xt1.conv_transpose1d(&wt1, 0, 0, 8, 1, 1)?)
    })?;

    // ---- strided input conv (post-tconv narrow view, as in the real flow) ----
    let xfull = Tensor::randn(0f32, 1.0, (1, 192, 364550), &dev)?;
    let xn = xfull.narrow(2, 0, 364544)?;
    let wn = Tensor::randn(0f32, 0.1, (192, 192, 7), &dev)?;
    let gn = 2.0 * 364544.0 * 192.0 * 192.0 * 7.0 / 1e9;
    bench("B3 k7d9 narrow-view input", &dev, gn, 5, || {
        Ok(xn.conv1d(&wn, 0, 1, 9, 1)?)
    })?;

    // ---- depthwise conv through the groups path (ConvNeXt dwconv) ----
    let xd = Tensor::randn(0f32, 1.0, (1, 1024, 712), &dev)?;
    let wd = Tensor::randn(0f32, 0.1, (1024, 1, 7), &dev)?;
    let ch = 2.0 * 1024.0 * 712.0 * 7.0 / 1e9;
    bench("dwconv [1,1024,712] k7 groups=1024", &dev, ch, 5, || {
        Ok(xd.conv1d(&wd, 0, 1, 1, 1024)?)
    })?;

    let _ = &dev;
    println!("=== done ===");
    Ok(())
}
