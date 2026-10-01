//! Isolated conv1d correctness check: CPU (im2col path) vs Metal (new
//! conv1d_direct implicit-GEMM path).
//!
//! Exercises the shapes that the DAC decoder actually produces, plus edge
//! cases that stress the tiling (l_out / c_out not multiples of TM=8 / TN=8,
//! stride>1, dilation>1, padding>0, and strided/narrow input views).
//!
//! Run:  cargo run --release --features metal --example conv1d_direct_check
use candle_core::{Device, Tensor};

fn to_cpu_f32(t: &Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn compare(label: &str, cpu: &Tensor, mtl: &Tensor) -> anyhow::Result<()> {
    let a = to_cpu_f32(cpu)?;
    let b = to_cpu_f32(mtl)?;
    let n = a.len().min(b.len());
    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    for i in 0..n {
        let d = (a[i] as f64 - b[i] as f64).abs();
        if d > max_diff { max_diff = d; }
        sum_diff += d;
        sa += a[i] as f64;
        sb += b[i] as f64;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut cov, mut va, mut vb) = (0.0f64, 0.0, 0.0);
    for i in 0..n {
        let x = a[i] as f64 - ma;
        let y = b[i] as f64 - mb;
        cov += x * y;
        va += x * x;
        vb += y * y;
    }
    let corr = cov / (va.sqrt() * vb.sqrt()).max(1e-30);
    let mean_diff = sum_diff / n as f64;
    let status = if corr > 0.9999 && max_diff < 1e-3 { "ok" } else { "MISMATCH" };
    println!("  [{status}] {label}: shape={:?} max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} corr={corr:.6}",
             cpu.shape().dims());
    Ok(())
}

/// Build a conv1d input on `cpu`, run on both backends, compare.
fn check_conv1d(
    label: &str,
    cpu: &Device,
    mtl: &Device,
    b: usize, c_in: usize, l_in: usize,
    c_out: usize, k: usize,
    stride: usize, padding: usize, dilation: usize,
) -> anyhow::Result<()> {
    let x = Tensor::randn(0f32, 1.0, (b, c_in, l_in), cpu)?;
    let w = Tensor::randn(0f32, 0.1, (c_out, c_in, k), cpu)?;
    let y_cpu = x.to_device(cpu)?.conv1d(&w.to_device(cpu)?, padding, stride, dilation, 1)?;
    let y_mtl = x.to_device(mtl)?.conv1d(&w.to_device(mtl)?, padding, stride, dilation, 1)?;
    compare(label, &y_cpu, &y_mtl)
}

/// Same as check_conv1d but the input is a narrow (non-contiguous) view,
/// matching the real decoder pattern: transconv → narrow → Snake → pad → conv1d.
fn check_conv1d_strided(
    label: &str,
    cpu: &Device,
    mtl: &Device,
    b: usize, c_in: usize, l_full: usize, narrow_offset: usize, l_in: usize,
    c_out: usize, k: usize,
    stride: usize, padding: usize, dilation: usize,
) -> anyhow::Result<()> {
    let x_full = Tensor::randn(0f32, 1.0, (b, c_in, l_full), cpu)?;
    let x = x_full.narrow(2, narrow_offset, l_in)?;
    let w = Tensor::randn(0f32, 0.1, (c_out, c_in, k), cpu)?;
    let y_cpu = x.to_device(cpu)?.conv1d(&w.to_device(cpu)?, padding, stride, dilation, 1)?;
    let y_mtl = x.to_device(mtl)?.conv1d(&w.to_device(mtl)?, padding, stride, dilation, 1)?;
    compare(label, &y_cpu, &y_mtl)
}

/// Check half/bfloat16 precision paths.
/// Note: CPU f16 uses f16 accumulation (less accurate), Metal uses f32 accumulation.
/// So we compare both against f32 reference, not against each other.
fn check_conv1d_dtype(
    label: &str,
    cpu: &Device,
    mtl: &Device,
    dtype: candle_core::DType,
    b: usize, c_in: usize, l_in: usize,
    c_out: usize, k: usize,
) -> anyhow::Result<()> {
    let x_f32 = Tensor::randn(0f32, 1.0, (b, c_in, l_in), cpu)?;
    let w_f32 = Tensor::randn(0f32, 0.1, (c_out, c_in, k), cpu)?;
    let x = x_f32.to_dtype(dtype)?;
    let w = w_f32.to_dtype(dtype)?;
    let y_cpu = x.to_device(cpu)?.conv1d(&w.to_device(cpu)?, 0, 1, 1, 1)?;
    let y_mtl = x.to_device(mtl)?.conv1d(&w.to_device(mtl)?, 0, 1, 1, 1)?;
    let y_ref = x_f32.to_device(cpu)?.conv1d(&w_f32.to_device(cpu)?, 0, 1, 1, 1)?;
    // Compare both against f32 reference
    let cpu_f32 = to_cpu_f32(&y_cpu)?;
    let mtl_f32 = to_cpu_f32(&y_mtl)?;
    let ref_f32 = to_cpu_f32(&y_ref)?;
    let n = cpu_f32.len();
    let mut cpu_err = 0.0f64;
    let mut mtl_err = 0.0f64;
    for i in 0..n {
        cpu_err += (cpu_f32[i] as f64 - ref_f32[i] as f64).abs();
        mtl_err += (mtl_f32[i] as f64 - ref_f32[i] as f64).abs();
    }
    let cpu_mean = cpu_err / n as f64;
    let mtl_mean = mtl_err / n as f64;
    // Metal should be close to ref; bf16 has less precision than f16
    let threshold = if dtype == candle_core::DType::BF16 { 1e-2 } else { 1e-3 };
    let status = if mtl_mean < threshold { "ok" } else { "MISMATCH" };
    println!("  [{status}] {label}: shape={:?} cpu_mean_err={cpu_mean:.2e} mtl_mean_err={mtl_mean:.2e}",
             y_ref.shape().dims());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cpu = Device::Cpu;
    let mtl = Device::new_metal(0)?;

    println!("=== conv1d_direct vs CPU im2col correctness check ===\n");

    // 1. DAC decoder actual shapes (from src/config.rs: rates [8,8,4,2], dim 1536)
    println!("[DAC decoder shapes]");
    // conv_in: (1, 1536, l) from latent 1024
    check_conv1d("dec0 conv_in",    &cpu, &mtl, 1, 1024, 1560, 1536, 7, 1, 0, 1)?;
    // dec0 RU conv1d: c_in=c_out=768, k=7, dilations 1/3/9
    check_conv1d("dec0 ru dil=1",   &cpu, &mtl, 1, 768,  3120, 768,  7, 1, 3, 1)?;
    check_conv1d("dec0 ru dil=3",   &cpu, &mtl, 1, 768,  3120, 768,  7, 1, 9, 3)?;
    check_conv1d("dec0 ru dil=9",   &cpu, &mtl, 1, 768,  3120, 768,  7, 1, 27, 9)?;
    // dec2 (channels 384->192) at 391 frames: l_in ~ 400k
    check_conv1d("dec2 ru dil=1 (391fr)", &cpu, &mtl, 1, 192, 400384, 192, 7, 1, 3, 1)?;

    // 2. Tiling edge cases: l_out / c_out not multiples of TM=8, TN=8
    println!("\n[tiling edge cases]");
    check_conv1d("l_out=1 (c_out=8)",   &cpu, &mtl, 1, 4, 8,   8,  7, 1, 3, 1)?;
    check_conv1d("l_out=7 (partial M)", &cpu, &mtl, 1, 4, 13,  8,  7, 1, 3, 1)?;
    check_conv1d("l_out=9 (M+1)",       &cpu, &mtl, 1, 4, 15,  8,  7, 1, 3, 1)?;
    check_conv1d("c_out=1 (partial N)", &cpu, &mtl, 1, 4, 20,  1,  7, 1, 3, 1)?;
    check_conv1d("c_out=3 (partial N)", &cpu, &mtl, 1, 4, 20,  3,  7, 1, 3, 1)?;
    check_conv1d("c_out=9 (N+1)",       &cpu, &mtl, 1, 4, 20,  9,  7, 1, 3, 1)?;
    check_conv1d("c_out=17 l_out=17",   &cpu, &mtl, 1, 5, 23, 17,  5, 1, 2, 1)?;

    // 3. stride / dilation / padding combinations
    println!("\n[stride/dilation/padding]");
    check_conv1d("stride=2",            &cpu, &mtl, 1, 16, 100, 32, 5, 2, 0, 1)?;
    check_conv1d("stride=3",            &cpu, &mtl, 1, 16, 100, 32, 5, 3, 0, 1)?;
    check_conv1d("padding=2",           &cpu, &mtl, 1, 16, 100, 32, 5, 1, 2, 1)?;
    check_conv1d("dilation=2",          &cpu, &mtl, 1, 16, 100, 32, 5, 1, 0, 2)?;
    check_conv1d("dilation=4 pad=4",    &cpu, &mtl, 1, 16, 100, 32, 5, 1, 4, 4)?;
    check_conv1d("stride=2 dil=2 pad=2",&cpu, &mtl, 1, 16, 200, 32, 7, 2, 2, 2)?;

    // 4. Batched
    println!("\n[batched]");
    check_conv1d("batch=4",             &cpu, &mtl, 4, 32, 256, 64, 5, 1, 2, 1)?;

    // 4b. multi-tile x batch>1: tiled im2col must scatter each batch slice
    // into the (b, l_out, n) accumulator at the right offset.
    println!("\n[multi-tile x batched]");
    check_conv1d("batch=4 multi-tile",  &cpu, &mtl, 4, 32, 40_000, 32, 7, 1, 3, 1)?;
    check_conv1d("batch=2 multi-tile pad/dil", &cpu, &mtl, 2, 8, 210_000, 16, 5, 1, 4, 2)?;

    // 5. Strided / narrow input views (real decoder pattern)
    println!("\n[strided input views (narrow)]");
    check_conv1d_strided("narrow offset=10",  &cpu, &mtl, 1, 64, 500, 10, 400, 32, 7, 1, 3, 1)?;
    check_conv1d_strided("narrow offset=100", &cpu, &mtl, 1, 64, 500, 100, 300, 32, 7, 1, 3, 1)?;
    check_conv1d_strided("narrow offset=1 l=400384 (391fr)", &cpu, &mtl, 1, 192, 400400, 1, 400384, 192, 7, 1, 3, 1)?;

    // 6. Long sequences (the problematic regime)
    println!("\n[long sequences]");
    check_conv1d("l_in=100k",           &cpu, &mtl, 1, 192, 100_000, 192, 7, 1, 3, 1)?;
    check_conv1d("l_in=400k (391fr)",   &cpu, &mtl, 1, 192, 400_384, 192, 7, 1, 3, 1)?;
    check_conv1d("l_in=800k",           &cpu, &mtl, 1, 96,  800_000, 96,  7, 1, 3, 1)?;

    // 7. dtype paths (f16 / bf16 with f32 accumulation)
    println!("\n[dtype paths]");
    check_conv1d_dtype("f16",           &cpu, &mtl, candle_core::DType::F16,  1, 64, 1000, 128, 7)?;
    check_conv1d_dtype("bf16",          &cpu, &mtl, candle_core::DType::BF16, 1, 64, 1000, 128, 7)?;
    check_conv1d_dtype("f32",           &cpu, &mtl, candle_core::DType::F32,  1, 64, 1000, 128, 7)?;

    println!("\n=== done ===");
    Ok(())
}
