//! Focused f16 conv1d investigation: is the bug in conv1d_direct, or in the
//! CPU im2col path's f16 handling?
//!
//! Run:  cargo run --release --features metal --example conv1d_f16_debug
use candle_core::{Device, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let cpu = Device::Cpu;
    let mtl = Device::new_metal(0)?;

    println!("=== f16 conv1d debug ===\n");

    // The shape that failed in conv1d_direct_check
    let (b, c_in, l_in, c_out, k) = (1, 64, 1000, 128, 7);
    let x_f32 = Tensor::randn(0f32, 1.0, (b, c_in, l_in), &cpu)?;
    let w_f32 = Tensor::randn(0f32, 0.1, (c_out, c_in, k), &cpu)?;

    println!("Input shape: {:?}, Weight shape: {:?}", x_f32.shape(), w_f32.shape());

    // Convert to f16
    let x_f16 = x_f32.to_dtype(DType::F16)?;
    let w_f16 = w_f32.to_dtype(DType::F16)?;

    // CPU f16
    let y_cpu_f16 = x_f16.conv1d(&w_f16, 3, 1, 1, 1)?;
    println!("\nCPU f16 output (first 10 elements):");
    let cpu_f16_vec = y_cpu_f16.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    for (i, v) in cpu_f16_vec.iter().take(10).enumerate() {
        print!("{:8.4} ", v);
        if (i + 1) % 10 == 0 { println!(); }
    }

    // Metal f16
    let y_mtl_f16 = x_f16.to_device(&mtl)?.conv1d(&w_f16.to_device(&mtl)?, 3, 1, 1, 1)?;
    println!("\nMetal f16 output (first 10 elements):");
    let mtl_f16_vec = y_mtl_f16.to_device(&cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    for (i, v) in mtl_f16_vec.iter().take(10).enumerate() {
        print!("{:8.4} ", v);
        if (i + 1) % 10 == 0 { println!(); }
    }

    // CPU f32 (reference)
    let y_cpu_f32 = x_f32.conv1d(&w_f32, 3, 1, 1, 1)?;
    println!("\nCPU f32 output (first 10 elements) [reference]:");
    let cpu_f32_vec = y_cpu_f32.flatten_all()?.to_vec1::<f32>()?;
    for (i, v) in cpu_f32_vec.iter().take(10).enumerate() {
        print!("{:8.4} ", v);
        if (i + 1) % 10 == 0 { println!(); }
    }

    // Compare
    println!("\n=== Differences ===");
    let mut max_cpu_mtl = 0.0f64;
    let mut max_cpu_ref = 0.0f64;
    for i in 0..cpu_f16_vec.len() {
        let d1 = (cpu_f16_vec[i] as f64 - mtl_f16_vec[i] as f64).abs();
        let d2 = (cpu_f16_vec[i] as f64 - cpu_f32_vec[i] as f64).abs();
        if d1 > max_cpu_mtl { max_cpu_mtl = d1; }
        if d2 > max_cpu_ref { max_cpu_ref = d2; }
    }
    println!("CPU f16 vs Metal f16: max_diff = {:.4}", max_cpu_mtl);
    println!("CPU f16 vs CPU f32:   max_diff = {:.4}  (expected f16 rounding)", max_cpu_ref);

    // Check if Metal f16 is closer to CPU f32 than CPU f16 is
    let mut sum_mtl = 0.0f64;
    let mut sum_cpu = 0.0f64;
    for i in 0..cpu_f32_vec.len() {
        sum_mtl += (mtl_f16_vec[i] as f64 - cpu_f32_vec[i] as f64).abs();
        sum_cpu += (cpu_f16_vec[i] as f64 - cpu_f32_vec[i] as f64).abs();
    }
    println!("\nMean abs diff from CPU f32 reference:");
    println!("  Metal f16: {:.6}", sum_mtl / cpu_f32_vec.len() as f64);
    println!("  CPU f16:   {:.6}", sum_cpu / cpu_f32_vec.len() as f64);

    Ok(())
}
