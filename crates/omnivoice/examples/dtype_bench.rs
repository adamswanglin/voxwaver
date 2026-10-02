//! Micro-benchmark: f32 vs bf16 throughput of the ops that dominate the
//! Qwen3 backbone pass (GEMM, sdpa, rms_norm, elementwise) on Metal.
//!
//! Run with: cargo run --release --features metal --example dtype_bench

use candle_core::{DType, Device, Tensor};

fn timeit(label: &str, dtype: DType, runs: usize, f: impl Fn() -> candle_core::Result<Tensor>) {
    // warmup + sync (Metal executes async)
    let warm = f().unwrap();
    warm.device().synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        let out = f().unwrap();
        out.device().synchronize().unwrap();
        std::hint::black_box(&out);
    }
    let dt_ms = t0.elapsed().as_secs_f32() / runs as f32 * 1000.0;
    println!("{label:24} {dtype:?}  {dt_ms:8.3} ms/op");
}

fn main() -> candle_core::Result<()> {
    let dev = Device::new_metal(0)?;
    let t = 494usize; // typical cond sequence length
    let hidden = 1024usize;
    let inter = 3072usize;
    let heads = 16usize;
    let kv = 8usize;
    let hd = 128usize;
    let runs = 50usize;

    for dtype in [DType::F32, DType::BF16] {
        let mk = |r: usize, c: usize| -> candle_core::Result<Tensor> {
            Tensor::randn(0f32, 1f32, (r, c), &dev)?.to_dtype(dtype)
        };
        // GEMM shapes of one Qwen3 layer: qkv/o projections + MLP
        let x = mk(t, hidden)?;
        let wq = mk(hidden, heads * hd)?;
        let wg = mk(hidden, inter)?;
        let wd = mk(inter, hidden)?;
        timeit("gemm [t,1024]x[1024,2048]", dtype, runs, || x.matmul(&wq));
        timeit("gemm [t,1024]x[1024,3072]", dtype, runs, || x.matmul(&wg));
        timeit("gemm [t,3072]x[3072,1024]", dtype, runs, || {
            let h = mk(t, inter)?;
            h.matmul(&wd)
        });

        // sdpa (bidirectional, full)
        let q = mk(1 * heads * t * hd, 1)?.reshape((1, heads, t, hd))?;
        let k = mk(1 * kv * t * hd, 1)?.reshape((1, kv, t, hd))?;
        let v = mk(1 * kv * t * hd, 1)?.reshape((1, kv, t, hd))?;
        let rep = heads / kv;
        let k = k
            .unsqueeze(2)?
            .expand((1, kv, rep, t, hd))?
            .reshape((1, heads, t, hd))?;
        let v = v
            .unsqueeze(2)?
            .expand((1, kv, rep, t, hd))?
            .reshape((1, heads, t, hd))?;
        let scale = 1.0f64 / (hd as f64).sqrt();
        timeit("sdpa [1,16,t,128]", dtype, runs, || {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)
        });

        // rms_norm (alpha must be rank-1)
        let alpha = Tensor::randn(0f32, 1f32, (hidden,), &dev)?.to_dtype(dtype)?;
        timeit("rms_norm [t,1024]", dtype, runs, || {
            candle_nn::ops::rms_norm(&x, &alpha, 1e-6f32)
        });

        // rope (via candle_nn)
        let half = hd / 2;
        let cos = Tensor::randn(0f32, 1f32, (t, half), &dev)?.to_dtype(dtype)?;
        let sin = Tensor::randn(0f32, 1f32, (t, half), &dev)?.to_dtype(dtype)?;
        let qr = mk(1 * heads * t * hd, 1)?.reshape((1, heads, t, hd))?;
        timeit("rope [1,16,t,128]", dtype, runs, || {
            candle_nn::rotary_emb::rope(&qr, &cos, &sin)
        });

        // elementwise chain (silu(gate)*up)
        let g = mk(t, inter)?;
        let u = mk(t, inter)?;
        timeit("silu*up [t,3072]", dtype, runs, || {
            candle_nn::ops::silu(&g)? * &u
        });
        println!();
    }
    Ok(())
}
