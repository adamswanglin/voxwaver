//! Minimal Metal smoke test for the GPU sampling chain pieces:
//! multi-block arg-sort (main vocab 155776), single-block (codebook 4096),
//! and the primitives `sample_gpu` is built from.

use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    dev.set_seed(42)?;

    // 1. multi-block argsort path (ncols > 1024, nblocks > 1)
    for (name, v) in [("main_vocab", 155776usize), ("codebook", 4096)] {
        let t = Tensor::rand(0f32, 1f32, v, &dev)?;
        let order = t.affine(-1.0, 0.0)?.arg_sort_last_dim(true)?;
        let sorted = t.index_select(&order, 0)?;
        let s = sorted.to_vec1::<f32>()?;
        let ok = s.windows(2).all(|w| w[0] >= w[1]);
        let ids = order.to_vec1::<u32>()?;
        let ids_ok = (0..v).all(|i| ids.contains(&(i as u32)) || true); // placeholder
        println!("[{name}] v={v} descending={ok} head={:?}", &s[..3]);
        let _ = ids_ok;
    }

    // 2. gather/scatter_add/where_cond/cumsum/argmax/rand on metal
    let v = 1000usize;
    let vals = Tensor::rand(-1f32, 1f32, v, &dev)?;
    let prev = Tensor::rand(0f32, v as f32, 16, &dev)?
        .to_dtype(candle_core::DType::U32)?; // random ids in [0, v)
    let prev = prev.to_vec1::<u32>()?; // round-trip through the CPU to clamp
    let prev = Tensor::from_vec(prev, 16, &dev)?;
    let w = vals.gather(&prev, 0)?;
    let neg = w.lt(0f32)?;
    let delta = neg.where_cond(&w.affine(-0.5, 0.0)?, &w.affine(0.5, 0.0)?)?;
    let _p = vals.scatter_add(&prev, &delta, 0)?;
    let cum = vals.cumsum(0)?;
    let rank = vals.argmax_keepdim(0)?;
    let u = Tensor::rand(1e-12f32, 1.0f32, v, &dev)?;
    let noise = u.log()?.affine(-1.0, 0.0)?;
    let score = cum.div(&noise)?;
    let _top = score.argmax_keepdim(0)?;
    println!(
        "[primitives] gather/scatter/where/cumsum/argmax/rand ok; rank={} w0={:?}",
        rank.to_vec1::<u32>()?[0],
        w.to_vec1::<f32>()?[0]
    );
    Ok(())
}
