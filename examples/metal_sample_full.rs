//! Full replication of the `sample_gpu` chain on Metal, from a realistic
//! F16 [1,1,V] logits tensor, to isolate which primitive fails in the
//! real generation loop.

use candle_core::{DType, Device, Tensor};

fn triu_ones(n: usize, dev: &Device) -> anyhow::Result<Tensor> {
    let r = Tensor::arange(0u32, n as u32, dev)?;
    let row = r.reshape((n, 1))?.broadcast_as((n, n))?;
    let col = r.reshape((1, n))?.broadcast_as((n, n))?;
    Ok(row.le(&col)?.to_dtype(DType::F32)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    dev.set_seed(42)?;
    let v = 155776usize;

    // mimic the real logits: [1,1,V] f16
    let logits = Tensor::rand(-5f32, 5f32, (1, 1, v), &dev)?.to_dtype(DType::F16)?;

    let mut vals = logits.reshape(v)?.to_dtype(DType::F32)?; // [V]
    println!("[1] cast ok");

    // penalty chain with a zero window (like a fresh history)
    let prev = Tensor::zeros(16, DType::U32, &dev)?;
    let w = vals.gather(&prev, 0)?;
    let rp = 1.5f32;
    let neg = w.lt(0f32)?;
    let delta = neg.where_cond(
        &w.affine((rp - 1f32) as f64, 0.0)?,
        &w.affine((1f32 / rp - 1f32) as f64, 0.0)?,
    )?;
    vals = vals.scatter_add(&prev, &delta, 0)?;
    println!("[2] penalty ok");

    let order = vals.affine(-1.0, 0.0)?.arg_sort_last_dim(true)?; // [V] u32
    println!("[3] argsort ok");
    let sorted = vals.index_select(&order, 0)?; // [V] descending
    println!("[4] index_select ok");

    // ---- the segment that fails in the real loop ----
    let mx = sorted.get(0)?; // []
    println!("[5] get(0) ok, mx shape {:?}", mx.dims());
    let e = sorted.broadcast_sub(&mx)?.exp()?;
    println!("[6] broadcast_sub+exp ok");
    let sum = e.sum_all()?;
    println!("[7] sum_all ok, shape {:?}", sum.dims());
    // two-level blocked scan (mirrors cumsum1 in src/sampling.rs)
    const B: usize = 256;
    let c = v.div_ceil(B);
    let e_pad = if c * B != v {
        let tail = Tensor::zeros(c * B - v, DType::F32, &dev)?;
        Tensor::cat(&[&e, &tail], 0)?
    } else {
        e.clone()
    };
    let triu_b = triu_ones(B, &dev)?;
    let intra = e_pad.reshape((c, B))?.matmul(&triu_b)?;
    let tot = intra.narrow(1, B - 1, 1)?.reshape(c)?;
    let triu_c = triu_ones(c, &dev)?;
    let inc = tot.unsqueeze(0)?.matmul(&triu_c)?.reshape(c)?;
    let exc = inc.sub(&tot)?;
    let cum = intra
        .broadcast_add(&exc.reshape((c, 1))?)?
        .reshape(c * B)?
        .narrow(0, 0, v)?
        .broadcast_div(&sum)?;
    println!("[8] blocked-scan cumsum+broadcast_div ok");
    let keep_p = cum.le(0.7f64)?.to_dtype(DType::F32)?;
    println!("[9] le+cast ok");
    let rank0 = Tensor::arange(0u32, v as u32, &dev)?
        .eq(0u32)?
        .to_dtype(DType::F32)?;
    println!("[10] rank0 ok");
    let keep = keep_p.add(&rank0)?.gt(0.5f64)?.to_dtype(DType::F32)?;
    println!("[11] keep ok");

    let temp = 0.7f32;
    let t = sorted
        .broadcast_sub(&mx)?
        .affine(1f64 / temp as f64, 0.0)?
        .exp()?
        .mul(&keep)?;
    println!("[12] temperature ok");
    let total = t.sum_all()?;
    println!("[13] total ok");

    let u = Tensor::rand(1e-12f32, 1.0f32, v, &dev)?;
    let noise = u.log()?.affine(-1.0, 0.0)?;
    println!("[14] gumbel noise ok");
    let score = t.div(&noise)?.broadcast_div(&total)?;
    println!("[15] score ok");
    let rank = score.argmax_keepdim(0)?; // [1] u32
    println!("[16] argmax ok");
    let token = order.index_select(&rank, 0)?;
    let token = token.to_vec1::<u32>()?[0];
    println!("[17] all ok, token={token}");
    Ok(())
}
