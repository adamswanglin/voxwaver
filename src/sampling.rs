use candle_core::{DType, Tensor};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Sampling parameters mirroring the s1-mini serving defaults:
/// temperature 0.7, top_p 0.7, repetition_penalty 1.5.
#[derive(Debug, Clone, Copy)]
pub struct SampleParams {
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

/// Windowed repetition-penalty history, mirroring `decode_n_tokens`:
/// a per-row ring of the last REP_WIN_SIZE sampled tokens, zero-padded
/// (token id 0 participates in the penalty exactly like upstream).
pub const REP_WIN_SIZE: usize = 16;

/// Upper-triangular (incl. diagonal) all-ones `[n, n]` matrix, cached per
/// (n, dtype): the scan masks are shape-stable across decode frames, and
/// rebuilding one costs several kernels plus an `n*n` device buffer.
fn triu_ones(n: usize, dtype: DType, dev: &candle_core::Device) -> anyhow::Result<Tensor> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, DType), Tensor>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if let Some(t) = cache.get(&(n, dtype)) {
        if t.device().same_device(dev) {
            return Ok(t.clone()); // Arc clone
        }
    }
    let r = Tensor::arange(0u32, n as u32, dev)?;
    let row = r.reshape((n, 1))?.broadcast_as((n, n))?;
    let col = r.reshape((1, n))?.broadcast_as((n, n))?;
    let t = row.le(&col)?.to_dtype(dtype)?;
    cache.insert((n, dtype), t.clone());
    Ok(t)
}

/// `[v]` f32 one-hot mask at position 0 (top-1 always kept), cached per
/// length: the decode loop calls this with a constant vocab every frame.
fn rank0_mask(v: usize, dev: &candle_core::Device) -> anyhow::Result<Tensor> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Tensor>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if let Some(t) = cache.get(&v) {
        if t.device().same_device(dev) {
            return Ok(t.clone());
        }
    }
    let t = Tensor::arange(0u32, v as u32, dev)?
        .eq(0u32)?
        .to_dtype(DType::F32)?;
    cache.insert(v, t.clone());
    Ok(t)
}

/// Inclusive prefix sum of a rank-1 f32 tensor via a two-level blocked
/// matmul scan. candle's `Tensor::cumsum` materializes a `[V, V]` triangular
/// matrix (~97 GB at the main vocab size), so it cannot be used here.
///
/// The vector is zero-padded to a multiple of `B`, each `[B]` block is
/// scanned with a `[B, B]` triu matmul (`[C, B] x [B, B]`, C = V/B blocks in
/// one batched matmul), and the exclusive scan of the block totals (a `[C]`
/// matmul, C stays far below the second-level limit) is broadcast back.
fn cumsum1(x: &Tensor) -> anyhow::Result<Tensor> {
    const B: usize = 256;
    let dev = x.device();
    let v = x.elem_count();
    anyhow::ensure!(x.rank() == 1, "cumsum1 expects a rank-1 tensor");
    let c = v.div_ceil(B);
    anyhow::ensure!(c <= 1024, "cumsum1: {v} elements need a third scan level");
    let x = if c * B != v {
        let tail = Tensor::zeros(c * B - v, x.dtype(), dev)?;
        Tensor::cat(&[x, &tail], 0)?
    } else {
        x.clone()
    };
    let blocks = x.reshape((c, B))?;
    let triu = triu_ones(B, x.dtype(), dev)?;
    let intra = blocks.matmul(&triu)?; // [C, B] per-block inclusive scans
    let tot = intra.narrow(1, B - 1, 1)?.reshape(c)?; // [C] block totals
    let triu_c = triu_ones(c, x.dtype(), dev)?;
    let inc = tot.unsqueeze(0)?.matmul(&triu_c)?.reshape(c)?; // [C]
    let exc = inc.sub(&tot)?; // [C] exclusive block offsets
    Ok(intra
        .broadcast_add(&exc.reshape((c, 1))?)?
        .reshape(c * B)?
        .narrow(0, 0, v)?)
}

pub struct Sampler {
    /// CPU-side RNG, used only by the reference CPU `sample` below (tests
    /// and regression debugging); the decode loop runs `sample_gpu`.
    #[allow(dead_code)]
    rng: rand::rngs::StdRng,
    /// round logits through bf16 before sampling (parity with the bf16 reference)
    round_bf16: bool,
}

impl Sampler {
    pub fn new(seed: u64, round_bf16: bool) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            round_bf16,
        }
    }

    /// Copy a logits row tensor to a CPU f32 vector, optionally rounding
    /// through bf16 (matching the reference which computes logits in bf16).
    /// Reference CPU path; the decode loop samples on-device instead.
    #[allow(dead_code)]
    pub fn logits_row(&self, logits: &Tensor) -> anyhow::Result<Vec<f32>> {
        let row = logits.reshape(logits.elem_count())?;
        let row = if self.round_bf16 {
            row.to_dtype(candle_core::DType::BF16)?
                .to_dtype(candle_core::DType::F32)?
        } else {
            row.to_dtype(candle_core::DType::F32)?
        };
        Ok(row.to_vec1::<f32>()?)
    }

    /// Exact port of `logits_to_probs` + `multinomial_sample_one_no_sync`.
    /// Reference CPU implementation (tests, debugging); the decode loop
    /// runs the on-device `sample_gpu`.
    ///
    /// Repetition penalty first (score < 0 -> *rp, else /rp, scattered back),
    /// then top-p over the *untempered* softmax cumulative masses (top-1 always
    /// kept), then temperature, then a Gumbel-max draw.
    #[allow(dead_code)]
    pub fn sample(
        &mut self,
        logits: &[f32],
        params: &SampleParams,
        previous_tokens: Option<&[u32]>,
    ) -> u32 {
        let mut vals = logits.to_vec();

        // windowed repetition penalty
        if let Some(prev) = previous_tokens {
            for &t in prev {
                let i = t as usize;
                if i < vals.len() {
                    let s = vals[i];
                    vals[i] = if s < 0.0 {
                        s * params.repetition_penalty as f32
                    } else {
                        s / params.repetition_penalty as f32
                    };
                }
            }
        }

        // sort descending
        let mut order: Vec<usize> = (0..vals.len()).collect();
        order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));

        // cumulative softmax over the sorted logits (untempered)
        let max = vals[order[0]];
        let mut sum = 0f32;
        let mut exp_sorted = Vec::with_capacity(order.len());
        for &i in &order {
            let e = (vals[i] - max).exp();
            sum += e;
            exp_sorted.push(e);
        }
        let mut cum = Vec::with_capacity(order.len());
        let mut acc = 0f32;
        for e in &exp_sorted {
            acc += e / sum;
            cum.push(acc);
        }

        // top-p mask: remove cum > top_p; always keep top-1
        // temperature + renormalize over kept entries
        let temp = params.temperature.max(1e-5) as f32;
        let mut kept: Vec<(usize, f32)> = Vec::new(); // (index, unnormalized prob)
        let mut total = 0f32;
        for (rank, &i) in order.iter().enumerate() {
            if rank == 0 || !(cum[rank] > params.top_p as f32) {
                let p = ((vals[i] - max) / temp).exp();
                total += p;
                kept.push((i, p));
            }
        }
        if kept.is_empty() {
            return order[0] as u32;
        }

        // gumbel-max style: argmax(probs / -log(u))
        let u: f32 = self.rng.gen::<f32>().clamp(1e-12, 1.0);
        let noise = -u.ln();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for &(i, p) in &kept {
            let score = (p / total) / noise;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best as u32
    }

    /// GPU-resident port of `sample`: the same windowed repetition penalty
    /// -> descending sort -> top-p prefix over the untempered cumulative
    /// softmax -> temperature -> Gumbel draw `argmax(probs / -ln(u))`, built
    /// from candle primitives so the logits row never crosses back to the
    /// CPU. Returns a `[1]` u32 tensor on the logits' device, ready to feed
    /// the next embedding lookup.
    ///
    /// Accepted differences from the CPU sampler:
    /// - the RNG is candle's device RNG (Metal tausworthe/lcg) instead of
    ///   `StdRng`, so seeded runs diverge from the CPU path token-for-token
    ///   (same distribution);
    /// - repeated ids inside the penalty window accumulate one scatter_add
    ///   delta against the original logit, instead of the CPU's sequential
    ///   re-application of the transform (only matters for id repeats inside
    ///   the 16-token window);
    /// - the Metal multi-block sort kernels are ascending-only, so the
    ///   descending order comes from sorting the negation.
    pub fn sample_gpu(
        &self,
        logits: &Tensor,
        params: &SampleParams,
        prev: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        let dev = logits.device();
        let v = logits.elem_count();
        let mut vals = logits.reshape(v)?.to_dtype(DType::F32)?; // [V]
        if self.round_bf16 {
            vals = vals.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        }

        // windowed repetition penalty: score' = score < 0 ? score * rp :
        // score / rp, applied as an additive delta at the window ids
        let vals = if let Some(prev) = prev {
            let w = vals.gather(prev, 0)?; // [W]
            let rp = params.repetition_penalty as f32;
            let neg = w.lt(0f32)?; // u8 mask
            let delta = neg.where_cond(
                &w.affine((rp - 1f32) as f64, 0.0)?,
                &w.affine((1f32 / rp - 1f32) as f64, 0.0)?,
            )?;
            vals.scatter_add(prev, &delta, 0)?
        } else {
            vals
        };

        // descending argsort: Metal's multi-block kernels are ascending-only
        let order = vals.affine(-1.0, 0.0)?.arg_sort_last_dim(true)?; // [V] u32
        let sorted = vals.index_select(&order, 0)?; // [V] descending

        // top-p prefix over the untempered cumulative softmax; top-1 kept
        let mx = sorted.get(0)?; // []
        let d = sorted.broadcast_sub(&mx)?; // [V], shared by both softmaxes
        let e = d.exp()?;
        let cum = cumsum1(&e)?.broadcast_div(&e.sum_all()?)?; // [V]
        let keep_p = cum.le(params.top_p as f64)?.to_dtype(DType::F32)?;
        // boolean OR without a bit-or op: elementwise max on 0/1 masks
        let keep = keep_p.maximum(&rank0_mask(v, dev)?)?;

        // temperature + renormalize over the kept entries
        let temp = params.temperature.max(1e-5) as f32;
        let t = d.affine(1f64 / temp as f64, 0.0)?.exp()?.mul(&keep)?;
        let total = t.sum_all()?; // []

        // Gumbel-max style draw (same form as the CPU sampler): u is drawn
        // in [1e-12, 1) so -ln(u) is strictly positive
        let u = Tensor::rand(1e-12f32, 1.0f32, v, dev)?;
        let noise = u.log()?.affine(-1.0, 0.0)?;
        let score = t.div(&noise)?.broadcast_div(&total)?;
        let rank = score.argmax_keepdim(0)?; // [1] u32
        Ok(order.index_select(&rank, 0)?) // [1] u32 token id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_zero_is_argmax() {
        let mut s = Sampler::new(42, false);
        let p = SampleParams {
            temperature: 1e-6,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        for _ in 0..20 {
            assert_eq!(s.sample(&logits, &p, None), argmax);
        }
    }

    #[test]
    fn repetition_penalty_demotes() {
        let mut s = Sampler::new(3, false);
        let cold = SampleParams {
            temperature: 1e-6,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let hot = SampleParams {
            repetition_penalty: 1.5,
            ..cold
        };
        let mut logits = vec![0f32; 10];
        logits[3] = 5.0; // clear winner
        logits[7] = 4.9;
        assert_eq!(s.sample(&logits, &cold, Some(&[3])), 3);
        // positive logits get divided by rp: 5/1.5 < 4.9 -> 7 wins
        assert_eq!(s.sample(&logits, &hot, Some(&[3])), 7);
    }

    // ---- sample_gpu (device-agnostic semantics tests; run on CPU) ----

    use candle_core::Device;

    /// Independent reference of the kept support after penalty + top-p,
    /// mirroring the CPU sampler's first half (used to bound the GPU
    /// sampler's output support).
    fn kept_support(logits: &[f32], params: &SampleParams, prev: Option<&[u32]>) -> Vec<u32> {
        let mut vals = logits.to_vec();
        if let Some(prev) = prev {
            for &t in prev {
                let i = t as usize;
                if i < vals.len() {
                    let s = vals[i];
                    vals[i] = if s < 0.0 {
                        s * params.repetition_penalty as f32
                    } else {
                        s / params.repetition_penalty as f32
                    };
                }
            }
        }
        let mut order: Vec<usize> = (0..vals.len()).collect();
        order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
        let max = vals[order[0]];
        let mut sum = 0f32;
        let exp_sorted: Vec<f32> = order
            .iter()
            .map(|&i| {
                let e = (vals[i] - max).exp();
                sum += e;
                e
            })
            .collect();
        let mut acc = 0f32;
        let mut kept = Vec::new();
        for (rank, &i) in order.iter().enumerate() {
            acc += exp_sorted[rank] / sum;
            if rank == 0 || !(acc > params.top_p as f32) {
                kept.push(i as u32);
            }
        }
        kept
    }

    #[test]
    fn cumsum1_matches_sequential() {
        let dev = Device::Cpu;
        for v in [1usize, 255, 256, 257, 1000, 4096, 10000] {
            let src: Vec<f32> = (0..v).map(|i| ((i as f32) * 0.03).sin() * 2.0).collect();
            let t = Tensor::from_vec(src.clone(), v, &dev).unwrap();
            let got = cumsum1(&t).unwrap().to_vec1::<f32>().unwrap();
            let mut acc = 0f32;
            for (i, x) in src.iter().enumerate() {
                acc += x;
                let want = acc;
                assert!(
                    (got[i] - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "v={v} i={i} got={} want={}",
                    got[i],
                    want
                );
            }
        }
    }

    #[test]
    fn sample_gpu_temp_zero_is_argmax() {
        let dev = Device::Cpu;
        let s = Sampler::new(42, false);
        let p = SampleParams {
            temperature: 1e-6,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        let t = Tensor::from_vec(logits, 100, &dev).unwrap();
        for _ in 0..20 {
            let out = s.sample_gpu(&t, &p, None).unwrap();
            assert_eq!(out.to_vec1::<u32>().unwrap()[0], argmax);
        }
    }

    #[test]
    fn sample_gpu_penalty_demotes() {
        let dev = Device::Cpu;
        let s = Sampler::new(3, false);
        let cold = SampleParams {
            temperature: 1e-6,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let hot = SampleParams {
            repetition_penalty: 1.5,
            ..cold
        };
        let mut logits = vec![0f32; 10];
        logits[3] = 5.0; // clear winner
        logits[7] = 4.9;
        let t = Tensor::from_vec(logits, 10, &dev).unwrap();
        let prev = Tensor::from_vec(vec![3u32], 1, &dev).unwrap();
        assert_eq!(
            s.sample_gpu(&t, &cold, None).unwrap().to_vec1::<u32>().unwrap()[0],
            3
        );
        // positive logits get divided by rp: 5/1.5 < 4.9 -> 7 wins
        assert_eq!(
            s.sample_gpu(&t, &hot, Some(&prev))
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0],
            7
        );
    }

    #[test]
    fn sample_gpu_support_within_top_p() {
        let dev = Device::Cpu;
        let s = Sampler::new(7, false);
        let p = SampleParams {
            temperature: 0.8,
            top_p: 0.5,
            repetition_penalty: 1.0,
        };
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).cos() * 3.0).collect();
        let kept = kept_support(&logits, &p, None);
        assert!(kept.len() >= 2, "reference support too small: {kept:?}");
        let t = Tensor::from_vec(logits, 64, &dev).unwrap();
        for _ in 0..300 {
            let tok = s
                .sample_gpu(&t, &p, None)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0];
            assert!(
                kept.contains(&tok),
                "sampled {tok} outside kept support {kept:?}"
            );
        }
    }
}
