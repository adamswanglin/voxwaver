use crate::config::DualArConfig;
use anyhow::{bail, ensure, Context, Result};
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::{rms_norm, silu};
use candle_nn::rotary_emb::rope_i;
use std::path::Path;


/// Scaled dot-product attention: softmax(q k^T * scale + mask) v.
/// q/k/v: [1, h, t, d] / [1, h, s, d]; `mask` additive, broadcastable to
/// [1, h, t, s]; `causal` adds a lower-triangular -inf mask.
/// On Metal this uses candle's fused SDPA kernels when safe (see the dispatch
/// note in the body): flash-attention style, no materialized score matrix,
/// handles GQA internally so k/v may stay un-repeated. Everything else takes
/// the manual matmul/softmax path and requires k/v repeated to the query
/// head count.
pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: f32,
) -> Result<Tensor> {
    // The sdpa_full kernel's causal path reads garbage at tile edges (NaN
    // outputs, data-dependent; see candle issue), so causal with q_seq > 1
    // uses the manual path. Non-causal and single-query causal (the vector
    // kernels) are safe.
    if q.device().is_metal() && (!causal || q.dim(2)? == 1) {
        return Ok(candle_nn::ops::sdpa(q, k, v, mask, causal, scale, 1.0)?);
    }
    let att = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?; // [1, h, t, s]
    let att = if causal {
        let (t, s) = (att.dim(2)?, att.dim(3)?);
        // queries sit at absolute positions s-t .. s-1 (kv-cache offset)
        let off = s - t;
        let mut m = vec![0f32; t * s];
        for r in 0..t {
            for c in (r + 1 + off)..s {
                m[r * s + c] = f32::NEG_INFINITY;
            }
        }
        let m = Tensor::from_vec(m, (1, 1, t, s), q.device())?.to_dtype(att.dtype())?;
        att.broadcast_add(&m)?
    } else if let Some(m) = mask {
        att.broadcast_add(m)?
    } else {
        att
    };
    let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
    Ok(att.matmul(v)?)
}

#[derive(Clone, Copy)]
struct HeadCfg {
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
}

impl HeadCfg {
    fn q_total(&self) -> usize {
        self.n_head * self.head_dim
    }
    fn kv_total(&self) -> usize {
        self.n_local_heads * self.head_dim
    }
}

pub struct Block {
    wqkv: Tensor,
    wo: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    w1: Tensor,
    w3: Tensor,
    w2: Tensor,
    attn_norm: Tensor,
    ffn_norm: Tensor,
}

pub struct DualArModel {
    cfg: DualArConfig,
    device: Device,
    dtype: DType,
    eps: f32,

    emb: Tensor,
    layers: Vec<Block>,
    norm: Tensor,
    head: Tensor,
    // fast transformer
    fast_emb: Tensor,
    fast_project_in: Option<Tensor>, // Linear if present, identity otherwise
    fast_layers: Vec<Block>,
    fast_norm: Tensor,
    fast_head: Tensor,

    // rope tables, bf16-rounded f32: [seq, head_dim/2]
    cos: Tensor,
    sin: Tensor,
    fcos: Tensor,
    fsin: Tensor,

    // slow kv cache per layer: [1, n_local_heads, capacity, head_dim]
    k_caches: Vec<Tensor>,
    v_caches: Vec<Tensor>,
    // fast kv cache per layer: [1, fast_n_local_heads, num_codebooks, fast_head_dim]
    fk_caches: Vec<Tensor>,
    fv_caches: Vec<Tensor>,

    scale: f64, // 1/sqrt(num_codebooks+1) if scale_codebook_embeddings else 1.0

    // on-device constants for embed_values: per-codebook views into the fused
    // codebook embedding table (ids index the view directly, no per-frame
    // offset kernel) and the semantic-range bounds `[1]` u32
    cb_tables: Vec<Tensor>,
    sem_lo: Tensor,
    sem_hi: Tensor,
}

/// bf16-rounded interleaved RoPE tables, cos/sin of shape [seq, head_dim/2].
fn rope_tables(
    seq_len: usize,
    head_dim: usize,
    base: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut freqs = vec![0f32; half];
    for i in 0..half {
        freqs[i] = base.powf(-(2.0 * i as f64) / head_dim as f64) as f32;
    }
    let mut cos_v = vec![0f32; seq_len * half];
    let mut sin_v = vec![0f32; seq_len * half];
    for t in 0..seq_len {
        for i in 0..half {
            let (s, c) = ((t as f32) * freqs[i]).sin_cos();
            cos_v[t * half + i] = c;
            sin_v[t * half + i] = s;
        }
    }
    let round = |v: Vec<f32>| -> Result<Tensor> {
        let t = Tensor::from_vec(v, (seq_len, half), &Device::Cpu)?;
        Ok(t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.to_device(dev)?)
    };
    Ok((round(cos_v)?, round(sin_v)?))
}

/// Maps logical weight names to raw pth keys, handling the optional
/// `state_dict` wrapper and `model.` prefix like `BaseTransformer::from_pretrained`.
struct KeyMap {
    prefix: String,
    pth: PthTensors,
}

impl KeyMap {
    fn open(path: &Path) -> Result<Self> {
        for (k, prefix) in [
            (None, ""),
            (Some("state_dict"), ""),
            (None, "model."),
        ] {
            if let Ok(candidate) = PthTensors::new(path, k) {
                let probe = format!("{prefix}embeddings.weight");
                if candidate.tensor_infos().contains_key(&probe) {
                    return Ok(Self {
                        prefix: prefix.to_string(),
                        pth: candidate,
                    });
                }
            }
        }
        bail!("no embeddings.weight found in {}", path.display())
    }

    fn get(&self, dtype: DType, dev: &Device, name: &str) -> Result<Tensor> {
        let full = format!("{}{}", self.prefix, name);
        match self.pth.get(&full)? {
            Some(t) => Ok(t.to_dtype(dtype)?.to_device(dev)?),
            None => bail!("missing tensor {full}"),
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.pth
            .tensor_infos()
            .contains_key(&format!("{}{}", self.prefix, name))
    }
}

impl DualArModel {
    pub fn load(
        model_pth: &Path,
        cfg: &DualArConfig,
        sem_begin: u32,
        sem_end: u32,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let km = KeyMap::open(model_pth)?;
        let eps = cfg.norm_eps as f32;
        let g = |name: &str| km.get(dtype, dev, name);

        let emb = g("embeddings.weight")?;
        let cb_emb = g("codebook_embeddings.weight")?;
        let head = if cfg.tie_word_embeddings {
            emb.clone()
        } else {
            g("output.weight")?
        };
        let norm = g("norm.weight")?;

        let hc = HeadCfg {
            n_head: cfg.n_head,
            n_local_heads: cfg.n_local_heads(),
            head_dim: cfg.head_dim,
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            layers.push(load_block(&km, &format!("layers.{i}."), hc, cfg, false, dtype, dev)?);
        }

        let fhc = HeadCfg {
            n_head: cfg.fast_n_head(),
            n_local_heads: cfg.fast_n_local_heads(),
            head_dim: cfg.fast_head_dim(),
        };
        let fast_emb = g("fast_embeddings.weight")?;
        let mut fast_layers = Vec::with_capacity(cfg.n_fast_layer);
        for i in 0..cfg.n_fast_layer {
            fast_layers.push(load_block(
                &km,
                &format!("fast_layers.{i}."),
                fhc,
                cfg,
                true,
                dtype,
                dev,
            )?);
        }
        let fast_norm = g("fast_norm.weight")?;
        let fast_head = g("fast_output.weight")?;
        let fast_project_in = if cfg.fast_dim() != cfg.dim {
            Some(g("fast_project_in.weight")?)
        } else {
            ensure!(
                !km.contains("fast_project_in.weight"),
                "fast_dim == dim but fast_project_in.weight present"
            );
            None
        };

        let (cos, sin) = rope_tables(cfg.max_seq_len, cfg.head_dim, cfg.rope_base, dev)?;
        let (fcos, fsin) = rope_tables(cfg.num_codebooks, cfg.fast_head_dim(), cfg.rope_base, dev)?;

        let scale = if cfg.scale_codebook_embeddings {
            (cfg.num_codebooks as f64 + 1.0).sqrt().recip()
        } else {
            1.0
        };
        let cb_tables: Vec<Tensor> = (0..cfg.num_codebooks)
            .map(|i| {
                Ok(cb_emb.narrow(0, i * cfg.codebook_size, cfg.codebook_size)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let sem_lo = Tensor::from_vec(vec![sem_begin], 1, dev)?;
        let sem_hi = Tensor::from_vec(vec![sem_end], 1, dev)?;

        Ok(Self {
            cfg: cfg.clone(),
            device: dev.clone(),
            dtype,
            eps,
            emb,
            layers,
            norm,
            head,
            fast_emb,
            fast_project_in,
            fast_layers,
            fast_norm,
            fast_head,
            cos,
            sin,
            fcos,
            fsin,
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            fk_caches: Vec::new(),
            fv_caches: Vec::new(),
            scale,
            cb_tables,
            sem_lo,
            sem_hi,
        })
    }

    /// Allocate the KV caches for a generation with a prompt of `prompt_len`
    /// tokens and at most `max_new` generated frames.
    pub fn setup_caches(&mut self, prompt_len: usize, max_new: usize) -> Result<()> {
        let capacity = prompt_len + max_new;
        let shape = (1usize, self.cfg.n_local_heads(), capacity, self.cfg.head_dim);
        self.k_caches = vec![Tensor::zeros(shape, self.dtype, &self.device)?; self.layers.len()];
        self.v_caches = self.k_caches.clone();
        let fshape = (
            1usize,
            self.cfg.fast_n_local_heads(),
            self.cfg.num_codebooks,
            self.cfg.fast_head_dim(),
        );
        self.fk_caches =
            vec![Tensor::zeros(fshape, self.dtype, &self.device)?; self.fast_layers.len()];
        self.fv_caches = self.fk_caches.clone();
        Ok(())
    }

    /// Drop the KV caches (they are re-allocated by the next `setup_caches`).
    /// Call between generations: an idle cache for the full context capacity
    /// is hundreds of MB that the codec decode no longer needs to coexist
    /// with.
    pub fn release_caches(&mut self) {
        self.k_caches = Vec::new();
        self.v_caches = Vec::new();
        self.fk_caches = Vec::new();
        self.fv_caches = Vec::new();
    }

    /// fish `BaseTransformer.forward_generate` embedding logic:
    /// semantic positions get (emb + sum of codebook embeddings) / sqrt(n_cb+1).
    ///
    /// Entirely on-device: the caller uploads the `[rows, T]` u32 ids once
    /// and every row is sliced as a view (the old per-row readback cost 11
    /// blocking device-to-host syncs per decode frame).
    fn embed_values(&self, values: &Tensor) -> Result<Tensor> {
        let (rows, t) = values.dims2()?;
        ensure!(rows == self.cfg.num_codebooks + 1, "values rows {rows}");
        let flat = values.reshape(rows * t)?; // [rows*t] u32 contiguous view
        let ids0 = flat.narrow(0, 0, t)?; // [t] semantic row
        let mut x = self.emb.embedding(&ids0)?.unsqueeze(0)?; // [1, T, D]
        if self.scale != 1.0 {
            let mut cb_sum: Option<Tensor> = None;
            for i in 0..self.cfg.num_codebooks {
                // codebook i ids index the per-codebook view directly (the view
                // starts at row i*codebook_size of the fused table)
                let ids = flat.narrow(0, (i + 1) * t, t)?;
                let e = self.cb_tables[i].embedding(&ids)?; // [t, D]
                cb_sum = Some(match cb_sum {
                    None => e,
                    Some(s) => s.add(&e)?,
                });
            }
            let cb_sum = cb_sum.context("no codebooks")?;
            // semantic-position mask, evaluated on-device
            let m = ids0
                .broadcast_ge(&self.sem_lo)?
                .mul(&ids0.broadcast_le(&self.sem_hi)?)?; // [t] u8
            let mask = m.to_dtype(x.dtype())?.reshape((1, t, 1))?;
            let combined = x.add(&cb_sum.unsqueeze(0)?.broadcast_mul(&mask)?)?;
            let scaled = combined.affine(self.scale as f64, 0.0)?;
            x = combined
                .broadcast_mul(&mask.affine(-1.0, 1.0)?)?
                .add(&scaled.broadcast_mul(&mask)?)?;
        }
        Ok(x)
    }

    /// One forward over `values` ([11, T] u32) placed at `start_pos`.
    /// Returns (last-position logits [1,1,V], last-position pre-norm hidden [1,1,D]).
    pub fn forward(&mut self, values: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let t = values.dim(1)?;
        ensure!(start_pos + t <= self.cache_capacity(), "kv cache overflow");
        let mut x = self.embed_values(values)?;
        let hc = HeadCfg {
            n_head: self.cfg.n_head,
            n_local_heads: self.cfg.n_local_heads(),
            head_dim: self.cfg.head_dim,
        };
        // prefill from position 0 is plain causal (kernel flag, no mask tensor);
        // a chunked prefill with a cache prefix needs an explicit additive mask
        // [1, n_head, T, start_pos + T] (Metal sdpa requires the head dim)
        let (mask, _causal) = if t > 1 && start_pos > 0 {
            let k_len = start_pos + t;
            let mut m = vec![0f32; t * k_len];
            for r in 0..t {
                for c in (start_pos + r + 1)..k_len {
                    m[r * k_len + c] = f32::NEG_INFINITY;
                }
            }
            let m = Tensor::from_vec(m, (1, 1, t, k_len), &self.device)?
                .to_dtype(self.dtype)?
                .expand((1, hc.n_head, t, k_len))?;
            (Some(m), false)
        } else {
            (None, t > 1)
        };

        for i in 0..self.layers.len() {
            let mut kc = self.k_caches[i].clone();
            let mut vc = self.v_caches[i].clone();
            x = self.block_forward(
                &self.layers[i],
                hc,
                &x,
                start_pos,
                &mut kc,
                &mut vc,
                &self.cos,
                &self.sin,
                mask.as_ref(),
            )?;
            self.k_caches[i] = kc;
            self.v_caches[i] = vc;
        }
        let xs = x.narrow(1, t - 1, 1)?; // [1, 1, D] pre-norm hidden
        let slow_out = rms_norm(&xs, &self.norm, self.eps)?;
        let logits = slow_out.matmul(&self.head.t()?.unsqueeze(0)?)?; // [1, 1, V]
        Ok((logits, xs))
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        &self,
        block: &Block,
        hc: HeadCfg,
        x: &Tensor,
        start_pos: usize,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t = x.dim(1)?;
        let h = rms_norm(x, &block.attn_norm, self.eps)?;
        let qkv = h.matmul(&block.wqkv.t()?.unsqueeze(0)?)?; // [1, T, q+2kv]
        let (q_total, _kv_total) = (hc.q_total(), hc.kv_total());
        let (q, k, v) = split_qkv(&qkv, hc, t)?;

        let (q, k) = if let (Some(qn), Some(kn)) = (&block.q_norm, &block.k_norm) {
            (rms_norm(&q, qn, self.eps)?, rms_norm(&k, kn, self.eps)?)
        } else {
            (q, k)
        };

        // layout (1, h, t, d) for rope + attention; rope_i indexes dims4 as
        // (b, h, seq, head_dim) with cos [seq, head_dim/2]
        let cos = cos.narrow(0, start_pos, t)?.contiguous()?;
        let sin = sin.narrow(0, start_pos, t)?.contiguous()?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let q = rope_f32(&q, &cos, &sin, self.dtype)?;
        let k = rope_f32(&k, &cos, &sin, self.dtype)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let r = 0..1;
        let rh = 0..hc.n_local_heads;
        let rd = 0..hc.head_dim;
        *k_cache = k_cache.slice_assign(&[r.clone(), rh.clone(), start_pos..start_pos + t, rd.clone()], &k)?;
        *v_cache = v_cache.slice_assign(&[r, rh, start_pos..start_pos + t, rd], &v)?;
        let k_all = k_cache.narrow(2, 0, start_pos + t)?;
        let v_all = v_cache.narrow(2, 0, start_pos + t)?;

        // fused Metal SDPA applies GQA itself; the manual path needs repeated
        // k/v. The sdpa *full* kernel's causal path reads garbage at tile
        // edges (NaN outputs, data-dependent; see candle issue), so causal
        // prefill (t > 1, no mask) falls back to the manual matmul/softmax;
        // single-token decode keeps the fast vector kernel.
        let causal_prefill = mask.is_none() && t > 1;
        let (k_all, v_all) =
            if self.device.is_metal() && !causal_prefill {
                (k_all, v_all)
            } else {
                let rep = hc.n_head / hc.n_local_heads;
                (repeat_kv(&k_all, rep)?, repeat_kv(&v_all, rep)?)
            };

        let y = attention(
            &q,
            &k_all,
            &v_all,
            mask,
            causal_prefill, // plain causal prefill
            1.0 / (hc.head_dim as f32).sqrt(),
        )?;
        let y = y
            .transpose(1, 2)?
            .reshape((1, t, q_total))?
            .matmul(&block.wo.t()?.unsqueeze(0)?)?;

        let attn_out = x.add(&y)?;
        ffn(block, &attn_out, self.eps)
    }

    /// One fast-transformer step at position `idx` (0..num_codebooks).
    fn fast_step(&mut self, x: &Tensor, idx: usize) -> Result<Tensor> {
        let mut x = x.clone();
        let fhc = HeadCfg {
            n_head: self.cfg.fast_n_head(),
            n_local_heads: self.cfg.fast_n_local_heads(),
            head_dim: self.cfg.fast_head_dim(),
        };
        for i in 0..self.fast_layers.len() {
            let mut kc = self.fk_caches[i].clone();
            let mut vc = self.fv_caches[i].clone();
            x = self.fast_block_forward(&self.fast_layers[i], fhc, &x, idx, &mut kc, &mut vc)?;
            self.fk_caches[i] = kc;
            self.fv_caches[i] = vc;
        }
        let out = rms_norm(&x, &self.fast_norm, self.eps)?;
        Ok(out.matmul(&self.fast_head.t()?.unsqueeze(0)?)?) // [1, 1, codebook_size]
    }

    fn fast_block_forward(
        &self,
        block: &Block,
        fhc: HeadCfg,
        x: &Tensor,
        idx: usize,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
    ) -> Result<Tensor> {
        let h = rms_norm(x, &block.attn_norm, self.eps)?;
        let qkv = h.matmul(&block.wqkv.t()?.unsqueeze(0)?)?;
        let (q_total, _kv_total) = (fhc.q_total(), fhc.kv_total());
        let (q, k, v) = split_qkv(&qkv, fhc, 1)?;

        let (q, k) = if let (Some(qn), Some(kn)) = (&block.q_norm, &block.k_norm) {
            (rms_norm(&q, qn, self.eps)?, rms_norm(&k, kn, self.eps)?)
        } else {
            (q, k)
        };

        let cos = self.fcos.narrow(0, idx, 1)?.contiguous()?;
        let sin = self.fsin.narrow(0, idx, 1)?.contiguous()?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let q = rope_f32(&q, &cos, &sin, self.dtype)?;
        let k = rope_f32(&k, &cos, &sin, self.dtype)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let r = 0..1;
        let rh = 0..fhc.n_local_heads;
        let rd = 0..fhc.head_dim;
        *k_cache = k_cache.slice_assign(&[r.clone(), rh.clone(), idx..idx + 1, rd.clone()], &k)?;
        *v_cache = v_cache.slice_assign(&[r, rh, idx..idx + 1, rd], &v)?;
        let k_all = k_cache.narrow(2, 0, idx + 1)?;
        let v_all = v_cache.narrow(2, 0, idx + 1)?;
        // fused Metal SDPA applies GQA itself; the manual path needs repeated k/v
        let (k_all, v_all) = if self.device.is_metal() {
            (k_all, v_all)
        } else {
            let rep = fhc.n_head / fhc.n_local_heads;
            (repeat_kv(&k_all, rep)?, repeat_kv(&v_all, rep)?)
        };
        let y = attention(&q, &k_all, &v_all, None, false, 1.0 / (fhc.head_dim as f32).sqrt())?;
        let y = y.transpose(1, 2)?.reshape((1, 1, q_total))?.matmul(&block.wo.t()?.unsqueeze(0)?)?;
        let attn_out = x.add(&y)?;
        ffn(block, &attn_out, self.eps)
    }

    /// One fast-transformer frame: seed with the projected slow hidden, then
    /// predict codebooks 1..9 sampled from the full codebook_size logits.
    /// Returns [num_codebooks] codes ([semantic, residual...]).
    ///
    /// `cb_windows[k]` is the repetition-penalty window for codebook k+1
    /// (`previous_tokens[codebook_idx + 1]` upstream); codebook 0 is the
    /// clamped semantic code and is never penalized.
    /// Debug: one greedy (argmax) fast frame; returns codes and per-step
    /// logits rows for comparison against a reference implementation.
    pub fn fast_frame_greedy(
        &mut self,
        hidden: &Tensor,
        sem_code: u32,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let x0 = match &self.fast_project_in {
            Some(w) => hidden.matmul(&w.t()?.unsqueeze(0)?)?,
            None => hidden.clone(),
        };
        self.fast_step(&x0, 0)?;
        let a0 = sem_code.min(self.cfg.codebook_size as u32 - 1);
        let mut codes = vec![a0];
        let mut rows = Vec::new();
        let mut x = embed_one(&self.fast_emb, a0, &self.device)?;
        for idx in 1..self.cfg.num_codebooks {
            let logits = self.fast_step(&x, idx)?;
            let row = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let a = row
                .iter()
                .enumerate()
                .max_by(|(_, p), (_, q)| p.total_cmp(q))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            rows.push(row);
            codes.push(a);
            x = embed_one(&self.fast_emb, a, &self.device)?;
        }
        Ok((codes, rows))
    }

    pub fn fast_frame(
        &mut self,
        hidden: &Tensor,
        sem_code: u32,
        sampler: &crate::sampling::Sampler,
        params: &crate::sampling::SampleParams,
        cb_windows: &[Option<Tensor>],
    ) -> Result<Vec<u32>> {
        let x0 = match &self.fast_project_in {
            Some(w) => hidden.matmul(&w.t()?.unsqueeze(0)?)?,
            None => hidden.clone(),
        };
        // position 0: warm the cache with the (projected) hidden; logits discarded
        {
            let _g = crate::prof::scope(crate::prof::FAST_SUBMIT);
            self.fast_step(&x0, 0)?;
        }

        let a0 = sem_code.min(self.cfg.codebook_size as u32 - 1);
        let mut ids = vec![Tensor::from_vec(vec![a0], 1, &self.device)?]; // [1] u32 each
        let mut x = embed_one(&self.fast_emb, a0, &self.device)?; // [1, 1, fast_dim]
        for idx in 1..self.cfg.num_codebooks {
            let logits = {
                let _g = crate::prof::scope(crate::prof::FAST_SUBMIT);
                self.fast_step(&x, idx)?
            };
            let window = cb_windows.get(idx - 1).and_then(|w| w.as_ref());
            let a = {
                let _g = crate::prof::scope(crate::prof::FAST_SAMPLE);
                sampler.sample_gpu(&logits, params, window)?
            };
            // stay on-device: embed straight from the [1] u32 id tensor
            x = self.fast_emb.embedding(&a)?.unsqueeze(0)?; // [1, 1, fast_dim]
            ids.push(a);
        }
        // single pipeline drain per frame instead of one per codebook
        let codes = {
            let _g = crate::prof::scope(crate::prof::FAST_READ);
            Tensor::cat(&ids, 0)?.to_vec1::<u32>()?
        };
        Ok(codes)
    }

    pub fn cache_capacity(&self) -> usize {
        if self.k_caches.is_empty() {
            0
        } else {
            self.k_caches[0].dim(2).unwrap_or(0)
        }
    }
}

/// Split fused qkv [1, t, q+2kv] into per-head layouts (1, t, h, d).
fn split_qkv(qkv: &Tensor, hc: HeadCfg, t: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (q_total, kv_total) = (hc.q_total(), hc.kv_total());
    let q = qkv.narrow(2, 0, q_total)?.reshape((1, t, hc.n_head, hc.head_dim))?;
    let k = qkv
        .narrow(2, q_total, kv_total)?
        .reshape((1, t, hc.n_local_heads, hc.head_dim))?;
    let v = qkv
        .narrow(2, q_total + kv_total, kv_total)?
        .reshape((1, t, hc.n_local_heads, hc.head_dim))?;
    Ok((q, k, v))
}

/// Apply interleaved rope in f32 (cos/sin already bf16-rounded), cast back.
/// Input/output layout: (1, h, t, d).
fn rope_f32(x: &Tensor, cos: &Tensor, sin: &Tensor, dtype: DType) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    Ok(rope_i(&x32, cos, sin)?.to_dtype(dtype)?)
}

fn embed_one(emb: &Tensor, id: u32, dev: &Device) -> Result<Tensor> {
    let ids = Tensor::from_vec(vec![id], 1, dev)?;
    Ok(emb.embedding(&ids)?.unsqueeze(0)?) // [1, 1, D]
}

fn ffn(block: &Block, attn_out: &Tensor, eps: f32) -> Result<Tensor> {
    let h2 = rms_norm(attn_out, &block.ffn_norm, eps)?;
    let a = h2.matmul(&block.w1.t()?.unsqueeze(0)?)?;
    let b = h2.matmul(&block.w3.t()?.unsqueeze(0)?)?;
    let ff = silu(&a)?.mul(&b)?.matmul(&block.w2.t()?.unsqueeze(0)?)?;
    Ok(attn_out.add(&ff)?)
}

fn load_block(
    km: &KeyMap,
    p: &str,
    hc: HeadCfg,
    cfg: &DualArConfig,
    is_fast: bool,
    dtype: DType,
    dev: &Device,
) -> Result<Block> {
    let q_total = hc.q_total();
    let kv_total = hc.kv_total();
    let qk_norm = if is_fast {
        cfg.fast_attention_qk_norm()
    } else {
        cfg.attention_qk_norm
    };
    let dim = if is_fast { cfg.fast_dim() } else { cfg.dim };
    let wqkv = km.get(dtype, dev, &format!("{p}attention.wqkv.weight"))?;
    ensure!(
        wqkv.dims() == [q_total + 2 * kv_total, dim],
        "unexpected wqkv shape {:?} for {p}",
        wqkv.dims()
    );
    Ok(Block {
        wqkv,
        wo: km.get(dtype, dev, &format!("{p}attention.wo.weight"))?,
        q_norm: if qk_norm {
            Some(km.get(dtype, dev, &format!("{p}attention.q_norm.weight"))?)
        } else {
            None
        },
        k_norm: if qk_norm {
            Some(km.get(dtype, dev, &format!("{p}attention.k_norm.weight"))?)
        } else {
            None
        },
        w1: km.get(dtype, dev, &format!("{p}feed_forward.w1.weight"))?,
        w3: km.get(dtype, dev, &format!("{p}feed_forward.w3.weight"))?,
        w2: km.get(dtype, dev, &format!("{p}feed_forward.w2.weight"))?,
        attn_norm: km.get(dtype, dev, &format!("{p}attention_norm.weight"))?,
        ffn_norm: km.get(dtype, dev, &format!("{p}ffn_norm.weight"))?,
    })
}

/// candle-transformers' repeat_kv, vendored to avoid the extra dependency.
fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b, h, t, d) = xs.dims4()?;
    Ok(xs
        .unsqueeze(2)?
        .expand((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))?)
}
