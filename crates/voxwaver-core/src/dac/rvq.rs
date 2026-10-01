//! Residual vector quantizer (descript-style `VectorQuantize` /
//! `ResidualVectorQuantize`) plus the fish `DownsampleResidualVectorQuantize`
//! wrapper with semantic + residual codebooks.
use super::layers::{ConvNeXt, ConvW, TransConvW};
use super::prof;
use super::transformer::Wlt;
use anyhow::Result;
use candle_core::Tensor;

/// l2-normalize rows (F.normalize with default eps 1e-12).
fn l2norm_rows(x: &Tensor) -> Result<Tensor> {
    let norm = x.powf(2.0)?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    let norm = norm.maximum(&Tensor::full(1e-12f32, norm.shape(), x.device())?)?;
    Ok(x.broadcast_div(&norm)?.contiguous()?)
}

pub struct Vq {
    in_proj: ConvW,   // k=1
    out_proj: ConvW,  // k=1
    codebook: Tensor, // [N, codebook_dim]
    pub codebook_size: usize,
}

impl Vq {
    pub fn new(in_proj: ConvW, out_proj: ConvW, codebook: Tensor) -> Self {
        let codebook_size = codebook.dims()[0];
        Self { in_proj, out_proj, codebook, codebook_size }
    }

    /// codes [T] -> [1, input_dim, T] (raw embedding lookup, no norm, then out_proj).
    pub fn decode_code(&self, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.flatten_all()?; // argmax keeps dims -> [T, 1]
        let e = self.codebook.embedding(&ids)?; // [T, D8]
        let e = e.t()?.unsqueeze(0)?.contiguous()?; // [1, D8, T]
        self.out_proj.forward(&e)
    }

    /// z [1, input_dim, T] -> (z_q [1, input_dim, T], indices [T] u32).
    /// Nearest neighbor in the l2-normalized low-dim space; ties/argmax follow
    /// torch (first max wins) via candle's argmax.
    pub fn quantize(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_e = self.in_proj.forward(z)?; // [1, D8, T]
        let t = z_e.dim(2)?;
        let zt = z_e.squeeze(0)?.t()?.contiguous()?; // [T, D8]
        let zn = l2norm_rows(&zt)?;
        let cbn = l2norm_rows(&self.codebook)?;
        // dist = |z|^2 - 2 z.cb + |cb|^2; the per-row |z|^2 constant does not
        // affect the argmin, so score = 2 z.cb - |cb|^2
        let scores = zn.matmul(&cbn.t()?)?.affine(2.0, 0.0)?; // [T, N]
        let cb_sq = cbn.powf(2.0)?.sum_keepdim(candle_core::D::Minus1)?; // [N, 1]
        let scores = scores.broadcast_sub(&cb_sq.t()?.unsqueeze(0)?)?;
        let indices = scores
            .argmax(candle_core::D::Minus1)?
            .flatten_all()?; // [T] u32
        let z_q = self.decode_code(&indices.to_device(z.device())?)?;
        let _ = t;
        Ok((z_q, indices))
    }
}

pub struct Rvq {
    pub quantizers: Vec<Vq>,
}

impl Rvq {
    /// Residual encoding: z [1, D, T] -> (z_q, codes [n][T]).
    pub fn encode(&self, z: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let mut residual = z.clone();
        let mut z_q: Option<Tensor> = None;
        let mut codes = Vec::with_capacity(self.quantizers.len());
        for q in &self.quantizers {
            let (z_q_i, idx) = q.quantize(&residual)?;
            z_q = Some(match &z_q {
                None => z_q_i.clone(),
                Some(acc) => acc.add(&z_q_i)?,
            });
            residual = residual.sub(&z_q_i)?;
            codes.push(idx);
        }
        Ok((z_q.unwrap(), codes))
    }

    /// codes: one row per quantizer, each of length T -> [1, D, T].
    pub fn from_codes(&self, codes: &[Vec<u32>], device: &candle_core::Device) -> Result<Tensor> {
        let mut z_q: Option<Tensor> = None;
        for (i, row) in codes.iter().enumerate() {
            let ids = Tensor::from_vec(row.clone(), row.len(), device)?;
            let z_q_i = self.quantizers[i].decode_code(&ids)?;
            z_q = Some(match &z_q {
                None => z_q_i,
                Some(acc) => acc.add(&z_q_i)?,
            });
        }
        Ok(z_q.expect("empty rvq"))
    }
}

pub struct DownRvq {
    semantic: Rvq,
    residual: Rvq,
    post_module: Wlt,
    up: Vec<(TransConvW, ConvNeXt)>,
    pub semantic_codebook_size: usize,
    pub residual_codebook_size: usize,
}

impl DownRvq {
    pub fn new(
        semantic: Rvq,
        residual: Rvq,
        post_module: Wlt,
        up: Vec<(TransConvW, ConvNeXt)>,
        semantic_codebook_size: usize,
        residual_codebook_size: usize,
    ) -> Self {
        Self { semantic, residual, post_module, up, semantic_codebook_size, residual_codebook_size }
    }

    /// Quantizer input [1, 1024, T] (already through the encoder half and
    /// `pre_module`, see `Dac::encode`) -> codes [10][T/4].
    pub fn encode_latent(&self, z: &Tensor) -> Result<Vec<Vec<u32>>> {
        let (sem_z, sem_codes) = self.semantic.encode(z)?;
        let residual_in = z.sub(&sem_z)?;
        let (_res_z, res_codes) = self.residual.encode(&residual_in)?;
        let mut codes: Vec<Vec<u32>> = Vec::with_capacity(1 + res_codes.len());
        codes.push(sem_codes[0].to_vec1::<u32>()?);
        for c in &res_codes {
            codes.push(c.to_vec1::<u32>()?);
        }
        Ok(codes)
    }

    /// codes [10][T] (already ordered semantic + 9 residual) -> [1, 1024, 4T].
    pub fn decode(&self, codes: &[Vec<u32>], device: &candle_core::Device) -> Result<Tensor> {
        let clamped: Vec<Vec<u32>> = codes
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let max = if i == 0 {
                    self.semantic_codebook_size - 1
                } else {
                    self.residual_codebook_size - 1
                };
                row.iter().map(|&v| v.min(max as u32)).collect()
            })
            .collect();
        let sem = prof::phase(prof::Q_FROM_CODES, device, || {
            let sem = self.semantic.from_codes(&clamped[..1], device)?;
            let res = self.residual.from_codes(&clamped[1..], device)?;
            Ok(sem.add(&res)?)
        })?;
        let z = prof::phase(prof::Q_POST, device, || self.post_module.forward(&sem))?;
        let z = prof::phase(prof::Q_UP, device, || {
            let mut z = z;
            for (i, (conv, cn)) in self.up.iter().enumerate() {
                z = prof::phase(prof::Q_UP0_TCONV + 2 * i, device, || conv.forward(&z))?;
                z = prof::phase(prof::Q_UP0_CN + 2 * i, device, || cn.forward(&z))?;
            }
            Ok(z)
        })?;
        Ok(z)
    }
}
