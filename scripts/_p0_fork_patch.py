#!/usr/bin/env python3
"""One-shot patch for the candle fork (../candle): tiled 32-bit im2col1d.

Patches:
  1. candle-metal-kernels/src/metal_src/conv.metal : im2col1d kernel + macro
  2. candle-metal-kernels/src/kernels/convolution.rs : call_im2col1d_strided
  3. candle-core/src/metal_backend/mod.rs : conv1d direct-path condition + tiled im2col loop

Each patch asserts the original snippet occurs exactly once, so re-running a
partially applied script fails loudly instead of corrupting the file.
"""
import sys

FORK = "/Users/wanglin/app/projects/github/candle"

PATCHES = []

# ---------------------------------------------------------------- conv.metal
PATCHES.append((f"{FORK}/candle-metal-kernels/src/metal_src/conv.metal", r'''template <typename T>
METAL_FUNC void im2col1d(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &l_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, l_out, c_in, l_k)
  // src: (b_size, c_in, l_in)
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t dst_s2 = l_k;
  const size_t dst_s1 = c_in * dst_s2;
  const size_t dst_s0 = l_out * dst_s1;

  size_t tmp_dst_i = tid;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t l_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= l_idx * dst_s1;
  const size_t c_idx = tmp_dst_i / dst_s2;
  tmp_dst_i -= c_idx * dst_s2;
  const size_t l_k_idx = tmp_dst_i;
  size_t src_l_idx = l_idx * stride + l_k_idx * dilation;
  if (src_l_idx < padding || src_l_idx >= l_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_l_idx -= padding;
    const size_t src_i = b_idx * src_strides[0] + c_idx * src_strides[1] + src_l_idx * src_strides[2];
    dst[tid] = src[src_i];
  }
}''', r'''// Tiled 1d im2col: one thread per (c_in, l, b) coordinate writes the l_k taps
// of a single output position as a contiguous run into dst: (b, l_out_tile, c_in, l_k).
//
// Unlike the previous flat one-thread-per-element form this does not derive its
// coordinates by dividing a flat thread id (three 64-bit divisions per output
// *element* dominated the runtime of large convolutions, making them ALU-bound
// rather than memory-bound), but from a 3D grid, and all index math is 32-bit.
// The l_start/l_out_tile pair lets the caller materialize the matrix in
// bounded tiles instead of one allocation of l_out * c_in * l_k elements
// (multiple GiB for long audio).
//
// Each thread reads up to l_k input taps (stride*dilation apart); consecutive
// threads along x cover consecutive channels of the same l, which keeps both
// the loads (128-byte lines are shared by neighbouring l threads) and the
// stores (each thread writes a contiguous l_k run, adjacent across threads)
// cache friendly.
template <typename T>
METAL_FUNC void im2col1d(
    constant uint &l_out_tile,
    constant uint &l_start,
    constant uint &c_in,
    constant uint &l_in,
    constant uint &l_k,
    constant uint &stride,
    constant uint &padding,
    constant uint &dilation,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint3 pos [[ thread_position_in_grid ]]
) {
  // dst: (b_size, l_out_tile, c_in, l_k); src: (b_size, c_in, l_in)
  // pos: (c_in index, l index within tile, batch index)
  const uint c_idx = pos.x;
  const uint l_t = pos.y;
  const uint b_idx = pos.z;
  device T *dst_row =
    dst + ((size_t)((b_idx * l_out_tile + l_t) * c_in + c_idx)) * (size_t)l_k;
  const size_t src_row = (size_t)b_idx * src_strides[0] + (size_t)c_idx * src_strides[1];
  const size_t l_pad = (size_t)(l_start + l_t) * stride;  // padded position of the first tap
  const size_t p_end = (size_t)l_in + padding;
  if (l_pad >= padding && l_pad + (l_k - 1) * dilation < p_end) {
    // Whole window in range: no per-tap bounds checks.
    const size_t src_first = src_row + (size_t)(l_pad - padding) * src_strides[2];
    const size_t step = (size_t)dilation * src_strides[2];
    for (uint j = 0; j < l_k; ++j) {
      dst_row[j] = src[src_first + (size_t)j * step];
    }
  } else {
    for (uint j = 0; j < l_k; ++j) {
      const size_t p = l_pad + (size_t)j * dilation;
      if (p < padding || p >= p_end) {
        dst_row[j] = static_cast<T>(0);
      } else {
        dst_row[j] = src[src_row + (size_t)(p - padding) * src_strides[2]];
      }
    }
  }
}'''))

PATCHES.append((f"{FORK}/candle-metal-kernels/src/metal_src/conv.metal", r'''#define IM2COL1D_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &dst_numel, \
    constant size_t &l_out, \
    constant size_t &l_k, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &dilation, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    device const T *src, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  im2col1d<T>(dst_numel, l_out, l_k, stride, padding, dilation, src_dims, src_strides, src, dst, tid); \
} \
''', r'''#define IM2COL1D_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant uint &l_out_tile, \
    constant uint &l_start, \
    constant uint &c_in, \
    constant uint &l_in, \
    constant uint &l_k, \
    constant uint &stride, \
    constant uint &padding, \
    constant uint &dilation, \
    constant size_t *src_strides, \
    device const T *src, \
    device T *dst, \
    uint3 pos [[ thread_position_in_grid ]] \
) {  \
  im2col1d<T>(l_out_tile, l_start, c_in, l_in, l_k, stride, padding, dilation, src_strides, src, dst, pos); \
} \
'''))

# ------------------------------------------------------------ convolution.rs
PATCHES.append((f"{FORK}/candle-metal-kernels/src/kernels/convolution.rs", r'''#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col1d {name} dst_el={dst_el}");
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            k_size,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}''', r'''/// Tiled 1d im2col: materializes the rows for `l in [l_start, l_start + l_out_tile)`
/// of the (b_size, l_out, c_in, k_size) matrix into a contiguous scratch buffer.
/// One thread per (c_in, l, b) coordinate, taken directly from a 3D grid, so no
/// flat-id divisions are needed and the kernel index math stays 32-bit.
#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    (l_start, l_out_tile): (usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let (b_size, c_in, l_in) = (shape[0], shape[1], shape[2]);
    if [b_size, c_in, l_in, k_size, stride, padding, dilation, l_start, l_out_tile]
        .iter()
        .any(|&v| v > u32::MAX as usize)
    {
        return Err(MetalKernelError::InvalidInput(format!(
            "im2col1d dimensions exceed u32: shape={shape:?} k={k_size} l_start={l_start} tile={l_out_tile}"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let threads_per_grid = MTLSize {
        width: c_in,
        height: l_out_tile,
        depth: b_size,
    };
    let tg_w = 64.min(c_in.max(1));
    let threads_per_threadgroup = MTLSize {
        width: tg_w,
        height: (256 / tg_w).max(1),
        depth: 1,
    };
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col1d {name} l_start={l_start} tile={l_out_tile}");
    set_params!(
        encoder,
        (
            l_out_tile as u32,
            l_start as u32,
            c_in as u32,
            l_in as u32,
            k_size as u32,
            stride as u32,
            padding as u32,
            dilation as u32,
            strides,
            &input,
            Output::new(output)
        )
    );
    encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    Ok(())
}'''))

# ------------------------------------------------- candle-core mod.rs part 1
PATCHES.append((f"{FORK}/candle-core/src/metal_backend/mod.rs", r'''        // Direct convolution path, in the spirit of the implicit-GEMM kernels
        // used by cuDNN / MPSGraph which never materialize the im2col matrix.
        // The explicit im2col buffer holds l_out * c_in * k elements (k times
        // the input size). Two cases favor the direct kernel, which computes
        // each output from the input taps on the fly:
        // - a small reduction dimension (c_in * k): candle lowers grouped
        //   convs to per-group dense calls at the tensor level, so a depthwise
        //   conv becomes many calls with c_in == 1, where the im2col + matmul
        //   round-trip per call is far more expensive than the direct compute;
        // - an im2col buffer above ~4 GiB, where materializing it (plus the
        //   buffer pool retaining it) costs multiple GiB of memory for a
        //   conv that only runs ~5x slower directly. Large dense convs keep
        //   the im2col path, whose matmul runs ~2 TFLOP/s on Apple GPUs vs
        //   ~0.1 for the scalar direct kernel.
        if matches!(self.dtype, DType::F32 | DType::F16 | DType::BF16) {
            let (b_size, c_in, _l_in) = (dims[0], dims[1], dims[2]);
            let (c_out, _k_c_in, k_size) = {
                let kd = kernel_l.shape().dims();
                (kd[0], kd[1], kd[2])
            };
            let l_out = params.l_out();
            let im2col_el = b_size * l_out * c_in * k_size;
            let small_reduction = c_in * k_size <= 64;
            let huge_im2col = im2col_el * self.dtype.size_in_bytes() > 1 << 32;
            if small_reduction || huge_im2col {''', r'''        // Direct convolution path, in the spirit of the implicit-GEMM kernels
        // used by cuDNN / MPSGraph which never materialize the im2col matrix.
        // The explicit im2col buffer holds l_out * c_in * k elements (k times
        // the input size). A small reduction dimension (c_in * k) favors the
        // direct kernel, which computes each output from the input taps on the
        // fly: candle lowers grouped convs to per-group dense calls at the
        // tensor level, so a depthwise conv becomes many calls with c_in == 1,
        // where the im2col + matmul round-trip per call is far more expensive
        // than the direct compute. Large dense convs take the tiled im2col
        // path below, whose matmul runs ~2 TFLOP/s on Apple GPUs vs ~0.1 for
        // the scalar direct kernel while the tiling keeps the materialized
        // matrix within a fixed scratch budget.
        if matches!(self.dtype, DType::F32 | DType::F16 | DType::BF16) {
            let (b_size, c_in, _l_in) = (dims[0], dims[1], dims[2]);
            let (c_out, _k_c_in, k_size) = {
                let kd = kernel_l.shape().dims();
                (kd[0], kd[1], kd[2])
            };
            let l_out = params.l_out();
            if c_in * k_size <= 64 {'''))

# ------------------------------------------------- candle-core mod.rs part 2
PATCHES.append((f"{FORK}/candle-core/src/metal_backend/mod.rs", r'''        let stride = params.stride;
        let dilation = params.dilation;
        let padding = params.padding;
        let k_size = params.k_size;
        let l_out = (dims[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
        let dst_el = dims[0] * l_out * dims[1] * k_size;
        let dst = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("conv1d_im2col")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let name = match self.dtype {
            DType::F32 => "im2col1d_f32",
            DType::F16 => "im2col1d_f16",
            DType::BF16 => "im2col1d_bf16",
            DType::U8 => "im2col1d_u8",
            DType::U32 => "im2col1d_u32",
            dtype => crate::bail!("Metal conv1d {dtype:?} not implemented"),
        };
        let src = buffer_o(&self.buffer, layout, self.dtype);
        candle_metal_kernels::call_im2col1d_strided(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            layout.shape().dims(),
            strides,
            (k_size, stride, padding, dilation),
            src,
            &dst,
        )
        .map_err(MetalError::from)?;
        drop(encoder);
        let col = Self {
            buffer: dst,
            device,
            count: dst_el,
            dtype: self.dtype,
        };
        let l_out = params.l_out();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), res.dtype())?;
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }''', r'''        // im2col + GEMM path. The matrix is materialized in tiles of at most
        // IM2COL_SCRATCH_BYTES so that a long sequence (l_out * c_in * k can
        // reach multiple GiB per conv) never exceeds a fixed scratch
        // footprint; each tile runs its own GEMM into the full
        // (b, l_out, c_out) accumulator, and the final layout conversion is
        // identical to the untiled path.
        const IM2COL_SCRATCH_BYTES: usize = 64 << 20;
        let stride = params.stride;
        let dilation = params.dilation;
        let padding = params.padding;
        let k_size = params.k_size;
        let l_out = (dims[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
        let b_size = dims[0];
        let c_in = dims[1];
        let n = params.c_out;
        let k = k_size * c_in;
        let row_el = c_in * k_size;
        let budget_el = (IM2COL_SCRATCH_BYTES / self.dtype.size_in_bytes()).max(1);
        let tile_l = (budget_el / (b_size.max(1) * row_el)).clamp(1, l_out.max(1));
        let n_tiles = l_out.div_ceil(tile_l);

        let name = match self.dtype {
            DType::F32 => "im2col1d_f32",
            DType::F16 => "im2col1d_f16",
            DType::BF16 => "im2col1d_bf16",
            DType::U8 => "im2col1d_u8",
            DType::U32 => "im2col1d_u32",
            dtype => crate::bail!("Metal conv1d {dtype:?} not implemented"),
        };
        // The GEMM reads the weights densely; copy strided ones once. The copy
        // lands at offset 0 of the fresh buffer, so its layout starts at 0.
        let kernel_c = if kernel_l.is_contiguous() {
            None
        } else {
            let mut kc = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kc, 0, kernel_l)?;
            Some(kc)
        };
        let (kernel_ref, kernel_offset) = match &kernel_c {
            None => (kernel, kernel_l.start_offset()),
            Some(kc) => (kc, 0),
        };
        let kernel_l_b = Layout::contiguous_with_offset((1, n, k), kernel_offset)
            .transpose(1, 2)?
            .broadcast_as((b_size, k, n))?;

        // (b, l_out, n) accumulator: the tiles below write every element
        // exactly once, so the buffer can stay uninitialized.
        let res_el = b_size * l_out * n;
        let res_buf = self
            .device
            .new_buffer_builder()
            .with_size_for(res_el, self.dtype)
            .with_label("conv1d_gemm")
            .build()?;
        let mut res_full = Self {
            buffer: res_buf,
            device: device.clone(),
            count: res_el,
            dtype: self.dtype,
        };

        for tile_idx in 0..n_tiles {
            let l_start = tile_idx * tile_l;
            let tile_len = tile_l.min(l_out - l_start);
            let scratch_el = b_size * tile_len * row_el;
            let dst = self
                .device
                .new_buffer_builder()
                .with_size_for(scratch_el, self.dtype)
                .with_label("conv1d_im2col")
                .build()?;
            let src = buffer_o(&self.buffer, layout, self.dtype);
            let encoder = self.device.command_encoder()?;
            candle_metal_kernels::call_im2col1d_strided(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                name,
                dims,
                strides,
                (k_size, stride, padding, dilation),
                (l_start, tile_len),
                src,
                &dst,
            )
            .map_err(MetalError::from)?;
            drop(encoder);
            let col = Self {
                buffer: dst,
                device: device.clone(),
                count: scratch_el,
                dtype: self.dtype,
            };
            let col_l = Layout::contiguous((b_size, tile_len, k));
            let res = col.matmul(kernel_ref, (b_size, tile_len, n, k), &col_l, &kernel_l_b)?;
            // res: (b_size, tile_len, n) contiguous -> scatter each batch slice
            // into the (b_size, l_out, n) accumulator (contiguous copy).
            for b_idx in 0..b_size {
                let tile_src =
                    Layout::contiguous_with_offset((1, tile_len, n), b_idx * tile_len * n)?;
                res.copy_strided_src(
                    &mut res_full,
                    b_idx * l_out * n + l_start * n,
                    &tile_src,
                )?;
            }
        }

        let res_l = Layout::contiguous((b_size, l_out, n)).transpose(1, 2)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), self.dtype)?;
        res_full.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }'''))

# --------------------------------------------------------------- include fix
PATCHES.append((f"{FORK}/candle-metal-kernels/src/kernels/convolution.rs", r'''use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};''', r'''use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source, MTLSize,
};'''))


def main():
    for i, (path, old, new) in enumerate(PATCHES):
        with open(path, "r") as f:
            text = f.read()
        count = text.count(old)
        if count != 1:
            print(f"PATCH {i} FAILED on {path}: original snippet occurs {count} times")
            sys.exit(1)
        with open(path, "w") as f:
            f.write(text.replace(old, new, 1))
        print(f"PATCH {i} ok: {path}")
    print("all patches applied")


if __name__ == "__main__":
    main()
