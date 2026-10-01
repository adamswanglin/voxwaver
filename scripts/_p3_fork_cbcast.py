#!/usr/bin/env python3
"""P3: channel-broadcast binary-op specialization for the candle Metal fork.

Adds a `binary_kernel_cbcast` Metal kernel + `call_binary_cbcast` dispatcher
and routes `{badd,bsub,bmul,bdiv,bminimum,bmaximum}` f32/f16/bf16 ops through
it when the right operand is a per-channel vector broadcast along the last
dim (stride pattern `(?, 1, 0)`, e.g. Snake's [1,C,1] alpha expanded to
[1,C,L]). The generic strided kernel was measured ~4x slower on this pattern
(44 ms vs 11 ms for [1,192,T]).

Idempotency: every patch asserts its `old` text is present exactly once; if
the repo already has the change, the patch fails loudly instead of double
applying.
"""
import sys

CANDLE = "/Users/wanglin/app/projects/github/candle"

METAL = f"{CANDLE}/candle-metal-kernels/src/metal_src/binary.metal"
BRIDGE = f"{CANDLE}/candle-metal-kernels/src/kernels/binary.rs"
BACKEND = f"{CANDLE}/candle-core/src/metal_backend/mod.rs"

patches = [
    # ---- 1. binary.metal: the kernel itself ----
    (
        METAL,
        """        uint l_idx = l_index(i, num_dims, dims, left_strides);
        uint r_idx = r_index(i, num_dims, dims, right_strides);
        output[i] = static_cast<U>(op(left[l_idx], right[r_idx]));
    }
}

// Macros to help initialize kernels""",
        """        uint l_idx = l_index(i, num_dims, dims, left_strides);
        uint r_idx = r_index(i, num_dims, dims, right_strides);
        output[i] = static_cast<U>(op(left[l_idx], right[r_idx]));
    }
}

// Specialized kernel for a right operand that varies only per channel and is
// broadcast along the last dim (e.g. a per-channel vector [1, C, 1] expanded
// to [B, C, L]). Each thread reads its channel value once, avoiding the
// per-element multi-dimensional index math of binary_kernel_strided (~4x
// slower on such channel broadcasts).
template <typename T, typename U, typename binary>
[[kernel]] void binary_kernel_cbcast(
    constant uint &l_dim,
    constant uint &c_dim,
    device const T *left,
    device const T *right,
    device U *output,
    uint2 pos [[ thread_position_in_grid ]]
) {
    binary op;
    const uint bc = pos.y;
    const T r = right[bc % c_dim];
    const uint i = bc * l_dim + pos.x;
    output[i] = static_cast<U>(op(left[i], r));
}

// Macros to help initialize kernels""",
    ),
    # ---- 2. binary.metal: init macro + instantiations ----
    (
        METAL,
        """// Initialize kernels
init_binary(badd);
init_binary(bsub);
init_binary(bmul);
init_binary(bdiv);
init_binary(bminimum);
init_binary(bmaximum);

init_boolean_binary(eq, beq);""",
        """// Initialize kernels
init_binary(badd);
init_binary(bsub);
init_binary(bmul);
init_binary(bdiv);
init_binary(bminimum);
init_binary(bmaximum);

// Channel-broadcast variants of the numeric ops (see binary_kernel_cbcast).
#define init_binary_cbcast(op_name, binary_op, tname, t, u) \\
    init_kernel(#op_name "_" #tname "_cbcast", binary_kernel_cbcast, t, u, binary_op)

#if defined(__HAVE_BFLOAT__)
#define init_binary_cb(bop) \\
    init_binary_cbcast(bop, bop, f32, float, float) \\
    init_binary_cbcast(bop, bop, f16, half, half) \\
    init_binary_cbcast(bop, bop, bf16, bfloat, bfloat)
#else
#define init_binary_cb(bop) \\
    init_binary_cbcast(bop, bop, f32, float, float) \\
    init_binary_cbcast(bop, bop, f16, half, half)
#endif

init_binary_cb(badd);
init_binary_cb(bsub);
init_binary_cb(bmul);
init_binary_cb(bdiv);
init_binary_cb(bminimum);
init_binary_cb(bmaximum);

init_boolean_binary(eq, beq);""",
    ),
    # ---- 3. binary.rs: import MTLSize ----
    (
        BRIDGE,
        """use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};""",
        """use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source, MTLSize,
};""",
    ),
    # ---- 4. binary.rs: the dispatcher ----
    (
        BRIDGE,
        """            right_strides,
            &left_input,
            &right_input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}""",
        """            right_strides,
            &left_input,
            &right_input,
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Channel-broadcast binary op: `right` holds `c_dim` per-channel values
/// broadcast along the last dim of a `[b, c, l]` left operand (e.g. the
/// [1, C, 1] Snake alpha expanded to [B, C, L]). One thread per output
/// element on a 2-D grid `(l, b * c)`; each thread reads its channel value
/// once instead of re-doing strided index math per element.
#[allow(clippy::too_many_arguments)]
pub fn call_binary_cbcast<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    l_dim: usize,
    c_dim: usize,
    bc_dim: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    if [l_dim, c_dim, bc_dim].iter().any(|&v| v > u32::MAX as usize) {
        return Err(MetalKernelError::InvalidInput(format!(
            "binary_cbcast dims exceed u32: l={l_dim} c={c_dim} bc={bc_dim}"
        )));
    }
    let kernel_name = kernel_name.to_string();
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.clone())?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let threads_per_grid = MTLSize {
        width: l_dim,
        height: bc_dim,
        depth: 1,
    };
    let tg_w = 64.min(l_dim.max(1));
    let threads_per_threadgroup = MTLSize {
        width: tg_w,
        height: (256 / tg_w).max(1),
        depth: 1,
    };
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "binary_cbcast {kernel_name} l={l_dim} c={c_dim} bc={bc_dim}");
    set_params!(
        encoder,
        (
            l_dim as u32,
            c_dim as u32,
            &left,
            &right,
            Output::new(output)
        )
    );
    encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    Ok(())
}""",
    ),
    # ---- 5. metal_backend: route per-channel broadcasts to cbcast ----
    (
        BACKEND,
        """        let lhs_is_scalar = lhs_l.is_scalar_like();
        let rhs_is_scalar = rhs_l.is_scalar_like();
        let lhs_contiguous = lhs_l.is_contiguous();
        let rhs_contiguous = rhs_l.is_contiguous();

        let contiguous_kernel = kernel_name(op, &self.dtype, "");""",
        """        let lhs_is_scalar = lhs_l.is_scalar_like();
        let rhs_is_scalar = rhs_l.is_scalar_like();
        let lhs_contiguous = lhs_l.is_contiguous();
        let rhs_contiguous = rhs_l.is_contiguous();

        let contiguous_kernel = kernel_name(op, &self.dtype, "");

        // Fast path for a right operand that varies only per channel and is
        // broadcast along the last dim (e.g. the [1, C, 1] alpha of a Snake
        // layer expanded to [B, C, L]). The dedicated kernel reads each
        // channel value once, skipping the per-element strided index math
        // that made this pattern ~4x slower than a contiguous op.
        if !lhs_is_scalar
            && !rhs_is_scalar
            && lhs_contiguous
            && dtype == self.dtype
            && matches!(self.dtype, DType::F32 | DType::F16 | DType::BF16)
            && lhs_l.dims().len() == 3
        {
            let dims = lhs_l.dims();
            let rs = rhs_l.stride();
            if rhs_l.dims() == dims
                && rs[1] == 1
                && rs[2] == 0
                && (rs[0] == 0 || dims[0] == 1)
                && el_count <= u32::MAX as usize
                && dims[1] <= u32::MAX as usize
                && dims[2] <= u32::MAX as usize
                && dims[0] * dims[1] <= u32::MAX as usize
            {
                let buffer = device
                    .new_buffer_builder()
                    .with_size_for(el_count, dtype)
                    .with_label(op)
                    .build()?;
                candle_metal_kernels::call_binary_cbcast(
                    &device.device,
                    &encoder,
                    &device.kernels,
                    kernel_name(op, &self.dtype, "_cbcast"),
                    dims[2],
                    dims[1],
                    dims[0] * dims[1],
                    lhs,
                    rhs,
                    &buffer,
                )
                .map_err(MetalError::from)?;
                return Ok(Self::new(buffer, device.clone(), el_count, dtype));
            }
        }""",
    ),
]

ok = True
for i, (path, old, new) in enumerate(patches, 1):
    with open(path) as f:
        src = f.read()
    count = src.count(old)
    if count != 1:
        print(f"patch {i} [{path.split('/')[-1]}]: FAIL expected 1 occurrence, found {count}")
        ok = False
        continue
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"patch {i} [{path.split('/')[-1]}]: applied ({len(old)} -> {len(new)} chars)")

sys.exit(0 if ok else 1)
