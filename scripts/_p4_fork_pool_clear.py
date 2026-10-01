#!/usr/bin/env python3
"""Patch the candle fork with an explicit Metal pool release API.

Adds:
1. MetalDevice::release_unused_buffers() in metal_backend/device.rs —
   flush+wait, then sweep every pooled buffer whose strong_count==1.
2. Device::clear_metal_pool() in candle-core/src/device.rs — public enum-level
   wrapper (no-op on CPU/Cuda).

Run once from the voxwaver workspace root. Idempotent: each replacement
asserts exactly one match.
"""
import pathlib

CANDLE = pathlib.Path("/Users/wanglin/app/projects/github/candle")

def patch(path, old, new, name):
    p = pathlib.Path(path)
    s = p.read_text()
    assert s.count(old) == 1, f"{name}: expected 1 match, got {s.count(old)}"
    p.write_text(s.replace(old, new))
    print(f"applied: {name}")

# --- 1. MetalDevice::release_unused_buffers ---------------------------------
device_rs = CANDLE / "candle-core/src/metal_backend/device.rs"
anchor = """    pub fn kernels(&self) -> &Kernels {
        &self.kernels"""
new_fn = """    /// Blocks until all submitted GPU work finishes, then releases every
    /// pooled buffer no longer referenced by the compute graph. The pool is
    /// also swept on each `synchronize()`; this explicit entry point targets
    /// generation boundaries in long-running apps, where idle pooled buffers
    /// still count towards the process memory footprint.
    pub fn release_unused_buffers(&self) -> Result<()> {
        self.commands
            .flush_and_wait_current()
            .map_err(MetalError::from)?;
        self.drop_unused_buffers()
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels"""
patch(device_rs, anchor, new_fn, "MetalDevice::release_unused_buffers")

# --- 2. Device::clear_metal_pool --------------------------------------------
core_device = CANDLE / "candle-core/src/device.rs"
anchor2 = """    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
        }
    }
}"""
new2 = """    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
        }
    }

    /// Metal only: wait for pending GPU work, then release every pooled
    /// buffer that is no longer alive in the compute graph. No-op on Cpu/Cuda.
    /// Call at generation boundaries to hand idle GPU memory back to the OS.
    pub fn clear_metal_pool(&self) -> Result<()> {
        match self {
            Self::Cpu | Self::Cuda(_) => Ok(()),
            Self::Metal(d) => d.release_unused_buffers(),
        }
    }
}"""
patch(core_device, anchor2, new2, "Device::clear_metal_pool")

print("all patches applied")
