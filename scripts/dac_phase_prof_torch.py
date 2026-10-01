"""Phase-level profiling of the official PyTorch DAC decode on MPS.

Run from the fish-speech repo root with the lightyear env:

  cd /Users/wanglin/app/projects/github/fish-speech
  PYTHONPATH=/tmp/audiotools_stub:/tmp/pypath_extra \
    ~/miniconda3/envs/lightyear/bin/python \
    /Users/wanglin/app/projects/github/voxwaver/scripts/dac_phase_prof_torch.py

Times: quantizer.decode (from_codes / post_module / upsample) and
decoder (conv_in / each block / tail) plus a few isolated heavy ops.
"""
import time
import numpy as np
import torch

from fish_speech.models.dac.inference import load_model

DEV = "mps"
torch.manual_seed(0)

model = load_model("modded_dac_vq", "/Users/wanglin/app/projects/github/s1-mini/codec.pth", device=DEV)
codes = torch.from_numpy(np.load("/tmp/fs_runA/codes_0.npy")).to(DEV).long()
if codes.ndim == 2:
    codes = codes.unsqueeze(0)
print(f"codes: {tuple(codes.shape)}, dtype={codes.dtype}, device={codes.device}")


def bench(fn, n=3, warm=1):
    out = None
    for _ in range(warm):
        out = fn()
        torch.mps.synchronize()
    ts = []
    for _ in range(n):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


with torch.no_grad():
    # ---- full pipeline ----
    t_total, (y, alen) = bench(lambda: model.decode(codes[0], torch.tensor([codes.shape[-1]], device=DEV)), n=2)
    print(f"[total] model.decode: {t_total*1000:.0f} ms  out={tuple(y.shape)}")

    # ---- quantizer.decode (full) ----
    t_q, z = bench(lambda: model.quantizer.decode(codes), n=2)
    print(f"[quantizer.decode] {t_q*1000:.0f} ms  out={tuple(z.shape)}")

    # ---- quantizer.decode internals (replicated) ----
    q = model.quantizer
    idx = codes

    def from_codes_part():
        new_indices = torch.zeros_like(idx)
        new_indices[:, 0] = torch.clamp(idx[:, 0], max=q.semantic_quantizer.codebook_size - 1)
        new_indices[:, 1:] = torch.clamp(idx[:, 1:], max=q.quantizer.codebook_size - 1)
        z_sem = q.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_res = q.quantizer.from_codes(new_indices[:, 1:])[0]
        return z_sem + z_res

    t_fc, z0 = bench(from_codes_part, n=3)
    t_pm, z1 = bench(lambda: q.post_module(z0), n=3)
    t_up, z2 = bench(lambda: q.upsample(z1), n=3)
    print(
        f"[quantizer] from_codes={t_fc*1000:.0f} ms  post_module={t_pm*1000:.0f} ms  "
        f"upsample={t_up*1000:.0f} ms (sum={(t_fc+t_pm+t_up)*1000:.0f})"
    )

    # upsample sub-modules
    up0 = q.upsample[0]
    up1 = q.upsample[1]
    t_u0, _ = bench(lambda: up0(z1), n=3)
    t_u1, _ = bench(lambda: up1(up0(z1)), n=3)
    print(f"[upsample] up[0]={t_u0*1000:.0f} ms  up[1]={t_u1*1000:.0f} ms")
    for bi, blk in enumerate(q.upsample):
        for mi, m in enumerate(blk):
            t_b, _ = bench(lambda m=m, z=z1 if bi == 0 else up0(z1): m(z), n=2)
            print(f"  upsample[{bi}][{mi}] {type(m).__name__}: {t_b*1000:.0f} ms")

    # ---- decoder (full) ----
    t_d, y2 = bench(lambda: model.decoder(z), n=2)
    print(f"[decoder] {t_d*1000:.0f} ms  out={tuple(y2.shape)}")

    # ---- decoder.model walk ----
    x = z
    for i, m in enumerate(model.decoder.model):
        t_i, x = bench(lambda m=m, x=x: m(x), n=2)
        print(f"  decoder.model[{i}] {type(m).__name__}: {t_i*1000:.0f} ms  out={tuple(x.shape)}")

    # ---- inside decoder.model[3] (B3, stride 4) ----
    blk = model.decoder.model[3]
    xi = z
    for j in range(3):
        xi = model.decoder.model[j](xi)
    for i, m in enumerate(blk.block):
        t_i, xi = bench(lambda m=m, x=xi: m(x), n=2)
        print(f"  B3.block[{i}] {type(m).__name__}: {t_i*1000:.0f} ms  out={tuple(xi.shape)}")

    # ---- isolated heavy ops ----
    conv_b3 = model.decoder.model[3].block[4].block[1]
    xs = torch.randn(1, 192, 364544, device=DEV)
    t_op, _ = bench(lambda: conv_b3(xs), n=3)
    print(f"[op] RU k7-dil9 conv [1,192,364544]: {t_op*1000:.0f} ms")

    conv_b2 = model.decoder.model[2].block[4].block[1]
    xs2 = torch.randn(1, 384, 91136, device=DEV)
    t_op2, _ = bench(lambda: conv_b2(xs2), n=3)
    print(f"[op] RU k7-dil9 conv [1,384,91136]: {t_op2*1000:.0f} ms")

    tconv_b3 = model.decoder.model[3].block[1]
    xs3 = torch.randn(1, 384, 91136, device=DEV)
    t_op3, _ = bench(lambda: tconv_b3(xs3), n=3)
    print(f"[op] tconv [1,384,91136]->s8: {t_op3*1000:.0f} ms")

    tconv_b1 = model.decoder.model[1].block[1]
    xs4 = torch.randn(1, 1536, 1424, device=DEV)
    t_op4, _ = bench(lambda: tconv_b1(xs4), n=3)
    print(f"[op] tconv [1,1536,1424]->s8: {t_op4*1000:.0f} ms")

    t_op5, _ = bench(lambda: up1[1](torch.randn(1, 1024, 712, device=DEV)), n=3)
    print(f"[op] convnext(1024ch, T=712): {t_op5*1000:.0f} ms")

print("done")
