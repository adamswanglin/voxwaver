"""Companion isolated-op bench for the ops missing from dac_phase_prof_torch.py:
k1 convs, B4/B1 k7d9 convs, conv_out (96->1), the Snake chain, and the
broadcast scale/shift ops, all at the exact 356-frame DAC shapes.

Run from the fish-speech repo root with the lightyear env:

  cd /Users/wanglin/app/projects/github/fish-speech
  PYTHONPATH=/Users/wanglin/app/projects/github/fish-speech:/tmp/audiotools_stub:/tmp/pypath_extra \
    ~/miniconda3/envs/lightyear/bin/python \
    /Users/wanglin/app/projects/github/voxwaver/scripts/dac_ops_prof_torch.py
"""
import time
import torch

from fish_speech.models.dac.modded_dac import CausalConvNet, Snake1d

DEV = "mps"
torch.manual_seed(0)


def bench(label, fn, n=5, warm=2):
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
    dt = min(ts)
    print(f"  {label}: {dt*1000:.1f} ms")
    return out


with torch.no_grad():
    print("=== torch MPS isolated ops (f32, 356-frame DAC shapes) ===")

    # k1 convs (RU c2) at each decoder stage
    for label, c, t in [
        ("B1 k1 [1,768,11392]", 768, 11392),
        ("B2 k1 [1,384,91136]", 384, 91136),
        ("B3 k1 [1,192,364544]", 192, 364544),
        ("B4 k1 [1,96,729088]", 96, 729088),
    ]:
        x = torch.randn(1, c, t, device=DEV)
        w = torch.randn(c, c, 1, device=DEV) * 0.1
        bench(label, lambda x=x, w=w: torch.conv1d(x, w, padding=0))

    # k7d9 convs at B1/B4 (B2/B3 already measured: ~91/90 ms)
    for label, c, t in [
        ("B1 k7d9 [1,768,11392]", 768, 11392),
        ("B4 k7d9 [1,96,729088]", 96, 729088),
    ]:
        x = torch.randn(1, c, t, device=DEV)
        w = torch.randn(c, c, 7, device=DEV) * 0.1
        bench(label, lambda x=x, w=w: torch.conv1d(x, w, padding=0, dilation=9))

    # conv_out 96 -> 1 (CausalConvNet k7)
    conv_out = CausalConvNet(96, 1, 7)
    conv_out = conv_out.to(DEV).float()
    x96 = torch.randn(1, 96, 729088, device=DEV)
    bench("conv_out k7 [1,96,729088]->[1,1]", lambda: conv_out(x96))

    # Snake chain at each scale (the model's own module, f32)
    for label, c, t in [
        ("B1 snake [1,768,11392]", 768, 11392),
        ("B2 snake [1,384,91136]", 384, 91136),
        ("B3 snake [1,192,364544]", 192, 364544),
        ("B4 snake [1,96,729088]", 96, 729088),
    ]:
        sn = Snake1d(c).to(DEV).float()
        x = torch.randn(1, c, t, device=DEV)
        bench(label, lambda sn=sn, x=x: sn(x))

    # broadcast scale/shift building blocks (Snake internals)
    x3 = torch.randn(1, 192, 364544, device=DEV)
    a1 = torch.randn(1, 192, 1, device=DEV)
    a_full = torch.randn(1, 192, 364544, device=DEV)
    bench("mul [1,192,T] * scalar-tensor()", lambda: x3 * 0.5)
    bench("mul [1,192,T] * [1,192,1] (broadcast)", lambda: x3 * a1, n=10)
    bench("mul [1,192,T] * [1,192,T] (full)", lambda: x3 * a_full, n=10)
    bench("sin [1,192,364544]", lambda: torch.sin(x3), n=10)
    bench("add [1,192,364544]", lambda: x3 + x3, n=10)

    # narrow (non-contiguous) input conv, as in the real flow
    xfull = torch.randn(1, 192, 364550, device=DEV)
    xn = xfull[:, :, :364544]
    wn = torch.randn(192, 192, 7, device=DEV) * 0.1
    bench("B3 k7d9 narrow-view input", lambda: torch.conv1d(xn, wn, padding=0, dilation=9))

print("done")
