"""Official PyTorch DAC decode total time, min of 3 runs in one fresh process.
Run from the fish-speech repo root with the lightyear env (see dac_phase_prof_torch.py).
"""
import time
import numpy as np
import torch

from fish_speech.models.dac.inference import load_model

DEV = "mps"
model = load_model("modded_dac_vq", "/Users/wanglin/app/projects/github/s1-mini/codec.pth", device=DEV)
codes = torch.from_numpy(np.load("/tmp/fs_runA/codes_0.npy")).to(DEV).long()
if codes.ndim == 2:
    codes = codes.unsqueeze(0)
print(f"codes: {tuple(codes.shape)}")

with torch.no_grad():
    for it in range(3):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        y, _ = model.decode(codes[0], torch.tensor([codes.shape[-1]], device=DEV))
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        print(f"run{it}: {dt*1000:.0f} ms  out={tuple(y.shape)}")
print("done")
