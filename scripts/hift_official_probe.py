# Official-code ground truth for the vocoder: build CausalHiFTGenerator from
# the yaml, load hift.pt, vocode the exported anchor mel, dump stats.
# Usage: python3 scripts/hift_official_probe.py
import sys

CV = '/Users/wanglin/app/projects/github/CosyVoice'
sys.path.insert(0, CV)
import numpy as np
import soundfile as sf
import torch
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from cosyvoice.hifigan.generator import CausalHiFTGenerator

MD = '/Users/wanglin/app/projects/github/Fun-CosyVoice3-0.5B-2512'

# Params from cosyvoice3.yaml hift: (mel_spec_transform in the yaml only
# matters for training, so we build the generator directly).
f0p = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
model = CausalHiFTGenerator(
    in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=24000,
    nsf_alpha=0.1, nsf_sigma=0.003, nsf_voiced_threshold=10,
    upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
    istft_params={'n_fft': 16, 'hop_len': 4},
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    source_resblock_kernel_sizes=[7, 7, 11],
    source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
    f0_predictor=f0p)
sd = torch.load(MD + '/hift.pt', map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing:', [k for k in missing][:6], 'unexpected:', [k for k in unexpected][:6])
model.eval()

mel = torch.from_numpy(np.fromfile('anchor_mel.f32', dtype=np.float32).reshape(-1, 80).T)[None]  # [1,80,T]

# Intercept mag/phase right before the ISTFT.
orig_istft = model._istft
def istft_probe(mag, phase):
    mag.detach().flatten().numpy().tofile('dump_mag.f32')
    phase.detach().flatten().numpy().tofile('dump_phase.f32')
    return orig_istft(mag, phase)
model._istft = istft_probe

with torch.no_grad():
    audio, s = model.inference(mel)
    # inference() has moved f0_predictor to float64 by now; re-run for the dump.
    f0 = model.f0_predictor(mel.to(torch.float64)).to(torch.float32)
    f0.flatten().numpy().tofile('dump_f0.f32')
audio = audio.squeeze().numpy()
s.squeeze().numpy().tofile('dump_s.f32')
audio.tofile('dump_audio.f32')
# Fixed per-process uniform noise buffer used by SineGen2 in eval+causal mode.
model.m_source.l_sin_gen.sine_waves[:, :len(audio), :].numpy().tofile('dump_sinebuf.f32')
# STFT of the source as consumed by decode() (cat[re, im]).
sr_, si_ = model._stft(s.squeeze(1))
torch.cat([sr_, si_], 1).flatten().numpy().tofile('dump_sstft.f32')

# Replay decode() stage by stage with the same dumps as the Rust side.
import torch.nn.functional as F
with torch.no_grad():
    s_stft = torch.cat([sr_, si_], 1)
    x = model.conv_pre(mel)
    x.flatten().numpy().tofile('dump_d_pre.f32')
    for i in range(3):
        x = F.leaky_relu(x, 0.1)
        x = model.ups[i](x)
        x.flatten().numpy().tofile(f'dump_d{i}_up.f32')
        if i == 2:
            x = model.reflection_pad(x)
        sij = model.source_downs[i](s_stft)
        if i == 0:
            sij.flatten().numpy().tofile('dump_d0_sd.f32')
            rb = model.source_resblocks[0]
            print('alpha1[0] first4:', rb.activations1[0].alpha.flatten()[:4].tolist())
            xt = rb.activations1[0](sij)
            xt.flatten().numpy().tofile('dump_d0_s10.f32')
            xt = rb.convs1[0](xt)
            xt.flatten().numpy().tofile('dump_d0_c10.f32')
        sij = model.source_resblocks[i](sij)
        sij.flatten().numpy().tofile(f'dump_d{i}_si.f32')
        x = x + sij
        xs = None
        for j in range(3):
            r = model.resblocks[i * 3 + j](x)
            xs = r if xs is None else xs + r
        x = xs / 3
        x.flatten().numpy().tofile(f'dump_d{i}_x.f32')
sf.write('voc_official.wav', audio, 24000)
rms = float(np.sqrt((audio**2).mean()))
zcr = float(np.diff(np.signbit(audio)).mean())
print(f'voc_official.wav: n={len(audio)} rms={rms:.3f} zcr={zcr:.3f} peak={np.abs(audio).max():.3f}')
print('source s stats: rms=%.3f zcr=%.3f' % (
    float(np.sqrt((s.numpy()**2).mean())), float(np.diff(np.signbit(s.numpy())).mean())))
with torch.no_grad():
    f0_probe = model.f0_predictor(mel.to(torch.float64)).flatten()[:8]
print('f0 head:', f0_probe.tolist())
