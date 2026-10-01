"""Reference dumps for the OmniVoice voice-clone encoder (Rust port parity).

Runs the HiggsAudioV2TokenizerModel.encode reference path step by step on a
reference wav and dumps every intermediate tensor as raw little-endian f32
(or i32 for codes) plus a manifest.json describing shapes.

Requires the arm64 venv with transformers>=5.13 + torch + torchaudio:

    /Users/wanglin/omni-ref-venv-arm64/bin/python scripts/omni_clone_dump.py \
        --model-dir ../OmniVoice --audio out.wav --out-dir /tmp/omni_clone_ref

Outputs (all shapes in the manifest, row-major):
    ref_in.f32      [L_in]          input wav, float32 mono at its native rate
    ref24k.f32      [L24]           resampled to 24 kHz (pipeline input)
    ref16k.f32      [L16]           resampled 24k -> 16k (semantic input)
    x16_pad.f32     [L16+320]       16k waveform padded by (160, 160)
    hub_hidden.f32  [13, T_hub, 768]  semantic_model hidden states (stacked)
    sem_mean.f32    [T_hub, 768]    mean over the 13 hidden states
    sem_feat.f32    [T_sem, 768]    after [:, ::2, :] downsample
    e_semantic.f32  [768, T_sem]    encoder_semantic output
    e_acoustic.f32  [256, T_sem]    acoustic encoder output (pad if applied)
    emb_fc.f32      [T_sem, 1024]   fc output
    codes.i32       [8, T_sem]      RVQ encode codes
    pos_w.f32       [768, 48, 128]  merged pos_conv weight (weight-norm)
    pos_g.f32       [1, 1, 128]     pos_conv original0
    pos_v.f32       [768, 48, 128]  pos_conv original1
    rsk.f32         [2, 1, 23]      resample kernel 24k->16k
    ref24k.wav                      24 kHz float32 wav (feed straight to Rust)
    ref_in.wav                      input wav re-saved as float32 (Rust input)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional.functional as taf


def dump_raw(out_dir: Path, name: str, arr: np.ndarray, manifest: dict):
    flat = np.ascontiguousarray(arr).reshape(-1)
    if flat.dtype == np.float32:
        suffix = "f32"
    elif flat.dtype == np.int64 or flat.dtype == np.int32:
        flat = flat.astype(np.int32)
        suffix = "i32"
    else:
        raise TypeError(f"unsupported dtype {flat.dtype}")
    path = out_dir / f"{name}.{suffix}"
    flat.tofile(path)
    manifest[name] = {"shape": list(arr.shape), "dtype": suffix}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="../OmniVoice")
    ap.add_argument("--audio", default="out.wav")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}

    from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

    model = HiggsAudioV2TokenizerModel.from_pretrained(Path(args.model_dir) / "audio_tokenizer").eval()

    wav, sr = sf.read(args.audio, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(-1)
    print(f"input: {wav.shape[0]} samples @ {sr} Hz")
    dump_raw(out_dir, "ref_in", wav, manifest)
    sf.write(out_dir / "ref_in.wav", wav, sr, subtype="FLOAT")

    x24 = torch.from_numpy(wav)[None]  # [1, L_in]
    if sr != 24000:
        x24 = torchaudio.functional.resample(x24, sr, 24000)
    dump_raw(out_dir, "ref24k", x24[0].numpy(), manifest)
    sf.write(out_dir / "ref24k.wav", x24[0].numpy(), 24000, subtype="FLOAT")

    # --- resample 24k -> 16k (the step encode() does internally) ------------
    p = model.config
    x24c = x24.unsqueeze(1)  # [1, 1, L24]
    x16 = torchaudio.functional.resample(x24c, p.sample_rate, p.semantic_sample_rate)
    dump_raw(out_dir, "ref16k", x16[0, 0].numpy(), manifest)

    kernel, width = taf._get_sinc_resample_kernel(
        p.sample_rate, p.semantic_sample_rate,
        math.gcd(p.sample_rate, p.semantic_sample_rate),
        dtype=torch.float32, device=torch.device("cpu"),
    )
    dump_raw(out_dir, "rsk", kernel.numpy(), manifest)

    # --- semantic features (HuBERT) -----------------------------------------
    hid = x16[:, 0, :]                       # [1, L16]
    hid_pad = F.pad(hid, (160, 160))         # [1, L16+320]
    dump_raw(out_dir, "x16_pad", hid_pad[0].numpy(), manifest)
    with torch.no_grad():
        outs = model.semantic_model(hid_pad, output_hidden_states=True)
        hs = outs.hidden_states               # 13 x [1, T_hub, 768]
    print(f"hidden_states: {len(hs)} x {tuple(hs[0].shape)}")
    stacked = torch.stack([h.to(hid.device) for h in hs], dim=1)  # [1, 13, T, 768]
    sem_mean = stacked.mean(dim=1)
    sem_feat = sem_mean[:, :: p.semantic_downsample_factor, :]
    dump_raw(out_dir, "hub_hidden", stacked[0].numpy(), manifest)
    dump_raw(out_dir, "sem_mean", sem_mean[0].numpy(), manifest)
    dump_raw(out_dir, "sem_feat", sem_feat[0].numpy(), manifest)

    e_semantic = model.encoder_semantic(sem_feat.transpose(1, 2))
    dump_raw(out_dir, "e_semantic", e_semantic[0].detach().numpy(), manifest)

    # --- acoustic encoder (with the pad decision) ---------------------------
    ac_len = model._get_conv1d_output_lengths(x24c.shape[2], model.acoustic_encoder)
    pad_applied = ac_len != e_semantic.shape[2]
    print(f"acoustic no-pad {ac_len} vs semantic {e_semantic.shape[2]} -> pad={pad_applied}")
    manifest["acoustic_pad"] = {"value": bool(pad_applied)}
    if pad_applied:
        e_acoustic = model.acoustic_encoder(F.pad(x24c, (model.pad, model.pad)))
    else:
        e_acoustic = model.acoustic_encoder(x24c)
    dump_raw(out_dir, "e_acoustic", e_acoustic[0].detach().numpy(), manifest)

    # --- fc + RVQ ------------------------------------------------------------
    embeddings = torch.cat([e_acoustic, e_semantic], dim=1)
    emb_fc = model.fc(embeddings.transpose(1, 2)).transpose(1, 2)
    dump_raw(out_dir, "emb_fc", emb_fc[0].detach().numpy(), manifest)

    codes = model.quantizer.encode(emb_fc, p.target_bandwidths[-1]).transpose(0, 1)
    dump_raw(out_dir, "codes", codes[0].numpy(), manifest)

    # cross-check against the real encode() (both [1, 8, T])
    with torch.inference_mode():
        ref = model.encode(x24c, return_dict=False)
    same = torch.equal(ref, codes)
    print(f"manual vs encode(): codes equal = {same}")
    assert same, "manual replication diverged from model.encode()"

    # --- pos_conv weight-norm merge check ------------------------------------
    conv = model.semantic_model.encoder.pos_conv_embed.conv
    g = conv.parametrizations.weight.original0.detach()
    v = conv.parametrizations.weight.original1.detach()
    merged = conv.weight.detach()
    manual = g * v / v.norm(dim=tuple(d for d in range(v.dim()) if d != 2), keepdim=True)
    diff = (merged - manual).abs().max().item()
    print(f"pos_conv merge max diff = {diff:.3e}")
    dump_raw(out_dir, "pos_w", merged.numpy(), manifest)
    dump_raw(out_dir, "pos_g", g.numpy(), manifest)
    dump_raw(out_dir, "pos_v", v.numpy(), manifest)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest)} tensors to {out_dir}")


if __name__ == "__main__":
    main()
