"""Per-layer HuBERT dumps for debugging the Rust port.

Runs the HF HuBERT forward on the padded 16 kHz waveform and dumps the
intermediate outputs of the feature extractor (selected conv layers), the
feature projection, the positional conv, the encoder layer norm, and selected
encoder layers, plus the official hidden_states[0/6/12] for semantic
comparison.

    /Users/wanglin/omni-ref-venv-arm64/bin/python scripts/omni_hubert_dump.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel


def save(dumps, name, t):
    if isinstance(t, (tuple, list)):
        t = t[0]
    arr = t.detach().float().cpu().numpy().astype(np.float32)
    dumps[name] = arr
    print(f"  {name:14s} {tuple(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="../OmniVoice")
    ap.add_argument("--ref-dir", default="/tmp/omni_clone_ref")
    args = ap.parse_args()

    out = Path(args.ref_dir)
    model = HiggsAudioV2TokenizerModel.from_pretrained(
        Path(args.model_dir) / "audio_tokenizer"
    ).eval()
    sm = model.semantic_model

    x16p = np.fromfile(out / "x16_pad.f32", dtype=np.float32)
    x = torch.from_numpy(x16p)[None]
    print(f"input {tuple(x.shape)}")

    dumps = {}
    hooks = []

    def hook(name):
        def fn(m, i, o):
            save(dumps, name, o)
        return fn

    for i in (0, 3, 6):
        hooks.append(
            sm.feature_extractor.conv_layers[i].register_forward_hook(hook(f"conv{i}"))
        )
    hooks.append(sm.feature_projection.register_forward_hook(hook("feat_proj")))
    hooks.append(sm.encoder.pos_conv_embed.register_forward_hook(hook("pos_conv")))
    hooks.append(sm.encoder.layer_norm.register_forward_hook(hook("enc_ln")))
    for i in (0, 5, 11):
        hooks.append(sm.encoder.layers[i].register_forward_hook(hook(f"layer{i}")))

    with torch.no_grad():
        outs = sm(x, output_hidden_states=True)
    hs = outs.hidden_states
    print(f"official hidden_states: {len(hs)} x {tuple(hs[0].shape)}")
    for i in (0, 6, 12):
        save(dumps, f"official_hs{i}", hs[i])

    for h in hooks:
        h.remove()

    # Cross-checks: which intermediate equals hidden_states[0]?
    for name in ("pos_conv", "enc_ln"):
        if name in dumps:
            a = dumps[name].reshape(-1)
            b = dumps["official_hs0"].reshape(-1)
            print(f"hs0 vs {name}: maxdiff {np.abs(a - b).max():.3e}")

    for name, arr in dumps.items():
        arr.tofile(out / f"hub_{name}.f32")
    print(f"wrote {len(dumps)} tensors to {out}")


if __name__ == "__main__":
    main()
