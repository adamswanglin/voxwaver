#!/usr/bin/env python3
"""Reference dumps for verifying the Rust port against the original PyTorch code.

Requires a fish-speech checkout (the reference implementation) and torch:

    pip install torch soundfile
    export FISH_SPEECH_ROOT=/path/to/fish-speech

Usage:
  # 1. DAC round-trip check: encode audio / decode codes, dump both to JSON
  python scripts/dump_reference.py dac --audio ref.wav --out /tmp/ref_dac.json

  # 2. Compare with the Rust implementation:
  #    voxwaver generate ... --dump-codes /tmp/rust_codes.json
  #    python scripts/dump_reference.py dac --codes /tmp/rust_codes.json \
  #        --out /tmp/ref_from_rust_codes.json
"""

import argparse
import json
import sys
from pathlib import Path

FISH_ROOT = Path(sys.argv[0]).resolve().parent.parent
FISH_SPEECH_ROOT = Path(__import__("os").environ.get(
    "FISH_SPEECH_ROOT", FISH_ROOT.parent / "fish-speech"))

sys.path.insert(0, str(FISH_SPEECH_ROOT))

import soundfile as sf
import torch

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def load_codec(ckpt: Path):
    from hydra.utils import instantiate

    with initialize_config_module(config_module="fish_speech.configs"):
        cfg = compose(config_name="modded_dac_vq.yaml")
    model = instantiate(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="../s1-mini")
    ap.add_argument("--audio")
    ap.add_argument("--codes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    model = load_codec(Path(args.model_dir) / "codec.pth")

    result = {}
    with torch.inference_mode():
        if args.audio:
            audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
            assert sr == args.sr, f"expected {args.sr} Hz, got {sr}"
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            codes, _lens = model.encode(audio_t)
            result["codes"] = codes.squeeze(0).tolist()
            print(f"encoded {audio_t.shape[-1]} samples -> {codes.shape[-1]} frames")
        if args.codes:
            with open(args.codes) as f:
                c = json.load(f)
            codes_t = torch.tensor(c, dtype=torch.long).unsqueeze(0)
            audio = model.from_indices(codes_t)
            result["audio"] = audio.squeeze().tolist()
            result["audio_len"] = len(result["audio"])
            print(f"decoded {codes_t.shape} -> {audio.shape[-1]} samples")
            sf.write(args.out + ".wav", audio.squeeze().numpy(), args.sr)

    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
