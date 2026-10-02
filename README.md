<div align="center">

<img src="app/app-icon.png" width="160" alt="VoxWeaver" />

# voxwaver

A local, offline text-to-speech (TTS) desktop app powered by [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)

Install and use right away · Fully offline · macOS Metal / NVIDIA CUDA acceleration

**English** | [简体中文](README.zh-CN.md)

</div>

## Installation

Grab the installer for your platform from
[GitHub Releases](https://github.com/adamswanglin/voxwaver/releases) — the app
works out of the box after installation:

| Platform | Installer | Acceleration |
| --- | --- | --- |
| macOS | `.dmg` | Metal |
| Windows | `.exe` installer | CUDA (automatic CPU fallback without an NVIDIA GPU) |
| Linux | `.AppImage` / `.deb` | CUDA (automatic CPU fallback without an NVIDIA GPU) |

## Getting Started

1. Install and open the app — no environment setup required.
2. On first use, download the OmniVoice model in **Settings → Models**
   (~3.3 GB, one time only).
   Prefer downloading it yourself? Fetch the model from
   [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (when cloning
   the repo, [Git LFS](https://git-lfs.com) is required for the audio
   tokenizer weights), then link it in **Settings → Models → Select local
   model folder**. The folder must contain `model.safetensors`,
   `tokenizer.json` and `audio_tokenizer/model.safetensors`.
3. Go to the **Workspace**, type your text, pick a voice and generate.
   The same seed always produces identical output.
4. Want to replicate someone's voice? Follow the wizard in the
   **Voice Library**: upload a WAV sample and clone it zero-shot.

All inference runs on your machine — the app works without a network
connection. Your data (settings / voices / history / audio / downloaded
models) is stored in the system app-data directory, e.g.
`~/Library/Application Support/com.voxwaver.app/` on macOS.

## Features

- **Workspace** — text editing (word count / duration estimate, .txt import),
  voice selection, style instructions, sampling parameters
  (temperature / seed, fully reproducible with the same seed), generation
  progress with mid-run cancel
- **Voice Library** — zero-shot cloning wizard
  (define a person → upload a WAV sample → confirm transcription & encode).
  Reference audio is encoded once into a token cache and reused for every
  generation; voice cards offer instant preview playback
- **History** — local persistence (JSON metadata + wav), batch MP3 export /
  delete / reveal in folder
- **Player** — real-waveform playback bar with seek / fast-forward / rewind
- **Settings** — device switching (auto/CPU/Metal/CUDA), model management
  (HuggingFace download, import local directory, delete copies), UI language
  (13 languages, also used as the synthesis language tag)

## Documentation

Build instructions, the inference pipeline and the codebase layout are
covered in [TECHNICAL.md](TECHNICAL.md).

## License

The code in this repository is released under [Apache-2.0](LICENSE); model
weights follow their original license (CC-BY-NC, constrained by training
data such as Emilia), see
[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).
