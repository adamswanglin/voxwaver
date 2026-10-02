# Technical Notes

Developer-facing documentation for voxwaver: building from source, the
inference pipeline, and the codebase layout. For installation and usage,
see [README.md](README.md) / [README.zh-CN.md](README.zh-CN.md).

## Build

```bash
# Desktop app (recommended entry point)
cd app
pnpm install
pnpm tauri dev      # development
pnpm tauri build    # bundle dmg / app (Metal feature enabled automatically on macOS)

# Plain CLI
cargo run -p omnivoice-cli --release -- --help
```

The candle dependency points to the fork branch
[adamswanglin/candle `voxwaver`](https://github.com/adamswanglin/candle/tree/voxwaver)
(a git dependency; the branch contains: precise bucketing fix for the large
buffer pool, a direct Metal conv1d kernel, and the MLX-ported multi-block
argsort wiring) — no local checkout required.

## Manual Model Setup

Models can be downloaded in one click from within the app (see the README);
alternatively fetch them manually and point the app at the local directory:

```bash
git lfs install
git clone https://huggingface.co/k2-fsa/OmniVoice
# ~3.3 GB; loading requires model.safetensors, tokenizer.json and
# audio_tokenizer/model.safetensors (LFS). The download_files list also
# includes config.json, tokenizer_config.json, chat_template.jinja,
# audio_tokenizer/config.json and audio_tokenizer/preprocessor_config.json.
```

The app validates a local folder against the three required files above
(`models::check_model_dir`); the remaining files are fetched by the in-app
downloader but are not required when importing a local copy.

## Inference Pipeline

A two-stage pipeline aligned with the upstream vllm-omn PyTorch implementation:

1. **generator** — bidirectional Qwen3 backbone + 32-step iterative unmasking,
   producing 8-codebook audio tokens (25 fps)
2. **dac** — HiggsAudioV2 RVQ + DAC decoder → 24 kHz audio

Reference audio (voice cloning) is encoded by the HuBERT semantic encoder +
the audio tokenizer into a reusable token cache; style instructions
(instruct) and language tags are fed directly into generation.

The engine is implemented in Rust (the `omnivoice` crate), on top of a
[huggingface/candle](https://github.com/huggingface/candle) fork, and
embedded directly into the **Tauri v2 + React + Vite + TypeScript** desktop
app — no sidecar process.

## Codebase Layout

| File | Contents |
| --- | --- |
| `crates/omnivoice/src/engine.rs` | Inference engine: resident model, progress callbacks, cancellation, reference-audio encoding, long-text chunking and post-processing |
| `crates/omnivoice/src/generator.rs` | Qwen3 backbone + iterative unmasking sampling |
| `crates/omnivoice/src/qwen3.rs` | Qwen3 transformer weights and forward pass |
| `crates/omnivoice/src/dac.rs` | HiggsAudioV2 RVQ + DAC decoder |
| `crates/omnivoice/src/encoder.rs` | Cloning reference encoding (HuBERT + audio tokenizer) |
| `crates/omnivoice/src/hubert.rs` | HuBERT semantic encoder |
| `crates/omnivoice/src/duration.rs` | Duration estimation (reference calibration + speed) |
| `crates/omnivoice/src/text.rs` | Long-text splitting (punctuation-based sentence split, abbreviation protection, min-chunk merging) |
| `crates/omnivoice/src/bin/cli.rs` | `omnivoice-cli` command line |
| `crates/tts-common/` | Cancel flags, progress events, WAV I/O and resampling |
| `app/` | Tauri desktop app (Rust backend + React frontend) |
