# voxwaver

[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) 文本转语音（TTS）
本地推理：**Tauri 桌面应用（VoxWeaver）+ Rust 推理库（omnivoice）**，
基于 [huggingface/candle](https://github.com/huggingface/candle) fork，
支持 macOS Metal / CUDA 加速，完全离线运行。

## 构建

```bash
# 桌面应用（推荐入口）
cd app
pnpm install
pnpm tauri dev      # 开发
pnpm tauri build    # 打包 dmg / app（macOS 自动启用 Metal feature）

# 纯 CLI
cargo run -p omnivoice-cli --release -- --help
```

candle 依赖指向 fork 分支 [adamswanglin/candle `voxwaver`](https://github.com/adamswanglin/candle/tree/voxwaver)
（git 依赖，分支含：大 buffer 池精确分桶修复、直接式 Metal conv1d kernel、
MLX 移植的多块 argsort 接线），无需本地 checkout。

## 模型准备

在应用「设置 → 模型」中从 HuggingFace 下载，或手动获取后选择本地目录：

```bash
git clone https://huggingface.co/k2-fsa/OmniVoice
# 需要 model.safetensors、tokenizer.json、audio_tokenizer/ (~3.3 GB)
```

## 推理链路

两阶段流水线，对齐上游 vllm-omn PyTorch 实现：

1. **generator** — Qwen3 双向 backbone + 32 步迭代 unmasking，
   产出 8 codebook 音频 token（25 fps）
2. **dac** — HiggsAudioV2 RVQ + DAC 解码器 → 24 kHz 音频

参考音频（声音克隆）经 HuBERT 语义编码器 + 音频 tokenizer 编码为
token 缓存复用；风格指令（instruct）与语言标签直接传入生成。

## 桌面应用（VoxWeaver）

`app/` 基于 **Tauri v2 + React + Vite + TypeScript**，推理引擎直接内嵌为库
（`omnivoice` crate），无 sidecar 进程。

功能：

- **工作台**：文本编辑（字数 / 时长估计、导入 .txt）、声音选择、风格指令、
  采样参数（温度 / 种子，同种子完全可复现）、生成进度与中途取消
- **声音库**：zero-shot 克隆向导（定义人物 → 上传 WAV 样本 → 转写确认与编码），
  参考音频编码为 token 缓存后复用，生成时不再重复编码；卡片可直接试听
- **历史记录**：本地持久化（JSON 元数据 + wav），批量导出 MP3 / 删除 /
  在文件夹中显示
- **播放器**：真实波形播放条，seek / 快进快退
- **设置**：设备（auto/CPU/Metal/CUDA）切换、模型管理
  （HuggingFace 下载、导入本地目录、删除副本）、界面语言（13 种，
  同时作为合成语言标签）

桌面端数据（设置 / 声音 / 历史 / 音频 / 下载的模型）存放在系统应用数据目录：
`~/Library/Application Support/com.voxwaver.app/`（macOS）。

## 结构

| 文件 | 内容 |
| --- | --- |
| `crates/omnivoice/src/engine.rs` | 推理引擎：模型常驻、进度回调、取消、参考音频编码 |
| `crates/omnivoice/src/generator.rs` | Qwen3 backbone + 迭代 unmasking 采样 |
| `crates/omnivoice/src/qwen3.rs` | Qwen3 transformer 权重与前向 |
| `crates/omnivoice/src/dac.rs` | HiggsAudioV2 RVQ + DAC 解码器 |
| `crates/omnivoice/src/encoder.rs` | 克隆参考编码（HuBERT + 音频 tokenizer） |
| `crates/omnivoice/src/hubert.rs` | HuBERT 语义编码器 |
| `crates/omnivoice/src/duration.rs` | 时长估计（长文本切块） |
| `crates/omnivoice/src/bin/cli.rs` | `omnivoice-cli` 命令行 |
| `crates/tts-common/` | 取消标志、进度事件、WAV 读写与重采样 |
| `app/` | Tauri 桌面应用（Rust 后端 + React 前端） |

## 许可

本仓库代码以 Apache-2.0 发布；模型权重遵循其原始许可，
详见 [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)。
