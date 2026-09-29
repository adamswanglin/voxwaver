# voxwaver

fishaudio/s1-mini 文本转语音（TTS）推理的纯 Rust 实现，基于
[huggingface/candle](https://github.com/huggingface/candle) 0.11。
仅命令行，支持 CUDA / macOS Metal 加速，依赖保持最小。

实现了完整推理链路：

- **Dual-AR 语言模型**（`model.pth`）：慢速 AR transformer 逐帧预测 semantic token，
  快速 transformer（共享 hidden state 的小 transformer）预测其余 9 个 codebook。
  完整移植了 s1-mini 推理细节：GQA、qk-norm、interleaved RoPE（bf16 圆整的
  cos/sin 表）、VQ embedding 注入（`/sqrt(num_codebooks+1)`）。
- **提示格式**：`ContentSequence(modality="interleave")` 布局——
  `<|interleave|><|speaker:0|>{REF_TEXT}{semantic tokens}<|im_end|><|speaker:0|>{TEXT}`
  （注意：`<|speaker:0|>` 不是特殊 token，按普通 BPE 编码，与官方一致）。
  文本过 `clean_text`（trim、弯引号映射、去 emoji、连续逗号折叠）。
- **采样**：官方服务默认参数 temperature 0.7 / top-p 0.7 / repetition penalty 1.5
  （每 codebook 行独立、最近 16 token 滑窗、正负 logit 不对称惩罚），
  top-p 累积掩码在未除温度的 softmax 上计算，Gumbel-max 多项采样。
- **tokenizer**：tiktoken BPE（**FISH_TIKTOKEN_PATTERN**，非标准 GPT-4o 模式：
  标点 `\p{P}` 单独成词、数字逐位切分）+ fish 特殊 token 的最小实现
  （与 tiktoken `allowed_special="all"` 逐 token 对齐，含回归测试）。
- **modded-DAC codec**（`codec.pth`）：44.1 kHz、帧长 2048、10 本 RVQ
  （semantic 4096 + 9×1024）。包含编码（voice cloning 参考）与解码（wav 输出），
  权重加载时自动折叠 weight-norm（新旧两种 parametrization 均支持），
  解码支持分块（128 帧左上下文，与整段解码等价）。

## 构建

```bash
# macOS (Metal + Accelerate)
cargo build --release --features metal,accelerate

# CUDA
cargo build --release --features cuda,cudnn

# CPU
cargo build --release
```

默认通过 path 依赖使用本地 candle checkout（`/Users/wanglin/app/projects/github/candle`），
换机器请把 `Cargo.toml` 中改为 crates.io 版本 `candle-core = "0.11.0"`。

## 模型准备

```bash
git clone https://huggingface.co/fishaudio/s1-mini
cd s1-mini && git lfs pull   # model.pth + codec.pth (~3.6 GB)
```

## 使用

```bash
# 基本合成（zero-shot）
voxwaver generate --model-dir /path/to/s1-mini \
    --text "你好，世界。" --out out.wav

# 声音克隆（参考音频 + 文本）
voxwaver generate --model-dir /path/to/s1-mini \
    --text "今天天气不错。" \
    --ref-audio ref.wav --ref-text "参考音频里说的话" \
    --out cloned.wav

# 常用参数
#   --temperature 0.7 --top-p 0.7 --repetition-penalty 1.5  # 官方默认采样参数
#   --seed N              # 固定采样种子
#   --max-new-tokens N    # 最多生成帧数（每帧 ~46ms 音频）
#   --device auto|cpu|cuda|metal --dtype auto|bf16|f16|f32
#   --chunk-frames N      # codec 解码分块（0 = 整段一次）
#   --pcm16               # 输出 16-bit PCM 而非 float32 wav
#   --dump-codes c.json   # 导出生成的 codebook 行（与 Python 对拍用）
#   --print-prompt        # 打印提示 token 布局（调试）

# 只跑 codec：codes JSON -> wav
voxwaver decode --model-dir /path/to/s1-mini --codes c.json --out out.wav

# 查看权重键名（调试）
voxwaver keys /path/to/s1-mini/model.pth
```

参考音频支持任意常见采样率的 WAV（内部 Kaiser 窗 sinc 重采样到 44.1 kHz，
自动混为单声道）。

## 结构

| 文件 | 内容 |
| --- | --- |
| `src/tokenizer.rs` | tiktoken BPE（fish pattern）+ 特殊 token |
| `src/prompt.rs` | interleave 提示模板 + clean_text（含参考音频内联 semantic token） |
| `src/dual_ar.rs` | Dual-AR transformer + KV cache + 快速帧解码 |
| `src/sampling.rs` | top-p + 窗口化 repetition penalty 采样（CPU 参考实现 + 全 GPU 链） |
| `src/dac/` | codec（layers / transformer / rvq / 加载与编排） |
| `src/wavio.rs` | WAV 读写 + 重采样 |
| `src/config.rs` | config.json 与 codec 超参 |
| `scripts/dump_reference.py` | 用 fish-speech/PyTorch 生成参考输出做数值对拍 |

## 验证状态

- tokenizer 与 tiktoken（FISH_TIKTOKEN_PATTERN）逐 token 一致；
  prompt 布局与官方 `ContentSequence` 编码逐 token 一致（`cargo test`）。
- LM 前向 logits 与 PyTorch 逐元素一致（4 位小数）。
- DAC 编码与 PyTorch 100% 一致（10 本 codebook 全对）；
  DAC 解码输出与 PyTorch 相关系数 1.0000（曾因 upsample 阶段加载顺序错误而损坏，
  已修复并验证）。
- 端到端（文本 → LM 采样 → codec 解码 → wav）在 macOS Metal 上验证：
  参考音频 3.8 s / 提示 118 token / 生成 59 帧（2.7 s 音频），
  与 PyTorch 官方推理路径的生成帧数一致。

## 许可

模型权重与 fish-speech 代码遵循其原始许可（CC-BY-NC-SA-4.0，
仅限非商业用途）；本仓库代码同样以 CC-BY-NC-SA-4.0 发布。
