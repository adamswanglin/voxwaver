<div align="center">

<img src="app/app-icon.png" width="160" alt="VoxWeaver" />

# voxwaver

基于 [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) 的本地离线文本转语音（TTS）桌面应用

安装即可直接使用 · 完全离线 · macOS Metal / NVIDIA CUDA 加速

[English](README.md) | **简体中文**

</div>

## 安装

从 [GitHub Releases](https://github.com/adamswanglin/voxwaver/releases) 下载对应平台的安装包，安装后即可直接使用：

| 平台 | 安装包 | 推理加速 |
| --- | --- | --- |
| macOS | `.dmg` | Metal |
| Windows | `.exe` 安装器 | CUDA（无 NVIDIA 显卡时自动回退 CPU） |
| Linux | `.AppImage` / `.deb` | CUDA（无 NVIDIA 显卡时自动回退 CPU） |

## 快速上手

1. 安装并打开应用，无需配置任何环境；
2. 首次使用在「设置 → 模型」一键下载 OmniVoice 模型（约 3.3 GB，仅需一次）。
   也可自己从 [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
   下载（git clone 需安装 [Git LFS](https://git-lfs.com)，否则音频
   tokenizer 权重是残缺的指针文件），然后在「设置 → 模型 → 选择本地
   模型文件夹」关联；文件夹须包含 `model.safetensors`、`tokenizer.json`
   和 `audio_tokenizer/model.safetensors`。
3. 在「工作台」输入文本、选择声音，点击生成；同一种子结果完全可复现；
4. 想复刻某个人的声音？在「声音库」按向导上传一段 WAV 样本，
   即可零样本（zero-shot）克隆。

全部推理在本机完成，断网可用。数据（设置 / 声音 / 历史 / 音频 / 下载的模型）
保存在系统应用数据目录：`~/Library/Application Support/com.voxwaver.app/`（macOS）。

## 功能

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

## 技术文档

构建方法、推理链路与代码结构见 [TECHNICAL.md](TECHNICAL.md)（英文）。

## 许可

本仓库代码以 [Apache-2.0](LICENSE) 发布；模型权重遵循其原始许可
（CC-BY-NC，受训练数据如 Emilia 的约束），详见
[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)。
