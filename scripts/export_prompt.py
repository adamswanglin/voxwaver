#!/usr/bin/env python
"""Export a zero-shot prompt bundle for `cosyvoice-cli generate`.

From a prompt wav it extracts (mirroring `cosyvoice/cli/frontend.py`):
  - speech_token  I32 [N]      whisper log-mel(128) @16 kHz -> speech_tokenizer_v3.onnx
  - spk_embedding F32 [1,192]  kaldi fbank(80) @16 kHz -> campplus.onnx
  - feat          F32 [1,T,80] matcha mel_spectrogram @24 kHz
then applies the reference 24 kHz alignment (feat = 2 * tokens) and saves a
safetensors bundle loadable by `cosyvoice::engine::PromptBundle::load`.

Usage:
  python scripts/export_prompt.py --model-dir <Fun-CosyVoice3-0.5B-2512> \
      --wav asset/zero_shot_prompt.wav --out prompts/anchor.safetensors
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from safetensors.torch import save_file


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    speech, sr = torchaudio.load(path, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)
    return speech


# ---------------------------------------------------------------- whisper log-mel
# openai-whisper audio.py `log_mel_spectrogram`, hand-rolled (only dep: librosa).
_N_FFT, _HOP, _SR = 400, 160, 16000
_mel128 = None


def whisper_logmel128(speech: torch.Tensor) -> torch.Tensor:
    global _mel128
    if _mel128 is None:
        import librosa
        _mel128 = torch.from_numpy(
            librosa.filters.mel(sr=_SR, n_fft=_N_FFT, n_mels=128)).float()
    window = torch.hann_window(_N_FFT)
    stft = torch.stft(speech, _N_FFT, _HOP, window=window, return_complex=True)
    mag = stft[..., :-1].abs() ** 2
    log_spec = torch.clamp(_mel128 @ mag, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


# ------------------------------------------------------------------- matcha mel
# matcha.utils.audio.mel_spectrogram (the `feat_extractor` of cosyvoice3.yaml).
_MEL80 = None


def matcha_mel(speech: torch.Tensor, sr: int = 24000, n_fft: int = 1920,
               num_mels: int = 80, hop: int = 480, win: int = 1920,
               fmin: float = 0.0, fmax=None) -> torch.Tensor:
    global _MEL80
    if _MEL80 is None:
        import librosa
        _MEL80 = torch.from_numpy(
            librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mels,
                                fmin=fmin, fmax=fmax)).float()
    y = torch.nn.functional.pad(
        speech.unsqueeze(1), ((n_fft - hop) // 2, (n_fft - hop) // 2),
        mode='reflect').squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop,
                      window=torch.hann_window(win), center=False,
                      pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-8)  # torch>=2.1 eps
    mel_spec = _MEL80 @ spec  # [1, 80, F]
    return torch.log(torch.clamp(mel_spec, min=1e-5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True,
                    help='model dir holding campplus.onnx + speech_tokenizer_v3.onnx')
    ap.add_argument('--wav', required=True, help='prompt wav (any sr, mono-mixed)')
    ap.add_argument('--out', required=True, help='output bundle .safetensors')
    args = ap.parse_args()

    campplus_path = os.path.join(args.model_dir, 'campplus.onnx')
    st_path = os.path.join(args.model_dir, 'speech_tokenizer_v3.onnx')

    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 4
    campplus = ort.InferenceSession(campplus_path, sess_options=opt,
                                    providers=['CPUExecutionProvider'])
    st_tok = ort.InferenceSession(st_path, sess_options=opt,
                                  providers=['CPUExecutionProvider'])

    # speech tokens: whisper log-mel(128) @16 kHz
    speech_16k = load_wav(args.wav, 16000)
    assert speech_16k.shape[1] / 16000 <= 30, 'prompt wav must be <= 30 s'
    feat = whisper_logmel128(speech_16k)  # [128, T]
    token = st_tok.run(None, {
        st_tok.get_inputs()[0].name: feat.numpy(),
        st_tok.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
    })[0].flatten().astype(np.int32)

    # speaker embedding: kaldi fbank(80) @16 kHz, mean-removed
    fbank = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0,
                        sample_frequency=16000)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    spk = campplus.run(None, {
        campplus.get_inputs()[0].name: fbank.unsqueeze(0).numpy()
    })[0].flatten().astype(np.float32)

    # log-mel feat @24 kHz, then the reference alignment: tokens = feat frames / 2
    speech_24k = load_wav(args.wav, 24000)
    mel = matcha_mel(speech_24k).squeeze(0).T.unsqueeze(0)  # [1, T, 80]
    token_len = min(mel.shape[1] // 2, token.shape[0])
    token = token[:token_len]
    mel = mel[:, :2 * token_len, :]

    save_file({
        'speech_token': torch.from_numpy(token),
        'spk_embedding': torch.from_numpy(spk).unsqueeze(0),
        'feat': mel.contiguous(),
    }, args.out)
    print(f'{args.out}: speech_token {token.shape} spk_embedding [1,{spk.shape[0]}] '
          f'feat [1,{mel.shape[1]},{mel.shape[2]}]')


if __name__ == '__main__':
    main()
