# Torch parity probe for the LLM prefill: identical inputs to the Rust engine
# (sos/text/task/prompt_speech embedding concat), hand-rolled Qwen2 forward.
# Compare the printed top8 with COSYVOICE_DEBUG step 0 from cosyvoice-cli.
# Usage: python3 scripts/llm_torch_probe.py [--steps N]
import argparse, math, sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

MD = '/Users/wanglin/app/projects/github/Fun-CosyVoice3-0.5B-2512'
BUNDLE = '/Users/wanglin/app/projects/github/voxwaver/prompts/anchor.safetensors'
PROMPT_TEXT = '希望你以后能够做的比我还好呦。<|endofprompt|>'
TEXT = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。'
SOS, TASK_ID, STS = 6561, 6563, 6561
THETA, EPS, HEADS, KV, HD = 1_000_000.0, 1e-6, 14, 2, 64

# tokenizer (same registration as the runtime ground truth)
src = open('/Users/wanglin/app/projects/github/CosyVoice/cosyvoice/tokenizer/tokenizer.py').read().splitlines()
starts = [i for i, l in enumerate(src) if 'additional_special_tokens' in l]
body_lines = []
for l in src[starts[-1] + 1:]:
    if l.strip() == '}':
        break
    body_lines.append(l)
toks = eval('[' + '\n'.join(body_lines).strip())
tok = AutoTokenizer.from_pretrained(MD + '/CosyVoice-BlankEN')
tok.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>',
                        'additional_special_tokens': toks})

# weights
sd = torch.load(MD + '/llm.rl.pt', map_location='cpu', weights_only=True)
print('dtypes:', {str(v.dtype) for v in sd.values()}, file=sys.stderr)
emb = sd['llm.model.model.embed_tokens.weight'].float()
spk_emb_all = sd['speech_embedding.weight'].float()
dec = sd['llm_decoder.weight'].float()
norm_w = sd['llm.model.model.norm.weight'].float()
L = 24

# inputs, identical to engine.tts
prompt_ids = tok.encode(PROMPT_TEXT)
text_ids = tok.encode(TEXT)
full = prompt_ids + text_ids
speech_token = load_file(BUNDLE)['speech_token'].long()
print(f'prompt_ids={len(prompt_ids)} text_ids={len(text_ids)} prompt_speech={speech_token.numel()}', file=sys.stderr)

def rope_cache(t, pos0):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
    f = torch.arange(pos0, pos0 + t).float()[:, None] * inv[None]  # [t, 32]
    e = torch.cat([f, f], dim=-1)
    return e.cos(), e.sin()

def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

def forward(x, cache_kv, pos0):
    t = x.shape[1]
    cos, sin = rope_cache(t, pos0)
    # q at absolute positions pos0..pos0+t may attend kv 0..pos0+t
    qpos = torch.arange(pos0, pos0 + t)[:, None]
    kvpos = torch.arange(pos0 + t)[None, :]
    mask = torch.where(qpos < kvpos, float('-inf'), 0.0)
    new_kv = []
    for i in range(L):
        p = f'llm.model.model.layers.{i}.'
        h = rms(x, sd[p + 'input_layernorm.weight'].float())
        q = (h @ sd[p + 'self_attn.q_proj.weight'].float().T + sd[p + 'self_attn.q_proj.bias'].float())
        k = (h @ sd[p + 'self_attn.k_proj.weight'].float().T + sd[p + 'self_attn.k_proj.bias'].float())
        v = (h @ sd[p + 'self_attn.v_proj.weight'].float().T + sd[p + 'self_attn.v_proj.bias'].float())
        q = q.view(1, t, HEADS, HD).transpose(1, 2)
        k = k.view(1, t, KV, HD).transpose(1, 2)
        v = v.view(1, t, KV, HD).transpose(1, 2)
        q = q * cos * 1.0 + rotate_half(q) * sin
        k = k * cos * 1.0 + rotate_half(k) * sin
        new_kv.append((k, v))
        if cache_kv is not None:
            k = torch.cat([cache_kv[i][0], k], dim=2)
            v = torch.cat([cache_kv[i][1], v], dim=2)
        k = k.repeat_interleave(HEADS // KV, dim=1)
        v = v.repeat_interleave(HEADS // KV, dim=1)
        att = q @ k.transpose(2, 3) / HD ** 0.5 + mask
        o = att.softmax(-1) @ v
        o = o.transpose(1, 2).reshape(1, t, -1) @ sd[p + 'self_attn.o_proj.weight'].float().T
        x = x + o
        h = rms(x, sd[p + 'post_attention_layernorm.weight'].float())
        g = h @ sd[p + 'mlp.gate_proj.weight'].float().T
        u = h @ sd[p + 'mlp.up_proj.weight'].float().T
        x = x + (torch.nn.functional.silu(g) * u) @ sd[p + 'mlp.down_proj.weight'].float().T
    return rms(x, norm_w), new_kv

sos = spk_emb_all[SOS].view(1, 1, -1)
task = spk_emb_all[TASK_ID].view(1, 1, -1)
text_e = emb[torch.tensor(full)].unsqueeze(0)
prompt_e = spk_emb_all[speech_token].unsqueeze(0)
lm_input = torch.cat([sos, text_e, task, prompt_e], dim=1)
print('lm_input', tuple(lm_input.shape), file=sys.stderr)

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=6)
args = ap.parse_args()

kv, x, pos0 = None, lm_input, 0
for step in range(args.steps):
    x, new_kv = forward(x, kv, pos0)
    kv = [(kv[i][0], kv[i][1]) for i in range(L)] if kv else None
    kv = [(torch.cat([new_kv[i][0]], 0), torch.cat([new_kv[i][1]], 0)) if kv is None else
          (torch.cat([kv[i][0], new_kv[i][0]], dim=2), torch.cat([kv[i][1], new_kv[i][1]], dim=2))
          for i in range(L)]
    pos0 += x.shape[1]
    logits = x[:, -1] @ dec.T
    logp = logits.log_softmax(-1)
    top = logp.topk(8)
    tops = ' '.join(f'{i}:{math.exp(p):.4f}' for i, p in zip(top.indices[0].tolist(), top.values[0].tolist()))
    print(f'[torch] step {step} top8: {tops}')
    if step == 0:
        k0 = kv[0][0]
        print('[torch] cache k[0] 2f8:', k0[0, 0, :, :4].flatten().tolist())
        print('[torch] hidden pos0 6:', x[0, 0, :6].tolist())
    x = spk_emb_all[top.indices[0][0]].view(1, 1, -1)  # greedy for comparison
