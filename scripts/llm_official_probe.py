# Official-code ground truth: build CosyVoice3LM from the CosyVoice repo,
# load llm.rl.pt, run the stock inference() loop, count tokens until stop.
# Usage: python3 scripts/llm_official_probe.py [seed]
import sys

sys.path.insert(0, '/Users/wanglin/app/projects/github/CosyVoice')
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from safetensors.torch import load_file

MD = '/Users/wanglin/app/projects/github/Fun-CosyVoice3-0.5B-2512'
BUNDLE = '/Users/wanglin/app/projects/github/voxwaver/prompts/anchor.safetensors'
PROMPT_TEXT = '希望你以后能够做的比我还好呦。<|endofprompt|>'
TEXT = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。'

from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
from cosyvoice.utils.common import ras_sampling

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(seed)

cfg = Qwen2Config(hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                  num_attention_heads=14, num_key_value_heads=2, rms_norm_eps=1e-6,
                  rope_theta=1_000_000, vocab_size=151936, max_position_embeddings=32768)
enc = Qwen2Encoder.__new__(Qwen2Encoder)
torch.nn.Module.__init__(enc)  # bypass from_pretrained('') in __init__
enc.model = Qwen2ForCausalLM(cfg)
lm = CosyVoice3LM(896, 896, 6561, enc, ras_sampling, mix_ratio=[5, 15])
sd = torch.load(MD + '/llm.rl.pt', map_location='cpu', weights_only=True)
missing, unexpected = lm.load_state_dict(sd, strict=False)
print('missing:', [k for k in missing][:5], 'unexpected:', [k for k in unexpected][:5])
lm.eval()

if len(sys.argv) > 2 and sys.argv[2] == 'greedy':
    lm.sampling_ids = lambda weighted_scores, decoded_tokens, sampling, ignore_eos=True: int(weighted_scores.argmax())

# tokenizer identical to the runtime ground truth
src = open('/Users/wanglin/app/projects/github/CosyVoice/cosyvoice/tokenizer/tokenizer.py').read().splitlines()
starts = [i for i, l in enumerate(src) if 'additional_special_tokens' in l]
body_lines = []
for l in src[starts[-1] + 1:]:
    if l.strip() == '}':
        break
    body_lines.append(l)
toks = eval('[' + '\n'.join(body_lines).strip())
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MD + '/CosyVoice-BlankEN')
tok.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>',
                        'additional_special_tokens': toks})

p_ids = tok.encode(PROMPT_TEXT)
t_ids = tok.encode(TEXT)
speech_token = load_file(BUNDLE)['speech_token'].long()
with torch.no_grad():
    out = list(lm.inference(
        torch.tensor([t_ids]), torch.tensor([len(t_ids)]),
        torch.tensor([p_ids]), torch.tensor([len(p_ids)]),
        speech_token.unsqueeze(0), torch.tensor([speech_token.numel()]),
        None, sampling=25))
print(f'seed={seed}{" GREEDY" if len(sys.argv) > 2 else ""} tokens={len(out)}')
print('first20:', out[:20])
print('last20:', out[-20:])
import collections
c = collections.Counter(out)
print('top repeats:', c.most_common(6))
