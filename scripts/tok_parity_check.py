# Tokenizer parity: registered HF (CosyVoice3Tokenizer style) vs cosyvoice-cli encode.
# Usage: python scripts/tok_parity_check.py
import json, subprocess, sys

# 1. extract the additional_special_tokens list from CosyVoice3Tokenizer (source of truth)
src = open('/Users/wanglin/app/projects/github/CosyVoice/cosyvoice/tokenizer/tokenizer.py').read().splitlines()
starts = [i for i, l in enumerate(src) if 'additional_special_tokens' in l]
start = starts[-1]  # CosyVoice3Tokenizer block (last one)
body_lines = []
for l in src[start + 1:]:
    if l.strip() == '}':
        break
    body_lines.append(l)
body = '\n'.join(body_lines).strip()
assert body.endswith(']'), body[-50:]
toks = eval('[' + body)
print(f'extracted {len(toks)} additional special tokens', file=sys.stderr)

# 2. registered tokenizer = runtime ground truth
from transformers import AutoTokenizer
MD = '/Users/wanglin/app/projects/github/Fun-CosyVoice3-0.5B-2512'
tok = AutoTokenizer.from_pretrained(MD + '/CosyVoice-BlankEN')
tok.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>',
                        'additional_special_tokens': toks})
print(f'vocab: {len(tok)} after add_special_tokens', file=sys.stderr)

# 3. parity over diverse samples
samples = [
    'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
    '你好，世界！Hello, world! 123 456.',
    "It's a test -- really? Yes... [breath] OK.",
    '<|endofprompt|>',
    '<|im_start|>system<|im_end|>',
    ' CAPS lock AND lowercase äöü 你好',
    'multiple    spaces\tand\ttab',
    '3.14159 and 2/3 and 100%',
    '[AA][AE][AH] phonemes [zh]',
    'trailing space ',
    ' leading space',
    '<strong>emphasized</strong> [laughter] haha',
    'Mixed中文English混排 with<br>tags',
    'long: ' + 'word ' * 50,
    'a',
    '',
    '<|endofprompt|><|endofprompt|>',
    '数字混合文本2026年09月30日',
    "don't we'll I'm you're",
    'emoji test 🙂🎧 end',
]
fails = 0
for i, s in enumerate(samples):
    ids_ref = tok.encode(s)
    out = subprocess.run(
        ['cargo', 'run', '-q', '-p', 'cosyvoice', '--bin', 'cosyvoice-cli',
         '--features', 'metal', '--', 'encode', '--model-dir', MD, '--text', s],
        capture_output=True, text=True,
        cwd='/Users/wanglin/app/projects/github/voxwaver')
    so = out.stdout.strip()
    ids_rust = eval(so) if so else None
    ok = ids_rust == ids_ref
    if not ok:
        fails += 1
    print(f'[{i:2d}] {"OK  " if ok else "FAIL"} len={len(ids_ref):3d}/{len(ids_rust) if ids_rust is not None else -1:3d}  {s[:36]!r}')
    if not ok:
        print('   ref :', ids_ref)
        print('   rust:', ids_rust)
        if out.returncode != 0:
            print('   stderr:', out.stderr[-300:])
print('RESULT:', 'ALL PASS' if fails == 0 else f'{fails} FAILURES')
