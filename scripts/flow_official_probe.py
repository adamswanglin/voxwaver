# Official-code ground truth for the flow module: build CausalMaskedDiffWithDiT,
# load flow.pt, synthesize mel from fixed tokens, dump stats + CFM noise buffer.
# Usage: python3 scripts/flow_official_probe.py
import sys

CV = '/Users/wanglin/app/projects/github/CosyVoice'
sys.path.insert(0, CV)
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

# The Matcha-TTS submodule is not checked out; stub the only symbol that
# cosyvoice.flow.flow_matching imports (BASECFM, as in the upstream source).
import types


class BASECFM(torch.nn.Module):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=64, estimator=None):
        super().__init__()
        self.n_feats = n_feats
        self.cfm_params = cfm_params
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.estimator = estimator
        self.sigma_min = cfm_params.sigma_min


_m = types.ModuleType('matcha')
_m_models = types.ModuleType('matcha.models')
_m_comp = types.ModuleType('matcha.models.components')
_m_fm = types.ModuleType('matcha.models.components.flow_matching')
_m_fm.BASECFM = BASECFM
_m_comp.flow_matching = _m_fm
_m_models.components = _m_comp
_m.models = _m_models
for name, mod in [('matcha', _m), ('matcha.models', _m_models),
                  ('matcha.models.components', _m_comp),
                  ('matcha.models.components.flow_matching', _m_fm)]:
    sys.modules[name] = mod

from cosyvoice.flow.DiT.dit import DiT
from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.flow.flow_matching import CausalConditionalCFM
from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

MD = '/Users/wanglin/app/projects/github/Fun-CosyVoice3-0.5B-2512'

cfm_params = OmegaConf.create({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                               'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7,
                               'reg_loss_type': 'l1'})
dit = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80,
          spk_dim=80, out_channels=80, static_chunk_size=50, num_decoding_left_chunks=-1)
dec = CausalConditionalCFM(in_channels=240, cfm_params=cfm_params, n_spks=1, spk_emb_dim=80, estimator=dit)
pal = PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
model = CausalMaskedDiffWithDiT(input_size=80, output_size=80, spk_embed_dim=192, vocab_size=6561,
                                input_frame_rate=25, token_mel_ratio=2, pre_lookahead_len=3,
                                pre_lookahead_layer=pal, decoder=dec)
sd = torch.load(MD + '/flow.pt', map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing:', [k for k in missing][:6], 'unexpected:', [k for k in unexpected][:6])
model.eval()

b = load_file('/Users/wanglin/app/projects/github/voxwaver/prompts/anchor.safetensors')
prompt_token = b['speech_token'].long()[None]        # [1, 87]
prompt_feat = b['feat']                              # [1, 174, 80]
embedding = b['spk_embedding']                       # [1, 192]

torch.manual_seed(42)
token = torch.randint(0, 6561, size=(1, 25))
token_len = torch.tensor([25])
prompt_token_len = torch.tensor([prompt_token.shape[1]])
prompt_feat_len = torch.tensor([prompt_feat.shape[1]])
print('tokens:', token.flatten().tolist())

with torch.inference_mode():
    # dump the encoder-side intermediates (same names as the Rust side)
    import torch.nn.functional as F
    emb = F.normalize(embedding, dim=1)
    emb = model.spk_embed_affine_layer(emb)
    emb.flatten().numpy().tofile('dump_flowspk.f32')
    tok_all = torch.concat([prompt_token, token], dim=1)
    tok = model.input_embedding(torch.clamp(tok_all, min=0))  # no padding -> mask is all ones
    tok.flatten().numpy().tofile('dump_flowtok.f32')
    h = model.pre_lookahead_layer(tok)
    h = h.repeat_interleave(model.token_mel_ratio, dim=1)
    h.flatten().numpy().tofile('dump_flowmu.f32')
    feat, _ = model.inference(token, token_len, prompt_token, prompt_token_len,
                              prompt_feat, prompt_feat_len, embedding,
                              streaming=True, finalize=True)
feat = feat.squeeze(0)  # [80, mel_len2]
feat.numpy().tofile('dump_mel.f32')
# CFM noise buffer (seed 0, deterministic) for Rust parity.
model.decoder.rand_noise.numpy().tofile('dump_cfmnoise.f32')
rms = float(np.sqrt((feat.numpy()**2).mean()))
print(f'mel: shape={tuple(feat.shape)} rms={rms:.4f} min={feat.min():.3f} max={feat.max():.3f}')
print('mel head (bin0 first8):', feat[0, :8].tolist())
