"""Official Kimodo two-stage blocks, adapted to the existing 266D protocol.

This is NOT the native Kimodo representation or a reproduction of its recipe.
Root output = existing local root4 + explicit pelvis XYZ3. Body output = the
remaining 259 features. The body stage consumes the predicted local root4;
no unknown ground-truth root, velocity, rotation or heading is supplied.
"""
from types import SimpleNamespace
import torch
from torch import nn
from .kimodo_components import official_twostage


class RootBody266:
    motion_rep_dim = 266
    global_root_dim = 7  # Interface name from upstream; root4 + global pelvis3.
    local_root_dim = 4
    body_slice = slice(7, 266)
    skeleton = SimpleNamespace(nbjoints=22)

    def global_root_to_local_root(self, root_features, normalized, lengths):
        if not normalized:
            raise ValueError('Adapter expects the existing normalized 266D features')
        # These four predicted channels ALREADY encode local root motion.
        # Their training-time detach is performed by the official forward.
        return root_features[..., :4]


class Kimodo266(nn.Module):
    def __init__(self, width=1024, depth=16, heads=8, ffn_dim=2048,
                 register_tokens=50, sampling_steps=32):
        super().__init__()
        if sampling_steps < 1 or sampling_steps > 1000:
            raise ValueError('Invalid sampling lattice')
        self.sampling_steps = sampling_steps
        self.register_tokens = register_tokens
        self.backbone = official_twostage()(
            motion_rep=RootBody266(), motion_mask_mode='concat',
            llm_shape=[1, 4096], use_text_mask=False,
            latent_dim=width, ff_size=ffn_dim, num_layers=depth,
            num_heads=heads, activation='gelu', dropout=0., pe_dropout=0.,
            norm_first=False, num_text_tokens_override=register_tokens,
            input_first_heading_angle=False)

    def forward(self, x, sigma, known, text, text_valid, valid, frame_text_allowed=None):
        if any(v is not None for v in [text, text_valid, frame_text_allowed]):
            raise ValueError('This control-only ablation does not accept text/GT heading')
        if x.ndim != 3 or x.shape[-1] != 266 or sigma.shape != x.shape or known.shape != x.shape:
            raise ValueError('Expected matching B,T,266 tensors')
        if known.dtype != torch.bool or valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError('Invalid masks')
        if not valid.any(-1).all() or (sigma[known] != 0).any():
            raise ValueError('Invalid validity mask or noisy anchor')
        live = valid[..., None].expand_as(x)
        unknown = live & ~known
        # Kimodo has one time prefix token per sequence. Reject joint-time
        # fields rather than silently averaging them into an unrelated time.
        hi = sigma.masked_fill(~unknown, -torch.inf).flatten(1).amax(-1)
        lo = sigma.masked_fill(~unknown, torch.inf).flatten(1).amin(-1)
        if not torch.isfinite(hi).all() or (hi < 0).any() or (hi > 1).any() or not torch.allclose(hi, lo, atol=1e-6, rtol=0):
            raise ValueError('Kimodo adapter requires a uniform unknown-channel time field')
        lattice = (hi * self.sampling_steps).round()
        if not torch.allclose(hi, lattice / self.sampling_steps, atol=1e-6, rtol=0):
            raise ValueError('Time field is not on the configured inference lattice')
        # Index official sinusoidal timestep table. This is a time embedding
        # coordinate, not adoption of Kimodo DDPM/DDIM dynamics.
        timestep = (hi * 1000).round().long()
        x = x.masked_fill(~live, 0)
        mask = known & live
        observed = torch.where(mask, x, 0)
        null_text = x.new_zeros(x.shape[0], 1, 4096)
        null_mask = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool)
        result = self.backbone(x, valid, null_text, null_mask, timestep,
                               motion_mask=mask.to(x.dtype), observed_motion=observed)
        return result.masked_fill(~live, 0)
