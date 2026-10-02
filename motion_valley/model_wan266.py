"""Standard Wan blocks on explicit joint-time motion tokens; no motion VAE."""
import torch
from torch import nn
from .representation266 import Layout266
from .wan_components import WanAttentionBlock,WanHead,sinusoidal_embedding,grid_rope


class WanMotion266(nn.Module):
    def __init__(self,width=256,depth=8,heads=8,ffn_dim=1024,freq_dim=256,
                 text_dim=4096,time_embedding_scale=1.,text_enabled=True,input_mode='split_linear'):
        super().__init__()
        if depth<1 or freq_dim<2 or freq_dim%2 or time_embedding_scale<=0:
            raise ValueError('Invalid model configuration')
        if input_mode not in ['split_linear','concat_mask']: raise ValueError('Invalid input mode')
        self.config=dict(architecture='wan_joint_time_v1',width=width,depth=depth,heads=heads,
            ffn_dim=ffn_dim,freq_dim=freq_dim,text_dim=text_dim,
            time_embedding_scale=time_embedding_scale,text_enabled=text_enabled,input_mode=input_mode)
        self.layout=Layout266(); self.motion_embedding=nn.Linear(13,width)
        self.observed_embedding=nn.Linear(13,width,bias=False); self.joint=nn.Embedding(22,width)
        # Minimal scalar->vector time adapter: preserve channel identity, never
        # average clean XYZ and noisy rotations/velocities into one scalar.
        self.time_embedding=nn.Sequential(nn.Linear(13*freq_dim,width),nn.SiLU(),nn.Linear(width,width))
        self.time_projection=nn.Sequential(nn.SiLU(),nn.Linear(width,6*width))
        self.text_embedding=nn.Sequential(nn.Linear(text_dim,width),nn.GELU(approximate='tanh'),nn.Linear(width,width))
        self.blocks=nn.ModuleList([WanAttentionBlock(width,ffn_dim,heads) for _ in range(depth)])
        self.head=WanHead(width,13)
        self.initialize_weights()
        # Prune AFTER common initialization to keep both arms' motion weights identical.
        if not text_enabled:
            del self.text_embedding
            for block in self.blocks: block.remove_text()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
        for stack in [self.text_embedding,self.time_embedding]:
            for module in stack.modules():
                if isinstance(module,nn.Linear): nn.init.normal_(module.weight,std=.02)
        # Released Wan uses nonzero modulation offsets but a zero output head.
        # This is not an assertion that every block uses AdaLN-Zero initialization.
        nn.init.zeros_(self.head.head.weight)

    def time_condition(self,sigma):
        s=self.layout.pack(sigma)*self.config['time_embedding_scale']
        with torch.autocast(device_type=s.device.type,enabled=False):
            features=sinusoidal_embedding(self.config['freq_dim'],s.float())
            features=features*self.layout.live[...,None]
            e=self.time_embedding(features.flatten(-2))
            modulation=self.time_projection(e).unflatten(-1,(6,self.config['width']))
        return e,modulation

    def forward(self,x,sigma,known,text,text_valid,valid,frame_text_allowed=None):
        if x.ndim!=3 or x.shape[-1]!=266 or x.shape!=sigma.shape or known.shape!=x.shape:
            raise ValueError('Expected matching B,T,266 motion, sigma, known')
        b,t,_=x.shape; width=self.config['width']
        if known.dtype!=torch.bool or valid.dtype!=torch.bool or valid.shape!=(b,t) or not valid.any(-1).all():
            raise ValueError('Boolean masks with at least one valid frame required')
        if not torch.isfinite(sigma).all() or (sigma<0).any() or (sigma>1).any():
            raise ValueError('Sigma must be finite and in [0,1]')
        if (sigma[known]!=0).any(): raise ValueError('Known channels must remain noise-free')
        # Sanitize padding BEFORE projections; padded motion/text cannot contaminate valid outputs.
        x=x.masked_fill(~valid[...,None],0); sigma=sigma.masked_fill(~valid[...,None],0)
        known=known & valid[...,None]
        packed=self.layout.pack(x); mask=self.layout.pack(known.float())
        if self.config['input_mode']=='concat_mask':
            # Explicit [motion, known-mask] concatenation. Preserve the existing
            # parameter layout and initialization; only GEMM rounding can differ.
            h=torch.nn.functional.linear(torch.cat([packed,mask],-1),
                torch.cat([self.motion_embedding.weight,self.observed_embedding.weight],-1),
                self.motion_embedding.bias)
        else:
            h=self.motion_embedding(packed)+self.observed_embedding(mask)
        h=h+self.joint.weight[None,None]
        h=h.reshape(b,t*22,width); token_valid=valid[:,:,None].expand(b,t,22).reshape(b,t*22)
        time,modulation=self.time_condition(sigma)
        time=time.reshape(b,t*22,width); modulation=modulation.reshape(b,t*22,6,width)
        frames,joints=torch.meshgrid(torch.arange(t,device=x.device),torch.arange(22,device=x.device),indexing='ij')
        coords=torch.stack([frames,joints,torch.zeros_like(frames)],-1).reshape(-1,3)
        frequencies=grid_rope(coords,width//self.config['heads'])
        context=None; allowed=None
        if self.config['text_enabled']:
            if text is None or text.ndim!=3 or text.shape[0]!=b or text.shape[-1]!=self.config['text_dim']:
                raise ValueError('Expected official UMT5 token features')
            if text_valid is None or text_valid.dtype!=torch.bool or text_valid.shape!=text.shape[:2] or not text_valid.any(-1).all():
                raise ValueError('At least one valid text token required')
            context=self.text_embedding(text.masked_fill(~text_valid[...,None],0).to(h.dtype))
            allowed=text_valid[:,None,:].expand(b,t*22,text.shape[1])
            if frame_text_allowed is not None:
                if frame_text_allowed.dtype!=torch.bool or frame_text_allowed.shape!=(b,t,text.shape[1]):
                    raise ValueError('Invalid frame-to-text alignment')
                selected=frame_text_allowed & text_valid[:,None]
                if not selected.any(-1)[valid].all(): raise ValueError('Valid frames need an active text token')
                selected=torch.where(valid[...,None],selected,text_valid[:,None])
                allowed=selected[:,:,None].expand(b,t,22,text.shape[1]).reshape(b,t*22,-1)
        elif text is not None or text_valid is not None or frame_text_allowed is not None:
            raise ValueError('No-text model does not accept dummy text')
        for block in self.blocks: h=block(h,modulation,token_valid,frequencies,context,allowed)
        # Match Wan output-head full precision even when attention uses BF16 autocast.
        with torch.autocast(device_type=x.device.type,enabled=False):
            output=self.head(h.float(),time.float()).reshape(b,t,22,13)
        return self.layout.unpack(output).masked_fill(~valid[...,None],0)
