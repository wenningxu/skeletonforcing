"""Ablation with text projection and all text cross-attention physically removed."""
import torch
from torch import nn
from .model266 import Motion266Flow
from .model import fourier


class NoTextMotion266Flow(Motion266Flow):
    def __init__(self,width=256,depth=6,heads=8,text_dim=4096):
        # Initialize identically before pruning so common motion weights match
        # the text-enabled arm under an identical torch seed.
        super().__init__(width,depth,heads,text_dim)
        del self.text
        self.blocks=nn.ModuleList([b.motion for b in self.blocks])
        self.config['text_enabled']=False

    def forward(self,x,sigma,known,text,text_valid,valid,frame_text_allowed=None):
        if text is not None or text_valid is not None or frame_text_allowed is not None:
            raise ValueError('No-text model must not receive text or dummy tokens')
        _,t,_=x.shape; width=self.config['width']; s=self.layout.pack(sigma)
        h=self.input(self.layout.pack(x))+self.observed(self.layout.pack(known.float()))+self.noise_channels(s)
        mean_sigma=s.sum(-1)/self.layout.live.sum(-1).clamp_min(1)
        h=h+self.noise_time(fourier(mean_sigma*1000,width))
        h=h+self.joint.weight[None,None]+fourier(torch.arange(t,device=x.device),width)[None,:,None]
        for block in self.blocks: h=block(h,valid)
        return self.layout.unpack(self.output(h))
