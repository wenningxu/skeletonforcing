"""Joint-time denoiser on a lossless packing of raw HumanML3D 263D channels."""
import torch
from torch import nn
from .model import AxialBlock, fourier
from .representation263 import Layout263


class TextAxialBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.motion=AxialBlock(width,heads)
        self.norm=nn.LayerNorm(width)
        self.cross=nn.MultiheadAttention(width,heads,batch_first=True)
        self.heads=heads

    def forward(self,x,valid,text,text_valid,frame_text_allowed=None):
        x=self.motion(x,valid); b,t,j,d=x.shape
        query=x.reshape(b,t*j,d)
        attention_mask=None
        if frame_text_allowed is not None:
            if frame_text_allowed.shape!=(b,t,text.shape[1]): raise ValueError('invalid frame text alignment')
            if not (frame_text_allowed & text_valid[:,None]).any(-1).all(): raise ValueError('each frame must see at least one valid text token')
            forbidden=~frame_text_allowed[:,:,None,:].expand(b,t,j,text.shape[1]).reshape(b,t*j,-1)
            attention_mask=forbidden[:,None].expand(b,self.heads,t*j,text.shape[1]).reshape(b*self.heads,t*j,-1)
        delta=self.cross(self.norm(query),text,text,key_padding_mask=~text_valid,
                         attn_mask=attention_mask,need_weights=False)[0]
        return (query+delta).reshape(b,t,j,d)


class Motion263Flow(nn.Module):
    def __init__(self,width=256,depth=6,heads=8,text_dim=4096):
        super().__init__()
        self.config=dict(width=width,depth=depth,heads=heads,text_dim=text_dim)
        self.layout=Layout263()
        self.input=nn.Linear(13,width)
        self.observed=nn.Linear(13,width,bias=False)
        self.noise_channels=nn.Linear(13,width,bias=False)
        self.noise_time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.joint=nn.Embedding(22,width)
        self.text=nn.Sequential(nn.Linear(text_dim,width),nn.GELU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([TextAxialBlock(width,heads) for _ in range(depth)])
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,13))

    def forward(self,x,sigma,known,text,text_valid,valid,frame_text_allowed=None):
        if text.shape[-1]!=self.config['text_dim']: raise ValueError('wrong text embedding width; expected UMT5 tokens')
        if not text_valid.any(-1).all(): raise ValueError('all-padding text is invalid')
        _,t,_=x.shape; width=self.config['width']
        packed=self.layout.pack(x); s=self.layout.pack(sigma)
        mask=self.layout.pack(known.float())
        h=self.input(packed)+self.observed(mask)+self.noise_channels(s)
        mean_sigma=s.sum(-1)/self.layout.live.sum(-1).clamp_min(1)
        h=h+self.noise_time(fourier(mean_sigma*1000,width))
        h=h+self.joint.weight[None,None]+fourier(torch.arange(t,device=x.device),width)[None,:,None]
        tokens=self.text(text.to(h.dtype))
        for block in self.blocks: h=block(h,valid,tokens,text_valid,frame_text_allowed)
        return self.layout.unpack(self.output(h))
