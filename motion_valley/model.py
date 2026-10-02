import math
import torch
from torch import nn


def fourier(x, width):
    freq = torch.exp(torch.arange(width//2, device=x.device) * (-math.log(10000)/(width//2)))
    phase = x[..., None] * freq
    return torch.cat([phase.sin(), phase.cos()], -1)


class AxialBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(width, heads, 4*width, dropout=0., batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(width, heads, 4*width, dropout=0., batch_first=True, norm_first=True)

    def forward(self, x, valid):
        b,t,j,d = x.shape
        x = self.spatial(x.reshape(b*t,j,d)).reshape(b,t,j,d)
        x = x.permute(0,2,1,3).reshape(b*j,t,d)
        padding = ~valid[:,None,:].expand(b,j,t).reshape(b*j,t)
        x = self.temporal(x, src_key_padding_mask=padding)
        return x.reshape(b,j,t,d).permute(0,2,1,3)


class JointTimeFlow(nn.Module):
    """No motion encoder/decoder: XYZ -> joint/time transformer -> XYZ velocity.

    text is a per-frame mean GloVe vector (frozen external word embeddings).
    This small pilot text conditioner is explicitly different from Flood's UMT5.
    """
    def __init__(self, width=128, depth=4, heads=4, text_dim=300):
        super().__init__()
        self.config = dict(width=width, depth=depth, heads=heads, text_dim=text_dim)
        self.input = nn.Linear(4, width)
        self.sigma = nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.text = nn.Linear(text_dim,width)
        self.joint = nn.Embedding(22,width)
        self.blocks = nn.ModuleList([AxialBlock(width,heads) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, sigma, mask, text, valid):
        b,t,j,_ = x.shape
        width = self.config['width']
        h = self.input(torch.cat([x,mask[...,None].to(x.dtype)],-1))
        h = h + self.sigma(fourier(sigma*1000,width))
        h = h + self.joint.weight[None,None] + fourier(torch.arange(t,device=x.device),width)[None,:,None]
        h = h + self.text(text)[:,:,None]
        for block in self.blocks:
            h = block(h,valid)
        return self.output(h)
