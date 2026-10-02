"""Portable Flood/Wan blocks with the released parameter names and equations.

Adapted from third_party/FloodDiffusion/models/tools/wan_model.py (Apache-2.0).
Original source: https://github.com/Wan-Video/Wan2.2
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
Changes: PyTorch SDPA backend, explicit boolean masks, device-neutral FP32
modulation, optional removal of text, and RoPE coordinates supplied by caller.
These are component ports, not pretrained motion-generator weights.
"""
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(dim, position):
    if dim % 2: raise ValueError('Frequency dimension must be even')
    half=dim//2
    frequency=10000.**(-torch.arange(half,device=position.device,dtype=torch.float64)/half)
    phase=position.double()[...,None]*frequency
    return torch.cat([phase.cos(),phase.sin()],-1).float()


def grid_rope(position, head_dim):
    """Exactly Wan's three-axis frequency allocation, evaluated at explicit coords.

    position: L,3; motion adapter uses (frame, joint index, 0).
    Joint indices are positional identities, not anatomical graph distances.
    """
    if position.ndim!=2 or position.shape[-1]!=3 or head_dim % 2:
        raise ValueError('Expected L,3 positions and an even head dimension')
    sizes=[head_dim-4*(head_dim//6),2*(head_dim//6),2*(head_dim//6)]
    phases=[]
    for axis,size in enumerate(sizes):
        if size:
            frequency=10000.**(-torch.arange(0,size,2,device=position.device,dtype=torch.float64)/size)
            phases.append(position[:,axis,None].double()*frequency)
    angle=torch.cat(phases,-1)
    return torch.polar(torch.ones_like(angle),angle)


def apply_rope(x, frequencies):
    # B,L,H,D, with adjacent real pairs as in Wan's complex implementation.
    pair=torch.view_as_complex(x.double().reshape(*x.shape[:-1],-1,2))
    result=torch.view_as_real(pair*frequencies[None,:,None]).flatten(-2)
    return result.float()


class WanRMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))

    def forward(self,x):
        normalized=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return normalized.to(x.dtype)*self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self,dim,eps=1e-6,elementwise_affine=False):
        super().__init__(dim,eps=eps,elementwise_affine=elementwise_affine)

    def forward(self,x):
        return super().forward(x.float()).to(x.dtype)


class WanSelfAttention(nn.Module):
    def __init__(self,dim,heads,eps=1e-6):
        super().__init__()
        if dim % heads or (dim//heads) % 2: raise ValueError('Invalid attention dimensions')
        self.num_heads=heads; self.head_dim=dim//heads
        self.q=nn.Linear(dim,dim); self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim); self.o=nn.Linear(dim,dim)
        # Wan normalizes the full projected width BEFORE splitting heads.
        self.norm_q=WanRMSNorm(dim,eps); self.norm_k=WanRMSNorm(dim,eps)

    def attend(self,query,context,allowed,frequencies=None):
        b,l,d=query.shape; n=self.num_heads
        q=self.norm_q(self.q(query)).reshape(b,l,n,self.head_dim)
        k=self.norm_k(self.k(context)).reshape(b,context.shape[1],n,self.head_dim)
        v=self.v(context).reshape(b,context.shape[1],n,self.head_dim)
        if frequencies is not None:
            q=apply_rope(q,frequencies); k=apply_rope(k,frequencies)
        # SDPA requires matching dtypes; retain BF16 matmuls under autocast.
        result=F.scaled_dot_product_attention(q.transpose(1,2).to(v.dtype),
            k.transpose(1,2).to(v.dtype),v.transpose(1,2),attn_mask=allowed,
            dropout_p=0.,is_causal=False)
        return self.o(result.transpose(1,2).reshape(b,l,d))

    def forward(self,x,valid,frequencies):
        return self.attend(x,x,valid[:,None,None,:],frequencies)


class WanCrossAttention(WanSelfAttention):
    def forward(self,x,context,allowed):
        return self.attend(x,context,allowed[:,None])


class WanAttentionBlock(nn.Module):
    def __init__(self,dim,ffn_dim,heads,eps=1e-6):
        super().__init__()
        self.norm1=WanLayerNorm(dim,eps)
        self.self_attn=WanSelfAttention(dim,heads,eps)
        self.norm3=WanLayerNorm(dim,eps,elementwise_affine=True)
        self.cross_attn=WanCrossAttention(dim,heads,eps)
        self.norm2=WanLayerNorm(dim,eps)
        self.ffn=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(approximate='tanh'),nn.Linear(ffn_dim,dim))
        self.modulation=nn.Parameter(torch.randn(1,6,dim)/dim**.5)

    def remove_text(self):
        del self.norm3; del self.cross_attn

    def forward(self,x,e,valid,frequencies,context=None,text_allowed=None):
        # Shared e is B,L,6,D. Each block has its own learned modulation offset.
        shift1,scale1,gate1,shift2,scale2,gate2=(e.float()+self.modulation.float()).unbind(-2)
        y=self.self_attn(self.norm1(x).float()*(1+scale1)+shift1,valid,frequencies)
        x=x.float()+gate1*y.float()
        if hasattr(self,'cross_attn'):
            if context is None or text_allowed is None: raise ValueError('Text context required')
            x=x+self.cross_attn(self.norm3(x),context,text_allowed)
        elif context is not None: raise ValueError('Text supplied to a no-text block')
        y=self.ffn(self.norm2(x).float()*(1+scale2)+shift2)
        return (x+gate2*y.float()).masked_fill(~valid[...,None],0)


class WanHead(nn.Module):
    def __init__(self,dim,out_dim,eps=1e-6):
        super().__init__()
        self.norm=WanLayerNorm(dim,eps); self.head=nn.Linear(dim,out_dim)
        self.modulation=nn.Parameter(torch.randn(1,2,dim)/dim**.5)

    def forward(self,x,e):
        shift,scale=(e.float().unsqueeze(-2)+self.modulation.float()).unbind(-2)
        return self.head(self.norm(x).float()*(1+scale)+shift)
