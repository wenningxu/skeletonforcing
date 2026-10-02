"""Lossless layout of the SAME 263 raw channels used by FloodDiffusion's VAE.

No XYZ roundtrip, learned compression, channel dropping, or alternative features.
Official order: root(4), RIC(63), rot6d(126), local velocity(66), contact(4).
"""
import torch
from torch import nn

CONTACT_JOINTS = (7, 10, 8, 11)


def joint_channel_groups():
    groups = [list(range(4)) + list(range(193, 196))]
    for j in range(1, 22):
        groups.append(list(range(4+3*(j-1), 4+3*j))
                      + list(range(67+6*(j-1), 67+6*j))
                      + list(range(193+3*j, 196+3*j)))
    for k,j in enumerate(CONTACT_JOINTS): groups[j].append(259+k)
    assert sorted(c for g in groups for c in g) == list(range(263))
    return groups


GROUPS = joint_channel_groups()
CHANNEL_OWNER = torch.tensor([next(j for j,g in enumerate(GROUPS) if c in g) for c in range(263)])


class Layout263(nn.Module):
    def __init__(self):
        super().__init__()
        index=torch.zeros(22,13,dtype=torch.long)
        live=torch.zeros(22,13,dtype=torch.bool)
        reverse=torch.empty(263,dtype=torch.long)
        for j,g in enumerate(GROUPS):
            index[j,:len(g)]=torch.tensor(g); live[j,:len(g)]=True
            for k,c in enumerate(g): reverse[c]=j*13+k
        self.register_buffer('index',index); self.register_buffer('live',live)
        self.register_buffer('reverse',reverse); self.register_buffer('owner',CHANNEL_OWNER.clone())

    def pack(self,x):
        if x.shape[-1]!=263: raise ValueError('expected last dimension 263')
        return x[...,self.index]*self.live.to(x.dtype)

    def unpack(self,x):
        if x.shape[-2:]!=(22,13): raise ValueError('expected last dimensions 22,13')
        return x.flatten(-2)[...,self.reverse]


class Normalizer263(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        mean=torch.as_tensor(mean).float(); std=torch.as_tensor(std).float()
        if mean.shape!=(263,) or std.shape!=(263,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
            raise ValueError('invalid official Mean.npy/Std.npy')
        self.register_buffer('mean',mean); self.register_buffer('std',std)
    def normalize(self,x): return (x-self.mean)/self.std
    def denormalize(self,x): return x*self.std+self.mean


def local_ric_mask(joint_mask):
    """Observing local RIC position does NOT reveal rotation/velocity/contact.

    Root has only absolute height as a direct positional channel. Reject root
    XYZ constraints rather than silently treating height as root position.
    """
    if joint_mask.shape[-1]!=22 or joint_mask.dtype!=torch.bool: raise ValueError('expected boolean ...,22')
    if joint_mask[...,0].any(): raise ValueError('root XYZ needs an integrated-trajectory constraint; unsupported here')
    mask=torch.zeros(*joint_mask.shape[:-1],263,dtype=torch.bool,device=joint_mask.device)
    for j in range(1,22): mask[...,4+3*(j-1):4+3*j]=joint_mask[...,j,None]
    return mask
