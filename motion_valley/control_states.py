"""Coherent articulated motion variants and paired control-response metrics."""
import torch
from .representation266 import xyz266, to263, Normalizer266
from .diagnostics266 import fk_xyz
from utils.math.quaternion import cont6d_to_matrix


def arm_variant(raw, degrees, joint=17, center=48, radius=12):
    """Postmultiply local shoulder rotation by smooth Rz; recompute FK/velocity.

    Root and feet are unchanged. This ensures kinematic consistency, not
    collision avoidance, anatomical limits, or unchanged text semantics.
    """
    if raw.ndim!=3 or raw.shape[-1]!=266 or not 1<=joint<=21: raise ValueError('Invalid motion/joint')
    frame=torch.arange(raw.shape[1],device=raw.device,dtype=raw.dtype)
    u=(frame-center)/radius
    envelope=torch.where(u.abs()<1,torch.cos(torch.pi*u/2).square(),0)
    angle=envelope*torch.as_tensor(degrees,device=raw.device,dtype=raw.dtype)*torch.pi/180
    c,s=angle.cos(),angle.sin(); rotation=torch.zeros(len(frame),3,3,device=raw.device,dtype=raw.dtype)
    rotation[:,0,0]=c; rotation[:,0,1]=-s; rotation[:,1,0]=s; rotation[:,1,1]=c; rotation[:,2,2]=1
    result=raw.clone(); start=70+6*(joint-1)
    transformed=cont6d_to_matrix(raw[...,start:start+6])@rotation[None]
    result[...,start:start+6]=torch.cat([transformed[...,0],transformed[...,1]],-1)
    xyz,invalid=fk_xyz(result,raw)
    if invalid.any(): raise ValueError('Degenerate source rotation')
    result[...,4:70]=xyz.flatten(-2)
    export=to263(result)
    result[...,1:4]=export[...,1:4]
    result[...,196:262]=export[...,193:259]
    return result


def fit_state_normalizer(train_raw,official_mean,official_std):
    xyz=train_raw[...,4:70].reshape(-1,66)
    return Normalizer266(torch.cat([official_mean[:4],xyz.mean(0),official_mean[67:]]),
                         torch.cat([official_std[:4],xyz.std(0,unbiased=False).clamp_min(.05),official_std[67:]]))


def coherent_response(pred,base,truth,base_truth,known,first=36,last=60,joints=(17,19,21)):
    """Compare actual and correct free-joint changes; a clamp-only model fails."""
    delta=xyz266(pred)-xyz266(base); target=xyz266(truth)-xyz266(base_truth)
    selected=torch.zeros(*known.shape[:2],22,device=known.device,dtype=torch.bool)
    selected[:,first:last+1,list(joints)]=True
    selected &= ~known[...,4:70].reshape(*known.shape[:2],22,3).all(-1)
    a=delta[selected]; b=target[selected]; energy=b.square().sum()
    if not torch.isfinite(energy) or energy<=1e-10: raise ValueError('Intervention has no measurable free-joint target change')
    return dict(response_gain=(a*b).sum().div(energy).item(),
                response_relative_error=((a-b).square().sum()/energy).sqrt().item(),
                response_target_rms_m=b.square().sum(-1).mean().sqrt().item(),
                response_actual_rms_m=a.square().sum(-1).mean().sqrt().item())
