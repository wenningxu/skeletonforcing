"""Joint-time valley front with equal local denoising budgets and shifted starts.

No user anchors, ground-truth root, or angle/state identifiers enter sampling.
The origin is a scheduling source, NOT a known motion value. This is not a
causal attention/streaming implementation.
"""
from dataclasses import dataclass
import torch
from .schedule import PARENTS,distance_field
from .representation266 import Layout266,to263


def joint_depths():
    depths=[]
    for joint,parent in enumerate(PARENTS):
        if parent>=joint:raise ValueError('Expected topologically ordered skeleton')
        depths.append(0 if parent<0 else depths[parent]+1)
    return depths


@dataclass(frozen=True)
class RootFirstSchedule:
    local_steps: int=32
    delay_scale_steps: int=32
    temporal_scale: float=20.
    joint_scale: float=2.
    spread: float=.75

    def __post_init__(self):
        if type(self.local_steps)!=int or self.local_steps<1 or type(self.delay_scale_steps)!=int or self.delay_scale_steps<1 or self.temporal_scale<=0 or self.joint_scale<=0 or not 0<self.spread<1:
            raise ValueError('Invalid valley schedule parameters')

    def start_steps(self,frames,device=None):
        if type(frames)!=int or frames<1:raise ValueError('Invalid frame count')
        origin=torch.zeros(1,frames,22,device=device,dtype=torch.bool);origin[:,0,0]=True
        distance=distance_field(origin,self.temporal_scale,self.joint_scale)
        delay=self.spread*distance.double()/(1+distance.double())
        # Preserve the old propagation-distance map, but quantize starts to a
        # global step boundary. Tolerance prevents roundoff adding a whole step
        # at mathematically integral boundaries (e.g. 8.0000000001 -> 8).
        return torch.ceil(delay[0]*self.delay_scale_steps-1e-6).long().clamp_min(0)

    def total_steps(self,frames):
        return int(self.start_steps(frames).max())+self.local_steps

    def lattice(self,frames,device=None):
        starts=self.start_steps(frames,device)
        owner=Layout266().owner.to(device)
        total=int(starts.max())+self.local_steps
        k=torch.arange(total+1,device=device)[:,None,None]
        local_index=(k-starts[None]).clamp(0,self.local_steps)
        sigma=1-local_index.float()/self.local_steps
        return sigma[...,owner]

    def step(self,k,like):
        if like.ndim!=3 or like.shape[-1]!=266:raise ValueError('Expected B,T,266')
        table=self.lattice(like.shape[1],like.device)
        if type(k)!=int or not 0<=k<len(table)-1:raise ValueError('Invalid inference step')
        return table[k][None].expand_as(like),table[k+1][None].expand_as(like)


def advance(x,pred,sigma,next_sigma):
    active=next_sigma<sigma
    ratio=next_sigma/sigma.clamp_min(1e-8)
    # At sigma_next=0, commit the model prediction exactly. Never overwrite it
    # with GT. Completed and not-yet-started features remain bitwise unchanged.
    return torch.where(active,ratio*x+(1-ratio)*pred,x)


def rollout(model,noise,valid,schedule,stop_step=None,grad_last=0,callback=None):
    table=schedule.lattice(noise.shape[1],noise.device);total=len(table)-1
    end=total if stop_step is None else stop_step
    if not 0<=end<=total or not 0<=grad_last<=end:raise ValueError('Invalid truncation')
    known=torch.zeros_like(noise,dtype=torch.bool)
    x=noise.clone()
    if callback:callback(0,x)
    for k in range(end):
        a=table[k][None].expand_as(x);b=table[k+1][None].expand_as(x)
        with torch.set_grad_enabled(grad_last>0 and k>=end-grad_last):
            pred=model(x,a,known,None,None,valid)
            x=advance(x,pred,a,b)
        if callback:callback(k+1,x)
    return x


def training_prediction(model,clean,noise,valid,k,schedule,use_rollout=False,gradient_tail=4):
    a,b=schedule.step(k,clean)
    if use_rollout:
        state=rollout(model,noise,valid,schedule,stop_step=k,grad_last=min(k,gradient_tail))
    else:
        # Only teacher training mixes GT into states. Rollout training and
        # inference instead use the previous generated joint-time state.
        state=(1-a)*clean+a*noise
    pred=model(state,a,torch.zeros_like(clean,dtype=torch.bool),None,None,valid)
    return pred,a,b,state


def active_redundancy_loss(raw,valid,active,available=None):
    """Root/velocity consistency only on channels advancing in this step.

    Caller combines active predictions with the actual inactive sampler state.
    Unstarted noisy children must not dominate the physical consistency loss.
    """
    export=to263(raw)
    residual=torch.cat([raw[...,1:4]-export[...,1:4],raw[...,196:262]-export[...,193:259]],-1)
    available=active if available is None else available
    pos=available[...,4:70].reshape(*raw.shape[:2],22,3).all(-1)
    # A forward velocity needs BOTH endpoint positions, and root orientation
    # needs the angular prefix. Do not supervise consistency against untouched
    # next-frame Gaussian positions at the advancing wavefront.
    pair=pos[:,:-1]&pos[:,1:]
    heading=available[...,0].long().cumprod(1).bool()[:,:-1]
    root_mask=active[:,:-1,1:4].clone()
    root_mask[...,:2] &= (pair[...,0]&heading)[...,None]
    root_mask[...,2] &= pos[:,:-1,0]
    velocity_mask=active[:,:-1,196:262].reshape(*pair.shape,3)
    velocity_mask=velocity_mask & pair[...,None] & heading[...,None,None]
    mask=torch.cat([root_mask,velocity_mask.flatten(-2)],-1)
    mask &= (valid[:,:-1]&valid[:,1:])[...,None]
    if not mask.any():return raw.sum()*0
    return residual[:,:-1].square()[mask].mean()
