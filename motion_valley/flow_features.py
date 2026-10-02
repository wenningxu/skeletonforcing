"""Shared discrete feature-time fields and actual-state rollout, no motion VAE."""
from dataclasses import dataclass
import torch
from .representation263 import CHANNEL_OWNER
from .schedule import distance_field, trajectory_step


@dataclass
class FeatureControls:
    values: torch.Tensor  # B,T,C; only values at known mask may be used
    known: torch.Tensor
    joints: torch.Tensor  # B,T,22; anchors for distance, not blanket channel mask

    def __post_init__(self):
        if self.known.dtype!=torch.bool or self.joints.dtype!=torch.bool: raise ValueError('boolean masks required')
        if self.values.shape!=self.known.shape or self.values.shape[-1] not in (263,266): raise ValueError('invalid feature controls')
        if self.joints.shape!=self.values.shape[:-1]+(22,): raise ValueError('invalid joint mask shape')
        if not torch.isfinite(self.values[self.known]).all(): raise ValueError('nonfinite observations')

    def project(self,x):
        return torch.where(self.known,self.values,x)


def field_step(k,steps,controls,distance,mode='valley'):
    # Anchored position does not reveal that joint's rotations/velocities.
    # Nonobserved channels at that location have delay=0 but start at sigma=1.
    blank=torch.zeros_like(controls.joints)
    a,b=trajectory_step(k,steps,blank,distance,mode)
    if controls.values.shape[-1]==266:
        from .representation266 import Layout266
        owner=Layout266().owner.to(a.device)
    else:
        owner=CHANNEL_OWNER.to(a.device)
    return a[...,owner].masked_fill(controls.known,0),b[...,owner].masked_fill(controls.known,0)


def advance(x,clean_prediction,sigma,next_sigma,controls):
    # x0 prediction yields a well-defined supervised target even off the
    # analytic data-noise interpolation path. Do not label rolled-out x with
    # the original epsilon-x0 as if it were still on that path.
    active=next_sigma<sigma
    velocity=(x-clean_prediction)/sigma.clamp_min(1e-8)
    updated=torch.where(active,x+(next_sigma-sigma)*velocity,x)
    return controls.project(updated)


def rollout(model,noise,controls,text,text_valid,valid,steps=32,mode='valley',
            stop_step=None,grad_last=0,callback=None,frame_text_allowed=None):
    """Start from noise and feed every previous model output into the next step.

    grad_last=0 is inference/detached prefix. Positive grad_last differentiates
    through only the last N executed steps, explicitly truncated BPTT.
    """
    end=steps if stop_step is None else stop_step
    if not 0<=end<=steps or not 0<=grad_last<=end: raise ValueError('invalid rollout/truncation')
    d=distance_field(controls.joints)
    x=controls.project(noise)
    if callback: callback(0,x)
    for k in range(end):
        a,b=field_step(k,steps,controls,d,mode)
        with torch.set_grad_enabled(grad_last>0 and k>=end-grad_last):
            pred=model(x,a,controls.known,text,text_valid,valid,frame_text_allowed)
            x=advance(x,pred,a,b,controls)
        if callback: callback(k+1,x)
    return x


def teacher_state(clean,noise,k,steps,controls,mode='valley'):
    a,b=field_step(k,steps,controls,distance_field(controls.joints),mode)
    x=controls.project((1-a)*clean+a*noise)
    return x,a,b
