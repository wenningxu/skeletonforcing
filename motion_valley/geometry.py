import numpy as np
import torch
from .schedule import PARENTS


def geometric_metrics(pred,truth,mask,valid,fps=20):
    """Metres; derivatives have SI units. These jerk metrics are NOT paper PJ/AUJ."""
    error=(pred-truth).norm(dim=-1)
    known=mask & valid[...,None]
    free=~mask & valid[...,None]
    anchors=error[known]
    edges=(pred[:,:,1:]-pred[:,:,PARENTS[1:]]).norm(dim=-1)
    true_edges=(truth[:,:,1:]-truth[:,:,PARENTS[1:]]).norm(dim=-1)
    bone=(edges-true_edges).abs()[valid]
    result=dict(anchor_mean_m=anchors.mean().item() if len(anchors) else None,
                anchor_max_m=anchors.max().item() if len(anchors) else None,
                anchor_success_1mm=(anchors<=.001).float().mean().item() if len(anchors) else None,
                free_mpjpe_m=error[free].mean().item() if free.any() else None,
                bone_length_mae_m=bone.mean().item())
    peaks=[]; integrals=[]; acc=[]
    for x,n in zip(pred,valid.sum(1)):
        x=x[:int(n)]
        acc.append(x.diff(n=2,dim=0).norm(dim=-1).mean().item()*fps**2)
        jerk=x.diff(n=3,dim=0).norm(dim=-1)*fps**3
        peaks.append(jerk.max().item())
        integrals.append((jerk.mean(-1).sum()/fps).item())
    result.update(jerk_peak_m_s3=float(np.mean(peaks)),jerk_integral_m_s2=float(np.mean(integrals)),
                  acceleration_mean_m_s2=float(np.mean(acc)))
    return result
