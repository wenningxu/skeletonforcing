"""Diagnostic loss for fixed [-12, 0, +12] coherent state triplets."""
import torch


def state_difference_loss(pred, target, std, active, known):
    if pred.shape != target.shape or pred.shape[0] != 3 or pred.shape[-1] != 266:
        raise ValueError('Expected ordered three-state batch')
    if not torch.equal(active, active[:1].expand_as(active)) or not torch.equal(known, known[:1].expand_as(known)):
        raise ValueError('Paired states require identical fields/masks')
    region=torch.zeros(pred.shape[1],22,3,device=pred.device,dtype=torch.bool)
    region[36:61,[17,19,21]]=True
    mask=region & (active[1, :,4:70]&~known[1,:,4:70]).reshape(-1,22,3)
    a=(pred[[0,2],:,4:70].float()-pred[1:2,:,4:70].float())*std[4:70]
    b=(target[[0,2],:,4:70].float()-target[1:2,:,4:70].float())*std[4:70]
    a=a.reshape(2,-1,22,3)[:,mask]; b=b.reshape(2,-1,22,3)[:,mask]
    energy=b.square().sum()
    if energy.detach() <= 1e-10: return pred.float().sum()*0
    return (a-b).square().sum()/energy
