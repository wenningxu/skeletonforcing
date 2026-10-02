"""Do not hide disagreement between redundant coordinates, rotation and root."""
import torch
from .representation266 import xyz266,to263,recover_root_rot_pos
from .schedule import PARENTS
from third_party.HumanML3D.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
from utils.math.quaternion import quaternion_to_matrix


def fk_xyz(raw,reference):
    pos=xyz266(raw); ref=xyz266(reference)
    lengths=(ref[:,0,1:]-ref[:,0,PARENTS[1:]]).norm(dim=-1)
    lengths=torch.cat([torch.zeros_like(lengths[:,:1]),lengths],-1)
    offsets=torch.as_tensor(t2m_raw_offsets,device=raw.device,dtype=raw.dtype)[None]*lengths[...,None]
    six=raw[...,70:196].reshape(*raw.shape[:-1],21,6)
    first=six[...,:3]; second=six[...,3:]
    first_norm=first.norm(dim=-1,keepdim=True)
    x=first/first_norm.clamp_min(1e-8)
    z=torch.cross(x,second,dim=-1); z_norm=z.norm(dim=-1,keepdim=True); z=z/z_norm.clamp_min(1e-8)
    y=torch.cross(z,x,dim=-1); rotation=torch.stack([x,y,z],-1)
    q,_=recover_root_rot_pos(raw[...,:4]); root_rotation=quaternion_to_matrix(q)
    joints=[None]*22; joints[0]=pos[...,0,:]
    for chain in t2m_kinematic_chain:
        r=root_rotation
        for a,b in zip(chain[:-1],chain[1:]):
            r=r@rotation[...,b-1,:,:]
            joints[b]=joints[a]+(r@offsets[:,None,b,:,None]).squeeze(-1)
    degenerate=(first_norm[...,0]<1e-6)|(z_norm[...,0]<1e-6)
    return torch.stack(joints,-2),degenerate


def consistency_metrics(raw,reference):
    export=to263(raw)
    pred_fk,invalid=fk_xyz(raw,reference); gt_fk,gt_invalid=fk_xyz(reference,reference)
    pos=xyz266(raw); gt=xyz266(reference)
    pred_edges=(pos[...,1:,:]-pos[...,PARENTS[1:],:]).norm(dim=-1)
    gt_edges=(gt[...,1:,:]-gt[...,PARENTS[1:],:]).norm(dim=-1)
    velocity_error=(pos.diff(dim=1)-gt.diff(dim=1)).norm(dim=-1).mean()
    static_velocity_error=gt.diff(dim=1).norm(dim=-1).mean()
    return dict(
        root_velocity_consistency_mae_m_frame=(raw[:,:-1,1:3]-export[:,:-1,1:3]).abs().mean().item(),
        local_velocity_consistency_mae_m_frame=(raw[:,:-1,196:262]-export[:,:-1,193:259]).abs().mean().item(),
        root_height_consistency_mae_m=(raw[...,3]-pos[...,0,1]).abs().mean().item(),
        fk_position_disagreement_m=(pred_fk-pos).norm(dim=-1).mean().item(),
        gt_fk_position_disagreement_m=(gt_fk-gt).norm(dim=-1).mean().item(),
        degenerate_rotation_count=int(invalid.sum()),gt_degenerate_rotation_count=int(gt_invalid.sum()),
        bone_length_mae_m=(pred_edges-gt_edges).abs().mean().item(),
        motion_velocity_error_m_frame=velocity_error.item(),
        static_pose_velocity_error_m_frame=static_velocity_error.item(),
        velocity_error_vs_static_ratio=(velocity_error/static_velocity_error.clamp_min(1e-8)).item())
