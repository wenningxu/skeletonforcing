"""CPU data/schedule audit: is the control informative and is its loss diluted?"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from motion_valley.flow_features import FeatureControls,field_step
from motion_valley.representation266 import position_mask
from motion_valley.schedule import distance_field


def main():
    cfg=json.loads((ROOT/'configs/overfit_wan_v1_a.json').read_text())
    with np.load(ROOT/cfg['bank_file']) as bank:
        states=bank['states'][0].astype(np.float64); angles=bank['angles'].tolist()
    train=[angles.index(a) for a in cfg['train_angles']]
    xyz=states[:,:,4:70].reshape(5,64,22,3)
    std=xyz[train].reshape(-1,66).std(0).clip(.05).reshape(22,3)
    result={'layouts':{},'angle_order':angles,'train_angles':cfg['train_angles']}
    for layout in ['A','D']:
        anchors=cfg['layouts'][layout]
        observations=np.stack([xyz[:,t,j] for t,j in anchors],1).reshape(5,-1)
        # Fit only three training states; no angle or state id is an input.
        center=observations[train].mean(0); scale=np.maximum(observations[train].std(0),1e-6)
        features=np.concatenate([(observations-center)/scale,np.ones((5,1))],axis=1)
        coefficients=np.linalg.lstsq(features[train],states[train].reshape(3,-1),rcond=None)[0]
        prediction=(features@coefficients).reshape(states.shape)
        joints=torch.zeros(1,64,22,dtype=torch.bool)
        for t,j in anchors: joints[0,t,j]=True
        known=position_mask(joints)
        controls=FeatureControls(torch.zeros(1,64,266),known,joints)
        free=~joints[0].numpy()
        pred_xyz=prediction[:,:,4:70].reshape(5,64,22,3)
        oracle_errors=np.linalg.norm(pred_xyz-xyz,axis=-1)[:,free].mean(1)
        mean_xyz=xyz[train].mean(0)
        raw_variance=(xyz[train]-mean_xyz)**2
        variance=(raw_variance/std**2).mean(0)
        varying=raw_variance.mean(0).sum(-1)>1e-12
        step_rows=[]
        for k in range(32):
            s,nxt=field_step(k,32,controls,distance_field(joints))
            active=((nxt<s)&~known)[0].numpy()
            active_xyz=active[:,4:70].reshape(64,22,3)
            step_rows.append(dict(k=k,active_unknown_channels=int(active.sum()),
                xyz_state_variance_contribution_to_full_mse=float(variance[active_xyz].sum()/active.sum()),
                varying_xyz_fraction_in_full_loss=float((active_xyz&varying[...,None]).sum()/active.sum())))
        delta=observations[angles.index(12)]-observations[angles.index(0)]
        base=observations[angles.index(0)]
        result['layouts'][layout]=dict(control_delta_12_vs_0_m=delta.tolist(),
            controls_distinct=len(np.unique(observations,axis=0))==5,
            linear_control_oracle_free_mpjpe_mm={str(a):float(e*1000) for a,e in zip(angles,oracle_errors)},
            oracle_training_design_rank=int(np.linalg.matrix_rank(features[train])),
            oracle_uses_only_control_xyz=True,oracle_is_not_a_generative_model=True,
            affected_free_xyz_points=int((varying&free).sum()),total_free_xyz_points=int(free.sum()),
            xyz_only_training_mean_normalized_mse=float(variance[free].mean()),
            xyz_signal_full_loss_lower_bound_mean=float(np.mean([r['xyz_state_variance_contribution_to_full_mse'] for r in step_rows])),
            steps=step_rows)
    out=ROOT/'outputs/OF006-preflight'; out.mkdir(exist_ok=True)
    (out/'control_signal.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({l:{k:v for k,v in r.items() if k!='steps'} for l,r in result['layouts'].items()},indent=2))


if __name__=='__main__': main()
