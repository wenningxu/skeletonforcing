"""Post-hoc CPU-only information probe, not a generative model or GPU run."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'configs/kimodo266_overfit_v1.json').read_text())
plan=json.loads((root/'configs/control_factors_v1.json').read_text())
with np.load(root/cfg['bank_file']) as d:
    states=d['states'][0].astype(np.float64); angles=d['angles'].tolist()
xyz=states[:, :,4:70].reshape(5,64,22,3)
anchors=cfg['layouts']['A']; train=[angles.index(a) for a in cfg['train_angles']]
control=np.stack([xyz[:,f,j] for f,j in anchors],axis=1).reshape(5,-1)
center=control[train].mean(0);scale=float(np.linalg.norm(control[train]-center,axis=1).max())
x=np.concatenate([np.ones((5,1)),(control-center)/scale],axis=1)
target=states.reshape(5,-1);target_center=target[train].mean(0)
weights,_,rank,singular=np.linalg.lstsq(x[train],target[train]-target_center,rcond=1e-6)
pred=(x@weights+target_center).reshape(states.shape)
for f,j in anchors:pred[:,f,4+3*j:7+3*j]=states[:,f,4+3*j:7+3*j]
outxyz=pred[:,:,4:70].reshape(5,64,22,3)
known=np.zeros((64,22),bool)
for f,j in anchors:known[f,j]=True
region=np.zeros_like(known);region[36:61,[17,19,21]]=True
for layout in plan['eval_layouts'].values():
    for f,j in layout:region[f,j]=False
assert region.sum()==48
zero=angles.index(0);rows=[]
for i,angle in enumerate(angles):
    row=dict(angle=angle,heldout=angle not in cfg['train_angles'],
             free_xyz_mpjpe_m=float(np.linalg.norm(outxyz[i]-xyz[i],axis=-1)[~known].mean()))
    if angle:
        a=(outxyz[i]-outxyz[zero])[region];b=(xyz[i]-xyz[zero])[region];energy=(b*b).sum()
        row.update(common_response_gain=float((a*b).sum()/energy),
                   common_response_relative_error=float(np.sqrt(((a-b)**2).sum()/energy)))
    rows.append(row)
record=dict(method='Post-hoc affine least-squares map from layout A known XYZ to full 266D; clean controls only',
    training_angles=cfg['train_angles'],input_angle_or_state_id=False,training_design_rank=int(rank),
    training_design_singular_values=singular.tolist(),input_scale_m=scale,
    limitations=['Single clip and fixed training layout A only','Deterministic supervised interpolation; no noise or generation',
                 'No test state used for fitting or normalization','Does not establish diffusion optimization or generalization to other clips/layouts'],
    rows=rows)
dest=root/'outputs/OF008-analysis';dest.mkdir(exist_ok=True)
(dest/'cpu_condition_probe.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
