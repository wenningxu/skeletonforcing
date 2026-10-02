import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from motion_valley.control_factors import common_known
from motion_valley.control_states import coherent_response
plan=json.loads((root/'configs/control_factors_v1.json').read_text())
with np.load(root/plan['bank_file']) as data:states=torch.from_numpy(data['states']);angles=data['angles'].tolist()
xyz=states[0,[angles.index(a) for a in [-12,0,12]],:,4:70].reshape(-1,66)
mean=xyz.mean(0);std=xyz.std(0,unbiased=False).clamp_min(.05)
common=common_known(plan['eval_layouts']); rows=[]
for i in range(8):
    norm=(states[i,:,:,4:70]-mean)/std
    base=states[i:i+1,angles.index(0)]
    targets=[]
    for angle in [-12,-6,6,12]:
        raw=states[i:i+1,angles.index(angle)]
        targets.append(coherent_response(raw,base,raw,base,common)['response_target_rms_m'])
    rows.append(dict(bank_index=i,normalized_xyz_abs_p99=float(norm.abs().flatten().quantile(.99)),
        normalized_xyz_abs_max=float(norm.abs().max()),minimum_common_target_rms_mm=min(targets)*1000))
out=root/'outputs/OF007-preflight';out.mkdir(exist_ok=True)
result=dict(common_free_response_points=48,bank_clips=rows,layout_counts={k:len(v) for k,v in plan['eval_layouts'].items()})
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
