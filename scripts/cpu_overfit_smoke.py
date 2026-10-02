"""Synthetic structural overfit ONLY; not HumanML3D evidence or UMT5 evaluation."""
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from motion_valley.model266 import Motion266Flow
from motion_valley.representation266 import position_mask
from motion_valley.flow_features import FeatureControls,rollout
from motion_valley.overfit import training_prediction


def main():
    torch.set_num_threads(2); torch.manual_seed(91)
    model=Motion266Flow(width=32,depth=1,heads=4).cpu()
    clean=torch.randn(1,8,266)*.1
    # Synthetic context checks tensor plumbing, NOT pretrained language quality.
    text=torch.randn(1,4,4096); tv=torch.ones(1,4,dtype=torch.bool); valid=torch.ones(1,8,dtype=torch.bool)
    joints=torch.zeros(1,8,22,dtype=torch.bool); joints[:,4,20]=True; joints[:,5,0]=True
    known=position_mask(joints); controls=FeatureControls(torch.where(known,clean,0),known,joints)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0)
    heldout=torch.randn_like(clean); start=time.monotonic()
    before=rollout(model,heldout,controls,text,tv,valid,steps=8)
    initial=(before-clean).square()[~known].mean().item()
    zero_baseline=clean.square()[~known].mean().item()
    steps=2500
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        noise=torch.randn_like(clean); k=step%8
        pred,_,_,_=training_prediction(model,clean,noise,controls,text,tv,valid,k,8,
                         use_rollout=step>=40,gradient_tail=2)
        loss=(pred-clean).square()[~known].mean()
        for group in optimizer.param_groups:
            group['lr']=.002*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
    after=rollout(model,heldout,controls,text,tv,valid,steps=8)
    final=(after-clean).square()[~known].mean().item()
    result=dict(kind='CPU synthetic structural overfit; not real motion or actual UMT5 outputs',
                steps=steps,initial_new_noise_rollout_mse=initial,final_new_noise_rollout_mse=final,
                zero_prediction_mse=zero_baseline,ratio_to_zero_predictor=final/zero_baseline,
                reduction=1-final/initial,anchor_exact=torch.equal(after[known],clean[known]),
                elapsed_seconds=time.monotonic()-start,torch=torch.__version__)
    root=Path(__file__).resolve().parents[1]
    previous=root/'reports/cpu_overfit_v2.json'; history=root/'reports/cpu_overfit_attempts.json'
    if previous.exists():
        attempts=json.loads(history.read_text()) if history.exists() else []
        attempts.append(json.loads(previous.read_text())); history.write_text(json.dumps(attempts,indent=2))
    (root/'reports/cpu_overfit_v2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if final>=zero_baseline*.1 or not result['anchor_exact']: raise SystemExit('Structural overfit check failed: must beat the trivial zero predictor by 10x')


if __name__=='__main__': main()
