import json
from pathlib import Path
import torch
from motion_valley.control_factors import common_known
from motion_valley.control_experiment import fit_experiment_normalizer
from motion_valley.control_states import coherent_response
from motion_valley.representation266 import position_mask


def test_fixed_normalizer_independent_of_training_population():
    x=torch.randn(8,5,64,266);mean=torch.zeros(263);std=torch.ones(263);idx=[0,2,4]
    a=fit_experiment_normalizer(x,x[:1],idx,mean,std,{})
    b=fit_experiment_normalizer(x,x,idx,mean,std,{'normalizer_bank_clip_indices':[0]})
    torch.testing.assert_close(a.mean,b.mean,rtol=0,atol=0);torch.testing.assert_close(a.std,b.std,rtol=0,atol=0)
    x[1:]*=100
    c=fit_experiment_normalizer(x,x,idx,mean,std,{'normalizer_bank_clip_indices':[0]})
    torch.testing.assert_close(a.std,c.std,rtol=0,atol=0)


def test_common_score_cannot_reward_extra_clamping():
    plan=json.loads((Path(__file__).resolve().parents[1]/'configs/control_factors_v1.json').read_text())
    common=common_known(plan['eval_layouts'])
    region=torch.zeros(1,64,22,dtype=torch.bool);region[:,36:61,[17,19,21]]=True
    assert int((region&~common[...,4:70].reshape(1,64,22,3).all(-1)).sum())==48
    base=torch.zeros(1,64,266);truth=torch.randn_like(base)
    for anchors in plan['eval_layouts'].values():
        js=torch.zeros(1,64,22,dtype=torch.bool)
        for t,j in anchors:js[:,t,j]=True
        mask=position_mask(js)
        clamp=torch.where(mask,truth,base)
        r=coherent_response(clamp,base,truth,base,common)
        assert abs(r['response_gain'])<1e-8 and abs(r['response_relative_error']-1)<1e-6
    r=coherent_response(truth,base,truth,base,common)
    assert abs(r['response_gain']-1)<1e-6 and r['response_relative_error']==0


def test_factor_evaluation_reuses_noise_and_excludes_all_clamps(tmp_path):
    import numpy as np
    from motion_valley.control_factors import evaluate_factor
    from motion_valley.representation266 import Normalizer266
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'configs/control_factors_v1.json').read_text());plan['eval_seeds']=[2026]
    with np.load(root/plan['bank_file']) as data:
        states=torch.from_numpy(data['states']);angles=data['angles'].tolist()
    class Identity(torch.nn.Module):
        def forward(self,x,sigma,known,*args):
            assert torch.all(sigma[known]==0)
            assert torch.all(sigma[~known]==sigma[~known][0])
            return x
    evaluate_factor(Identity(),Normalizer266(torch.zeros(266),torch.ones(266)),states,angles,plan,tmp_path/'evaluation',lambda:9999)
    rows=[json.loads(s) for s in (tmp_path/'evaluation/evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==20
    assert all(abs(r['common_response_gain'])<1e-8 and abs(r['common_response_relative_error']-1)<1e-6 for r in rows if r['angle'])
