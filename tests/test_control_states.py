import json
from pathlib import Path
import numpy as np
import pytest
import torch
from motion_valley.control_states import arm_variant,coherent_response,fit_state_normalizer
from motion_valley.model_no_text import NoTextMotion266Flow
from motion_valley.model266 import Motion266Flow
from motion_valley.representation266 import xyz266,position_mask,Normalizer266
from motion_valley.multi_overfit import make_controls
from motion_valley.control_experiment import text_batch,evaluate,summarize
from motion_valley.schedule import PARENTS

ROOT=Path(__file__).resolve().parents[1]
def bank(): return torch.from_numpy(np.load(ROOT/'prepared/OF003/states.npz')['states'])
def config(): return json.loads((ROOT/'configs/control_no_text_v4.json').read_text())

def test_articulated_states_preserve_bones_root_feet_and_update_velocity():
    data=np.load(ROOT/'prepared/OF003/states.npz'); original=torch.from_numpy(data['source266'])
    changed=arm_variant(original,12); pos=xyz266(changed); ref=xyz266(original)
    edges=(pos[...,1:,:]-pos[...,PARENTS[1:],:]).norm(dim=-1)
    gt_edges=(ref[...,1:,:]-ref[...,PARENTS[1:],:]).norm(dim=-1)
    torch.testing.assert_close(edges,gt_edges,atol=5e-6,rtol=1e-4)
    torch.testing.assert_close(pos[...,:16,:],ref[...,:16,:],atol=5e-6,rtol=1e-4)
    assert torch.equal(changed[...,262:],original[...,262:])
    torch.testing.assert_close(changed[:,[0,63],4:70],original[:,[0,63],4:70],atol=5e-6,rtol=1e-4)
    assert (pos[:,48,21]-ref[:,48,21]).norm(dim=-1).min()>.01
    # Rotation preserves vector norm, independently checking local velocities.
    stored=changed[:,:-1,196:262].reshape(8,63,22,3).norm(dim=-1)
    world=pos.diff(dim=1).norm(dim=-1)
    torch.testing.assert_close(stored,world,atol=2e-6,rtol=1e-4)
    assert not torch.allclose(changed[...,196:262],original[...,196:262])

def test_response_metric_rejects_clamp_only_and_accepts_exact_coherent_change():
    states=bank()[:1]; base=states[:,2]; target=states[:,4]
    norm=Normalizer266(torch.zeros(266),torch.ones(266)); controls=make_controls(target,norm,config()['layouts']['A'])
    copied=torch.where(controls.known,target,base)
    r=coherent_response(copied,base,target,base,controls.known)
    assert r['response_gain']==0 and r['response_relative_error']==1
    r=coherent_response(target,base,target,base,controls.known)
    assert r['response_gain']==1 and r['response_relative_error']==0

def test_no_text_branch_has_no_text_parameters_and_matching_initial_motion_weights():
    torch.manual_seed(17); full=Motion266Flow(width=32,depth=1,heads=4)
    torch.manual_seed(17); net=NoTextMotion266Flow(width=32,depth=1,heads=4)
    assert not any('text' in name or 'cross' in name for name,_ in net.named_parameters())
    assert torch.equal(net.input.weight,full.input.weight)
    for a,b in zip(net.blocks[0].parameters(),full.blocks[0].motion.parameters()): assert torch.equal(a,b)
    x=torch.randn(2,8,266); sigma=torch.ones_like(x); known=torch.zeros_like(x,dtype=torch.bool); valid=torch.ones(2,8,dtype=torch.bool)
    result=net(x,sigma,known,None,None,valid); result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    with pytest.raises(ValueError): net(x,sigma,known,torch.zeros(2,1,4096),None,valid)
    assert text_batch(None,[0,1],'cpu')==(None,None)

def test_normalization_fits_training_states_only():
    states=bank(); mean=torch.zeros(263); std=torch.ones(263)
    normalizer=fit_state_normalizer(states[:,[0,2,4]],mean,std)
    changed=states.clone(); changed[:,[1,3]]+=10000
    other=fit_state_normalizer(changed[:,[0,2,4]],mean,std)
    assert torch.equal(normalizer.mean,other.mean) and torch.equal(normalizer.std,other.std)

def test_full_coherent_evaluation_on_cpu(tmp_path):
    states=bank()[:1]; angles=[-12,-6,0,6,12]; cfg=config(); cfg.update(heldout_noise_seeds=[2026],sampling_steps=4)
    normalizer=Normalizer266(torch.zeros(266),torch.ones(266))
    class ControlLookup:
        def __call__(self,x,sigma,known,text,*args):
            assert text is None
            errors=((states[0]-x[0]).square()*known[0]).sum((1,2))
            return states[:,int(errors.argmin())]
    rows,complete=evaluate(ControlLookup(),states,angles,normalizer,None,cfg,tmp_path,lambda:1000)
    assert complete and len(rows)==20
    assert max(r['free_xyz_mpjpe_m'] for r in rows)<1e-6
    assert all(r['response_relative_error']<1e-5 for r in rows if r['angle']!=0)
    assert all(r['overwrite_baseline']['response_relative_error']==1 for r in rows if r['angle']!=0)
    assert not summarize([],config(),False,0)['trained_layout_trained_angle']['reconstruction_passed']
