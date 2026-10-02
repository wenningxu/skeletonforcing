import json
from pathlib import Path
import numpy as np
import pytest
import torch
from motion_valley.multi_overfit import make_controls, paired_response, select_clips, summarize, evaluate
from motion_valley.representation266 import Normalizer266, from263, xyz266
from motion_valley.flow_features import teacher_state, field_step
from motion_valley.schedule import distance_field

def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/overfit_multi_v3.json').read_text())

def test_counterfactual_only_changes_observed_position_no_gt_leakage():
    raw=from263(torch.randn(2,64,263)*.01)
    norm=Normalizer266(torch.randn(266),torch.rand(266)+.1)
    cfg=config(); base=make_controls(raw,norm,cfg['layouts']['A'])
    moved=make_controls(raw,norm,cfg['layouts']['A'],(48,21,0,.1))
    assert torch.equal(base.known,moved.known)
    changed=(base.values!=moved.values).nonzero()
    assert changed.tolist()==[[0,48,67],[1,48,67]]
    assert (base.values[~base.known]==0).all()
    other=raw.clone(); other[~base.known]=1e5
    assert torch.equal(base.values,make_controls(other,norm,cfg['layouts']['A']).values)
    with pytest.raises(ValueError): make_controls(raw,norm,cfg['layouts']['A'],(49,21,0,.1))

def test_pure_clamping_cannot_pass_free_response_metric():
    raw=from263(torch.randn(1,64,263)*.01)
    norm=Normalizer266(torch.zeros(266),torch.ones(266)); cfg=config()
    controls=make_controls(raw,norm,cfg['layouts']['A'],(48,21,0,.1))
    copied=torch.where(controls.known,controls.values,raw)
    result=paired_response(copied,raw,controls,48,21,0,.1)
    assert result['neighborhood_displacement_m']==0 and result['free_displacement_m']==0
    copied[:,47,67]+=.05
    result=paired_response(copied,raw,controls,48,21,0,.1)
    assert result['neighborhood_displacement_m']>0 and result['neighborhood_signed_axis_gain']>0

def test_every_layout_training_field_matches_inference_and_zero_anchor_sigma():
    cfg=config(); raw=from263(torch.randn(1,64,263)*.01)
    norm=Normalizer266(torch.zeros(266),torch.ones(266)); noise=torch.randn_like(raw)
    for layout in cfg['layouts'].values():
        c=make_controls(raw,norm,layout); d=distance_field(c.joints)
        for k in range(32):
            state,a,b=teacher_state(raw,noise,k,32,c)
            ia,ib=field_step(k,32,c,d)
            assert torch.equal(a,ia) and torch.equal(b,ib)
            assert (a[c.known]==0).all() and torch.equal(state[c.known],raw[c.known])

def test_selection_reproducible_train_only_no_mirrors_or_duplicate_captions(tmp_path):
    root=tmp_path/'raw_data/HumanML3D'; root.mkdir(parents=True)
    for folder in ['new_joint_vecs','texts']: (root/folder).mkdir()
    names=['000001','000002','000003','M000001']
    (root/'train.txt').write_text('\n'.join(names)); (root/'val.txt').write_text('900000')
    (root/'test.txt').write_text('900001')
    for i,name in enumerate(names):
        a=np.zeros((64,263),np.float32); a[:,1]=.01*(i+1); a[:,3]=1
        np.save(root/'new_joint_vecs'/(name+'.npy'),a)
        (root/'texts'/(name+'.txt')).write_text(f'action {i}#token#0#0')
    cfg=config(); cfg.update(clips=2,exclude_ids=['000002'],minimum_pairwise_mpjpe_m=0)
    a=select_clips(tmp_path,cfg); b=select_clips(tmp_path,cfg)
    assert a[1:]==b[1:]
    assert {r['id'] for r in a[2]['selected']}=={'000001','000003'}
    (root/'val.txt').write_text('000001')
    with pytest.raises(ValueError,match='overlap'): select_clips(tmp_path,cfg)

def test_incomplete_evaluation_never_passes_and_case_count_is_fixed():
    cfg=config(); result=summarize([],cfg,False,0)
    assert result['expected_cases']==256
    assert not result['training_layout_overfit_passed']
    assert not result['heldout_layout_overfit_passed']
    assert not result['shift_response_diagnostic_passed']

def test_full_paired_evaluation_matrix_on_cpu(tmp_path):
    cfg=config(); cfg.update(clips=2,heldout_noise_seeds=[2026],sampling_steps=4)
    raw=from263(torch.randn(2,64,263)*.01)
    norm=Normalizer266(torch.zeros(266),torch.ones(266))
    contexts=[torch.full((2,4096),float(i)) for i in range(3)]
    class TextLookup:
        def __call__(self,x,sigma,known,text,*args):
            index=int(text[0,0,0]); return raw[index:index+1] if index<2 else torch.zeros_like(x)
    rows,complete=evaluate(TextLookup(),raw,raw,norm,contexts,cfg,tmp_path,lambda:1000)
    assert complete and len(rows)==16
    assert all(r['all_steps_known_exact'] for r in rows)
    correct=[r for r in rows if r['shift_m'] is None and r['text_mode']=='correct']
    assert max(r['free_xyz_mpjpe_m'] for r in correct)<1e-6
    assert all(r['free_xyz_mpjpe_m']>0 for r in rows if r['text_mode']=='wrong')
    shifted=[r for r in rows if r['shift_m'] is not None]
    assert len(shifted)==4
    assert all(r['overwrite_baseline']['neighborhood_displacement_m']==0 for r in shifted)
    # This oracle ignores controls and must not be mistaken for a responsive model.
    assert not summarize(rows,cfg,True,cfg['steps'])['shift_response_diagnostic_passed']
