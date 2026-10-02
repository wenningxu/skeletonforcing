import json
import torch
import pytest
from motion_valley.representation263 import Layout263,local_ric_mask
from motion_valley.representation266 import (Layout266,from263,to263,recover_xyz263,xyz266,
                                            position_mask,Normalizer266,redundancy_loss)
from motion_valley.flow_features import FeatureControls,field_step,rollout,teacher_state
from motion_valley.schedule import distance_field
from motion_valley.model266 import Motion266Flow
from motion_valley.overfit import training_prediction,approval_check
from motion_valley.diagnostics266 import consistency_metrics


def setup():
    torch.manual_seed(77)
    raw=torch.randn(2,8,263)*.1
    raw[...,3]=1
    clean=from263(raw)
    joints=torch.zeros(2,8,22,dtype=torch.bool)
    joints[:,3,20]=True; joints[:,5,0]=True
    known=position_mask(joints)
    c=FeatureControls(torch.where(known,clean,0),known,joints)
    return clean,c


def test_lossless_all_channels_and_redundant_features_preserved():
    for size,cls in [(263,Layout263),(266,Layout266)]:
        a=torch.randn(2,8,size); layout=cls()
        assert torch.equal(a,layout.unpack(layout.pack(a)))
    original=torch.randn(2,8,263)
    expanded=from263(original)
    assert torch.equal(expanded[...,:4],original[...,:4])
    assert torch.equal(expanded[...,70:],original[...,67:])


def test_export_recovers_generated_world_positions_and_keeps_rotations():
    x,_=setup()
    # Include a moved root and changed world-space wrist.
    x[...,4:70]+=torch.randn_like(x[...,4:70])*.01
    restored=recover_xyz263(to263(x)); reference=xyz266(x)
    origin=reference[:,0:1,0:1,:].clone(); origin[...,1]=0
    torch.testing.assert_close(restored,reference-origin,atol=2e-6,rtol=2e-5)
    assert torch.equal(to263(x)[...,67:193],x[...,70:196])


def test_observing_xyz_never_reveals_velocity_rotation_contact():
    clean,c=setup(); d=distance_field(c.joints)
    a,b=field_step(0,32,c,d)
    assert (a[c.known]==0).all() and (a[~c.known]==1).all()
    assert c.known.sum()==2*2*3
    assert not c.known[...,70:].any()
    # Native263 local masks explicitly reject root XYZ instead of lying.
    with pytest.raises(ValueError): local_ric_mask(c.joints)


def test_exact_shared_field_and_oracle_rollout():
    clean,c=setup(); noise=torch.randn_like(clean); d=distance_field(c.joints)
    class Oracle:
        def __call__(self,*args): return clean
    seen=[]
    out=rollout(Oracle(),noise,c,None,None,None,steps=8,
                callback=lambda k,x: seen.append(torch.equal(x[c.known],clean[c.known])))
    assert len(seen)==9 and all(seen)
    torch.testing.assert_close(out,clean,atol=1e-6,rtol=1e-5)
    for k in range(8):
        _,train,next_train=teacher_state(clean,noise,k,8,c)
        infer,next_infer=field_step(k,8,c,d)
        assert torch.equal(train,infer) and torch.equal(next_train,next_infer)


def test_rollout_consumes_own_previous_state_and_truncated_gradient():
    clean,c=setup(); noise=torch.randn_like(clean)
    class Recorder(torch.nn.Module):
        def __init__(self): super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(.2)); self.inputs=[]
        def forward(self,x,*args): self.inputs.append(x.detach().clone()); return x*self.scale
    net=Recorder(); valid=torch.ones(2,8,dtype=torch.bool)
    pred,a,b,state=training_prediction(net,clean,noise,c,None,None,valid,4,8,True,2)
    assert len(net.inputs)==5
    torch.testing.assert_close(net.inputs[-1],state.detach())
    teacher,_,_=teacher_state(clean,noise,4,8,c)
    assert not torch.allclose(state,teacher)
    pred.square().mean().backward()
    assert net.scale.grad is not None and torch.isfinite(net.scale.grad)
    assert torch.equal(state[c.known],clean[c.known])


def test_full_umt5_token_cross_attention_and_padding():
    clean,c=setup(); valid=torch.ones(2,8,dtype=torch.bool)
    model=Motion266Flow(width=32,depth=1,heads=4).eval()
    text=torch.randn(2,5,4096); text_valid=torch.tensor([[1,1,1,0,0],[1,1,1,0,0]],dtype=torch.bool)
    sigma=torch.ones_like(clean)
    a=model(clean,sigma,c.known,text,text_valid,valid)
    text[:,3:]=1e4
    b=model(clean,sigma,c.known,text,text_valid,valid)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    text[:,0]+=3
    changed=model(clean,sigma,c.known,text,text_valid,valid)
    assert not torch.allclose(a,changed)
    assert a.shape==clean.shape


def test_redundancy_loss_differentiates_and_approval_fails_closed(tmp_path):
    x,_=setup(); x.requires_grad_(); valid=torch.ones(2,8,dtype=torch.bool)
    loss=redundancy_loss(x,valid); loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    config=tmp_path/'config.json'; config.write_text('{}')
    approval=tmp_path/'approval.json'; approval.write_text(json.dumps({'approved':False}))
    with pytest.raises(PermissionError): approval_check(approval,config)


def test_rotation_diagnostics_compare_against_gt_floor():
    x,_=setup(); diagnostics=consistency_metrics(x,x)
    assert diagnostics['bone_length_mae_m']==0
    assert diagnostics['motion_velocity_error_m_frame']==0
    assert diagnostics['fk_position_disagreement_m']==diagnostics['gt_fk_position_disagreement_m']
    assert diagnostics['degenerate_rotation_count']==0
