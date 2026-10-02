import torch
import pytest
from motion_valley.model_kimodo266 import Kimodo266
from motion_valley.flow_features import FeatureControls, rollout
from motion_valley.representation266 import position_mask


def fixture():
    torch.manual_seed(808)
    model = Kimodo266(width=32, depth=2, heads=4, ffn_dim=64, sampling_steps=4)
    x = torch.randn(2, 5, 266)
    joints = torch.zeros(2, 5, 22, dtype=torch.bool); joints[:, 2, 21] = True
    mask = position_mask(joints)
    valid = torch.ones(2, 5, dtype=torch.bool); valid[1, 4] = False
    sigma = torch.full_like(x, .5).masked_fill(mask, 0)
    return model, x, sigma, mask, valid, joints


def test_official_forward_parity_and_position_only_injection():
    model,x,sigma,mask,valid,_ = fixture(); model.eval()
    observed = {}
    def capture(module,args): observed['root_input'] = args[0].detach().clone()
    hook = model.backbone.root_model.register_forward_pre_hook(capture)
    a = model(x,sigma,mask,None,None,valid)
    live = valid[...,None].expand_as(x); clean_input=x.masked_fill(~live,0)
    b = model.backbone(clean_input,valid,torch.zeros(2,1,4096),torch.ones(2,1,dtype=torch.bool),
        torch.tensor([500,500]),motion_mask=mask.float(),observed_motion=torch.where(mask,clean_input,0))
    hook.remove()
    torch.testing.assert_close(a,b.masked_fill(~live,0),rtol=0,atol=0)
    torch.testing.assert_close(observed['root_input'][...,:266],clean_input)
    assert torch.equal(observed['root_input'][...,266:].bool(),mask)
    assert not mask[...,:4].any() and not mask[...,70:].any()
    # Padding never contaminates valid frames.
    poisoned=x.clone(); poisoned[~live]=float('nan')
    torch.testing.assert_close(model(poisoned,sigma,mask,None,None,valid),a)


def test_two_stages_train_and_official_detached_root_handoff():
    model,x,sigma,mask,valid,_ = fixture(); model.train()
    out=model(x,sigma,mask,None,None,valid)
    out[...,7:].square().mean().backward()
    root_grads=[p.grad for p in model.backbone.root_model.parameters()]
    assert all(g is None or torch.count_nonzero(g)==0 for g in root_grads)
    assert model.backbone.body_model.input_linear.weight.grad.norm()>0
    model.zero_grad(set_to_none=True)
    out=model(x,sigma,mask,None,None,valid)
    out[...,:7].square().mean().backward()
    assert model.backbone.root_model.input_linear.weight.grad.norm()>0


def test_reject_valley_field_and_text_instead_of_silent_approximation():
    model,x,sigma,mask,valid,_=fixture()
    sigma[:,0,90]=.25
    with pytest.raises(ValueError,match='uniform'):model(x,sigma,mask,None,None,valid)
    sigma[:,0,90]=.5
    with pytest.raises(ValueError,match='text'):model(x,sigma,mask,torch.zeros(1),None,valid)


def test_real_model_rollout_keeps_anchors_clean_and_has_control_path():
    model,x,_,mask,valid,joints=fixture(); model.eval()
    controls=FeatureControls(torch.where(mask,x,0),mask,joints)
    checked=[]
    def check(k,state):
        assert torch.isfinite(state).all()
        assert torch.equal(state[mask],x[mask]); checked.append(k)
    result=rollout(model,torch.randn_like(x),controls,None,None,valid,steps=4,mode='uniform',callback=check)
    assert checked==list(range(5)) and result.shape==x.shape
    # Nonzero untrained input sensitivity proves a graph path, not learned control.
    x=x.requires_grad_(); sigma=torch.full_like(x,.5).masked_fill(mask,0)
    prediction=model(x,sigma,mask,None,None,valid)
    prediction[:,1,67:70].sum().backward()
    assert x.grad[mask].norm()>0
