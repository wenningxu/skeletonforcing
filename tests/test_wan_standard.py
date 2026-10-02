"""Upstream numerical parity, mask isolation and real rollout integration."""
import ast
import json
import math
from pathlib import Path
import warnings
import pytest
import torch
from torch import nn
from motion_valley.wan_components import (WanAttentionBlock,WanHead,grid_rope,
                                          sinusoidal_embedding,apply_rope)
from motion_valley.model_wan266 import WanMotion266
from motion_valley.model_factory import build_control_model
from motion_valley.flow_features import FeatureControls,rollout,field_step
from motion_valley.representation266 import position_mask
from motion_valley.schedule import distance_field
from motion_valley.overfit import training_prediction

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def upstream_classes():
    """Load unmodified class bodies from the checked-in upstream source.

    Only the unavailable FlashAttention CUDA backend is substituted by an
    independent explicit softmax/matmul CPU oracle. No package monkeypatching.
    """
    file=ROOT/'third_party/FloodDiffusion/models/tools/wan_model.py'
    selected={'sinusoidal_embedding_1d','rope_params','rope_apply','WanRMSNorm',
              'WanLayerNorm','WanSelfAttention','WanCrossAttention','WanAttentionBlock','Head'}
    tree=ast.parse(file.read_text(encoding='utf-8'))
    module=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in selected],type_ignores=[])
    def reference_attention(q,k,v,k_lens=None,**kwargs):
        scores=torch.einsum('bqhd,bkhd->bhqk',q,k)/q.shape[-1]**.5
        if k_lens is not None:
            valid=torch.arange(k.shape[1])[None]<k_lens[:,None]
            scores=scores.masked_fill(~valid[:,None,None],-torch.inf)
        return torch.einsum('bhqk,bkhd->bqhd',scores.softmax(-1),v)
    namespace=dict(torch=torch,nn=nn,math=math,flash_attention=reference_attention)
    exec(compile(module,str(file),'exec'),namespace)
    return namespace


def small(text_enabled=True,depth=2):
    return WanMotion266(width=32,depth=depth,heads=4,ffn_dim=64,freq_dim=16,
                        text_dim=24,text_enabled=text_enabled)


def inputs():
    x=torch.randn(2,4,266); sigma=torch.full_like(x,.6)
    anchors=torch.zeros(2,4,22,dtype=torch.bool); anchors[:,1,21]=True
    known=position_mask(anchors); sigma[known]=0
    text=torch.randn(2,5,24); text_valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    valid=torch.tensor([[1,1,1,1],[1,1,1,0]],dtype=torch.bool)
    return x,sigma,known,text,text_valid,valid


def open_head(net):
    # Wan intentionally initializes head weights to zero. Nonzero test weights
    # are needed to exercise dependencies beyond the head, not to claim learning.
    with torch.no_grad(): nn.init.normal_(net.head.head.weight,std=.03)


def test_wan_block_and_head_match_upstream_forward_and_gradients():
    torch.manual_seed(19); upstream=upstream_classes()
    ref=upstream['WanAttentionBlock'](32,64,4,cross_attn_norm=True)
    net=WanAttentionBlock(32,64,4); net.load_state_dict(ref.state_dict(),strict=True)
    grids=torch.tensor([[3,2,1],[2,2,1]]); lens=torch.tensor([6,4]); lengths=torch.tensor([5,3])
    coords=torch.tensor([[t,j,0] for t in range(3) for j in range(2)])
    freq=grid_rope(coords,8)
    reference_freq=torch.cat([upstream['rope_params'](16,4),upstream['rope_params'](16,2),upstream['rope_params'](16,2)],1)
    valid=torch.arange(6)[None]<lens[:,None]
    x=torch.randn(2,6,32,requires_grad=True); e=torch.randn(2,6,6,32,requires_grad=True)
    context=torch.randn(2,5,32,requires_grad=True)
    other=[v.detach().clone().requires_grad_() for v in [x,e,context]]
    allowed=(torch.arange(5)[None]<lengths[:,None])[:,None].expand(2,6,5)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='User provided device_type of')
        expected=ref(x,e,lens,grids,reference_freq,context,lengths)
    actual=net(other[0],other[1],valid,freq,other[2],allowed)
    torch.testing.assert_close(actual[valid],expected[valid],atol=3e-6,rtol=3e-5)
    expected[valid].square().mean().backward(); actual[valid].square().mean().backward()
    for a,b in zip([x,e,context],other): torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=4e-5)
    for (an,a),(bn,b) in zip(ref.named_parameters(),net.named_parameters()):
        assert an==bn
        torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=4e-5)
    ref_head=upstream['Head'](32,13,(1,1,1)); head=WanHead(32,13)
    head.load_state_dict(ref_head.state_dict(),strict=True)
    hx=torch.randn(2,6,32); he=torch.randn(2,6,32)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='User provided device_type of')
        expected=ref_head(hx,he)
    torch.testing.assert_close(head(hx,he),expected,atol=1e-6,rtol=1e-6)


def test_rotary_and_sinusoidal_encoding_match_upstream():
    upstream=upstream_classes(); d=32
    coords=torch.tensor([[t,j,0] for t in range(4) for j in range(3)])
    sizes=[d-4*(d//6),2*(d//6),2*(d//6)]
    reference_freq=torch.cat([upstream['rope_params'](32,s) for s in sizes],1)
    x=torch.randn(2,12,2,d)
    torch.testing.assert_close(apply_rope(x,grid_rope(coords,d)),
        upstream['rope_apply'](x,torch.tensor([[4,3,1],[4,3,1]]),reference_freq),atol=0,rtol=0)
    times=torch.tensor([0.,.1,.9,1.])
    torch.testing.assert_close(sinusoidal_embedding(16,times),
        upstream['sinusoidal_embedding_1d'](16,times).float(),atol=0,rtol=0)


def test_time_vector_preserves_channel_identity_and_reaches_every_block():
    torch.manual_seed(23); net=small(depth=3); open_head(net)
    x,sigma,known,text,text_valid,valid=inputs()
    a=torch.full_like(sigma,.5); b=a.clone()
    # Swap noise levels between a joint's XYZ and rotation channels: equal mean.
    a[...,7]=0.; a[...,70]=1.; b[...,7]=1.; b[...,70]=0.
    assert torch.equal(net.layout.pack(a).sum(-1),net.layout.pack(b).sum(-1))
    ea,_=net.time_condition(a); eb,_=net.time_condition(b)
    assert (ea[:,:,1]-eb[:,:,1]).abs().max()>1e-5
    per_layer=[]
    def capture(module,args):
        changed=list(args); changed[1]=args[1].clone(); changed[1].retain_grad()
        per_layer.append(changed[1]); return tuple(changed)
    handles=[block.register_forward_pre_hook(capture) for block in net.blocks]
    x.requires_grad_(); result=net(x,sigma,known,text,text_valid,valid)
    result[(~known)&valid[...,None]].square().mean().backward()
    for handle in handles: handle.remove()
    assert len(per_layer)==3
    assert all(value.grad is not None and value.grad.abs().sum()>0 for value in per_layer)
    assert x.grad[known].abs().sum()>0  # Free output has a differentiable path from clean anchors.


def test_padding_and_text_alignment_are_enforced():
    torch.manual_seed(29); net=small(depth=1); open_head(net)
    args=list(inputs()); x,sigma,known,text,text_valid,valid=args
    alignment=torch.zeros(2,4,5,dtype=torch.bool)
    alignment[:,:,0]=True
    base=net(*args,frame_text_allowed=alignment)
    # Text tokens disallowed for every frame must be irrelevant, including NaN padding.
    dirty=text.clone(); dirty[:,1:]=10000.; dirty[~text_valid]=float('nan')
    dirty_x=x.clone(); dirty_x[~valid]=float('nan')
    actual=net(dirty_x,sigma,known,dirty,text_valid,valid,alignment)
    torch.testing.assert_close(actual,base,atol=0,rtol=0)
    assert torch.equal(actual[~valid],torch.zeros_like(actual[~valid]))
    with pytest.raises(ValueError): net(*args,frame_text_allowed=torch.zeros_like(alignment))


def test_no_text_pruning_initialization_and_dispatch():
    torch.manual_seed(37); full=small(); torch.manual_seed(37); empty=small(False)
    assert not any('text' in n or 'cross_attn' in n for n,_ in empty.named_parameters())
    state=full.state_dict()
    for name,value in empty.state_dict().items(): assert torch.equal(value,state[name])
    x,sigma,known,text,text_valid,valid=inputs()
    assert torch.equal(empty(x,sigma,known,None,None,valid),torch.zeros_like(x))
    with pytest.raises(ValueError): empty(x,sigma,known,text,text_valid,valid)
    cfg=json.loads((ROOT/'configs/control_no_text_v4.json').read_text())
    cfg.update(width=32,depth=1,heads=4)
    assert type(build_control_model(cfg)).__name__=='NoTextMotion266Flow'
    cfg.update(architecture='wan_joint_time_v1',ffn_dim=64,freq_dim=16,time_embedding_scale=1.)
    assert isinstance(build_control_model(cfg),WanMotion266)
    cfg['architecture']='typo'
    with pytest.raises(ValueError): build_control_model(cfg)


def test_exact_fields_clean_anchor_rollout_and_training_bf16_cpu():
    torch.manual_seed(41); net=small(False,depth=2)
    x,_,known,_,_,valid=inputs(); valid[:]=True
    joints=known[...,4:70].reshape(2,4,22,3).all(-1)
    controls=FeatureControls(torch.where(known,x,0),known,joints)
    noise=torch.randn_like(x); checked=[]
    def check(k,state):
        assert torch.isfinite(state).all()
        assert torch.equal(state[known],x[known]); checked.append(k)
    # Exercise the Wan zero-head first step, then verify gradient flow beyond it.
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            prediction,s,nxt,_=training_prediction(net,x,noise,controls,None,None,valid,2,4,True,2)
            expected=field_step(2,4,controls,distance_field(joints))
            assert torch.equal(s,expected[0]) and torch.equal(nxt,expected[1])
            loss=(prediction.float()-x).square()[(nxt<s)&~known].mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    assert net.time_embedding[0].weight.grad.abs().sum()>0
    assert net.blocks[0].self_attn.q.weight.grad.abs().sum()>0
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        result=rollout(net,noise,controls,None,None,valid,steps=4,callback=check)
    assert checked==list(range(5)) and result.shape==x.shape
    with pytest.raises(ValueError): net(x,torch.ones_like(x),known,None,None,valid)
