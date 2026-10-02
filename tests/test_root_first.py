import pytest
import torch
from motion_valley.root_first import (RootFirstSchedule,joint_depths,advance,rollout,
                                     training_prediction,active_redundancy_loss)
from motion_valley.schedule import PARENTS,distance_field
from motion_valley.representation266 import Layout266,Normalizer266
from motion_valley.model_wan266 import WanMotion266


def test_root_first_complete_lattice_and_feature_ownership():
    schedule=RootFirstSchedule();table=schedule.lattice(64);depths=joint_depths()
    assert schedule.total_steps(64)==53 and table.shape==(54,64,266)
    assert torch.equal(table[0],torch.ones(64,266)) and torch.equal(table[-1],torch.zeros(64,266))
    assert (table[1:]<=table[:-1]).all()
    layout=Layout266()
    for j in range(22):
        own=layout.owner==j
        assert torch.equal(table[...,own],table[...,4+3*j,None].expand_as(table[...,own]))
        if j:
            assert (table[...,4+3*PARENTS[j]]<=table[...,4+3*j]).all()
    assert (table[:,:-1]<=table[:,1:]).all(), 'Earlier frames must lead later frames'
    assert table[1,0,4]<1 and table[1,1,4]==1 and table[1,0,7]==1
    # Same distance: 20 frames == two bone edges with the original scales.
    assert torch.equal(table[:,20,4],table[:,0,4+3*4])
    for kwargs in [{'local_steps':0},{'delay_scale_steps':0},{'temporal_scale':0},{'spread':1}]:
        with pytest.raises(ValueError):RootFirstSchedule(**kwargs)
    counts=(table[1:]<table[:-1]).sum(0)
    assert torch.equal(counts,torch.full_like(counts,32))
    assert table[32,0,4]==0 and table[32,63,67]>0
    starts=schedule.start_steps(64)
    assert [int(starts[0,0]),int(starts[0,21]),int(starts[63,21])]==[0,19,21]


class Recorder(torch.nn.Module):
    def __init__(self,clean=None):super().__init__();self.clean=clean;self.inputs=[]
    def forward(self,x,sigma,known,text,text_valid,valid):
        assert not known.any() and text is None and text_valid is None
        self.inputs.append((x.detach().clone(),sigma.detach().clone()))
        return torch.full_like(x,.25) if self.clean is None else self.clean


def test_sampling_uses_model_parents_and_never_clean_observations():
    noise=torch.randn(1,3,266);valid=torch.ones(1,3,dtype=torch.bool);s=RootFirstSchedule()
    net=Recorder();states=[]
    result=rollout(net,noise,valid,s,callback=lambda k,x:states.append(x.clone()))
    assert torch.equal(net.inputs[0][0],noise)
    owner=Layout266().owner;root=owner==0
    assert torch.equal(states[1][:,1:],noise[:,1:])
    assert torch.equal(states[1][:,0,~root],noise[:,0,~root])
    assert not torch.equal(states[1][:,0,root],noise[:,0,root])
    assert torch.equal(states[1][:,0,root],noise[:,0,root]*(31/32)+.25/32)
    assert torch.equal(result,torch.full_like(result,.25))
    # A wrong model parent is retained even when a different GT is available
    # to the training loss. Changing GT cannot affect a rolled-out state.
    clean=torch.randn_like(noise)
    _,a,b,state=training_prediction(net,clean,noise,valid,3,s,True,0)
    _,_,_,other=training_prediction(net,clean+10,noise,valid,3,s,True,0)
    assert torch.equal(state,other)
    assert torch.equal(a,net.inputs[-1][1]) and (b<a).any()
    oracle=Recorder(clean);assert torch.equal(rollout(oracle,noise,valid,s),clean)


def test_teacher_and_sampler_share_every_field_and_freeze_inactive():
    s=RootFirstSchedule();x=torch.randn(2,3,266);noise=torch.randn_like(x)
    valid=torch.ones(2,3,dtype=torch.bool);net=Recorder()
    rollout(net,noise,valid,s)
    captured=[a for _,a in net.inputs]
    for k in range(s.total_steps(x.shape[1])):
        p,a,b,state=training_prediction(net,x,noise,valid,k,s)
        assert torch.equal(captured[k],a)
        assert torch.equal(state,(1-a)*x+a*noise)
        y=advance(state,p,a,b);inactive=b==a
        assert torch.equal(y[inactive],state[inactive])


def test_wan_root_first_bf16_backward_and_inactive_loss_gradients():
    torch.set_num_threads(2);torch.manual_seed(9009)
    net=WanMotion266(width=32,depth=2,heads=4,ffn_dim=64,freq_dim=16,text_enabled=False)
    torch.nn.init.normal_(net.head.head.weight,std=.01)
    clean=torch.randn(1,8,266);noise=torch.randn_like(clean);valid=torch.ones(1,8,dtype=torch.bool)
    s=RootFirstSchedule()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        pred,a,b,state=training_prediction(net,clean,noise,valid,5,s,True,2)
        loss=(pred.float()-clean).square()[b<a].mean()
    loss.backward()
    assert torch.isfinite(loss) and net.time_projection[1].weight.grad.abs().sum()>0
    raw=torch.randn(1,8,266,requires_grad=True);active=torch.zeros_like(raw,dtype=torch.bool)
    active[...,4:7]=True;active[...,0:4]=True;active[...,196:199]=True
    consistency=active_redundancy_loss(raw,valid,active);consistency.backward()
    assert torch.isfinite(consistency) and raw.grad[...,199:262].abs().sum()==0


def test_temporal_front_excludes_unstarted_velocity_endpoints():
    raw=torch.randn(1,4,266,requires_grad=True);valid=torch.ones(1,4,dtype=torch.bool)
    s=RootFirstSchedule();a,b=s.step(0,raw);active=b<a
    loss=active_redundancy_loss(raw,valid,active,b<1);loss.backward()
    assert raw.grad[:,1:].abs().sum()==0
    assert raw.grad[...,196:262].abs().sum()==0
    assert raw.grad[...,1:3].abs().sum()==0
    assert raw.grad[:,0,3].abs().sum()>0


@pytest.mark.parametrize('frames,local_steps',[(1,8),(3,32),(64,32),(101,7)])
def test_every_joint_frame_traverses_the_identical_local_noise_sequence(frames,local_steps):
    s=RootFirstSchedule(local_steps=local_steps);table=s.lattice(frames)
    starts=s.start_steps(frames);curve=1-torch.arange(local_steps+1).float()/local_steps
    for f in range(frames):
        for j in range(22):
            start=int(starts[f,j]);channel=4+3*j
            assert torch.equal(table[start:start+local_steps+1,f,channel],curve)
            assert (table[:start+1,f,channel]==1).all()
            assert (table[start+local_steps:,f,channel]==0).all()
    assert ((table[1:]<table[:-1]).sum(0)==local_steps).all()


def test_completed_model_output_is_frozen_while_other_nodes_keep_denoising():
    class ChangingRecorder(Recorder):
        def forward(self,*args):
            x=super().forward(*args)
            return torch.full_like(x,len(self.inputs)/100)
    x=torch.randn(1,3,266);valid=torch.ones(1,3,dtype=torch.bool);s=RootFirstSchedule()
    states=[];net=ChangingRecorder()
    rollout(net,x,valid,s,callback=lambda k,z:states.append(z.clone()))
    root=Layout266().owner==0
    assert torch.equal(states[32][:,0,root],torch.full_like(states[32][:,0,root],.32))
    for state in states[32:]:assert torch.equal(state[:,0,root],states[32][:,0,root])
    assert not torch.equal(states[-1][:,0,67:70],states[32][:,0,67:70])
