import torch
from motion_valley.schedule import distance_field, noise_levels, corrupt, sample, trajectory_step
from motion_valley.model import JointTimeFlow


def setup():
    clean = torch.randn(2,12,22,3)
    mask = torch.zeros(2,12,22,dtype=torch.bool)
    mask[0,6,20] = True
    mask[1,0,0] = True
    return clean,mask


def test_endpoints_monotonic_front():
    clean,m = setup(); d = distance_field(m)
    prev = noise_levels(0,m,d)
    assert (prev[~m] == 1).all() and (prev[m] == 0).all()
    for p in torch.linspace(0,1,101):
        s = noise_levels(p,m,d)
        assert (s <= prev+1e-6).all() and (s[m] == 0).all()
        prev = s
    assert (prev == 0).all()
    assert d[0,6,18] < d[0,6,0]  # wrist's parent is closer than pelvis


def test_training_fields_exactly_equal_sampler_lattice():
    _,m=setup(); d=distance_field(m)
    for mode in ['uniform','temporal','valley']:
        for step in range(32):
            train,train_next=trajectory_step(torch.tensor([step,step]),32,m,d,mode)
            infer,infer_next=trajectory_step(step,32,m,d,mode)
            assert torch.equal(train,infer) and torch.equal(train_next,infer_next)
            assert (train[m]==0).all() and (train_next[m]==0).all()


def test_clean_anchors_every_step_and_oracle_transport():
    clean,m = setup(); noise = torch.randn_like(clean)
    class Oracle:
        def __call__(self,*args): return noise-clean
    seen=[]
    out=sample(Oracle(),clean,m,None,None,steps=13,initial_noise=noise,
               callback=lambda k,x: seen.append(torch.equal(x[m],clean[m])))
    assert all(seen) and len(seen)==14
    torch.testing.assert_close(out,clean,atol=2e-6,rtol=1e-5)
    s=noise_levels(.4,m,distance_field(m))
    assert torch.equal(corrupt(clean,noise,s,m)[m],clean[m])


def test_no_anchors_and_all_anchors():
    clean,m=setup()
    for m in [torch.zeros_like(m),torch.ones_like(m)]:
        d=distance_field(m)
        assert torch.isfinite(d).all()
        assert torch.isfinite(noise_levels(.5,m,d)).all()


def test_padding_does_not_change_valid_predictions():
    torch.manual_seed(8)
    net=JointTimeFlow(width=32,depth=1).eval()
    torch.nn.init.normal_(net.output[-1].weight)
    x,m=setup(); valid=torch.ones(2,12,dtype=torch.bool); valid[:,-3:]=False
    s=torch.rand(2,12,22); text=torch.randn(2,12,300)
    a=net(x,s,m,text,valid)
    x[:,-3:]=1000
    b=net(x,s,m,text,valid)
    torch.testing.assert_close(a[:,:9],b[:,:9],atol=1e-5,rtol=1e-5)


def test_backward_and_sampler():
    clean,m=setup(); net=JointTimeFlow(width=32,depth=1)
    text=torch.randn(2,12,300); valid=torch.ones(2,12,dtype=torch.bool)
    s=noise_levels(.6,m,distance_field(m)); noise=torch.randn_like(clean)
    pred=net(corrupt(clean,noise,s,m),s,m,text,valid)
    loss=(pred[~m]-(noise-clean)[~m]).square().mean(); loss.backward()
    assert torch.isfinite(loss) and net.output[-1].weight.grad.abs().sum()>0
    out=sample(net,clean,m,text,valid,steps=3)
    assert torch.equal(out[m],clean[m]) and torch.isfinite(out).all()
