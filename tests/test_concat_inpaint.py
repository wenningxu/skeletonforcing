import torch
from motion_valley.model_wan266 import WanMotion266


def test_explicit_concat_matches_split_projection_and_input_gradient():
    torch.manual_seed(51)
    a=WanMotion266(width=32,depth=2,heads=4,ffn_dim=64,freq_dim=16,text_enabled=False)
    b=WanMotion266(width=32,depth=2,heads=4,ffn_dim=64,freq_dim=16,text_enabled=False,input_mode='concat_mask')
    with torch.no_grad(): a.head.head.weight.normal_(std=.03)
    b.load_state_dict(a.state_dict(),strict=True)
    x=torch.randn(1,4,266,requires_grad=True); y=x.detach().clone().requires_grad_(True)
    known=torch.zeros_like(x,dtype=torch.bool); known[:,2,4:7]=True
    sigma=torch.full_like(x,.6).masked_fill(known,0); valid=torch.ones(1,4,dtype=torch.bool)
    out_a=a(x,sigma,known,None,None,valid); out_b=b(y,sigma,known,None,None,valid)
    torch.testing.assert_close(out_a,out_b,atol=2e-6,rtol=3e-5)
    out_a.square().mean().backward(); out_b.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-7,rtol=1e-4)
    for name in ['motion_embedding','observed_embedding']:
        torch.testing.assert_close(getattr(a,name).weight.grad,getattr(b,name).weight.grad,atol=1e-7,rtol=1e-4)


def test_independent_evaluator_uses_configured_uniform_field(tmp_path):
    import numpy as np
    from pathlib import Path
    from motion_valley.control_diagnosis import independent_evaluate
    from motion_valley.representation266 import Normalizer266
    with np.load(Path(__file__).resolve().parents[1]/'prepared/OF003/states.npz') as bank:
        states=torch.from_numpy(bank['states'][0]); angles=bank['angles'].tolist()
    class CheckUniform(torch.nn.Module):
        def forward(self,x,sigma,known,*args):
            unknown=sigma[~known]
            assert torch.all(unknown==unknown[0])
            assert torch.all(sigma[known]==0)
            return states[angles.index(0)][None].expand_as(x)
    cfg=dict(layouts={'A':[[16,0]],'D':[[10,0]]},mode='uniform',sampling_steps=2,independent_noise_samples=1)
    (tmp_path/'samples').mkdir()
    independent_evaluate(CheckUniform(),states,angles,Normalizer266(torch.zeros(266),torch.ones(266)),cfg,tmp_path,lambda:9999)
    assert len(list((tmp_path/'samples').glob('*.npz')))==10
