import torch
from motion_valley.response_supervision import state_difference_loss


def test_difference_loss_correct_collapse_and_excludes_anchors():
    target=torch.zeros(3,64,266)
    for b,sign in [(0,-1),(2,1)]:
        target[b,40:57,4+17*3:4+18*3]=sign*.05
        target[b,48,4+21*3:4+22*3]=sign*.1
    known=torch.zeros_like(target,dtype=torch.bool); known[:,48,4+21*3:4+22*3]=True
    active=~known; std=torch.ones(266)
    assert state_difference_loss(target,target,std,active,known)==0
    collapsed=torch.zeros_like(target,requires_grad=True)
    loss=state_difference_loss(collapsed,target,std,active,known)
    torch.testing.assert_close(loss,torch.tensor(1.))
    loss.backward()
    assert collapsed.grad[known].abs().sum()==0
    assert collapsed.grad[:,40:57,4+17*3:4+18*3].abs().sum()>0
    assert state_difference_loss(collapsed,target,std,torch.zeros_like(active),known)==0


def test_difference_loss_ignores_inactive_and_unrelated_channels():
    target=torch.randn(3,64,266); active=torch.zeros_like(target,dtype=torch.bool)
    active[:,48,4+19*3:4+20*3]=True; known=torch.zeros_like(active)
    pred=target.clone(); pred[~active]+=100
    assert state_difference_loss(pred,target,torch.ones(266),active,known)==0
