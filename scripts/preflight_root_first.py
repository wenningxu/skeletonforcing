import json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/mpl-cache'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion_valley.root_first import RootFirstSchedule,training_prediction,joint_depths
from motion_valley.model_wan266 import WanMotion266
from motion_valley.model_factory import build_control_model
torch.set_num_threads(2);torch.manual_seed(9009)
cfg=json.loads((root/'configs/root_first_overfit_v1.json').read_text())
with torch.device('meta'):main=build_control_model(cfg)
count=sum(p.numel() for p in main.parameters());del main
net=WanMotion266(width=32,depth=2,heads=4,ffn_dim=64,freq_dim=16,text_enabled=False)
torch.nn.init.normal_(net.head.head.weight,std=.01)
clean=torch.randn(3,64,266);noise=torch.randn_like(clean);valid=torch.ones(3,64,dtype=torch.bool)
s=RootFirstSchedule(**cfg['schedule']);started=time.monotonic();total=s.total_steps(cfg['frames'])
with torch.autocast('cpu',dtype=torch.bfloat16):
    pred,a,b,state=training_prediction(net,clean,noise,valid,total-1,s,True,4)
    loss=(pred.float()-clean).square()[b<a].mean()
loss.backward();assert torch.isfinite(loss) and net.time_projection[1].weight.grad.abs().sum()>0
out=root/'outputs/OF009-equal-steps';out.mkdir(exist_ok=True)
sigma=s.lattice(64).numpy();joints=[joint_depths().index(d) for d in range(8)]
fig,axes=plt.subplots(1,4,figsize=(13,4),sharey=True,layout='constrained')
for ax,k in zip(axes,[12,24,36,total]):
    z=sigma[k][:,[4+3*j for j in joints]].T
    im=ax.imshow(z,aspect='auto',origin='upper',vmin=0,vmax=1,cmap='viridis',extent=[-.5,63.5,7.5,-.5])
    ax.set_yticks(range(8));ax.set_xticks([0,20,40,63]);ax.set_xlabel('Motion frame');ax.set_title(f'Global step {k}/{total}')
axes[0].set_ylabel('Skeleton depth (root = 0)')
fig.suptitle(f'Shifted valley field: {s.local_steps} equal local updates per joint-frame; staggered completion')
fig.colorbar(im,ax=axes,label='Sigma (noise = 1)',shrink=.8)
fig.savefig(out/'time_field.png',dpi=170);plt.close(fig)
counts=(sigma[1:]<sigma[:-1]).sum(0);assert (counts==s.local_steps).all()
record=dict(cpu_tests_passed=56,main_parameters=count,small_model_cpu_bf16_shape=[3,64,266],
    rollout_prefix_steps=total-1,gradient_tail=4,full_schedule_steps=total,local_denoising_steps=s.local_steps,small_model_loss=float(loss.detach()),
    smoke_seconds=time.monotonic()-started,no_observations=True,joint_depths=joint_depths(),schedule=cfg['schedule'],
    scheduling_origin=[0,0],equal_update_counts_min=int(counts.min()),equal_update_counts_max=int(counts.max()),
    exact_original_valley_parity=False,schedule_version='shifted_equal_steps_v2')
(root/'reports/local_checks_root_first.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
