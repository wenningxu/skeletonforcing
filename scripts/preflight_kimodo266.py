"""CPU-only preparation. Does not provision or execute a paid experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_valley.model_factory import build_control_model
from motion_valley.model_kimodo266 import Kimodo266
from motion_valley.flow_features import FeatureControls,rollout
from motion_valley.representation266 import position_mask

torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/inpaint_baseline_v1.json').read_text())
cfg.update(experiment='OF008-kimodo266',architecture='kimodo_frame_twostage_266_v1',depth=16,
           register_tokens=50,normalizer_bank_clip_indices=[0],max_gpu_minutes=75,
           max_total_usd=2.,evaluation_reserve_seconds=300,heldout_noise_seeds=list(range(2026,2034)))
cfg.pop('independent_noise_samples',None)
with torch.device('meta'):
    main=build_control_model(cfg)
parameters=sum(p.numel() for p in main.parameters())
del main
torch.manual_seed(808)
model=Kimodo266(width=32,depth=2,heads=4,ffn_dim=64,sampling_steps=32).train()
x=torch.randn(3,64,266)
joints=torch.zeros(3,64,22,dtype=torch.bool)
for frame,joint in cfg['layouts']['A']:joints[:,frame,joint]=True
known=position_mask(joints);controls=FeatureControls(torch.where(known,x,0),known,joints)
valid=torch.ones(3,64,dtype=torch.bool)
with torch.autocast('cpu',dtype=torch.bfloat16):
    result=rollout(model,torch.randn_like(x),controls,None,None,valid,steps=32,
                   stop_step=4,grad_last=2,mode='uniform')
loss=(result[~known]-x[~known]).square().mean();loss.backward()
assert torch.equal(result[known],x[known]) and torch.isfinite(result).all()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
upstream=ROOT/'third_party/kimodo/kimodo'
files=['model/backbone.py','model/twostage_denoiser.py','tools.py','model/loading.py','model/registry.py']
record=dict(parameters=parameters,main_model_meta_only=True,small_model_cpu_bf16_backward=True,
    shape=list(x.shape),known_xyz_only=True,full_sampling_lattice_steps=32,
    smoke_rollout_executed_steps=4,smoke_gradient_steps=2,
    upstream=json.loads((ROOT/'reports/kimodo_upstream.json').read_text()),
    official_files={f:hashlib.sha256((upstream/f).read_bytes()).hexdigest() for f in files},
    local_files={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in [
        'motion_valley/model_kimodo266.py','motion_valley/kimodo_components.py',
        'motion_valley/model_factory.py','motion_valley/kimodo_suite.py',
        'motion_valley/control_experiment.py','motion_valley/control_factors.py',
        'motion_valley/flow_features.py','scripts/run_of008.sh']},
    dependencies={n:importlib.metadata.version(n) for n in ['torch','omegaconf','pydantic','einops','hydra-core','safetensors']})
(ROOT/'reports/local_checks_kimodo266.json').write_text(json.dumps(record,indent=2)+'\n')
cfg['component_checks_sha256']=hashlib.sha256((ROOT/'reports/local_checks_kimodo266.json').read_bytes()).hexdigest()
cfg['evaluation_plan_sha256']=hashlib.sha256((ROOT/'configs/control_factors_v1.json').read_bytes()).hexdigest()
(ROOT/'configs/kimodo266_overfit_v1.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps(record,indent=2))
