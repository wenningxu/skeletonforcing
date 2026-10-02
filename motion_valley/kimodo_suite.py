"""One approval-gated Kimodo266 training run and the frozen common evaluator."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np
from .overfit import approval_check
from .control_factors import evaluate_factor
from .model_factory import build_control_model
from .representation266 import Normalizer266


def main():
    config='configs/kimodo266_overfit_v1.json'; approval='approvals/of008-kimodo266.json'
    approval_check(approval,config)
    cfg=json.loads(Path(config).read_text())
    check_path=Path('reports/local_checks_kimodo266.json')
    if hashlib.sha256(check_path.read_bytes()).hexdigest()!=cfg['component_checks_sha256']:
        raise RuntimeError('Component audit changed')
    checks=json.loads(check_path.read_text())
    for name,sha in checks['official_files'].items():
        path=Path('third_party/kimodo/kimodo')/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise RuntimeError('Upstream component changed')
    for name,sha in checks['local_files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Local component changed')
    plan_path=Path('configs/control_factors_v1.json')
    if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=cfg['evaluation_plan_sha256']:
        raise RuntimeError('Common evaluation plan changed')
    plan=json.loads(plan_path.read_text())
    reference_path=Path(plan['baseline_checkpoint'])
    if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=plan['baseline_checkpoint_sha256']:
        raise RuntimeError('Reference checkpoint changed')
    baseline=torch.load(reference_path,map_location='cpu',weights_only=False)
    fixed_norm={k:v.clone() for k,v in baseline['normalizer'].items()};del baseline
    root=Path('/workspace/outputs/OF008')
    if root.exists():raise RuntimeError('OF008 already exists; refuse duplicate run')
    deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00'))
    remaining=lambda:(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining()<900:raise RuntimeError('Insufficient allocation time')
    root.mkdir(parents=True)
    (root/'source_checks.json').write_text(json.dumps(checks,indent=2))
    subprocess.run([sys.executable,'-m','motion_valley.control_experiment','--config',config,
        '--approval-file',approval,'--data','/workspace/data','--output',str(root/'train')],check=True)
    cp=torch.load(root/'train/last.pt',map_location='cpu',weights_only=False)
    assert cp['step']==cfg['steps'] and cp['config']==cfg
    assert all(torch.equal(cp['normalizer'][k],v) for k,v in fixed_norm.items())
    model=build_control_model(cfg).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
    norm=Normalizer266(cp['normalizer']['mean'],cp['normalizer']['std']).cuda()
    with np.load(cfg['bank_file']) as bank:states=torch.from_numpy(bank['states']).cuda();angles=bank['angles'].tolist()
    evaluate_factor(model=model,normalizer=norm,states=states,angles=angles,plan=plan,
                    out=root/'common_eval',remaining=remaining)
    (root/'SUITE_COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__=='__main__':main()
