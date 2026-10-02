"""Train OF006-inpaint once, then independently sample its final checkpoint."""
import json,os,subprocess,sys,time
from pathlib import Path
import datetime
import numpy as np
import torch
from .overfit import approval_check
from .model_factory import build_control_model
from .representation266 import Normalizer266
from .control_diagnosis import independent_evaluate


def main():
    cfgpath='configs/inpaint_baseline_v1.json'; approval='approvals/of006-inpaint.json'
    approval_check(approval,cfgpath)
    cfg=json.loads(Path(cfgpath).read_text()); out=Path('/workspace/outputs/OF006-inpaint')
    if out.exists(): raise FileExistsError('Do not relaunch an existing experiment')
    subprocess.run([sys.executable,'-m','motion_valley.control_experiment','--config',cfgpath,
        '--approval-file',approval,'--data','/workspace/data','--output',str(out)],check=True)
    checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    if checkpoint['step']!=4000 or checkpoint['config']!=cfg: raise ValueError('Unexpected checkpoint')
    model=build_control_model(cfg); model.load_state_dict(checkpoint['model'],strict=True)
    normalizer=Normalizer266(checkpoint['normalizer']['mean'],checkpoint['normalizer']['std']).cuda()
    del checkpoint
    model=model.cuda().eval().requires_grad_(False); torch.backends.cuda.matmul.allow_tf32=True
    with np.load(cfg['bank_file']) as bank:
        states=torch.from_numpy(bank['states'][0]).cuda(); angles=bank['angles'].tolist()
    deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00')).timestamp()
    independent_evaluate(model,states,angles,normalizer,cfg,out,lambda:deadline-time.time())
    (out/'SUITE_COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    print('OF006-inpaint: 4000 steps,40 paired cases,160 independent cases complete',flush=True)


if __name__=='__main__': main()
