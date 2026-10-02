"""OF007: isolated control density, data count, and layout-coverage interventions."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .overfit import approval_check,reconstruction_metrics
from .model_factory import build_control_model
from .representation266 import Normalizer266,xyz266,position_mask
from .multi_overfit import make_controls
from .control_states import coherent_response
from .flow_features import rollout


def common_known(layouts,frames=64,device='cpu'):
    """Exclude the union of every evaluation anchor, not just each arm's mask."""
    joints=torch.zeros(1,frames,22,dtype=torch.bool,device=device)
    for layout in layouts.values():
        if len(layout)!=len({tuple(x) for x in layout}):raise ValueError('Duplicate anchors')
        for t,j in layout:
            if not 0<=t<frames or not 0<=j<22:raise ValueError('Invalid anchor')
            joints[:,t,j]=True
    return position_mask(joints)


def evaluate_factor(model,normalizer,states,angles,plan,out,remaining):
    out.mkdir(); samples=out/'samples';samples.mkdir()
    device=states.device; valid=torch.ones(1,64,dtype=torch.bool,device=device)
    common=common_known(plan['eval_layouts'],device=device)
    np.save(out/'common_known.npy',common.cpu().numpy())
    rows=[]
    with (out/'evaluation.jsonl').open('w',buffering=1) as log:
        for layout,anchors in plan['eval_layouts'].items():
            saved=[]; references=[]; masks=[]; seed_values=[]; angle_values=[]
            for seed in plan['eval_seeds']:
                noise=torch.randn(1,64,266,device=device,generator=torch.Generator(device=device).manual_seed(seed))
                base=None
                for angle in [0,-12,-6,6,12]:
                    if remaining()<120:raise RuntimeError('Allocation reserve reached; incomplete evaluation')
                    raw=states[0:1,angles.index(angle)];clean=normalizer.normalize(raw)
                    controls=make_controls(raw,normalizer,anchors)
                    def check(k,x):
                        if not torch.isfinite(x).all():raise RuntimeError('Nonfinite generation')
                        if not torch.equal(x[controls.known],controls.values[controls.known]):raise AssertionError('Anchor drift')
                    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        generated=rollout(model,noise.clone(),controls,None,None,valid,32,'uniform',callback=check)
                    physical=normalizer.denormalize(generated.float())
                    r=reconstruction_metrics(generated.float(),clean,normalizer,controls.known)
                    r['anchor_max_m']=(xyz266(physical)-xyz266(raw)).norm(dim=-1)[controls.joints].max().item()
                    r.update(layout=layout,angle=angle,seed=seed,clip_index=0,all_steps_known_exact=True,anchor_count=len(anchors))
                    if angle==0:base=physical.clone()
                    else:
                        truth0=states[0:1,angles.index(0)]
                        r.update(coherent_response(physical,base,raw,truth0,controls.known))
                        r.update({'common_'+k:v for k,v in coherent_response(physical,base,raw,truth0,common).items()})
                    rows.append(r);log.write(json.dumps(r)+'\n')
                    saved.append(physical[0].cpu().numpy());references.append(raw[0].cpu().numpy())
                    masks.append(controls.known[0].cpu().numpy());seed_values.append(seed);angle_values.append(angle)
            np.savez_compressed(samples/f'{layout}.npz',generated266=np.stack(saved),reference266=np.stack(references),
                known=np.stack(masks),seeds=seed_values,angles=angle_values)
            print(json.dumps(dict(factor_evaluation=str(out),layout=layout,cases=len(rows))),flush=True)
    assert len(rows)==len(plan['eval_layouts'])*len(plan['eval_seeds'])*5
    (out/'COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan_path='configs/control_factors_v1.json'; approval=approval_check('approvals/of007.json',plan_path)
    plan=json.loads(Path(plan_path).read_text()); root=Path('/workspace/outputs/OF007')
    if root.exists():raise FileExistsError('Never restart an existing suite')
    if not torch.cuda.is_available():raise RuntimeError('Approved GPU required')
    for item in plan['arms']:
        if sha(item['config'])!=item['config_sha256']:raise ValueError('Arm config changed')
        approval_check(item['approval'],item['config'])
    if sha(plan['baseline_checkpoint'])!=plan['baseline_checkpoint_sha256'] or sha(plan['bank_file'])!=plan['bank_sha256']:
        raise ValueError('Frozen input changed')
    root.mkdir(); deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00')).timestamp()
    def remaining():return deadline-time.time()
    with np.load(plan['bank_file']) as data:states=torch.from_numpy(data['states']).cuda();angles=data['angles'].tolist()
    baseline=torch.load(plan['baseline_checkpoint'],map_location='cpu',weights_only=False)
    assert baseline['step']==4000 and baseline['config']==json.loads(Path(plan['baseline_config']).read_text())
    fixed_norm={k:v.clone() for k,v in baseline['normalizer'].items()}
    manifest=dict(plan=plan,approval=approval,gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.backends.cuda.matmul.allow_tf32=True
    def eval_checkpoint(cp,dest):
        for k,v in fixed_norm.items():
            if not torch.equal(cp['normalizer'][k],v):raise ValueError('Normalizer differs from frozen baseline')
        model=build_control_model(cp['config']);model.load_state_dict(cp['model'],strict=True)
        norm=Normalizer266(cp['normalizer']['mean'],cp['normalizer']['std']).cuda()
        model=model.cuda().eval().requires_grad_(False)
        evaluate_factor(model,norm,states,angles,plan,dest,remaining)
        del model,norm;torch.cuda.empty_cache()
    eval_checkpoint(baseline,root/'baseline_eval');del baseline
    for item in plan['arms']:
        if remaining()<plan['minimum_arm_start_seconds']:raise RuntimeError('Not enough allocation for next arm; do not restart')
        print('START_ARM '+item['name'],flush=True)
        subprocess.run([sys.executable,'-m','motion_valley.control_experiment','--config',item['config'],
            '--approval-file',item['approval'],'--data','/workspace/data','--output',str(root/item['name'])],check=True)
        cp=torch.load(root/item['name']/'last.pt',map_location='cpu',weights_only=False)
        assert cp['step']==4000 and cp['config']==json.loads(Path(item['config']).read_text())
        eval_checkpoint(cp,root/(item['name']+'_eval'));del cp
        print('END_ARM '+item['name'],flush=True)
    assert sha(plan['baseline_checkpoint'])==plan['baseline_checkpoint_sha256']
    (root/'SUITE_COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    print('OF007_SUITE_COMPLETED: 3 training arms,400 routine evaluations,640 common-protocol evaluations',flush=True)


if __name__=='__main__':main()
