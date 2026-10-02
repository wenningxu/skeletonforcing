"""OF006: frozen precision/teacher-state probes and a one-loss continuation test."""
import datetime,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .overfit import approval_check,training_prediction,reconstruction_metrics
from .multi_overfit import make_controls
from .control_experiment import evaluate,summarize
from .control_states import coherent_response
from .representation266 import Normalizer266,xyz266,redundancy_loss
from .flow_features import rollout,field_step
from .schedule import distance_field
from .model_factory import build_control_model
from .noise_evaluation import sha256
from .response_supervision import state_difference_loss


def restore(checkpoint):
    cfg=checkpoint['config']
    model=build_control_model(cfg)
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.cuda()


def probes(model,states,angles,normalizer,cfg,out):
    raw=states[[angles.index(a) for a in [-12,0,12]]]
    clean=normalizer.normalize(raw)
    controls=make_controls(raw,normalizer,cfg['layouts']['A'])
    base_ctrl=make_controls(raw[1:2],normalizer,cfg['layouts']['A'])
    valid=torch.ones(3,64,device='cuda',dtype=torch.bool)
    noise=torch.randn(1,64,266,device='cuda',generator=torch.Generator(device='cuda').manual_seed(77123))
    cache={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        rollout(model,noise,base_ctrl,None,None,valid[:1],32,callback=lambda k,x:cache.update({k:x.detach().clone()}))
    records=[]
    for precision in ['bf16','fp32']:
        for k in [0,8,16,24,31]:
            sigma,nxt=field_step(k,32,controls,distance_field(controls.joints))
            teacher=controls.project((1-sigma)*clean+sigma*noise)
            teacher_zero=torch.where(controls.known,base_ctrl.values,teacher)
            swap=controls.project(cache[k].expand(3,-1,-1).clone())
            for kind,x in [('teacher_correct',teacher),('teacher_zero_anchor',teacher_zero),('rollout_anchor_swap',swap)]:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                    prediction=model(x,sigma,controls.known,None,None,valid)
                # All response measurements exclude known coordinates.
                physical=normalizer.denormalize(prediction.float())
                for i in [0,2]:
                    response=coherent_response(physical[i:i+1],physical[1:2],raw[i:i+1],raw[1:2],controls.known[i:i+1])
                    records.append(dict(precision=precision,k=k,state=kind,angle=[-12,0,12][i],**response))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            generated=rollout(model,noise.expand(3,-1,-1).clone(),controls,None,None,valid,32)
        physical=normalizer.denormalize(generated.float())
        np.savez_compressed(out/f'precision_{precision}.npz',generated266=physical.cpu().numpy(),reference266=raw.cpu().numpy(),known=controls.known.cpu().numpy())
        for i in [0,2]:
            records.append(dict(precision=precision,k=32,state='full_rollout',angle=[-12,0,12][i],
                **coherent_response(physical[i:i+1],physical[1:2],raw[i:i+1],raw[1:2],controls.known[i:i+1])))
    # Input derivative along the actual +12-degree anchor perturbation.
    gradient_rows=[]
    for k in [0,16]:
        sigma,_=field_step(k,32,base_ctrl,distance_field(base_ctrl.joints))
        x=cache[k].detach().requires_grad_(True)
        model.requires_grad_(False)
        pred=model(x,sigma,base_ctrl.known,None,None,valid[:1])
        physical=xyz266(normalizer.denormalize(pred.float()))
        target=xyz266(raw[2:3]-raw[1:2]); region=torch.zeros(1,64,22,device='cuda',dtype=torch.bool)
        region[:,36:61,[17,19,21]]=True; region &= ~base_ctrl.joints
        score=(physical[region]*target[region]).sum()/target[region].square().sum()
        grad=torch.autograd.grad(score,x)[0]
        direction=torch.where(base_ctrl.known,controls.values[2:3]-base_ctrl.values,0.)
        gradient_rows.append(dict(k=k,projected_anchor_direction_derivative=float((grad*direction).sum()),
                                  anchor_gradient_norm=float(grad[base_ctrl.known].norm())))
    (out/'probes.json').write_text(json.dumps(dict(rows=records,input_gradients=gradient_rows),indent=2))
    model.requires_grad_(True)


def independent_evaluate(model,states,angles,normalizer,cfg,out,remaining):
    records=[]; device=states.device; valid=torch.ones(1,64,device=device,dtype=torch.bool)
    for li,layout in enumerate(['A','D']):
        for ai,angle in enumerate([-12,-6,0,6,12]):
            raw=states[angles.index(angle)][None]; clean=normalizer.normalize(raw)
            controls=make_controls(raw,normalizer,cfg['layouts'][layout]); saved=[]; seeds=[]
            for i in range(cfg.get('independent_noise_samples',16)):
                if remaining()<90: raise RuntimeError('Insufficient time for complete evaluation')
                seed=820000+li*10000+ai*1000+i
                noise=torch.randn(clean.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
                def check(k,x):
                    if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite sample')
                    if not torch.equal(x[controls.known],controls.values[controls.known]): raise AssertionError('Anchor drift')
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    pred=rollout(model,noise,controls,None,None,valid,cfg['sampling_steps'],cfg['mode'],callback=check)
                physical=normalizer.denormalize(pred.float())
                r=reconstruction_metrics(pred.float(),clean,normalizer,controls.known)
                r.update(layout=layout,angle=angle,seed=seed,sample_index=i,all_steps_known_exact=True)
                records.append(r); saved.append(physical[0].cpu().numpy()); seeds.append(seed)
            np.savez_compressed(out/'samples'/f'independent_{layout}_{angle}.npz',generated266=np.stack(saved),
                reference266=raw[0].cpu().numpy(),known=controls.known[0].cpu().numpy(),seeds=seeds)
    (out/'independent.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))


def main():
    plan_path=Path('configs/control_diagnosis_v1.json'); plan=json.loads(plan_path.read_text())
    approval=approval_check('approvals/of006.json',plan_path)
    root=Path('/workspace/outputs/OF006')
    if root.exists(): raise FileExistsError('Never restart this diagnostic suite')
    deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00')).timestamp()
    def remaining(): return deadline-time.time()
    if not torch.cuda.is_available() or remaining()<600: raise RuntimeError('Approved allocation required')
    if sha256(plan['checkpoint'])!=plan['checkpoint_sha256'] or sha256(plan['bank_file'])!=plan['bank_sha256']:
        raise ValueError('Frozen input hash mismatch')
    checkpoint=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    cfg=checkpoint['config']; assert checkpoint['step']==4000 and cfg['architecture']=='wan_joint_time_v1'
    if cfg!=json.loads(Path(plan['model_config']).read_text()): raise ValueError('Model config mismatch')
    checkpoint.pop('optimizer')
    root.mkdir(parents=True); (root/'manifest.json').write_text(json.dumps(dict(plan=plan,model_config=cfg,approval=approval,torch=torch.__version__,gpu=torch.cuda.get_device_name()),indent=2))
    normalizer=Normalizer266(checkpoint['normalizer']['mean'],checkpoint['normalizer']['std']).cuda()
    np.savez(root/'normalizer.npz',mean=normalizer.mean.cpu().numpy(),std=normalizer.std.cpu().numpy())
    with np.load(plan['bank_file']) as bank:
        states=torch.from_numpy(bank['states'][0]).cuda(); angles=bank['angles'].tolist()
    torch.backends.cuda.matmul.allow_tf32=True
    model=restore(checkpoint).eval(); probes(model,states,angles,normalizer,cfg,root); del model; torch.cuda.empty_cache()
    print('Frozen checkpoint probes completed',flush=True)
    for arm in ['baseline','response']:
        if remaining()<plan['minimum_arm_start_seconds']: raise RuntimeError('Not enough time for the next arm; do not restart')
        out=root/arm; out.mkdir(); rng=random.Random(plan['seed'])
        torch.manual_seed(plan['seed']); noise_rng=torch.Generator(device='cuda').manual_seed(plan['seed']+1)
        model=restore(checkpoint).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],betas=(.9,.99),eps=1e-8,weight_decay=0.)
        raw=states[[angles.index(a) for a in [-12,0,12]]]; clean=normalizer.normalize(raw)
        controls=make_controls(raw,normalizer,cfg['layouts']['A']); valid=torch.ones(3,64,device='cuda',dtype=torch.bool)
        start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        with (out/'train.jsonl').open('w',buffering=1) as log:
            for step in range(1,plan['steps']+1):
                if remaining()<plan['evaluation_reserve_seconds']: raise RuntimeError('Deadline reserve; incomplete run')
                k=rng.randrange(32); use_rollout=rng.random()<.5
                noise=torch.randn(clean.shape,device='cuda',generator=noise_rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred,sigma,nxt,_=training_prediction(model,clean,noise,controls,None,None,valid,k,32,use_rollout,4,'valley')
                    active=(nxt<sigma)&~controls.known
                    reconstruction=(pred.float()-clean).square()[active].mean()
                    consistency=redundancy_loss(normalizer.denormalize(pred.float()),valid)
                    response=state_difference_loss(pred,clean,normalizer.std,active,controls.known)
                    loss=reconstruction+.01*consistency+(plan['response_weight'] if arm=='response' else 0.)*response
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                for group in optimizer.param_groups: group['lr']=plan['learning_rate']*min(step/plan['warmup_steps'],1.)
                optimizer.step()
                if step==1 or step%25==0:
                    row=dict(arm=arm,step=step,k=k,rollout=use_rollout,reconstruction=float(reconstruction),
                        redundancy=float(consistency),response=float(response),loss=float(loss),gradient_norm=float(grad),
                        elapsed_s=time.monotonic()-start,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
                    log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
        # Inference artifact only; this does not promise exact optimizer resumption.
        torch.save(dict(model=model.state_dict(),normalizer=normalizer.state_dict(),config=cfg,
                        diagnosis_plan=plan,arm=arm,source_step=4000,additional_steps=plan['steps']),out/'model.pt')
        model.eval(); eval_cfg=dict(cfg,steps=plan['steps'])
        rows,complete=evaluate(model,states[None],angles,normalizer,None,eval_cfg,out,remaining)
        summary=summarize(rows,eval_cfg,complete,plan['steps'])
        if not complete: raise RuntimeError('Incomplete paired evaluation')
        independent_evaluate(model,states,angles,normalizer,cfg,out,remaining)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)); (out/'COMPLETED').write_text('complete')
        print(json.dumps(dict(arm=arm,summary=summary)),flush=True)
        del model,optimizer,pred,loss; torch.cuda.empty_cache()
    if sha256(plan['checkpoint'])!=plan['checkpoint_sha256']: raise ValueError('Source checkpoint altered')
    (root/'COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__=='__main__': main()
