"""Approval-gated real HumanML3D overfitting with the official UMT5 encoder.

No provisioning here. The approved operator starts/stops the paid pod.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import os
import datetime
import time
import numpy as np
import torch
from .data import assert_disjoint
from .flow_features import FeatureControls, teacher_state, rollout, field_step
from .schedule import distance_field
from .model266 import Motion266Flow
from .representation266 import from263, to263, xyz266, position_mask, fit_overfit_normalizer, redundancy_loss
from .text_umt5 import cache_texts, read_cache
from .diagnostics266 import consistency_metrics


def approval_check(path,config_path):
    approval=json.loads(Path(path).read_text())
    digest=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    if approval.get('approved') is not True or approval.get('config_sha256')!=digest or not approval.get('user_approval_reference'):
        raise PermissionError('This exact run configuration has not been explicitly approved')
    report=Path(__file__).resolve().parents[1]/approval.get('report_file','missing')
    if not report.is_file() or hashlib.sha256(report.read_bytes()).hexdigest()!=approval.get('report_sha256'):
        raise PermissionError('The approved report is missing or has changed')
    return approval


def load_clips(root,count,frames):
    root=Path(root)/'raw_data/HumanML3D'; audit=assert_disjoint(root)
    selected=[]; captions=[]; names=[]
    for name in sorted((root/'train.txt').read_text().splitlines()):
        feature=root/'new_joint_vecs'/(name+'.npy'); text=root/'texts'/(name+'.txt')
        if not feature.exists() or not text.exists(): continue
        a=np.load(feature)
        if a.ndim!=2 or a.shape[1]!=263 or len(a)<frames or len(a)>200 or not np.isfinite(a).all(): continue
        descriptions=[]
        for line in text.read_text(encoding='utf-8').splitlines():
            fields=line.split('#')
            if len(fields)==4 and float(fields[2])==0 and float(fields[3])==0:
                descriptions.append(fields[0])
        if not descriptions: continue
        selected.append(torch.from_numpy(a[:frames].astype(np.float32))); captions.append(descriptions[0]); names.append(name)
        if len(selected)==count: break
    if len(selected)!=count: raise ValueError('Insufficient verified train clips; no fallback to another split')
    return selected,captions,names,audit


def reconstruction_metrics(pred,truth,normalizer,known):
    raw=normalizer.denormalize(pred); gt=normalizer.denormalize(truth)
    xyz=xyz266(raw); ref=xyz266(gt)
    pos_known=known[...,4:70].reshape(*known.shape[:-1],22,3).all(-1)
    d=(xyz-ref).norm(dim=-1)
    result=dict(feature_mse=(pred-truth).square()[~known].mean().item(),
                xyz_mpjpe_m=d.mean().item(),free_xyz_mpjpe_m=d[~pos_known].mean().item(),
                anchor_max_m=d[pos_known].max().item() if pos_known.any() else 0.,
                root_mpjpe_m=d[...,0].mean().item(),
                rot6d_normalized_mse=(pred[...,70:196]-truth[...,70:196]).square().mean().item(),
                velocity_normalized_mse=(pred[...,196:262]-truth[...,196:262]).square().mean().item())
    result.update(consistency_metrics(raw,gt))
    result['zero_normalized_predictor_mse']=truth.square()[~known].mean().item()
    return result


def training_prediction(model,clean,noise,controls,text,text_valid,valid,k,steps,
                        use_rollout=False,gradient_tail=4,mode='valley'):
    if use_rollout:
        state=rollout(model,noise,controls,text,text_valid,valid,steps,mode,
                      stop_step=k,grad_last=min(k,gradient_tail))
        sigma,nxt=field_step(k,steps,controls,distance_field(controls.joints),mode)
    else:
        state,sigma,nxt=teacher_state(clean,noise,k,steps,controls,mode)
    pred=model(state,sigma,controls.known,text,text_valid,valid)
    return controls.project(pred),sigma,nxt,state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--approval-file',required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); approval=approval_check(args.approval_file,args.config)
    cfg=json.loads(Path(args.config).read_text()); root=Path(args.data); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('Real-data run requires the approved GPU; use CPU unit tests separately')
    device=torch.device('cuda'); start=time.monotonic()
    # The allocation clock includes image startup, downloads and dependencies.
    # This does NOT terminate billing: the active operator still deletes the pod.
    allocated_deadline=os.environ.get('EXPERIMENT_DEADLINE_UTC')
    if not allocated_deadline: raise RuntimeError('Allocation deadline required from supervisor')
    deadline=datetime.datetime.fromisoformat(allocated_deadline.replace('Z','+00:00')).timestamp()
    def remaining(): return min(cfg['max_gpu_minutes']*60-(time.monotonic()-start),deadline-time.time())
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    clips,captions,names,audit=load_clips(root,cfg['clips'],cfg['frames'])
    text_dir=root/'umt5_cache'
    # The heavyweight frozen encoder is loaded once then released before training.
    cache_texts(captions,root/'deps',text_dir,device='cuda')
    torch.cuda.empty_cache()
    contexts=[read_cache(c,text_dir).to(device) for c in captions]
    text=torch.zeros(len(clips),max(len(c) for c in contexts),4096,device=device,dtype=torch.bfloat16)
    text_valid=torch.zeros(text.shape[:2],device=device,dtype=torch.bool)
    for i,c in enumerate(contexts): text[i,:len(c)]=c; text_valid[i,:len(c)]=True
    official_mean=torch.from_numpy(np.load(root/'raw_data/HumanML3D/Mean.npy')).float()
    official_std=torch.from_numpy(np.load(root/'raw_data/HumanML3D/Std.npy')).float()
    normalizer=fit_overfit_normalizer(clips,official_mean,official_std).to(device)
    raw=from263(torch.stack(clips).to(device)); clean=normalizer.normalize(raw)
    valid=torch.ones(clean.shape[:2],device=device,dtype=torch.bool)
    joints=torch.zeros(*clean.shape[:2],22,device=device,dtype=torch.bool)
    for frame,joint in zip(cfg['control_frames'],cfg['control_joints']):
        if frame>=cfg['frames']: raise ValueError('control outside clip')
        joints[:,frame,joint]=True
    known=position_mask(joints)
    controls=FeatureControls(torch.where(known,clean,0),known,joints)
    model=Motion266Flow(**{k:cfg[k] for k in ['width','depth','heads','text_dim']}).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    metadata=dict(config=cfg,approval=approval,names=names,captions=captions,split_audit=audit,
                  parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(),text_cache=json.loads((text_dir/'manifest.json').read_text()))
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    log=(out/'train.jsonl').open('w',buffering=1)
    final_step=0
    for step in range(1,cfg['steps']+1):
        if remaining()<120:
            print('JOB_TIME_LIMIT: checkpoint and terminate pod via supervisor',flush=True); break
        model.train(); optimizer.zero_grad(set_to_none=True)
        k=int(torch.randint(cfg['sampling_steps'],()).item())
        noise=torch.randn_like(clean)
        use_rollout=step>cfg['rollout_warmup_steps'] and random.random()<cfg['rollout_probability']
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred,sigma,nxt,state=training_prediction(model,clean,noise,controls,text,text_valid,valid,k,
                         cfg['sampling_steps'],use_rollout,cfg['rollout_gradient_tail'],cfg['mode'])
            active=(nxt<sigma)&~known
            loss_recon=(pred.float()-clean).square()[active].mean()
            loss_consistency=redundancy_loss(normalizer.denormalize(pred.float()),valid)
            loss=loss_recon+cfg['redundancy_weight']*loss_consistency
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss; do not auto-restart')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups: group['lr']=cfg['learning_rate']*min(step/cfg['warmup_steps'],1)
        optimizer.step(); final_step=step
        if step==1 or step%25==0:
            record=dict(step=step,k=k,rollout=use_rollout,loss=loss.item(),reconstruction=loss_recon.item(),
                        redundancy=loss_consistency.item(),elapsed_s=time.monotonic()-start)
            log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
        if step%250==0:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                            config=cfg,normalizer=normalizer.state_dict()),out/'last.pt')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=final_step,
                    config=cfg,normalizer=normalizer.state_dict()),out/'last.pt')
    model.eval(); results=[]; artifact=None
    # Fixed unseen noise seeds; no teacher forcing and no GT access outside controls.
    for seed in cfg['heldout_noise_seeds']:
        if remaining()<30: break
        generator=torch.Generator(device=device).manual_seed(seed)
        noise=torch.randn(clean.shape,device=device,generator=generator)
        def check(k,state):
            if not torch.equal(state[known],clean[known]): raise AssertionError('valley changed during rollout')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            generated=rollout(model,noise,controls,text,text_valid,valid,cfg['sampling_steps'],cfg['mode'],callback=check)
        result=reconstruction_metrics(generated,clean,normalizer,known)
        result['seed']=seed; result['all_steps_known_exact']=True; results.append(result)
        if artifact is None:
            # Original physical control values survive normalization roundoff.
            generated_raw=torch.where(known,raw,normalizer.denormalize(generated))
            artifact=dict(generated266=generated_raw.cpu().numpy(),reference266=raw.cpu().numpy(),
                          generated263=to263(generated_raw).cpu().numpy(),reference263=torch.stack(clips).numpy(),
                          known=known.cpu().numpy())
    log.close()
    if artifact: np.savez_compressed(out/'overfit_samples.npz',**artifact)
    result=dict(step=final_step,tests=results,completed_requested_steps=final_step==cfg['steps'],
                completed_noise_tests=len(results)==len(cfg['heldout_noise_seeds']),
                note='Real single-clip overfit; no FID or generalization claim; inspect rotation/XYZ consistency separately')
    gate=cfg['overfit_gate']
    result['overfit_gate_passed']=len(results)==len(cfg['heldout_noise_seeds']) and all(
        r['feature_mse']<gate['feature_mse_max'] and r['free_xyz_mpjpe_m']<gate['free_xyz_mpjpe_m_max'] and r['anchor_max_m']<gate['anchor_error_m_max']
        and r['rot6d_normalized_mse']<gate['rotation_normalized_mse_max'] and r['velocity_normalized_mse']<gate['velocity_normalized_mse_max']
        and r['velocity_error_vs_static_ratio']<gate['motion_velocity_error_vs_static_max'] and r['degenerate_rotation_count']==0
        and r['fk_position_disagreement_m']<r['gt_fk_position_disagreement_m']+gate['fk_disagreement_above_gt_m_max']
        for r in results)
    (out/'overfit_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
