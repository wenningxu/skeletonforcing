"""OF002: multi-clip memory and paired conditioning diagnostics; approval gated."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .data import assert_disjoint
from .overfit import approval_check, reconstruction_metrics, training_prediction
from .representation266 import (from263, xyz266, position_mask, fit_overfit_normalizer,
                                redundancy_loss)
from .flow_features import FeatureControls, rollout
from .diagnostics266 import consistency_metrics
from .model266 import Motion266Flow
from .text_umt5 import cache_texts, read_cache
from .schedule import PARENTS


def select_clips(root, cfg):
    root=Path(root)/'raw_data/HumanML3D'
    if not all((root/(s+'.txt')).is_file() for s in ['train','val','test']):
        raise ValueError('All official split files are required')
    audit=assert_disjoint(root)
    names=sorted(set((root/'train.txt').read_text().splitlines()))
    names=[n for n in names if n and not n.startswith('M') and n not in cfg['exclude_ids']]
    random.Random(cfg['selection_seed']).shuffle(names)
    clips=[]; captions=[]; records=[]; rejected={}; seen=set()
    for name in names:
        def reject(reason): rejected[reason]=rejected.get(reason,0)+1
        feature=root/'new_joint_vecs'/(name+'.npy'); text=root/'texts'/(name+'.txt')
        if not feature.is_file() or not text.is_file(): reject('missing'); continue
        a=np.load(feature,allow_pickle=False)
        if a.ndim!=2 or a.shape[1]!=263 or not cfg['frames']<=len(a)<=200 or not np.isfinite(a).all():
            reject('shape_length_or_finite'); continue
        descriptions=[]
        for line in text.read_text(encoding='utf-8').splitlines():
            f=line.split('#')
            if len(f)!=4: continue
            try: full=float(f[2])==0 and float(f[3])==0
            except ValueError: continue
            key=' '.join(f[0].lower().split())
            if full and key and key not in seen: descriptions.append((f[0],key))
        if not descriptions: reject('no_unique_full_caption'); continue
        x=torch.from_numpy(a[:cfg['frames']].astype(np.float32)); pos=xyz266(from263(x))
        speed=pos.diff(dim=0).norm(dim=-1).mean().item()
        if speed<cfg['minimum_motion_m_per_frame']: reject('near_static'); continue
        if any((pos-xyz266(from263(old))).norm(dim=-1).mean().item()<cfg['minimum_pairwise_mpjpe_m'] for old in clips):
            reject('near_duplicate_motion'); continue
        caption,key=descriptions[0]; seen.add(key); clips.append(x); captions.append(caption)
        records.append(dict(id=name,caption=caption,source_frames=len(a),crop_start=0,
                            motion_m_per_frame=speed,feature_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),
                            text_sha256=hashlib.sha256(text.read_bytes()).hexdigest()))
        if len(clips)==cfg['clips']: break
    if len(clips)!=cfg['clips']: raise ValueError('Insufficient distinct verified train clips')
    return clips,captions,dict(splits=audit,selected=records,rejected_counts=rejected)


def make_controls(raw, normalizer, layout, shift=None):
    """Only observed XYZ enters values; no other GT channel is conditioned on."""
    joints=torch.zeros(*raw.shape[:2],22,dtype=torch.bool,device=raw.device)
    for frame,joint in layout:
        if not 0<=frame<raw.shape[1] or not 0<=joint<22: raise ValueError('Invalid anchor')
        joints[:,frame,joint]=True
    physical=raw.clone()
    if shift is not None:
        frame,joint,axis,amount=shift
        if not joints[:,frame,joint].all(): raise ValueError('Shift must target an observed point')
        physical[:,frame,4+3*joint+axis]+=amount
    known=position_mask(joints)
    values=torch.where(known,normalizer.normalize(physical),0)
    return FeatureControls(values,known,joints)


def paired_response(changed, baseline, controls, frame, joint, axis, amount, radius=4):
    if amount==0: raise ValueError('Nonzero intervention required')
    delta=xyz266(changed)-xyz266(baseline)
    known=controls.joints
    neighborhood=torch.zeros_like(known)
    lo=max(0,frame-radius); hi=min(delta.shape[1],frame+radius+1)
    neighborhood[:,lo:hi,joint]=True
    if PARENTS[joint]>=0: neighborhood[:,lo:hi,PARENTS[joint]]=True
    neighborhood &= ~known
    if not neighborhood.any(): raise ValueError('No unobserved neighborhood')
    return dict(free_displacement_m=delta.norm(dim=-1)[~known].mean().item(),
                neighborhood_displacement_m=delta.norm(dim=-1)[neighborhood].mean().item(),
                neighborhood_signed_axis_gain=(delta[...,axis][neighborhood].mean()/amount).item(),
                acceleration_change_m_frame2=delta.diff(n=2,dim=1).norm(dim=-1).mean().item())


def passes_reconstruction(r,g):
    return (r['feature_mse']<g['feature_mse_max'] and r['free_xyz_mpjpe_m']<g['free_xyz_mpjpe_m_max']
        and r['anchor_max_m']<g['anchor_error_m_max'] and r['rot6d_normalized_mse']<g['rotation_normalized_mse_max']
        and r['velocity_normalized_mse']<g['velocity_normalized_mse_max']
        and r['velocity_error_vs_static_ratio']<g['motion_velocity_error_vs_static_max']
        and r['degenerate_rotation_count']==0
        and r['fk_position_disagreement_m']<r['gt_fk_position_disagreement_m']+g['fk_disagreement_above_gt_m_max'])


def save_checkpoint(path,model,optimizer,step,cfg,normalizer):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                    config=cfg,normalizer=normalizer.state_dict(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate()),temporary)
    temporary.replace(path)


def evaluate(model,raw,clean,normalizer,contexts,cfg,out,remaining):
    rows=[]; baseline={}; sample_dir=out/'samples'; sample_dir.mkdir(exist_ok=True)
    with (out/'evaluation.jsonl').open('w',encoding='utf-8',buffering=1) as stream:
        for seed in cfg['heldout_noise_seeds']:
            for i in range(len(raw)):
                truth=clean[i:i+1]; physical=raw[i:i+1]
                generator=torch.Generator(device=raw.device).manual_seed(seed+i*10000)
                noise=torch.randn(truth.shape,device=raw.device,generator=generator)
                valid=torch.ones(truth.shape[:2],device=raw.device,dtype=torch.bool)
                # A/B/C/D with correct text; A with wrong/empty text; A with +/- shift.
                cases=[(name,'correct',None) for name in cfg['eval_layouts']]
                cases += [('A','wrong',None),('A','empty',None)]
                cases += [('A','correct',amount) for amount in cfg['shift_meters']]
                for layout_name,text_mode,amount in cases:
                    if remaining()<45: return rows,False
                    shift=None if amount is None else (*cfg['shift_target'],amount)
                    controls=make_controls(physical,normalizer,cfg['layouts'][layout_name],shift)
                    text_index=i if text_mode=='correct' else ((i+1)%len(raw) if text_mode=='wrong' else len(raw))
                    text=contexts[text_index][None]; text_valid=torch.ones(text.shape[:2],device=raw.device,dtype=torch.bool)
                    def check(k,state):
                        if not torch.isfinite(state).all(): raise RuntimeError('Nonfinite sampling state')
                        if not torch.equal(state[controls.known],controls.values[controls.known]):
                            raise AssertionError('Clean anchor changed during sampling')
                    with torch.no_grad(),torch.autocast(raw.device.type,dtype=torch.bfloat16,enabled=raw.device.type=='cuda'):
                        generated=rollout(model,noise.clone(),controls,text,text_valid,valid,cfg['sampling_steps'],
                                          cfg['mode'],callback=check)
                    result=dict(clip_index=i,seed=seed,actual_noise_seed=seed+i*10000,layout=layout_name,
                                text_mode=text_mode,shift_m=amount,all_steps_known_exact=True)
                    generated_raw=normalizer.denormalize(generated.float())
                    # Measure error before any final physical-coordinate replacement.
                    target=physical.clone()
                    if shift is not None:
                        frame,joint,axis,value=shift
                        target[:,frame,4+3*joint+axis]+=value
                    target=torch.where(controls.known,target,0)
                    anchor_error=(xyz266(generated_raw)-xyz266(target)).norm(dim=-1)[controls.joints].max().item()
                    if amount is None:
                        result.update(reconstruction_metrics(generated.float(),truth,normalizer,controls.known))
                        result['anchor_max_m']=anchor_error
                        if text_mode=='correct' and layout_name=='A': baseline[(i,seed)]=generated_raw.detach()
                        if text_mode!='correct':
                            result['paired_xyz_change_m']=(xyz266(generated_raw)-xyz266(baseline[(i,seed)])).norm(dim=-1)[~controls.joints].mean().item()
                    else:
                        base=baseline[(i,seed)]
                        result.update(paired_response(generated_raw,base,controls,*cfg['shift_target'],amount,cfg['response_radius']))
                        result['consistency']=consistency_metrics(generated_raw,physical)
                        # A pure overwrite has zero free response despite exact anchors.
                        overwrite=torch.where(controls.known,target,base)
                        result['overwrite_baseline']=paired_response(overwrite,base,controls,*cfg['shift_target'],amount,cfg['response_radius'])
                    result['target_anchor_max_m']=anchor_error
                    rows.append(result); stream.write(json.dumps(result)+'\n')
                    if seed==cfg['heldout_noise_seeds'][0]:
                        label=f'{i:02d}_{layout_name}_{text_mode}_{amount}'
                        np.savez_compressed(sample_dir/(label+'.npz'),generated266=generated_raw.cpu().numpy(),
                            reference266=physical.cpu().numpy(),known=controls.known.cpu().numpy(),
                            target266=target.cpu().numpy())
                print(json.dumps(dict(evaluated_clip=i,seed=seed,completed_cases=len(rows))),flush=True)
    return rows,True


def summarize(rows,cfg,complete,step):
    correct=[r for r in rows if r['text_mode']=='correct' and r['shift_m'] is None]
    train=[r for r in correct if r['layout'] in cfg['train_layouts']]
    heldout=[r for r in correct if r['layout'] not in cfg['train_layouts']]
    shifted=[r for r in rows if r['shift_m'] is not None]
    expected=cfg['clips']*len(cfg['heldout_noise_seeds'])
    success=lambda group: sum(passes_reconstruction(r,cfg['overfit_gate']) for r in group)
    response=sum(r['neighborhood_displacement_m']>=cfg['response_min_m'] for r in shifted)
    result=dict(completed_steps=step,completed_requested_steps=step==cfg['steps'],completed_evaluation=complete,
        expected_cases=expected*(len(cfg['eval_layouts'])+2+len(cfg['shift_meters'])),actual_cases=len(rows),
        train_layout_pass_count=success(train),train_layout_count=len(train),
        heldout_layout_pass_count=success(heldout),heldout_layout_count=len(heldout),
        training_layout_overfit_passed=complete and step==cfg['steps'] and len(train)==expected*len(cfg['train_layouts']) and success(train)==len(train),
        heldout_layout_overfit_passed=complete and len(heldout)==expected and success(heldout)==len(heldout),
        shift_nontrivial_response_count=response,shift_count=len(shifted),
        shift_response_diagnostic_passed=complete and len(shifted)==expected*len(cfg['shift_meters']) and response/len(shifted)>=cfg['response_fraction_min'])
    for mode in ['correct','wrong','empty']:
        group=[r for r in rows if r['layout']=='A' and r['text_mode']==mode and r['shift_m'] is None]
        if group: result[mode+'_text_mean_free_mpjpe_m']=sum(r['free_xyz_mpjpe_m'] for r in group)/len(group)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--approval-file',required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    approval=approval_check(args.approval_file,args.config)
    cfg=json.loads(Path(args.config).read_text()); root=Path(args.data); out=Path(args.output)
    assets=json.loads((root/'download_manifest.json').read_text())
    if assets.get('repo')!='ShandaAI/FloodDiffusionDownloads' or assets.get('revision')!=cfg['asset_revision']:
        raise ValueError('Publisher data manifest does not match the approved revision')
    if {f['file'] for f in assets['files']}!={'deps.zip','HumanML3D.zip'}:
        raise ValueError('Expected verified dependency and HumanML3D archives')
    if out.exists() and any(out.iterdir()): raise FileExistsError('Refuse to overwrite previous run')
    if not torch.cuda.is_available(): raise RuntimeError('Approved GPU required')
    deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00')).timestamp()
    start=time.monotonic()
    def remaining(): return min(deadline-time.time(),cfg['max_gpu_minutes']*60-(time.monotonic()-start))
    if remaining()<300: raise RuntimeError('Insufficient allocation time')
    if cfg['clips']%cfg['batch_size']: raise ValueError('Balanced epoch requires divisible batch size')
    device=torch.device('cuda'); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    clips,captions,audit=select_clips(root,cfg)
    (out/'selection.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    cache_texts(captions+[''],root/'deps',root/'umt5_cache',device='cuda'); torch.cuda.empty_cache()
    contexts=[read_cache(c,root/'umt5_cache').to(device) for c in captions+['']]
    mean=torch.from_numpy(np.load(root/'raw_data/HumanML3D/Mean.npy')).float()
    std=torch.from_numpy(np.load(root/'raw_data/HumanML3D/Std.npy')).float()
    normalizer=fit_overfit_normalizer(clips,mean,std).to(device)
    raw=from263(torch.stack(clips).to(device)); clean=normalizer.normalize(raw)
    model=Motion266Flow(**{k:cfg[k] for k in ['width','depth','heads','text_dim']}).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    metadata=dict(config=cfg,approval=approval,selection=audit,captions=captions,assets=assets,parameters=sum(p.numel() for p in model.parameters()),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),text_cache=json.loads((root/'umt5_cache/manifest.json').read_text()))
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    step=0; order=[]
    with (out/'train.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for current in range(1,cfg['steps']+1):
            if remaining()<cfg['evaluation_reserve_seconds']: break
            if not order: order=list(range(cfg['clips'])); random.shuffle(order)
            ids=[order.pop() for _ in range(cfg['batch_size'])]
            layout=random.choice(cfg['train_layouts'])
            target=clean[ids]; physical=raw[ids]
            controls=make_controls(physical,normalizer,cfg['layouts'][layout])
            text=torch.zeros(len(ids),max(len(contexts[i]) for i in ids),4096,device=device,dtype=torch.bfloat16)
            text_valid=torch.zeros(text.shape[:2],device=device,dtype=torch.bool)
            for j,i in enumerate(ids): text[j,:len(contexts[i])]=contexts[i]; text_valid[j,:len(contexts[i])]=True
            valid=torch.ones(target.shape[:2],device=device,dtype=torch.bool)
            k=int(torch.randint(cfg['sampling_steps'],()).item()); noise=torch.randn_like(target)
            use_rollout=current>cfg['rollout_warmup_steps'] and random.random()<cfg['rollout_probability']
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,sigma,nxt,_=training_prediction(model,target,noise,controls,text,text_valid,valid,k,
                    cfg['sampling_steps'],use_rollout,cfg['rollout_gradient_tail'],cfg['mode'])
                active=(nxt<sigma)&~controls.known
                reconstruction=(pred.float()-target).square()[active].mean()
                consistency=redundancy_loss(normalizer.denormalize(pred.float()),valid)
                loss=reconstruction+cfg['redundancy_weight']*consistency
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; no automatic restart')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for group in optimizer.param_groups: group['lr']=cfg['learning_rate']*min(current/cfg['warmup_steps'],1.)
            optimizer.step(); step=current
            if step==1 or step%25==0:
                record=dict(step=step,clip_indices=ids,layout=layout,k=k,rollout=use_rollout,loss=loss.item(),
                            reconstruction=reconstruction.item(),redundancy=consistency.item(),gradient_norm=float(norm),elapsed_s=time.monotonic()-start)
                log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
            if step%500==0: save_checkpoint(out/'last.pt',model,optimizer,step,cfg,normalizer)
    save_checkpoint(out/'last.pt',model,optimizer,step,cfg,normalizer)
    model.eval(); rows,complete=evaluate(model,raw,clean,normalizer,contexts,cfg,out,remaining)
    summary=summarize(rows,cfg,complete,step)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
    if complete and step==cfg['steps']: (out/'COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__=='__main__': main()
