"""OF003 paired experiments on the same kinematically coherent state bank."""
import argparse,datetime,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .overfit import approval_check,training_prediction,reconstruction_metrics
from .multi_overfit import make_controls,passes_reconstruction
from .control_states import fit_state_normalizer,coherent_response
from .representation266 import xyz266,redundancy_loss
from .flow_features import rollout
from .text_umt5 import cache_texts,read_cache
from .model_factory import build_control_model


def text_batch(contexts,ids,device):
    if contexts is None: return None,None
    text=torch.zeros(len(ids),max(len(contexts[i]) for i in ids),4096,device=device,dtype=torch.bfloat16)
    mask=torch.zeros(text.shape[:2],device=device,dtype=torch.bool)
    for j,i in enumerate(ids): text[j,:len(contexts[i])]=contexts[i]; mask[j,:len(contexts[i])]=True
    return text,mask


def save_checkpoint(path,model,optimizer,step,cfg,normalizer,rng,noise_rng,order):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
        config=cfg,normalizer=normalizer.state_dict(),training_rng=rng.getstate(),
        noise_rng=noise_rng.get_state(),pending_epoch_order=order,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state()),temporary)
    temporary.replace(path)


def evaluate(model,states,angles,normalizer,contexts,cfg,out,remaining):
    rows=[]; samples=out/'samples'; samples.mkdir(exist_ok=True)
    base_index=angles.index(0); order=[0]+[a for a in cfg['eval_angles'] if a!=0]
    with (out/'evaluation.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for seed in cfg['heldout_noise_seeds']:
            for i in range(len(states)):
                noise=torch.randn(states[:,0][:1].shape,device=states.device,
                    generator=torch.Generator(device=states.device).manual_seed(seed+i*10000))
                text,text_valid=text_batch(contexts,[i],states.device)
                for layout in cfg['eval_layouts']:
                    base=None
                    for angle in order:
                        if remaining()<45: return rows,False
                        raw=states[i:i+1,angles.index(angle)]; clean=normalizer.normalize(raw)
                        controls=make_controls(raw,normalizer,cfg['layouts'][layout])
                        valid=torch.ones(clean.shape[:2],device=states.device,dtype=torch.bool)
                        def check(k,x):
                            if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite rollout')
                            if not torch.equal(x[controls.known],controls.values[controls.known]): raise AssertionError('Clean anchor changed')
                        with torch.no_grad(),torch.autocast(states.device.type,dtype=torch.bfloat16,enabled=states.device.type=='cuda'):
                            generated=rollout(model,noise.clone(),controls,text,text_valid,valid,cfg['sampling_steps'],cfg['mode'],callback=check)
                        physical=normalizer.denormalize(generated.float())
                        result=reconstruction_metrics(generated.float(),clean,normalizer,controls.known)
                        result['anchor_max_m']=(xyz266(physical)-xyz266(raw)).norm(dim=-1)[controls.joints].max().item()
                        result.update(clip_index=i,seed=seed,layout=layout,angle=angle,all_steps_known_exact=True,
                                      heldout_angle=angle not in cfg['train_angles'],heldout_layout=layout not in cfg['train_layouts'])
                        if angle==0: base=physical.detach()
                        else:
                            result.update(coherent_response(physical,base,raw,states[i:i+1,base_index],controls.known))
                            overwrite=torch.where(controls.known,raw,base)
                            result['overwrite_baseline']=coherent_response(overwrite,base,raw,states[i:i+1,base_index],controls.known)
                        rows.append(result); log.write(json.dumps(result)+'\n')
                        if seed==cfg['heldout_noise_seeds'][0]:
                            np.savez_compressed(samples/f'{i:02d}_{layout}_{angle}.npz',generated266=physical.cpu().numpy(),
                                reference266=raw.cpu().numpy(),known=controls.known.cpu().numpy())
                print(json.dumps(dict(evaluated_clip=i,seed=seed,completed_cases=len(rows))),flush=True)
    return rows,True


def summarize(rows,cfg,complete,step):
    expected=cfg['clips']*len(cfg['eval_angles'])*len(cfg['eval_layouts'])*len(cfg['heldout_noise_seeds'])
    result=dict(completed_steps=step,completed_requested_steps=step==cfg['steps'],completed_evaluation=complete,
                expected_cases=expected,actual_cases=len(rows),text_enabled=cfg['text_enabled'])
    for layout_hold in [False,True]:
        for angle_hold in [False,True]:
            label=f"{'unseen' if layout_hold else 'trained'}_layout_{'unseen' if angle_hold else 'trained'}_angle"
            group=[r for r in rows if r['heldout_layout']==layout_hold and r['heldout_angle']==angle_hold]
            expected_group=cfg['clips']*len(cfg['heldout_noise_seeds'])*sum((a not in cfg['train_angles'])==angle_hold for a in cfg['eval_angles'])*sum((l not in cfg['train_layouts'])==layout_hold for l in cfg['eval_layouts'])
            count=sum(passes_reconstruction(r,cfg['overfit_gate']) for r in group)
            response=[r for r in group if r['angle']!=0]
            response_pass=sum(cfg['response_gain_min']<=r['response_gain']<=cfg['response_gain_max'] and r['response_relative_error']<cfg['response_error_max'] for r in response)
            result[label]=dict(count=len(group),expected=expected_group,reconstruction_pass_count=count,
                reconstruction_passed=complete and step==cfg['steps'] and len(group)==expected_group and count==len(group),
                response_count=len(response),response_pass_count=response_pass,
                response_passed=complete and step==cfg['steps'] and len(group)==expected_group and len(response)>0 and response_pass/len(response)>=cfg['response_fraction_min'])
    return result


def select_bank_subset(states,meta,cfg):
    indices=cfg.get('bank_clip_indices',list(range(len(states))))
    if len(indices)!=cfg['clips'] or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<len(states) for i in indices):
        raise ValueError('Invalid or duplicate bank clip selection')
    if cfg['clips']*len(cfg['train_angles']) % cfg['batch_size']:
        raise ValueError('Balanced epoch must be divisible by batch size')
    return states[indices],[meta['captions'][i] for i in indices],indices


def fit_experiment_normalizer(all_states,selected_states,train_indices,mean,std,cfg):
    """Optional fixed normalization population for controlled data-size studies."""
    indices=cfg.get('normalizer_bank_clip_indices')
    source=selected_states
    if indices is not None:
        if not indices or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<len(all_states) for i in indices):
            raise ValueError('Invalid fixed normalizer population')
        source=all_states[indices]
    return fit_state_normalizer(source[:,train_indices],mean,std)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--approval-file',required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    approval=approval_check(args.approval_file,args.config); cfg=json.loads(Path(args.config).read_text())
    project=Path(__file__).resolve().parents[1]; bank=project/cfg['bank_file']; meta_path=bank.with_name('manifest.json')
    if hashlib.sha256(meta_path.read_bytes()).hexdigest()!=cfg['bank_manifest_sha256']: raise ValueError('State-bank metadata hash mismatch')
    meta=json.loads(meta_path.read_text())
    if hashlib.sha256(bank.read_bytes()).hexdigest()!=cfg['bank_sha256'] or meta['bank_sha256']!=cfg['bank_sha256']:
        raise ValueError('Approved state bank hash mismatch')
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Never overwrite or restart a prior run')
    if not torch.cuda.is_available(): raise RuntimeError('Approved GPU required')
    root=Path(args.data); assets=json.loads((root/'download_manifest.json').read_text())
    if assets['revision']!=cfg['asset_revision']: raise ValueError('Unexpected official assets')
    deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00')).timestamp()
    start=time.monotonic()
    def remaining(): return min(deadline-time.time(),cfg['max_gpu_minutes']*60-(time.monotonic()-start))
    if remaining()<300: raise RuntimeError('Insufficient allocation time')
    out.mkdir(parents=True); device=torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32=True
    data=np.load(bank)
    all_states=torch.from_numpy(data['states']).to(device)
    states,captions,selected_indices=select_bank_subset(all_states,meta,cfg)
    angles=data['angles'].tolist()
    contexts=None
    if cfg['text_enabled']:
        cache_texts(captions,root/'deps',root/'umt5_cache',device='cuda'); torch.cuda.empty_cache()
        contexts=[read_cache(c,root/'umt5_cache').to(device) for c in captions]
    mean=torch.from_numpy(np.load(root/'raw_data/HumanML3D/Mean.npy')).float().to(device)
    std=torch.from_numpy(np.load(root/'raw_data/HumanML3D/Std.npy')).float().to(device)
    train_indices=[angles.index(a) for a in cfg['train_angles']]
    normalizer=fit_experiment_normalizer(all_states,states,train_indices,mean,std,cfg).to(device)
    # Re-seed after UMT5 so the two arms share initial motion weights and RNGs.
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    model=build_control_model(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.,
        betas=tuple(cfg.get('optimizer_betas',[.9,.999])),eps=1e-8)
    rng=random.Random(cfg['seed']); noise_rng=torch.Generator(device=device).manual_seed(cfg['seed']+1)
    manifest=dict(config=cfg,approval=approval,bank=meta,assets=assets,selected_bank_indices=selected_indices,
        captions=captions,parameters=sum(p.numel() for p in model.parameters()),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),text_cache=json.loads((root/'umt5_cache/manifest.json').read_text()) if contexts is not None else None)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    order=[]; step=0
    with (out/'train.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for current in range(1,cfg['steps']+1):
            if remaining()<cfg['evaluation_reserve_seconds']: break
            if not order:
                order=[(i,j) for i in range(len(states)) for j in train_indices]; rng.shuffle(order)
            batch=[order.pop() for _ in range(cfg['batch_size'])]; ids=[x[0] for x in batch]
            raw=torch.stack([states[i,j] for i,j in batch]); clean=normalizer.normalize(raw)
            layout=rng.choice(cfg['train_layouts']); controls=make_controls(raw,normalizer,cfg['layouts'][layout])
            text,text_valid=text_batch(contexts,ids,device); valid=torch.ones(clean.shape[:2],device=device,dtype=torch.bool)
            k=rng.randrange(cfg['sampling_steps']); noise=torch.randn(clean.shape,device=device,generator=noise_rng)
            use_rollout=current>cfg['rollout_warmup_steps'] and rng.random()<cfg['rollout_probability']
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,sigma,nxt,_=training_prediction(model,clean,noise,controls,text,text_valid,valid,k,cfg['sampling_steps'],use_rollout,cfg['rollout_gradient_tail'],cfg['mode'])
                reconstruction=(pred.float()-clean).square()[(nxt<sigma)&~controls.known].mean()
                consistency=redundancy_loss(normalizer.denormalize(pred.float()),valid)
                loss=reconstruction+cfg['redundancy_weight']*consistency
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for group in optimizer.param_groups: group['lr']=cfg['learning_rate']*min(current/cfg['warmup_steps'],1.)
            optimizer.step(); step=current
            if step==1 or step%25==0:
                record=dict(step=step,batch=batch,layout=layout,k=k,rollout=use_rollout,reconstruction=reconstruction.item(),
                            redundancy=consistency.item(),loss=loss.item(),gradient_norm=float(grad),elapsed_s=time.monotonic()-start,
                            peak_memory_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
            if step%500==0: save_checkpoint(out/'last.pt',model,optimizer,step,cfg,normalizer,rng,noise_rng,order)
    save_checkpoint(out/'last.pt',model,optimizer,step,cfg,normalizer,rng,noise_rng,order)
    model.eval(); rows,complete=evaluate(model,states,angles,normalizer,contexts,cfg,out,remaining)
    summary=summarize(rows,cfg,complete,step); (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    if complete and step==cfg['steps']: (out/'COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    else: raise RuntimeError('Run incomplete; do not start another arm automatically')


if __name__=='__main__': main()
