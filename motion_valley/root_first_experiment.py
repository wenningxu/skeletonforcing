"""One bounded, unconditional Wan overfit; no GPU provisioning or auto-retry."""
import argparse,datetime,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .overfit import approval_check,reconstruction_metrics
from .control_states import fit_state_normalizer
from .model_factory import build_control_model
from .representation266 import Layout266,xyz266
from .root_first import RootFirstSchedule,joint_depths,rollout,training_prediction,active_redundancy_loss
from .multi_overfit import passes_reconstruction


def checkpoint(out,model,optimizer,step,cfg,norm,rng,noise_rng,pending):
    tmp=out/'last.tmp'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,config=cfg,
                    normalizer=norm.state_dict(),training_rng=rng.getstate(),noise_rng=noise_rng.get_state(),
                    pending_steps=pending,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),tmp)
    tmp.replace(out/'last.pt')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--approval-file',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    approval=approval_check(args.approval_file,args.config);cfg=json.loads(Path(args.config).read_text())
    root=Path(__file__).resolve().parents[1];out=Path(args.output)
    if out.exists():raise FileExistsError('Never overwrite/restart this experiment')
    if cfg['text_enabled'] or cfg['train_angle']!=0 or cfg['clips']!=1:raise ValueError('Single original clip, no text')
    if cfg['scheduling_origin']!=[0,0] or cfg['mode']!='valley_default_origin_equal_steps':
        raise ValueError('Only the default frame-zero/root valley entry is supported')
    audit=json.loads((root/cfg['source_audit_file']).read_text())
    if hashlib.sha256((root/cfg['source_audit_file']).read_bytes()).hexdigest()!=cfg['source_audit_sha256']:
        raise ValueError('Source audit changed')
    for path,digest in audit['files'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed: '+path)
    bank=root/cfg['bank_file']
    if hashlib.sha256(bank.read_bytes()).hexdigest()!=cfg['bank_sha256']:raise ValueError('Data changed')
    if hashlib.sha256(bank.with_name('manifest.json').read_bytes()).hexdigest()!=cfg['bank_manifest_sha256']:
        raise ValueError('Data metadata changed')
    if not torch.cuda.is_available():raise RuntimeError('GPU entry; CPU tests are separate')
    deadline=datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z','+00:00')).timestamp()
    started=time.monotonic();remaining=lambda:min(deadline-time.time(),cfg['max_gpu_minutes']*60-(time.monotonic()-started))
    if remaining()<600:raise RuntimeError('Insufficient allocation time')
    device=torch.device('cuda');data=Path(args.data)
    assets=json.loads((data/'download_manifest.json').read_text())
    if assets['revision']!=cfg['asset_revision']:raise ValueError('Wrong official assets')
    with np.load(bank) as d:
        raw=torch.from_numpy(d['states'][cfg['bank_clip_index']:cfg['bank_clip_index']+1,d['angles'].tolist().index(0)]).to(device)
    assert raw.shape==(1,cfg['frames'],266)
    mean=torch.from_numpy(np.load(data/'raw_data/HumanML3D/Mean.npy')).to(device)
    std=torch.from_numpy(np.load(data/'raw_data/HumanML3D/Std.npy')).to(device)
    norm=fit_state_normalizer(raw,mean,std).to(device);clean=norm.normalize(raw)
    schedule=RootFirstSchedule(**cfg['schedule']);total_steps=schedule.total_steps(cfg['frames'])
    assert total_steps==cfg['sampling_steps']
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    model=build_control_model(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),eps=1e-8,weight_decay=0.)
    rng=random.Random(cfg['seed']);noise_rng=torch.Generator(device=device).manual_seed(cfg['seed']+1)
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg,approval=approval,source_audit=audit,
        parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        no_observations=True,no_text=True,joint_depths=joint_depths(),asset_revision=assets['revision']),indent=2))
    np.savez_compressed(out/'time_field.npz',sigma=schedule.lattice(cfg['frames']).numpy(),owner=Layout266().owner.numpy())
    # Small normalizer artifact permits independent local physical-metric checks.
    np.savez_compressed(out/'normalizer.npz',mean=norm.mean.cpu().numpy(),std=norm.std.cpu().numpy())
    batch=clean.expand(cfg['batch_size'],-1,-1);valid=torch.ones(batch.shape[:2],device=device,dtype=torch.bool)
    pending=[];step=0;coverage=np.zeros(total_steps,dtype=np.int64)
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,cfg['steps']+1):
            if remaining()<cfg['evaluation_reserve_seconds']:raise RuntimeError('Time limit; do not auto-retry')
            if not pending:pending=list(range(total_steps));rng.shuffle(pending)
            k=pending.pop();coverage[k]+=1
            noise=torch.randn(batch.shape,device=device,generator=noise_rng)
            use_rollout=step>cfg['rollout_warmup_steps'] and rng.random()<cfg['rollout_probability']
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,a,b,state=training_prediction(model,batch,noise,valid,k,schedule,use_rollout,cfg['rollout_gradient_tail'])
                active=(b<a)&valid[...,None]
                reconstruction=(pred.float()-batch).square()[active].mean()
                assembled=torch.where(active,pred.float(),state.float())
                consistency=active_redundancy_loss(norm.denormalize(assembled),valid,active,b<1)
                loss=reconstruction+cfg['redundancy_weight']*consistency
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*min(step/cfg['warmup_steps'],1.)
            optimizer.step()
            if step==1 or step%25==0:
                live_depths=sorted(set(joint_depths()[int(j)] for j in Layout266().owner[(b[0]<a[0]).any(0).cpu()].tolist()))
                r=dict(step=step,k=k,active_depths=live_depths,active_fraction=active.float().mean().item(),rollout=use_rollout,reconstruction=reconstruction.item(),
                       redundancy=consistency.item(),loss=loss.item(),gradient_norm=float(grad),
                       elapsed_s=time.monotonic()-started,peak_memory_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
            if step%500==0:checkpoint(out,model,optimizer,step,cfg,norm,rng,noise_rng,pending)
    checkpoint(out,model,optimizer,step,cfg,norm,rng,noise_rng,pending)
    np.save(out/'training_step_coverage.npy',coverage)
    model.eval();rows=[];valid=torch.ones(clean.shape[:2],device=device,dtype=torch.bool)
    owner=Layout266().owner.to(device);depths=torch.tensor(joint_depths(),device=device)
    table=schedule.lattice(cfg['frames'],device);samples=out/'samples';samples.mkdir()
    for seed in cfg['heldout_noise_seeds']:
        if remaining()<45:raise RuntimeError('Not enough time to complete evaluation')
        noise=torch.randn(clean.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
        snapshots=[];wave_errors=[];previous=None
        def check(k,x):
            nonlocal previous
            if not torch.isfinite(x).all():raise RuntimeError('Nonfinite sample')
            if k==0:assert torch.equal(x,noise)
            else:
                inactive=(table[k]>=table[k-1])[None].expand_as(x)
                assert torch.equal(x[inactive],previous[inactive]),'Inactive state changed'
            previous=x.detach().clone()
            if k in cfg['snapshot_steps']:
                physical=norm.denormalize(x.float());snapshots.append(physical.cpu().numpy())
                wave_errors.append(dict(step=k,sigma_min=float(table[k].min()),sigma_max=float(table[k].max()),
                    xyz_mpjpe_m=(xyz266(physical)-xyz266(raw)).norm(dim=-1).mean().item(),
                    note='Intermediate state includes intentionally noisy positions; not a final reconstruction score'))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            generated=rollout(model,noise,valid,schedule,callback=check)
        known=torch.zeros_like(clean,dtype=torch.bool)
        result=reconstruction_metrics(generated.float(),clean,norm,known)
        result.pop('anchor_max_m');result.update(seed=seed,all_inactive_steps_exact=True,trajectory=wave_errors)
        physical=norm.denormalize(generated.float());x=xyz266(physical);gt=xyz266(raw)
        result['temporal_rms_ratio']=((x-x.mean(1,keepdim=True)).square().sum(-1).mean()/
                                      (gt-gt.mean(1,keepdim=True)).square().sum(-1).mean()).sqrt().item()
        result['by_depth']={str(d):dict(xyz_mpjpe_m=(x-gt)[...,depths==d,:].norm(dim=-1).mean().item(),
            feature_mse=(generated-clean).square()[...,depths[owner]==d].mean().item()) for d in range(8)}
        result['by_frame_block']={str(f):dict(xyz_mpjpe_m=(x[:,f:f+16]-gt[:,f:f+16]).norm(dim=-1).mean().item())
                                 for f in range(0,cfg['frames'],16)}
        # The old reconstruction gate includes an anchor criterion: supply 0
        # ONLY to reuse its remaining checks, never report anchor success.
        result['overfit_passed']=passes_reconstruction(dict(result,anchor_max_m=0.),cfg['overfit_gate']) and \
            cfg['temporal_ratio_min']<=result['temporal_rms_ratio']<=cfg['temporal_ratio_max']
        rows.append(result)
        np.savez_compressed(samples/f'{seed}.npz',generated266=physical.cpu().numpy(),reference266=raw.cpu().numpy(),
                            trajectory_generated266=np.stack(snapshots),snapshot_steps=np.array(cfg['snapshot_steps']),
                            initial_noise=noise.cpu().numpy())
        print(json.dumps(dict(evaluation_seed=seed,mpjpe_m=result['xyz_mpjpe_m'],passed=result['overfit_passed'])),flush=True)
    (out/'evaluation.json').write_text(json.dumps(rows,indent=2))
    summary=dict(completed_steps=step,cases=len(rows),reconstruction_pass_count=sum(r['overfit_passed'] for r in rows),
                 overfit_passed=all(r['overfit_passed'] for r in rows),no_control=True,no_text=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
