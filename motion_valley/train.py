import argparse
from contextlib import nullcontext
import copy
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import MotionDataset,collate,random_controls
from .model import JointTimeFlow
from .schedule import distance_field,trajectory_step,corrupt,PARENTS


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    raise SystemExit('Obsolete XYZ/GloVe v1 runner disabled. Use the approval-gated motion_valley.overfit v2 runner.')
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=['uniform','temporal','valley'],default='valley')
    ap.add_argument('--dataset',default='HumanML3D')
    ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--frames',type=int,default=64); ap.add_argument('--width',type=int,default=128)
    ap.add_argument('--depth',type=int,default=4); ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--sampling-steps',type=int,default=32)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--resume')
    ap.add_argument('--max-hours',type=float,default=2.); ap.add_argument('--save-every',type=int,default=500)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=True
    dataset=MotionDataset(args.data,frames=args.frames,limit=args.limit,dataset=args.dataset)
    (out/'data_audit.json').write_text(json.dumps(dataset.audit,indent=2))
    loader=DataLoader(dataset,batch_size=args.batch,shuffle=True,collate_fn=collate,
                      num_workers=args.workers,pin_memory=device.type=='cuda',drop_last=False)
    model=JointTimeFlow(width=args.width,depth=args.depth).to(device)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    start_step=0
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        if ck['args']['mode']!=args.mode: raise ValueError('resume schedule mismatch')
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema']); optimizer.load_state_dict(ck['optimizer'])
        start_step=ck['step']
    meta=dict(args=vars(args),parameters=sum(p.numel() for p in model.parameters()),
              torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
    (out/'config.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta),flush=True)
    iterator=iter(loader); start=time.monotonic(); log=(out/'train.jsonl').open('a',buffering=1)
    def save(step):
        payload=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                     step=step,args=vars(args),model_config=model.config)
        torch.save(payload,out/'last.tmp'); (out/'last.tmp').replace(out/'last.pt')
    for step in range(start_step+1,args.steps+1):
        if time.monotonic()-start>args.max_hours*3600:
            save(step-1); print('TIME_LIMIT',flush=True); break
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        x=batch['x'].to(device); valid=batch['valid'].to(device); text=batch['text'].to(device)
        mask=random_controls(valid,clean_prefix=True)
        d=distance_field(mask)
        # Select one complete field from the exact inference lattice per example.
        index=torch.randint(args.sampling_steps,(len(x),),device=device)
        s,s_next=trajectory_step(index,args.sampling_steps,mask,d,args.mode)
        noise=torch.randn_like(x); noisy=corrupt(x,noise,s,mask)
        text=text*(torch.rand(len(x),1,1,device=device)>.1)
        free=(~mask)&valid[...,None]&(s_next<s)
        optimizer.zero_grad(set_to_none=True)
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with amp:
            velocity=model(noisy,s,mask,text,valid)
            squared=(velocity.float()-(noise-x)).square().mean(-1)
            main_loss=(squared*free).sum()/free.sum().clamp_min(1)
            estimate=noisy-s[...,None]*velocity.float()
            # Finite geometry losses in uncompressed position space.
            bones=(estimate[:,:,1:]-estimate[:,:,PARENTS[1:]]).norm(dim=-1)
            gt_bones=(x[:,:,1:]-x[:,:,PARENTS[1:]]).norm(dim=-1)
            bone_loss=(bones-gt_bones).square()[valid].mean()
            vp=valid[:,1:]&valid[:,:-1]
            vel_loss=(estimate.diff(dim=1)-x.diff(dim=1)).square().mean((-1,-2))[vp].mean()
            # Apply geometric supervision only on steps with actual updates.
            # Direct flow supervision above follows the exact active time field.
            loss=main_loss+.05*bone_loss+.02*vel_loss
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        lr=args.lr*min(step/100,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.step()
        with torch.no_grad():
            decay=min(.999,(1+step)/(10+step))
            for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,1-decay)
        if step%50==0 or step==1:
            record=dict(step=step,loss=loss.item(),flow=main_loss.item(),bone=bone_loss.item(),
                        elapsed_s=time.monotonic()-start,lr=lr)
            log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
        if step%args.save_every==0 or step==args.steps: save(step)
    log.close()


if __name__=='__main__': main()
