import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import MotionDataset,collate,random_controls
from .model import JointTimeFlow
from .schedule import sample
from .geometry import geometric_metrics
from .features import xyz_to_features
from .evaluator import Evaluator,embedding_metrics
from .train import seed_all


def main():
    raise SystemExit('Obsolete XYZ v1 evaluator disabled. v2 overfit evaluates raw 266D plus deterministic 263D export.')
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--split',default='val',choices=['val','test'])
    ap.add_argument('--dataset',default='HumanML3D'); ap.add_argument('--limit',type=int,default=64)
    ap.add_argument('--frames',type=int,default=196); ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--batch',type=int,default=8); ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--skip-embeddings',action='store_true')
    ap.add_argument('--control',choices=['random','wrist','keyframe'],default='random')
    args=ap.parse_args(); seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if args.steps!=ck['args']['sampling_steps']:
        raise ValueError('Inference step lattice must match training --sampling-steps')
    net=JointTimeFlow(**ck['model_config']).to(device).eval(); net.load_state_dict(ck['ema'])
    dataset=MotionDataset(args.data,split=args.split,frames=args.frames,limit=args.limit,dataset=args.dataset)
    loader=DataLoader(dataset,batch_size=args.batch,shuffle=False,collate_fn=collate)
    evaluator=None if args.skip_embeddings else Evaluator(args.data,device)
    result_dir=Path(args.output); result_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; gen_emb=[]; gt_emb=[]; raw_emb=[]; txt_emb=[]; failures=[]; times=[]; peak_anchor=0.
    manifest=[]; first=True; feature_errors=[]
    generator=torch.Generator(device=device).manual_seed(args.seed)
    for batch in loader:
        x=batch['x'].to(device); text=batch['text'].to(device); valid=batch['valid'].to(device)
        mask=random_controls(valid,generator,max_anchors=8)
        if args.control!='random':
            mask.zero_()
            for i,n in enumerate(batch['lengths']):
                if args.control=='wrist': mask[i,n//2,20]=True
                else: mask[i,n//2,:]=True
        # Fixed mask/noise generator makes schedule comparisons paired.
        noise=torch.randn(x.shape,device=device,generator=generator)
        def check(k,state):
            nonlocal peak_anchor
            if not torch.equal(state[mask],x[mask]): raise AssertionError('anchor changed during sampling')
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with amp:
            if first:
                sample(net,x,mask,text,valid,steps=2,mode=ck['args']['mode'],initial_noise=noise)
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            pred=sample(net,x,mask,text,valid,steps=args.steps,mode=ck['args']['mode'],initial_noise=noise,callback=check)
            if device.type=='cuda': torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        for i,n in enumerate(batch['lengths']):
            metrics=geometric_metrics(pred[i:i+1],x[i:i+1],mask[i:i+1],valid[i:i+1])
            metrics['name']=batch['names'][i]; rows.append(metrics)
            manifest.append(dict(name=batch['names'][i],length=n,anchors=mask[i].nonzero().cpu().tolist()))
        if first:
            np.savez_compressed(result_dir/'samples.npz',pred=pred.cpu().numpy(),truth=x.cpu().numpy(),
                                mask=mask.cpu().numpy(),lengths=np.array(batch['lengths']))
        first=False
        if evaluator:
            generated=[]; reference=[]; original=[]; tokens=[]
            for i,n in enumerate(batch['lengths']):
                try:
                    f,_=xyz_to_features(pred[i,:n].cpu().numpy())
                    g,_=xyz_to_features(x[i,:n].cpu().numpy())
                    generated.append(f); reference.append(g); original.append(batch['feature'][i,:n-1].numpy()); tokens.append(batch['tokens'][i])
                    feature_errors.append(float(np.mean((g-original[-1])**2)))
                except (ValueError,FloatingPointError) as e:
                    failures.append(dict(name=batch['names'][i],reason=str(e)))
            if generated:
                gen_emb.append(evaluator.motions(generated)); gt_emb.append(evaluator.motions(reference))
                raw_emb.append(evaluator.motions(original)); txt_emb.append(evaluator.texts(tokens))
        print('evaluated',len(rows),'/',len(dataset),flush=True)
    result=dict(args=vars(args),checkpoint_step=ck['step'],mode=ck['args']['mode'],count=len(rows),
                stage='pilot' if args.limit else 'full_split',conversion_failures=failures,
                latency_kind='offline batch; includes per-step anchor assertions; NOT streaming latency',
                batch_latency_p50_s=float(np.median(times)),batch_latency_p95_s=float(np.percentile(times,95)),
                offline_frames_per_second=sum(x['length'] for x in manifest)/sum(times),
                hard_anchor_all_steps_exact=True)
    for key in rows[0]:
        if key=='name': continue
        values=[r[key] for r in rows if r[key] is not None]
        result[key]=float(np.mean(values)) if values else None
    result['anchor_max_m']=max(r['anchor_max_m'] or 0 for r in rows)
    if evaluator and gen_emb and not failures:
        gen=np.concatenate(gen_emb); gt=np.concatenate(gt_emb); raw=np.concatenate(raw_emb); txt=np.concatenate(txt_emb)
        result['evaluator_vs_xyz_roundtrip_reference']=embedding_metrics(gen,gt,txt,args.seed)
        result['evaluator_vs_original_features']=embedding_metrics(gen,raw,txt,args.seed)
        result['ground_truth_conversion_floor']=embedding_metrics(gt,raw,txt,args.seed)
        result['ground_truth_feature_roundtrip_mse']=float(np.mean(feature_errors))
        np.savez_compressed(result_dir/'embeddings.npz',generated=gen,reference=gt,original=raw,text=txt)
    elif evaluator:
        result['embedding_metrics_withheld']='conversion failures; no silent subset selection'
    (result_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    (result_dir/'per_motion.json').write_text(json.dumps(rows,indent=2))
    (result_dir/'controls.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
