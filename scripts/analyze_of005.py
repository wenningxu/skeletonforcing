"""Audit all OF005 samples and analyze conditional noise ensembles on CPU."""
import csv
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion_valley.noise_statistics import sample_seed, response_region, distribution_response


def main():
    src = ROOT/'outputs/OF005'; dest = ROOT/'outputs/OF005-analysis'; dest.mkdir(exist_ok=True)
    manifest = json.loads((src/'manifest.json').read_text()); cfg = manifest['config']
    rows = [json.loads(s) for s in (src/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows) == 640 and len({(r['protocol'],r['layout'],r['angle'],r['sample_index']) for r in rows}) == 640
    lookup = {(r['protocol'],r['layout'],r['angle'],r['sample_index']):r for r in rows}
    summary = json.loads((src/'summary.json').read_text())
    assert summary['completed_cases'] == len(rows) and summary['checkpoint_unchanged']
    analysis = dict(config=cfg, summary=summary, protocols={}, audited_samples=0)
    contrasts=[]; arrays={}; condition_rows=[]
    for protocol in cfg['protocols']:
        selected = [r for r in rows if r['protocol']==protocol]
        result = dict(cases=len(selected), reconstruction_pass_count=sum(r['reconstruction_passed'] for r in selected),
            free_mpjpe_m_mean=float(np.mean([r['free_xyz_mpjpe_m'] for r in selected])),
            fk_disagreement_m_mean=float(np.mean([r['fk_position_disagreement_m'] for r in selected])),
            anchor_max_m=max(r['anchor_max_m'] for r in selected), conditions={}, contrasts={})
        for li,layout in enumerate(cfg['layouts']):
            for ai,angle in enumerate(cfg['angles']):
                with np.load(src/'samples'/f'{protocol}_{layout}_{angle}.npz') as data:
                    pred=data['generated266']; truth=data['reference266']; known=data['known']; seeds=data['seeds']
                assert pred.shape==(32,64,266) and truth.shape==(64,266) and known.shape==truth.shape
                xyz=pred[:,:,4:70].reshape(32,64,22,3); gt=truth[:,4:70].reshape(64,22,3)
                pos_known=known[:,4:70].reshape(64,22,3).all(-1)
                for i in range(32):
                    row=lookup[protocol,layout,angle,i]
                    assert int(seeds[i])==row['seed']==sample_seed(protocol,li,ai,i)
                    assert row['all_steps_known_exact']
                    error=np.linalg.norm(xyz[i]-gt,axis=-1)
                    np.testing.assert_allclose(error[~pos_known].mean(),row['free_xyz_mpjpe_m'],rtol=1e-5,atol=1e-7)
                    np.testing.assert_allclose(error[pos_known].max(),row['anchor_max_m'],rtol=1e-5,atol=1e-7)
                    analysis['audited_samples']+=1
                arrays[protocol,layout,angle]=(xyz,gt,known)
                cr=[lookup[protocol,layout,angle,i] for i in range(32)]
                region=response_region(known)
                within=np.sqrt(np.square(xyz.astype(np.float64)-xyz.mean(0,dtype=np.float64))[:,region].sum(-1).mean())
                stats=dict(protocol=protocol,layout=layout,angle=angle,count=32,
                    reconstruction_pass_count=sum(r['reconstruction_passed'] for r in cr),
                    mpjpe_mm_mean=float(np.mean([r['free_xyz_mpjpe_m']*1000 for r in cr])),
                    mpjpe_mm_p05=float(np.quantile([r['free_xyz_mpjpe_m']*1000 for r in cr],.05)),
                    mpjpe_mm_p95=float(np.quantile([r['free_xyz_mpjpe_m']*1000 for r in cr],.95)),
                    within_rms_mm=float(within*1000))
                result['conditions'][f'{layout}/{angle}']=stats; condition_rows.append(stats)
            base,base_truth,_=arrays[protocol,layout,0]
            for ai,angle in enumerate(cfg['angles']):
                if angle==0: continue
                pred,truth,known=arrays[protocol,layout,angle]
                r=distribution_response(pred,base,truth,base_truth,response_region(known),
                    paired=protocol=='matched',bootstrap=cfg['bootstrap_replicates'],
                    seed=cfg['bootstrap_seed']+li*100+ai)
                result['contrasts'][f'{layout}/{angle}']=r
                contrasts.append(dict(protocol=protocol,layout=layout,angle=angle,**r))
        result['distribution_pass_count']=sum(r['distribution_response_passed'] for r in result['contrasts'].values())
        result['mean_gain']=float(np.mean([r['mean_gain'] for r in result['contrasts'].values()]))
        result['mean_relative_error']=float(np.mean([r['mean_relative_error'] for r in result['contrasts'].values()]))
        result['mean_response_rms_mm']=float(np.mean([r['mean_response_rms_m']*1000 for r in result['contrasts'].values()]))
        result['within_rms_mm']=float(np.mean([r['within_rms_mm'] for r in result['conditions'].values()]))
        if protocol=='matched': result['paired_pass_count']=sum(r['paired_pass_count'] for r in result['contrasts'].values())
        analysis['protocols'][protocol]=result
    (dest/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    for filename,records in [('conditions.csv',condition_rows),('contrasts.csv',contrasts)]:
        keys=list(dict.fromkeys(k for row in records for k in row))
        with (dest/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(records)
    colors={'independent':'tab:blue','matched':'tab:orange'}
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for li,layout in enumerate(cfg['layouts']):
        ax=axes[li]
        _,zero_truth,known=arrays['independent',layout,0]
        _,plus_truth,_=arrays['independent',layout,12]
        region=response_region(known)
        direction=(plus_truth-zero_truth)[region].reshape(-1).astype(np.float64)
        energy=direction@direction
        gt_curve=[]
        for angle in cfg['angles']:
            _,gt,_=arrays['independent',layout,angle]
            gt_curve.append(float((gt-zero_truth)[region].reshape(-1)@direction/energy))
        ax.plot(cfg['angles'],gt_curve,'k--',label='Ground truth')
        for pi,protocol in enumerate(cfg['protocols']):
            base=arrays[protocol,layout,0][0].mean(0,dtype=np.float64)
            means=[]
            for angle in cfg['angles']:
                xyz=arrays[protocol,layout,angle][0]
                values=(xyz-base)[:,region].reshape(32,-1)@direction/energy
                means.append(float(values.mean()))
                ax.scatter(np.full(32,angle)+(pi-.5)*.5,values,s=9,alpha=.35,color=colors[protocol])
            ax.plot(cfg['angles'],means,'o-',color=colors[protocol],label=protocol)
        ax.set_title(f'Layout {layout}: all 32 samples / condition')
        ax.set_xlabel('Control state angle (degrees)'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Projection along +12-degree GT change\n(centered on each protocol zero-state mean)')
    fig.tight_layout(); fig.savefig(dest/'conditional_response.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=[f'{l}/{a:+}' for l in cfg['layouts'] for a in cfg['angles'] if a]
    for pi,protocol in enumerate(cfg['protocols']):
        records=list(analysis['protocols'][protocol]['contrasts'].values())
        xs=np.arange(8)+(pi-.5)*.15; gains=np.array([r['mean_gain'] for r in records])
        lo=np.array([r['gain_family_ci'][0] for r in records]); hi=np.array([r['gain_family_ci'][1] for r in records])
        axes[0].vlines(xs,lo,hi,color=colors[protocol]); axes[0].scatter(xs,gains,s=18,color=colors[protocol],label=protocol)
        axes[1].plot(xs,[r['mean_relative_error'] for r in records],'o-',color=colors[protocol],label=protocol)
    axes[0].axhline(0,c='gray',ls='--'); axes[0].set_title('Mean gain with 99.375% bootstrap intervals')
    axes[1].axhline(.5,c='gray',ls='--'); axes[1].set_title('Mean-response relative error (ideal 0)')
    for ax in axes:
        ax.set_xticks(range(8),labels,rotation=40); ax.set_xlabel('Layout / nonzero control angle'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(dest/'contrast_intervals.png',dpi=170); plt.close(fig)
    print(json.dumps({p:{k:v for k,v in r.items() if k not in ['conditions','contrasts']} for p,r in analysis['protocols'].items()},indent=2))
    print('Independent NumPy sample audits:',analysis['audited_samples'])


if __name__=='__main__': main()
