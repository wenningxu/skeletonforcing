"""Audit the fixed traditional inpainting experiment; CPU only."""
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_of003 import GROUPS, group_name, statistics, failure_reasons
from motion_valley.noise_statistics import response_region, distribution_response


def main():
    src=ROOT/'outputs/OF006-inpaint'; dest=ROOT/'outputs/OF006-analysis'; dest.mkdir(exist_ok=True)
    assert (src/'SUITE_COMPLETED').exists()
    manifest=json.loads((src/'manifest.json').read_text()); cfg=manifest['config']
    assert cfg['mode']=='uniform' and cfg['input_mode']=='concat_mask'
    summary=json.loads((src/'summary.json').read_text())
    rows=[json.loads(s) for s in (src/'evaluation.jsonl').read_text().splitlines()]
    independent=[json.loads(s) for s in (src/'independent.jsonl').read_text().splitlines()]
    logs=[json.loads(s) for s in (src/'train.jsonl').read_text().splitlines()]
    assert len(rows)==40 and len(independent)==160 and logs[-1]['step']==4000
    assert summary['completed_evaluation'] and summary['completed_requested_steps']
    assert len({(r['seed'],r['layout'],r['angle']) for r in rows})==40
    assert len({r['seed'] for r in independent})==160
    assert all(r['all_steps_known_exact'] for r in rows+independent)
    reconstruction=lambda r:not failure_reasons(r,cfg['overfit_gate'])
    response=lambda r:.5<=r['response_gain']<=1.5 and r['response_relative_error']<.5
    result=dict(config=cfg,summary=summary,parameters=manifest['parameters'],gpu=manifest['gpu'],
        paired=dict(overall=statistics(rows),groups={},reconstruction_pass_count=sum(map(reconstruction,rows)),
            response_pass_count=sum(response(r) for r in rows if r['angle'])),
        independent=dict(overall=statistics(independent),reconstruction_pass_count=sum(map(reconstruction,independent)),conditions={},contrasts={}),
        audited_samples=0,train_seconds=logs[-1]['elapsed_s'],
        peak_memory_allocated_gb=max(r['peak_memory_allocated_gb'] for r in logs))
    for g in GROUPS:
        selected=[r for r in rows if group_name(r)==g]
        st=dict(statistics=statistics(selected),reconstruction_pass_count=sum(map(reconstruction,selected)),
                response_pass_count=sum(response(r) for r in selected if r['angle']))
        assert st['reconstruction_pass_count']==summary[g]['reconstruction_pass_count']
        assert st['response_pass_count']==summary[g]['response_pass_count']
        result['paired']['groups'][g]=st
    failures={}
    for r in rows+independent:
        for key in failure_reasons(r,cfg['overfit_gate']): failures[key]=failures.get(key,0)+1
    result['failure_counts_all200']=failures
    for r in rows:
        if r['seed']!=cfg['heldout_noise_seeds'][0]:continue
        prefix=f"00_{r['layout']}"
        with np.load(src/'samples'/f"{prefix}_{r['angle']}.npz") as data:
            xyz=data['generated266'][0,:,4:70].reshape(64,22,3)
            gt=data['reference266'][0,:,4:70].reshape(64,22,3)
            known=data['known'][0]; position_known=known[:,4:70].reshape(64,22,3).all(-1)
        err=np.linalg.norm(xyz-gt,axis=-1)
        np.testing.assert_allclose(err[~position_known].mean(),r['free_xyz_mpjpe_m'],rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose(err[position_known].max(),r['anchor_max_m'],atol=2e-7)
        if r['angle']:
            with np.load(src/'samples'/f'{prefix}_0.npz') as data:
                delta=xyz-data['generated266'][0,:,4:70].reshape(64,22,3)
                target=gt-data['reference266'][0,:,4:70].reshape(64,22,3)
            region=response_region(known); a=delta[region]; b=target[region]; energy=(b*b).sum()
            np.testing.assert_allclose([(a*b).sum()/energy,np.sqrt(((a-b)**2).sum()/energy)],
                [r['response_gain'],r['response_relative_error']],rtol=1e-5,atol=1e-6)
        result['audited_samples']+=1
    lookup={(r['layout'],r['angle'],r['sample_index']):r for r in independent}; arrays={}
    anchor_max_raw=0.
    for li,layout in enumerate(['A','D']):
        for ai,angle in enumerate([-12,-6,0,6,12]):
            with np.load(src/'samples'/f'independent_{layout}_{angle}.npz') as data:
                pred=data['generated266']; truth=data['reference266']; known=data['known']; seeds=data['seeds']
            assert pred.shape==(16,64,266) and truth.shape==(64,266)
            xyz=pred[:,:,4:70].reshape(16,64,22,3); gt=truth[:,4:70].reshape(64,22,3)
            position_known=known[:,4:70].reshape(64,22,3).all(-1)
            for i in range(16):
                r=lookup[layout,angle,i]
                assert seeds[i]==r['seed']==820000+li*10000+ai*1000+i
                err=np.linalg.norm(xyz[i]-gt,axis=-1)
                np.testing.assert_allclose(err[~position_known].mean(),r['free_xyz_mpjpe_m'],rtol=1e-5,atol=1e-7)
                # Log metric uses normalized-roundtrip GT; audit against original raw GT.
                np.testing.assert_allclose(err[position_known].max(),r['anchor_max_m'],atol=2e-7)
                anchor_max_raw=max(anchor_max_raw,float(err[position_known].max()))
                result['audited_samples']+=1
            arrays[layout,angle]=(xyz,gt,known)
            group=[lookup[layout,angle,i] for i in range(16)]
            region=response_region(known)
            within=np.sqrt(np.square(xyz.astype(np.float64)-xyz.mean(0,dtype=np.float64))[:,region].sum(-1).mean())
            result['independent']['conditions'][f'{layout}/{angle}']=dict(statistics=statistics(group),within_rms_mm=float(within*1000))
        base,base_truth,_=arrays[layout,0]
        for ai,angle in enumerate([-12,-6,0,6,12]):
            if angle==0:continue
            xyz,gt,known=arrays[layout,angle]
            result['independent']['contrasts'][f'{layout}/{angle}']=distribution_response(xyz,base,gt,base_truth,response_region(known),bootstrap=4000,seed=923+li*100+ai)
    result['anchor_max_raw_m']=anchor_max_raw
    ind=result['independent']; cs=list(ind['contrasts'].values())
    ind.update(distribution_pass_count=sum(r['distribution_response_passed'] for r in cs),
        mean_gain=float(np.mean([r['mean_gain'] for r in cs])),
        mean_relative_error=float(np.mean([r['mean_relative_error'] for r in cs])),
        mean_response_rms_mm=float(np.mean([r['mean_response_rms_m']*1000 for r in cs])),
        within_rms_mm=float(np.mean([r['within_rms_mm'] for r in ind['conditions'].values()])))
    old=json.loads((ROOT/'outputs/OF004-analysis/analysis.json').read_text())['arms']['A']
    old_noise=json.loads((ROOT/'outputs/OF005-analysis/analysis.json').read_text())['protocols']['independent']
    result['comparison']=dict(OF004_paired=old['overall'],OF005_independent={k:v for k,v in old_noise.items() if k not in ['conditions','contrasts']})
    (dest/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for layout,ax in zip(['A','D'],axes):
        _,gt0,known=arrays[layout,0]; _,gt12,_=arrays[layout,12]; region=response_region(known)
        direction=(gt12-gt0)[region].reshape(-1).astype(np.float64); energy=direction@direction
        base=arrays[layout,0][0].mean(0,dtype=np.float64); means=[]; truths=[]
        for angle in [-12,-6,0,6,12]:
            xyz,gt,_=arrays[layout,angle]
            values=(xyz-base)[:,region].reshape(16,-1)@direction/energy
            ax.scatter(np.full(16,angle),values,s=12,alpha=.35,color='tab:blue')
            means.append(values.mean()); truths.append((gt-gt0)[region].reshape(-1)@direction/energy)
        ax.plot([-12,-6,0,6,12],truths,'k--',label='Ground truth')
        ax.plot([-12,-6,0,6,12],means,'o-',label='Inpainting mean')
        ax.set_title(f'Layout {layout}; 16 independent samples/state'); ax.set_xlabel('State angle (degrees)'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Free-region response along GT +12-degree change')
    fig.tight_layout();fig.savefig(dest/'conditional_response.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,factor,label in zip(axes,['free_xyz_mpjpe_m','response_gain','response_relative_error'],[1000,1,1],['Free MPJPE (mm)','Response gain (ideal 1)','Response relative error (ideal 0)']):
        ax.bar(['Valley OF004','Inpaint OF006'],[old['overall'][key]['mean']*factor,result['paired']['overall'][key]['mean']*factor])
        ax.set_title(label);ax.grid(axis='y',alpha=.2)
    # Keep the practical ideal visible; avoid magnifying near-zero gains into
    # apparently substantial positive/negative effects.
    axes[1].set_ylim(-.05,1.05);axes[1].axhline(1,color='gray',ls='--')
    for x,gain in enumerate([old['overall']['response_gain']['mean'],result['paired']['overall']['response_gain']['mean']]):
        axes[1].text(x,.06,f'{gain:.6f}',ha='center')
    fig.tight_layout();fig.savefig(dest/'paired_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(paired=result['paired']['overall'],independent={k:v for k,v in ind.items() if k not in ['conditions','contrasts','overall']},audited=result['audited_samples'],failure_counts=failures),indent=2))


if __name__=='__main__':main()
