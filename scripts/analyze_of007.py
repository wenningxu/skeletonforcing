"""CPU evidence audit and paired, common-region control-factor comparisons."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_of003 import statistics,failure_reasons


def stats(rows):
    nonzero=[r for r in rows if r['angle']]
    result=dict(cases=len(rows),response_cases=len(nonzero),
        response_pass_count=sum(.5<=r['common_response_gain']<=1.5 and r['common_response_relative_error']<.5 for r in nonzero))
    for key in ['common_response_gain','common_response_relative_error','common_response_actual_rms_m','common_response_target_rms_m','common_region_mpjpe_m','free_xyz_mpjpe_m','fk_position_disagreement_m']:
        values=[r[key] for r in rows if key in r]
        result[key]=dict(mean=float(np.mean(values)),min=float(np.min(values)),max=float(np.max(values)))
    return result


def main():
    root=ROOT/'outputs/OF007';dest=ROOT/'outputs/OF007-analysis';dest.mkdir(exist_ok=True)
    assert (root/'SUITE_COMPLETED').exists()
    manifest=json.loads((root/'manifest.json').read_text());plan=manifest['plan']
    cfg=json.loads((ROOT/'configs/inpaint_baseline_v1.json').read_text());gates=cfg['overfit_gate']
    results={};arrays={};all_rows={};audit=0;routine_audit=0
    with np.load(ROOT/plan['bank_file']) as bank:bank_states=bank['states'];bank_angles=bank['angles'].tolist()
    for model in ['baseline','density','data','coverage']:
        src=root/(model+'_eval');rows=[json.loads(x) for x in (src/'evaluation.jsonl').read_text().splitlines()]
        assert len(rows)==160 and all(r['all_steps_known_exact'] for r in rows)
        lookup={(r['layout'],r['seed'],r['angle']):r for r in rows};assert len(lookup)==160
        common=np.load(src/'common_known.npy')[0,:,4:70].reshape(64,22,3).all(-1)
        region=np.zeros((64,22),bool);region[36:61,[17,19,21]]=True;region &= ~common;assert region.sum()==48
        for layout in plan['eval_layouts']:
            with np.load(src/'samples'/f'{layout}.npz') as d:
                gen=d['generated266'];truth=d['reference266'];known=d['known'];seeds=d['seeds'];angles=d['angles']
            assert gen.shape==truth.shape==known.shape==(40,64,266)
            assert np.isfinite(gen).all() and known.dtype==np.bool_
            expected_known=np.zeros((64,266),bool)
            for frame,joint in plan['eval_layouts'][layout]:expected_known[frame,4+3*joint:7+3*joint]=True
            assert np.all(known==expected_known[None])
            xyz=gen[:,:,4:70].reshape(40,64,22,3);gt=truth[:,:,4:70].reshape(40,64,22,3)
            ix={(int(s),int(a)):i for i,(s,a) in enumerate(zip(seeds,angles))};assert len(ix)==40
            for i,(seed,angle) in enumerate(zip(seeds,angles)):
                r=lookup[layout,int(seed),int(angle)]
                np.testing.assert_array_equal(truth[i],bank_states[0,bank_angles.index(int(angle))])
                mask=known[i,:,4:70].reshape(64,22,3).all(-1)
                err=np.linalg.norm(xyz[i]-gt[i],axis=-1)
                np.testing.assert_allclose([err[~mask].mean(),err[mask].max()],[r['free_xyz_mpjpe_m'],r['anchor_max_m']],rtol=1e-5,atol=2e-7)
                r['common_region_mpjpe_m']=float(err[region].mean())
                arrays[model,layout,int(seed),int(angle)]=(xyz[i],gt[i])
                if angle:
                    base=ix[int(seed),0];a=xyz[i]-xyz[base];b=gt[i]-gt[base]
                    native=np.zeros((64,22),bool);native[36:61,[17,19,21]]=True;native &= ~mask
                    for prefix,where in [('common_',region),('',native)]:
                        aa=a[where];bb=b[where];energy=(bb*bb).sum()
                        values=[float((aa*bb).sum()/energy),float(np.sqrt(((aa-bb)**2).sum()/energy))]
                        np.testing.assert_allclose(values,[r[prefix+'response_gain'],r[prefix+'response_relative_error']],rtol=1e-5,atol=1e-6)
                audit+=1
        primary_layouts=['A15','D15'] if model=='density' else ['A3','D3']
        primary=[r for r in rows if r['layout'] in primary_layouts]
        result=dict(primary=stats(primary),by_layout={l:stats([r for r in rows if r['layout']==l]) for l in plan['eval_layouts']},
            primary_layouts=primary_layouts,reconstruction_pass_count=sum(not failure_reasons(r,gates) for r in primary),
            primary_conditions={})
        for layout in primary_layouts:
            for angle in [-12,-6,6,12]:
                group=[r for r in primary if r['layout']==layout and r['angle']==angle]
                st=stats(group);st['group_passed']=st['response_pass_count']>=6
                result['primary_conditions'][f'{layout}/{angle}']=st
        result['primary_groups_passed']=sum(r['group_passed'] for r in result['primary_conditions'].values())
        if model!='baseline':
            routine=[json.loads(s) for s in (root/model/'evaluation.jsonl').read_text().splitlines()]
            logs=[json.loads(s) for s in (root/model/'train.jsonl').read_text().splitlines()]
            summary=json.loads((root/model/'summary.json').read_text())
            assert logs[-1]['step']==4000 and summary['actual_cases']==len(routine)
            for row in routine:
                if row['seed']!=2026:continue
                filename=f"{row['clip_index']:02d}_{row['layout']}_{row['angle']}.npz"
                with np.load(root/model/'samples'/filename) as d:
                    pred=d['generated266'][0,:,4:70].reshape(64,22,3)
                    truth=d['reference266'][0,:,4:70].reshape(64,22,3)
                    mask=d['known'][0,:,4:70].reshape(64,22,3).all(-1)
                error=np.linalg.norm(pred-truth,axis=-1)
                np.testing.assert_allclose([error[~mask].mean(),error[mask].max()],[row['free_xyz_mpjpe_m'],row['anchor_max_m']],rtol=1e-5,atol=2e-7)
                if row['angle']:
                    zero=f"{row['clip_index']:02d}_{row['layout']}_0.npz"
                    with np.load(root/model/'samples'/zero) as d:
                        a=pred-d['generated266'][0,:,4:70].reshape(64,22,3)
                        b=truth-d['reference266'][0,:,4:70].reshape(64,22,3)
                    free=np.zeros((64,22),bool);free[36:61,[17,19,21]]=True;free &= ~mask
                    a=a[free];b=b[free];energy=(b*b).sum()
                    np.testing.assert_allclose([(a*b).sum()/energy,np.sqrt(((a-b)**2).sum()/energy)],
                        [row['response_gain'],row['response_relative_error']],rtol=1e-5,atol=1e-6)
                routine_audit+=1
            result['routine']=dict(statistics=statistics(routine),summary=summary,
                by_clip={str(i):statistics([r for r in routine if r['clip_index']==i]) for i in sorted({r['clip_index'] for r in routine})})
            result['train_seconds']=logs[-1]['elapsed_s'];result['peak_allocated_gb']=max(r['peak_memory_allocated_gb'] for r in logs)
            result['last_1000_logged_training_steps']={}
            for kind in [False,True]:
                values=[r['reconstruction'] for r in logs if r['step']>3000 and r['rollout']==kind]
                result['last_1000_logged_training_steps']['rollout' if kind else 'teacher']=dict(
                    logged_count=len(values),mean=float(np.mean(values)),median=float(np.median(values)))
        results[model]=result;all_rows[model]=primary
    # Resample seed blocks; angles/layouts sharing noise are not independent replicates.
    rng=np.random.default_rng(1707);weights=rng.multinomial(8,np.ones(8)/8,size=4000)/8
    seed_values={m:np.array([np.mean([r['common_response_gain'] for r in rows if r['angle'] and r['seed']==s]) for s in plan['eval_seeds']]) for m,rows in all_rows.items()}
    for model in ['density','data','coverage']:
        delta=seed_values[model]-seed_values['baseline'];boot=weights@delta
        results[model]['gain_delta_vs_baseline']=dict(mean=float(delta.mean()),seed_block_deltas=delta.tolist(),
            bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
            bootstrap_family_ci_98_333=np.quantile(boot,[.05/(2*3),1-.05/(2*3)]).tolist(),
            note='Eight sampling-noise blocks, one checkpoint per model; not training-seed uncertainty')
    output=dict(plan=plan,models=results,audited_common_protocol_samples=audit,audited_routine_samples=routine_audit)
    (dest/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4));names=list(results)
    for ax,key,mult,label in zip(axes,['common_response_gain','common_response_relative_error','common_region_mpjpe_m'],[1,1,1000],['Common-region gain (ideal 1)','Response relative error (ideal 0)','Common-region MPJPE (mm)']):
        vals=[results[m]['primary'][key]['mean']*mult for m in names]
        ax.bar(names,vals);ax.set_title(label);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
        if key=='common_response_gain':ax.axhline(1,c='gray',ls='--');ax.set_ylim(min(-.05,min(vals)*1.2),max(1.05,max(vals)*1.2))
    fig.tight_layout();fig.savefig(dest/'factor_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for axis,(sparse,dense) in zip(axes,[('A3','A15'),('D3','D15')]):
        for model in names:
            layout=dense if model=='density' else sparse
            _,gt0=arrays[model,layout,2026,0];_,gt12=arrays[model,layout,2026,12]
            direction=(gt12-gt0)[region].reshape(-1).astype(np.float64);energy=direction@direction
            means=[];stds=[]
            for angle in [-12,-6,0,6,12]:
                values=[]
                for seed in plan['eval_seeds']:
                    pred,_=arrays[model,layout,seed,angle];base,_=arrays[model,layout,seed,0]
                    values.append((pred-base)[region].reshape(-1)@direction/energy)
                means.append(np.mean(values));stds.append(np.std(values))
            axis.errorbar([-12,-6,0,6,12],means,yerr=stds,marker='o',label=model,capsize=2)
        truth_curve=[]
        for angle in [-12,-6,0,6,12]:
            _,gt=arrays['baseline',sparse,2026,angle]
            truth_curve.append((gt-gt0)[region].reshape(-1)@direction/energy)
        axis.plot([-12,-6,0,6,12],truth_curve,'k--',label='GT');axis.set_title(sparse+' / '+dense);axis.set_xlabel('Control angle (degrees)');axis.grid(alpha=.2);axis.legend()
    axes[0].set_ylabel('Paired common-region response along GT change')
    fig.tight_layout();fig.savefig(dest/'paired_response.png',dpi=170);plt.close(fig)
    print(json.dumps({m:dict(primary=r['primary'],groups_passed=r['primary_groups_passed'],reconstruction_pass_count=r['reconstruction_pass_count'],gain_delta=r.get('gain_delta_vs_baseline')) for m,r in results.items()},indent=2))
    print('Audited samples',audit)


if __name__=='__main__':main()
