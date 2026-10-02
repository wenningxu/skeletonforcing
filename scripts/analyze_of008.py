"""Independent CPU audit of Kimodo266 response and paired comparison to OF007 baseline."""
import json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_of003 import statistics,failure_reasons
from analyze_of007 import stats


def main():
    output=root/'outputs/OF008-analysis';output.mkdir(exist_ok=True)
    cfg=json.loads((root/'configs/kimodo266_overfit_v1.json').read_text())
    plan=json.loads((root/'configs/control_factors_v1.json').read_text())
    with np.load(root/cfg['bank_file']) as d:truth_bank=d['states'];bank_angles=d['angles'].tolist()
    models={};primary_rows={};arrays={};audits={}
    for name,src in [('baseline',root/'outputs/OF007/baseline_eval'),('kimodo266',root/'outputs/OF008/common_eval')]:
        rows=[json.loads(s) for s in (src/'evaluation.jsonl').read_text().splitlines()]
        assert len(rows)==160 and all(r['all_steps_known_exact'] for r in rows)
        lookup={(r['layout'],r['seed'],r['angle']):r for r in rows};assert len(lookup)==160
        known_union=np.load(src/'common_known.npy')[0,:,4:70].reshape(64,22,3).all(-1)
        region=np.zeros((64,22),bool);region[36:61,[17,19,21]]=True;region &= ~known_union
        assert region.sum()==48
        audited=0
        for layout,anchors in plan['eval_layouts'].items():
            with np.load(src/'samples'/f'{layout}.npz') as d:
                generated=d['generated266'];refs=d['reference266'];masks=d['known'];seeds=d['seeds'];angles=d['angles']
            assert generated.shape==refs.shape==masks.shape==(40,64,266)
            assert np.isfinite(generated).all() and masks.dtype==np.bool_
            expected=np.zeros((64,266),bool)
            for f,j in anchors:expected[f,4+3*j:7+3*j]=True
            assert (masks==expected).all()
            ix={(int(seed),int(angle)):i for i,(seed,angle) in enumerate(zip(seeds,angles))}
            assert len(ix)==40
            xyz=generated[:,:,4:70].reshape(40,64,22,3);gt=refs[:,:,4:70].reshape(40,64,22,3)
            for i,(seed,angle) in enumerate(zip(seeds,angles)):
                row=lookup[layout,int(seed),int(angle)]
                np.testing.assert_array_equal(refs[i],truth_bank[0,bank_angles.index(int(angle))])
                known=masks[i,:,4:70].reshape(64,22,3).all(-1);error=np.linalg.norm(xyz[i]-gt[i],axis=-1)
                np.testing.assert_allclose([error[~known].mean(),error[known].max()],
                    [row['free_xyz_mpjpe_m'],row['anchor_max_m']],rtol=1e-5,atol=2e-7)
                row['common_region_mpjpe_m']=float(error[region].mean())
                arrays[name,layout,int(seed),int(angle)]=(xyz[i],gt[i])
                if angle:
                    zero=ix[int(seed),0];a=xyz[i]-xyz[zero];b=gt[i]-gt[zero]
                    own=np.zeros((64,22),bool);own[36:61,[17,19,21]]=True;own &= ~known
                    for prefix,where in [('common_',region),('',own)]:
                        aa=a[where];bb=b[where];energy=(bb*bb).sum()
                        np.testing.assert_allclose([(aa*bb).sum()/energy,np.sqrt(((aa-bb)**2).sum()/energy)],
                            [row[prefix+'response_gain'],row[prefix+'response_relative_error']],rtol=1e-5,atol=1e-6)
                audited+=1
        primary=[r for r in rows if r['layout'] in ['A3','D3']];primary_rows[name]=primary
        groups={}
        for layout in ['A3','D3']:
            for angle in [-12,-6,6,12]:
                st=stats([r for r in primary if r['layout']==layout and r['angle']==angle])
                st['passed']=st['response_pass_count']>=6;groups[f'{layout}/{angle}']=st
        models[name]=dict(primary=stats(primary),groups=groups,groups_passed=sum(s['passed'] for s in groups.values()),
            by_layout={l:stats([r for r in rows if r['layout']==l]) for l in plan['eval_layouts']},
            reconstruction_pass_count=sum(not failure_reasons(r,cfg['overfit_gate']) for r in primary),
            max_physical_anchor_error_m=max(r['anchor_max_m'] for r in rows))
        audits[name]=audited
    seeds=plan['eval_seeds']
    gains={m:np.array([np.mean([r['common_response_gain'] for r in rows if r['angle'] and r['seed']==s]) for s in seeds]) for m,rows in primary_rows.items()}
    delta=gains['kimodo266']-gains['baseline'];rng=np.random.default_rng(1808)
    boot=(rng.multinomial(8,np.ones(8)/8,size=4000)/8)@delta
    difference=dict(mean=float(delta.mean()),seed_block_deltas=delta.tolist(),ci95=np.quantile(boot,[.025,.975]).tolist(),
        limitation='One training seed per model; interval describes sampling noise only')
    src=root/'outputs/OF008/train';rows=[json.loads(s) for s in (src/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==80 and all(r['all_steps_known_exact'] for r in rows)
    routine_audit=0
    for row in rows:
        if row['seed']!=2026:continue
        prefix=f"00_{row['layout']}"
        with np.load(src/'samples'/f"{prefix}_{row['angle']}.npz") as d:
            pred=d['generated266'][0,:,4:70].reshape(64,22,3);truth=d['reference266'][0,:,4:70].reshape(64,22,3)
            mask=d['known'][0,:,4:70].reshape(64,22,3).all(-1)
        error=np.linalg.norm(pred-truth,axis=-1)
        np.testing.assert_allclose([error[~mask].mean(),error[mask].max()],
            [row['free_xyz_mpjpe_m'],row['anchor_max_m']],rtol=1e-5,atol=2e-7)
        if row['angle']:
            with np.load(src/'samples'/f'{prefix}_0.npz') as d:
                a=pred-d['generated266'][0,:,4:70].reshape(64,22,3)
                b=truth-d['reference266'][0,:,4:70].reshape(64,22,3)
            where=np.zeros((64,22),bool);where[36:61,[17,19,21]]=True;where &= ~mask
            a=a[where];b=b[where];energy=(b*b).sum()
            np.testing.assert_allclose([(a*b).sum()/energy,np.sqrt(((a-b)**2).sum()/energy)],
                [row['response_gain'],row['response_relative_error']],rtol=1e-5,atol=1e-6)
        routine_audit+=1
    logs=[json.loads(s) for s in (src/'train.jsonl').read_text().splitlines()]
    assert logs[-1]['step']==4000
    record=dict(models=models,gain_difference=difference,common_samples_audited=audits,
        new_routine_samples_audited=routine_audit,routine=statistics(rows),
        routine_summary=json.loads((src/'summary.json').read_text()),train_seconds=logs[-1]['elapsed_s'],
        peak_memory_allocated_gb=max(r['peak_memory_allocated_gb'] for r in logs))
    # Post-hoc description of temporal collapse; this is not a causal test.
    temporal={}
    for name in models:
        values=[]
        for layout in ['A3','D3']:
            known=np.zeros((64,22),bool)
            for f,j in plan['eval_layouts'][layout]:known[f,j]=True
            for seed in seeds:
                for angle in [-12,-6,0,6,12]:
                    pred,gt=arrays[name,layout,seed,angle]
                    count=(~known).sum(0)[:,None]
                    pm=(pred*(~known)[...,None]).sum(0)/count
                    gm=(gt*(~known)[...,None]).sum(0)/count
                    pvar=float(np.mean(np.sum((pred-pm)**2,-1)[~known]))
                    gvar=float(np.mean(np.sum((gt-gm)**2,-1)[~known]))
                    values.append((np.sqrt(pvar),np.sqrt(gvar),np.sqrt(pvar/gvar)))
        temporal[name]=dict(zip(['prediction_temporal_rms_m','gt_temporal_rms_m','temporal_rms_ratio'],
                               np.mean(values,axis=0).tolist()))
    record['posthoc_temporal_variation']=temporal
    record['last_1000_logged_steps']={}
    for kind in [False,True]:
        values=[r['reconstruction'] for r in logs if r['step']>3000 and r['rollout']==kind]
        record['last_1000_logged_steps']['rollout' if kind else 'teacher']=dict(
            logged_count=len(values),mean=float(np.mean(values)),median=float(np.median(values)))
    record['routine_failure_counts']={key:sum(key in failure_reasons(r,cfg['overfit_gate']) for r in rows)
        for key in ['feature_mse','free_xyz_mpjpe_m','rot6d_normalized_mse','velocity_normalized_mse',
                    'velocity_error_vs_static_ratio','anchor_max_m','degenerate_rotation_count','fk_position_disagreement_m']}
    (output/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,layout in zip(axes,['A3','D3']):
        _,gt0=arrays['baseline',layout,2026,0];_,gt12=arrays['baseline',layout,2026,12]
        direction=(gt12-gt0)[region].reshape(-1).astype(np.float64);energy=direction@direction
        for name in models:
            means=[];stds=[]
            for angle in [-12,-6,0,6,12]:
                values=[]
                for seed in seeds:
                    pred,_=arrays[name,layout,seed,angle];base,_=arrays[name,layout,seed,0]
                    values.append((pred-base)[region].reshape(-1)@direction/energy)
                means.append(np.mean(values));stds.append(np.std(values))
            ax.errorbar([-12,-6,0,6,12],means,yerr=stds,marker='o',capsize=2,label=name)
        actual=[]
        for angle in [-12,-6,0,6,12]:
            _,gt=arrays['baseline',layout,2026,angle];actual.append((gt-gt0)[region].reshape(-1)@direction/energy)
        ax.plot([-12,-6,0,6,12],actual,'k--',label='GT');ax.set_title(layout);ax.set_xlabel('Control angle (degrees)');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Paired response along GT change');fig.tight_layout();fig.savefig(output/'paired_response.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    _,gt=arrays['baseline','A3',2026,0]
    for axis,ax in enumerate(axes):
        ax.plot(gt[:,21,axis],label='GT',color='black',linestyle='--')
        for name in models:
            pred,_=arrays[name,'A3',2026,0]
            ax.plot(pred[:,21,axis],label=name)
        ax.axvline(48,color='gray',alpha=.4,linestyle=':')
        ax.set_ylabel(f'Wrist {"XYZ"[axis]} (m)');ax.grid(alpha=.2)
    axes[0].legend();axes[0].set_title('A3, seed 2026, zero angle; dotted line = known wrist frame')
    axes[-1].set_xlabel('Frame');fig.tight_layout();fig.savefig(output/'temporal_wrist.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for kind,label in [(False,'Teacher state'),(True,'Model rollout')]:
        selected=[r for r in logs if r['rollout']==kind]
        ax.plot([r['step'] for r in selected],[r['reconstruction'] for r in selected],'.',label=label,markersize=3)
    ax.set_xlabel('Training step');ax.set_ylabel('Unknown-channel x0 MSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'training_loss.png',dpi=170);plt.close(fig)
    print(json.dumps({m:r['primary'] for m,r in models.items()},indent=2));print(json.dumps(difference))


if __name__=='__main__':main()
