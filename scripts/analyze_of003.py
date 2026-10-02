"""CPU-only analysis of frozen OF003 evidence; never loads a model checkpoint."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = ['trained_layout_trained_angle', 'trained_layout_unseen_angle',
          'unseen_layout_trained_angle', 'unseen_layout_unseen_angle']
KEYS = ['free_xyz_mpjpe_m', 'feature_mse', 'anchor_max_m', 'rot6d_normalized_mse',
        'velocity_normalized_mse', 'velocity_error_vs_static_ratio',
        'fk_position_disagreement_m', 'bone_length_mae_m',
        'response_gain', 'response_relative_error', 'response_target_rms_m', 'response_actual_rms_m']
CHAINS = [[0,2,5,8,11], [0,1,4,7,10], [0,3,6,9,12,15], [9,14,17,19,21], [9,13,16,18,20]]


def group_name(row):
    return f"{'unseen' if row['heldout_layout'] else 'trained'}_layout_{'unseen' if row['heldout_angle'] else 'trained'}_angle"


def statistics(rows):
    result = {'count': len(rows)}
    for key in KEYS:
        values = np.array([r[key] for r in rows if key in r])
        if len(values):
            result[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                               p05=float(np.quantile(values,.05)), p95=float(np.quantile(values,.95)),
                               min=float(values.min()), max=float(values.max()))
    return result


def failure_reasons(row, gates):
    mapping = {'feature_mse':'feature_mse_max', 'free_xyz_mpjpe_m':'free_xyz_mpjpe_m_max',
               'rot6d_normalized_mse':'rotation_normalized_mse_max',
               'velocity_normalized_mse':'velocity_normalized_mse_max',
               'velocity_error_vs_static_ratio':'motion_velocity_error_vs_static_max',
               'anchor_max_m':'anchor_error_m_max'}
    failures = [key for key, gate in mapping.items() if row[key] >= gates[gate]]
    if row['degenerate_rotation_count']: failures.append('degenerate_rotation_count')
    if row['fk_position_disagreement_m'] >= row['gt_fk_position_disagreement_m'] + gates['fk_disagreement_above_gt_m_max']:
        failures.append('fk_position_disagreement_m')
    return failures


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--outputs',type=Path,default=Path('outputs'))
    root=parser.parse_args().outputs; dest=root/'OF003-analysis'; dest.mkdir(exist_ok=True)
    all_rows={}; analysis={}; csv_rows=[]; logs={}
    for arm in ['no_text','same_text']:
        directory=root/f'OF003-{arm}'
        rows=[json.loads(s) for s in (directory/'evaluation.jsonl').read_text().splitlines()]
        manifest=json.loads((directory/'manifest.json').read_text()); cfg=manifest['config']
        summary=json.loads((directory/'summary.json').read_text())
        logs[arm]=[json.loads(s) for s in (directory/'train.jsonl').read_text().splitlines()]
        assert summary['actual_cases']==len(rows)==640
        assert len({(r['clip_index'],r['seed'],r['layout'],r['angle']) for r in rows})==640
        assert all(r['all_steps_known_exact'] for r in rows)
        failures=[]
        for r in rows:
            failed=failure_reasons(r,cfg['overfit_gate'])
            response_pass=None if r['angle']==0 else (.5 <= r['response_gain'] <= 1.5 and r['response_relative_error'] < .5)
            failures.append(dict(clip_index=r['clip_index'],seed=r['seed'],layout=r['layout'],angle=r['angle'],
                                 reconstruction_failed=failed,response_passed=response_pass))
            csv_rows.append(dict(arm=arm,clip_index=r['clip_index'],seed=r['seed'],layout=r['layout'],angle=r['angle'],
                                 group=group_name(r),**{k:r.get(k,'') for k in KEYS},reconstruction_passed=not failed,response_passed=response_pass))
            if r['angle']!=0:
                assert abs(r['overwrite_baseline']['response_gain'])<1e-8
                assert abs(r['overwrite_baseline']['response_relative_error']-1)<1e-6
        per_group={g:statistics([r for r in rows if group_name(r)==g]) for g in GROUPS}
        for g in GROUPS:
            selected=[r for r in rows if group_name(r)==g]
            assert summary[g]['reconstruction_pass_count']==sum(not failure_reasons(r,cfg['overfit_gate']) for r in selected)
            assert summary[g]['response_pass_count']==sum(.5<=r['response_gain']<=1.5 and r['response_relative_error']<.5 for r in selected if r['angle']!=0)
        analysis[arm]=dict(summary=summary,overall=statistics(rows),groups=per_group,
            by_layout={l:statistics([r for r in rows if r['layout']==l]) for l in cfg['eval_layouts']},
            by_angle={str(a):statistics([r for r in rows if r['angle']==a]) for a in cfg['eval_angles']},
            failure_counts={k:sum(k in f['reconstruction_failed'] for f in failures) for k in
                ['feature_mse','free_xyz_mpjpe_m','rot6d_normalized_mse','velocity_normalized_mse',
                 'velocity_error_vs_static_ratio','anchor_max_m','degenerate_rotation_count','fk_position_disagreement_m']},
            parameters=manifest['parameters'],train_seconds=logs[arm][-1]['elapsed_s'])
        # Independently audit saved physical samples with NumPy, including exclusion of clamped points.
        audited=0; max_metric_delta=0.
        for row in rows:
            if row['seed']!=cfg['heldout_noise_seeds'][0]: continue
            prefix=f"{row['clip_index']:02d}_{row['layout']}"
            data=np.load(directory/'samples'/f"{prefix}_{row['angle']}.npz")
            pred=data['generated266'][0,:,4:70].reshape(64,22,3)
            truth=data['reference266'][0,:,4:70].reshape(64,22,3)
            known=data['known'][0,:,4:70].reshape(64,22,3).all(-1)
            error=np.linalg.norm(pred-truth,axis=-1)
            np.testing.assert_allclose(error[~known].mean(),row['free_xyz_mpjpe_m'],rtol=1e-5,atol=1e-7)
            np.testing.assert_allclose(error[known].max(),row['anchor_max_m'],rtol=1e-5,atol=1e-7)
            if row['angle']:
                base=np.load(directory/'samples'/f'{prefix}_0.npz')
                actual=pred-base['generated266'][0,:,4:70].reshape(64,22,3)
                target=truth-base['reference266'][0,:,4:70].reshape(64,22,3)
                region=np.zeros((64,22),bool); region[36:61,[17,19,21]]=True; region &= ~known
                a,b=actual[region],target[region]; energy=(b*b).sum()
                gain=float((a*b).sum()/energy); relative=float(np.sqrt(((a-b)**2).sum()/energy))
                np.testing.assert_allclose([gain,relative],[row['response_gain'],row['response_relative_error']],rtol=1e-5,atol=1e-6)
                max_metric_delta=max(max_metric_delta,abs(gain-row['response_gain']),abs(relative-row['response_relative_error']))
            audited+=1
        analysis[arm]['numpy_sample_audit']=dict(count=audited,max_response_metric_difference=max_metric_delta)
        (directory/'gate_failures.json').write_text(json.dumps(failures,indent=2))
        all_rows[arm]=rows
    # Verify identical sampled training conditions and intervention protocol in the two arms.
    matched=all(all(a[k]==b[k] for k in ['step','batch','layout','k','rollout']) for a,b in zip(logs['no_text'],logs['same_text']))
    assert len(logs['no_text'])==len(logs['same_text']) and matched
    analysis['paired_training_logged_states_match']=matched
    (dest/'analysis.json').write_text(json.dumps(analysis,indent=2))
    with (dest/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    colors={'no_text':'tab:blue','same_text':'tab:orange'}
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for arm,rows in all_rows.items():
        for ax,key,scale in zip(axes,['free_xyz_mpjpe_m','response_gain','response_relative_error'],[1000,1,1]):
            y=[analysis[arm]['groups'][g][key]['mean']*scale for g in GROUPS]
            ax.plot(range(4),y,'o-',color=colors[arm],label=arm)
            ax.set_xticks(range(4),['train/train','train/new','new/train','new/new'],rotation=15)
            ax.set_xlabel('layout / state angle'); ax.grid(alpha=.2)
    axes[0].axhline(20,c='gray',ls='--'); axes[0].set_title('Free MPJPE (mm)')
    axes[1].axhline(1,c='gray',ls='--'); axes[1].axhspan(.5,1.5,color='green',alpha=.08); axes[1].set_title('Paired free-joint response gain')
    axes[2].axhline(.5,c='gray',ls='--'); axes[2].set_title('Paired response relative error')
    axes[0].legend(); fig.tight_layout(); fig.savefig(dest/'comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(arm,rows) in zip(axes,all_rows.items()):
        for held,color in [(False,'tab:blue'),(True,'tab:orange')]:
            selected=[r for r in rows if r['angle'] and r['heldout_angle']==held]
            ax.scatter([r['response_gain'] for r in selected],[r['response_relative_error'] for r in selected],
                       s=10,alpha=.3,color=color,label='new angle' if held else 'trained angle')
        ax.axvspan(.5,1.5,color='green',alpha=.08); ax.axhline(.5,c='gray',ls='--')
        ax.set_xlabel('Gain (ideal 1)'); ax.set_ylabel('Relative error (ideal 0)'); ax.set_title(arm); ax.legend()
    fig.tight_layout(); fig.savefig(dest/'response_distribution.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,(arm,records) in zip(axes,logs.items()):
        for kind in [False,True]:
            values=[r for r in records if r['rollout']==kind]
            ax.semilogy([r['step'] for r in values],[r['reconstruction'] for r in values],'.',alpha=.55,label='model rollout' if kind else 'teacher state')
        ax.set_title(arm); ax.set_xlabel('Training step'); ax.legend()
    axes[0].set_ylabel('Normalized reconstruction MSE'); fig.tight_layout(); fig.savefig(dest/'loss.png',dpi=160); plt.close(fig)
    # Fixed first seed, every clip: no selection of visually favorable cases.
    for layout in ['A','D']:
        fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
        for i,ax in enumerate(axes.flat):
            for arm in all_rows:
                a=np.load(root/f'OF003-{arm}'/'samples'/f'{i:02d}_{layout}_12.npz')
                b=np.load(root/f'OF003-{arm}'/'samples'/f'{i:02d}_{layout}_0.npz')
                predicted=(a['generated266']-b['generated266'])[0,:,4:70].reshape(64,22,3)
                target=(a['reference266']-b['reference266'])[0,:,4:70].reshape(64,22,3)
                known=a['known'][0,:,4:70].reshape(64,22,3).all(-1)
                # Project free right-arm displacement onto the true displacement direction at each frame.
                mask=np.zeros((64,22),bool); mask[:,[17,19,21]]=True; mask &= ~known
                energy=(target**2*mask[...,None]).sum((1,2)); norm=np.sqrt(energy)
                signed=(predicted*target*mask[...,None]).sum((1,2))/np.maximum(norm,1e-12)
                ax.plot(range(64),signed*1000,color=colors[arm],label=arm)
            ax.plot(range(64),norm*1000,'k--',label='GT change')
            ax.set_title(f'clip {i} | +12 deg'); ax.set_xlim(32,63); ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8); fig.suptitle(f'Layout {layout}, fixed first seed: free-arm signed response (mm)\nProjection onto framewise GT change; zero curve means no response')
        fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(dest/f'paired_response_{layout}.png',dpi=140); plt.close(fig)
        fig=plt.figure(figsize=(15,8))
        for i in range(8):
            ax=fig.add_subplot(2,4,i+1,projection='3d'); motions=[]
            for arm in all_rows:
                data=np.load(root/f'OF003-{arm}'/'samples'/f'{i:02d}_{layout}_12.npz')
                motions.append((arm,data['generated266'][0,48,4:70].reshape(22,3)))
            motions.insert(0,('GT',data['reference266'][0,48,4:70].reshape(22,3)))
            points=np.concatenate([p for _,p in motions]); center=(points.max(0)+points.min(0))/2
            radius=max(np.ptp(points,axis=0).max()*.55,.7)
            for name,points in motions:
                for c,chain in enumerate(CHAINS):
                    p=points[chain]
                    ax.plot(p[:,0],p[:,2],p[:,1],color='gray' if name=='GT' else colors[name],
                            alpha=.5 if name=='GT' else .8,lw=3 if name=='GT' else 1.5,label=name if c==0 else None)
            ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[2]-radius,center[2]+radius)
            ax.set_zlim(center[1]-radius,center[1]+radius); ax.set_box_aspect((1,1,1)); ax.view_init(15,-65)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(f'clip {i}')
            if i==0: ax.legend(fontsize=8)
        fig.suptitle(f'Layout {layout}, +12 deg target, frame 48, first fixed seed\nXYZ snapshots; rotation consistency is evaluated separately')
        fig.subplots_adjust(left=0,right=1,bottom=0,top=.90,wspace=0,hspace=0)
        fig.savefig(dest/f'poses_{layout}.png',dpi=140); plt.close(fig)
    print(json.dumps({arm:dict(overall=analysis[arm]['overall'],summary=analysis[arm]['summary']) for arm in all_rows},indent=2))


if __name__=='__main__': main()
