"""Local postprocessing of saved OF002 evidence. No model execution."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); root=ap.parse_args().directory
    rows=[json.loads(s) for s in (root/'evaluation.jsonl').read_text().splitlines()]
    manifest=json.loads((root/'manifest.json').read_text()); cfg=manifest['config']
    groups={}
    keys=['feature_mse','free_xyz_mpjpe_m','rot6d_normalized_mse','velocity_normalized_mse',
          'velocity_error_vs_static_ratio','fk_position_disagreement_m','target_anchor_max_m',
          'neighborhood_displacement_m','neighborhood_signed_axis_gain','paired_xyz_change_m']
    for row in rows:
        label=f"{row['layout']}/{row['text_mode']}/{row['shift_m']}"
        groups.setdefault(label,[]).append(row)
    stats={}
    for label,group in groups.items():
        stats[label]={'count':len(group)}
        for key in keys:
            values=[r[key] for r in group if key in r]
            if values: stats[label][key]=dict(mean=float(np.mean(values)),min=min(values),max=max(values))
        if group[0]['shift_m'] is not None:
            for key in ['fk_position_disagreement_m','bone_length_mae_m','motion_velocity_error_m_frame']:
                values=[r['consistency'][key] for r in group]
                stats[label][key]=dict(mean=float(np.mean(values)),min=min(values),max=max(values))
    (root/'analysis.json').write_text(json.dumps(stats,indent=2))
    metrics=[('feature_mse','feature_mse_max'),('free_xyz_mpjpe_m','free_xyz_mpjpe_m_max'),
             ('rot6d_normalized_mse','rotation_normalized_mse_max'),('velocity_normalized_mse','velocity_normalized_mse_max'),
             ('velocity_error_vs_static_ratio','motion_velocity_error_vs_static_max'),('anchor_max_m','anchor_error_m_max')]
    failures=[]
    for r in rows:
        if r['shift_m'] is not None or r['text_mode']!='correct': continue
        failed=[key for key,gate in metrics if r[key]>=cfg['overfit_gate'][gate]]
        if r['degenerate_rotation_count']!=0: failed.append('degenerate_rotation_count')
        if r['fk_position_disagreement_m']>=r['gt_fk_position_disagreement_m']+cfg['overfit_gate']['fk_disagreement_above_gt_m_max']:
            failed.append('fk_position_disagreement_m')
        if failed: failures.append(dict(clip_index=r['clip_index'],seed=r['seed'],layout=r['layout'],failed=failed))
    (root/'gate_failures.json').write_text(json.dumps(failures,indent=2))
    matrix=np.zeros((cfg['clips'],4))
    for i in range(cfg['clips']):
        for j,layout in enumerate(cfg['eval_layouts']):
            matrix[i,j]=1000*np.mean([r['free_xyz_mpjpe_m'] for r in rows if r['clip_index']==i and r['layout']==layout and r['text_mode']=='correct' and r['shift_m'] is None])
    labels=[r['id'] for r in manifest['selection']['selected']]
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(matrix,cmap='YlOrRd',vmin=0)
    ax.set_xticks(range(4),['A trained','B trained','C trained','D held out']); ax.set_yticks(range(cfg['clips']),labels)
    for i in range(cfg['clips']):
        for j in range(4): ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='black')
    ax.set_title('Free-joint MPJPE (mm), mean of four noise seeds')
    fig.colorbar(im,ax=ax,label='mm'); fig.tight_layout(); fig.savefig(root/'reconstruction.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode in ['correct','wrong','empty']:
        y=[1000*np.mean([r['free_xyz_mpjpe_m'] for r in rows if r['clip_index']==i and r['layout']=='A' and r['text_mode']==mode and r['shift_m'] is None]) for i in range(cfg['clips'])]
        axes[0].plot(range(cfg['clips']),y,'o-',label=mode)
    axes[0].set_xticks(range(cfg['clips']),labels,rotation=40); axes[0].set_ylabel('Free MPJPE to original motion (mm)'); axes[0].legend(); axes[0].set_title('Text intervention: fixed noise and anchors')
    for shift in cfg['shift_meters']:
        y=[1000*np.mean([r['neighborhood_displacement_m'] for r in rows if r['clip_index']==i and r['shift_m']==shift]) for i in range(cfg['clips'])]
        axes[1].plot(range(cfg['clips']),y,'o-',label=f'{shift:+.1f} m target')
    axes[1].axhline(cfg['response_min_m']*1000,c='gray',ls='--',label='2 mm diagnostic threshold')
    axes[1].set_xticks(range(cfg['clips']),labels,rotation=40); axes[1].set_ylabel('Unclamped neighborhood change (mm)'); axes[1].set_title('Control intervention vs original output'); axes[1].legend()
    fig.tight_layout(); fig.savefig(root/'conditioning.png',dpi=150); plt.close(fig)
    logs=[json.loads(s) for s in (root/'train.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(8,4))
    for kind in [False,True]:
        group=[r for r in logs if r['rollout']==kind]
        ax.semilogy([r['step'] for r in group],[r['reconstruction'] for r in group],'.',label='rollout' if kind else 'analytic',alpha=.6)
    ax.set_xlabel('Step'); ax.set_ylabel('Normalized reconstruction MSE'); ax.legend(); fig.tight_layout(); fig.savefig(root/'loss.png',dpi=150); plt.close(fig)
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
