"""Scientific plots and side-by-side generated/reference skeleton animation."""
import json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from motion_valley.schedule import PARENTS

out=root/'outputs/OF009';dest=root/'outputs/OF009-analysis';dest.mkdir(exist_ok=True)
rows=json.loads((out/'evaluation.json').read_text());logs=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for flag,color,label in [(False,'tab:blue','Teacher'),(True,'tab:orange','Model rollout')]:
    subset=[r for r in logs if r['rollout']==flag]
    axes[0].plot([r['step'] for r in subset],[r['reconstruction'] for r in subset],'.-',ms=2,lw=.5,color=color,label=label)
axes[0].set_yscale('log');axes[0].set_xlabel('Training step');axes[0].set_ylabel('Active normalized x0 MSE');axes[0].legend()
axes[1].bar(range(8),[r['xyz_mpjpe_m']*1000 for r in rows]);axes[1].axhline(20,color='red',ls='--',label='20 mm gate')
axes[1].set_xticks(range(8),[r['seed'] for r in rows],rotation=45);axes[1].set_ylabel('Final world XYZ MPJPE (mm)');axes[1].set_xlabel('Held-out noise seed');axes[1].legend()
for r in rows:axes[2].plot(range(8),[r['by_depth'][str(d)]['xyz_mpjpe_m']*1000 for d in range(8)],alpha=.6)
axes[2].set_xlabel('Skeleton depth (root = 0)');axes[2].set_ylabel('Final world XYZ MPJPE (mm)')
fig.suptitle('OF009: one-clip unconditional overfit; equal 32 local denoising updates')
fig.savefig(dest/'diagnostics.png',dpi=160);plt.close(fig)
with np.load(out/'samples/2026.npz') as d:
    pred=d['generated266'][0,:,4:70].reshape(64,22,3);gt=d['reference266'][0,:,4:70].reshape(64,22,3)
all_points=np.concatenate([pred,gt]).reshape(-1,3);lo=all_points.min(0);hi=all_points.max(0);center=(lo+hi)/2;radius=max((hi-lo).max()/2,.5)*1.1
fig=plt.figure(figsize=(10,5));axes=[fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)]
lines=[]
for ax,title,color in zip(axes,['Reference','OF009 generated (seed 2026)'],['tab:blue','tab:orange']):
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-radius,center[2]+radius),zlim=(center[1]-radius,center[1]+radius),xlabel='World X (m)',ylabel='World Z (m)',zlabel='World Y (m)',title=title)
    ax.set_box_aspect([1,1,1]);ax.view_init(elev=15,azim=-60)
    lines.append([ax.plot([],[],[],color=color,lw=2)[0] for _ in range(21)])
writer=PillowWriter(fps=20)
with writer.saving(fig,str(dest/'reference_vs_generated.gif'),dpi=95):
    for f in range(64):
        for pts,group in zip([gt,pred],lines):
            for j,line in enumerate(group,start=1):
                pair=pts[f,[PARENTS[j],j]];line.set_data(pair[:,0],pair[:,2]);line.set_3d_properties(pair[:,1])
        fig.suptitle(f'OF009 final generated motion, frame {f}/63 (not a denoising animation)');writer.grab_frame()
plt.close(fig)
print(dest/'diagnostics.png');print(dest/'reference_vs_generated.gif')
