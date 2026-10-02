"""Render saved real-data outputs locally; does not run the trained model."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args(); root=args.directory
    data=np.load(root/'overfit_samples.npz')
    pred=data['generated266'][0,:,4:70].reshape(-1,22,3)
    truth=data['reference266'][0,:,4:70].reshape(-1,22,3)
    known=data['known'][0,:,4:70].reshape(-1,22,3).all(-1)
    caption=json.loads((root/'manifest.json').read_text())['captions'][0]
    points=np.concatenate([pred,truth]).reshape(-1,3)
    center=(points.max(0)+points.min(0))/2
    radius=max(float(np.ptp(points,axis=0).max())*.55,.5)
    fig=plt.figure(figsize=(10,5))
    axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    lines=[]
    for ax,title in zip(axes,['Reference','Generated: unseen noise seed 2026']):
        ax.set_title(title); ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[2]-radius,center[2]+radius); ax.set_zlim(center[1]-radius,center[1]+radius)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Y (m)')
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=15,azim=-65)
        lines.append([ax.plot([],[],[],lw=2)[0] for chain in CHAINS])
        locations=truth[known]
        ax.scatter(locations[:,0],locations[:,2],locations[:,1],c='red',s=25,marker='x')
    label=fig.suptitle(caption,fontsize=10)
    def update(frame):
        for arrays,artists in zip([truth,pred],lines):
            for chain,line in zip(CHAINS,artists):
                x=arrays[frame,chain]; line.set_data_3d(x[:,0],x[:,2],x[:,1])
        label.set_text(f'{caption}\nFrame {frame}/{len(truth)-1}; red crosses: spatial targets at their specified frames')
        return sum(lines,[])
    update(len(truth)//2); fig.savefig(root/'comparison.png',dpi=150)
    animation=FuncAnimation(fig,update,frames=len(truth),interval=50)
    animation.save(root/'comparison.gif',writer=PillowWriter(fps=20)); plt.close(fig)
    log=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(8,4))
    for mode,name in [(False,'Teacher-state'),(True,'Model-rollout state')]:
        rows=[row for row in log if row['rollout']==mode]
        ax.semilogy([r['step'] for r in rows],[r['reconstruction'] for r in rows],'.-',label=name,alpha=.75)
    ax.set_xlabel('Training step'); ax.set_ylabel('Normalized feature reconstruction MSE')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(root/'loss.png',dpi=150)

if __name__=='__main__': main()
