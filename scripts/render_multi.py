"""Render all eight OF002 motions from saved samples, without inference."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); args=ap.parse_args(); root=args.directory
    manifest=json.loads((root/'manifest.json').read_text())
    for layout in ['A','D']:
        fig=plt.figure(figsize=(13,7)); artists=[]; arrays=[]
        for i,record in enumerate(manifest['selection']['selected']):
            data=np.load(root/'samples'/f'{i:02d}_{layout}_correct_None.npz')
            gt=data['reference266'][0,:,4:70].reshape(-1,22,3)
            pred=data['generated266'][0,:,4:70].reshape(-1,22,3)
            ax=fig.add_subplot(2,4,i+1,projection='3d'); ax.set_title(record['id'],fontsize=10)
            points=np.concatenate([gt,pred]).reshape(-1,3)
            center=(points.max(0)+points.min(0))/2; r=max(np.ptp(points,axis=0).max()*.53,.6)
            ax.set_xlim(center[0]-r,center[0]+r); ax.set_ylim(center[2]-r,center[2]+r); ax.set_zlim(center[1]-r,center[1]+r)
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=15,azim=-65)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            known=data['known'][0,:,4:70].reshape(-1,22,3).all(-1)
            target=data['target266'][0,:,4:70].reshape(-1,22,3)[known]
            ax.scatter(target[:,0],target[:,2],target[:,1],c='red',marker='x',s=15)
            for motion,is_gt in [(gt,True),(pred,False)]:
                arrays.append(motion)
                artists.append([ax.plot([],[],[],c='gray' if is_gt else 'tab:blue',lw=3 if is_gt else 1.5,alpha=.4 if is_gt else 1)[0] for _ in CHAINS])
        label=fig.suptitle('')
        def update(frame):
            for data,lines in zip(arrays,artists):
                for chain,line in zip(CHAINS,lines):
                    x=data[frame,chain]; line.set_data_3d(x[:,0],x[:,2],x[:,1])
            label.set_text(f'Layout {layout} | seed 2026 (+10000 per clip) | frame {frame}/63\nGray: reference; blue: generated; red: spatial targets')
            return sum(artists,[])
        fig.subplots_adjust(left=0,right=1,bottom=0,top=.86,wspace=0,hspace=.02)
        update(32); fig.savefig(root/f'motions_{layout}.png',dpi=120)
        FuncAnimation(fig,update,frames=64,interval=50).save(root/f'motions_{layout}.gif',writer=PillowWriter(fps=20))
        plt.close(fig)

if __name__=='__main__': main()
