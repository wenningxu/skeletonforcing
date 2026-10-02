import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data=np.load(ROOT/'prepared/OF003/states.npz')['states']
meta=json.loads((ROOT/'prepared/OF003/manifest.json').read_text())
chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
fig=plt.figure(figsize=(12,7))
for i in range(8):
    ax=fig.add_subplot(2,4,i+1,projection='3d')
    pos=data[i,:,48,4:70].reshape(5,22,3)
    center=(pos.max((0,1))+pos.min((0,1)))/2; radius=max(np.ptp(pos.reshape(-1,3),axis=0).max()*.55,.6)
    ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[2]-radius,center[2]+radius); ax.set_zlim(center[1]-radius,center[1]+radius)
    for variant,color in [(2,'gray'),(0,'tab:blue'),(4,'tab:orange')]:
        for chain in chains:
            xyz=pos[variant,chain]; ax.plot(xyz[:,0],xyz[:,2],xyz[:,1],color=color,lw=1.4,alpha=.75)
    ax.set_title(meta['selection']['selected'][i]['id'],fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1,1,1)); ax.view_init(15,-65)
fig.suptitle('Prepared supervision at frame 48 (not generated model outputs)\nGray: 0 deg; blue: -12 deg; orange: +12 deg')
fig.subplots_adjust(left=0,right=1,bottom=0,top=.86,wspace=0,hspace=.02)
fig.savefig(ROOT/'prepared/OF003/preview.png',dpi=140)
