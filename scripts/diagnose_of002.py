"""Post-hoc diagnostics on saved seed-2026 samples, no additional GPU runs."""
import json
import os
from pathlib import Path
import argparse
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); p=ap.parse_args().directory
    def sample(i,mode='correct',shift=None): return np.load(p/'samples'/f'{i:02d}_A_{mode}_{shift}.npz')
    def xyz(a): return a[:,:,4:70].reshape(-1,64,22,3)
    refs=np.concatenate([xyz(sample(i)['reference266']) for i in range(8)])
    rows=[]
    for i in range(8):
        a=sample(i,'wrong'); generated=xyz(a['generated266']); known=a['known'][0,:,4:70].reshape(64,22,3).all(-1)
        distances=np.linalg.norm(generated-refs,axis=-1)[:,~known].mean(axis=1)
        rows.append(dict(clip=i,wrong_caption_index=(i+1)%8,nearest_reference=int(distances.argmin()),
                        to_wrong_caption_motion_m=float(distances[(i+1)%8]),to_original_motion_m=float(distances[i])))
    (p/'caption_lookup_diagnostic.json').write_text(json.dumps(rows,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4)); report={}
    for shift in [-0.1,0.1]:
        base=np.concatenate([xyz(sample(i)['generated266']) for i in range(8)])
        changed=np.concatenate([xyz(sample(i,shift=shift)['generated266']) for i in range(8)])
        delta=(changed-base)*1000
        report[str(shift)]=dict(wrist_frame48_x_change_mm=float(delta[:,48,21,0].mean()),
                              elbow_frame48_x_change_mm=float(delta[:,48,19,0].mean()),
                              wrist_adjacent_frames_xyz_change_mm=float(np.linalg.norm(delta[:,[47,49],21],axis=-1).mean()))
        for ax,joint in zip(axes,[21,19]):
            y=delta[:,:,joint,0]; ax.plot(range(40,57),y.mean(axis=0)[40:57],'.-',label=f'{shift:+.1f} m target')
            ax.fill_between(range(40,57),y.min(axis=0)[40:57],y.max(axis=0)[40:57],alpha=.15)
            ax.axvline(48,c='gray',ls='--'); ax.set_xlabel('Frame'); ax.set_ylabel('X change vs baseline (mm)')
    axes[0].set_title('Right wrist: mean and range across eight clips')
    axes[1].set_title('Right elbow: mean and range across eight clips')
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(p/'local_response.png',dpi=150)
    (p/'local_response.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
