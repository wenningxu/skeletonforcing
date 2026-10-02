"""Animate the actual equal-local-step noise lattice; schematic geometry only."""
import hashlib,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/mpl-cache'))
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.patches import Rectangle
from motion_valley.root_first import RootFirstSchedule
from motion_valley.schedule import PARENTS


def main():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':10,'figure.facecolor':'#f5f7fb'})
    cfg=json.loads((root/'configs/root_first_overfit_v1.json').read_text())
    s=RootFirstSchedule(**cfg['schedule'])
    total=s.total_steps(cfg['frames'])
    field=s.lattice(cfg['frames']).numpy()[:,:,4:70].reshape(total+1,cfg['frames'],22,3)
    assert np.array_equal(field[...,0],field[...,1]) and np.array_equal(field[...,1],field[...,2])
    sigma=field[...,0];assert np.all(sigma[0]==1) and np.all(sigma[-1]==0)
    assert np.all(np.diff(sigma,axis=0)<=0)
    counts=(sigma[1:]<sigma[:-1]).sum(0);assert (counts==s.local_steps).all()
    dest=root/'outputs/OF009-equal-steps';dest.mkdir(exist_ok=True)
    cmap=colors.LinearSegmentedColormap.from_list('noise_wave',['#2462ab','#35b9bb','#e9dd79','#f39a38'])
    norm=colors.Normalize(0,1)
    # Clear schematic geometry with every one of the actual 22 graph vertices.
    xy=np.array([[0,0],[.23,-.13],[-.23,-.13],[0,.23],[.25,-.67],[-.25,-.67],
                 [0,.47],[.28,-1.15],[-.28,-1.15],[0,.74],[.43,-1.27],[-.43,-1.27],
                 [0,1.08],[.22,.87],[-.22,.87],[0,1.40],[.43,.81],[-.43,.81],
                 [.69,.43],[-.69,.43],[.89,.05],[-.89,.05]])
    fig=plt.figure(figsize=(14.2,8.4),dpi=110)
    fig.text(.04,.952,'等步数山谷去噪：错峰开始，错峰完成',fontsize=23,weight='bold',color='#132b46')
    fig.text(.04,.909,f'每个关节帧均去噪 {s.local_steps} 步；波前从第 0 帧 root 出发，完成区域保持生成值。',fontsize=11,color='#4b6079')
    status=fig.text(.96,.952,'',ha='right',fontsize=17,weight='bold',color='#2462ab')
    fig.text(.04,.859,'同一去噪步骤下的 4 个动作帧',fontsize=13,weight='bold',color='#132b46')
    fig.text(.495,.859,'完整时间场：22 个独立关节 × 64 帧',fontsize=13,weight='bold',color='#132b46')
    skeletons=[]
    frames=[0,16,32,63]
    for i,f in enumerate(frames):
        col=i%2;row=i//2
        ax=fig.add_axes([.04+col*.215,.48-row*.345,.20,.315],facecolor='#f5f7fb')
        for j,p in enumerate(PARENTS):
            if p>=0:ax.plot(xy[[p,j],0],xy[[p,j],1],color='#abb9c8',linewidth=3,zorder=1)
        scatter=ax.scatter(xy[:,0],xy[:,1],s=100,c=np.ones(22),cmap=cmap,norm=norm,
                           edgecolors='#f5f7fb',linewidths=1,zorder=3)
        if f==0:
            ax.scatter([0],[0],s=230,facecolors='none',edgecolors='#152d48',linewidths=1.4,zorder=4)
            ax.annotate('入口 root 0',xy=(0,0),xytext=(-1.08,.25),fontsize=8,color='#152d48',
                        arrowprops=dict(arrowstyle='-',color='#152d48',linewidth=.9),ha='left')
            for j,(x,y) in enumerate(xy):
                if j==0:continue
                dx=.11 if x>=0 else -.11
                ax.text(x+dx,y,str(j),fontsize=7.5,ha='left' if dx>0 else 'right',va='center',color='#344b66')
        ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.42,1.57);ax.set_aspect('equal');ax.axis('off')
        ax.set_title(f'动作第 {f} 帧',fontsize=11,pad=8,color='#152d48')
        skeletons.append(scatter)
    ax=fig.add_axes([.515,.205,.405,.60])
    heat=ax.imshow(sigma[0].T,origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,
                   extent=[-.5,63.5,21.5,-.5])
    ax.set_yticks(range(22),['0 root']+[str(i) for i in range(1,22)],fontsize=8)
    ax.set_xticks([0,16,32,48,63]);ax.set_xlabel('动作帧编号（不是去噪步数）',labelpad=8)
    ax.set_ylabel('关节编号',labelpad=6)
    for f in frames:ax.axvline(f,color='white',linewidth=.6,alpha=.42)
    ax.add_patch(Rectangle((-.5,-.5),1,1,fill=False,edgecolor='#152d48',linewidth=1.5))
    for spine in ax.spines.values():spine.set_edgecolor('#b1bfcd')
    cbax=fig.add_axes([.936,.205,.013,.60])
    bar=fig.colorbar(heat,cax=cbax,ticks=[0,.25,.5,.75,1])
    bar.ax.tick_params(labelsize=9)
    bar.ax.set_title('σ',fontsize=12,pad=10)
    fig.text(.50,.119,'蓝色 σ=0：完成去噪     橙色 σ=1：纯噪声',fontsize=11,color='#344b66')
    values=fig.text(.50,.076,'',fontsize=10,color='#344b66')
    fig.text(.04,.077,'固定示意骨架，仅展示噪声场；不是模型生成的动作。',fontsize=9,color='#5c6f83')
    completed=fig.text(.04,.044,'',fontsize=9,color='#5c6f83')
    progress=fig.add_axes([.04,.019,.91,.008]);progress.set_xlim(0,total);progress.set_ylim(0,1);progress.axis('off')
    progress.add_patch(Rectangle((0,0),total,1,color='#dce4ee'))
    fill=Rectangle((0,0),0,1,color='#2462ab');progress.add_patch(fill)
    rendered=[]
    for k in range(total+1):
        status.set_text(f'全局 {k:02d} / {total}')
        heat.set_data(sigma[k].T)
        for scatter,f in zip(skeletons,frames):scatter.set_array(sigma[k,f])
        fill.set_width(k)
        completed.set_text(f'已完成 {int((sigma[k]==0).sum())} / {sigma.shape[1]*sigma.shape[2]} 个关节帧 · 每点恰好 {s.local_steps} 次有效更新')
        values.set_text(f'第一帧 root：{sigma[k,0,0]:.2f}    第一帧手腕：{sigma[k,0,21]:.2f}    末帧手腕：{sigma[k,63,21]:.2f}')
        fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
        rendered.append(Image.fromarray(rgb))
        if k in [0,16,32,40,total]:rendered[-1].save(dest/f'valley_animation_step_{k:02d}.png')
    plt.close(fig)
    # One palette across the entire animation avoids frame-to-frame color drift.
    swatch=np.vstack([np.asarray(im.resize((280,166))) for im in rendered])
    palette=Image.fromarray(swatch).quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in rendered]
    gif=dest/'default_valley_denoising.gif'
    durations=[160]*len(indexed);durations[0]=900;durations[-1]=1700
    indexed[0].save(gif,save_all=True,append_images=indexed[1:],duration=durations,loop=0,optimize=False,disposal=2)
    with Image.open(gif) as check:
        assert check.n_frames==total+1 and check.size==rendered[0].size
        actual=[]
        for i in range(check.n_frames):check.seek(i);actual.append(check.info['duration'])
    np.save(dest/'animated_joint_sigma.npy',sigma)
    record=dict(gif=str(gif),bytes=gif.stat().st_size,frames=len(indexed),resolution=list(rendered[0].size),
        duration_ms=sum(actual),motion_frames=64,joints=22,global_denoising_steps=total,local_denoising_steps=s.local_steps,
        active_count_min=int(counts.min()),active_count_max=int(counts.max()),
        schedule=cfg['schedule'],scheduling_origin=[0,0],geometry='Fixed schematic; colors are actual schedule sigma',
        config_sha256=hashlib.sha256((root/'configs/root_first_overfit_v1.json').read_bytes()).hexdigest())
    (dest/'animation_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
