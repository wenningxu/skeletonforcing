"""Render the implemented WanMotion266 component graph."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig,ax=plt.subplots(figsize=(14,11)); ax.set(xlim=(0,14),ylim=(0,11)); ax.axis('off')
fig.patch.set_facecolor('#f7f9fc')
def text(x,y,s,size=11,color='#24354b'):
    ax.text(x,y,s,ha='center',va='center',fontproperties=font,fontsize=size,color=color,linespacing=1.6)
def box(x,y,w,h,s,color='#eaf2fc',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.1',fc=color,ec='#a9bed8',lw=1.2))
    text(x+w/2,y+h/2,s,size)
def arrow(x,y,u,v):
    ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=14,lw=1.5,color='#597a9e'))
text(7,10.58,'已实现：WanMotion266 标准组件主干',19)
text(7,10.13,'无运动 VAE；关节×时间 token；共享时间 MLP 在每层执行自适应调制',11)
box(.6,8.83,7.4,.98,'当前动作状态 x：B × T × 266\n已知 XYZ 为干净值；保留 root / rotation / velocity / contact')
box(8.65,8.83,4.75,.98,'完整通道时间场 σ：B × T × 266\n训练与推理使用同一个时间场',color='#fff0df',size=10.5)
arrow(4.3,8.80,4.3,8.53); arrow(11,8.80,11,8.53)
box(.6,7.52,7.4,.98,'无损打包22个关节 → Linear(13→D)\n+ 已知通道mask投影 + 关节embedding → B × (T·22) × D')
box(8.65,7.52,4.75,.98,'每个通道分别做 sinusoidal embedding\n按槽位拼接 → 共享MLP → e：B × (T·22) × D',color='#fff0df',size=9.8)
arrow(4.3,7.49,4.3,7.14); arrow(11,7.49,11,7.14)
box(.6,2.72,7.4,4.38,'',color='#f0f5fd')
text(4.3,6.77,'标准 WanAttentionBlock × 8',13)
box(.9,5.36,6.8,.97,'AdaLN(e) → 双向 Self-Attention → gate(e)残差\nQ/K RMSNorm；Wan 3轴RoPE位置=(帧, 关节序号, 0)',size=10.3)
arrow(4.3,5.32,4.3,5.03)
box(.9,4.17,6.8,.82,'LayerNorm → 文本 Cross-Attention → 残差\n支持每帧文本允许mask；无文本组删除该分支',size=10.5)
arrow(4.3,4.13,4.3,3.85)
box(.9,2.98,6.8,.82,'AdaLN(e) → GELU FFN → gate(e)残差\n每个block有独立可学习的modulation偏置',size=10.5)
box(8.65,6.12,4.75,.97,'e → SiLU + Linear(D→6D)\n每个token的两组 shift / scale / gate',color='#fff0df',size=10)
arrow(8.60,6.60,8.02,6.60)
box(8.65,4.10,4.75,1.28,'官方 UMT5 token（4096D）\n→ Linear / GELU / Linear\n作为各层 cross-attention 的 K/V',color='#e8f4eb',size=10.5)
arrow(8.60,4.57,8.02,4.57)
text(11,5.63,'每层均接收调制，不只在输入相加',9.5,color='#946020')
arrow(4.3,2.68,4.3,2.36)
box(.6,1.40,7.4,.91,'时间调制输出头：LN(h) × (1+scale(e)) + shift(e)\n→ Linear(D→13) → 无损解包为266D干净动作预测 x0',size=10.5)
box(8.65,1.40,4.75,1.91,'主尺寸：D=1024 / FFN=2048\n诊断尺寸：D=256 / FFN=1024\n两者均8层、8头、相同组件\n尚未运行GPU实验',color='#ffffff',size=10.5)
arrow(4.3,1.36,4.3,1.08)
box(.6,.17,7.4,.87,'采样器：按当前/下一时间场更新自由通道 → 投影干净锚点\n生成状态回到下一步网络输入；时间条件随步骤重算',size=10.3)
text(11,.55,'图示对应 model_wan266.py\nFP32调制；SDPA注意力后端',9.5)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
out=ROOT/'outputs/architecture'; out.mkdir(exist_ok=True)
for ext in ['png','svg']: fig.savefig(out/f'wan_standard_v1.{ext}',dpi=150,facecolor=fig.get_facecolor())
plt.close(fig)
