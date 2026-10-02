"""Render a source-audited architecture diagram; no model execution."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'outputs/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/architecture'; OUT.mkdir(exist_ok=True)
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig,ax=plt.subplots(figsize=(16,12)); ax.set(xlim=(0,16),ylim=(0,12)); ax.axis('off')
fig.patch.set_facecolor('#f7f9fc')

def label(x,y,text,size=11,color='#24354b',weight='normal',ha='center'):
    ax.text(x,y,text,fontproperties=FONT,fontsize=size,color=color,weight=weight,ha=ha,va='center',linespacing=1.6)

def box(x,y,w,h,text,fill='#ffffff',edge='#ced7e4',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.045,rounding_size=0.10',fc=fill,ec=edge,lw=1.2))
    label(x+w/2,y+h/2,text,size)

def arrow(x,y,x2,y2,color='#7a8da7'):
    ax.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=color,lw=1.4))

label(8,11.63,'网络结构核对：FloodDiffusion 与当前 OF003',20,weight='bold')
label(8,11.22,'左：官方主模型配置 / 右：已实际训练的代码。不是同一个去噪主干。',11)
label(4,10.71,'FloodDiffusion：潜空间 Wan-style DiT',14,color='#2563a6',weight='bold')
label(12,10.71,'OF003：动作空间轴向 Transformer',14,color='#8b4c16',weight='bold')

box(.7,9.60,6.6,.77,'HumanML3D 动作：T × 263\nroot / RIC / rotation / velocity / contact',fill='#eaf2fc')
box(8.7,9.60,6.6,.77,'当前采样状态：T × 266（本轮 T=64）\n保留冗余表示；观察点 XYZ 已固定为干净目标',fill='#fff2e5')
arrow(4,9.58,4,9.34); arrow(12,9.58,12,9.34)
box(.7,8.55,6.6,.76,'因果运动 VAE 编码器 → 4D latent / 时间压缩约 4 倍\n这是训练数据路径；生成时从潜空间噪声开始',size=10.5)
box(8.7,8.55,6.6,.76,'按关节无损打包：T × 22 × 13\n13 是补零后的槽宽；没有运动编码器或潜空间',size=10.5)
arrow(4,8.53,4,8.29); arrow(12,8.53,12,8.29)
box(.7,7.48,6.6,.78,'潜空间噪声状态 → 1×1×1 patch projection\n一个 latent 时刻对应一个 token；宽度 1024',fill='#eaf2fc')
box(8.7,7.48,6.6,.78,'Linear(13→256) + mask投影 + 通道σ投影\n+ 平均σ的Fourier/MLP + 关节embedding + 帧编码',fill='#fff2e5',size=10.4)
arrow(4,7.46,4,7.18); arrow(12,7.46,12,7.18)

box(.7,3.60,6.6,3.55,'',fill='#f0f5fd',edge='#7fa6d9')
box(8.7,3.60,6.6,3.55,'',fill='#fff5eb',edge='#d8aa7b')
label(4,6.83,'WanAttentionBlock × 8  |  8 heads',12,weight='bold')
label(12,6.83,'TextAxialBlock × 6  |  8 heads',12,weight='bold')
box(1,5.86,6,.60,'双向时间 Self-Attention：RoPE + Q/K normalization',size=10.4)
arrow(4,5.83,4,5.60)
box(1,5.00,6,.57,'文本 Cross-Attention：每帧关联对应的文本 token',size=10.5)
arrow(4,4.98,4,4.75)
box(1,4.14,6,.58,'FFN：1024 → 2048 → 1024',size=11)
label(4,3.83,'σ → 共享时间 MLP → 每层的 shift / scale / 残差 gate',10,color='#2563a6')
box(9,5.86,6,.60,'同一帧：22 关节 Self-Attention + FFN',size=10.7)
arrow(12,5.83,12,5.60)
box(9,5.00,6,.57,'同一关节：T 帧双向 Self-Attention + FFN',size=10.7)
arrow(12,4.98,12,4.75)
box(9,4.14,6,.58,'文本 Cross-Attention（无文本组删除这一层）',size=10.7)
label(12,3.83,'普通 Pre-LN；噪声/控制标志仅在输入端显式相加',10,color='#8b4c16')
arrow(4,3.58,4,3.31); arrow(12,3.58,12,3.31)
box(.7,2.52,6.6,.76,'时间调制输出头 → 4D latent 的 flow velocity\n三角时间调度更新活动窗口，窗口逐步滑动',fill='#eaf2fc',size=10.6)
box(8.7,2.52,6.6,.76,'LayerNorm + Linear(256→13) → 解包为 266D x0\njoint×time 谷底时间场更新自由通道；每步投影锚点',fill='#fff2e5',size=10.6)
arrow(4,2.50,4,2.24); arrow(12,2.50,12,2.24)
box(.7,1.48,6.6,.73,'因果 VAE 解码器 → 263D 动作流\n去噪器是双向注意力，因果性不要与 VAE 混淆',size=10.5)
box(8.7,1.48,6.6,.73,'下一步直接使用模型生成的状态，共 32 步\n本轮生成完整 64 帧；未验证流式缓冲区或实时性能',size=10.5)
label(4,1.00,'文本：官方 UMT5-XXL（4096D token）→ 每个 DiT block',10,color='#2563a6')
label(12,1.00,'同文本组复用官方 UMT5；12.33M 参数 / 无文本组 9.63M',10,color='#8b4c16')
label(8,.43,'依据：官方 ldf.yaml、diffusion_forcing_wan.py、wan_model.py；本地 model263.py、model.py、flow_features.py。2026-09-25',9,color='#64748b')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'comparison.png',dpi=160,facecolor=fig.get_facecolor())
fig.savefig(OUT/'comparison.svg',facecolor=fig.get_facecolor())
plt.close(fig)
print(OUT/'comparison.png')
