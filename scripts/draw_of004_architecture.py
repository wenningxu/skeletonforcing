"""Actual no-text OF004 graph; keep the historical standardization figure intact."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig, ax = plt.subplots(figsize=(14, 10)); ax.set(xlim=(0, 14), ylim=(0, 10)); ax.axis('off')
fig.patch.set_facecolor('#f7f9fc')

def label(x, y, value, size=11):
    ax.text(x, y, value, ha='center', va='center', fontproperties=font, fontsize=size, color='#24354b', linespacing=1.55)

def box(x, y, w, h, value, color='#eaf2fc', size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.04,rounding_size=.1', fc=color, ec='#a9bed8', lw=1.2))
    label(x+w/2, y+h/2, value, size)

def arrow(x, y, u, v):
    ax.add_patch(FancyArrowPatch((x, y), (u, v), arrowstyle='-|>', mutation_scale=14, lw=1.5, color='#597a9e'))

label(7, 9.55, 'OF004 实际运行结构：无文本 WanMotion266', 19)
label(7, 9.08, 'D=1024 · 8层 · 8头 · FFN=2048 · 78,053,389参数 · 无运动VAE', 11)
box(.5, 7.7, 7.4, 1.0, '当前状态 x：B × 64 × 266\nroot 4 + world XYZ 66 + rotation 126 + velocity 66 + contact 4', size=10.3)
box(8.5, 7.7, 5, 1.0, '完整通道时间场 σ：B × 64 × 266\n仅观察位置 σ=0；其余通道按距离依次去噪', '#fff0df', 10)
arrow(4.2, 7.65, 4.2, 7.37); arrow(11, 7.65, 11, 7.37)
box(.5, 6.3, 7.4, 1.0, '无损打包：22关节 × 13槽位\n动作投影 + 已知通道mask投影 + 关节embedding\n→ 1408个token，每个1024D', size=10.5)
box(8.5, 6.3, 5, 1.0, '13通道分别 sinusoidal 编码 → 拼接\n共享 Linear → SiLU → Linear → e\n→ SiLU → Linear → 每token的6组调制量', '#fff0df', 10)
arrow(4.2, 6.25, 4.2, 5.97)
box(.5, 3.12, 7.4, 2.8, '', '#f0f5fd')
label(4.2, 5.65, '标准 WanAttentionBlock × 8', 13)
box(.8, 4.4, 6.8, .93, 'AdaLN(e) → 完整关节×时间 Self-Attention → gate残差\nQ/K RMSNorm；RoPE=(帧, 关节序号, 0)', size=10.2)
arrow(4.2, 4.36, 4.2, 4.08)
box(.8, 3.35, 6.8, .7, 'AdaLN(e) → Linear/GELU/Linear FFN → gate残差\n每层独立可学习modulation偏置', size=10.3)
arrow(8.45, 5.5, 7.96, 5.5)
box(8.5, 4.35, 5, 1.57, 'shift / scale / gate 调制每一层\n时间与调制保持FP32\n无文本投影、无文本Cross-Attention\n不输入角度ID或动作ID', '#fff0df', 10.5)
arrow(4.2, 3.07, 4.2, 2.79)
box(.5, 1.77, 7.4, .95, '时间调制输出头 → Linear(1024→13) → 无损解包\n预测干净动作 x0：B × 64 × 266\n最后Linear零初始化；训练中学习', size=10.4)
arrow(4.2, 1.72, 4.2, 1.43)
box(.5, .32, 7.4, 1.04, '采样更新自由通道，再投影干净锚点\n32步推理始终接收模型上一步状态\n训练：准确同源时间场 + 50% rollout，截断梯度尾长≤4步', size=10.2)
box(8.5, .85, 5, 2.8, 'OF004-A：1条动作 × 3种训练状态\n4000步；4噪声种子 × 5状态 × 2布局\n\n仅A达到重建与响应门槛才启动B\nOF004-B：4条动作 × 3状态，2000步\n\n硬约束满足、自由关节响应、物理一致性分别评价', '#ffffff', 10.2)
fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
out = ROOT / 'outputs/OF004-analysis'; out.mkdir(exist_ok=True)
for ext in ['png', 'svg']:
    fig.savefig(out / f'architecture.{ext}', dpi=160, facecolor=fig.get_facecolor())
plt.close(fig)
