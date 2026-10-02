"""Plot the exact frozen OF004 schedule, not a learned influence estimate."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from motion_valley.flow_features import FeatureControls, field_step
from motion_valley.representation266 import position_mask
from motion_valley.schedule import distance_field


def main():
    cfg = json.loads((ROOT / 'configs/overfit_wan_v1_a.json').read_text())
    joints = torch.zeros(1, 64, 22, dtype=torch.bool)
    for frame, joint in cfg['layouts']['A']:
        joints[0, frame, joint] = True
    known = position_mask(joints)
    controls = FeatureControls(torch.zeros(1, 64, 266), known, joints)
    distance = distance_field(joints)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True, constrained_layout=True)
    for column, step in enumerate([0, 8, 16, 24, 32]):
        before, after = field_step(min(step, 31), 32, controls, distance, cfg['mode'])
        sigma = after if step == 32 else before
        position = sigma[0, :, 4:70].reshape(64, 22, 3)[..., 0]
        velocity = sigma[0, :, 196:262].reshape(64, 22, 3)[..., 0]
        for row, field in enumerate([position, velocity]):
            ax = axes[row, column]
            picture = ax.imshow(field.T.numpy(), origin='lower', aspect='auto', vmin=0, vmax=1,
                                cmap='viridis', interpolation='nearest', extent=(-.5, 63.5, -.5, 21.5))
            ax.set_xticks([0, 16, 32, 48, 63]); ax.set_yticks([0, 7, 14, 21])
            if row == 0:
                ax.scatter([p[0] for p in cfg['layouts']['A']], [p[1] for p in cfg['layouts']['A']],
                           s=35, marker='x', c='red', linewidths=1.3)
                ax.set_title(f'Sampler state {step}/32')
            else:
                ax.set_xlabel('Motion frame')
        axes[0, 0].set_ylabel('World position: joint index')
        axes[1, 0].set_ylabel('Unknown local velocity: joint index')
    fig.colorbar(picture, ax=axes, shrink=.8, label='Noise level sigma (0 = clean, 1 = noise)')
    fig.suptitle('OF004-A exact time fields | red crosses: clean XYZ anchors | not learned attention', fontsize=13)
    output = ROOT / 'outputs/OF004-analysis'
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'time_fields.png', dpi=170)
    fig.savefig(output / 'time_fields.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
