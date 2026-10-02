"""CPU audit of OF004: physical samples, frozen gates, paired control response."""
import csv
import json
from pathlib import Path

from analyze_of003 import GROUPS, KEYS, failure_reasons, group_name, statistics
import numpy as np
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1] / 'outputs'
    dest = root / 'OF004-analysis'
    dest.mkdir(exist_ok=True)
    suite = json.loads((root / 'OF004-suite.json').read_text())
    analysis = {'suite': suite, 'arms': {}}
    csv_rows = []
    for arm in ['A', 'B']:
        directory = root / f'OF004-{arm}'
        if not (directory / 'summary.json').exists():
            continue
        manifest = json.loads((directory / 'manifest.json').read_text())
        cfg = manifest['config']
        summary = json.loads((directory / 'summary.json').read_text())
        rows = [json.loads(s) for s in (directory / 'evaluation.jsonl').read_text().splitlines()]
        logs = [json.loads(s) for s in (directory / 'train.jsonl').read_text().splitlines()]
        assert summary['actual_cases'] == len(rows)
        assert len({(r['clip_index'], r['seed'], r['layout'], r['angle']) for r in rows}) == len(rows)
        if summary['completed_evaluation']:
            assert len(rows) == cfg['clips'] * len(cfg['eval_angles']) * len(cfg['eval_layouts']) * len(cfg['heldout_noise_seeds'])
        assert all(r['all_steps_known_exact'] for r in rows)
        failures = []
        for r in rows:
            failed = failure_reasons(r, cfg['overfit_gate'])
            response_pass = None if r['angle'] == 0 else (
                cfg['response_gain_min'] <= r['response_gain'] <= cfg['response_gain_max']
                and r['response_relative_error'] < cfg['response_error_max'])
            failures.append(dict(clip_index=r['clip_index'], seed=r['seed'], layout=r['layout'],
                                 angle=r['angle'], reconstruction_failed=failed, response_passed=response_pass))
            csv_rows.append(dict(arm=arm, clip_index=r['clip_index'], seed=r['seed'], layout=r['layout'],
                                 angle=r['angle'], group=group_name(r), **{k: r.get(k, '') for k in KEYS},
                                 reconstruction_passed=not failed, response_passed=response_pass))
            if r['angle']:
                assert abs(r['overwrite_baseline']['response_gain']) < 1e-8
                assert abs(r['overwrite_baseline']['response_relative_error'] - 1) < 1e-6
        groups = {g: statistics([r for r in rows if group_name(r) == g]) for g in GROUPS}
        for g in GROUPS:
            selected = [r for r in rows if group_name(r) == g]
            assert summary[g]['reconstruction_pass_count'] == sum(not failure_reasons(r, cfg['overfit_gate']) for r in selected)
            assert summary[g]['response_pass_count'] == sum(
                cfg['response_gain_min'] <= r['response_gain'] <= cfg['response_gain_max']
                and r['response_relative_error'] < cfg['response_error_max'] for r in selected if r['angle'])
        audited, max_delta = 0, 0.
        for row in rows:
            if row['seed'] != cfg['heldout_noise_seeds'][0]:
                continue
            prefix = f"{row['clip_index']:02d}_{row['layout']}"
            with np.load(directory / 'samples' / f"{prefix}_{row['angle']}.npz") as data:
                pred = data['generated266'][0, :, 4:70].reshape(64, 22, 3)
                truth = data['reference266'][0, :, 4:70].reshape(64, 22, 3)
                known = data['known'][0, :, 4:70].reshape(64, 22, 3).all(-1)
            error = np.linalg.norm(pred - truth, axis=-1)
            np.testing.assert_allclose(error[~known].mean(), row['free_xyz_mpjpe_m'], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(error[known].max(), row['anchor_max_m'], rtol=1e-5, atol=1e-7)
            if row['angle']:
                with np.load(directory / 'samples' / f'{prefix}_0.npz') as base:
                    actual = pred - base['generated266'][0, :, 4:70].reshape(64, 22, 3)
                    target = truth - base['reference266'][0, :, 4:70].reshape(64, 22, 3)
                region = np.zeros((64, 22), bool)
                region[36:61, [17, 19, 21]] = True
                region &= ~known
                a, b = actual[region], target[region]
                energy = (b * b).sum()
                gain = float((a * b).sum() / energy)
                relative = float(np.sqrt(((a - b) ** 2).sum() / energy))
                np.testing.assert_allclose([gain, relative], [row['response_gain'], row['response_relative_error']], rtol=1e-5, atol=1e-6)
                max_delta = max(max_delta, abs(gain - row['response_gain']), abs(relative - row['response_relative_error']))
            audited += 1
        failure_keys = sorted({k for f in failures for k in f['reconstruction_failed']})
        result = dict(summary=summary, overall=statistics(rows), groups=groups,
                      by_layout={l: statistics([r for r in rows if r['layout'] == l]) for l in cfg['eval_layouts']},
                      by_angle={str(a): statistics([r for r in rows if r['angle'] == a]) for a in cfg['eval_angles']},
                      failure_counts={k: sum(k in f['reconstruction_failed'] for f in failures) for k in failure_keys},
                      parameters=manifest['parameters'], selected_bank_indices=manifest['selected_bank_indices'],
                      train_seconds=logs[-1]['elapsed_s'], peak_memory_allocated_gb=max(r['peak_memory_allocated_gb'] for r in logs),
                      numpy_sample_audit=dict(count=audited, max_response_metric_difference=max_delta))
        # Post-hoc descriptive control-ignoring reference, never used to alter gates.
        # XYZ averaging need not satisfy FK/dynamics, so this is not a physical-quality baseline.
        bank_path = root.parent / cfg['bank_file']
        with np.load(bank_path) as bank:
            angles = bank['angles'].tolist()
            states = bank['states'][manifest['selected_bank_indices']]
        mean_state = states[:, [angles.index(a) for a in cfg['train_angles']]].mean(1)
        reference_errors = []
        for row in rows:
            truth = states[row['clip_index'], angles.index(row['angle']), :, 4:70].reshape(64, 22, 3)
            average = mean_state[row['clip_index'], :, 4:70].reshape(64, 22, 3)
            free = np.ones((64, 22), bool)
            for frame, joint in cfg['layouts'][row['layout']]:
                free[frame, joint] = False
            reference_errors.append(float(np.linalg.norm(average - truth, axis=-1)[free].mean()))
        result['posthoc_training_mean_reference'] = dict(
            description='Mean XYZ of three training states, with exact anchors overwritten; not a physically validated motion',
            free_xyz_mpjpe_m_mean=float(np.mean(reference_errors)),
            free_xyz_mpjpe_m_max=float(np.max(reference_errors)), response_gain=0., response_relative_error=1.)
        analysis['arms'][arm] = result
        (directory / 'gate_failures.json').write_text(json.dumps(failures, indent=2))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, scale, title in zip(axes, ['free_xyz_mpjpe_m', 'response_gain', 'response_relative_error'],
                                       [1000, 1, 1], ['Free MPJPE (mm)', 'Paired response gain', 'Paired response relative error']):
            ax.plot(range(4), [groups[g][key]['mean'] * scale for g in GROUPS], 'o-')
            ax.set_xticks(range(4), ['train/train', 'train/new', 'new/train', 'new/new'], rotation=15)
            ax.set_xlabel('Layout / state angle'); ax.set_title(title); ax.grid(alpha=.2)
        axes[0].axhline(20, c='gray', ls='--')
        axes[1].axhspan(.5, 1.5, color='green', alpha=.1); axes[1].axhline(1, c='gray', ls='--')
        axes[2].axhline(.5, c='gray', ls='--')
        fig.tight_layout(); fig.savefig(dest / f'{arm}-metrics.png', dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        for kind in [False, True]:
            points = [r for r in logs if r['rollout'] == kind]
            ax.semilogy([r['step'] for r in points], [r['reconstruction'] for r in points], '.', alpha=.6,
                        label='Model rollout' if kind else 'Teacher state')
        ax.set_xlabel('Training step'); ax.set_ylabel('Normalized reconstruction MSE'); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(dest / f'{arm}-loss.png', dpi=160); plt.close(fig)
        # Every chosen clip, first fixed noise seed, +12 minus 0 degree. No favorable sample selection.
        for layout in cfg['eval_layouts']:
            fig, axes = plt.subplots(1, cfg['clips'], figsize=(6 * cfg['clips'], 4), squeeze=False)
            for i, ax in enumerate(axes.flat):
                with np.load(directory / 'samples' / f'{i:02d}_{layout}_12.npz') as a, np.load(directory / 'samples' / f'{i:02d}_{layout}_0.npz') as b:
                    predicted = (a['generated266'] - b['generated266'])[0, :, 4:70].reshape(64, 22, 3)
                    target = (a['reference266'] - b['reference266'])[0, :, 4:70].reshape(64, 22, 3)
                    known = a['known'][0, :, 4:70].reshape(64, 22, 3).all(-1)
                mask = np.zeros((64, 22), bool); mask[:, [17, 19, 21]] = True; mask &= ~known
                energy = (target ** 2 * mask[..., None]).sum((1, 2)); norm = np.sqrt(energy)
                signed = (predicted * target * mask[..., None]).sum((1, 2)) / np.maximum(norm, 1e-12)
                ax.plot(norm * 1000, c='black', label='Ground truth')
                ax.plot(np.where(norm > 1e-8, signed * 1000, np.nan), label='Generated')
                ax.set_xlabel('Frame'); ax.set_ylabel('Free-arm projected displacement (mm)')
                ax.set_title(f"Bank {manifest['selected_bank_indices'][i]}, layout {layout}"); ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(dest / f'{arm}-response-{layout}.png', dpi=160); plt.close(fig)
    (dest / 'analysis.json').write_text(json.dumps(analysis, indent=2))
    if csv_rows:
        with (dest / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({a: dict(cases=r['overall']['count'], audit=r['numpy_sample_audit'],
                            free_mpjpe_mm=r['overall']['free_xyz_mpjpe_m']['mean'] * 1000,
                            response_gain=r['overall']['response_gain']['mean']) for a, r in analysis['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
