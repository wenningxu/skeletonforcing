"""CPU distribution response statistics. Hard anchors never enter the metric."""
import numpy as np


def sample_seed(protocol, layout_index, angle_index, sample_index):
    if protocol not in ['independent', 'matched'] or not 0 <= sample_index < 32:
        raise ValueError('Invalid protocol/sample index')
    if not 0 <= layout_index < 2 or not 0 <= angle_index < 5:
        raise ValueError('Invalid condition')
    return (610000 + layout_index * 10000 + angle_index * 1000 + sample_index
            if protocol == 'independent' else 510000 + layout_index * 1000 + sample_index)


def response_region(known):
    mask = np.zeros((64, 22), bool)
    mask[36:61, [17, 19, 21]] = True
    return mask & ~np.asarray(known).reshape(64, 266)[:, 4:70].reshape(64, 22, 3).all(-1)


def distribution_response(pred, base, truth, base_truth, region, paired=False,
                          bootstrap=4000, seed=923):
    """Inputs: N,T,J,3 samples, T,J,3 references. CI conditions on ONE model.

    Same-noise bootstrap resamples paired indices; independent bootstrap uses
    separate samples. Family interval uses Bonferroni for eight contrasts.
    """
    a = np.asarray(pred, dtype=np.float64)[:, region].reshape(len(pred), -1)
    b = np.asarray(base, dtype=np.float64)[:, region].reshape(len(base), -1)
    target = (np.asarray(truth, dtype=np.float64) - base_truth)[region].reshape(-1)
    energy = target @ target
    if a.shape != b.shape or len(a) < 2 or energy <= 1e-10:
        raise ValueError('Invalid ensembles or zero response target')
    delta = a.mean(0) - b.mean(0)
    gain = float(delta @ target / energy)
    relative = float(np.linalg.norm(delta - target) / np.sqrt(energy))
    rng = np.random.default_rng(seed); n = len(a)
    weights_a = rng.multinomial(n, np.full(n, 1/n), size=bootstrap) / n
    weights_b = weights_a if paired else rng.multinomial(n, np.full(n, 1/n), size=bootstrap) / n
    # Bootstrap means rather than materializing bootstrap*N*all-features tensors.
    differences = weights_a @ a - weights_b @ b
    boot_gain = differences @ target / energy
    boot_error = np.linalg.norm(differences - target, axis=-1) / np.sqrt(energy)
    ci95 = np.quantile(boot_gain, [.025, .975]).tolist()
    family_ci = np.quantile(boot_gain, [.05/(2*8), 1-.05/(2*8)]).tolist()
    within = np.sqrt(((a-a.mean(0))**2).sum(1).mean() / region.sum())
    actual = np.sqrt((delta**2).sum() / region.sum())
    result = dict(mean_gain=gain, mean_relative_error=relative,
                  gain_ci95=ci95, gain_family_ci=family_ci,
                  relative_error_ci95=np.quantile(boot_error, [.025, .975]).tolist(),
                  target_rms_m=float(np.sqrt(energy/region.sum())),
                  mean_response_rms_m=float(actual), within_condition_rms_m=float(within),
                  baseline_within_rms_m=float(np.sqrt(((b-b.mean(0))**2).sum(1).mean()/region.sum())),
                  distribution_response_passed=bool(.5 <= gain <= 1.5 and relative < .5 and family_ci[0] > 0),
                  n=n, bootstrap=bootstrap, paired=paired)
    if paired:
        gains = (a-b) @ target / energy
        errors = np.linalg.norm(a-b-target, axis=1) / np.sqrt(energy)
        result.update(paired_gain_mean=float(gains.mean()), paired_error_mean=float(errors.mean()),
                      paired_pass_count=int(((gains >= .5) & (gains <= 1.5) & (errors < .5)).sum()))
    return result
