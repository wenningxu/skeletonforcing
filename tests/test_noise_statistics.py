import numpy as np
from motion_valley.noise_statistics import sample_seed, response_region, distribution_response


def test_noise_coupling_and_disjoint_conditions():
    independent = [sample_seed('independent', l, a, i) for l in range(2) for a in range(5) for i in range(32)]
    assert len(set(independent)) == 320
    for l in range(2):
        for i in range(32):
            assert len({sample_seed('matched', l, a, i) for a in range(5)}) == 1
    assert not set(independent) & {sample_seed('matched', l, a, i) for l in range(2) for a in range(5) for i in range(32)}


def test_anchor_exclusion_and_correct_response():
    known = np.zeros((64, 266), bool); known[48, 4+21*3:4+22*3] = True
    region = response_region(known)
    assert region.sum() == 74 and not region[48, 21]
    truth = np.zeros((64, 22, 3)); truth[region] = .02
    base_truth = np.zeros_like(truth)
    rng = np.random.default_rng(4)
    base = rng.normal(0, .001, (32, 64, 22, 3)); pred = base+truth
    pred[:, 48, 21] = 10000  # Fixed coordinates cannot improve or ruin free response.
    r = distribution_response(pred, base, truth, base_truth, region, paired=True, bootstrap=100)
    assert abs(r['mean_gain']-1) < 1e-10 and r['mean_relative_error'] < 1e-10
    assert r['distribution_response_passed'] and r['paired_pass_count'] == 32


def test_ignoring_control_fails_and_bootstrap_respects_coupling():
    region = response_region(np.zeros((64, 266), bool))
    truth = np.zeros((64, 22, 3)); truth[region] = .02
    base = np.random.default_rng(5).normal(0, .01, (32, 64, 22, 3))
    paired = distribution_response(base, base, truth, np.zeros_like(truth), region, paired=True, bootstrap=200)
    independent = distribution_response(base, base, truth, np.zeros_like(truth), region, paired=False, bootstrap=200)
    assert paired['mean_gain'] == 0 and paired['mean_relative_error'] == 1
    assert not paired['distribution_response_passed'] and paired['gain_ci95'] == [0, 0]
    assert independent['gain_ci95'][0] < 0 < independent['gain_ci95'][1]
