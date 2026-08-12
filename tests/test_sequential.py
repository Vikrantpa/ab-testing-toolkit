"""
The headline correctness test.

Under the null (no true effect), an analyst who peeks at a classical p-value
and stops the first time it dips below 0.05 has a false-positive rate FAR above
5%. The mSPRT always-valid p-value must keep that rate at or below 5% even
under continuous peeking. We verify both claims by simulation.
"""

import numpy as np
from scipy import stats

from abtest import MSPRT, always_valid_pvalue, confidence_sequence


def _naive_peeking_false_positive(rng, n_max, step, alpha):
    """Run one null experiment, peeking with a classical t-test."""
    c = rng.normal(0, 1, n_max)
    t = rng.normal(0, 1, n_max)  # no true effect
    for n in range(step, n_max + 1, step):
        _, p = stats.ttest_ind(t[:n], c[:n], equal_var=False)
        if p < alpha:
            return True
    return False


def _msprt_false_positive(rng, n_max, step, alpha):
    c = rng.normal(0, 1, n_max)
    t = rng.normal(0, 1, n_max)  # no true effect
    res = MSPRT(alpha=alpha, tau=0.2, burn_in=30).analyze(c, t, step=step)
    return res.rejected


def test_naive_peeking_inflates_type_I_error():
    rng = np.random.default_rng(0)
    sims, n_max, step, alpha = 400, 2000, 100, 0.05
    fp = np.mean([_naive_peeking_false_positive(rng, n_max, step, alpha)
                  for _ in range(sims)])
    # Peeking ~20 times should blow well past the nominal 5%.
    assert fp > 0.15


def test_msprt_controls_type_I_error_under_peeking():
    rng = np.random.default_rng(1)
    sims, n_max, step, alpha = 400, 2000, 100, 0.05
    fp = np.mean([_msprt_false_positive(rng, n_max, step, alpha)
                  for _ in range(sims)])
    # Allow simulation noise + small-sample slack, but it must stay near alpha.
    assert fp <= 0.08


def test_msprt_detects_a_real_effect():
    rng = np.random.default_rng(2)
    c = rng.normal(0, 1, 5000)
    t = rng.normal(0.15, 1, 5000)  # real, modest effect
    res = MSPRT(alpha=0.05, tau=0.15).analyze(c, t, step=100)
    assert res.rejected


def test_confidence_sequence_wider_than_fixed_ci():
    # Anytime-valid CIs pay for peeking with extra width vs a one-shot CI.
    d, V, alpha = 0.1, 1.0 / 500, 0.05
    lo, hi = confidence_sequence(d, V, tau=0.2, alpha=alpha)
    fixed_half = stats.norm.ppf(1 - alpha / 2) * np.sqrt(V)
    seq_half = (hi - lo) / 2
    assert seq_half > fixed_half


def test_always_valid_p_in_unit_interval():
    assert 0.0 <= always_valid_pvalue(0.0, 1.0 / 100, tau=0.2) <= 1.0
