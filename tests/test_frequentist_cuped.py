import numpy as np
from scipy import stats

from abtest import two_sample_test, two_proportion_test, run_cuped_test, cuped_adjust


def test_two_sample_matches_scipy():
    rng = np.random.default_rng(0)
    c = rng.normal(0, 1, 500)
    t = rng.normal(0.3, 1, 500)
    res = two_sample_test(c, t)
    t_ref, p_ref = stats.ttest_ind(t, c, equal_var=False)
    assert abs(res.p_value - p_ref) < 1e-9
    assert abs(res.statistic - t_ref) < 1e-9


def test_two_sample_ci_contains_true_effect():
    rng = np.random.default_rng(1)
    c = rng.normal(10, 2, 5000)
    t = rng.normal(11, 2, 5000)  # true diff = 1
    res = two_sample_test(c, t)
    assert res.ci_low < 1.0 < res.ci_high


def test_two_proportion_detects_difference():
    res = two_proportion_test(1000, 10000, 1200, 10000)  # 10% vs 12%
    assert res.significant
    assert res.estimate > 0


def test_cuped_reduces_variance_and_is_unbiased():
    rng = np.random.default_rng(2)
    n = 10000
    x_c = rng.normal(0, 1, n)
    x_t = rng.normal(0, 1, n)
    # Y strongly correlated with X (rho ~ 0.8), plus a real treatment effect.
    y_c = 0.8 * x_c + rng.normal(0, 0.6, n)
    y_t = 0.8 * x_t + rng.normal(0, 0.6, n) + 0.2

    res = run_cuped_test(y_c, x_c, y_t, x_t)

    # Variance reduction should be positive and close to rho^2.
    assert res.variance_reduction > 0.3
    assert abs(res.variance_reduction - res.correlation ** 2) < 1e-9

    # Adjusted variance is actually smaller than raw variance.
    raw_var = np.var(np.concatenate([y_c, y_t]), ddof=1)
    adj_var = np.var(
        np.concatenate([res.y_control_adj, res.y_treatment_adj]), ddof=1
    )
    assert adj_var < raw_var

    # CUPED must not bias the estimated effect (true effect = 0.2).
    assert abs(res.test.estimate - 0.2) < 0.05


def test_cuped_adjust_theta_sign():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 5000)
    y = 2.0 * x + rng.normal(0, 0.5, 5000)  # theta should be ~2.0
    theta, _ = cuped_adjust(y, x)
    assert abs(theta - 2.0) < 0.1
