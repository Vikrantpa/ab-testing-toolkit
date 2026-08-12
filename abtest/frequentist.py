"""
Fixed-horizon (classical) two-sample tests.

These are the tests you run *once*, at a pre-committed sample size. They are
the baseline the sequential methods in ``sequential.py`` improve upon. Both
return a rich :class:`TestResult` so downstream code / reports read cleanly.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    estimate: float          # treatment - control
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    method: str
    n_control: int
    n_treatment: int

    def __str__(self) -> str:
        star = "significant" if self.significant else "not significant"
        return (
            f"[{self.method}] estimate={self.estimate:+.4f} "
            f"{100*(1-self.alpha):.0f}% CI=({self.ci_low:+.4f}, {self.ci_high:+.4f}) "
            f"p={self.p_value:.4f} -> {star}"
        )


def two_sample_test(
    control,
    treatment,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Welch's t-test for a difference in means (unequal variances)."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    nc, nt = len(control), len(treatment)

    t_stat, p = stats.ttest_ind(
        treatment, control, equal_var=False, alternative=alternative
    )

    diff = treatment.mean() - control.mean()
    vc, vt = control.var(ddof=1), treatment.var(ddof=1)
    se = np.sqrt(vc / nc + vt / nt)
    # Welch–Satterthwaite degrees of freedom
    df = (vc / nc + vt / nt) ** 2 / (
        (vc / nc) ** 2 / (nc - 1) + (vt / nt) ** 2 / (nt - 1)
    )
    tcrit = stats.t.ppf(1 - alpha / 2, df)

    return TestResult(
        estimate=float(diff),
        ci_low=float(diff - tcrit * se),
        ci_high=float(diff + tcrit * se),
        statistic=float(t_stat),
        p_value=float(p),
        alpha=alpha,
        significant=bool(p < alpha),
        method="Welch t-test",
        n_control=nc,
        n_treatment=nt,
    )


def two_proportion_test(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Two-proportion z-test.

    Pooled variance is used for the test statistic (correct under the null);
    unpooled variance is used for the confidence interval (correct under the
    observed rates).
    """
    p_c = x_control / n_control
    p_t = x_treatment / n_treatment
    diff = p_t - p_c

    p_pool = (x_control + x_treatment) / (n_control + n_treatment)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = diff / se_pool

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative in ("larger", "one-sided"):
        p = 1 - stats.norm.cdf(z)
    else:  # smaller
        p = stats.norm.cdf(z)

    se_un = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    zcrit = stats.norm.ppf(1 - alpha / 2)

    return TestResult(
        estimate=float(diff),
        ci_low=float(diff - zcrit * se_un),
        ci_high=float(diff + zcrit * se_un),
        statistic=float(z),
        p_value=float(p),
        alpha=alpha,
        significant=bool(p < alpha),
        method="Two-proportion z-test",
        n_control=n_control,
        n_treatment=n_treatment,
    )
