"""
CUPED — Controlled-experiment Using Pre-Experiment Data.

Idea (Deng, Xu, Kohavi & Walker, 2013): if you have a covariate X measured
*before* the experiment that correlates with the outcome Y, you can subtract
off its explainable variation and shrink the metric's variance without biasing
the treatment effect. Less variance -> tighter CIs -> smaller samples / faster
reads for the same power.

    Y_cuped = Y - theta * (X - E[X]),     theta = Cov(Y, X) / Var(X)

The variance of the adjusted metric is Var(Y) * (1 - rho^2), so the fractional
variance reduction is exactly rho^2, the squared correlation between Y and X.

IMPORTANT: X must be a *pre-treatment* covariate (e.g. the same metric measured
in the weeks before the test). Using a post-treatment covariate biases results.
"""

from dataclasses import dataclass, field

import numpy as np

from .frequentist import TestResult, two_sample_test


@dataclass
class CUPEDResult:
    theta: float
    correlation: float
    variance_reduction: float           # fraction of variance removed (= rho^2)
    y_control_adj: np.ndarray = field(repr=False)
    y_treatment_adj: np.ndarray = field(repr=False)
    test: TestResult | None = None

    def __str__(self) -> str:
        line = (
            f"CUPED: theta={self.theta:.4f} corr(Y,X)={self.correlation:.3f} "
            f"variance reduction={self.variance_reduction*100:.1f}%"
        )
        return line if self.test is None else f"{line}\n  adjusted {self.test}"


def cuped_adjust(y, x):
    """Return ``(theta, y_adjusted)`` for a single pooled sample."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    return float(theta), y_adj


def run_cuped_test(
    y_control,
    x_control,
    y_treatment,
    x_treatment,
    alpha: float = 0.05,
) -> CUPEDResult:
    """Apply CUPED and run the adjusted difference-in-means test.

    ``theta`` and the covariate mean are estimated on the *pooled* data (both
    arms), which is the standard unbiased choice because X is pre-treatment.
    """
    y_control = np.asarray(y_control, dtype=float)
    x_control = np.asarray(x_control, dtype=float)
    y_treatment = np.asarray(y_treatment, dtype=float)
    x_treatment = np.asarray(x_treatment, dtype=float)

    y_all = np.concatenate([y_control, y_treatment])
    x_all = np.concatenate([x_control, x_treatment])

    theta = np.cov(y_all, x_all, ddof=1)[0, 1] / np.var(x_all, ddof=1)
    x_bar = x_all.mean()
    rho = np.corrcoef(y_all, x_all)[0, 1]

    yc_adj = y_control - theta * (x_control - x_bar)
    yt_adj = y_treatment - theta * (x_treatment - x_bar)

    result = two_sample_test(yc_adj, yt_adj, alpha=alpha)

    return CUPEDResult(
        theta=float(theta),
        correlation=float(rho),
        variance_reduction=float(rho ** 2),
        y_control_adj=yc_adj,
        y_treatment_adj=yt_adj,
        test=result,
    )
