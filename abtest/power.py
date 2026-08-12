"""
Power analysis: how many users do you need, and what power do you have?

Two families of formulas are provided, for the two most common metric types:

* means / continuous metrics (revenue, time-on-site, ...)   -> t/z approximation
* proportions / binary metrics (conversion, retention, ...)  -> two-proportion z

All functions use the normal approximation, which is standard for the sample
sizes seen in online experiments. Sample-size results are *per group*.
"""

import math
from dataclasses import dataclass

from scipy.stats import norm


def _z_alpha(alpha: float, alternative: str) -> float:
    """Critical z for the type-I error rate, one- or two-sided."""
    if alternative == "two-sided":
        return norm.ppf(1 - alpha / 2)
    if alternative in ("larger", "smaller", "one-sided"):
        return norm.ppf(1 - alpha)
    raise ValueError("alternative must be 'two-sided' or 'one-sided'")


# ---------------------------------------------------------------------------
# Continuous metrics (means)
# ---------------------------------------------------------------------------
def sample_size_means(
    effect_size: float | None = None,
    mean_diff: float | None = None,
    sd: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """Per-group sample size to detect a difference in means.

    Provide either a standardised ``effect_size`` (Cohen's d), or the raw
    ``mean_diff`` together with the metric standard deviation ``sd``.
    """
    if effect_size is None:
        if mean_diff is None or sd is None:
            raise ValueError("Provide effect_size, or both mean_diff and sd.")
        effect_size = mean_diff / sd
    if effect_size == 0:
        raise ValueError("effect_size must be non-zero.")

    za = _z_alpha(alpha, alternative)
    zb = norm.ppf(power)
    n = 2 * (za + zb) ** 2 / effect_size ** 2
    return math.ceil(n)


def power_means(
    n_per_group: int,
    effect_size: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Achieved power for a given per-group sample size (means)."""
    za = _z_alpha(alpha, alternative)
    ncp = abs(effect_size) * math.sqrt(n_per_group / 2)  # non-centrality
    return float(norm.cdf(ncp - za))


# ---------------------------------------------------------------------------
# Binary metrics (proportions)
# ---------------------------------------------------------------------------
def sample_size_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """Per-group sample size to detect a change from baseline ``p1`` to ``p2``.

    Uses the pooled-variance term under the null and the unpooled term under
    the alternative (the standard, slightly-conservative formulation).
    """
    if p2 == p1:
        raise ValueError("p1 and p2 must differ.")
    za = _z_alpha(alpha, alternative)
    zb = norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (
        za * math.sqrt(2 * pbar * (1 - pbar))
        + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    den = (p2 - p1) ** 2
    return math.ceil(num / den)


def power_proportions(
    n_per_group: int,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Achieved power for a given per-group sample size (proportions)."""
    za = _z_alpha(alpha, alternative)
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n_per_group)
    se1 = math.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z = (abs(p2 - p1) - za * se0) / se1
    return float(norm.cdf(z))


@dataclass
class PowerCurve:
    """Convenience container to sweep sample size vs power."""

    ns: list
    powers: list

    def min_n_for(self, target_power: float) -> int:
        for n, p in zip(self.ns, self.powers):
            if p >= target_power:
                return n
        return self.ns[-1]
