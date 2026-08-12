"""
Sequential / always-valid inference for a difference in means.

The problem with classical tests: they are only valid at a *single*,
pre-committed sample size. If you watch the p-value and stop as soon as it
dips below 0.05 ("peeking"), your real false-positive rate is far above 5%.
This is the single most common way online experiments produce false wins.

The fix implemented here is the **mixture Sequential Probability Ratio Test
(mSPRT)** (Robbins; popularised for A/B testing by Johari, Pekelis & Walsh),
which yields an *always-valid* p-value and an *anytime-valid confidence
sequence*: you may look as often as you like, stop whenever you want, and the
type-I error is still controlled at ``alpha``.

For a difference-in-means estimator ``d`` with variance ``V`` (= s_c^2/n_c +
s_t^2/n_t) and a mixing prior N(0, tau^2) on the true effect:

    Lambda = sqrt(V / (V + tau^2)) *
             exp( tau^2 * d^2 / (2 * V * (V + tau^2)) )

``Lambda`` is a non-negative martingale under H0 with expectation 1, so by
Ville's inequality  P(sup_n Lambda_n >= 1/alpha) <= alpha. The always-valid
p-value is therefore ``p = min(1, 1/Lambda)``, and inverting the test gives the
confidence sequence

    d +/- sqrt( 2 * V * (V + tau^2) / tau^2 * log( sqrt((V+tau^2)/V) / alpha ) )

Variance is estimated from data (plug-in), so guarantees are asymptotic; a
short burn-in avoids the small-sample regime.

``tau`` is the analyst's prior std on the true effect. It affects *power only*,
not validity: larger ``tau`` favours detecting large effects quickly, smaller
``tau`` favours small effects. A good default is your minimum detectable effect.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SequentialResult:
    n: np.ndarray               # cumulative per-arm sample size at each look
    estimate: np.ndarray        # running difference in means
    p_value: np.ndarray         # always-valid (running-min) p-value
    ci_low: np.ndarray          # confidence-sequence lower bound
    ci_high: np.ndarray         # confidence-sequence upper bound
    tau: float
    alpha: float
    stopped_at: int | None      # first per-arm n where H0 was rejected, else None

    @property
    def rejected(self) -> bool:
        return self.stopped_at is not None


def _lambda(d, V, tau2):
    """mSPRT mixture likelihood ratio (vectorised over looks)."""
    return np.sqrt(V / (V + tau2)) * np.exp(tau2 * d ** 2 / (2 * V * (V + tau2)))


def _cs_halfwidth(V, tau2, alpha):
    """Half-width of the anytime-valid confidence sequence."""
    return np.sqrt(
        2 * V * (V + tau2) / tau2 * np.log(np.sqrt((V + tau2) / V) / alpha)
    )


class MSPRT:
    """Mixture SPRT for the difference in means of two streams."""

    def __init__(self, alpha: float = 0.05, tau: float | None = None,
                 mde: float | None = None, burn_in: int = 30):
        self.alpha = alpha
        self.tau = tau
        self.mde = mde
        self.burn_in = burn_in

    def _resolve_tau(self, control, treatment) -> float:
        if self.tau is not None:
            return self.tau
        if self.mde is not None:
            return self.mde
        # Fallback prior: 20% of the pooled outcome standard deviation.
        pooled_sd = np.sqrt(
            (np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2
        )
        return 0.2 * pooled_sd

    def analyze(self, control, treatment, step: int = 1) -> SequentialResult:
        """Replay two arrays as if they had streamed in, look-by-look.

        ``control`` and ``treatment`` are the per-user observations in arrival
        order. A "look" is taken every ``step`` new users per arm.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        n_max = min(len(control), len(treatment))
        tau = self._resolve_tau(control, treatment)
        tau2 = tau ** 2

        looks = list(range(max(self.burn_in, step), n_max + 1, step))
        if not looks or looks[-1] != n_max:
            looks.append(n_max)

        ns, est, raw_p, lo, hi = [], [], [], [], []
        for n in looks:
            c, t = control[:n], treatment[:n]
            d = t.mean() - c.mean()
            V = c.var(ddof=1) / n + t.var(ddof=1) / n
            if V <= 0:
                continue
            lam = _lambda(d, V, tau2)
            h = _cs_halfwidth(V, tau2, self.alpha)
            ns.append(n)
            est.append(d)
            raw_p.append(min(1.0, 1.0 / lam))
            lo.append(d - h)
            hi.append(d + h)

        raw_p = np.array(raw_p)
        # Always-valid p-value is monotone non-increasing (running minimum).
        av_p = np.minimum.accumulate(raw_p)

        stopped_at = None
        below = np.where(av_p <= self.alpha)[0]
        if below.size:
            stopped_at = int(np.array(ns)[below[0]])

        return SequentialResult(
            n=np.array(ns),
            estimate=np.array(est),
            p_value=av_p,
            ci_low=np.array(lo),
            ci_high=np.array(hi),
            tau=float(tau),
            alpha=self.alpha,
            stopped_at=stopped_at,
        )


def always_valid_pvalue(d: float, V: float, tau: float) -> float:
    """Single-look always-valid p-value for difference ``d`` with variance ``V``."""
    return float(min(1.0, 1.0 / _lambda(d, V, tau ** 2)))


def confidence_sequence(d: float, V: float, tau: float, alpha: float = 0.05):
    """Single-look anytime-valid CI ``(low, high)``."""
    h = float(_cs_halfwidth(V, tau ** 2, alpha))
    return d - h, d + h
