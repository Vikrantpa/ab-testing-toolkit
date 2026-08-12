import math

from abtest import (
    sample_size_means,
    power_means,
    sample_size_proportions,
    power_proportions,
)


def test_sample_size_means_textbook_value():
    # Classic result: d=0.2, alpha=0.05 two-sided, power=0.8 -> ~393 per group.
    n = sample_size_means(effect_size=0.2, alpha=0.05, power=0.80)
    assert 390 <= n <= 396


def test_sample_size_means_from_raw_inputs():
    n_d = sample_size_means(effect_size=0.5)
    n_raw = sample_size_means(mean_diff=5.0, sd=10.0)  # d = 0.5
    assert n_d == n_raw


def test_power_increases_with_n():
    p_small = power_means(100, effect_size=0.2)
    p_large = power_means(1000, effect_size=0.2)
    assert p_large > p_small
    assert 0.0 <= p_small <= 1.0 <= 1.0 + 1e-9


def test_power_roundtrip_means():
    # The n from sample_size should deliver approximately the target power.
    n = sample_size_means(effect_size=0.3, power=0.80)
    assert abs(power_means(n, effect_size=0.3) - 0.80) < 0.02


def test_sample_size_proportions_reasonable():
    # Detecting 10% -> 11% needs a large sample; sanity-check the ballpark.
    n = sample_size_proportions(0.10, 0.11)
    assert 12_000 < n < 22_000


def test_power_roundtrip_proportions():
    n = sample_size_proportions(0.10, 0.12, power=0.80)
    assert abs(power_proportions(n, 0.10, 0.12) - 0.80) < 0.03
