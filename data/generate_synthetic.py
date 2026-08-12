"""
Generate a realistic synthetic experiment so the demo is fully reproducible.

Scenario: an e-commerce checkout test. Each user has a *pre-period* spend
(the two weeks before the test) that correlates with their *in-experiment*
spend. Treatment lifts in-experiment spend by a small, realistic amount.

The pre-period spend is exactly the kind of covariate CUPED exploits, so this
dataset lets the demo show a large, honest variance reduction.

Output: data/experiment.csv with columns
    user_id, group ('control'|'treatment'), pre_spend, spend, converted
"""

import numpy as np
import pandas as pd


def generate(
    n_per_group: int = 20_000,
    true_lift: float = 0.80,      # absolute lift in mean spend (treatment)
    base_spend: float = 25.0,
    corr_strength: float = 0.75,  # how strongly pre-period predicts in-period
    base_conv: float = 0.110,     # control conversion rate
    conv_lift: float = 0.006,     # absolute lift in conversion (treatment)
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    group = np.array(["control"] * n_per_group + ["treatment"] * n_per_group)
    is_t = (group == "treatment").astype(float)

    # Pre-period spend: right-skewed, like real revenue.
    pre_spend = rng.gamma(shape=2.0, scale=base_spend / 2.0, size=n)

    # In-experiment spend correlated with pre_spend, plus the treatment lift.
    noise = rng.normal(0, base_spend * 0.6, size=n)
    spend = (
        base_spend
        + corr_strength * (pre_spend - pre_spend.mean())
        + true_lift * is_t
        + noise
    )
    spend = np.clip(spend, 0, None)

    # Conversion (binary) metric.
    p = base_conv + conv_lift * is_t
    converted = rng.binomial(1, np.clip(p, 0, 1))

    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "group": group,
            "pre_spend": pre_spend.round(2),
            "spend": spend.round(2),
            "converted": converted,
        }
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    out = "data/experiment.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    print(df.groupby("group")[["pre_spend", "spend", "converted"]].mean())
