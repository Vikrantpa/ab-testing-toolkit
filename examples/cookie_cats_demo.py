"""
Real-world demo: the Cookie Cats mobile-game A/B test.

Cookie Cats moved a progression gate from level 30 to level 40 and measured the
effect on player retention. This is a genuine, widely-used experiment dataset.

How to get the data (one file, ~4 MB):
  1. Download 'cookie_cats.csv' from Kaggle:
     https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing
  2. Place it at data/cookie_cats.csv
  3. Run:  python examples/cookie_cats_demo.py

Columns: userid, version (gate_30 | gate_40), sum_gamerounds,
         retention_1 (bool), retention_7 (bool)
"""

import os

import pandas as pd

from abtest import (
    two_proportion_test,
    sample_size_proportions,
    two_sample_test,
    MSPRT,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "cookie_cats.csv")


def main():
    if not os.path.exists(DATA):
        raise SystemExit(
            "data/cookie_cats.csv not found.\n"
            "Download it from Kaggle (see the docstring at the top of this "
            "file) and place it in the data/ folder."
        )

    df = pd.read_csv(DATA)
    # 'gate_30' is the control (original), 'gate_40' the treatment.
    ctrl = df[df.version == "gate_30"]
    trt = df[df.version == "gate_40"]

    print(f"Control (gate_30): {len(ctrl):,} players | "
          f"Treatment (gate_40): {len(trt):,} players\n")

    # 7-day retention — the metric the team actually cared about.
    for metric in ["retention_1", "retention_7"]:
        xc, xt = int(ctrl[metric].sum()), int(trt[metric].sum())
        res = two_proportion_test(xc, len(ctrl), xt, len(trt))
        print(f"{metric}: control={xc/len(ctrl):.4f}  "
              f"treatment={xt/len(trt):.4f}")
        print("   ", res, "\n")

    # Would we have been powered to see a 1-point retention drop?
    p0 = ctrl.retention_7.mean()
    n_needed = sample_size_proportions(p0, p0 - 0.01, power=0.80)
    print(f"To detect a 1pp drop in 7-day retention you'd need "
          f"{n_needed:,}/arm; you have {min(len(ctrl), len(trt)):,}.\n")

    # Sequential read on game rounds played (a continuous metric).
    seq = MSPRT(alpha=0.05, burn_in=200).analyze(
        ctrl.sum_gamerounds.values, trt.sum_gamerounds.values, step=1000
    )
    verdict = (f"significant @ n={seq.stopped_at:,}/arm"
               if seq.rejected else "not significant")
    print(f"Sequential test on game rounds played: {verdict} "
          f"(final always-valid p={seq.p_value[-1]:.3f})")


if __name__ == "__main__":
    main()
