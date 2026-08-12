"""
End-to-end walkthrough of the toolkit on the synthetic checkout experiment.

Run from the project root:

    python data/generate_synthetic.py     # once, to create data/experiment.csv
    python examples/demo.py

It prints a full experiment read and saves two figures to figures/.
"""

import os

import numpy as np
import pandas as pd

from abtest import (
    sample_size_means,
    sample_size_proportions,
    two_sample_test,
    two_proportion_test,
    run_cuped_test,
    MSPRT,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "experiment.csv")
FIGS = os.path.join(HERE, "figures")


def section(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def main():
    if not os.path.exists(DATA):
        raise SystemExit("Run `python data/generate_synthetic.py` first.")

    df = pd.read_csv(DATA)
    ctrl = df[df.group == "control"]
    trt = df[df.group == "treatment"]

    # ------------------------------------------------------------------ #
    section("1. POWER ANALYSIS  (planning, before you launch)")
    base_spend_sd = df.spend.std()
    n_means = sample_size_means(mean_diff=1.0, sd=base_spend_sd, power=0.80)
    n_prop = sample_size_proportions(0.11, 0.12, power=0.80)
    print(f"Spend metric: to detect a $1.00 lift (sd=${base_spend_sd:.1f}) "
          f"at 80% power you need {n_means:,} users/arm.")
    print(f"Conversion:   to detect 11% -> 12% at 80% power you need "
          f"{n_prop:,} users/arm.")

    # ------------------------------------------------------------------ #
    section("2. FIXED-HORIZON TESTS  (the classical read)")
    spend_res = two_sample_test(ctrl.spend, trt.spend)
    conv_res = two_proportion_test(
        int(ctrl.converted.sum()), len(ctrl),
        int(trt.converted.sum()), len(trt),
    )
    print("Spend      ", spend_res)
    print("Conversion ", conv_res)

    # ------------------------------------------------------------------ #
    section("3. CUPED  (variance reduction using pre-experiment spend)")
    cuped = run_cuped_test(
        ctrl.spend.values, ctrl.pre_spend.values,
        trt.spend.values, trt.pre_spend.values,
    )
    print(cuped)
    naive_ci_w = spend_res.ci_high - spend_res.ci_low
    cuped_ci_w = cuped.test.ci_high - cuped.test.ci_low
    print(f"CI width: naive=${naive_ci_w:.3f}  CUPED=${cuped_ci_w:.3f}  "
          f"-> {(1 - cuped_ci_w / naive_ci_w) * 100:.1f}% narrower")
    equiv = sample_size_means(mean_diff=1.0,
                              sd=base_spend_sd * np.sqrt(1 - cuped.variance_reduction),
                              power=0.80)
    print(f"Equivalent sample size with CUPED: {equiv:,}/arm "
          f"(vs {n_means:,} without) -> same power, "
          f"{(1 - equiv / n_means) * 100:.0f}% fewer users.")

    # ------------------------------------------------------------------ #
    section("4. SEQUENTIAL / ALWAYS-VALID TEST  (safe to peek)")
    # Stream users in arrival order; look every 250 users per arm.
    seq = MSPRT(alpha=0.05, mde=1.0, burn_in=100).analyze(
        ctrl.spend.values, trt.spend.values, step=250
    )
    if seq.rejected:
        print(f"Significant at n={seq.stopped_at:,}/arm "
              f"(the fixed-horizon plan wanted {n_means:,}/arm).")
    else:
        print("Not significant within the observed data.")
    print(f"Final always-valid p={seq.p_value[-1]:.4f}, "
          f"CI=({seq.ci_low[-1]:+.3f}, {seq.ci_high[-1]:+.3f})")

    _make_figures(seq)
    print(f"\nFigures written to {FIGS}/")


def _make_figures(seq):
    """Save (1) the confidence sequence and (2) the peeking-FPR comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("(matplotlib not installed; skipping figures)")
        return

    os.makedirs(FIGS, exist_ok=True)

    # Figure 1: confidence sequence shrinking over time.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(seq.n, seq.ci_low, seq.ci_high, alpha=0.25,
                    label="anytime-valid CI")
    ax.plot(seq.n, seq.estimate, lw=2, label="estimated lift")
    ax.axhline(0, color="k", lw=1, ls="--")
    if seq.stopped_at:
        ax.axvline(seq.stopped_at, color="C3", ls=":",
                   label=f"stopped @ {seq.stopped_at:,}")
    ax.set_xlabel("users per arm")
    ax.set_ylabel("treatment - control (spend)")
    ax.set_title("Confidence sequence: valid at every look")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "confidence_sequence.png"), dpi=130)
    plt.close(fig)

    # Figure 2: false-positive rate under peeking, naive vs always-valid.
    rng = np.random.default_rng(0)
    sims, n_max, step, alpha = 600, 3000, 100, 0.05
    naive = msprt = 0
    for _ in range(sims):
        c = rng.normal(0, 1, n_max)
        t = rng.normal(0, 1, n_max)  # null
        rej = False
        for n in range(step, n_max + 1, step):
            _, p = stats.ttest_ind(t[:n], c[:n], equal_var=False)
            if p < alpha:
                rej = True
                break
        naive += rej
        msprt += MSPRT(alpha=alpha, tau=0.2, burn_in=30).analyze(
            c, t, step=step).rejected

    fig, ax = plt.subplots(figsize=(6, 4.5))
    rates = [naive / sims, msprt / sims]
    bars = ax.bar(["naive peeking\n(t-test)", "always-valid\n(mSPRT)"], rates,
                  color=["C3", "C0"])
    ax.axhline(alpha, color="k", ls="--", label=f"nominal alpha = {alpha}")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.005, f"{r:.1%}",
                ha="center", fontweight="bold")
    ax.set_ylabel("false-positive rate under H0")
    ax.set_title(f"Type-I error with {n_max // step} looks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "peeking_false_positives.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
