"""Generic seed-level direction test, matching the exact methodology used
for the original SGD architecture-fixed ablation
(activation_ablation_seedlevel_stats.csv): one statistic per independent
seed trajectory (never pooled epochs across seeds -- that was the
pseudoreplication bug caught and discarded earlier in this project), then
an exact binomial sign test across seeds.

For each (group_cols..., seed), computes the Pearson correlation between
epoch index and the metric across that seed's own 25-epoch trajectory
(r > 0 => rising, r < 0 => declining), plus the metric's value at the
first and last logged epoch. Then, per group (excluding seed), reports how
many of the seeds show a negative vs. positive trend and the two-sided
binomial sign-test p-value against a coin-flip null.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def seedlevel_stats(df, group_cols, metric_cols=("mean_active_frac", "mean_effective_rank")):
    rows = []
    for key, g in df.groupby(group_cols + ["seed"]):
        g = g.sort_values("epoch")
        row = dict(zip(group_cols + ["seed"], key if isinstance(key, tuple) else (key,)))
        row["n_epochs"] = len(g)
        for metric in metric_cols:
            r = float(np.corrcoef(g["epoch"], g[metric])[0, 1]) if g[metric].std() > 0 else float("nan")
            row[f"r_{metric}"] = r
            row[f"{metric}_start"] = float(g[metric].iloc[0])
            row[f"{metric}_end"] = float(g[metric].iloc[-1])
            row[f"{metric}_delta"] = float(g[metric].iloc[-1] - g[metric].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def sign_test_summary(seedlevel_df, group_cols, metric="mean_active_frac"):
    """Two statistics per group, reported side by side because they can
    disagree on non-monotonic trajectories (observed on Tiny-ImageNet: an
    early overshoot followed by partial relaxation gives a negative
    linear-trend r even when the net start-to-end displacement is
    consistently positive): the LINEAR-TREND sign (r_<metric>, this
    project's primary methodology elsewhere -- Pearson r between epoch
    index and the metric across the full trajectory) and the simpler RAW
    NET DISPLACEMENT sign (end value minus start value). Both are reported;
    neither is silently preferred."""
    rcol, dcol = f"r_{metric}", f"{metric}_delta"
    rows = []
    for key, g in seedlevel_df.groupby(group_cols):
        n = len(g)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        for col, label in ((rcol, "trend"), (dcol, "delta")):
            n_neg = int((g[col] < 0).sum())
            n_pos = int((g[col] > 0).sum())
            majority = "decline" if n_neg > n_pos else "rise"
            n_majority = max(n_neg, n_pos)
            p = stats.binomtest(n_majority, n, p=0.5, alternative="two-sided").pvalue if n else float("nan")
            row.update({f"n_seeds": n, f"n_decline_{label}": n_neg, f"n_rise_{label}": n_pos,
                        f"majority_{label}": majority, f"all_agree_{label}": (n_majority == n),
                        f"sign_test_p_{label}": p})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, nargs="+", help="one or more CSVs to concatenate")
    ap.add_argument("--pool-cols", nargs="+", default=["activation", "optimizer"],
                     help="columns to pool over when reporting the sign test (e.g. pool across "
                          "arch+dataset to get one statistic per activation). Trajectory identity "
                          "(arch/activation/optimizer/dataset/seed, whichever are present) is always "
                          "used in full for the underlying per-seed statistic -- pooling only happens "
                          "at the sign-test-summary step, never by merging distinct runs' epoch rows.")
    ap.add_argument("--metric", default="mean_active_frac")
    ap.add_argument("--out-seedlevel", default=None)
    ap.add_argument("--out-summary", default=None)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)
    id_cols = [c for c in ("arch", "activation", "optimizer", "dataset") if c in df.columns]
    pool_cols = [c for c in args.pool_cols if c in df.columns]
    seedlevel = seedlevel_stats(df, id_cols)
    summary = sign_test_summary(seedlevel, pool_cols, metric=args.metric)

    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))

    if args.out_seedlevel:
        seedlevel.to_csv(args.out_seedlevel, index=False)
    if args.out_summary:
        summary.to_csv(args.out_summary, index=False)


if __name__ == "__main__":
    main()
