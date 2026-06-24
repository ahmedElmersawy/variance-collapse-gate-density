"""Publication-style figure for Phase 4: does gate collapse emerge during
NORMAL supervised training? Plots mean +/- 95% CI (across seeds) of gate
density (active_frac), effective rank, and test accuracy vs epoch, faceted
by architecture/dataset, from gradient_gate_outputs/csv/training_dynamics.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs")


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "csv", "training_dynamics.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "training_dynamics_gate_collapse.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    combos = sorted(df.groupby(["arch", "dataset"]).size().index.tolist())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(combos), 3, figsize=(13, 3.2 * len(combos)), squeeze=False)
    for i, (arch, dataset) in enumerate(combos):
        sub = df[(df.arch == arch) & (df.dataset == dataset)]
        n_seeds = sub["seed"].nunique()
        agg = sub.groupby("epoch").agg(
            active_mean=("mean_active_frac", "mean"), active_ci=("mean_active_frac", ci95),
            rank_mean=("mean_effective_rank", "mean"), rank_ci=("mean_effective_rank", ci95),
            acc_mean=("test_acc", "mean"), acc_ci=("test_acc", ci95),
        ).reset_index()

        ax = axes[i][0]
        ax.plot(agg.epoch, agg.active_mean, "-o", ms=3, color="C0")
        ax.fill_between(agg.epoch, agg.active_mean - agg.active_ci, agg.active_mean + agg.active_ci, alpha=0.25, color="C0")
        ax.set_ylabel(f"{arch}\n{dataset}\nactive_frac")
        if i == 0:
            ax.set_title("gate density (active_frac)")

        ax = axes[i][1]
        ax.plot(agg.epoch, agg.rank_mean, "-o", ms=3, color="C1")
        ax.fill_between(agg.epoch, agg.rank_mean - agg.rank_ci, agg.rank_mean + agg.rank_ci, alpha=0.25, color="C1")
        ax.set_ylabel("effective_rank")
        if i == 0:
            ax.set_title("representation rank")

        ax = axes[i][2]
        ax.plot(agg.epoch, agg.acc_mean, "-o", ms=3, color="C2")
        ax.fill_between(agg.epoch, agg.acc_mean - agg.acc_ci, agg.acc_mean + agg.acc_ci, alpha=0.25, color="C2")
        ax.set_ylabel("test_acc")
        if i == 0:
            ax.set_title("test accuracy")
        ax.text(0.97, 0.05, f"n={n_seeds} seeds", transform=ax.transAxes, ha="right", fontsize=8, color="gray")

        for j in range(3):
            axes[i][j].set_xlabel("epoch")

    fig.suptitle("Gate collapse emerges during normal supervised training\n"
                  "(shaded = 95% CI across seeds)", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # quick hypothesis test per (arch,dataset): is active_frac trend with
    # epoch significantly negative (collapsing) and rank trend positive?
    from scipy.stats import spearmanr
    print("\nSpearman(epoch, active_frac) / Spearman(epoch, effective_rank) per (arch,dataset):")
    for arch, dataset in combos:
        sub = df[(df.arch == arch) & (df.dataset == dataset)]
        r1, p1 = spearmanr(sub.epoch, sub.mean_active_frac)
        r2, p2 = spearmanr(sub.epoch, sub.mean_effective_rank)
        print(f"  {arch}/{dataset}: active_frac rho={r1:+.3f} p={p1:.2e}   "
              f"effective_rank rho={r2:+.3f} p={p2:.2e}   (n_rows={len(sub)})")


if __name__ == "__main__":
    main()
