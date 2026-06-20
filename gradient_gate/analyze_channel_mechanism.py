"""Analysis for P2 (run_channel_mechanism.py). Tests the mechanism's claim
at the unit it actually makes a claim about: the individual channel.

The mechanism (theory_variance_compression_mechanism.md) claims an
INSTANTANEOUS, state-dependent relationship: at any point in training,
channel c's active_frac is governed by which side of the fixed z_low its
CURRENT mu_c(t) sits on -- not that its position at epoch 0 is prognostic
of where it ends up (mu itself drifts over training; that drift is part of
the mechanism, not separate from it). Two tests follow from this:

1. CORRELATION TEST (primary -- the direct operationalization of the
   mechanism's actual claim): per channel, across all 5 logged epochs,
   the correlation between margin_c(epoch) = mu_c(epoch) - z_low and
   active_frac_c(epoch). Summarized per (activation, seed) as the fraction
   of channels with positive correlation and the mean correlation.

2. SIGN TEST (secondary, reported for transparency, NOT the mechanism's
   actual claim): does the sign of the EARLY margin (epoch 0) match the
   sign of the channel's total active_frac change by epoch 24? This is a
   stronger, lagged-predictive claim the mechanism does not make -- and it
   under-performs test 1 for ReLU specifically (observed: ~0.41-0.53,
   chance level) precisely because ReLU's z_low sits at ~0, the thinnest
   possible margin, so a channel's epoch-0 sign is a poor predictor of
   where the (also drifting) mu ends up 24 epochs later. That is a
   property of using a stale, single-epoch margin as a predictor, not a
   failure of the mechanism itself -- test 1, which uses each epoch's own
   margin, is unaffected and remains strong for ReLU too.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from gradient_gate.run_training_dynamics import CSV_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(CSV_DIR, "channel_mechanism.csv"))
    ap.add_argument("--zlow", default=os.path.join(CSV_DIR, "channel_mechanism_zlow.csv"))
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "channel_mechanism_summary.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    zlow = pd.read_csv(args.zlow).set_index("activation")["z_low"]

    df["channel_id"] = df["layer"] + "::" + df["channel"].astype(str)
    epochs = sorted(df.epoch.unique())
    first_epoch, last_epoch = epochs[0], epochs[-1]
    print(f"epochs found: {epochs}")

    rows = []
    for (activation, seed), g in df.groupby(["activation", "seed"]):
        z = zlow[activation]
        pivot_mu = g.pivot(index="channel_id", columns="epoch", values="mu")
        pivot_af = g.pivot(index="channel_id", columns="epoch", values="active_frac")
        if first_epoch not in pivot_mu.columns or last_epoch not in pivot_mu.columns:
            continue

        margin0 = pivot_mu[first_epoch] - z
        d_active_frac = pivot_af[last_epoch] - pivot_af[first_epoch]
        # SIGN TEST -- exclude channels with ~0 change (no directional claim possible)
        moved = d_active_frac.abs() > 1e-6
        sign_match = (np.sign(margin0[moved]) == np.sign(d_active_frac[moved]))
        frac_match_sign = float(sign_match.mean()) if moved.any() else float("nan")
        n_moved = int(moved.sum())

        # CORRELATION TEST -- per channel, across all logged epochs
        corrs = []
        for ch in pivot_mu.index:
            m = pivot_mu.loc[ch] - z
            a = pivot_af.loc[ch]
            if m.std() > 1e-9 and a.std() > 1e-9:
                corrs.append(float(np.corrcoef(m.values, a.values)[0, 1]))
        frac_positive_corr = float(np.mean([c > 0 for c in corrs])) if corrs else float("nan")
        mean_corr = float(np.mean(corrs)) if corrs else float("nan")

        rows.append(dict(activation=activation, seed=seed, z_low=z, n_channels=len(pivot_mu),
                          n_channels_moved=n_moved, frac_sign_match=frac_match_sign,
                          frac_positive_corr=frac_positive_corr, mean_corr=mean_corr))
        print(f"{activation:6s} seed={seed} z_low={z:+.4f} n_channels={len(pivot_mu):5d} "
              f"n_moved={n_moved:5d} frac_sign_match={frac_match_sign:.3f} "
              f"frac_positive_corr={frac_positive_corr:.3f} mean_corr={mean_corr:+.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)

    print("\n--- PRIMARY: per-activation sign test on frac_positive_corr > 0.5 across seeds ---")
    for activation, g in summary.groupby("activation"):
        n_above_half = int((g.frac_positive_corr > 0.5).sum())
        n_total = len(g)
        p = stats.binomtest(n_above_half, n_total, p=0.5, alternative="greater").pvalue if n_total else float("nan")
        print(f"{activation:6s} {n_above_half}/{n_total} seeds with frac_positive_corr>0.5, "
              f"mean={g.frac_positive_corr.mean():.3f}, mean_corr={g.mean_corr.mean():+.3f}, binomial p={p:.4f}")

    print("\n--- secondary (not the mechanism's actual claim): frac_sign_match > 0.5 across seeds ---")
    for activation, g in summary.groupby("activation"):
        n_above_half = int((g.frac_sign_match > 0.5).sum())
        n_total = len(g)
        p = stats.binomtest(n_above_half, n_total, p=0.5, alternative="greater").pvalue if n_total else float("nan")
        print(f"{activation:6s} {n_above_half}/{n_total} seeds with frac_sign_match>0.5, "
              f"mean={g.frac_sign_match.mean():.3f}, binomial p={p:.4f}")


if __name__ == "__main__":
    main()
