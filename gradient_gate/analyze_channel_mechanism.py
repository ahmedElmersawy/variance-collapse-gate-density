"""Analysis for P2 (run_channel_mechanism.py). Tests the mechanism's claim
at the unit it actually makes a claim about: the individual channel.

THREE tests were tried, in the order they were actually discovered to be
wrong/incomplete -- kept here deliberately, because the progression is part
of the evidence, not just the final number:

1. RAW-MARGIN CORRELATION (first attempt, kept for transparency): per
   channel, correlate margin_c(epoch) = mu_c(epoch) - z_low against
   active_frac_c(epoch) across the 5 logged epochs. Strong and positive for
   relu (frac_positive_corr ~0.87, mean_corr ~+0.61) but weak/slightly
   NEGATIVE for gelu/silu/mish (~0.41-0.46, mean_corr ~-0.13). This is not
   evidence against the mechanism -- diagnosing it (see point 3) is.

2. LAGGED SIGN TEST (does epoch-0 margin's sign predict the epoch-24 total
   active_frac delta's sign?): the mirror image of test 1 -- weak for relu
   (~0.46, chance), strong for gelu/silu/mish (~0.85-0.90).

3. Z-SCORE-MARGIN CORRELATION (the corrected, primary test): tests 1 and 2
   disagreeing by activation is explained by checking each channel's margin
   in UNITS OF ITS OWN SPREAD: at epoch 0, mean (margin / sigma) is ~0.0 for
   relu (z_low sits at the CENTER of the channel distribution -- the
   "thin-margin" case) but ~0.9-1.5 for gelu/silu/mish (z_low sits CLOSE TO
   but not deep inside the bulk -- a "thick-margin" case). Sigma shrinks
   ~3x over training for every activation (0.83-0.92 -> 0.19-0.31); mu also
   drifts slightly negative for every activation (the population-level
   finding this project already established). For relu, mu's drift directly
   crosses the live threshold -- margin alone tracks active_frac. For
   gelu/silu/mish, mu's small negative drift actually pulls the RAW margin
   down even as sigma's much larger relative shrinkage pulls active_frac UP
   by thinning the sub-threshold left tail (the project's earlier
   "quantile-compression" finding, now localized to the per-channel level)
   -- two different forces moving the raw margin and active_frac in
   opposite apparent directions, which is exactly why test 1 came out
   negative for those three. Replacing the margin with a sigma-normalized
   z-score, (mu - z_low) / sigma -- the standard, correct way to ask "how
   many channel-widths is the mean from the boundary," not "how many raw
   units" -- removes the confound: ALL FOUR activations now show strong,
   uniform, positive per-channel correlation (frac_positive_corr 0.90-0.96,
   mean_corr +0.66-0.80), including a STRONGER result for relu itself than
   the unnormalized test gave it.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from gradient_gate.run_training_dynamics import CSV_DIR


def per_channel_corr(pivot_x, pivot_y):
    corrs = []
    for ch in pivot_x.index:
        x, y = pivot_x.loc[ch], pivot_y.loc[ch]
        if x.std() > 1e-9 and y.std() > 1e-9:
            corrs.append(float(np.corrcoef(x.values, y.values)[0, 1]))
    return corrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(CSV_DIR, "channel_mechanism.csv"))
    ap.add_argument("--zlow", default=os.path.join(CSV_DIR, "channel_mechanism_zlow.csv"))
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "channel_mechanism_summary.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    zlow = pd.read_csv(args.zlow).set_index("activation")["z_low"]
    df["channel_id"] = df["layer"] + "::" + df["channel"].astype(str)
    df["zscore_margin"] = df.apply(lambda r: (r.mu - zlow[r.activation]) / r.sigma if r.sigma > 0 else np.nan, axis=1)

    epochs = sorted(df.epoch.unique())
    first_epoch, last_epoch = epochs[0], epochs[-1]
    print(f"epochs found: {epochs}")

    rows = []
    for (activation, seed), g in df.groupby(["activation", "seed"]):
        z = zlow[activation]
        pivot_mu = g.pivot(index="channel_id", columns="epoch", values="mu")
        pivot_af = g.pivot(index="channel_id", columns="epoch", values="active_frac")
        pivot_z = g.pivot(index="channel_id", columns="epoch", values="zscore_margin")
        if first_epoch not in pivot_mu.columns or last_epoch not in pivot_mu.columns:
            continue

        margin0 = pivot_mu[first_epoch] - z
        d_active_frac = pivot_af[last_epoch] - pivot_af[first_epoch]
        moved = d_active_frac.abs() > 1e-6
        sign_match = (np.sign(margin0[moved]) == np.sign(d_active_frac[moved]))
        frac_match_sign = float(sign_match.mean()) if moved.any() else float("nan")

        corrs_raw = per_channel_corr(pivot_mu.sub(z), pivot_af)
        corrs_z = per_channel_corr(pivot_z, pivot_af)

        rows.append(dict(
            activation=activation, seed=seed, z_low=z, n_channels=len(pivot_mu),
            frac_sign_match=frac_match_sign,
            frac_positive_corr_raw=float(np.mean([c > 0 for c in corrs_raw])) if corrs_raw else np.nan,
            mean_corr_raw=float(np.mean(corrs_raw)) if corrs_raw else np.nan,
            frac_positive_corr_zscore=float(np.mean([c > 0 for c in corrs_z])) if corrs_z else np.nan,
            mean_corr_zscore=float(np.mean(corrs_z)) if corrs_z else np.nan,
        ))
        r = rows[-1]
        print(f"{activation:6s} seed={seed} z_low={z:+.4f} n_channels={len(pivot_mu):5d} "
              f"frac_sign_match={frac_match_sign:.3f} "
              f"frac_pos_corr_raw={r['frac_positive_corr_raw']:.3f} mean_corr_raw={r['mean_corr_raw']:+.3f}  "
              f"frac_pos_corr_zscore={r['frac_positive_corr_zscore']:.3f} mean_corr_zscore={r['mean_corr_zscore']:+.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)

    print("\n--- PRIMARY (z-score margin, sigma-normalized): sign test on frac_positive_corr_zscore > 0.5 across seeds ---")
    for activation, g in summary.groupby("activation"):
        n_above_half = int((g.frac_positive_corr_zscore > 0.5).sum())
        n_total = len(g)
        p = stats.binomtest(n_above_half, n_total, p=0.5, alternative="greater").pvalue if n_total else float("nan")
        print(f"{activation:6s} {n_above_half}/{n_total} seeds, mean_frac_pos={g.frac_positive_corr_zscore.mean():.3f}, "
              f"mean_corr={g.mean_corr_zscore.mean():+.3f}, binomial p={p:.4f}")

    print("\n--- diagnostic only (raw, unnormalized margin -- see module docstring for why this fails for gelu/silu/mish) ---")
    for activation, g in summary.groupby("activation"):
        print(f"{activation:6s} mean_frac_pos_corr_raw={g.frac_positive_corr_raw.mean():.3f}, "
              f"mean_corr_raw={g.mean_corr_raw.mean():+.3f}, mean_frac_sign_match(lagged)={g.frac_sign_match.mean():.3f}")


if __name__ == "__main__":
    main()
