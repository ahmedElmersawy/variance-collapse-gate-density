"""Phase 5: does gradient-gate collapse predict gradient-inversion attack
success? Sweeps the victim AlphaGateClassifier's gate stiffness alpha across
20 seeds, runs THREE attacks per (alpha, seed) against the same true
gradient — DLG (Zhu et al. 2019), iDLG (Zhao et al. 2020), and the stronger
cosine-similarity "Inverting Gradients" baseline (Geiping et al. 2020) — and
correlates the victim's active-gradient-fraction against attack success.
Replaces run_experiments.py's ext_e_gradient_leakage() IoU-threshold PROXY
with real optimization-based reconstruction attacks.

Synthetic "private" image (a randomly placed disc) is used instead of a
downloaded dataset, consistent with the rest of the project's synthetic-
target convention and keeping the experiment fully self-contained.
"""
import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import pearsonr

from gradient_gate.dlg_attack import (AlphaGateClassifier, dlg_attack,
                                       inverting_gradients_attack, reconstruction_quality)

ROOT = os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs")
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_SEEDS = 20
N_BOOT = 2000


def make_private_example(seed: int, img: int = 28, n_classes: int = 10):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(-1, 1, img), np.linspace(-1, 1, img))
    cx, cy = rng.uniform(-0.5, 0.5, 2)
    img_arr = (((xx - cx) ** 2 + (yy - cy) ** 2) < 0.25).astype(np.float32)
    x = torch.from_numpy(img_arr).reshape(1, 1, img, img)
    y = torch.tensor([int(rng.integers(0, n_classes))])
    return x, y


def run_attack(name, model, x_true, y_true, iters, device):
    if name == "dlg":
        return dlg_attack(model, x_true, y_true, iters=iters, use_idlg=False, device=device)
    if name == "idlg":
        return dlg_attack(model, x_true, y_true, iters=iters, use_idlg=True, device=device)
    if name == "inverting_gradients":
        return inverting_gradients_attack(model, x_true, y_true, iters=iters, device=device)
    raise ValueError(name)


def bootstrap_ci_mean(values, n_boot=N_BOOT, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--attacks", nargs="+", default=["dlg", "idlg", "inverting_gradients"])
    ap.add_argument("--out", default=os.path.join(ROOT, "csv", "privacy_gate_collapse.csv"))
    ap.add_argument("--fig", default=os.path.join(ROOT, "figures", "privacy_gate_collapse.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for alpha in ALPHAS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = AlphaGateClassifier(alpha=alpha)
            x_true, y_true = make_private_example(seed)
            for attack in args.attacks:
                x_recon, hist = run_attack(attack, model, x_true, y_true, args.iters, device)
                q = reconstruction_quality(x_true, x_recon)
                rows.append(dict(attack=attack, alpha=alpha, seed=seed,
                                  active_frac=model.last_active_frac,
                                  gate_mean=model.last_gate_mean, final_loss=hist[-1], **q))
            print(f"[privacy] alpha={alpha:5.1f} seed={seed:2d} active_frac={model.last_active_frac:.3f}  "
                  + "  ".join(f"{a}:psnr={r['psnr']:6.1f}dB" for a, r in
                              zip(args.attacks, rows[-len(args.attacks):])))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[privacy] wrote {len(df)} rows -> {args.out}")

    df["converged"] = df["active_frac"] > 0.01

    print("\n" + "=" * 70)
    print("PER-ATTACK ANALYSIS")
    print("=" * 70)
    summary_rows = []
    for attack in args.attacks:
        sub = df[df.attack == attack]
        print(f"\n--- {attack} ---")

        # 1) Convergence rate vs alpha: logistic regression of converged ~ log(alpha).
        #    More principled than a Pearson r on a 7-point aggregate — uses
        #    every seed-level observation (n=alpha_count*seeds) and returns a
        #    proper Wald p-value on the slope, the right tool for a binary
        #    outcome with a monotone-trend hypothesis.
        X = sm.add_constant(np.log(sub["alpha"].values))
        yb = sub["converged"].astype(int).values
        try:
            logit_res = sm.Logit(yb, X).fit(disp=0)
            slope, p_slope = logit_res.params[1], logit_res.pvalues[1]
            print(f"  logit(converged) ~ log(alpha):  slope={slope:+.3f}  p={p_slope:.4g}")
        except Exception as e:
            slope, p_slope = float("nan"), float("nan")
            print(f"  logistic fit failed ({e}) — likely separation (all-0/all-1 outcome)")

        rate_by_alpha = sub.groupby("alpha")["converged"].mean()
        for a in ALPHAS:
            n = int((sub.alpha == a).sum())
            k = int(sub[sub.alpha == a]["converged"].sum())
            mean_b, lo_b, hi_b = bootstrap_ci_mean(sub[sub.alpha == a]["converged"].values)
            print(f"    alpha={a:5.1f}  converged {k:2d}/{n:2d}  rate={mean_b:.2f}  "
                  f"95% CI=[{lo_b:.2f}, {hi_b:.2f}]")

        # 2) Quality | converged vs alpha — Pearson on alpha-level means
        #    (only where n>=5 converged runs so the mean is not a 1-2 point
        #    fluke), plus bootstrap CI per alpha.
        given = sub[sub.converged]
        n_conv = given.groupby("alpha").size()
        valid = n_conv.index[n_conv >= 5]
        qual_by_alpha = given[given.alpha.isin(valid)].groupby("alpha")["corr"].mean()
        if len(valid) >= 3:
            r_qual, p_qual = pearsonr(qual_by_alpha.index, qual_by_alpha.values)
            print(f"  quality|converged corr vs alpha (n_alpha={len(valid)}):  r={r_qual:+.3f}  p={p_qual:.4g}")
        else:
            r_qual, p_qual = float("nan"), float("nan")
            print(f"  quality|converged: insufficient alphas with >=5 converged seeds (n_alpha={len(valid)})")

        summary_rows.append(dict(attack=attack, logit_slope=slope, logit_p=p_slope,
                                  quality_r=r_qual, quality_p=p_qual,
                                  overall_convergence_rate=float(sub["converged"].mean())))

    print("\n" + "=" * 70)
    print("CROSS-ATTACK COMPARISON (does the stronger baseline change the story?)")
    print("=" * 70)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(os.path.dirname(args.out), "privacy_attack_comparison.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        colors = {"dlg": "C0", "idlg": "C1", "inverting_gradients": "C2"}
        for attack in args.attacks:
            sub = df[df.attack == attack]
            rate = sub.groupby("alpha")["converged"].mean()
            cis = [bootstrap_ci_mean(sub[sub.alpha == a]["converged"].values) for a in rate.index]
            lo = [c[1] for c in cis]; hi = [c[2] for c in cis]
            axes[0].plot(rate.index, rate.values, "o-", color=colors.get(attack), label=attack)
            axes[0].fill_between(rate.index, lo, hi, color=colors.get(attack), alpha=0.15)

            given = sub[sub.converged]
            qual = given.groupby("alpha")["corr"].mean()
            axes[1].plot(qual.index, qual.values, "o-", color=colors.get(attack), label=attack)

        axes[0].set_xlabel("alpha (gate stiffness)"); axes[0].set_ylabel("attack convergence rate")
        axes[0].set_xscale("log"); axes[0].set_title("P(attack converges) vs gate collapse")
        axes[0].legend()
        axes[1].set_xlabel("alpha (gate stiffness)"); axes[1].set_ylabel("recon. corr | converged")
        axes[1].set_xscale("log"); axes[1].set_title("reconstruction quality given convergence")
        axes[1].legend()
        fig.suptitle(f"Gate collapse vs gradient-inversion attacks (n={args.seeds} seeds/alpha)")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        fig.savefig(args.fig, dpi=150)
        print(f"\n[privacy] wrote figure -> {args.fig}")
    except Exception as e:
        print(f"[privacy] plotting skipped: {e}")


if __name__ == "__main__":
    main()
