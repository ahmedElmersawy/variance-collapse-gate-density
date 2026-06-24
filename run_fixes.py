#!/usr/bin/env python3
"""
Targeted runner for three specific fixes:
  Fix 1 — Effective rank K_EIG: rerun with min(n-2, max(150, n//3))
  Fix 2 — Oracle ablation: adam+oracle_pgd+pgd for alpha=[1,2,5,10,20,40], steps=200
  Fix 3 — Phase transition fit: sigmoid fit on Adam data from alpha_sweep_results.csv
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Point at the existing full-data output directory so we have alpha_sweep_results.csv
ROOT_DIR_TARGET = os.path.join(os.path.expanduser("~"), "gradient_gate_outputs")
sys.argv = [
    "run_fixes.py",
    "--profile", "full",
    "--skip-mnist",
    "--skip-deepnet",
    "--root", ROOT_DIR_TARGET,
]

import run_experiments as exp  # imports module-level setup with correct ROOT_DIR

print(f"[setup] ROOT_DIR : {exp.ROOT_DIR}")
print(f"[setup] CSV_DIR  : {exp.CSV_DIR}")
print(f"[setup] FIG_DIR  : {exp.FIG_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — Effective rank sweep with corrected K_EIG
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FIX 1: Effective rank sweep — K_EIG = min(n-2, max(150, n//3))")
print("="*60)

for fname in ("effective_rank_vs_alpha.csv", "effective_rank_vs_kernel.csv",
              "eigenvalue_spectra.csv"):
    fpath = os.path.join(exp.CSV_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f"[deleted] {fname}")

erank_df = exp.run_effective_rank_sweep()


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — Oracle ablation: adam + oracle_pgd + pgd, steps=200
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FIX 2: Oracle ablation — adam / oracle_pgd / pgd, steps=200")
print("="*60)

oracle_path = os.path.join(exp.CSV_DIR, "oracle_ablation.csv")
if os.path.exists(oracle_path):
    os.remove(oracle_path)
    print("[deleted] oracle_ablation.csv (will rerun with steps=200)")

oracle_alphas = [1, 2, 5, 10, 20, 40]
oracle_opts = [
    ("adam",       {"lr": 0.03, "steps": 200}),
    ("oracle_pgd", {"lr": 0.1,  "steps": 200}),
    ("pgd",        {"lr": 0.1,  "steps": 200}),
]

oracle_rows = []
for alpha in oracle_alphas:
    for opt, kw in oracle_opts:
        for seed in range(5):
            p = exp.run_single_experiment(
                alpha=float(alpha),
                optimizer_name=opt,
                optimizer_kwargs=kw,
                seed=seed,
                kernel_name="sobel_x",
                target_name="checkerboard",
            )
            s = p["summary"]
            oracle_rows.append(s)
            print(f"  alpha={alpha:>4.0f} {opt:<12s} seed={seed}: IoU={s['output_iou_final']:.3f}")

oracle_df = pd.DataFrame(oracle_rows)
oracle_df.to_csv(oracle_path, index=False)
print(f"[saved] oracle_ablation.csv ({len(oracle_df)} rows)")

# Regenerate oracle_ablation.png
exp.plot_oracle_ablation(oracle_df)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — Phase transition sigmoid fit
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FIX 3: Phase transition sigmoid fit from alpha_sweep_results.csv")
print("="*60)

alpha_df = exp.load_csv("alpha_sweep_results.csv")
if alpha_df is None:
    print("[error] alpha_sweep_results.csv not found — cannot fit phase transition")
else:
    adam = (alpha_df[alpha_df["optimizer"] == "adam"]
            .groupby("alpha")["output_iou_final"]
            .agg(["mean", "std", "count"])
            .reset_index())

    def sigmoid_model(alpha, iou_max, iou_min, alpha_star, delta):
        return iou_min + (iou_max - iou_min) / (1.0 + np.exp((alpha - alpha_star) / (delta + 1e-8)))

    popt, _ = curve_fit(
        sigmoid_model, adam["alpha"], adam["mean"],
        p0=[0.894, 0.625, 11.7, 6.6],
        bounds=([0.3, -0.05, 0.5, 0.1], [1.1, 0.99, 60.0, 20.0]),
        maxfev=10000,
    )
    print(f"alpha_star={popt[2]:.2f}  delta={popt[3]:.2f}  "
          f"IoU_max={popt[0]:.3f}  IoU_min={popt[1]:.3f}")

    a_dense = np.linspace(float(adam["alpha"].min()), float(adam["alpha"].max()), 300)
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        adam["alpha"], adam["mean"],
        yerr=adam["std"] / np.sqrt(adam["count"]),
        fmt="o", capsize=4, color="#1f77b4", label="Empirical (Adam)",
    )
    plt.plot(
        a_dense, sigmoid_model(a_dense, *popt),
        color="#d62728", lw=2.2,
        label=f"Fit: a*={popt[2]:.2f}, D={popt[3]:.2f}",
    )
    plt.axvline(popt[2], color="gray", linestyle=":", lw=1.5)
    plt.xlabel("Sigmoid stiffness alpha")
    plt.ylabel("Mean reconstruction IoU")
    plt.title(f"Phase transition: alpha*={popt[2]:.2f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig_path = os.path.join(exp.FIG_DIR, "phase_transition_fit.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(f"[fig] saved phase_transition_fit.png")

    csv_path = os.path.join(exp.CSV_DIR, "phase_transition_fit.csv")
    pd.DataFrame([{
        "alpha_star": popt[2], "delta": popt[3],
        "iou_max": popt[0], "iou_min": popt[1],
    }]).to_csv(csv_path, index=False)
    print(f"[csv] saved phase_transition_fit.csv")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

print("oracle_ablation.png :",
      os.path.exists(os.path.join(exp.FIG_DIR, "oracle_ablation.png")))
print("phase_transition_fit.png:",
      os.path.exists(os.path.join(exp.FIG_DIR, "phase_transition_fit.png")))

er = pd.read_csv(os.path.join(exp.CSV_DIR, "effective_rank_vs_alpha.csv"))
grp = er.groupby("alpha")["effective_rank"].mean()
print("Effective rank:")
print(grp.round(1).to_string())
print("Saturated (all >= 38):", all(grp >= 38))

print("\nOracle ablation at alpha=10:")
at10 = oracle_df[oracle_df["alpha"].sub(10).abs() < 1e-9]
at10_grp = at10.groupby("optimizer")["output_iou_final"].mean()
print(at10_grp.round(3))
oracle_v = float(at10_grp.get("oracle_pgd", 0))
pgd_v    = float(at10_grp.get("pgd",        0))
adam_v   = float(at10_grp.get("adam",       0))
print(f"Oracle < PGD  : {oracle_v:.3f} < {pgd_v:.3f} = {oracle_v < pgd_v}")
print(f"Adam > both   : {adam_v:.3f} > max({oracle_v:.3f},{pgd_v:.3f}) = {adam_v > max(oracle_v, pgd_v)}")
