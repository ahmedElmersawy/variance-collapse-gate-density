"""Three new figures for the optimizer/per-channel/scale-and-architecture
upgrade results (P1-P4). Run from the project root.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "gradient_gate_outputs", "csv")
FIG = os.path.join(ROOT, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACT_ORDER = ["relu", "gelu", "silu", "mish"]
ACT_COLOR = {"relu": "C0", "gelu": "C1", "silu": "C2", "mish": "C3"}


def fig_optimizer_generalization():
    sgd_relu = pd.read_csv(os.path.join(CSV, "training_dynamics.csv"))
    sgd_relu = sgd_relu[sgd_relu.arch.isin(["resnet18", "vgg11"])].copy()
    sgd_relu["activation"] = "relu"
    sgd_smooth = pd.read_csv(os.path.join(CSV, "activation_ablation.csv"))
    sgd = pd.concat([sgd_relu[["arch", "activation", "dataset", "seed", "epoch", "mean_active_frac"]],
                      sgd_smooth[["arch", "activation", "dataset", "seed", "epoch", "mean_active_frac"]]],
                     ignore_index=True)
    sgd["optimizer"] = "sgd"
    adam = pd.read_csv(os.path.join(CSV, "optimizer_ablation_adam.csv"))
    adam["optimizer"] = "adam"
    adamw = pd.read_csv(os.path.join(CSV, "optimizer_ablation_adamw.csv"))
    adamw["optimizer"] = "adamw"
    df = pd.concat([sgd, adam[["arch", "activation", "dataset", "seed", "epoch", "mean_active_frac", "optimizer"]],
                     adamw[["arch", "activation", "dataset", "seed", "epoch", "mean_active_frac", "optimizer"]]],
                    ignore_index=True)

    rows = []
    for (opt, act, arch, ds, seed), g in df.groupby(["optimizer", "activation", "arch", "dataset", "seed"]):
        g = g.sort_values("epoch")
        rows.append(dict(optimizer=opt, activation=act,
                          delta=float(g.mean_active_frac.iloc[-1] - g.mean_active_frac.iloc[0])))
    sl = pd.DataFrame(rows)
    agg = sl.groupby(["optimizer", "activation"]).delta.agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    fig, ax = plt.subplots(figsize=(7, 4))
    optimizers = ["sgd", "adam", "adamw"]
    width = 0.2
    x = np.arange(len(ACT_ORDER))
    for i, opt in enumerate(optimizers):
        means, ses = [], []
        for act in ACT_ORDER:
            row = agg[(agg.optimizer == opt) & (agg.activation == act)]
            means.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
            ses.append(float(row["se"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 1) * width, means, width, yerr=ses, capsize=3, label=opt.upper())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ACT_ORDER])
    ax.set_ylabel(r"$\Delta$ active_frac (epoch 24 $-$ epoch 0)")
    ax.set_title("Optimizer generalization: SGD/Adam match, AdamW collapses the split")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG, "optimizer_generalization.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def fig_channel_mechanism():
    df = pd.read_csv(os.path.join(CSV, "channel_mechanism.csv"))
    zlow = pd.read_csv(os.path.join(CSV, "channel_mechanism_zlow.csv")).set_index("activation")["z_low"]
    df["channel_id"] = df["layer"] + "::" + df["channel"].astype(str)
    df["zscore_margin"] = df.apply(lambda r: (r.mu - zlow[r.activation]) / r.sigma if r.sigma > 0 else np.nan, axis=1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, act in zip(axes, ACT_ORDER):
        corrs = []
        for seed in [0, 1, 2]:
            g = df[(df.activation == act) & (df.seed == seed)]
            pivot_z = g.pivot(index="channel_id", columns="epoch", values="zscore_margin")
            pivot_a = g.pivot(index="channel_id", columns="epoch", values="active_frac")
            for ch in pivot_z.index:
                zz, aa = pivot_z.loc[ch], pivot_a.loc[ch]
                if zz.std() > 1e-9 and aa.std() > 1e-9:
                    corrs.append(np.corrcoef(zz.values, aa.values)[0, 1])
        ax.hist(corrs, bins=40, color=ACT_COLOR[act], alpha=0.85)
        frac_pos = np.mean([c > 0 for c in corrs])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{act.upper()}\n{frac_pos:.0%} channels $\\rho>0$")
        ax.set_xlabel(r"per-channel $\rho$" + "((mu-z_low)/sigma, active_frac)")
    axes[0].set_ylabel("channel count")
    fig.suptitle("Per-channel mechanism verification (sigma-normalized margin), pooled across 3 seeds")
    fig.tight_layout()
    out = os.path.join(FIG, "channel_mechanism_zscore.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def fig_scale_and_architecture():
    tin = pd.concat([pd.read_csv(os.path.join(CSV, "tinyimagenet_dynamics_a.csv")),
                      pd.read_csv(os.path.join(CSV, "tinyimagenet_dynamics_b.csv"))], ignore_index=True)
    seq = pd.read_csv(os.path.join(CSV, "sequence_model_ablation.csv"))

    def deltas(df, group_cols):
        rows = []
        for key, g in df.groupby(group_cols + ["seed"]):
            g = g.sort_values("epoch")
            row = dict(zip(group_cols + ["seed"], key if isinstance(key, tuple) else (key,)))
            row["delta"] = float(g.mean_active_frac.iloc[-1] - g.mean_active_frac.iloc[0])
            rows.append(row)
        return pd.DataFrame(rows)

    tin_d = deltas(tin, ["activation"])
    seq_d = deltas(seq, ["arch", "activation"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    agg = tin_d.groupby("activation").delta.agg(["mean", "std", "count"])
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    x = np.arange(len(ACT_ORDER))
    means = [agg.loc[a, "mean"] if a in agg.index else np.nan for a in ACT_ORDER]
    ses = [agg.loc[a, "se"] if a in agg.index else np.nan for a in ACT_ORDER]
    ax.bar(x, means, yerr=ses, capsize=3, color=[ACT_COLOR[a] for a in ACT_ORDER])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in ACT_ORDER])
    ax.set_title("Tiny-ImageNet-200\n(ResNet-18, SGD)")
    ax.set_ylabel(r"$\Delta$ active_frac")

    for ax, arch in zip(axes[1:], ["mlp_mixer", "transformer_encoder"]):
        sub = seq_d[seq_d.arch == arch]
        agg = sub.groupby("activation").delta.agg(["mean", "std", "count"])
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        means = [agg.loc[a, "mean"] if a in agg.index else np.nan for a in ACT_ORDER]
        ses = [agg.loc[a, "se"] if a in agg.index else np.nan for a in ACT_ORDER]
        ax.bar(x, means, yerr=ses, capsize=3, color=[ACT_COLOR[a] for a in ACT_ORDER])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in ACT_ORDER])
        ax.set_title(("MLP-Mixer" if arch == "mlp_mixer" else "Transformer-Encoder") + "\n(LayerNorm, SGD)")

    fig.suptitle(r"$\Delta$ active_frac (epoch end $-$ start), pooled across datasets/seeds")
    fig.tight_layout()
    out = os.path.join(FIG, "generalization_scale_architecture.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def fig_adamw_prediction():
    sgd = pd.read_csv(os.path.join(CSV, "channel_mechanism.csv"))
    adamw = pd.read_csv(os.path.join(CSV, "channel_mechanism_adamw.csv"))
    zlow_sgd = pd.read_csv(os.path.join(CSV, "channel_mechanism_zlow.csv")).set_index("activation")["z_low"]
    zlow_adamw = pd.read_csv(os.path.join(CSV, "channel_mechanism_adamw_zlow.csv")).set_index("activation")["z_low"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for name, df, zlow, ls in [("SGD", sgd, zlow_sgd, "-"), ("AdamW", adamw, zlow_adamw, "--")]:
        df = df.copy()
        df["zscore"] = df.apply(lambda r: (r.mu - zlow[r.activation]) / r.sigma if r.sigma > 0 else np.nan, axis=1)
        sigma_agg = df.groupby(["activation", "epoch"]).sigma.mean().reset_index()
        margin_agg = df.groupby(["activation", "epoch"]).zscore.mean().reset_index()
        for act in ACT_ORDER:
            s = sigma_agg[sigma_agg.activation == act].sort_values("epoch")
            m = margin_agg[margin_agg.activation == act].sort_values("epoch")
            axes[0, 0 if name == "SGD" else 1].plot(s.epoch, s.sigma, ls, color=ACT_COLOR[act], label=act.upper())
            axes[1, 0 if name == "SGD" else 1].plot(m.epoch, m.zscore, ls, color=ACT_COLOR[act], label=act.upper())

    axes[0, 0].set_title("SGD: sigma"); axes[0, 1].set_title("AdamW: sigma")
    axes[1, 0].set_title("SGD: z-score margin"); axes[1, 1].set_title("AdamW: z-score margin")
    for ax in axes[0]: ax.set_ylabel("mean per-channel sigma")
    for ax in axes[1]: ax.set_ylabel("mean z-score margin"); ax.set_xlabel("epoch")
    axes[0, 0].axhline(0, color="gray", lw=0.5); axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[1, 0].axhline(0, color="black", lw=0.8); axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Why the predictor gets AdamW right: sigma collapses under SGD, not under AdamW")
    fig.tight_layout()
    out = os.path.join(FIG, "adamw_prediction.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    fig_optimizer_generalization()
    fig_channel_mechanism()
    fig_scale_and_architecture()
    fig_adamw_prediction()
