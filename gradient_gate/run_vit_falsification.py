"""Falsification attempt for the vit_b_16-at-random-init gradient-vanishing
finding surfaced by run_gate_profile.py: at standard torchvision init,
gradients reaching the encoder body of a 12-layer Pre-LN ViT-B/16 underflow
to ~0 in float32 (verified by direct tensor hooks, not an instrumentation
artifact — see ROADMAP.md). That finding is a single data point (one depth,
one init scheme, one batch size); a skeptical reviewer's first question is
"is this a real depth-induced phenomenon, or an artifact of one specific
init/batch-size choice you happened to pick?"

This sweeps three axes that could each independently explain or fix it:
  1. DEPTH (1..12 layers) — does vanishing onset have a depth threshold,
     connecting this to the project's own depth-compounding theme
     (Theorem 4.3) but now measured on a real attention architecture
     instead of the synthetic alpha-sigmoid stack?
  2. INIT SCHEME — torchvision's default vs. the standard "ViT paper" init
     (truncated-normal, std=0.02) — known in practice to stabilize deep
     transformer training. If this alone fixes vanishing at depth=12, the
     finding is an init artifact, not a structural property of attention
     stacks.
  3. BATCH SIZE (at fixed depth=12) — gradient *magnitude* is batch-size
     dependent (mean over the batch), so this checks whether vanishing is
     a real per-example effect or a batch-averaging artifact.

Each run is a single forward+backward pass (no training) — cheap, exact, and
directly comparable to the original finding.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import fisher_exact, spearmanr
from torchvision.models.vision_transformer import VisionTransformer

from gradient_gate.instrumentation import GateInstrumentor

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs", "csv")
VANISH_THRESHOLD = 1e-8  # stem grad norm below this counts as "vanished" (vs. ~O(1-10) when healthy)


def build_vit(num_layers, init="default", hidden_dim=768, num_heads=12, mlp_dim=3072,
              image_size=224, patch_size=16, num_classes=10):
    model = VisionTransformer(image_size=image_size, patch_size=patch_size, num_layers=num_layers,
                               num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim,
                               num_classes=num_classes)
    if init == "vit_paper":
        # Dosovitskiy et al. 2021 / common ViT implementations: truncated-
        # normal(std=0.02) on all Linear/Conv weights and the class token,
        # zero bias — the standard fix cited for deep-ViT training stability.
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(model.class_token, std=0.02)
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


def run_one(num_layers, init, batch_size, seed, device):
    torch.manual_seed(seed)
    model = build_vit(num_layers, init=init).to(device).train()
    try:
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        instr = GateInstrumentor(model)
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        stats = instr.collect()
        instr.remove()
    except torch.cuda.OutOfMemoryError:
        # this is a SHARED login-node GPU (other users' jobs fluctuate free
        # memory); skip rather than crash the whole sweep — the depth x init
        # result does not depend on reaching large batch sizes.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return dict(num_layers=num_layers, init=init, batch_size=batch_size, seed=seed,
                    stem_grad_norm=float("nan"), mean_active_frac=float("nan"),
                    n_layers_with_signal=0, n_layers_total=0, vanished=float("nan"))

    stem_grad = float(model.conv_proj.weight.grad.norm().item())
    active_fracs = [s.active_frac for s in stats if s.active_frac is not None]
    result = dict(num_layers=num_layers, init=init, batch_size=batch_size, seed=seed,
                  stem_grad_norm=stem_grad, mean_active_frac=float(np.mean(active_fracs)) if active_fracs else 0.0,
                  n_layers_with_signal=len(active_fracs), n_layers_total=len(stats),
                  vanished=float(stem_grad < VANISH_THRESHOLD))
    del model, x, y, out, loss, instr, stats
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10, 12])
    ap.add_argument("--inits", nargs="+", default=["default", "vit_paper"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "vit_falsification.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    print("[falsify] Axis 1+2: depth x init sweep (batch_size=32 fixed)")
    for d in args.depths:
        for init in args.inits:
            for seed in range(args.seeds):
                rows.append(run_one(d, init, 32, seed, device))
            vr = np.mean([r["vanished"] for r in rows if r["num_layers"] == d and r["init"] == init])
            print(f"  depth={d:2d} init={init:10s} vanish_rate={vr:.2f}")

    print("\n[falsify] Axis 3: batch-size sweep (depth=12 fixed)")
    for init in args.inits:
        for bs in args.batch_sizes:
            for seed in range(args.seeds):
                rows.append(run_one(12, init, bs, seed, device))
            vals = [r["vanished"] for r in rows
                    if r["num_layers"] == 12 and r["init"] == init and r["batch_size"] == bs
                    and r["vanished"] is not None]
            vr = np.mean(vals) if vals else float("nan")
            n_oom = sum(1 for r in rows if r["num_layers"] == 12 and r["init"] == init
                        and r["batch_size"] == bs and r["vanished"] is None)
            print(f"  batch_size={bs:4d} init={init:10s} vanish_rate={vr:.2f}  (n_oom_skipped={n_oom})")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[falsify] wrote {len(df)} rows -> {args.out}")

    # ---- statistics ----
    print("\n[falsify] === Depth-threshold test (default init, batch_size=32) ===")
    depth_df = df[(df.init == "default") & (df.batch_size == 32)]
    vanish_by_depth = depth_df.groupby("num_layers")["vanished"].mean()
    print(vanish_by_depth)
    rho, p = spearmanr(depth_df["num_layers"], depth_df["vanished"].astype(int))
    print(f"Spearman(depth, vanished): rho={rho:+.3f}  p={p:.4g}")
    if p < 0.05 and rho > 0:
        print("VERDICT: vanishing rate significantly increases with depth — genuine")
        print("         depth-induced phenomenon, not a one-off artifact.")
    else:
        print("VERDICT: no significant depth trend detected at this n — do not claim")
        print("         a depth-induced effect beyond the single depth=12 anecdote.")

    print("\n[falsify] === Does 'proper ViT init' fix it? (depth=12, batch_size=32) ===")
    sub = df[(df.num_layers == 12) & (df.batch_size == 32)]
    for init in args.inits:
        vr = sub[sub.init == init]["vanished"].mean()
        print(f"  init={init:10s} vanish_rate={vr:.2f}  (n={len(sub[sub.init==init])})")
    if set(args.inits) >= {"default", "vit_paper"}:
        is_v = sub.vanished == 1.0
        is_ok = sub.vanished == 0.0
        table = [[int(((sub.init == "default") & is_v).sum()),
                  int(((sub.init == "default") & is_ok).sum())],
                 [int(((sub.init == "vit_paper") & is_v).sum()),
                  int(((sub.init == "vit_paper") & is_ok).sum())]]
        odds_ratio, p_fisher = fisher_exact(table)
        print(f"  Fisher exact test (default vs vit_paper vanish rate): "
              f"OR={odds_ratio:.3g}  p={p_fisher:.4g}")
        if p_fisher < 0.05:
            print("  VERDICT: init scheme significantly changes vanishing — the original")
            print("           finding is (at least partly) an init artifact, not an")
            print("           unconditional property of deep ViTs.")
        else:
            print("  VERDICT: init scheme does NOT significantly change vanishing at this")
            print("           n — the standard fix does not rescue it here; treat the")
            print("           original finding as more structural than init-specific.")

    print("\n[falsify] === Batch-size sensitivity (depth=12, default init) ===")
    bs_df = df[(df.num_layers == 12) & (df.init == "default")]
    print(bs_df.groupby("batch_size")["vanished"].mean())


if __name__ == "__main__":
    main()
