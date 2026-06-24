"""Phase 2/3 (generalized architecture support + automatic gate analysis),
runnable. Profiles per-layer gate density, gradient-flow, and rank-collapse
metrics on REAL torchvision architectures (resnet18, resnet50, vgg11,
vit_b_16, convnext_tiny) exactly as shipped — no AlphaSigmoid replacement,
generalizing run_experiments.py's Phase 3A beyond "ResNet/VGG-shaped sigmoid
networks" to the canonical architectures plus a transformer (ViT) and a
modern conv net (ConvNeXt).

Uses random (not real-data) inputs at a fixed seed per run, matching
run_deepnet_gate_sweep's existing convention (random CIFAR-shaped inputs) —
this profiles the architecture+activation landscape itself, not a
data-dependent training trajectory (see gradient_gate/run_training_dynamics.py in the
roadmap for the latter).
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.architectures import SUPPORTED, build_model, input_size_for
from gradient_gate.instrumentation import GateInstrumentor

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs", "csv")


def profile_one(name: str, seed: int, device: str, batch: int = 8):
    torch.manual_seed(seed)
    size = input_size_for(name)
    model = build_model(name).to(device).train()
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(0, 10, (batch,), device=device)

    instr = GateInstrumentor(model)
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    stats = instr.collect()
    instr.remove()

    n = len(stats)
    return [
        dict(arch=name, seed=seed, layer_idx=s.order, n_layers=n, layer_name=s.name,
             active_frac=s.active_frac, gate_mean=s.gate_mean, grad_norm=s.grad_norm,
             grad_sparsity=s.grad_sparsity, grad_entropy=s.grad_entropy,
             effective_rank=s.effective_rank, stable_rank=s.stable_rank)
        for s in stats
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(SUPPORTED))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "real_arch_gate_profile.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_rows = []
    for name in args.archs:
        for seed in args.seeds:
            print(f"[gate] profiling {name} seed={seed} on {device}")
            rows = profile_one(name, seed, device)
            all_rows.extend(rows)
            print(f"  {len(rows)} elementwise-activation layers instrumented")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\n[gate] wrote {len(df)} rows -> {args.out}")

    summary = df.groupby("arch")[["active_frac", "effective_rank", "stable_rank"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
