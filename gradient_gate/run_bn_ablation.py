"""Experiment 1: BatchNorm Necessity Ablation.

Objective: determine whether BatchNorm is responsible for the
activation-class-dependent gate-density direction split (ReLU declining,
GELU/SiLU/Mish rising), or whether it survives when BatchNorm is replaced
by GroupNorm on the exact same ResNet-18 skeleton.

Reuses the CIFAR dataloaders/eval loop from run_training_dynamics.py and the
pooled raw-gate-distribution instrumentation from run_threshold_robustness.py
(same recovery trick: grad_input/grad_output ratio, pooled across all
elementwise-activation layers on the fixed held-out batch, capped per layer
to bound memory).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.cifar_models import build_cifar_model
from gradient_gate.run_threshold_robustness import PooledGateCollector, QUANTILES as _UNUSED
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, evaluate, get_dataloaders

THRESHOLD = 0.01
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def instrument(model, x, y, device):
    model.train()
    x, y = x.to(device), y.to(device)
    collector = PooledGateCollector(model)
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    model.zero_grad()
    loss.backward()
    gate_vals = collector.pooled()
    collector.remove()
    model.zero_grad()

    if gate_vals.size == 0:
        result = dict(active_frac=float("nan"), gate_mean=float("nan"),
                      gate_median=float("nan"), gate_std=float("nan"))
        for q in QUANTILES:
            result[f"gate_q{q:02d}"] = float("nan")
        return result

    result = dict(active_frac=float((gate_vals > THRESHOLD).mean()),
                   gate_mean=float(gate_vals.mean()), gate_median=float(np.median(gate_vals)),
                   gate_std=float(gate_vals.std()))
    for q in QUANTILES:
        result[f"gate_q{q:02d}"] = float(np.percentile(gate_vals, q))
    return result


def already_done(out_path, norm, activation, seed, final_epoch):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path)
    mask = (df.norm == norm) & (df.activation == activation) & (df.seed == seed) & (df.epoch == final_epoch)
    return len(df[mask]) > 0


def run_one(norm, activation, seed, epochs, batch_size, lr, data_root, out_path, device, num_workers):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_dataloaders("cifar10", "resnet18", batch_size, data_root, num_workers)
    model = build_cifar_model("resnet18", num_classes=10, activation=activation, norm=norm).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    instr_x, instr_y = next(iter(test_loader))
    instr_x, instr_y = instr_x[:64], instr_y[:64]

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_correct = train_total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            train_correct += (out.detach().argmax(1) == y).sum().item()
            train_total += y.size(0)
        sched.step()
        train_acc = train_correct / train_total

        test_loss, test_acc = evaluate(model, test_loader, device)
        gate_stats = instrument(model, instr_x, instr_y, device)
        row = dict(norm=norm, activation=activation, seed=seed, epoch=epoch,
                   train_acc=train_acc, test_acc=test_acc, epoch_time=time.time() - t0, **gate_stats)
        print(f"[bn-abl] {norm}/{activation} seed={seed} epoch={epoch:3d}/{epochs} "
              f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} active_frac={gate_stats['active_frac']:.3f} "
              f"gate_mean={gate_stats['gate_mean']:.4f} ({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norms", nargs="+", default=["batchnorm", "groupnorm"])
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "silu", "mish"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "bn_vs_gn_gate_dynamics.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[bn-abl] device={device} norms={args.norms} activations={args.activations} "
          f"seeds={args.seeds} epochs={args.epochs}")

    for norm in args.norms:
        for activation in args.activations:
            for seed in args.seeds:
                if already_done(args.out, norm, activation, seed, args.epochs - 1):
                    print(f"[bn-abl] [skip] {norm}/{activation} seed={seed} already complete")
                    continue
                t0 = time.time()
                run_one(norm, activation, seed, args.epochs, args.batch_size, args.lr, args.data_root,
                        args.out, device, args.num_workers)
                print(f"[bn-abl] {norm}/{activation} seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
