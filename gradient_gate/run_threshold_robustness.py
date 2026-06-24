"""Threshold-robustness and gate-magnitude-distribution analysis for the
activation-class claim (Contribution 2). The training-dynamics and
activation-ablation campaigns both logged active_frac at a single fixed
threshold (0.01), averaged per-layer, with no record of the raw gate
distribution. That is not enough to answer "is the direction an artifact of
the threshold, or of smooth activations merely sitting near a ceiling from
the start?" This script logs, every epoch, on the same fixed instrumentation
batch used throughout this project:

  - active_frac at 5 thresholds (0.001, 0.005, 0.01, 0.05, 0.10)
  - gate_mean, gate_median, gate_std (threshold-free, continuous summaries)
  - gate quantiles (1/5/25/50/75/95/99%)

computed from the POOLED raw gate magnitudes across every instrumented
layer (each layer's tensor randomly subsampled to a cap before pooling, to
bound memory/compute -- this gives an unbiased estimate of the network-wide
gate-magnitude distribution, not just a per-layer-averaged scalar).

Reuses the exact training protocol, dataloaders, and model builder from
run_training_dynamics.py -- only the instrumentation step differs.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.instrumentation import ELEMENTWISE_ACT_TYPES, GRAD_RATIO_EPS
from gradient_gate.run_training_dynamics import (CSV_DIR, DATA_ROOT, build_model_for,
                                                  evaluate, get_dataloaders)

THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.10)
QUANTILES = (1, 5, 25, 50, 75, 95, 99)
PER_LAYER_CAP = 50_000  # subsample cap per layer before pooling, bounds memory/compute


class PooledGateCollector:
    """Backward-hooks every elementwise activation and pools a random
    subsample of each layer's |grad_input/grad_output| gate values into one
    flat array per forward+backward pass -- the same recovery trick as
    GateInstrumentor, but keeping raw values (subsampled) instead of
    collapsing immediately to a single per-layer scalar."""

    def __init__(self, model, act_types=ELEMENTWISE_ACT_TYPES, cap=PER_LAYER_CAP):
        self.cap = cap
        self._chunks = []
        self._handles = []
        for _, module in model.named_modules():
            if isinstance(module, act_types):
                self._handles.append(module.register_full_backward_hook(self._hook))

    def _hook(self, module, grad_input, grad_output):
        if not grad_input or grad_input[0] is None or not grad_output or grad_output[0] is None:
            return
        gi, go = grad_input[0].detach(), grad_output[0].detach()
        valid = go.abs() > GRAD_RATIO_EPS
        if not valid.any():
            return
        gate = (gi[valid].abs() / go[valid].abs()).clamp(max=1e6).flatten()
        if gate.numel() > self.cap:
            idx = torch.randperm(gate.numel(), device=gate.device)[: self.cap]
            gate = gate[idx]
        self._chunks.append(gate.cpu())

    def pooled(self) -> np.ndarray:
        if not self._chunks:
            return np.array([])
        return torch.cat(self._chunks).numpy()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def instrument_gate_distribution(model, x, y, device):
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
        result = {f"active_frac_t{t}": float("nan") for t in THRESHOLDS}
        result.update(gate_mean=float("nan"), gate_median=float("nan"), gate_std=float("nan"),
                      n_gate_values=0)
        for q in QUANTILES:
            result[f"gate_q{q}"] = float("nan")
        return result

    result = {f"active_frac_t{t}": float((gate_vals > t).mean()) for t in THRESHOLDS}
    result["gate_mean"] = float(gate_vals.mean())
    result["gate_median"] = float(np.median(gate_vals))
    result["gate_std"] = float(gate_vals.std())
    result["n_gate_values"] = int(gate_vals.size)
    for q in QUANTILES:
        result[f"gate_q{q}"] = float(np.percentile(gate_vals, q))
    return result


def already_done(out_path, arch, activation, dataset, seed, final_epoch):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path)
    mask = ((df.arch == arch) & (df.activation == activation) & (df.dataset == dataset)
            & (df.seed == seed) & (df.epoch == final_epoch))
    return len(df[mask]) > 0


def run_one(arch, activation, dataset, seed, epochs, batch_size, lr, data_root, out_path, device,
            num_workers):
    num_classes = 10 if dataset == "cifar10" else 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_dataloaders(dataset, arch, batch_size, data_root, num_workers)
    model = build_model_for(arch, num_classes, activation=activation).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    instr_x, instr_y = next(iter(test_loader))
    instr_x, instr_y = instr_x[:64], instr_y[:64]

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        gate_dist = instrument_gate_distribution(model, instr_x, instr_y, device)
        row = dict(arch=arch, activation=activation, dataset=dataset, seed=seed, epoch=epoch,
                   test_loss=test_loss, test_acc=test_acc, epoch_time=time.time() - t0, **gate_dist)
        print(f"[thresh-rob] {arch}({activation}) seed={seed} epoch={epoch:3d}/{epochs} "
              f"test_acc={test_acc:.3f}  af@0.01={gate_dist['active_frac_t0.01']:.3f}  "
              f"gate_mean={gate_dist['gate_mean']:.4f} gate_median={gate_dist['gate_median']:.4f} "
              f"({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["resnet18"])
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "silu", "mish"])
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "threshold_robustness.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[thresh-rob] device={device} archs={args.archs} activations={args.activations} "
          f"datasets={args.datasets} seeds={args.seeds} epochs={args.epochs}")

    for dataset in args.datasets:
        for arch in args.archs:
            for activation in args.activations:
                for seed in args.seeds:
                    if already_done(args.out, arch, activation, dataset, seed, args.epochs - 1):
                        print(f"[thresh-rob] [skip] {arch}({activation})/{dataset} seed={seed} done")
                        continue
                    t0 = time.time()
                    run_one(arch, activation, dataset, seed, args.epochs, args.batch_size, args.lr,
                            args.data_root, args.out, device, args.num_workers)
                    print(f"[thresh-rob] {arch}({activation})/{dataset} seed={seed} done in "
                          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
