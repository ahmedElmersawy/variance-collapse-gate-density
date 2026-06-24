"""Validates the central conjecture in theory_variance_compression_mechanism.md:
does the pre-activation mean (mu, signed, not just |beta|) drift by a
similar amount across activations during training, so that the
activation-class direction split is explained by where each activation's
fixed z_low(theta) sits relative to a shared mu-drift -- rather than by a
qualitatively different mu-trajectory for each activation (which would make
the z_low argument circular)?

For a representative subset spanning the empirical sign transition
(relu, leaky_relu, softplus at beta=50/20/10/5, gelu, silu, mish), logs,
at epochs {0,6,12,18,24}, the per-channel pre-activation mean and std
(forward hooks only, no backward pass needed), pooled across all
instrumented layers' channels.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.cifar_models import cifar_resnet18
from gradient_gate.instrumentation import ELEMENTWISE_ACT_TYPES
from gradient_gate.run_smoothness_sweep import ACTIVATIONS
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, evaluate, get_dataloaders

MECH_EPOCHS = (0, 6, 12, 18, 24)
CONDITIONS = ["relu", "leaky_relu_0.01", "softplus_beta50", "softplus_beta20",
              "softplus_beta10", "softplus_beta5", "gelu", "silu", "mish"]


class PerChannelPreActStats:
    """Forward-hook only (no backward pass needed): per-channel mean/std of
    the INPUT to each elementwise activation layer, pooled across channels
    from every instrumented layer into one flat array per quantity."""

    def __init__(self, model, act_types=ELEMENTWISE_ACT_TYPES):
        self.means, self.stds = [], []
        self.handles = []
        for _, module in model.named_modules():
            if isinstance(module, act_types):
                self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        x = inp[0].detach()
        if x.dim() == 4:
            m = x.mean(dim=(0, 2, 3))
            s = x.std(dim=(0, 2, 3))
        else:
            m = x.mean(dim=0)
            s = x.std(dim=0)
        self.means.append(m.cpu())
        self.stds.append(s.cpu())

    def pooled(self):
        means = torch.cat(self.means).numpy() if self.means else np.array([])
        stds = torch.cat(self.stds).numpy() if self.stds else np.array([])
        return means, stds

    def remove(self):
        for h in self.handles:
            h.remove()


def instrument(model, x, y, device):
    model.train()
    x = x.to(device)
    collector = PerChannelPreActStats(model)
    with torch.no_grad():
        model(x)
    means, stds = collector.pooled()
    collector.remove()
    if means.size == 0:
        return dict(mean_mu=float("nan"), median_mu=float("nan"), mean_sigma=float("nan"),
                    frac_mu_negative=float("nan"), mean_standardized_mu=float("nan"))
    standardized = means / (stds + 1e-8)
    return dict(
        mean_mu=float(means.mean()), median_mu=float(np.median(means)),
        mean_sigma=float(stds.mean()), frac_mu_negative=float((means < 0).mean()),
        mean_standardized_mu=float(standardized.mean()),
    )


def already_done(out_path, activation, seed, final_epoch):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path)
    mask = (df.activation == activation) & (df.seed == seed) & (df.epoch == final_epoch)
    return len(df[mask]) > 0


def run_one(activation, seed, epochs, batch_size, lr, data_root, out_path, device, num_workers):
    group, act_factory = ACTIVATIONS[activation]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_dataloaders("cifar10", "resnet18", batch_size, data_root, num_workers)
    model = cifar_resnet18(num_classes=10, act_layer=act_factory).to(device)

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

        if epoch in MECH_EPOCHS:
            test_loss, test_acc = evaluate(model, test_loader, device)
            stats = instrument(model, instr_x, instr_y, device)
            row = dict(activation=activation, seed=seed, epoch=epoch, test_acc=test_acc, **stats)
            print(f"[preact-mean] {activation} seed={seed} epoch={epoch:3d}/{epochs} "
                  f"test_acc={test_acc:.3f} mean_mu={stats['mean_mu']:+.5f} "
                  f"mean_sigma={stats['mean_sigma']:.4f} std_mu={stats['mean_standardized_mu']:+.5f} "
                  f"({time.time()-t0:.1f}s)")
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        else:
            print(f"[preact-mean] {activation} seed={seed} epoch={epoch:3d}/{epochs} ({time.time()-t0:.1f}s)")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=CONDITIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "preactivation_mean_check.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[preact-mean] device={device} activations={args.activations} seeds={args.seeds} epochs={args.epochs}")

    for activation in args.activations:
        for seed in args.seeds:
            if already_done(args.out, activation, seed, args.epochs - 1):
                print(f"[preact-mean] [skip] {activation} seed={seed} already complete")
                continue
            t0 = time.time()
            run_one(activation, seed, args.epochs, args.batch_size, args.lr, args.data_root, args.out,
                    device, args.num_workers)
            print(f"[preact-mean] {activation} seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
