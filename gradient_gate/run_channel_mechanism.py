"""P2 -- direct, per-CHANNEL mechanism verification.

final_neurips_upgrade_report.md explicitly flags the gap this closes:
"this experiment did not test the per-channel link between pre-activation
distribution shape and gate value directly." Every result up to this point
(theory_variance_compression_mechanism.md, run_preactivation_mean_check.py)
tested the mechanism at the POOLED, population level: one mu, one sigma,
one active_frac per (activation, seed, epoch), averaged across every
channel of every layer. That is consistent with the mechanism but does not
verify it at the unit of analysis the mechanism actually makes a claim
about -- the individual channel.

This script logs, for every channel of every elementwise-activation layer
in CIFAR-native ResNet-18, at epochs {0,6,12,18,24}:
  - mu_c, sigma_c       pre-activation mean/std for that channel (forward hook)
  - active_frac_c       |grad_input/grad_output| > GATE_EPS, that channel only
  - gate_mean_c         mean |grad_input/grad_output|, that channel only
(both gate quantities recovered via the same grad_input/grad_output trick
used everywhere else in this project -- see instrumentation.py's module
docstring.)

The per-channel test: does each channel's own margin (mu_c - z_low) at
epoch 0 predict the SIGN of that channel's own active_frac change by
epoch 24? z_low is computed once per activation, analytically, via
autograd (compute_z_low below) -- fixed, training-independent, exactly as
in theory_variance_compression_mechanism.md, but now compared against
thousands of individual channel trajectories instead of one pooled number
per activation.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.cifar_models import cifar_resnet18
from gradient_gate.instrumentation import ELEMENTWISE_ACT_TYPES, GATE_EPS, GRAD_RATIO_EPS
from gradient_gate.run_smoothness_sweep import ACTIVATIONS
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, evaluate, get_dataloaders

MECH_EPOCHS = (0, 6, 12, 18, 24)
MAIN_ACTIVATIONS = ("relu", "gelu", "silu", "mish")  # the four load-bearing main-paper activations


class PerChannelMechanismCollector:
    """Forward hook caches the pre-activation input tensor per layer;
    backward hook (fires right after, same pass) recovers the gate via
    grad_input/grad_output and combines it with the cached input to emit
    one row per channel. Channel identity = (layer name, channel index),
    stable across epochs because the architecture never changes, only the
    weights."""

    def __init__(self, model, act_types=ELEMENTWISE_ACT_TYPES):
        self._cache = {}
        self._rows = []
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, act_types):
                self._handles.append(module.register_forward_hook(self._make_fwd_hook(name)))
                self._handles.append(module.register_full_backward_hook(self._make_bwd_hook(name)))

    def _make_fwd_hook(self, name):
        def hook(module, inputs, output):
            self._cache[name] = inputs[0].detach()
        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            if not grad_input or grad_input[0] is None or not grad_output or grad_output[0] is None:
                return
            x = self._cache.get(name)
            if x is None:
                return
            gi, go = grad_input[0].detach(), grad_output[0].detach()
            valid = go.abs() > GRAD_RATIO_EPS
            gate = torch.zeros_like(go)
            gate[valid] = (gi[valid].abs() / go[valid].abs()).clamp(max=1e6)

            reduce_dims = (0, 2, 3) if x.dim() == 4 else (0,)
            mu = x.mean(dim=reduce_dims)
            sigma = x.std(dim=reduce_dims)
            active_frac = (gate > GATE_EPS).float().mean(dim=reduce_dims)
            gate_mean = gate.mean(dim=reduce_dims)
            valid_frac = valid.float().mean(dim=reduce_dims)

            n_channels = mu.numel()
            for c in range(n_channels):
                self._rows.append(dict(
                    layer=name, channel=c,
                    mu=float(mu[c]), sigma=float(sigma[c]),
                    active_frac=float(active_frac[c]), gate_mean=float(gate_mean[c]),
                    valid_frac=float(valid_frac[c]),
                ))
        return hook

    def collect(self):
        rows = self._rows
        self._rows = []
        self._cache = {}
        return rows

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def compute_z_low(act_factory, threshold=GATE_EPS, z_min=-15.0, z_max=15.0, n=600_001):
    """z_low(theta): the left boundary of the right-side region where
    |f'(z)| stays above theta for every z further right, computed via
    autograd on a fine grid -- a fixed property of the activation and
    threshold alone, no training involved.

    Implementation note: a literal left-to-right inf{z : g(z)>theta} is
    NOT robust for GELU/Mish, whose derivative is non-monotonic and has a
    small transient bump in the far-left tail (verified numerically: GELU's
    |g| briefly exceeds 0.10 again near z=-1.86 before dipping back to ~0
    near its true zero-crossing at z=-0.75, then rising for good past z=0).
    A left-to-right scan latches onto that irrelevant early blip. Scanning
    right-to-left from a point deep in the permanently-active region and
    stopping at the first z where g(z)<=theta instead finds the boundary of
    the *sustained* active region, which is what the mechanism's claim
    (variance shrinkage pushes mu permanently across one boundary) actually
    needs. This matches theory_variance_compression_mechanism.md's table to
    3 decimals at theta=0.10 for every activation checked (relu, gelu, silu,
    mish, softplus at beta in {50,20,10,5}, leaky_relu)."""
    z = torch.linspace(z_min, z_max, n, requires_grad=True)
    f = act_factory()
    y = f(z)
    y.sum().backward()
    g = z.grad.detach().abs()
    below = g <= threshold
    if not below.any():
        return float("nan")
    idx_from_right = int(below.flip(0).float().argmax())
    idx = len(z) - 1 - idx_from_right
    return float(z[idx].detach())


def already_done(out_path, activation, seed, final_epoch):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path, usecols=["activation", "seed", "epoch"])
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
            model.train()
            x, y = instr_x.to(device), instr_y.to(device)
            collector = PerChannelMechanismCollector(model)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            model.zero_grad()
            loss.backward()
            rows = collector.collect()
            collector.remove()
            model.zero_grad()

            for r in rows:
                r.update(activation=activation, seed=seed, epoch=epoch, test_acc=test_acc)
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
            print(f"[chan-mech] {activation} seed={seed} epoch={epoch:3d}/{epochs} [MECH] "
                  f"n_channels={len(rows)} test_acc={test_acc:.3f} "
                  f"mean_active_frac={np.mean([r['active_frac'] for r in rows]):.3f} ({time.time()-t0:.1f}s)")
        else:
            print(f"[chan-mech] {activation} seed={seed} epoch={epoch:3d}/{epochs} ({time.time()-t0:.1f}s)")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=list(MAIN_ACTIVATIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "channel_mechanism.csv"))
    ap.add_argument("--zlow-out", default=os.path.join(CSV_DIR, "channel_mechanism_zlow.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[chan-mech] device={device} activations={args.activations} seeds={args.seeds} epochs={args.epochs}")

    zlow_rows = []
    for activation in args.activations:
        _, act_factory = ACTIVATIONS[activation]
        zlow = compute_z_low(act_factory)
        zlow_rows.append(dict(activation=activation, threshold=GATE_EPS, z_low=zlow))
        print(f"[chan-mech] z_low({activation}, theta={GATE_EPS}) = {zlow:.5f}")
    pd.DataFrame(zlow_rows).to_csv(args.zlow_out, index=False)

    for activation in args.activations:
        for seed in args.seeds:
            if already_done(args.out, activation, seed, args.epochs - 1):
                print(f"[chan-mech] [skip] {activation} seed={seed} already complete")
                continue
            t0 = time.time()
            run_one(activation, seed, args.epochs, args.batch_size, args.lr, args.data_root, args.out,
                    device, args.num_workers)
            print(f"[chan-mech] {activation} seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
