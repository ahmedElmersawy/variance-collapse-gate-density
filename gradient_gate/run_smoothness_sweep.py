"""Smoothness-mechanism sweep: tests whether activation-derivative
smoothness, not ReLU-vs-architecture or hard-zero-vs-not, is the governing
factor behind the gate-density direction split (ReLU declines, GELU/SiLU/
Mish rise).

13 activation conditions spanning hard-gated (ReLU) through near-ReLU
(LeakyReLU at increasing slope), learnable (PReLU), a continuous-stiffness
family (Softplus(beta)), to fully smooth (GELU/SiLU/Mish). NOTE: Softplus's
derivative is EXACTLY sigmoid(beta*x) -- this is not just "another smooth
activation," it is a real-network realization of this project's original
synthetic gate formalism Gamma=sigmoid(alpha*(z-c)), with beta playing the
role of alpha. The Softplus-beta family is the most direct test available
of whether the synthetic theory's gate parameter and this project's
real-network activation-class finding are the same underlying object.

Same ResNet-18/CIFAR-10 protocol used throughout (SGD+momentum+cosine,
batch 128, lr 0.1, 25 epochs), only the activation varies.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.cifar_models import cifar_resnet18
from gradient_gate.run_threshold_robustness import PooledGateCollector
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, evaluate, get_dataloaders

QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
THRESHOLDS = (0.01, 0.10)

ACTIVATIONS = {
    "relu":              ("A_hard_gated",   lambda: nn.ReLU()),
    "leaky_relu_0.001":  ("B_near_relu",    lambda: nn.LeakyReLU(negative_slope=0.001)),
    "leaky_relu_0.01":   ("B_near_relu",    lambda: nn.LeakyReLU(negative_slope=0.01)),
    "leaky_relu_0.05":   ("B_near_relu",    lambda: nn.LeakyReLU(negative_slope=0.05)),
    "leaky_relu_0.10":   ("B_near_relu",    lambda: nn.LeakyReLU(negative_slope=0.10)),
    "prelu":             ("C_learnable",    lambda: nn.PReLU()),
    "softplus_beta50":   ("D_softplus",     lambda: nn.Softplus(beta=50)),
    "softplus_beta20":   ("D_softplus",     lambda: nn.Softplus(beta=20)),
    "softplus_beta10":   ("D_softplus",     lambda: nn.Softplus(beta=10)),
    "softplus_beta5":    ("D_softplus",     lambda: nn.Softplus(beta=5)),
    "gelu":              ("E_fully_smooth", lambda: nn.GELU()),
    "silu":              ("E_fully_smooth", lambda: nn.SiLU()),
    "mish":              ("E_fully_smooth", lambda: nn.Mish()),
}


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
        result = {f"active_frac_t{t}": float("nan") for t in THRESHOLDS}
        result.update(gate_mean=float("nan"), gate_std=float("nan"), gate_var=float("nan"))
        for q in QUANTILES:
            result[f"gate_q{q:02d}"] = float("nan")
        return result

    result = {f"active_frac_t{t}": float((gate_vals > t).mean()) for t in THRESHOLDS}
    result["gate_mean"] = float(gate_vals.mean())
    result["gate_std"] = float(gate_vals.std())
    result["gate_var"] = float(gate_vals.var())
    for q in QUANTILES:
        result[f"gate_q{q:02d}"] = float(np.percentile(gate_vals, q))
    return result


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
        train_correct = train_total = 0
        train_loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            train_correct += (out.detach().argmax(1) == y).sum().item()
            train_total += y.size(0)
            train_loss_sum += loss.item() * y.size(0)
        sched.step()
        train_acc = train_correct / train_total
        train_loss = train_loss_sum / train_total

        test_loss, test_acc = evaluate(model, test_loader, device)
        gate_stats = instrument(model, instr_x, instr_y, device)
        row = dict(activation=activation, group=group, seed=seed, epoch=epoch,
                   train_acc=train_acc, train_loss=train_loss, test_acc=test_acc, test_loss=test_loss,
                   epoch_time=time.time() - t0, **gate_stats)
        print(f"[smooth-sweep] {activation} seed={seed} epoch={epoch:3d}/{epochs} "
              f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
              f"af@0.01={gate_stats['active_frac_t0.01']:.3f} gate_mean={gate_stats['gate_mean']:.4f} "
              f"gate_var={gate_stats['gate_var']:.5f} ({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=list(ACTIVATIONS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "smoothness_sweep_results.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[smooth-sweep] device={device} activations={args.activations} seeds={args.seeds} "
          f"epochs={args.epochs}")

    for activation in args.activations:
        for seed in args.seeds:
            if already_done(args.out, activation, seed, args.epochs - 1):
                print(f"[smooth-sweep] [skip] {activation} seed={seed} already complete")
                continue
            t0 = time.time()
            run_one(activation, seed, args.epochs, args.batch_size, args.lr, args.data_root, args.out,
                    device, args.num_workers)
            print(f"[smooth-sweep] {activation} seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
