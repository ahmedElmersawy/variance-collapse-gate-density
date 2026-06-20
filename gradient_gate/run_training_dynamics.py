"""Phase 4: does Gradient Gate Collapse emerge during NORMAL supervised
training — not the synthetic alpha-sweep, but real SGD training on a real
classification dataset? Trains each architecture on CIFAR-10/CIFAR-100 and
logs gate density (active_frac), rank-collapse metrics (effective_rank,
stable_rank), and test accuracy every epoch.

CIFAR-native archs (resnet18, resnet50, vgg11; see cifar_models.py) use real
nn.ReLU with a CIFAR-appropriate stem at native 32x32 resolution. vit_b_16
and convnext_tiny keep their canonical torchvision form (real nn.GELU) with
CIFAR images resized to 224 — see cifar_models.py's module docstring for why
this split exists (a 32x32-native ViT-B/16 wouldn't be "ViT-B/16" anymore).

Checkpoint-aware: a run's rows are appended incrementally per epoch, and an
(arch, dataset, seed) triple already present up to args.epochs-1 is skipped
entirely on restart.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from gradient_gate.architectures import build_model as build_real_model
from gradient_gate.cifar_models import CIFAR_NATIVE_ARCHS, build_cifar_model
from gradient_gate.instrumentation import GateInstrumentor
from gradient_gate.sequence_models import SEQUENCE_NATIVE_ARCHS, build_sequence_model

ARCHS = ("resnet18", "resnet50", "vgg11", "vit_b_16", "convnext_tiny")
# Architectures that, like the CIFAR-native CNNs, run at native 32x32 and
# accept the same --activations swap (used for P4's "is the smooth-activation
# rise CNN-specific?" check) -- everything else (vit_b_16/convnext_tiny) keeps
# its canonical torchvision form, resized input, and fixed activation.
ACTIVATION_CONFIGURABLE_ARCHS = CIFAR_NATIVE_ARCHS + SEQUENCE_NATIVE_ARCHS
DATASETS = ("cifar10", "cifar100")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs", "csv")


def build_model_for(arch, num_classes, activation="relu"):
    if arch in CIFAR_NATIVE_ARCHS:
        return build_cifar_model(arch, num_classes=num_classes, activation=activation)
    if arch in SEQUENCE_NATIVE_ARCHS:
        return build_sequence_model(arch, num_classes=num_classes, activation=activation)
    return build_real_model(arch, num_classes=num_classes)


def get_dataloaders(dataset, arch, batch_size, data_root, num_workers=4):
    resize_224 = arch not in ACTIVATION_CONFIGURABLE_ARCHS
    train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    test_tf = []
    if resize_224:
        train_tf.append(T.Resize(224))
        test_tf.append(T.Resize(224))
    train_tf += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    test_tf += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]

    ds_cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    train_ds = ds_cls(data_root, train=True, download=True, transform=T.Compose(train_tf))
    test_ds = ds_cls(data_root, train=False, download=True, transform=T.Compose(test_tf))
    # 'spawn' context: workers fork lazily on first iteration, by which point
    # the main process has already initialized a CUDA context (model.to A
    # device)) -- forking a CUDA-initialized process hangs/breaks, so workers
    # must be spawned fresh instead of forked.
    mp_ctx = "spawn" if num_workers > 0 else None
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                                multiprocessing_context=mp_ctx, persistent_workers=num_workers > 0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False,
                                               num_workers=num_workers, pin_memory=True,
                                               multiprocessing_context=mp_ctx, persistent_workers=num_workers > 0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += nn.functional.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def instrument_one_batch(model, x, y, device):
    """Gate/rank/gradient-norm stats on a FIXED held-out batch (never used
    for a gradient step), so the per-epoch trajectory tracks the same inputs
    throughout training rather than confounding 'changed input' with
    'changed gate'. Gradient norm is computed on this same fixed batch for
    the same reason -- comparable across epochs/runs rather than reflecting
    whatever images happened to be in a given training minibatch."""
    model.train()
    x, y = x.to(device), y.to(device)
    instr = GateInstrumentor(model)
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    model.zero_grad()
    loss.backward()
    stats = instr.collect()
    instr.remove()

    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += float(p.grad.detach().norm() ** 2)
    grad_norm = grad_norm_sq ** 0.5
    model.zero_grad()

    active_fracs = [s.active_frac for s in stats if s.active_frac is not None]
    eff_ranks = [s.effective_rank for s in stats if s.effective_rank is not None]
    stable_ranks = [s.stable_rank for s in stats if s.stable_rank is not None]
    return dict(
        mean_active_frac=float(np.mean(active_fracs)) if active_fracs else float("nan"),
        min_active_frac=float(np.min(active_fracs)) if active_fracs else float("nan"),
        mean_effective_rank=float(np.mean(eff_ranks)) if eff_ranks else float("nan"),
        mean_stable_rank=float(np.mean(stable_ranks)) if stable_ranks else float("nan"),
        n_layers_measured=len(active_fracs), n_layers_total=len(stats),
        grad_norm=grad_norm,
    )


def already_done(out_path, arch, dataset, seed, final_epoch, activation=None, optimizer=None):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path)
    mask = (df.arch == arch) & (df.dataset == dataset) & (df.seed == seed) & (df.epoch == final_epoch)
    if activation is not None and "activation" in df.columns:
        mask &= (df.activation == activation)
    if optimizer is not None and "optimizer" in df.columns:
        mask &= (df.optimizer == optimizer)
    return len(df[mask]) > 0


def build_optimizer(optimizer, model, lr):
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    raise ValueError(f"unknown optimizer {optimizer}")


def run_one(arch, dataset, seed, epochs, batch_size, lr, data_root, out_path, device, num_workers,
            activation="relu", optimizer="sgd"):
    num_classes = 10 if dataset == "cifar10" else 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_dataloaders(dataset, arch, batch_size, data_root, num_workers)
    model = build_model_for(arch, num_classes, activation=activation).to(device)

    opt = build_optimizer(optimizer, model, lr)
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
        gate_stats = instrument_one_batch(model, instr_x, instr_y, device)
        row = dict(arch=arch, activation=activation, optimizer=optimizer, dataset=dataset, seed=seed,
                   epoch=epoch, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
                   lr=sched.get_last_lr()[0], epoch_time=time.time() - t0, **gate_stats)
        print(f"[train-dyn] {arch}({activation},{optimizer})/{dataset} seed={seed} epoch={epoch:3d}/{epochs} "
              f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} active_frac={gate_stats['mean_active_frac']:.3f} "
              f"eff_rank={gate_stats['mean_effective_rank']:.2f} grad_norm={gate_stats['grad_norm']:.3g} "
              f"({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--activations", nargs="+", default=["relu"],
                     help="Only applies to CIFAR-native archs (resnet18/resnet50/vgg11); "
                          "vit_b_16/convnext_tiny always use their canonical activation.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=None,
                     help="defaults to 0.1 for sgd, 1e-3 for adam/adamw if not given")
    ap.add_argument("--num-workers", type=int, default=0,
                     help="0 is the safe default verified to work end-to-end; >0 uses 'spawn' "
                          "workers (see get_dataloaders) which were unreliable/slow to start in "
                          "interactive testing on this cluster's login node — verify manually "
                          "before relying on it for an unattended multi-hour job.")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "training_dynamics.csv"))
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 0.1 if args.optimizer == "sgd" else 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[train-dyn] device={device}  archs={args.archs}  datasets={args.datasets}  "
          f"optimizer={args.optimizer}  lr={args.lr}  seeds={args.seeds}  epochs={args.epochs}")

    for dataset in args.datasets:
        for arch in args.archs:
            activations = args.activations if arch in ACTIVATION_CONFIGURABLE_ARCHS else ["relu"]
            for activation in activations:
                for seed in args.seeds:
                    if already_done(args.out, arch, dataset, seed, args.epochs - 1, activation, args.optimizer):
                        print(f"[train-dyn] [skip] {arch}({activation},{args.optimizer})/{dataset} "
                              f"seed={seed} already complete")
                        continue
                    t0 = time.time()
                    # vit_b_16/convnext_tiny at 224 need a smaller batch to fit
                    # alongside other jobs on a shared GPU; halve batch (and
                    # correspondingly halve lr, linear scaling rule) for those.
                    bs = args.batch_size if arch in ACTIVATION_CONFIGURABLE_ARCHS else max(32, args.batch_size // 4)
                    lr = args.lr if bs == args.batch_size else args.lr * bs / args.batch_size
                    run_one(arch, dataset, seed, args.epochs, bs, lr, args.data_root, args.out, device,
                            args.num_workers, activation=activation, optimizer=args.optimizer)
                    print(f"[train-dyn] {arch}({activation},{args.optimizer})/{dataset} seed={seed} done in "
                          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
