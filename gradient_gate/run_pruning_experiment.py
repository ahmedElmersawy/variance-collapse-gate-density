"""Experiment 2 (gate-density-guided pruning) + Experiment 3 (mechanism
logging), bundled into one training run since both need the same
checkpoints and the same fixed instrumentation batch.

Trains the standard (BatchNorm) CIFAR-native ResNet-18 across 4 activations
x 3 seeds x 25 epochs, saving a final checkpoint for the pruning analysis
(run_pruning_analysis.py) and logging, at epochs {0,6,12,18,24}:
  - gate_mean, active_frac (train-mode, fixed batch — same definition as
    elsewhere in this project)
  - pre-/post-activation variance (train-mode, fixed batch)
  - BatchNorm gamma/beta statistics, pooled and per-layer
  - a Hutchinson trace estimate and a power-iteration top-eigenvalue
    estimate of the loss Hessian (eval-mode, fixed batch, to avoid BN
    running-stat contamination from the extra forward passes sharpness
    estimation requires)

This is exploratory mechanism logging, not a causal claim -- see
mechanism_summary.md for the analysis this feeds.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.cifar_models import build_cifar_model
from gradient_gate.run_threshold_robustness import PooledGateCollector
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, build_optimizer, evaluate, get_dataloaders

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs", "checkpoints")
MECH_EPOCHS = (0, 6, 12, 18, 24)


def gate_and_activation_stats(model, x, y, device):
    """Train-mode pass: pooled gate_mean/active_frac (existing definition)
    plus pre-/post-activation variance via lightweight forward hooks on the
    same elementwise-activation layers."""
    model.train()
    x, y = x.to(device), y.to(device)

    pre_vars, post_vars = [], []

    def fwd_hook(module, inp, out):
        pre_vars.append(float(inp[0].detach().var()))
        post_vars.append(float(out.detach().var()))

    from gradient_gate.instrumentation import ELEMENTWISE_ACT_TYPES
    handles = []
    for _, m in model.named_modules():
        if isinstance(m, ELEMENTWISE_ACT_TYPES):
            handles.append(m.register_forward_hook(fwd_hook))

    collector = PooledGateCollector(model)
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    model.zero_grad()
    loss.backward()
    gate_vals = collector.pooled()
    collector.remove()
    for h in handles:
        h.remove()
    model.zero_grad()

    return dict(
        gate_mean=float(gate_vals.mean()) if gate_vals.size else float("nan"),
        active_frac=float((gate_vals > 0.01).mean()) if gate_vals.size else float("nan"),
        pre_activation_var=float(np.mean(pre_vars)) if pre_vars else float("nan"),
        post_activation_var=float(np.mean(post_vars)) if post_vars else float("nan"),
    )


def bn_gamma_beta_stats(model):
    """Pooled and per-layer BatchNorm scale/shift statistics."""
    all_gamma, all_beta = [], []
    per_layer = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            g = m.weight.detach().abs()
            b = m.bias.detach().abs()
            all_gamma.append(g)
            all_beta.append(b)
            per_layer.append(dict(layer=name, mean_abs_gamma=float(g.mean()), mean_abs_beta=float(b.mean())))
    pooled = dict(
        bn_mean_abs_gamma=float(torch.cat(all_gamma).mean()) if all_gamma else float("nan"),
        bn_mean_abs_beta=float(torch.cat(all_beta).mean()) if all_beta else float("nan"),
    )
    return pooled, per_layer


def sharpness_proxy(model, x, y, device, n_hutchinson=5, n_power_iter=10):
    """Hutchinson trace estimate + power-iteration top-eigenvalue estimate
    of the loss Hessian, on the fixed instrumentation batch. eval() mode to
    avoid BatchNorm running-stat updates from the repeated forward passes
    this requires. Exploratory proxy, not an exact Hessian computation."""
    model.eval()
    x, y = x.to(device), y.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    def hvp(v_list):
        dot = sum((g * v).sum() for g, v in zip(grads, v_list))
        return torch.autograd.grad(dot, params, retain_graph=True)

    traces = []
    for _ in range(n_hutchinson):
        v = [torch.randint(0, 2, p.shape, device=device, dtype=p.dtype) * 2 - 1 for p in params]
        hv = hvp(v)
        traces.append(sum(float((hvi * vi).sum()) for hvi, vi in zip(hv, v)))
    trace_est = float(np.mean(traces))

    v = [torch.randn_like(p) for p in params]
    vnorm = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    v = [vi / vnorm for vi in v]
    eigval = None
    for _ in range(n_power_iter):
        hv = hvp(v)
        hv_norm = torch.sqrt(sum((hvi ** 2).sum() for hvi in hv)).item()
        v = [hvi / (hv_norm + 1e-12) for hvi in hv]
        eigval = hv_norm

    model.zero_grad()
    model.train()
    return dict(hessian_trace_estimate=trace_est, top_eigenvalue_estimate=eigval)


def run_one(activation, seed, epochs, batch_size, lr, data_root, mech_out, layer_out, device, num_workers,
            ckpt_dir, optimizer="sgd"):
    num_classes = 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_dataloaders("cifar10", "resnet18", batch_size, data_root, num_workers)
    model = build_cifar_model("resnet18", num_classes=num_classes, activation=activation).to(device)

    opt = build_optimizer(optimizer, model, lr)
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

        if epoch in MECH_EPOCHS:
            gate_stats = gate_and_activation_stats(model, instr_x, instr_y, device)
            bn_pooled, bn_layers = bn_gamma_beta_stats(model)
            sharp_stats = sharpness_proxy(model, instr_x, instr_y, device)
            row = dict(activation=activation, optimizer=optimizer, seed=seed, epoch=epoch, test_acc=test_acc,
                       epoch_time=time.time() - t0, **gate_stats, **bn_pooled, **sharp_stats)
            pd.DataFrame([row]).to_csv(mech_out, mode="a", header=not os.path.exists(mech_out), index=False)
            for rec in bn_layers:
                rec.update(activation=activation, optimizer=optimizer, seed=seed, epoch=epoch)
            pd.DataFrame(bn_layers).to_csv(layer_out, mode="a", header=not os.path.exists(layer_out),
                                            index=False)
            print(f"[pruning-exp] {activation} seed={seed} epoch={epoch:3d}/{epochs} [MECH] "
                  f"test_acc={test_acc:.3f} gate_mean={gate_stats['gate_mean']:.4f} "
                  f"bn_gamma={bn_pooled['bn_mean_abs_gamma']:.4f} trace={sharp_stats['hessian_trace_estimate']:.2f} "
                  f"top_eig={sharp_stats['top_eigenvalue_estimate']:.2f} ({time.time()-t0:.1f}s)")
        else:
            print(f"[pruning-exp] {activation} seed={seed} epoch={epoch:3d}/{epochs} "
                  f"test_acc={test_acc:.3f} ({time.time()-t0:.1f}s)")

    os.makedirs(ckpt_dir, exist_ok=True)
    # sgd (the original, default condition) keeps its original filename exactly --
    # existing checkpoints/run_pruning_analysis.py depend on this and must not be
    # orphaned or duplicated by this optimizer extension.
    suffix = "" if optimizer == "sgd" else f"_{optimizer}"
    ckpt_path = os.path.join(ckpt_dir, f"resnet18_{activation}{suffix}_seed{seed}.pt")
    torch.save(dict(state_dict=model.state_dict(), activation=activation, optimizer=optimizer, seed=seed,
                     final_test_acc=test_acc, instr_x=instr_x, instr_y=instr_y), ckpt_path)
    print(f"[pruning-exp] saved checkpoint -> {ckpt_path}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def already_done(ckpt_dir, activation, seed, optimizer="sgd"):
    suffix = "" if optimizer == "sgd" else f"_{optimizer}"
    return os.path.exists(os.path.join(ckpt_dir, f"resnet18_{activation}{suffix}_seed{seed}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "silu", "mish"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=None,
                     help="defaults to 0.1 for sgd, 1e-3 for adam/adamw if not given")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--mech-out", default=os.path.join(CSV_DIR, "mechanism_logging.csv"))
    ap.add_argument("--layer-out", default=os.path.join(CSV_DIR, "bn_gamma_layerwise.csv"))
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 0.1 if args.optimizer == "sgd" else 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.mech_out), exist_ok=True)
    print(f"[pruning-exp] device={device} activations={args.activations} optimizer={args.optimizer} "
          f"lr={args.lr} seeds={args.seeds} epochs={args.epochs} mech_epochs={MECH_EPOCHS}")

    for activation in args.activations:
        for seed in args.seeds:
            if already_done(args.ckpt_dir, activation, seed, args.optimizer):
                print(f"[pruning-exp] [skip] {activation}({args.optimizer}) seed={seed} checkpoint already exists")
                continue
            t0 = time.time()
            run_one(activation, seed, args.epochs, args.batch_size, args.lr, args.data_root,
                    args.mech_out, args.layer_out, device, args.num_workers, args.ckpt_dir,
                    optimizer=args.optimizer)
            print(f"[pruning-exp] {activation}({args.optimizer}) seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
