"""Post-hoc analysis for Experiment 2: gate-density-guided pruning.

Loads the checkpoints saved by run_pruning_experiment.py, computes per-
channel final gate density (on the same fixed instrumentation batch saved
in the checkpoint) and per-channel weight magnitude (L1 norm of the conv1
filter) for the 8 prunable layers (stages.{0..7}.conv1 / bn1, gated by
act1), then evaluates immediate test-accuracy drop (no retraining) for
three pruning criteria x four ratios, globally across all 8 layers' 1920
channels combined (not per-layer-uniform pruning).

Channels are "pruned" by zeroing conv1.weight[channel], bn1.weight[channel]
(gamma), and bn1.bias[channel] (beta) -- since every activation tested maps
f(0)=0, this makes the channel's contribution to the rest of the network
exactly zero, equivalent to removal, without needing to handle tensor
shape/dimension bookkeeping.
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel

from gradient_gate.cifar_models import build_cifar_model
from gradient_gate.instrumentation import GRAD_RATIO_EPS
from gradient_gate.run_pruning_experiment import CKPT_DIR
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, evaluate, get_dataloaders

PRUNABLE_LAYERS = [f"stages.{i}" for i in range(8)]
RATIOS = (0.10, 0.30, 0.50, 0.70)


class PerChannelGateCollector:
    """Per-channel (not pooled) gate density on the named act1 modules of
    each BasicBlock, for ranking which channels to prune."""

    def __init__(self, model, layer_prefixes):
        self.per_channel = {}
        self._handles = []
        for name, module in model.named_modules():
            if any(name == f"{p}.act1" for p in layer_prefixes):
                self._handles.append(module.register_full_backward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, grad_input, grad_output):
            gi, go = grad_input[0].detach(), grad_output[0].detach()
            valid = go.abs() > GRAD_RATIO_EPS
            gate = torch.zeros_like(go)
            gate[valid] = (gi[valid].abs() / go[valid].abs()).clamp(max=1e6)
            self.per_channel[name.replace(".act1", "")] = gate.mean(dim=(0, 2, 3)).cpu()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


def compute_channel_table(model, instr_x, instr_y, device):
    model.train()
    instr_x, instr_y = instr_x.to(device), instr_y.to(device)
    collector = PerChannelGateCollector(model, PRUNABLE_LAYERS)
    out = model(instr_x)
    loss = nn.functional.cross_entropy(out, instr_y)
    model.zero_grad()
    loss.backward()
    gate_by_layer = collector.per_channel
    collector.remove()
    model.zero_grad()

    rows = []
    for layer in PRUNABLE_LAYERS:
        conv = dict(model.named_modules())[f"{layer}.conv1"]
        gate = gate_by_layer[layer]
        weight_mag = conv.weight.detach().abs().sum(dim=(1, 2, 3))  # L1 norm per output channel
        for ch in range(conv.out_channels):
            rows.append(dict(layer=layer, channel=ch, gate_density=float(gate[ch]),
                              weight_magnitude=float(weight_mag[ch])))
    return pd.DataFrame(rows)


def prune_and_evaluate(model, channel_table, method, ratio, test_loader, device):
    model = copy.deepcopy(model)
    n_prune = int(round(len(channel_table) * ratio))
    if method == "gate_density":
        victims = channel_table.nsmallest(n_prune, "gate_density")
    elif method == "magnitude":
        victims = channel_table.nsmallest(n_prune, "weight_magnitude")
    elif method == "random":
        victims = channel_table.sample(n=n_prune, random_state=0)
    else:
        raise ValueError(method)

    modules = dict(model.named_modules())
    with torch.no_grad():
        for _, row in victims.iterrows():
            layer, ch = row["layer"], int(row["channel"])
            modules[f"{layer}.conv1"].weight[ch].zero_()
            modules[f"{layer}.bn1"].weight[ch].zero_()
            modules[f"{layer}.bn1"].bias[ch].zero_()

    _, acc = evaluate(model, test_loader, device)
    del model
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "silu", "mish"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "pruning_results.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_dataloaders("cifar10", "resnet18", 256, args.data_root, num_workers=0)

    rows = []
    for activation in args.activations:
        for seed in args.seeds:
            ckpt_path = os.path.join(args.ckpt_dir, f"resnet18_{activation}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[pruning-analysis] [skip] missing checkpoint: {ckpt_path}")
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_cifar_model("resnet18", activation=activation).to(device)
            model.load_state_dict(ckpt["state_dict"])
            instr_x, instr_y = ckpt["instr_x"], ckpt["instr_y"]

            channel_table = compute_channel_table(model, instr_x, instr_y, device)
            baseline_acc = ckpt["final_test_acc"]

            for ratio in RATIOS:
                for method in ("gate_density", "magnitude", "random"):
                    acc = prune_and_evaluate(model, channel_table, method, ratio, test_loader, device)
                    rows.append(dict(activation=activation, seed=seed, ratio=ratio, method=method,
                                      baseline_acc=baseline_acc, pruned_acc=acc,
                                      acc_drop=baseline_acc - acc))
                    print(f"[pruning-analysis] {activation} seed={seed} ratio={ratio:.0%} "
                          f"method={method:12s} acc={acc:.4f} (baseline={baseline_acc:.4f}, "
                          f"drop={baseline_acc-acc:+.4f})")
            del model

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[pruning-analysis] wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
