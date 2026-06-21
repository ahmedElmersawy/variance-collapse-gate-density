"""Task D (pre-registered in RESULTS_LOG.md before this script produced any
output): does a checkpoint's already-measured final active_frac predict
its low-data fine-tuning test accuracy? Fine-tunes every existing ResNet-18
checkpoint (SGD and AdamW, 4 activations, 3 seeds each, 24 total) on a
fixed, seeded 5%-of-train-set CIFAR-10 subsample, identical fine-tuning
recipe for every checkpoint regardless of its original optimizer.
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gradient_gate.cifar_models import build_cifar_model
from gradient_gate.run_training_dynamics import CSV_DIR, DATA_ROOT, evaluate, get_dataloaders

CKPT_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs", "checkpoints"),
    os.path.join(os.path.dirname(__file__), "..", "gradient_gate_outputs", "checkpoints_adamw"),
]
FINETUNE_FRACTION = 0.05
FINETUNE_SEED = 999  # fixed subsample seed, shared across every checkpoint


def get_finetune_loader(batch_size, data_root, num_workers=0):
    train_loader_full, test_loader = get_dataloaders("cifar10", "resnet18", batch_size, data_root, num_workers)
    full_ds = train_loader_full.dataset
    n_total = len(full_ds)
    n_sub = int(n_total * FINETUNE_FRACTION)
    rng = np.random.RandomState(FINETUNE_SEED)
    idx = rng.choice(n_total, size=n_sub, replace=False)
    sub_ds = torch.utils.data.Subset(full_ds, idx.tolist())
    finetune_loader = torch.utils.data.DataLoader(sub_ds, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=False)
    return finetune_loader, test_loader, n_sub


def parse_ckpt_name(path):
    name = os.path.basename(path).replace(".pt", "")
    m = re.match(r"resnet18_([a-z]+)(?:_(adamw))?_seed(\d)", name)
    activation, optimizer, seed = m.group(1), m.group(2) or "sgd", int(m.group(3))
    return activation, optimizer, seed


def run_one(ckpt_path, finetune_loader, test_loader, device, epochs, lr):
    activation, optimizer, seed = parse_ckpt_name(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_cifar_model("resnet18", num_classes=10, activation=activation).to(device)
    model.load_state_dict(ckpt["state_dict"])

    pre_loss, pre_acc = evaluate(model, test_loader, device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        for x, y in finetune_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

    post_loss, post_acc = evaluate(model, test_loader, device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(activation=activation, optimizer=optimizer, seed=seed, ckpt_path=ckpt_path,
                pre_finetune_test_acc=pre_acc, post_finetune_test_acc=post_acc,
                pre_finetune_test_loss=pre_loss, post_finetune_test_loss=post_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "lowdata_finetune.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = []
    for d in CKPT_DIRS:
        ckpts.extend(sorted(glob.glob(os.path.join(d, "*.pt"))))
    print(f"[lowdata-ft] device={device} found {len(ckpts)} checkpoints, "
          f"finetune_fraction={FINETUNE_FRACTION}, epochs={args.epochs}, lr={args.lr}")

    finetune_loader, test_loader, n_sub = get_finetune_loader(args.batch_size, args.data_root, args.num_workers)
    print(f"[lowdata-ft] fine-tune subsample size = {n_sub} images (seed={FINETUNE_SEED}, fixed for all checkpoints)")

    rows = []
    for ckpt_path in ckpts:
        t0 = time.time()
        row = run_one(ckpt_path, finetune_loader, test_loader, device, args.epochs, args.lr)
        rows.append(row)
        print(f"[lowdata-ft] {os.path.basename(ckpt_path)}: pre={row['pre_finetune_test_acc']:.4f} "
              f"post={row['post_finetune_test_acc']:.4f} ({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)

    print(f"[lowdata-ft] done, {len(rows)} checkpoints fine-tuned, results -> {args.out}")


if __name__ == "__main__":
    main()
