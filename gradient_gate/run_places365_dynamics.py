"""Task C -- removing the scale ceiling. ImageNet-1k itself requires
registration/license acceptance and has no anonymous, scriptable download
(confirmed before choosing an alternative -- not bypassed). Places365-Standard
(easyformat split) is the largest freely, anonymously downloadable
real-photograph dataset available: 365 scene classes, 256x256 native
resolution, no gating, downloaded directly from
https://data.csail.mit.edu/places/places365/ . This is a genuine step up
from Tiny-ImageNet-200 in both class count (365 vs 200) and native image
resolution (256x256 vs 64x64, downsampled here to 96x96 -- still 2.25x the
linear resolution of the Tiny-ImageNet experiment).

Trains the CIFAR-native ResNet-50 (same stem as resnet18 -- 3x3 stride-1, no
initial maxpool; strides (1,2,2,2) take a 96x96 input to a 12x12 feature
map before the global pool, not collapsed prematurely) on a fixed-size
per-class SUBSAMPLE of the full 1.8M-image training set (the full set is
far larger than this project's single-A100 budget can train on in the time
available; the subsample size is fixed in advance, see SAMPLES_PER_CLASS
below, and reported as exactly what it is, not silently treated as the
full dataset).
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from gradient_gate.cifar_models import cifar_resnet50
from gradient_gate.instrumentation import GateInstrumentor
from gradient_gate.run_smoothness_sweep import ACTIVATIONS
from gradient_gate.run_training_dynamics import CSV_DIR, build_optimizer

PLACES_ROOT = "/scratch/gilbreth/aelmersa/places365/places365_standard"
SAMPLES_PER_CLASS_TRAIN = 150  # fixed in advance; 365 * 150 = 54,750 train images
SAMPLES_PER_CLASS_VAL = 20     # 365 * 20 = 7,300 eval images
IMG_SIZE = 96
PLACES_MEAN = (0.485, 0.456, 0.406)
PLACES_STD = (0.229, 0.224, 0.225)


class Places365Subset(torch.utils.data.Dataset):
    """Fixed-size, seeded per-class subsample (not the full 1.8M-image
    train set) of Places365-Standard's easyformat split
    (root/{train,val}/<class>/<file>.jpg)."""

    def __init__(self, root, split, n_per_class, transform, seed=12345):
        self.transform = transform
        split_dir = os.path.join(root, split)
        classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        rng = random.Random(seed)
        self.samples = []
        for c in classes:
            class_dir = os.path.join(split_dir, c)
            files = sorted(os.listdir(class_dir))
            rng.shuffle(files)
            for fname in files[:n_per_class]:
                self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders(data_root, batch_size, num_workers=0):
    train_tf = T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(PLACES_MEAN, PLACES_STD)])
    test_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(PLACES_MEAN, PLACES_STD)])
    train_ds = Places365Subset(data_root, "train", SAMPLES_PER_CLASS_TRAIN, train_tf)
    test_ds = Places365Subset(data_root, "val", SAMPLES_PER_CLASS_VAL, test_tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
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
    model.train()
    x, y = x.to(device), y.to(device)
    instr = GateInstrumentor(model)
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    model.zero_grad()
    loss.backward()
    stats = instr.collect()
    instr.remove()
    model.zero_grad()
    active_fracs = [s.active_frac for s in stats if s.active_frac is not None]
    eff_ranks = [s.effective_rank for s in stats if s.effective_rank is not None]
    return dict(
        mean_active_frac=float(np.mean(active_fracs)) if active_fracs else float("nan"),
        mean_effective_rank=float(np.mean(eff_ranks)) if eff_ranks else float("nan"),
    )


def already_done(out_path, activation, seed, final_epoch):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path, usecols=["activation", "seed", "epoch"])
    mask = (df.activation == activation) & (df.seed == seed) & (df.epoch == final_epoch)
    return len(df[mask]) > 0


def run_one(activation, seed, epochs, batch_size, lr, data_root, out_path, device, num_workers,
            optimizer="sgd"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_dataloaders(data_root, batch_size, num_workers)
    _, act_factory = ACTIVATIONS[activation]
    model = cifar_resnet50(num_classes=365, act_layer=act_factory).to(device)

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
        row = dict(arch="resnet50", activation=activation, optimizer=optimizer, dataset="places365_subset",
                   seed=seed, epoch=epoch, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
                   lr=sched.get_last_lr()[0], epoch_time=time.time() - t0, **gate_stats)
        print(f"[places-dyn] resnet50({activation},{optimizer}) seed={seed} epoch={epoch:3d}/{epochs} "
              f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} active_frac={gate_stats['mean_active_frac']:.3f} "
              f"eff_rank={gate_stats['mean_effective_rank']:.2f} ({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "silu", "mish"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=PLACES_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "places365_dynamics.csv"))
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 0.1 if args.optimizer == "sgd" else 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[places-dyn] device={device} activations={args.activations} optimizer={args.optimizer} "
          f"lr={args.lr} seeds={args.seeds} epochs={args.epochs} "
          f"samples_per_class_train={SAMPLES_PER_CLASS_TRAIN} img_size={IMG_SIZE}")

    for activation in args.activations:
        for seed in args.seeds:
            if already_done(args.out, activation, seed, args.epochs - 1):
                print(f"[places-dyn] [skip] {activation} seed={seed} already complete")
                continue
            t0 = time.time()
            run_one(activation, seed, args.epochs, args.batch_size, args.lr, args.data_root, args.out,
                    device, args.num_workers, optimizer=args.optimizer)
            print(f"[places-dyn] {activation} seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
