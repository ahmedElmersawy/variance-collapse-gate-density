"""P3 -- one scale step beyond CIFAR. Trains CIFAR-native ResNet-18 (the
same 3x3 stride-1, no-initial-maxpool stem documented in cifar_models.py --
already appropriate for 64x64, not just 32x32: strides=(1,2,2,2) takes a
64x64 input to an 8x8 feature map before the global average pool, the same
"don't collapse the image in the stem" property the module docstring
requires, just scaled up) on Tiny-ImageNet-200 (64x64, 200 classes), for
relu/gelu/silu/mish, logging the same active_frac/effective_rank trajectory
as run_training_dynamics.py, to test whether the activation-class direction
split survives a non-CIFAR, larger-vocabulary dataset.

Expects data/tiny-imagenet-200/ (the standard public download from
http://cs231n.stanford.edu/tiny-imagenet-200.zip) already extracted, with
its standard layout: train/<wnid>/images/*.JPEG, val/images/*.JPEG +
val/val_annotations.txt mapping filename -> wnid. ImageFolder doesn't fit
this layout directly (there's an extra 'images/' level under each train
class), hence the small custom Dataset below.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from gradient_gate.cifar_models import cifar_resnet18
from gradient_gate.instrumentation import GateInstrumentor
from gradient_gate.run_smoothness_sweep import ACTIVATIONS
from gradient_gate.run_training_dynamics import CSV_DIR, build_optimizer, instrument_one_batch

TINY_IMAGENET_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tiny-imagenet-200")
# Tiny-ImageNet is a curated ImageNet-1k subset; reusing ImageNet's standard
# per-channel mean/std rather than re-deriving CIFAR-style stats from scratch.
TIN_MEAN = (0.485, 0.456, 0.406)
TIN_STD = (0.229, 0.224, 0.225)


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        self.transform = transform
        wnids = sorted(open(os.path.join(root, "wnids.txt")).read().split())
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples = []
        if split == "train":
            for wnid in wnids:
                img_dir = os.path.join(root, "train", wnid, "images")
                for fname in sorted(os.listdir(img_dir)):
                    self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        elif split == "val":
            ann_path = os.path.join(root, "val", "val_annotations.txt")
            img_dir = os.path.join(root, "val", "images")
            for line in open(ann_path):
                fname, wnid = line.split("\t")[:2]
                self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[wnid]))
        else:
            raise ValueError(f"unknown split {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders(data_root, batch_size, num_workers=0):
    train_tf = T.Compose([T.RandomCrop(64, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(TIN_MEAN, TIN_STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(TIN_MEAN, TIN_STD)])
    train_ds = TinyImageNetDataset(data_root, "train", train_tf)
    test_ds = TinyImageNetDataset(data_root, "val", test_tf)
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
    model = cifar_resnet18(num_classes=200, act_layer=act_factory).to(device)

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
        row = dict(arch="resnet18", activation=activation, optimizer=optimizer, dataset="tinyimagenet200",
                   seed=seed, epoch=epoch, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
                   lr=sched.get_last_lr()[0], epoch_time=time.time() - t0, **gate_stats)
        print(f"[tin-dyn] resnet18({activation},{optimizer})/tinyimagenet200 seed={seed} "
              f"epoch={epoch:3d}/{epochs} train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
              f"active_frac={gate_stats['mean_active_frac']:.3f} "
              f"eff_rank={gate_stats['mean_effective_rank']:.2f} ({time.time()-t0:.1f}s)")
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "silu", "mish"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default=TINY_IMAGENET_ROOT)
    ap.add_argument("--out", default=os.path.join(CSV_DIR, "tinyimagenet_dynamics.csv"))
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 0.1 if args.optimizer == "sgd" else 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[tin-dyn] device={device} activations={args.activations} optimizer={args.optimizer} "
          f"lr={args.lr} seeds={args.seeds} epochs={args.epochs} data_root={args.data_root}")

    for activation in args.activations:
        for seed in args.seeds:
            if already_done(args.out, activation, seed, args.epochs - 1):
                print(f"[tin-dyn] [skip] {activation} seed={seed} already complete")
                continue
            t0 = time.time()
            run_one(activation, seed, args.epochs, args.batch_size, args.lr, args.data_root, args.out,
                    device, args.num_workers, optimizer=args.optimizer)
            print(f"[tin-dyn] {activation} seed={seed} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
