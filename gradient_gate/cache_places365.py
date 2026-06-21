"""One-time fix for Task C's real I/O bottleneck: discovered via direct
diagnosis (nvidia-smi showing 0% GPU utilization, /proc/<pid>/io showing
~0.57MB/s sustained read throughput) that per-image PIL decode from
individual JPEG files on this scratch filesystem has severe per-file
latency -- at the observed rate, one epoch's data loading alone would take
roughly an hour, making the full run (25 epochs x 4 activations x 2 seeds)
take days, not the 24h budgeted. The fix is not "use more workers" (this
project's own established num_workers=0 rule, justified by a documented
hang risk under num_workers>0 with CUDA+fork, was learned for CIFAR-scale
data and is conservatively kept here too) but to pay the per-file I/O cost
exactly ONCE rather than once per epoch: this script reads every one of
the 62,050 needed images (150 train + 20 val per class, the same fixed
subsample run_places365_dynamics.py already uses) a single time, resizes
each to a fixed CACHE_SIZE (larger than the final training resolution so
RandomResizedCrop-style augmentation still has real room to operate), and
stores the whole subsample as two single tensor files. Loading two single
files is one I/O operation each, regardless of how many images are inside.
"""
import os
import time

import numpy as np
import torch
from PIL import Image

from gradient_gate.run_places365_dynamics import PLACES_ROOT, SAMPLES_PER_CLASS_TRAIN, SAMPLES_PER_CLASS_VAL

CACHE_DIR = "/scratch/gilbreth/aelmersa/places365/cache"
CACHE_SIZE = 144  # > IMG_SIZE=96, leaves real room for RandomResizedCrop(96, scale=(0.7,1.0))


def build_cache(split, n_per_class, seed=12345):
    import random
    split_dir = os.path.join(PLACES_ROOT, split)
    classes = sorted(os.listdir(split_dir))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)

    images = []
    labels = []
    t0 = time.time()
    n_done = 0
    for c in classes:
        class_dir = os.path.join(split_dir, c)
        files = sorted(os.listdir(class_dir))
        rng.shuffle(files)
        for fname in files[:n_per_class]:
            path = os.path.join(class_dir, fname)
            img = Image.open(path).convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(class_to_idx[c])
            n_done += 1
            if n_done % 5000 == 0:
                print(f"[cache] {split}: {n_done} images, {time.time()-t0:.1f}s elapsed", flush=True)

    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()  # [N,3,H,W] uint8
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"[cache] {split}: done, {len(labels)} images, shape {images.shape}, "
          f"{images.numel()/1e9:.2f}GB, {time.time()-t0:.1f}s total", flush=True)
    return images, labels, class_to_idx


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_path = os.path.join(CACHE_DIR, "train_cache.pt")
    val_path = os.path.join(CACHE_DIR, "val_cache.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("[cache] both cache files already exist, skipping rebuild")
        return

    train_images, train_labels, class_to_idx = build_cache("train", SAMPLES_PER_CLASS_TRAIN)
    torch.save(dict(images=train_images, labels=train_labels, class_to_idx=class_to_idx), train_path)
    print(f"[cache] wrote {train_path}")

    val_images, val_labels, val_class_to_idx = build_cache("val", SAMPLES_PER_CLASS_VAL)
    assert val_class_to_idx == class_to_idx, "train/val class ordering mismatch"
    torch.save(dict(images=val_images, labels=val_labels, class_to_idx=val_class_to_idx), val_path)
    print(f"[cache] wrote {val_path}")


if __name__ == "__main__":
    main()
