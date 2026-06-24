# Experiment 1: BatchNorm Necessity Ablation — Summary

**Setup:** CIFAR-native ResNet-18, CIFAR-10, BatchNorm vs. GroupNorm (32 groups,
or fewer if a layer has <32 channels), 4 activations (ReLU/GELU/SiLU/Mish),
3 seeds, 25 epochs. 24 runs total. Identical architecture, optimizer
(SGD+momentum+cosine), batch size, and augmentation in both conditions —
only the normalization layer differs.

## Direction and sign test (unit of replication: independent seed trajectory)

| norm | activation | seeds declining/rising | sign-test p (n=3) | start→end active_frac | final test_acc |
|---|---|---|---|---|---|
| batchnorm | relu | 3/3 declining | 0.125 | 0.510→0.368 | 0.927 |
| batchnorm | gelu | 3/3 rising | 0.125 | 0.983→0.994 | 0.925 |
| batchnorm | silu | 3/3 rising | 0.125 | 0.993→0.997 | 0.917 |
| batchnorm | mish | 3/3 rising | 0.125 | 0.992→0.997 | 0.918 |
| groupnorm | relu | 3/3 declining | 0.125 | 0.387→0.303 | 0.894 |
| groupnorm | gelu | 3/3 rising | 0.125 | 0.984→0.994 | 0.906 |
| groupnorm | silu | 3/3 rising | 0.125 | 0.994→0.997 | 0.893 |
| groupnorm | mish | 3/3 rising | 0.125 | 0.993→0.997 | 0.888 |

n=3 per cell cannot reach significance alone (best possible sign-test
p=0.125), but pooling within each norm condition across the four
activations (the correct broader unit — direction is being tested, not
magnitude): **ReLU declining 3/3, smooth activations rising 9/9, under
BOTH BatchNorm and GroupNorm.** Direction never flips for a single seed in
either condition.

## Comparison: BatchNorm vs. GroupNorm

- **Direction: identical in both conditions.** Every activation's
  qualitative direction is unchanged by removing BatchNorm.
- **Magnitude: noisier under GroupNorm**, especially for SiLU/Mish
  (e.g. Mish seed-level ρ under BatchNorm: 0.94/0.99/0.83; under GroupNorm:
  0.32/0.30/0.60 — same sign, weaker and more variable trend strength).
- **Final accuracy is lower under GroupNorm** (88.8–90.6% vs. 91.7–92.7%
  under BatchNorm), consistent with GroupNorm generally needing different
  hyperparameters to match BatchNorm's performance at this scale — not
  itself evidence about the gate-density phenomenon, just a reminder that
  this is an unmatched-tuning comparison, not an apples-to-apples optimum
  for GroupNorm.

## Conclusion (per the pre-specified interpretation rule)

**The activation-class split survives GroupNorm. BatchNorm is not
necessary for the phenomenon.** The direction (ReLU declining, smooth
activations rising) is a property of the activation function on this
architecture, not an artifact of BatchNorm specifically. We do not
over-interpret the magnitude differences under GroupNorm (noisier trends,
lower accuracy) as evidence about the underlying mechanism — they are at
least partly attributable to GroupNorm being unmatched in hyperparameters,
not necessarily to anything about gate dynamics.
