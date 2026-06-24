# Gradient Gate Collapse — Status and Roadmap

## Core findings (all verified by real execution, not assumed)

**1. Gate collapse during normal training (3 ReLU architectures × 2 datasets, 3 seeds, 25 epochs):**
ResNet-18, ResNet-50, VGG-11 on CIFAR-10/CIFAR-100 all show gate density
(active_frac) falling and effective rank rising monotonically as test
accuracy improves. Corrected (seed-level, not pooled-epoch) statistics:
every one of 18 independent ReLU training runs shows active_frac declining
and effective rank rising (sign-test p=3.8e-6 for both).

**2. ViT-B/16 (GELU) shows the opposite direction:** gate density rises
toward a ceiling instead of collapsing, replicated across both datasets
(4/4 independent runs positive). ConvNeXt-Tiny's training-dynamics result
remains inconclusive (1 seed, undertrained on CIFAR-100, 18.9% test acc) —
not used as evidence either way.

**3. Architecture-fixed, activation-varied ablation (NEW, job 11044075,
COMPLETED) — resolves "is this activation or architecture?":**
Held ResNet-18 and VGG-11's architecture exactly fixed (identical
depth/width/stem/BatchNorm, confirmed identical parameter counts) and
swapped only the activation: ReLU baseline vs. GELU/SiLU/Mish, 3 seeds,
both datasets, 25 epochs. Result: every one of 12 independent ReLU runs
(2 archs × 2 datasets × 3 seeds) shows active_frac declining; every one of
12 GELU, 12 SiLU, and 12 Mish runs shows it rising instead (sign-test
p=2.44e-4 for each activation's 12/12 consistency). **The direction flips
with the activation, not the architecture family.** Effective rank, by
contrast, rises in all 48/48 runs regardless of activation (p=3.6e-15) —
rank growth does not track the activation-class split; only gate-density
direction does. See `gradient_gate_outputs/csv/activation_ablation.csv` and
`activation_ablation_seedlevel_stats.csv`; figure at
`gradient_gate_outputs/figures/activation_ablation.png`.

**4. Privacy (DLG/iDLG/Inverting Gradients, 20 seeds × 7 α, toy victim
model):** gate stiffness significantly reduces iDLG's convergence
probability (p=3.2e-5) but provides no protection against the stronger,
scale-invariant Inverting Gradients attack (100% convergence regardless of
α). Attack-dependent, not a general privacy claim. Not demonstrated to
connect to the real-architecture gate dynamics above (different, toy model).

**5. ViT vanishing-gradient falsification:** the original "ViT-B/16 at
random init has vanishing gradients" finding is a pure initialization
artifact (100%→0% vanish rate at every depth 1–12 tested, default vs.
truncated-normal init), not depth-induced — replicated twice independently.
Separate from, and not to be conflated with, finding 1's training-time
phenomenon.

## Current paper framing

"Activation-class-dependent gate dynamics" — now directly supported by
finding 3, not just suggested by the single ResNet/ViT architecture pair.
"Architecture-dependent" framing is ruled out. See conversation record for
the full reviewer-simulation, related-work positioning, and final
Introduction draft built around this framing.

## Infrastructure notes

- `gradient_gate/cifar_models.py`: CIFAR-native ResNet-18/50/VGG-11 now
  accept an `activation` argument (`relu`/`gelu`/`silu`/`mish`) via
  `build_cifar_model(name, activation=...)`.
- `gradient_gate/run_training_dynamics.py`: now logs `train_acc` and
  `grad_norm` (on the same fixed instrumentation batch as gate/rank stats)
  per epoch, and accepts `--activations` for the CIFAR-native architectures.
- All SLURM jobs use no conda/module activation (system python3 + `$HOME/.local`
  packages) and `--num-workers 0` (verified reliable; `num_workers>0` hung in
  interactive testing on this cluster).
- Background (non-`sbatch`) processes die on session disconnect — everything
  multi-hour goes through `sbatch`.

## Remaining roadmap

1. ~~Architecture-vs-activation ablation~~ — **done**, resolved cleanly.
2. U-Net / MLP-Mixer / Transformer-Encoder model factories, if broader
   architecture coverage is wanted later (`GateInstrumentor` already works
   on them).
3. Larger-scale (Tiny-ImageNet/ImageNet-subset) validation, if reviewers
   push on CIFAR-only scope — not yet started, real GPU-time investment.
4. ConvNeXt-Tiny full 3-seed completion, if its inconclusive status becomes
   load-bearing for any claim (currently it is not).
