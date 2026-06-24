# Variance Collapse Predicts When Gate Density Diverges by Activation Class

Whether the fraction of gradient-carrying units (*gate density*, Γ(x) = |f'(x)|) rises or falls during training is often treated as activation-specific folklore — "ReLU dies, GELU doesn't." This work shows the direction is **derived**, not folklore: it is governed by a single, training-free quantity — whether the post-BatchNorm pre-activation variance collapses during training.

## Key result

A single predictor, validated three independent ways, with zero new free parameters:

```
Direction = sign( μ(z) − z_low(θ) )
```

- **48/48** architecture-fixed predictions correct (p = 2.44×10⁻⁴) — ResNet-18/VGG-11, activation swapped, architecture held identical
- **12/12** activation×seed cells correct under AdamW — same predictor, decoupled weight decay
- **9/9** mechanism predictions correct, including Softplus β=50
- Generalizes across optimizer (SGD/AdamW), scale (Tiny-ImageNet-200, Places365), and architecture (MLP-Mixer, Transformer-Encoder) without retraining

Three pre-registered negative controls are reported openly: gate density does not discriminate via representational rank, is actively misleading for channel pruning on smooth activations, and does not predict low-data fine-tuning accuracy.

## Repository layout

- `main.tex` — NeurIPS 2026 submission (paper)
- `appendix.tex`, `appendix_synthetic_theory.tex`, `checklist.tex` — paper appendix and reproducibility checklist
- `gradient_gate/` — instrumentation (gate-density hooks), architectures, and experiment runners
- `gradient_gate_outputs/` — CSV results, figures, and markdown summaries per experiment
- `figures/` — generated plots used in the paper and poster
- `poster/`, `poster_materials/` — conference poster source and assets
- `scripts/` — SLURM job scripts for individual experiments
- `ROADMAP.md` — running log of verified findings and remaining work

## Reproducing experiments

Each script in `scripts/` submits one experiment via `sbatch` (no conda/module activation required — system `python3` + `$HOME/.local` packages). See `gradient_gate/` for the underlying instrumentation (`GateInstrumentor`) and per-experiment entry points.

Datasets (CIFAR-10/100, Tiny-ImageNet-200, Places365) are not tracked in this repository — see `gradient_gate/` download/cache scripts to regenerate them locally.
