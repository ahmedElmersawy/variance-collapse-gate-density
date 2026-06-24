#!/bin/bash
#SBATCH --job-name=ggc-act-ablation
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-act-ablation-%j.out
#SBATCH --error=slurm-act-ablation-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Architecture-fixed, activation-varied ablation: separates activation
# effects from architecture effects by holding the CIFAR-native ResNet-18
# and VGG-11 skeletons exactly fixed (same depth/width/stem/BatchNorm) and
# swapping ONLY the activation (gelu/silu/mish vs the existing relu
# baseline already in training_dynamics.csv). Writes to a separate CSV
# (activation_ablation.csv) with its own activation/train_acc/grad_norm
# columns -- the relu condition is NOT re-run here, it is read from the
# existing training_dynamics.csv during analysis.
python3 -u -m gradient_gate.run_training_dynamics \
    --archs resnet18 vgg11 \
    --activations gelu silu mish \
    --datasets cifar10 cifar100 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/activation_ablation.csv

echo "Job complete at $(date)"
