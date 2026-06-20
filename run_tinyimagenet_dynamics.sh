#!/bin/bash
#SBATCH --job-name=ggc-tin-dyn
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-tin-dyn-%j.out
#SBATCH --error=slurm-tin-dyn-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P3: one scale step beyond CIFAR. ResNet-18 (CIFAR-native stem, already
# 64x64-appropriate -- see run_tinyimagenet_dynamics.py module docstring),
# relu/gelu/silu/mish, 3 seeds, 25 epochs, standard SGD, on Tiny-ImageNet-200
# (data/tiny-imagenet-200/, the standard public download, 100k train / 10k
# val images, 200 classes). Tests whether the activation-class active_frac
# direction split survives a non-CIFAR, larger-vocabulary, higher-resolution
# dataset. 4x the classes and 4x the pixels per image of CIFAR -- budget
# accordingly relative to the CIFAR ablation jobs.
python3 -u -m gradient_gate.run_tinyimagenet_dynamics \
    --activations relu gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer sgd \
    --num-workers 0

echo "Job complete at $(date)"
