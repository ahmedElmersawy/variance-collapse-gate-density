#!/bin/bash
#SBATCH --job-name=ggc-traindyn-vitcnx
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-train-dyn-vitcnx-%j.out
#SBATCH --error=slurm-train-dyn-vitcnx-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# ViT-B/16 and ConvNeXt-Tiny at 224x224 are the bottleneck (~6.4 min/epoch
# for ViT-B/16, observed in job 11039609) -- a full 3-seed x 25-epoch x
# 2-dataset sweep for both would need >24hr. Compute-vs-coverage tradeoff:
# epochs reduced 25->15 (the resnet18/50/vgg11 trend in training_dynamics.csv
# is already monotonic well before epoch 15, so 15 epochs is enough to see
# the same qualitative trend) and seeds reduced to 1 (vs 3 for the cheap
# CIFAR-native archs) for the combos not already covered. vit_b_16/cifar10
# already has 3 full seeds from job 11039609 (rows up to epoch 24) --
# already_done() will detect the existing epoch=14 row for each of those 3
# seeds and skip them; this job only computes the 3 remaining combos:
# vit_b_16/cifar100, convnext_tiny/cifar10, convnext_tiny/cifar100 (1 seed
# each). Report this n=1 status honestly -- it is preliminary single-run
# evidence, not a statistically powered claim like the n=3 CIFAR-native runs.
python3 -u -m gradient_gate.run_training_dynamics \
    --archs vit_b_16 convnext_tiny \
    --datasets cifar10 cifar100 \
    --seeds 0 \
    --epochs 15 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Job complete at $(date)"
