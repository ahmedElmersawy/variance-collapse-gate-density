#!/bin/bash
#SBATCH --job-name=ggc-thresh-robust
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-thresh-robust-%j.out
#SBATCH --error=slurm-thresh-robust-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Cheapest rigorous threshold-robustness check for Contribution 2: ResNet-18
# only, CIFAR-10 only, 4 activations x 3 seeds (12 runs), logging
# active_frac at 5 thresholds plus the raw (threshold-free) gate-magnitude
# distribution (mean/median/std/quantiles) every epoch. No new architectures,
# no scaling, no pruning -- exactly the smallest experiment that can answer
# "is the directional split a threshold artifact?"
python3 -u -m gradient_gate.run_threshold_robustness \
    --archs resnet18 \
    --activations relu gelu silu mish \
    --datasets cifar10 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Job complete at $(date)"
