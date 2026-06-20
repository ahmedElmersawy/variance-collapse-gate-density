#!/bin/bash
#SBATCH --job-name=ggc-tin-dyn-a
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-tin-dyn-a-%j.out
#SBATCH --error=slurm-tin-dyn-a-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P3 part A (relu, gelu) -- see run_tinyimagenet_dynamics.sh's header comment
# for the full rationale; split into two jobs across two GPUs since
# Tiny-ImageNet is ~2x the images and 4x the pixels/image of CIFAR, making
# all 4 activations x 3 seeds in one job a risky single wall-clock budget.
python3 -u -m gradient_gate.run_tinyimagenet_dynamics \
    --activations relu gelu \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer sgd \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/tinyimagenet_dynamics_a.csv

echo "Job complete at $(date)"
