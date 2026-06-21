#!/bin/bash
#SBATCH --job-name=ggc-lowdata-ft
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-lowdata-ft-%j.out
#SBATCH --error=slurm-lowdata-ft-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Task D (pre-registered in RESULTS_LOG.md before any output existed): does
# a checkpoint's already-measured final active_frac predict low-data
# fine-tuning test accuracy? Fine-tunes all 24 existing checkpoints (SGD +
# AdamW, 4 activations, 3 seeds) on a fixed 5% CIFAR-10 subsample, identical
# recipe for every checkpoint.
python3 -u -m gradient_gate.run_lowdata_finetune \
    --epochs 5 \
    --lr 0.01 \
    --batch-size 64 \
    --num-workers 0

echo "Job complete at $(date)"
