#!/bin/bash
#SBATCH --job-name=ggc-chan-mech-adamw
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-chan-mech-adamw-%j.out
#SBATCH --error=slurm-chan-mech-adamw-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Task A (NeurIPS-8 revision): per-channel mu, sigma, active_frac under
# AdamW -- the data needed to test whether the EXACT, unmodified
# sigma-normalized z-score predictor from the SGD per-channel mechanism
# result (analyze_channel_mechanism.py), fed AdamW's real measured
# trajectories, predicts the observed AdamW outcome (all four activations
# declining) instead of the SGD/Adam outcome (relu declines, smooth
# activations rise). Same protocol as the existing SGD channel_mechanism.csv
# (resnet18, cifar10, relu/gelu/silu/mish, 3 seeds, 25 epochs, MECH_EPOCHS
# {0,6,12,18,24}) so the analysis is a drop-in comparison, not a redefined
# test. New output file -- does not touch the existing SGD data.
python3 -u -m gradient_gate.run_channel_mechanism \
    --activations relu gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer adamw \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/channel_mechanism_adamw.csv \
    --zlow-out gradient_gate_outputs/csv/channel_mechanism_adamw_zlow.csv

echo "Job complete at $(date)"
