#!/bin/bash
#SBATCH --job-name=ggc-pruning-mech
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-pruning-mech-%j.out
#SBATCH --error=slurm-pruning-mech-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Experiment 2+3: ResNet-18, CIFAR-10, standard BatchNorm, 4 activations x
# 3 seeds = 12 runs, saving final checkpoints and logging BN-gamma/beta,
# sharpness proxy, and pre/post-activation variance at epochs 0/6/12/18/24.
python3 -u -m gradient_gate.run_pruning_experiment \
    --activations relu gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Training+mechanism-logging complete at $(date)"

# Post-hoc pruning analysis (no retraining, just forward passes) on the
# checkpoints just produced.
python3 -u -m gradient_gate.run_pruning_analysis \
    --activations relu gelu silu mish \
    --seeds 0 1 2

echo "Job complete at $(date)"
