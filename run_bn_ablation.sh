#!/bin/bash
#SBATCH --job-name=ggc-bn-ablation
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-bn-ablation-%j.out
#SBATCH --error=slurm-bn-ablation-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Experiment 1: BatchNorm necessity ablation. ResNet-18, CIFAR-10 only,
# BatchNorm vs GroupNorm x ReLU/GELU/SiLU/Mish x 3 seeds = 24 runs.
python3 -u -m gradient_gate.run_bn_ablation \
    --norms batchnorm groupnorm \
    --activations relu gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Job complete at $(date)"
