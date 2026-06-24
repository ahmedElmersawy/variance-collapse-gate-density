#!/bin/bash
#SBATCH --job-name=ggc-smoothness-sweep
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-smoothness-%j.out
#SBATCH --error=slurm-smoothness-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Smoothness-mechanism sweep: 13 activations x 3 seeds x 25 epochs on the
# same ResNet-18/CIFAR-10 protocol used throughout. Tests whether activation
# smoothness predicts gate-density trend direction/magnitude in a
# monotonic dose-response, spanning hard-gated (ReLU) through near-ReLU
# (LeakyReLU), learnable (PReLU), a continuous-stiffness sigmoid-derivative
# family (Softplus(beta)), to fully smooth (GELU/SiLU/Mish).
python3 -u -m gradient_gate.run_smoothness_sweep \
    --activations relu leaky_relu_0.001 leaky_relu_0.01 leaky_relu_0.05 leaky_relu_0.10 \
                  prelu softplus_beta50 softplus_beta20 softplus_beta10 softplus_beta5 \
                  gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Job complete at $(date)"
