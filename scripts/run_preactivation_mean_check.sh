#!/bin/bash
#SBATCH --job-name=ggc-preact-mean
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-preact-mean-%j.out
#SBATCH --error=slurm-preact-mean-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Validates the central conjecture in theory_variance_compression_mechanism.md:
# does pre-activation mean drift similarly (in sign/magnitude) across
# activations, so the activation-class split is explained by each
# activation's fixed z_low(theta) location rather than a qualitatively
# different mu-trajectory per activation.
python3 -u -m gradient_gate.run_preactivation_mean_check \
    --activations relu leaky_relu_0.01 softplus_beta50 softplus_beta20 softplus_beta10 softplus_beta5 \
                  gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Job complete at $(date)"
