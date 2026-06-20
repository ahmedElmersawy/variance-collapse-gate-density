#!/bin/bash
#SBATCH --job-name=ggc-chan-mech
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-chan-mech-%j.out
#SBATCH --error=slurm-chan-mech-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P2: direct, per-channel mechanism verification. Logs, for every channel of
# every elementwise-activation layer in CIFAR-native ResNet-18 (relu/gelu/
# silu/mish, 3 seeds, 25 epochs, standard SGD), per-channel pre-activation
# (mu, sigma) and per-channel active_frac/gate_mean at epochs {0,6,12,18,24}.
# Closes the gap final_neurips_upgrade_report.md explicitly flagged: the
# mu-vs-z_low mechanism was previously verified only at the pooled,
# population level (preactivation_mean_check.csv, 9/9 sign predictions across
# *activations*) -- this tests it at the unit of analysis the mechanism
# actually claims about, the individual channel, across thousands of
# channels per activation.
python3 -u -m gradient_gate.run_channel_mechanism \
    --activations relu gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --num-workers 0

echo "Job complete at $(date)"
