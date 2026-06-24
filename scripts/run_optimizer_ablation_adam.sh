#!/bin/bash
#SBATCH --job-name=ggc-opt-adam
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-opt-adam-%j.out
#SBATCH --error=slurm-opt-adam-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P1: does the activation-class direction split (relu declines; gelu/silu/mish
# rise) survive a different optimizer? Architecture-fixed ablation (resnet18 +
# vgg11, both datasets, 3 seeds, 25 epochs), identical to
# run_activation_ablation.sh except: (a) Adam instead of SGD, lr=1e-3 instead
# of 0.1, (b) relu included explicitly -- there is no existing Adam-trained
# relu baseline to fall back on, unlike the SGD case which reuses
# training_dynamics.csv. New CSV, does not touch any existing file.
python3 -u -m gradient_gate.run_training_dynamics \
    --archs resnet18 vgg11 \
    --activations relu gelu silu mish \
    --datasets cifar10 cifar100 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer adam \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/optimizer_ablation_adam.csv

echo "Job complete at $(date)"
