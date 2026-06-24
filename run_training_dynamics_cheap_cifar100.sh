#!/bin/bash
#SBATCH --job-name=ggc-traindyn-cheap
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-train-dyn-cheap-%j.out
#SBATCH --error=slurm-train-dyn-cheap-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Continuation of job 11039609 (TIMEOUT at 8hr): resnet18/resnet50/vgg11 on
# cifar10 (all 3 seeds, 25 epochs) already completed and checkpointed in
# training_dynamics.csv -- already_done() will skip them. This job only
# needs to add cifar100 for these three cheap, CIFAR-native architectures.
python3 -u -m gradient_gate.run_training_dynamics \
    --archs resnet18 resnet50 vgg11 \
    --datasets cifar100 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo "Job complete at $(date)"
