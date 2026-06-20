#!/bin/bash
#SBATCH --job-name=ggc-opt-adamw
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-opt-adamw-%j.out
#SBATCH --error=slurm-opt-adamw-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P1, AdamW variant: the paper's mechanism (theory_variance_compression_mechanism.md)
# is tied explicitly to weight-decay dynamics on scale-invariant conv+BN layers
# (van Laarhoven 2017 / Hoffer et al. 2018), which assumes COUPLED weight decay
# (as in SGD and torch.optim.Adam). AdamW's decoupled weight decay is the most
# direct test of whether the BatchNorm-gamma-shrinkage step of the mechanism is
# specific to that coupling. Same architecture-fixed ablation design as the
# Adam job, new CSV.
python3 -u -m gradient_gate.run_training_dynamics \
    --archs resnet18 vgg11 \
    --activations relu gelu silu mish \
    --datasets cifar10 cifar100 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer adamw \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/optimizer_ablation_adamw.csv

echo "Architecture-fixed ablation complete at $(date)"

# Direct gamma-shrinkage check under AdamW specifically (the part of the
# mechanism most likely to break under decoupled weight decay). Reuses
# run_pruning_experiment.py's existing BN-gamma logging machinery, new
# --optimizer flag, separate checkpoint dir and output CSVs so the existing
# SGD checkpoints/mechanism_logging.csv (used by the already-published
# pruning result) are never touched.
python3 -u -m gradient_gate.run_pruning_experiment \
    --activations relu gelu silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer adamw \
    --num-workers 0 \
    --mech-out gradient_gate_outputs/csv/mechanism_logging_adamw.csv \
    --layer-out gradient_gate_outputs/csv/bn_gamma_layerwise_adamw.csv \
    --ckpt-dir gradient_gate_outputs/checkpoints_adamw

echo "Job complete at $(date)"
