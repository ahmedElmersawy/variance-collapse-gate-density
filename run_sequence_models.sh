#!/bin/bash
#SBATCH --job-name=ggc-seqmodels
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-seqmodels-%j.out
#SBATCH --error=slurm-seqmodels-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P4: is the smooth-activation gate-density rise CNN-specific? New
# CIFAR-native (32x32), activation-configurable MLP-Mixer and small
# Transformer-Encoder (gradient_gate/sequence_models.py) -- both built with
# explicit act_layer() submodules (not nn.TransformerEncoderLayer's
# functional activation, which GateInstrumentor cannot see) so the existing
# hook-based gate recovery applies unchanged. relu vs gelu/silu/mish, both
# datasets, 3 seeds, 25 epochs -- same design as the CNN architecture-fixed
# ablation (run_activation_ablation.sh), new architecture family instead of
# new architectures within the CNN family.
python3 -u -m gradient_gate.run_training_dynamics \
    --archs mlp_mixer transformer_encoder \
    --activations relu gelu silu mish \
    --datasets cifar10 cifar100 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer sgd \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/sequence_model_ablation.csv

echo "Job complete at $(date)"
