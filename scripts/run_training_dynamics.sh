#!/bin/bash
#SBATCH --job-name=ggc-training-dynamics
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-train-dyn-%j.out
#SBATCH --error=slurm-train-dyn-%j.err

set -e

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time   : $(date)"
echo "========================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# No conda/module activation on purpose: torch/torchvision/scipy/statsmodels
# are installed under $HOME/.local with the system python3 (3.9), which is
# what produced every result in ROADMAP.md. `module load anaconda; source
# activate <env>` would switch to a DIFFERENT python (3.12/3.13) that does
# not see this user-site install — keep this identical to the interactive
# environment that generated/validated all prior results.
cd $SLURM_SUBMIT_DIR

echo ""
echo "Dataset check (already pre-downloaded to ./data on the login node,"
echo "shared over /home — compute nodes may not have outbound internet):"
ls -la data/

echo ""
echo "========================================"
echo "Running training-dynamics sweep (Phase 4)"
echo "========================================"
python3 -u -m gradient_gate.run_training_dynamics \
    --archs resnet18 resnet50 vgg11 vit_b_16 convnext_tiny \
    --datasets cifar10 cifar100 \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.1 \
    --num-workers 0

echo ""
echo "========================================"
echo "Job complete at $(date)"
echo "Rows written: $(wc -l < gradient_gate_outputs/csv/training_dynamics.csv 2>/dev/null || echo 0)"
echo "========================================"
