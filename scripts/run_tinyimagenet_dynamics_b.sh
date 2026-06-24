#!/bin/bash
#SBATCH --job-name=ggc-tin-dyn-b
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-tin-dyn-b-%j.out
#SBATCH --error=slurm-tin-dyn-b-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# P3 part B (silu, mish) -- companion to run_tinyimagenet_dynamics_a.sh.
# Separate output CSV from part A to avoid a header-write race if both jobs
# happen to start within the same instant (each independently checks
# os.path.exists(out_path) to decide whether to write a CSV header).
python3 -u -m gradient_gate.run_tinyimagenet_dynamics \
    --activations silu mish \
    --seeds 0 1 2 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer sgd \
    --num-workers 0 \
    --out gradient_gate_outputs/csv/tinyimagenet_dynamics_b.csv

echo "Job complete at $(date)"
