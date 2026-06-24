#!/bin/bash
#SBATCH --job-name=ggc-privacy
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-privacy-%j.out
#SBATCH --error=slurm-privacy-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1
cd $SLURM_SUBMIT_DIR

# Re-run from scratch: the earlier interactive attempt only reached
# alpha=0.5/seed=15 before the parent shell died on a session disconnect
# (background bash processes don't survive that the way a submitted batch
# job does) -- this script intentionally has no checkpoint/resume logic, so
# just let it redo all 7 alphas x 20 seeds x 3 attacks (~3hr at the rate
# observed interactively on a contended GPU; should be faster on a
# dedicated A100).
python3 -u -m gradient_gate.run_privacy_experiment \
    --seeds 20 \
    --iters 120 \
    --out gradient_gate_outputs/csv/privacy_gate_collapse.csv \
    --fig gradient_gate_outputs/figures/privacy_gate_collapse.png

echo "Job complete at $(date)"
