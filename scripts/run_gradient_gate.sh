#!/bin/bash
#SBATCH --job-name=gradient-gate-collapse
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-gradient-gate-%j.out
#SBATCH --error=slurm-gradient-gate-%j.err

set -e

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "Start time   : $(date)"
echo "========================================"

# ── Environment ────────────────────────────────────────────────────────────────
module load anaconda
source activate rlft          # reuse your existing env — has numpy/scipy/sklearn

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Output root — writes figures/ csv/ arrays/ here
export GRADIENT_GATE_ROOT_DIR=$HOME/gradient_gate_outputs

# NOTE: the one-time "rm -rf csv/figures/arrays" for the c=0.5->0.0 migration
# (commit 75099ae) has been removed -- that migration already completed
# successfully (job 10779210, 30 valid c=0.0 CSVs on disk). Keeping it would
# discard that run and force a full from-scratch regeneration of everything,
# including the new Phase 3A-D experiments. The pipeline is checkpoint-aware
# (_already_done): it will skip the 30 completed sweeps and run only the new
# Phase 3A/3B/3C/3D + gate-independence + activation-v2 experiments.
mkdir -p $GRADIENT_GATE_ROOT_DIR/csv $GRADIENT_GATE_ROOT_DIR/figures $GRADIENT_GATE_ROOT_DIR/arrays

# ── Copy script to scratch (faster I/O for figure writes) ─────────────────────
SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}/$SLURM_JOB_ID
mkdir -p $SCRATCH
cp $SLURM_SUBMIT_DIR/run_experiments.py $SCRATCH/
cd $SCRATCH

echo "Working dir  : $SCRATCH"
echo "Output dir   : $GRADIENT_GATE_ROOT_DIR"

# ── Dependency check & install ────────────────────────────────────────────────
python - << 'PYEOF'
import importlib, sys, subprocess

required = {
    "numpy":       "numpy",
    "pandas":      "pandas",
    "matplotlib":  "matplotlib",
    "scipy":       "scipy",
    "sklearn":     "scikit-learn",
    "joblib":      "joblib",
}

missing = []
for mod, pkg in required.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"[deps] Installing: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U",
                           "--quiet"] + missing)
else:
    print("[deps] All dependencies already installed.")

# TensorFlow for MNIST (optional — skip gracefully if unavailable)
try:
    import tensorflow
    print(f"[deps] TensorFlow {tensorflow.__version__} available — MNIST enabled.")
except ImportError:
    print("[deps] TensorFlow not found — will run with --skip-mnist.")
PYEOF

# ── Check if TF is available to decide on --skip-mnist flag ───────────────────
TF_FLAG=""
python -c "import tensorflow" 2>/dev/null || TF_FLAG="--skip-mnist"
if [ -n "$TF_FLAG" ]; then
    echo "[info] TensorFlow unavailable — MNIST experiment will be skipped."
fi

# ── Main experiment run ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Starting experiments at $(date)"
echo "Profile: full | Jobs: $SLURM_CPUS_PER_TASK CPUs"
echo "========================================"

python -u run_experiments.py \
    --profile full \
    --jobs $SLURM_CPUS_PER_TASK \
    --root $GRADIENT_GATE_ROOT_DIR \
    $TF_FLAG

# ── Copy outputs back to home ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Copying outputs to $HOME/gradient_gate_outputs"
echo "========================================"
mkdir -p $HOME/gradient_gate_outputs
rsync -av --progress $GRADIENT_GATE_ROOT_DIR/ $HOME/gradient_gate_outputs/

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Job complete at $(date)"
echo "Figures : $(ls $HOME/gradient_gate_outputs/figures/*.png 2>/dev/null | wc -l) PNGs"
echo "CSVs    : $(ls $HOME/gradient_gate_outputs/csv/*.csv    2>/dev/null | wc -l) CSVs"
echo "========================================"
