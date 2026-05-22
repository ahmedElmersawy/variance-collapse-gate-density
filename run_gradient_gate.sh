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
export FIXED_CNN_ROOT_DIR=$HOME/gradient_gate_outputs

# ── Ensure output directories exist ───────────────────────────────────────────
mkdir -p $FIXED_CNN_ROOT_DIR/csv $FIXED_CNN_ROOT_DIR/figures $FIXED_CNN_ROOT_DIR/arrays

# Delete only CSVs that depend on the IoU metric (output_iou = iou(x, binary) fix).
# Keep: stable_rank, effective_rank, nullspace, depth_scaling, eigenvalue_spectra,
#        convergence_rate, twolayer (will regenerate), optimizer_statistics.
CSV=$FIXED_CNN_ROOT_DIR/csv
for f in alpha_sweep_results threshold_sweep_results kernel_sweep_results \
          target_sweep_results oracle_ablation activation_comparison \
          noise_robustness grad_sparsity_all_optimizers phase_diagram_alpha_x_kernel \
          curvature_alpha_sweep scale_experiment twolayer_vs_onelayer \
          gradient_leakage_dense gradient_leakage \
          learned_weights learned_weights_wilcoxon \
          phase_transition_fit fisher_cramer_rao finite_size_scaling \
          per_kernel_alpha_star mutual_information_proxy; do
    rm -f $CSV/${f}.csv
done
echo "Deleted metric-dependent CSVs. Preserved rank/depth/convergence results."

# ── Copy script to scratch (faster I/O for figure writes) ─────────────────────
SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/$USER}/$SLURM_JOB_ID
mkdir -p $SCRATCH
cp $SLURM_SUBMIT_DIR/run_experiments.py $SCRATCH/
cd $SCRATCH

echo "Working dir  : $SCRATCH"
echo "Output dir   : $FIXED_CNN_ROOT_DIR"

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
    --root $FIXED_CNN_ROOT_DIR \
    $TF_FLAG

# ── Copy outputs back to home ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Copying outputs to $HOME/gradient_gate_outputs"
echo "========================================"
mkdir -p $HOME/gradient_gate_outputs
rsync -av --progress $FIXED_CNN_ROOT_DIR/ $HOME/gradient_gate_outputs/

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Job complete at $(date)"
echo "Figures : $(ls $HOME/gradient_gate_outputs/figures/*.png 2>/dev/null | wc -l) PNGs"
echo "CSVs    : $(ls $HOME/gradient_gate_outputs/csv/*.csv    2>/dev/null | wc -l) CSVs"
echo "========================================"
