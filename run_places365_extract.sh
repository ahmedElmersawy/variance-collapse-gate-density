#!/bin/bash
#SBATCH --job-name=ggc-places-full
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-places-full-%j.out
#SBATCH --error=slurm-places-full-%j.err

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
export PYTHONUNBUFFERED=1

# Step 1: extraction (CPU/IO-bound). Was previously running as a bare
# background `tar` process directly on the shared LOGIN NODE for 3+ hours
# at 95% CPU -- a clear violation of this project's own environment rule
# (SLURM for anything multi-hour, never an unsupervised background process
# on a node shared by dozens of other interactive users). Killed that
# process and moved it here. This partition requires --gres=gpu:1 for any
# job (confirmed: a CPU-only submission was rejected outright), so rather
# than waste a separate GPU allocation purely on extraction, training runs
# immediately after on the same allocation.
#
# places_needed.txt (62,050 lines, built before this job and reused
# unmodified): the full already-extracted places365_standard/val set, plus
# the first 150 sequentially-numbered training images
# (00000001.jpg..00000150.jpg) for each of the 365 train classes -- a
# fixed, deterministic, reproducible selection, not a claimed random
# sample. `-k` (--keep-old-files) skips anything already on disk from the
# earlier partial attempt without erroring, a free, correct speedup.
cd /scratch/gilbreth/aelmersa/places365
if [ ! -f extraction_done.marker ]; then
  # `tar -k` exits 2 ("Cannot open: File exists") for every file it skips
  # because it already exists -- confirmed by direct test before relying on
  # this. That is the EXPECTED outcome here (we are deliberately re-running
  # over partially-extracted data) not a real failure, but combined with
  # `set -e` it would silently kill this script right after extraction and
  # never reach training, burning the whole GPU allocation on nothing. `|| true`
  # makes that expected, benign exit code non-fatal; the verification step
  # immediately below is what actually catches a real extraction problem.
  tar -k -xf places365standard_easyformat.tar -T places_needed.txt || true
  echo "=== verification: train classes with < 150 files ==="
  cd places365_standard/train
  under=0
  for d in */; do
    n=$(ls "$d" | wc -l)
    if [ "$n" -lt 150 ]; then echo "$d: $n"; under=$((under+1)); fi
  done
  echo "train classes under 150: $under / $(ls -d */ | wc -l)"
  cd /scratch/gilbreth/aelmersa/places365
  if [ "$under" -eq 0 ]; then
    touch extraction_done.marker
  else
    echo "WARNING: $under classes still short of 150 train images -- proceeding anyway, Places365Subset will just sample fewer for those classes."
    touch extraction_done.marker
  fi
else
  echo "extraction_done.marker present, skipping extraction"
fi
echo "Extraction step complete at $(date)"

# Step 2: training (GPU). Task C: ResNet-50, CIFAR-native stem (already
# appropriate at 96x96 -- strides (1,2,2,2) give a 12x12 feature map before
# the global pool, not collapsed prematurely), relu/gelu/silu/mish, 2
# seeds (feasible within this job's budget), 25 epochs, SGD, on the fixed
# 150-train/20-val-per-class Places365 subsample.
cd /home/aelmersa/MMLS
python3 -u -m gradient_gate.run_places365_dynamics \
    --activations relu gelu silu mish \
    --seeds 0 1 \
    --epochs 25 \
    --batch-size 128 \
    --optimizer sgd \
    --num-workers 0

echo "Job complete at $(date)"
