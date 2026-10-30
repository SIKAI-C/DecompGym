#!/bin/bash
# Scale-up calibration (D1, 2026-10-28): monolithic MIP vs. the paper's decomposition reference,
# 1 hour each, one thread, on 9 XL instances (UFL Euclidean xl, UFL Koerkel-Ghosh xl_kg, GAP Chu-Beasley D xl).
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-scaleup-calib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=01:30:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out
#SBATCH --array=0-17
#SBATCH --requeue

set -euo pipefail

CODE_ROOT="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/code"
ARTIFACT_ROOT="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
OUTPUT_PATH="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/results/calib_20261028"

cd "$CODE_ROOT"

module load gurobi/13.0.1
module load mamba

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

INSTANCES=(ufl_xl_seed000 ufl_xl_seed001 ufl_xl_seed002 ufl_xlkg_seed000 ufl_xlkg_seed001 ufl_xlkg_seed002 gap_xl_seed000 gap_xl_seed001 gap_xl_seed002)
METHODS=(mip ref)
TASK=${SLURM_ARRAY_TASK_ID}
INSTANCE=${INSTANCES[$((TASK / 2))]}
METHOD=${METHODS[$((TASK % 2))]}
FAMILY=${INSTANCE%%_*}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "node=$(hostname)"
echo "account=gts-ae58-ups"
echo "partition=cpu-small"
echo "qos=embers"
echo "instance=$INSTANCE method=$METHOD family=$FAMILY"
echo "output_path=$OUTPUT_PATH"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "git_status_begin"
git status --short 2>/dev/null || true
echo "git_status_end"
echo "submodules_begin"
git submodule status --recursive 2>/dev/null || true
echo "submodules_end"
lscpu | grep -E "Model name|^CPU\(s\)" || true
echo "started=$(date -Is)"

mkdir -p "$OUTPUT_PATH"
mamba run -n pd python -m src.experiments.scaleup_calibration \
  --family "$FAMILY" --data "$ARTIFACT_ROOT/instances/$INSTANCE/data.json" \
  --method "$METHOD" --time-limit 3600 --out "$OUTPUT_PATH/${INSTANCE}_${METHOD}.json"

echo "finished=$(date -Is)"
