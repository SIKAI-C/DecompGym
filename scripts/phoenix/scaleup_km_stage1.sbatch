#!/bin/bash
# k-median (XL) Stage 1 / reference tasks (protocol v4). One array task per line of $TASKS (arguments for
# scripts/phoenix/scaleup_km_stage1.py, e.g. "--task <instance.json> eng" or "--reference <instance.json>").
# Each arm runs in its own supervised process group, hard-killed at T; the certificate check (<= 600 s) follows.
# Submit with
#   sbatch --array=0-N --export=ALL,CODE_ROOT=$BASE/<code dir>,TASKS=<file>,OUT=<dir>,T=900 scripts/phoenix/scaleup_km_stage1.sbatch
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-km-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=01:00:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out

set -euo pipefail
BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
: "${CODE_ROOT:?set CODE_ROOT}" "${TASKS:?set TASKS}" "${OUT:?set OUT}"
T="${T:-900}"
cd "$CODE_ROOT"
module load gurobi/13.0.1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$BASE/devenv/bin:$PATH"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
echo "node=$(hostname) cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') task=$SLURM_ARRAY_TASK_ID args=$LINE T=$T started=$(date -Is)"
# shellcheck disable=SC2086
python scripts/phoenix/scaleup_km_stage1.py $LINE --T "$T" --out "$OUT"
echo "finished=$(date -Is)"
