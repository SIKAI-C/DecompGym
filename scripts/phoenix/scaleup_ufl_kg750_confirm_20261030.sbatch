#!/bin/bash
# D1 scale-up, UFL Koerkel-Ghosh confirmation of the candidate tier n = m = 750, classes A and B (2026-10-30):
# strong MIP and engineered Benders reference on the 18 remaining sym/asym A/B instances, the paper's naive
# Benders on 4 of them; 1 hour each on one thread.
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-scaleup-ufl750
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=01:20:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out
#SBATCH --array=0-39
#SBATCH --requeue
# One CPU model for every task: the 32-core cpu-small nodes are excluded, and the model is asserted below.
#SBATCH --exclude=atl1-1-01-005-[21-36]-[1-2],atl1-1-01-002-[22-25]-[1-2]

set -euo pipefail

BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
CODE_ROOT="$BASE/code"
OUTPUT_PATH="$BASE/results/ufl_kg750_confirm_20261030"

cd "$CODE_ROOT"
module load gurobi/13.0.1
module load mamba

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TASKS=(
  "gs750a-2 mip"
  "gs750a-2 ref_xl"
  "gs750a-3 mip"
  "gs750a-3 ref_xl"
  "gs750a-4 mip"
  "gs750a-4 ref_xl"
  "gs750a-5 mip"
  "gs750a-5 ref_xl"
  "gs750b-2 mip"
  "gs750b-2 ref_xl"
  "gs750b-3 mip"
  "gs750b-3 ref_xl"
  "gs750b-4 mip"
  "gs750b-4 ref_xl"
  "gs750b-5 mip"
  "gs750b-5 ref_xl"
  "ga750a-1 mip"
  "ga750a-1 ref_xl"
  "ga750a-2 mip"
  "ga750a-2 ref_xl"
  "ga750a-3 mip"
  "ga750a-3 ref_xl"
  "ga750a-4 mip"
  "ga750a-4 ref_xl"
  "ga750a-5 mip"
  "ga750a-5 ref_xl"
  "ga750b-1 mip"
  "ga750b-1 ref_xl"
  "ga750b-2 mip"
  "ga750b-2 ref_xl"
  "ga750b-3 mip"
  "ga750b-3 ref_xl"
  "ga750b-4 mip"
  "ga750b-4 ref_xl"
  "ga750b-5 mip"
  "ga750b-5 ref_xl"
  "gs750a-2 ref"
  "gs750b-2 ref"
  "ga750a-1 ref"
  "ga750b-1 ref"
)
read -r INSTANCE METHOD <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "job_id=${SLURM_JOB_ID:-unknown} array_task_id=${SLURM_ARRAY_TASK_ID:-none} restart_count=${SLURM_RESTART_COUNT:-0}"
echo "node=$(hostname) instance=$INSTANCE method=$METHOD"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
lscpu | grep -E "Model name|^CPU\(s\)" || true
echo "nproc=$(nproc) affinity=$(taskset -cp $$ 2>/dev/null || true)"
cat /proc/self/cgroup 2>/dev/null | head -3 || true
CPU_MODEL=$(lscpu | sed -n 's/^Model name: *//p')
if [[ "$CPU_MODEL" != "Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz" ]]; then
  echo "abort: unexpected CPU model '$CPU_MODEL' (infrastructure; rerun this task)"; exit 3
fi
# The env's interpreter is called directly so that signal exit codes (137 = SIGKILL) reach this script.
PY=$(mamba run -n pd python -c 'import sys; print(sys.executable)')
GRB_VERSION=$("$PY" -c 'import gurobipy; print(".".join(map(str, gurobipy.gurobi.version())))')
if [[ "$GRB_VERSION" != "13.0.1" ]]; then
  echo "abort: gurobipy $GRB_VERSION, expected 13.0.1"; exit 3
fi
echo "cpu_model=$CPU_MODEL gurobi=$GRB_VERSION"
echo "started=$(date -Is)"

mkdir -p "$OUTPUT_PATH"
ARGS=(--family ufl --data "$BASE/instances/ufl/$INSTANCE/data.json" --method "$METHOD" --time-limit 3600 --out "$OUTPUT_PATH/${INSTANCE}_${METHOD}.json")
RC=0
"$PY" -m src.experiments.scaleup_calibration "${ARGS[@]}" || RC=$?
echo "solver_exit_code=$RC"
# Postmortem: a hard-killed solver (e.g. cgroup OOM) is scored from its journal.
"$PY" -m src.experiments.scaleup_calibration "${ARGS[@]}" --finalize --exit-code "$RC"

echo "finished=$(date -Is)"
