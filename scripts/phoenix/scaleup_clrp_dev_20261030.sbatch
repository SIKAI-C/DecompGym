#!/bin/bash
# D1 scale-up calibration, CLRP (Prodhon) development screen (2026-10-30): compact MIP, branch-and-price reference and
# heuristic control, 1 hour each on one thread, on the 18 development instances (variants 1-2; variant 3 is held out).
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-scaleup-clrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=01:20:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out
#SBATCH --array=0-53
#SBATCH --requeue
# One CPU model for every task: the 32-core cpu-small nodes are excluded, and the model is asserted below.
#SBATCH --exclude=atl1-1-01-005-[21-36]-[1-2],atl1-1-01-002-[22-25]-[1-2]

set -euo pipefail

BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
CODE_ROOT="$BASE/code"
OUTPUT_PATH="$BASE/results/clrp_dev_20261030"

cd "$CODE_ROOT"
module load gurobi/13.0.1
module load mamba

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TASKS=(
  "50-5-1 mip"
  "50-5-1 ref_xl"
  "50-5-1 ls"
  "50-5-1b mip"
  "50-5-1b ref_xl"
  "50-5-1b ls"
  "50-5-2 mip"
  "50-5-2 ref_xl"
  "50-5-2 ls"
  "50-5-2b mip"
  "50-5-2b ref_xl"
  "50-5-2b ls"
  "50-5-2BIS mip"
  "50-5-2BIS ref_xl"
  "50-5-2BIS ls"
  "50-5-2bBIS mip"
  "50-5-2bBIS ref_xl"
  "50-5-2bBIS ls"
  "100-5-1 mip"
  "100-5-1 ref_xl"
  "100-5-1 ls"
  "100-5-1b mip"
  "100-5-1b ref_xl"
  "100-5-1b ls"
  "100-5-2 mip"
  "100-5-2 ref_xl"
  "100-5-2 ls"
  "100-5-2b mip"
  "100-5-2b ref_xl"
  "100-5-2b ls"
  "100-10-1 mip"
  "100-10-1 ref_xl"
  "100-10-1 ls"
  "100-10-1b mip"
  "100-10-1b ref_xl"
  "100-10-1b ls"
  "100-10-2 mip"
  "100-10-2 ref_xl"
  "100-10-2 ls"
  "100-10-2b mip"
  "100-10-2b ref_xl"
  "100-10-2b ls"
  "200-10-1 mip"
  "200-10-1 ref_xl"
  "200-10-1 ls"
  "200-10-1b mip"
  "200-10-1b ref_xl"
  "200-10-1b ls"
  "200-10-2 mip"
  "200-10-2 ref_xl"
  "200-10-2 ls"
  "200-10-2b mip"
  "200-10-2b ref_xl"
  "200-10-2b ls"
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
ARGS=(--family clrp --data "$BASE/instances/clrp/coord${INSTANCE}.dat" --method "$METHOD" --time-limit 3600 --out "$OUTPUT_PATH/${INSTANCE}_${METHOD}.json")
RC=0
"$PY" -m src.experiments.scaleup_calibration "${ARGS[@]}" || RC=$?
echo "solver_exit_code=$RC"
# Postmortem: a hard-killed solver (e.g. cgroup OOM) is scored from its journal.
"$PY" -m src.experiments.scaleup_calibration "${ARGS[@]}" --finalize --exit-code "$RC"

echo "finished=$(date -Is)"
