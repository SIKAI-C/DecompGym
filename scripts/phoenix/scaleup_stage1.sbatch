#!/bin/bash
# D1 scale-up Stage 1 template (protocol v3.1): one task per line of $TASKS_FILE (family, instance, method, time limit,
# output directory). Same runner, guards and resources as the approved calibration jobs.
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-scaleup-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=01:20:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out
# Submit with: sbatch --array=0-<N-1> --export=ALL,TASKS_FILE=scripts/phoenix/tasks/<file> (N <= 50, Phoenix embers MaxSubmitJobs)
#SBATCH --requeue
# One CPU model for every task: the 32-core cpu-small nodes are excluded, and the model is asserted below.
#SBATCH --exclude=atl1-1-01-005-[21-36]-[1-2],atl1-1-01-002-[22-25]-[1-2]

set -euo pipefail

BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
CODE_ROOT="$BASE/code"

cd "$CODE_ROOT"
module load gurobi/13.0.1
module load mamba

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Task list: one line per array index, "family instance_path method time_limit_seconds output_dir";
# instance_path is relative to $BASE/instances, output_dir relative to $BASE/results.
: "${TASKS_FILE:?set TASKS_FILE (path relative to the code checkout) via sbatch --export=ALL,TASKS_FILE=...}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_FILE")
read -r FAMILY INSTANCE METHOD TLIMIT OUTDIR <<< "$LINE"
[[ -n "$OUTDIR" ]] || { echo "abort: no task line $SLURM_ARRAY_TASK_ID in $TASKS_FILE"; exit 3; }
OUTPUT_PATH="$BASE/results/$OUTDIR"

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
NAME=$(basename "$INSTANCE"); NAME=${NAME%.*}; NAME=${NAME#coord}
ARGS=(--family "$FAMILY" --data "$BASE/instances/$INSTANCE" --method "$METHOD" --time-limit "$TLIMIT" --out "$OUTPUT_PATH/${NAME}_${METHOD}.json")
RC=0
"$PY" -m src.experiments.scaleup_calibration "${ARGS[@]}" || RC=$?
echo "solver_exit_code=$RC"
# Postmortem: a hard-killed solver (e.g. cgroup OOM) is scored from its journal.
"$PY" -m src.experiments.scaleup_calibration "${ARGS[@]}" --finalize --exit-code "$RC"

echo "finished=$(date -Is)"
