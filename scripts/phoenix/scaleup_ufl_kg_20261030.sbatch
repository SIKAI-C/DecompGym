#!/bin/bash
# D1 scale-up calibration, UFL Koerkel-Ghosh (2026-10-30): strong MIP, engineered Benders reference,
# and the paper's (naive) Benders reference, 1 hour each on one thread, on 9 UflLib KG instances.
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-scaleup-ufl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=01:20:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out
#SBATCH --array=0-26
#SBATCH --requeue
# One CPU model for every task: the 32-core cpu-small nodes are excluded, and the model is asserted below.
#SBATCH --exclude=atl1-1-01-005-[21-36]-[1-2],atl1-1-01-002-[22-25]-[1-2]

set -euo pipefail

BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
CODE_ROOT="$BASE/code"
OUTPUT_PATH="$BASE/results/ufl_kg_20261030"

cd "$CODE_ROOT"
module load gurobi/13.0.1
module load mamba

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INSTANCES=(gs250a-1 gs250b-1 gs250c-1 gs500a-1 gs500b-1 gs500c-1 gs750a-1 gs750b-1 gs750c-1)
METHODS=(mip ref_xl ref)
TASK=${SLURM_ARRAY_TASK_ID}
INSTANCE=${INSTANCES[$((TASK / 3))]}
METHOD=${METHODS[$((TASK % 3))]}

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
GRB_VERSION=$(mamba run -n pd python -c 'import gurobipy; print(".".join(map(str, gurobipy.gurobi.version())))')
if [[ "$GRB_VERSION" != "13.0.1" ]]; then
  echo "abort: gurobipy $GRB_VERSION, expected 13.0.1"; exit 3
fi
echo "cpu_model=$CPU_MODEL gurobi=$GRB_VERSION"
echo "started=$(date -Is)"

mkdir -p "$OUTPUT_PATH"
mamba run -n pd python -m src.experiments.scaleup_calibration \
  --family ufl --data "$BASE/instances/ufl/$INSTANCE/data.json" \
  --method "$METHOD" --time-limit 3600 --out "$OUTPUT_PATH/${INSTANCE}_${METHOD}.json"

echo "finished=$(date -Is)"
