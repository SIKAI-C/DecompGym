#!/bin/bash
# Protocol v4 CG Stage 1 / holdout / development tasks (CLRP, CVRP, BPPC; primal-scored, T = 900 s). One array task per
# line of $TASKS ("--task <family> <instance path relative to CODE_ROOT> <arm>", written by scaleup_cg_panel.py).
# Each arm runs in its own supervised process group (xl_supervisor channel), hard-killed at T; evaluation follows.
# Submit (after the Codex-approved freeze; never with --requeue: a requeue would restart the clock silently) with
#   sbatch --no-requeue --array=0-N%25 --export=ALL,CODE_ROOT=$BASE/code_<sha>,TASKS=<file>,OUT=<dir>,PANEL=<panel>,T=900 \
#          scripts/phoenix/scaleup_cg_stage1.sbatch
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-cg-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out
# One CPU model for every task (speed comparisons): the 32-core cpu-small nodes are excluded, the model asserted below.
#SBATCH --exclude=atl1-1-01-005-[21-36]-[1-2],atl1-1-01-002-[22-25]-[1-2]

set -euo pipefail
BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
: "${CODE_ROOT:?set CODE_ROOT}" "${TASKS:?set TASKS}" "${OUT:?set OUT}" "${PANEL:?set PANEL}"
T="${T:-900}"
cd "$CODE_ROOT"
module load gurobi/13.0.1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$BASE/devenv/bin:$PATH"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
[[ -n "$LINE" ]] || { echo "abort: no task line $SLURM_ARRAY_TASK_ID in $TASKS"; exit 3; }
CPU_MODEL=$(lscpu | sed -n 's/^Model name: *//p')
if [[ "$CPU_MODEL" != "Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz" ]]; then
  echo "abort: unexpected CPU model '$CPU_MODEL' (infrastructure; rerun this task)"; exit 3
fi
VERSIONS=$(python -c 'import sys, gurobipy; print(sys.version.split()[0], ".".join(map(str, gurobipy.gurobi.version())))')
echo "numpy/scipy: $(python -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)' 2>&1) (recorded in every result identity)"
if [[ "$VERSIONS" != "3.11.6 13.0.1" ]]; then
  echo "abort: python/gurobipy $VERSIONS, expected 3.11.6 13.0.1 (infrastructure)"; exit 3
fi
echo "node=$(hostname) cpu=$CPU_MODEL versions=$VERSIONS task=$SLURM_ARRAY_TASK_ID args=$LINE T=$T commit=$(git rev-parse HEAD 2>/dev/null || echo none) started=$(date -Is)"
mkdir -p "$OUT"
# shellcheck disable=SC2086
python scripts/phoenix/scaleup_cg_stage1.py $LINE --T "$T" --out "$OUT" --panel "$PANEL"
echo "finished=$(date -Is)"
