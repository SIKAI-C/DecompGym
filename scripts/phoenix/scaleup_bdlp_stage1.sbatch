#!/bin/bash
# UFL / stochastic CFL (XL) Stage 1, holdout, LP* reference and instance-generation tasks (protocol v4 sections 6, 14-16).
# One array task per line of $TASKS: the arguments of scripts/phoenix/scaleup_bdlp_stage1.py, e.g.
#   --task ufl outputs/scaleup/instances/ufl/kgs4000b_g11.json eng
#   --reference scfl outputs/scaleup/instances/scfl/cap114_s5000_g11.json
#   --gen scfl outputs/scaleup/instances/scfl --raw <OR-Library CAP dir>
# Each arm runs in its own supervised process group, hard-killed at T; the canonical certificate check (<= 600 s)
# follows. The default allocation (1 h) covers an arm task (T 900 s + view build + checker 600 s + incumbent
# evaluation); reference tasks need --time=03:00:00 and T=3600; generation needs --time=00:30:00.
# Submit (only after the panel freeze and an explicit go), with --no-requeue, in arrays of <= 25 tasks:
#   sbatch --no-requeue --array=0-24 --export=ALL,CODE_ROOT=$BASE/<code dir>,TASKS=<file>,OUT=<dir>,T=900,PANEL=<panel> \
#          scripts/phoenix/scaleup_bdlp_stage1.sbatch
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-bdlp-stage1
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
: "${CODE_ROOT:?set CODE_ROOT}" "${TASKS:?set TASKS}" "${OUT:?set OUT}" "${PANEL:?set PANEL}"
T="${T:-900}"
cd "$CODE_ROOT"
module load gurobi/13.0.1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$BASE/devenv/bin:$PATH"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
[ -n "$LINE" ] || { echo "no task line $SLURM_ARRAY_TASK_ID in $TASKS"; exit 2; }
echo "node=$(hostname) cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') task=$SLURM_ARRAY_TASK_ID args=$LINE T=$T restarts=${SLURM_RESTART_COUNT:-0} started=$(date -Is)"
# shellcheck disable=SC2086
python scripts/phoenix/scaleup_bdlp_stage1.py $LINE --T "$T" --out "$OUT" --panel "$PANEL"
echo "finished=$(date -Is)"
