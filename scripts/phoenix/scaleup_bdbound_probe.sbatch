#!/bin/bash
# Bound-side probes of the remaining Benders scale-up candidates (UFL, FCTP, stochastic CFL; protocol v4; development
# only, not calibration data). One array task per line of $TASKS: "<family> <instance> <arm1,arm2,...>" (arms run one
# after the other, each supervised, 1 CPU, hard kill at T); results in $OUT. Submit with
#   sbatch --array=0-N --time=HH:MM:SS --export=ALL,CODE_ROOT=$BASE/dev_bdbound,TASKS=<file>,OUT=<dir>,T=300 \
#          scripts/phoenix/scaleup_bdbound_probe.sbatch
# Reference jobs (LP* intervals, NOT arms) use MODE=reference and may get more memory (--mem) and time; they are
# labeled "reference" in their outputs.
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-bdbound
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
T="${T:-300}"
MODE="${MODE:-task}"
cd "$CODE_ROOT"
module load gurobi/13.0.1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$BASE/devenv/bin:$PATH"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
read -r FAM INST ARMS <<< "$LINE"
echo "node=$(hostname) cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//') task=$SLURM_ARRAY_TASK_ID mode=$MODE line=$LINE T=$T restarts=${SLURM_RESTART_COUNT:-0} started=$(date -Is)"
if [ "$MODE" = "reference" ]; then
  python scripts/phoenix/scaleup_bdbound_probe.py --reference "$FAM" "$INST" --T "$T" --out "$OUT"
else
  python scripts/phoenix/scaleup_bdbound_probe.py --task "$FAM" "$INST" "$ARMS" --T "$T" --out "$OUT"
fi
echo "finished=$(date -Is)"
