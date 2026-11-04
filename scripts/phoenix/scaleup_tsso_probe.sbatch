#!/bin/bash
# Scale-up TSSO probes on Phoenix (development only). One array task per line of $TASKS (1-based line number =
# SLURM_ARRAY_TASK_ID); each line is the argument list of scripts/phoenix/scaleup_tsso_probe.py, e.g.
#   --single outputs/scaleup/instances/tsso/tsso_p40_t30_s1000_g1.json eng --time-limit 300
#   --cert outputs/scaleup/instances/tsso/tsso_p40_t30_s1000_g1.json
# Submit with
#   sbatch --array=1-N --export=ALL,CODE_ROOT=<dir>,TASKS=<file>,OUTDIR=<dir> scripts/phoenix/scaleup_tsso_probe.sbatch
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-tsso-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=00:15:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%A_%a.out

set -euo pipefail
BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
: "${CODE_ROOT:?set CODE_ROOT}" "${TASKS:?set TASKS}" "${OUTDIR:?set OUTDIR}"
cd "$CODE_ROOT"
module load gurobi/13.0.1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$BASE/devenv/bin:$PATH"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID args=$LINE started=$(date -Is)"
eval "python scripts/phoenix/scaleup_tsso_probe.py $LINE --outdir $OUTDIR"
echo "finished=$(date -Is)"
