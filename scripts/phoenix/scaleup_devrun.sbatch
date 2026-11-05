#!/bin/bash
# D1 scale-up development check on Phoenix (unit tests or a short probe of our own code), so that no optimization
# runs on the laptop. Not an experiment: results are code checks only. Submit with
#   sbatch --export=ALL,CODE_ROOT=$BASE/code_<sha>,DEV_CMD="python -m pytest -q tests/..." scripts/phoenix/scaleup_devrun.sbatch
#SBATCH --account=gts-ae58-ups
#SBATCH --job-name=dg-scaleup-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --time=00:20:00
#SBATCH --output=/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup/scripts/out/%x-%j.out

set -euo pipefail
BASE="/storage/home/hcoda1/1/scheng326/r-phentenryck3-0/decompgym-scaleup"
: "${CODE_ROOT:?set CODE_ROOT}" "${DEV_CMD:?set DEV_CMD}"
cd "$CODE_ROOT"
module load gurobi/13.0.1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$BASE/devenv/bin:$PATH"
echo "node=$(hostname) commit=$(git rev-parse --short HEAD) cmd=$DEV_CMD started=$(date -Is)"
eval "$DEV_CMD"
echo "finished=$(date -Is)"
