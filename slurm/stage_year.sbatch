#!/bin/bash
# Pre-stage one year of ARCO-ERA5 bbox to local Zarr. Submit with:
#   sbatch --job-name=stage_1995 --export=ALL,YEAR=1995 slurm/stage_year.sbatch
#
# Each year is ~4 GB on disk; the streaming load fits the io-partition's
# 20 GB cap because we only fetch one year of u/v/t/q/z/sp at 6-hourly.
# Subsequent period_contrast jobs can read the per-year zarrs from local
# disk on the standard partition (no internet needed).
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

: "${YEAR:?YEAR env var required (e.g. YEAR=1995)}"

python -m src.data.stage_arco_to_local --year "$YEAR"
