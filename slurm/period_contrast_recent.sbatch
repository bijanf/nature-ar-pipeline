#!/bin/bash
# Run period_contrast on the Recent (2015-2024) window. Reads the local
# CDS NetCDF cache via ERA5_LOCAL_CACHE — no internet needed, so this
# moves off the io partition and onto standard (no cgroup wall).
#SBATCH --job-name=arpc_recent
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

export ERA5_LOCAL_CACHE=/home/fallah/scripts/nature-ar-pipeline/data/cache/era5_bbox_local

python -m src.analysis.period_contrast \
    --periods recent_2015_2024 \
    --out-suffix recent
