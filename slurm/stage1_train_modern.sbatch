#!/bin/bash
# Stage 1 LightGBM training on the Modern (1980-2014) ERA5 window.
#
# Two-step inside the job:
#   1. era5_train_assembly --period 1980-2014  -> data/cache/train_rows/
#   2. stage1_ar_emulator --cv                -> data/cache/stage1_cv_report.json
#                                                + cached boosters/
#
# Standard partition (no internet needed — reads local CDS NetCDF cache via
# ERA5_LOCAL_CACHE). Submit with `--dependency=afterok:686013` so it
# auto-fires when cds_modern completes.
#SBATCH --job-name=stage1_modern
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

export ERA5_LOCAL_CACHE=/p/projects/poem/fallah/nature_ar_data/era5_bbox_local

# Step 1: assemble training rows from Modern.
python -m src.data.era5_train_assembly --period 1980-2014 --overwrite

# Step 2: 5-fold spatial-block CV.
python -m src.models.stage1_ar_emulator --cv

# Step 3: print the cv report tail for the SLURM log.
cat data/cache/stage1_cv_report.json
