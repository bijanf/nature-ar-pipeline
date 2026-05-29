#!/bin/bash
# Concatenate per-period parquets into unified outputs, then regenerate
# the trajectory + landfall-density figures. Pure local work; can run on
# the standard partition (no internet needed).
#SBATCH --job-name=arpc_concat
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

python -m src.analysis.period_contrast --concat presat modern recent
# ERA5-only manuscript: observational windows only, no synthetic CMIP6 half.
python -m src.figures.fig3_trajectory --observed-only
python -m src.figures.fig5_landfall_density --observed-only
