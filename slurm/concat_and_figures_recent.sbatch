#!/bin/bash
# Concat the Recent-only period_contrast output into the unified parquets
# and regenerate fig3 + fig5. Used by submit_recent_pipeline.sh as a
# first end-to-end validation before fetching Modern + Pre-sat.
#SBATCH --job-name=arpc_concat_recent
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

python -m src.analysis.period_contrast --concat recent
python -m src.figures.fig3_trajectory
python -m src.figures.fig5_landfall_density
