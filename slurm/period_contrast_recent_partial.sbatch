#!/bin/bash
# Period-contrast on the 8 complete Recent years (2015-2022) we already have
# from cds_recent — validates the production analysis chain end-to-end on
# real ERA5 + PIK tp while cds_recent finishes 2023 and 2024.
#
# Output:
#   data/cache/period_contrast_recent_partial.parquet
#   data/cache/observational_events_recent_partial.parquet
#
# Does NOT conflict with arpc_recent (703733) — different --out-suffix and
# separate ERA5_LOCAL_CACHE (only-complete-years symlink dir).
#SBATCH --job-name=arpc_recent_partial
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

export ERA5_LOCAL_CACHE=/home/fallah/scripts/nature-ar-pipeline/data/cache/era5_bbox_recent_partial

python -m src.analysis.period_contrast \
    --periods recent_2015_2024 \
    --out-suffix recent_partial
