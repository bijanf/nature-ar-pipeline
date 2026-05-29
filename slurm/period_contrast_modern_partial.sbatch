#!/bin/bash
# Period-contrast on the 7 complete Modern years (1980-1986) we already
# have from cds_modern — gives a real first-cut Modern parquet while
# cds_modern fetches the remaining 28 years (slow CDS queue).
#
# Output:
#   data/cache/period_contrast_modern_partial.parquet
#   data/cache/observational_events_modern_partial.parquet
#
# Does NOT conflict with arpc_modern (703774) — different --out-suffix
# and a separate ERA5_LOCAL_CACHE pointing at only-complete-years symlinks.
#SBATCH --job-name=arpc_modern_partial
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

export ERA5_LOCAL_CACHE=/home/fallah/scripts/nature-ar-pipeline/data/cache/era5_bbox_modern_partial

python -m src.analysis.period_contrast \
    --periods modern_1980_1999 \
    --time-range 1980-01-01:1986-12-31 \
    --out-suffix modern_partial
