#!/bin/bash
# Fetch hourly total_precipitation for the full Recent+Modern+Pre-sat
# record. Stage 2 cascade target. tp is fetched separately because its
# native cadence is hourly (24×/day) while pl/sfc are 6-hourly (4×/day);
# keeping it in distinct {YYYY}_{MM}_tp.nc files avoids re-fetching the
# pl/sfc data and lets the local-cache loader resample tp to 6h on read.
#SBATCH --job-name=cds_tp
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

# Recent + Modern + Pre-sat. The fetcher is idempotent; existing _tp.nc
# files are skipped. Order matters for partial-progress utility: do
# Recent first (smallest, needed for Stage 2 holdout), then Modern
# (training), then Pre-sat (observed-period analysis only).
for Y in $(seq 2015 2024) $(seq 1980 2014) $(seq 1940 1979); do
    python -m src.data.fetch_cds_era5 --year "$Y" --kinds tp --concurrency 4
done
