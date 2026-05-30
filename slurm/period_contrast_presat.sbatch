#!/bin/bash
# Run period_contrast on the Pre-satellite (1940-1959) window in 4 × 5-yr
# chunks. Reads the local CDS NetCDF cache via ERA5_LOCAL_CACHE. Chunking
# keeps each materialised cube ~5-yr (the 8-yr recent validation peaked
# ~98 GB), well inside the 192 GB cap; the period-internal climatology is
# computed over the full 1940-1959 window on the first chunk and cached.
#SBATCH --job-name=arpc_presat
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

export ERA5_LOCAL_CACHE=/p/projects/poem/fallah/nature_ar_data/era5_bbox_local

for RANGE in "1940-01-01:1944-12-31" \
             "1945-01-01:1949-12-31" \
             "1950-01-01:1954-12-31" \
             "1955-01-01:1959-12-31"; do
    SUFFIX="presat_${RANGE:0:4}"
    python -m src.analysis.period_contrast \
        --periods pre_sat_1940_1959 \
        --time-range "$RANGE" \
        --out-suffix "$SUFFIX"
done

# Concat the 4 chunks into a single Pre-sat parquet ready for the final
# all-period concat (observational_events_presat.parquet).
python -m src.analysis.period_contrast --concat \
    presat_1940 presat_1945 presat_1950 presat_1955 \
    --out-suffix presat
