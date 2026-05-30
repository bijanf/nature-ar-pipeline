#!/bin/bash
# Run period_contrast on the Pre-satellite (1940-1959) window in 4 × 5-yr
# chunks. Reads the local CDS NetCDF cache via ERA5_LOCAL_CACHE. Chunking
# keeps each materialised cube ~5-yr (the 8-yr recent validation peaked
# ~98 GB), well inside the 192 GB cap; the period-internal climatology is
# computed over the full 1940-1959 window on the first chunk and cached.
#SBATCH --job-name=arpc_presat
#SBATCH --partition=standard
# short QOS (higher CPU group limit, less contended than medium) + right-sized
# memory. With the climatology pre-cached and feats built per 5-yr chunk, peak
# is ~50 GB (validated) so 96G is ample; 192G needlessly forced ~36 CPUs and
# tripped the medium-QOS group CPU limit. Each window finishes in <1 h.
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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
