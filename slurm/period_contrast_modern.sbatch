#!/bin/bash
# Run period_contrast on the Modern (1980-1999) window in 4 × 5-yr chunks.
# The climatology is computed once over the full period (cached in
# data/cache/ivt_climatology_*.zarr); each --time-range pass then
# materialises a 5-yr cube only — peak ~64 GB observed in the 8-yr
# recent_partial validation — well inside the 192 GB cap.
#
# Outputs:
#   data/cache/observational_events_modern_198X.parquet
#   ... (one per chunk)
# Final concat to data/cache/observational_events.parquet is done by
# concat_and_figures.sbatch.
#SBATCH --job-name=arpc_modern
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

# Climatology will land on the first --time-range pass and be reused
# for the remainder. Each chunk writes its own per-suffix parquet.
for RANGE in "1980-01-01:1984-12-31" \
             "1985-01-01:1989-12-31" \
             "1990-01-01:1994-12-31" \
             "1995-01-01:1999-12-31"; do
    SUFFIX="modern_${RANGE:0:4}"
    python -m src.analysis.period_contrast \
        --periods modern_1980_1999 \
        --time-range "$RANGE" \
        --out-suffix "$SUFFIX"
done

# Concat the 4 chunks into a single Modern parquet ready for the final
# all-period concat (observational_events_modern.parquet).
python -m src.analysis.period_contrast --concat \
    modern_1980 modern_1985 modern_1990 modern_1995 \
    --out-suffix modern
