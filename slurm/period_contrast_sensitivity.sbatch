#!/bin/bash
# Fixed-threshold sensitivity test (manuscript robustness check). Re-runs the
# three observational windows with a UNIFORM 250 kg m-1 s-1 detection threshold
# instead of the period-internal 85th-percentile climatology. If the observed
# intensification were an artefact of threshold drift, a fixed threshold would
# erase it; instead it should preserve the direction of the shift while
# inflating later-window event counts.
#
# Reuses the same local CDS cache; runs after all three windows are downloaded.
# Writes data/cache/period_contrast_fixed250.parquet (and the per-period parts).
#SBATCH --job-name=arpc_fixed250
#SBATCH --partition=standard
# short QOS + right-sized mem (matches the main arpc jobs). The fixed-threshold
# path uses a constant climatology — no full-period open, no climatology compute
# — so it is lighter than the main run; 96G is ample and dodges the medium-QOS
# CPU group limit.
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

THRESH=250

# One chunked pass per window (5-yr chunks keep the materialised cube small).
declare -A WIN=(
    [pre_sat_1940_1959]="1940 1945 1950 1955"
    [modern_1980_1999]="1980 1985 1990 1995"
    [recent_2015_2024]="2015 2020"
)
declare -A LEN=( [2020]=2024 )  # recent's last chunk ends 2024 (2015->2019, 2020->2024)

for PERIOD in pre_sat_1940_1959 modern_1980_1999 recent_2015_2024; do
    SUFFIXES=""
    for Y0 in ${WIN[$PERIOD]}; do
        Y1=${LEN[$Y0]:-$((Y0 + 4))}
        SUFFIX="fx_${PERIOD%%_*}_${Y0}"
        python -m src.analysis.period_contrast \
            --periods "$PERIOD" \
            --time-range "${Y0}-01-01:${Y1}-12-31" \
            --fixed-threshold "$THRESH" \
            --out-suffix "$SUFFIX"
        SUFFIXES="$SUFFIXES $SUFFIX"
    done
    # Concat this window's chunks into one per-window fixed-threshold parquet.
    python -m src.analysis.period_contrast --concat $SUFFIXES \
        --out-suffix "fx_${PERIOD%%_*}"
done

# Final concat: the three fixed-threshold windows into one comparison table.
python -m src.analysis.period_contrast --concat \
    fx_pre fx_modern fx_recent \
    --out-suffix fixed250
