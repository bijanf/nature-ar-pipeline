#!/bin/bash
# Submit the Recent-only period_contrast pipeline against the local CDS
# cache. Use this once `slurm/fetch_cds_recent.sbatch` has finished — it
# validates the local-cache loader, AR detection, event extraction,
# topography aggregation and parquet write on a 10-year window before we
# commit cluster time to the 35-year Modern run.
#
# Usage:
#   bash slurm/submit_recent_pipeline.sh
#
# Produces:
#   data/cache/observational_events_recent.parquet
#   data/cache/period_contrast_recent.parquet
#   data/cache/observational_events.parquet   (unified — Recent only for now)
#   data/cache/period_contrast.parquet        (unified — Recent only for now)
#   figures/fig3_trajectory.pdf               (with Recent slot populated)
#   figures/fig5_landfall_density.pdf         (with Recent panel populated)

set -euo pipefail
cd "$(dirname "$0")/.."

# Sanity-check that the local cache has the full Recent window. Without
# 2015-2024 on disk, the climatology will be computed on whatever years
# are present and cached under the recent_2015_2024 key — that would
# poison subsequent runs.
MISSING=()
for Y in 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
    if [ ! -f "data/cache/era5_bbox_local/${Y}_12_pl.nc" ]; then
        MISSING+=("$Y")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ABORT: missing years from local cache: ${MISSING[*]}"
    echo "       run slurm/fetch_cds_recent.sbatch first."
    exit 1
fi

J1=$(sbatch --parsable slurm/period_contrast_recent.sbatch)
J2=$(sbatch --parsable --dependency=afterok:"$J1" \
            --export="ALL,CONCAT_SUFFIXES=recent" \
            slurm/concat_and_figures_recent.sbatch)

echo "submitted:"
echo "  recent : $J1"
echo "  concat : $J2  (afterok of recent)"
