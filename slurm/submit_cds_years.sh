#!/bin/bash
# Submit chained per-year CDS fetch jobs. Chained because the io QoS caps
# MaxSubmitPU=2 and we want to keep a slot free for ad-hoc work.
#
# Usage:
#   bash slurm/submit_cds_years.sh                 # 2015-2024 (Recent)
#   bash slurm/submit_cds_years.sh 1980 2014       # custom range
#
# The fetcher itself is idempotent — existing YYYY_pl.nc / YYYY_sfc.nc /
# static.nc are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

START=${1:-2015}
END=${2:-2024}

OUT_ROOT=data/cache/era5_bbox_local

PREV=
for Y in $(seq "$START" "$END"); do
    if [ -f "${OUT_ROOT}/${Y}_pl.nc" ] && [ -f "${OUT_ROOT}/${Y}_sfc.nc" ]; then
        echo "skip $Y (already fetched)"
        continue
    fi
    if [ -n "$PREV" ]; then
        DEP=("--dependency=afterany:${PREV}")
    else
        DEP=()
    fi
    JID=$(sbatch --parsable \
        --job-name="cds_${Y}" \
        --export="ALL,YEAR=${Y}" \
        "${DEP[@]}" \
        slurm/fetch_cds_year.sbatch)
    echo "submitted cds_${Y} as $JID"
    PREV=$JID
done
