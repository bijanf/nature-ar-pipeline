#!/bin/bash
# Poll for any period_contrast parquets and refill manuscript/main.md
# whenever a new one lands. Tracks which partial parquets have been
# observed; re-runs fill_placeholders.py + make pdf when the set changes.
# Exits after the full unified period_contrast.parquet has been processed.
#
# Output: appends to slurm/logs/watch_fill_manuscript.log.

set -euo pipefail
cd /home/fallah/scripts/nature-ar-pipeline

LOG=slurm/logs/watch_fill_manuscript.log
mkdir -p slurm/logs

source /home/fallah/mambaforge/etc/profile.d/conda.sh
conda activate nature-ar-pipeline
export PYTHONNOUSERSITE=1

echo "[$(date)] watcher started — polling every 2 min" >> "$LOG"

PREV_PARTIALS=""
did_full=0

while [[ "$did_full" -eq 0 ]]; do
    CUR_PARTIALS=""
    # Watch both validation partial parquets AND full per-period parquets;
    # fill_placeholders prefers the full file when both are present.
    for s in recent_partial modern_partial presat_partial recent modern presat; do
        if [[ -f data/cache/period_contrast_${s}.parquet ]]; then
            CUR_PARTIALS="$CUR_PARTIALS $s"
        fi
    done
    if [[ "$CUR_PARTIALS" != "$PREV_PARTIALS" && -n "$CUR_PARTIALS" ]]; then
        echo "[$(date)] partial set changed: '$CUR_PARTIALS' — refilling" >> "$LOG"
        python manuscript/fill_placeholders.py --source partial >> "$LOG" 2>&1 || true
        ( cd manuscript && make pdf >> "../$LOG" 2>&1 ) || true
        PREV_PARTIALS="$CUR_PARTIALS"
    fi
    if [[ "$did_full" -eq 0 && -f data/cache/period_contrast.parquet ]]; then
        echo "[$(date)] full parquet landed — filling all fields" >> "$LOG"
        python manuscript/fill_placeholders.py --source full >> "$LOG" 2>&1 || true
        ( cd manuscript && make pdf >> "../$LOG" 2>&1 ) || true
        did_full=1
    fi
    sleep 120
done

echo "[$(date)] full parquet processed; watcher exits" >> "$LOG"
