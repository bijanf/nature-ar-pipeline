#!/bin/bash
# Master orchestrator for the symmetric-window run (Pre-sat 1940-1959,
# Modern 1980-1999, Recent 2015-2024 [done]). Disk-driven and timeout-proof:
# it does NOT rely on SLURM afterok deps (the cds_* jobs hit walltime, which
# is not afterok and would stall the chain). Instead it polls the local CDS
# cache for the months each window needs and submits the next job the moment
# the data is on disk. One-shot submissions are guarded by sentinel files.
#
# Pipeline:
#   1. cds_modern times out -> submit cds_modern_finish (1994-1999, conc 8).
#   2. Pre-sat 1940-1959 complete on disk -> cancel the still-running
#      cds_presat (stop wasting CDS quota on unneeded 1960+), submit arpc_presat.
#   3. Modern 1980-1999 complete on disk -> submit arpc_modern.
#   4. All three period_contrast_{presat,modern,recent}.parquet present
#      -> submit concat_and_figures, then exit.
#
# Log: slurm/logs/orchestrate_completion.log
set -uo pipefail
cd /home/fallah/scripts/nature-ar-pipeline

LOG=slurm/logs/orchestrate_completion.log
SENT=slurm/logs/.sentinels
mkdir -p "$SENT"
echo "[$(date)] orchestrator started" >> "$LOG"

CACHE=data/cache/era5_bbox_local
PC=data/cache

PRESAT_YEARS=$(seq 1940 1959)
MODERN_YEARS=$(seq 1980 1999)

count_pl () {  # $1 = space-separated years -> count of *_pl.nc present AND non-empty
    # Size gate (>=4096 B): a 0-byte / truncated file from a failed download
    # must NOT count as complete, or we fire the analysis on a corrupt window
    # ("NetCDF: Unknown file format"). Matches fetch_cds_era5._MIN_VALID_BYTES.
    local n=0 y m f sz
    for y in $1; do
        for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
            f="$CACHE/${y}_${m}_pl.nc"
            [[ -f "$f" ]] || continue
            sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
            [[ "$sz" -ge 4096 ]] && n=$((n+1))
        done
    done
    echo "$n"
}

io_slots_used () { squeue -u fallah -h -t PD,R -p io | wc -l; }
job_in_queue () { squeue -j "$1" -h -o "%i" 2>/dev/null | grep -q .; }

while true; do
    # --- Step 1: resume Modern download once cds_modern leaves the queue ----
    if [[ ! -f "$SENT/modern_finish" ]]; then
        if ! job_in_queue 686013; then
            if [[ "$(io_slots_used)" -lt 2 ]]; then
                OUT=$(sbatch --parsable slurm/fetch_cds_modern_finish.sbatch 2>&1) \
                    && { echo "[$(date)] submitted cds_modern_finish: $OUT" >> "$LOG"; touch "$SENT/modern_finish"; } \
                    || echo "[$(date)] modern_finish sbatch failed: $OUT" >> "$LOG"
            fi
        fi
    fi

    # --- Step 2: Pre-sat data complete -> stop cds_presat, run arpc_presat --
    if [[ ! -f "$SENT/arpc_presat" && "$(count_pl "$PRESAT_YEARS")" -ge 240 ]]; then
        if job_in_queue 710290; then
            scancel 710290 2>/dev/null && echo "[$(date)] presat data complete; cancelled cds_presat 710290" >> "$LOG"
        fi
        OUT=$(sbatch --parsable slurm/period_contrast_presat.sbatch 2>&1) \
            && { echo "[$(date)] submitted arpc_presat: $OUT" >> "$LOG"; touch "$SENT/arpc_presat"; } \
            || echo "[$(date)] arpc_presat sbatch failed: $OUT" >> "$LOG"
    fi

    # --- Step 3: Modern data complete -> run arpc_modern --------------------
    if [[ ! -f "$SENT/arpc_modern" && "$(count_pl "$MODERN_YEARS")" -ge 240 ]]; then
        OUT=$(sbatch --parsable slurm/period_contrast_modern.sbatch 2>&1) \
            && { echo "[$(date)] submitted arpc_modern: $OUT" >> "$LOG"; touch "$SENT/arpc_modern"; } \
            || echo "[$(date)] arpc_modern sbatch failed: $OUT" >> "$LOG"
    fi

    # --- Step 4: all three per-period parquets present -> concat + figures --
    if [[ ! -f "$SENT/concat" \
          && -f "$PC/period_contrast_presat.parquet" \
          && -f "$PC/period_contrast_modern.parquet" \
          && -f "$PC/period_contrast_recent.parquet" ]]; then
        OUT=$(sbatch --parsable slurm/concat_and_figures.sbatch 2>&1) \
            && { echo "[$(date)] submitted concat_and_figures: $OUT" >> "$LOG"; touch "$SENT/concat"; } \
            || echo "[$(date)] concat sbatch failed: $OUT" >> "$LOG"
        echo "[$(date)] orchestrator done; exiting" >> "$LOG"
        exit 0
    fi

    sleep 120
done
