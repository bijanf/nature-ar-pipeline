#!/bin/bash
# When cds_modern (686013) times out / ends, an io QoS slot frees. Submit
# cds_modern_finish to pick up the remaining Modern years at concurrency 8.
# Then re-wire arpc_modern's dependency: the original arpc_modern (703774)
# depends afterok on cds_modern, which TIMES OUT (not afterok) and so will
# never fire. Cancel it and resubmit depending on the finish job instead.
# Also re-point arpc_concat (710292) onto the new arpc_modern jobid.
# Exits after the resubmission lands.

set -euo pipefail
cd /home/fallah/scripts/nature-ar-pipeline

LOG=slurm/logs/watch_io_submit_modern_finish.log
echo "[$(date)] starting watcher" >> "$LOG"

OLD_MODERN_JOB=686013
OLD_ARPC_MODERN=703774
OLD_ARPC_CONCAT=710292
ARPC_RECENT=707026   # already COMPLETED; afterok is satisfied instantly
ARPC_PRESAT=710291

while true; do
    # cds_modern gone from the queue (timed out or completed)?
    if ! squeue -j "$OLD_MODERN_JOB" -h -o "%i" 2>/dev/null | grep -q .; then
        # Need a free io slot (MaxSubmitPU=2). Wait until < 2 io jobs.
        N=$(squeue -u fallah -h -t PD,R -p io | wc -l)
        if [[ "$N" -lt 2 ]]; then
            OUT=$(sbatch --parsable slurm/fetch_cds_modern_finish.sbatch 2>&1) || {
                echo "[$(date)] sbatch modern_finish failed: $OUT" >> "$LOG"; sleep 60; continue; }
            FIN="$OUT"
            echo "[$(date)] submitted cds_modern_finish: $FIN" >> "$LOG"

            # Re-wire arpc_modern onto the finish job.
            scancel "$OLD_ARPC_MODERN" 2>/dev/null || true
            OUT2=$(sbatch --parsable --dependency=afterok:$FIN slurm/period_contrast_modern.sbatch 2>&1) || {
                echo "[$(date)] resubmit arpc_modern failed: $OUT2" >> "$LOG"; exit 0; }
            NEW_ARPC_MODERN="$OUT2"
            echo "[$(date)] resubmitted arpc_modern: $NEW_ARPC_MODERN (dep afterok:$FIN)" >> "$LOG"

            # Re-point arpc_concat onto the three current arpc jobs.
            scancel "$OLD_ARPC_CONCAT" 2>/dev/null || true
            OUT3=$(sbatch --parsable \
                --dependency=afterok:$ARPC_RECENT:$NEW_ARPC_MODERN:$ARPC_PRESAT \
                slurm/concat_and_figures.sbatch 2>&1) || {
                echo "[$(date)] resubmit arpc_concat failed: $OUT3" >> "$LOG"; exit 0; }
            echo "[$(date)] resubmitted arpc_concat: $OUT3" >> "$LOG"
            exit 0
        fi
    fi
    sleep 120
done
