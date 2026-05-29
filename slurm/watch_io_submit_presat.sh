#!/bin/bash
# Poll the io queue every 2 minutes; submit cds_presat the moment one
# of {cds_recent, cds_modern} finishes (freeing an io QoS slot).
# Exits after the submission lands.

set -euo pipefail
cd /home/fallah/scripts/nature-ar-pipeline

LOG=slurm/logs/watch_io_submit_presat.log
echo "[$(date)] starting watcher" >> "$LOG"

while true; do
    N=$(squeue -u fallah -h -t PD,R -p io | wc -l)
    if [[ "$N" -lt 2 ]]; then
        OUT=$(sbatch --parsable slurm/fetch_cds_presat.sbatch 2>&1) || {
            echo "[$(date)] sbatch presat failed: $OUT" >> "$LOG"
            # may be a transient race; try again on next tick
            sleep 60
            continue
        }
        JID="$OUT"
        echo "[$(date)] submitted cds_presat: $JID" >> "$LOG"
        # Also queue arpc_presat with dep on the cds_presat job
        OUT2=$(sbatch --parsable --dependency=afterok:$JID slurm/period_contrast_presat.sbatch 2>&1) || {
            echo "[$(date)] sbatch arpc_presat failed: $OUT2" >> "$LOG"
            exit 0
        }
        APID="$OUT2"
        echo "[$(date)] queued arpc_presat: $APID" >> "$LOG"
        # Finally, queue concat_and_figures depending on all three arpc_* jobs
        # so it runs as soon as the last-finishing arpc completes.
        # arpc_recent = 707026, arpc_modern = 703774, arpc_presat = $APID.
        OUT3=$(sbatch --dependency=afterok:707026:703774:$APID slurm/concat_and_figures.sbatch 2>&1) || {
            echo "[$(date)] sbatch concat_and_figures failed: $OUT3" >> "$LOG"
            exit 0
        }
        echo "[$(date)] queued concat_and_figures: $OUT3" >> "$LOG"
        exit 0
    fi
    sleep 120
done
