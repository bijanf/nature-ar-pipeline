#!/bin/bash
# Submit the next pending decade chunk from slurm/CHUNKS.txt that does not
# already have its output parquet on disk and is not currently in the queue.
# Returns 0 and prints the submitted job id, or returns 1 with a one-line
# explanation if nothing was submitted (all done, or io slot occupied).
#
# Designed to be safe to invoke repeatedly from the overnight monitor loop.

set -euo pipefail
cd "$(dirname "$0")/.."

USER_JOBS=$(squeue -h -u "$USER" -t pending,running --partition=io -o "%i %j" 2>/dev/null)
USER_SLOTS=$(echo "$USER_JOBS" | grep -v '^$' | wc -l)
if [ "$USER_SLOTS" -ge 2 ]; then
    echo "io slots full ($USER_SLOTS/2). Pending in queue:"
    echo "$USER_JOBS" | sed 's/^/  /'
    exit 1
fi

while IFS= read -r line; do
    case "$line" in '#'*|'') continue ;; esac
    set -- $line
    JOB_NAME=$1; PERIOD_NAME=$2; OUT_SUFFIX=$3; TIME_RANGE=$4

    out_parquet="data/cache/observational_events_${OUT_SUFFIX}.parquet"
    if [ -f "$out_parquet" ]; then
        continue
    fi
    if echo "$USER_JOBS" | grep -q "$JOB_NAME"; then
        echo "$JOB_NAME already queued."
        exit 1
    fi

    JID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --export="ALL,PERIOD_NAME=$PERIOD_NAME,OUT_SUFFIX=$OUT_SUFFIX,TIME_RANGE=$TIME_RANGE" \
        slurm/period_contrast_chunk.sbatch)
    echo "submitted $JOB_NAME as $JID"
    exit 0
done < slurm/CHUNKS.txt

echo "all chunks already have output parquets."
exit 2
