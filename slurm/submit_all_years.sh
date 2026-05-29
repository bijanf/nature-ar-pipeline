#!/bin/bash
# Submit one stage_year job per year in the contiguous 1940-2024 record.
# Each job is small (~4 GB on disk per year) and io-partition-bounded.
# Submitted sequentially via `afterany` since the io QoS caps the user
# at 2 concurrent submissions and there's already one held by 670945.
#
# Usage:
#   bash slurm/submit_all_years.sh         # 1940-2024
#   bash slurm/submit_all_years.sh 1980 2014   # custom range
#
# Important: this driver assumes path-2 (local-staging) is unblocked —
# i.e., the io partition's 20 GB cgroup cap doesn't OOM-kill a single-year
# stage. As of the night the streaming pipeline was first attempted, ONE
# year already OOMed at 1m07s (see slurm/STATUS.md). Do not run this
# blindly; either raise io MaxMemPerCPU first or experiment with a
# smaller cadence/sub-year window first.

set -euo pipefail
cd "$(dirname "$0")/.."

START=${1:-1940}
END=${2:-2024}

PREV=
for Y in $(seq "$START" "$END"); do
    if [ -d "data/cache/era5_bbox_local/${Y}.zarr" ]; then
        echo "skip $Y (already on disk)"
        continue
    fi
    if [ -n "$PREV" ]; then
        DEP=("--dependency=afterany:${PREV}")
    else
        DEP=()
    fi
    JID=$(sbatch --parsable \
        --job-name="stage_${Y}" \
        --export="ALL,YEAR=${Y}" \
        "${DEP[@]}" \
        slurm/stage_year.sbatch)
    echo "submitted stage_${Y} as $JID"
    PREV=$JID
done
