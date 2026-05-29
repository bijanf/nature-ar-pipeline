#!/bin/bash
# Submit the four-job step-1 chain: three period_contrast jobs in parallel,
# then concat + figures depending on all three finishing OK.
#
# Run from the repo root:
#   bash slurm/submit_step1.sh
#
# Recommended: smoke the cheapest period first to validate the env on a
# real compute node before burning the longer runs:
#   sbatch slurm/period_contrast_recent.sbatch
# Once that produces a sane observational_events_recent.parquet, submit the
# rest with this driver.

set -euo pipefail
cd "$(dirname "$0")/.."

J1=$(sbatch --parsable slurm/period_contrast_presat.sbatch)
J2=$(sbatch --parsable slurm/period_contrast_modern.sbatch)
J3=$(sbatch --parsable slurm/period_contrast_recent.sbatch)
J4=$(sbatch --parsable --dependency=afterok:"$J1":"$J2":"$J3" \
            slurm/concat_and_figures.sbatch)

echo "submitted:"
echo "  pre-sat : $J1"
echo "  modern  : $J2"
echo "  recent  : $J3"
echo "  concat  : $J4  (afterok of the three above)"
