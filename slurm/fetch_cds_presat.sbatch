#!/bin/bash
# Fetch the Pre-satellite period (1940-1979) from CDS in one allocation.
# Years 1940-1958 are served by the preliminary back-extension dataset
# (fetch_cds_era5._pl_dataset_name decides per-year).
#
# Walltime budget: 40 years × ~25 min/year = ~16.5 h serial inside the
# job. 48 h walltime leaves margin for slower back-extension queries.
#SBATCH --job-name=cds_presat
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

python -m src.data.fetch_cds_era5 --static-only

# Concurrency 8 (was 4): CDS allows up to ~8 concurrent requests per user
# and Recent/Modern are no longer running by the time presat fires, so we
# can use the full quota. Kinds=pl drops `sfc` (surface_pressure has no
# downstream consumer in the manuscript pipeline).
for Y in $(seq 1940 1979); do
    python -m src.data.fetch_cds_era5 --year "$Y" --concurrency 8 --kinds pl
done
