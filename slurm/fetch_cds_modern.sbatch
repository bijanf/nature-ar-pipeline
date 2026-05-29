#!/bin/bash
# Fetch the full Modern period (1980-2014) from CDS in one allocation.
# Runs in parallel with cds_recent (CDS allows ~8 concurrent requests
# per user; each sbatch uses 4 of those via the ThreadPoolExecutor).
#
# Walltime budget: 35 years × ~25 min/year = ~14.5 h serial inside the
# job, with 4-way parallel per-year that's the cap.
#SBATCH --job-name=cds_modern
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

for Y in $(seq 1980 2014); do
    python -m src.data.fetch_cds_era5 --year "$Y" --concurrency 4
done
