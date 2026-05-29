#!/bin/bash
# Finish cds_recent (it timed out at 24h with 2024 6/12 done). Just
# fetch the missing 2024 months (the per-month skip-if-exists logic
# in fetch_cds_era5 means we only re-download what's missing).
# Concurrency 8 + kinds=pl only (sfc has no downstream consumer).
#SBATCH --job-name=cds_recent_finish
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

python -m src.data.fetch_cds_era5 --year 2024 --concurrency 8 --kinds pl
