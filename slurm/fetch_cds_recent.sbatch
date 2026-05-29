#!/bin/bash
# Fetch every remaining year of the Recent period (2015-2024) from CDS in
# one allocation. Each year iterates inside a single job so we only use one
# io MaxSubmitPU slot regardless of the number of years.
#
# The fetcher is idempotent: years whose YYYY_*.nc files already exist are
# skipped. Submitting this twice does no harm.
#SBATCH --job-name=cds_recent
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

# Make the static fetch a no-op if already on disk; otherwise it fetches
# once at the top of the loop.
python -m src.data.fetch_cds_era5 --static-only

for Y in 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
    python -m src.data.fetch_cds_era5 --year "$Y" --concurrency 4
done
