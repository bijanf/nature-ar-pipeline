#!/bin/bash
# Fetch ERA5 monthly single-level moisture-flux integrals and 23-level q,u,v,T
# (1940-2024, 1 degree) for the full-column decomposition; both products in parallel.
#SBATCH --job-name=cds_fullcol
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
python -m src.data.fetch_era5_fullcolumn --kind sl --concurrency 3 &
python -m src.data.fetch_era5_fullcolumn --kind pl --concurrency 5 &
wait
# second pass retries any year that failed in the first
python -m src.data.fetch_era5_fullcolumn --kind sl --concurrency 2
python -m src.data.fetch_era5_fullcolumn --kind pl --concurrency 4
