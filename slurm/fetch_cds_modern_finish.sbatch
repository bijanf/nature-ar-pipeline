#!/bin/bash
# Finish cds_modern. The original (686013) ran at --concurrency 4 *with*
# sfc and timed out at 48 h with 1980-1994 done. This resumes at the
# faster --concurrency 8 --kinds pl (sfc has no downstream consumer) for
# the remaining years; the per-month skip-if-exists logic means only the
# missing 1994-12..2014 months are actually fetched.
#SBATCH --job-name=cds_modern_finish
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

for Y in $(seq 1994 1999); do
    python -m src.data.fetch_cds_era5 --year "$Y" --concurrency 8 --kinds pl
done
