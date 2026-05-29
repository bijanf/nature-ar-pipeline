#!/bin/bash
# Re-fetch the months whose earlier downloads left 0-byte / corrupt files
# (1951 = Pre-sat, 1995 = Modern). Uses the hardened fetch path (atomic
# .tmp -> rename, size validation), so only the missing months are pulled
# and a failed retrieve never leaves a fake-complete file again.
#SBATCH --job-name=cds_repair
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

for Y in 1951 1995; do
    python -m src.data.fetch_cds_era5 --year "$Y" --concurrency 8 --kinds pl
done
