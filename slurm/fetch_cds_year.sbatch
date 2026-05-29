#!/bin/bash
# Fetch one year of ERA5 from CDS (pressure-level + surface) into the
# local NetCDF cache. Bypasses the gcsfs/aiohttp streaming wall (see
# slurm/STATUS.md) by letting CDS do the bbox + time subset server-side
# and returning a single ~2 GB NetCDF per year.
#
# Submit with:
#   sbatch --job-name=cds_2018 --export=ALL,YEAR=2018 slurm/fetch_cds_year.sbatch
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh

: "${YEAR:?YEAR env var required (e.g. YEAR=2018)}"

python -m src.data.fetch_cds_era5 --year "$YEAR"
