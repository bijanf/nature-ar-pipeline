#!/bin/bash
# 1995-2100 ssp585 corridor extraction from the PIK CMIP6 archive (one member per model).
#SBATCH --job-name=c6_proj
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore
for M in BCC-CSM2-MR CAMS-CSM1-0 CNRM-ESM2-1 CanESM5 EC-Earth3 GFDL-CM4 GFDL-ESM4 MIROC6 \
         MPI-ESM1-2-HR NESM3 EC-Earth3-Veg IPSL-CM6A-LR MRI-ESM2-0 UKESM1-0-LL; do
    python -m src.data.extract_cmip6_corridors --source pik --model "$M" --projection
done
