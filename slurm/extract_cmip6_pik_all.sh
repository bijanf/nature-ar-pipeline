#!/bin/bash
# Run the PIK single-member extraction for every model with Amon hus/ua/va.
source slurm/_common.sh
export PYTHONWARNINGS=ignore
for M in BCC-CSM2-MR CAMS-CSM1-0 CNRM-ESM2-1 CanESM5 EC-Earth3 GFDL-CM4 GFDL-ESM4 MIROC6 \
         MPI-ESM1-2-HR NESM3 EC-Earth3-Veg IPSL-CM6A-LR MRI-ESM2-0 UKESM1-0-LL; do
    python -m src.data.extract_cmip6_corridors --source pik --model "$M"
done
