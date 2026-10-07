#!/bin/bash
# All CMIP6 corridor extractions in one io allocation (io QOS allows two jobs per user).
#SBATCH --job-name=c6_all
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore
run() { python -m src.data.extract_cmip6_corridors "$@" > "slurm/logs/c6_${3}.log" 2>&1; }
run --source pangeo --model CanESM5 &
run --source pangeo --model MIROC6 &
run --source pangeo --model ACCESS-ESM1-5 &
wait
run --source pangeo --model EC-Earth3 --max-members 10 &
bash slurm/extract_cmip6_pik_all.sh > slurm/logs/c6_pik.log 2>&1 &
wait
