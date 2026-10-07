#!/bin/bash
# Extract corridor-box hus/ua/va (1940-2024) for one CMIP6 model.
# Usage: sbatch --job-name=c6_<model> slurm/extract_cmip6.sbatch <source> <model> [max_members]
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore
SRC=$1; MODEL=$2; MAXM=${3:-0}
python -m src.data.extract_cmip6_corridors --source "$SRC" --model "$MODEL" --max-members "$MAXM"
