#!/bin/bash
# End-of-century CMIP6 decomposition (1995-2014 vs 2081-2100, SSP5-8.5).
#SBATCH --job-name=cmip6_proj
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m src.analysis.cmip6_projection
echo CMIP6_PROJ_DONE
