#!/bin/bash
# Map figures from the global ERA5 fields (Figs. 2 and 7).
#SBATCH --job-name=maps
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
set -e
mkdir -p figures/esd
python -m src.figures.fig_maps figures/esd/fig02_maps.pdf
python -m src.figures.fig_examples figures/esd/fig07_examples.pdf
echo MAPS_DONE
