#!/bin/bash
# CMIP6 decomposition of every extracted member, detection statistics and the CMIP6 figure.
#SBATCH --job-name=cmip6_det
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
set -e
D=/p/projects/poem/fallah/nature_ar_data
python -m src.analysis.cmip6_detection
python -m src.analysis.cmip6_trend_test $D/results_esd
mkdir -p figures/esd
python -m src.figures.fig_cmip6_detection $D/results_esd/cmip6_members.json figures/esd/fig06_cmip6.pdf
echo CMIP6_DONE
