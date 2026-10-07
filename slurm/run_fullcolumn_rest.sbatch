#!/bin/bash
# Remaining full-column chain after the decomposition: indices, index regression, first figures.
#SBATCH --job-name=fullcol_rest
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
set -e
D=/p/projects/poem/fallah/nature_ar_data
python -m src.analysis.indices_era5
python -m src.analysis.index_regression
mkdir -p figures/esd
python -m src.figures.fig_decomp_terms $D/results_esd/fullcolumn_headline.json figures/esd/fig01_decomposition_terms.pdf
python -m src.figures.fig_period_sensitivity $D/results_esd/fullcolumn_sensitivity.json figures/esd/fig03_period_sensitivity.pdf
echo FULLCOLUMN_REST_DONE
