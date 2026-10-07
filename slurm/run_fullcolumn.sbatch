#!/bin/bash
# Full-column ERA5 analysis chain: corridor extraction, decomposition (headline, sensitivity, seasons,
# CMIP-level variant), climate indices, index regression, and the first figures.
#SBATCH --job-name=fullcol_ana
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
set -e
D=/p/projects/poem/fallah/nature_ar_data
test "$(ls $D/era5_fullcolumn/sl/*.nc | wc -l)" -eq 85 && test "$(ls $D/era5_fullcolumn/pl/*.nc | wc -l)" -eq 85
python -m src.data.extract_era5_corridors
python -m src.analysis.fullcolumn_decomp --part headline --nboot 1000
python -m src.analysis.fullcolumn_decomp --part sensitivity --nboot 300
python -m src.analysis.fullcolumn_decomp --part seasons --nboot 500
python -m src.analysis.fullcolumn_decomp --part levels --nboot 1000
python -m src.analysis.indices_era5
python -m src.analysis.index_regression
mkdir -p figures/esd
python -m src.figures.fig_decomp_terms $D/results_esd/fullcolumn_headline.json figures/esd/fig01_decomposition_terms.pdf
python -m src.figures.fig_period_sensitivity $D/results_esd/fullcolumn_sensitivity.json figures/esd/fig03_period_sensitivity.pdf
echo FULLCOLUMN_DONE
