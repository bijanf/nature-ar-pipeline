#!/bin/bash
# Supplement data and figures: bootstrap distributions, SI figures, maps re-run.
#SBATCH --job-name=si_figs
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err

source slurm/_common.sh
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
set -e
D=/p/projects/poem/fallah/nature_ar_data
mkdir -p figures/esd
python -m src.figures.fig_maps figures/esd/fig02_maps.pdf
python -m src.figures.fig_examples figures/esd/fig07_examples.pdf
python -m src.analysis.bootstrap_distributions
python -m src.figures.fig_si_bootstrap $D/results_esd/bootstrap_wind.npz figures/esd/figS03_bootstrap.pdf
echo SI_DONE
