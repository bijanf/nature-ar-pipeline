#!/bin/bash
# Sourced by each period_contrast sbatch — keeps the env / Dask / repo
# bootstrap in one place. Not executed directly.

set -euo pipefail

# Conda bootstrap on this cluster: mambaforge under $HOME.
source /home/fallah/mambaforge/etc/profile.d/conda.sh
conda activate nature-ar-pipeline

# Suppress ~/.local site-packages — they were pip-installed under an older
# numpy 1.x and cause `_ARRAY_API not found` ImportErrors when the conda env
# loads xarray / shapely / xesmf with numpy 2.x.
export PYTHONNOUSERSITE=1

cd /home/fallah/scripts/nature-ar-pipeline

# Dask memory thresholds. Defaults trip OOM-killers on ARCO-ERA5 multi-year
# materialisations; we want spill *before* the SLURM cgroup cap fires.
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.4
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.5
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.75
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.90
# Encourage glibc to return freed pages to the OS sooner — helps when
# the cgroup-RSS ceiling is tight (io partition: ~20 GB hard cap).
export MALLOC_TRIM_THRESHOLD_=0
# Prevent BLAS/OpenMP libs from spawning a thread-per-core and replicating
# big intermediate arrays — under 4 CPUs each metpy call would otherwise
# duplicate its working set 4×.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Force float32 conversions to ignore irrelevant pint warnings during physics.
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning"
