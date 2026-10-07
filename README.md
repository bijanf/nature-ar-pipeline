# nature-ar-pipeline

Code for "Wind and moisture contributions to the observed change of eight moisture-transport corridors, 1940–2024" (Rostami, Fallah and Fu, submitted to Earth System Dynamics, 2026).

## Contents
- `src/data/`: retrieval of ERA5 monthly fields from the Copernicus Climate Data Store (single-level flux integrals and 23 pressure levels), extraction of corridor boxes from ERA5 and from CMIP6 (Pangeo cloud archive and ESGF-mirrored archives).
- `src/analysis/`: the level-resolved decomposition of the transport change into moisture-change, wind-change, covariance and transient terms; moving-block bootstrap and false-discovery control; Hamed–Rao trend tests; period-sensitivity matrix; seasonal split; ERA5-derived climate indices and their validation; regression of the yearly wind contribution on the indices; CMIP6 large-ensemble detection statistics and end-of-century projection.
- `src/figures/`: the main and supplementary figures.
- `src/tables/`: the LaTeX table bodies.
- `slurm/`: batch scripts for the PIK cluster, run in the order `fetch_era5_fullcolumn`, `extract_cmip6_all`, `extract_cmip6_proj`, `run_fullcolumn`, `run_fullcolumn_rest`, `run_cmip6_detection`, `run_cmip6_proj`, `run_maps`, `run_si`.
- Earlier modules (event detection, period contrasts, the moisture-weighted transport-speed split, EOF attribution) are kept for reference; the manuscript uses the modules listed above.

## Requirements
Python 3.11 with the packages in `environment.yml` (`conda env create -f environment.yml`). Access to the Copernicus Climate Data Store (a `~/.cdsapirc` file) is needed for the ERA5 retrieval; the CMIP6 large ensembles are read anonymously from the Pangeo Google Cloud archive.

## Usage
Paths to the data directories are set at the top of the modules in `src/data/` and `src/analysis/`. With the data in place:

```
python -m src.data.extract_era5_corridors
python -m src.analysis.fullcolumn_decomp --part headline
python -m src.analysis.fullcolumn_decomp --part sensitivity
python -m src.analysis.fullcolumn_decomp --part seasons
python -m src.analysis.fullcolumn_decomp --part levels
python -m src.analysis.indices_era5
python -m src.analysis.index_regression
python -m src.analysis.cmip6_detection
python -m src.analysis.cmip6_projection
python -m src.analysis.cmip6_trend_test RESULTS_DIR
python -m src.analysis.bootstrap_distributions
```

The figure modules take the result files as arguments; the batch scripts in `slurm/` give the exact calls.

## Citation
Rostami, M., Fallah, B. and Fu, L.-Y.: Wind and moisture contributions to the observed change of eight moisture-transport corridors, 1940–2024, Earth System Dynamics, submitted, 2026.

## License
MIT License (see `LICENSE`).
