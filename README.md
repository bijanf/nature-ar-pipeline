# nature-ar-pipeline

[![CI](https://github.com/bijanf/nature-ar-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/bijanf/nature-ar-pipeline/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A physically constrained, two-stage machine-learning pipeline that emulates
Atmospheric Rivers (ARs) and projects their cascading hydrometeorological
hazards over the US West Coast under SSP5-8.5. Accompanies a manuscript in
preparation for a *Nature* family journal.

---

## Why this exists

Atmospheric Rivers deliver up to half of US West Coast cool-season precipitation
and an outsized share of flood-day hazard. Projecting how ARs and their
compound hazards shift under end-of-century warming demands a model that:

1. **respects the underlying dynamics** — vertically integrated moisture
   transport, upper-level synoptic forcing, baroclinic energetics;
2. **can extrapolate** into thermodynamic regimes unseen during training,
   because SSP5-8.5 mid-latitude moisture loadings exceed any 20th-century
   reanalysis;
3. **quantifies uncertainty** for the downstream cascading hazard.

We address (1) by deriving Holton-dynamics features from ERA5 with
[`MetPy`](https://unidata.github.io/MetPy/), (2) by using `LightGBM` with
`linear_tree=True` leaves that admit linear extrapolation along a
Clausius-Clapeyron-scaled IVT axis, and (3) by quantile regression at
α ∈ {0.05, 0.50, 0.95} on AR-event total precipitation.

## Architecture

```
ARCO-ERA5  (Pangeo Zarr, streamed)
     │
     ▼
src/features/physics_pipeline.py        IVT, θ_e (850 hPa), QG-PV (250 hPa), Eady σ_BI
     │
     ▼
src/features/ar_detection.py            Guan & Waliser (2015) AR mask, in-repo
     │
     ▼
src/models/stage1_ar_emulator.py        LightGBM regressor, linear_tree=True
     │                                  target = GW_mask × IVT
     ▼
src/models/event_post.py                threshold + scipy.ndimage.label → events
     │
     ▼
src/models/stage2_cascade.py            LightGBM quantile regression, α = 0.05 / 0.50 / 0.95
     │                                  target = AR-event total precipitation
     ▼
src/models/explainability.py            SHAP θ_e (thermodynamic) vs PV (dynamic) attribution
     ▲
     │
CMIP6 SSP5-8.5 (Pangeo) ── xesmf conservative regrid ── direct & delta-change inference
```

## Domain & resolution

| | |
|--|--|
| Region | 25 °N – 60 °N, 210 °E – 250 °E (US West Coast landfall corridor) |
| Native resolution | 0.25° |
| Cadence | 6-hourly |
| Pressure levels | 850 / 500 / 250 hPa |

## Train / holdout / inference split

| set       | period     | source                       |
|-----------|------------|------------------------------|
| train     | 1980 – 2014 | ARCO-ERA5 (Pangeo)           |
| holdout   | 2015 – 2024 | ARCO-ERA5 (Pangeo)           |
| inference | 2070 – 2099 | CMIP6 SSP5-8.5 (Pangeo)      |

Spatial cross-validation: **5° × 5° blocks, K = 5, 1° outer buffer,
year-blocked, stratified on per-block AR climatology.**

## Install

The full pipeline depends on `xesmf` + `esmpy`, both of which are conda-only.
We recommend the conda environment:

```bash
git clone git@github.com:bijanf/nature-ar-pipeline.git
cd nature-ar-pipeline
conda env create -f environment.yml
conda activate nature-ar-pipeline
```

A pure-pip install works for everything except CMIP6 inference:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for lint + test tooling
```

## Reproduce

```bash
# 1. Build the training set (streams ERA5 from Pangeo)
python -m src.data.era5_train_assembly --period 1980-2014

# 2. Train Stage 1 with 5-fold spatial-block CV
python -m src.models.stage1_ar_emulator --cv

# 3. Build the event-level dataset for Stage 2
python -m src.data.event_dataset

# 4. Train Stage 2 (quantile regression)
python -m src.models.stage2_cascade --cv

# 5. CMIP6 SSP5-8.5 inference (direct headline + delta robustness)
python -m src.inference.cmip6_apply --source MPI-ESM1-2-HR --mode direct
python -m src.inference.cmip6_apply --source MPI-ESM1-2-HR --mode delta

# 6. Figures (Nature spec: vector PDF, 6–7 pt Helvetica, 88 / 180 mm widths)
python -m src.figures.fig3_ssp585_shift
```

A reviewer-facing end-to-end demo runs the full pipeline on a single year:

```bash
jupyter nbconvert --execute notebooks/e2e_demo.ipynb
```

## Engineering invariants

These are non-negotiable. They protect the validity of the projections and the
reproducibility of the results.

- **Streaming.** ARCO-ERA5 and CMIP6 are streamed from Pangeo via
  `xarray.open_zarr`. Raw data never lands on disk.
- **Lazy graph.** Every operation produces a Dask graph until the final
  ML-training or Parquet-write boundary. No `.compute()`, `.values`,
  `.to_numpy()` upstream of those points.
- **Conservative regridding.** CMIP6 → ERA5 grid via `xesmf` conservative
  regridder preserves column integrals of vapor transport.
- **Reproducibility.** Every trained model and intermediate artifact is cached
  to `data/cache/` with period, model id, and random seed embedded in the
  filename. The pipeline is idempotent end-to-end.

## Layout

```
src/
  config.py                  bbox, levels, Zarr URLs, ERA5 variable map
  features/
    physics_pipeline.py      Holton-dynamics features
    ar_detection.py          Guan-Waliser AR mask
    topography.py            event-level elevation aggregates
  data/
    era5_train_assembly.py   stratified training Parquet
    event_dataset.py         Stage-2 event dataset
    cache.py                 Zarr/Parquet helpers
  models/
    stage1_ar_emulator.py    LightGBM regressor
    event_post.py            scipy.ndimage.label
    stage2_cascade.py        LightGBM quantile regression
    explainability.py        SHAP attribution
  inference/
    cmip6_apply.py           direct + delta CMIP6 protocols
  figures/                   Nature-spec vector PDFs
  tests/                     pytest (network-free)
notebooks/
  e2e_demo.ipynb             reviewer-facing demo
```

## Continuous integration

GitHub Actions runs `ruff check`, `ruff format --check`, and `pytest`
(network-free tests only) on every push and pull request — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). Network-bound and slow
tests live behind the `network` and `slow` markers and are skipped in CI.

## License & data attribution

Code: license pending paper acceptance.
ERA5 reanalysis © ECMWF, distributed under CC-BY-4.0.
CMIP6 model output © respective modelling groups, licensed per ESGF terms.

## Citation

Manuscript in preparation (Fallah et al., 2026). If you use this code prior to
publication, please open an issue or contact `fallah@pik-potsdam.de`.
