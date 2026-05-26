# nature-ar-pipeline

[![CI](https://github.com/bijanf/nature-ar-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/bijanf/nature-ar-pipeline/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

US-West-Coast atmospheric rivers (ARs), three observational periods and four
projected SSP scenarios on one trajectory. Accompanies a manuscript in
preparation for a *Nature*-family journal.

---

## The claim, in one paragraph

The ERA5 reanalysis record (1940–2024) already shows AR intensification on
the US West Coast across three observational periods — pre-satellite
(1940–1979), modern (1980–2014), and recent (2015–2024). The same diagnostic,
extended through a physically constrained ML emulator into four SSP-2070-2099
futures (SSP2-4.5, SSP3-7.0, SSP4-6.0, SSP5-8.5), projects a further
intensification consistent with the observed trajectory. Driver attribution
via SHAP indicates that **thermodynamic moisture amplification** (rising θ_e
at 850 hPa and column IVT) dominates both the observed contrast and the
projected one, with **dynamic** (PV at 250 hPa, Eady growth) contributions
varying by landfall band and SSP.

The ML's role is to *extend* the observed trajectory, not to lead it. A
reviewer who distrusts the ML can still read the observed shift from the
left half of the headline trajectory figure.

## Observational + projected periods

| period | window | role | data source |
|---|---|---|---|
| Pre-satellite | 1940–1979 | climatological reference | ARCO-ERA5 (back-extended; reduced obs constraint) |
| Modern | 1980–2014 | well-observed warming | ARCO-ERA5; ML training window |
| Recent | 2015–2024 | recent decade | ARCO-ERA5; ML temporal-generalization holdout |
| SSP2-4.5 | 2070–2099 | middle-of-road | CMIP6 (Pangeo, conservative regrid) |
| SSP3-7.0 | 2070–2099 | CMIP7-aligned headline | CMIP6 |
| SSP4-6.0 | 2070–2099 | inequality / asymmetric forcing | CMIP6 |
| SSP5-8.5 | 2070–2099 | upper-bound stress test | CMIP6 |

## Architecture

```
ARCO-ERA5 1940-2024  (Pangeo Zarr, streamed)
     │
     ▼
src/features/physics_pipeline.py        IVT, θ_e (850 hPa), QG-PV (250 hPa), Eady σ_BI
     │
     ├─► src/analysis/period_contrast.py     ← observed shift across three periods
     │                                        (period-internal GW climatology so
     │                                         threshold drift doesn't confound
     │                                         genuine intensification)
     │
     ▼
src/features/ar_detection.py            Guan & Waliser (2015) AR mask
     │
     ▼
src/models/stage1_ar_emulator.py        LightGBM, linear_tree=True
     │                                  target = GW_mask × IVT
     ▼
src/models/event_post.py                threshold + scipy.ndimage.label → events
     │
     ▼
src/models/stage2_cascade.py            LightGBM quantile regression α=0.05/0.50/0.95
     │                                  target = AR-event total precipitation
     ▼
src/models/explainability.py            SHAP θ_e vs PV per landfall band,
                                        for observed AND projected contrasts
     ▲
     │
CMIP6 SSP2-4.5 / SSP3-7.0 / SSP4-6.0 / SSP5-8.5 (2070-2099)
     ── conservative regrid (xesmf) ── direct & delta-change inference
```

## Domain

US-West-Coast landfall corridor: 25 °N – 60 °N, 210 °E – 250 °E. Native
0.25° resolution, 6-hourly cadence, pressure-level fields at 100/250/500/700/850/1000 hPa.

## Spatial cross-validation (Stages 1 + 2)

5° × 5° blocks, K = 5, 1° outer buffer, year-blocked, stratified on per-block
AR climatology.

## Install

`xesmf` + `esmpy` are conda-only; use the conda environment for end-to-end:

```bash
git clone git@github.com:bijanf/nature-ar-pipeline.git
cd nature-ar-pipeline
conda env create -f environment.yml
conda activate nature-ar-pipeline
```

Pure pip works for everything except CMIP6 inference (no xesmf):

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # lint + tests
```

## Reproduce

```bash
# 1. Observed shift across the three ERA5 periods (the empirical core).
python -m src.analysis.period_contrast

# 2. Build the Stage 1 training set on the modern period.
python -m src.data.era5_train_assembly --period 1980-2014

# 3. Train Stage 1 with spatial-block CV.
python -m src.models.stage1_ar_emulator --cv

# 4. Build the event-level Stage 2 dataset.
python -m src.data.event_dataset --intensity <path-to-intensity>.zarr

# 5. Train Stage 2 quantile regression.
python -m src.models.stage2_cascade --cv

# 6. CMIP6 four-SSP inference (direct + delta).
for ssp in ssp245 ssp370 ssp460 ssp585; do
    for mode in direct delta; do
        python -m src.inference.cmip6_apply --source MPI-ESM1-2-HR --experiment $ssp --mode $mode
    done
done

# 7. SHAP attribution (per SSP and for the observed contrast).
python -m src.models.explainability \
    --booster data/cache/stage1_models/fold_0_seed42.txt \
    --events data/cache/cmip6_events_MPI-ESM1-2-HR_ssp370_direct.parquet \
    --name shap_attribution_ssp370

# 8. Headline figure: observed → projected trajectory.
python -m src.figures.fig3_trajectory
```

Reviewer-facing one-month demo:

```bash
jupyter nbconvert --execute notebooks/e2e_demo.ipynb
```

## Engineering invariants

- **Streaming.** ERA5 and CMIP6 are streamed from Pangeo via
  `xarray.open_zarr`. Raw data never lands on disk.
- **Lazy graph.** Every operation produces a Dask graph until the final
  ML-training or Parquet-write boundary.
- **Period-internal GW climatology.** Each observational period uses its
  *own* 85th-percentile IVT climatology, so trends in IVT don't artificially
  erode the AR detection threshold.
- **Conservative regridding.** CMIP6 → ERA5 via `xesmf` conservative
  regridder preserves column integrals of vapor transport.
- **Reproducibility.** Every trained model and intermediate artifact is
  cached to `data/cache/` with period, model id, and random seed embedded
  in the filename.
- **ML never trains on pre-satellite data.** The 1940–1979 window is
  *analysis-only*; the ML sees only 1980–2014.

## Layout

```
src/
  config.py                  bbox, levels, observational + SSP periods, Zarr URLs
  features/
    physics_pipeline.py      Holton-dynamics features
    ar_detection.py          Guan-Waliser AR mask, period-internal climatology
    topography.py            event-level elevation aggregates
  analysis/
    period_contrast.py       per-observational-period event statistics with bootstrap CIs
  data/
    era5_train_assembly.py   stratified Stage 1 training Parquet
    event_dataset.py         Stage 2 event dataset
  models/
    stage1_ar_emulator.py    LightGBM Stage 1 regressor
    event_post.py            scipy.ndimage.label → discrete events
    stage2_cascade.py        LightGBM quantile regression
    explainability.py        SHAP θ_e vs PV attribution
  inference/
    cmip6_apply.py           four-SSP direct + delta inference
  figures/                   Nature-spec vector PDFs
  tests/                     pytest (64 tests, network-free)
notebooks/
  e2e_demo.ipynb             reviewer-facing one-month walkthrough
```

## CI

GitHub Actions runs Ruff (lint + format) and Pytest (network-free, fast) on
every push and pull request. Network-bound and slow tests live behind the
`network` and `slow` markers and are skipped in CI.

## Caveats (must surface in the manuscript)

- **ERA5 1940–1978 back-extension** (HRES + ERA5-BE) has reduced
  observational constraint, particularly on moisture. We treat the pre-
  satellite period as a *climatological reference*: useful for trend
  detection at the basin scale, inadequate for individual-event validation.
  The ML never trains on pre-satellite data.
- **CMIP6 SSP4-6.0 model coverage on Pangeo at 6hrPlevPt is uneven.**
  The inference module raises `LookupError` (rather than silently
  substituting a different model) when the requested combination is absent,
  so the orchestration can tag missing scenarios honestly.
- **`linear_tree=True` extrapolation** is the load-bearing extrapolation
  argument for Stage 1; both `direct` and `delta` CMIP6 protocols are
  reported, and their agreement is itself a robustness check.

## License & data attribution

Code: license pending paper acceptance.
ERA5 reanalysis © ECMWF, distributed under CC-BY-4.0.
CMIP6 model output © respective modelling groups, licensed per ESGF terms.

## Citation

Manuscript in preparation (Fallah et al., 2026). If you use this code prior
to publication, please open an issue or contact `fallah@pik-potsdam.de`.
