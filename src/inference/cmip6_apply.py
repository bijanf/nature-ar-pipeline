"""
CMIP6 inference: run Stage 1 + Stage 2 over the four-SSP scenario set.

For each ``(source_id, experiment_id)`` we

  1. resolve the Pangeo zarr store via :mod:`intake_esm`;
  2. open it lazily, slice to the project bbox, conservative-regrid to the
     ERA5 0.25° grid via :mod:`xesmf`;
  3. rename CMIP6 variables to the ERA5 vocabulary so
     :func:`src.features.physics_pipeline.calculate_dynamics` works unchanged
     (``zg`` is multiplied by ``g`` so the ``/g`` step in physics_pipeline
     recovers heights correctly);
  4. apply the cached Stage 1 boosters to obtain a per-pixel intensity field,
     reduce to events via :func:`src.models.event_post.extract_events`, and
     score each event with Stage 2 quantile regression for the precipitation
     hazard.

Two protocols are run for each scenario:

* **direct**: features come from raw CMIP6 fields — this is the literal test
  of ``linear_tree=True`` extrapolation.
* **delta**: features = ``ERA5_baseline + (CMIP6_future_clim − CMIP6_hist_clim)``
  — the bias-cancelled robustness check.

Headline scenario is ``HEADLINE_EXPERIMENT`` (SSP3-7.0); the other SSPs (2-4.5,
4-6.0, 5-8.5) are reported alongside. SSP4-6.0 has uneven model coverage on
Pangeo — :func:`resolve_zarr` raises ``LookupError`` if MPI-ESM1-2-HR doesn't
publish the experiment at 6hrPlevPt, and the orchestration loop tags that
scenario as "missing" rather than silently substituting a different model.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import intake
import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.features import physics_pipeline, topography
from src.models import event_post, stage2_cascade

# xesmf (and its esmpy backend) is conda-only and not exercised in pip-based
# CI. We import it lazily inside :func:`regrid_and_rename` so the rest of this
# module is importable without it.

logger = logging.getLogger(__name__)

# Stage 1 feature ordering — must match the column order used by the
# era5_train_assembly Parquet so the booster's column indices line up.
_STAGE1_FEATURES = (
    "ivt",
    "ivt_u",
    "ivt_v",
    "theta_e_850",
    "pv_250",
    "eady_growth_rate",
    "lat_deg",
    "doy_sin",
    "doy_cos",
)


# =============================================================================
# Catalog access
# =============================================================================


def open_pangeo_catalog():
    """Open the Pangeo CMIP6 intake-esm catalog."""
    return intake.open_esm_datastore(config.CMIP6_CATALOG_URL)


def resolve_zarr(
    catalog,
    source_id: str,
    experiment_id: str,
    variable_ids: Iterable[str] = config.CMIP6_QUERY_BASE["variable_id"],
) -> xr.Dataset:
    """Return a single lazy ``xr.Dataset`` joining the requested variables for
    ``(source_id, experiment_id)`` at 6hrPlevPt. Raises ``LookupError`` if the
    catalog doesn't carry the combination (so the orchestrator can tag the
    scenario as 'missing' rather than silently picking a different model)."""
    query = {
        **{k: v for k, v in config.CMIP6_QUERY_BASE.items() if k != "variable_id"},
        "source_id": source_id,
        "experiment_id": experiment_id,
        "variable_id": list(variable_ids),
    }
    subset = catalog.search(**query)
    if len(subset.df) == 0:
        raise LookupError(
            f"Pangeo CMIP6 catalog has no entries for {source_id} / {experiment_id} "
            f"at table {config.CMIP6_QUERY_BASE['table_id']}."
        )
    ds_dict = subset.to_dataset_dict(
        storage_options={"token": "anon"},
        zarr_kwargs={"consolidated": True},
        progressbar=False,
    )
    return xr.merge(list(ds_dict.values()))


# =============================================================================
# Regrid + variable harmonisation
# =============================================================================


def _era5_target_grid() -> xr.Dataset:
    """The 0.25° ERA5 grid restricted to the project bbox, as an xesmf target."""
    lats = np.arange(config.BBOX["lat_min"], config.BBOX["lat_max"] + 0.25, 0.25)
    lons = np.arange(config.BBOX["lon_min"], config.BBOX["lon_max"] + 0.25, 0.25)
    return xr.Dataset({"lat": (("lat",), lats), "lon": (("lon",), lons)})


def regrid_and_rename(ds: xr.Dataset) -> xr.Dataset:
    """Conservative regrid CMIP6 -> ERA5 0.25° grid, then rename to the ERA5
    vocabulary so physics_pipeline works unchanged.

    The ``zg`` field is converted to ERA5 geopotential by multiplying by g
    so that physics_pipeline's ``Z = Φ/g`` step recovers heights correctly.
    """
    import xesmf as xe  # lazy: conda-only, not in CI pip install

    ds = physics_pipeline._apply_regional_bbox(ds)
    target = _era5_target_grid()
    regridder = xe.Regridder(ds, target, method="conservative", periodic=False)
    regridded = xr.Dataset(
        {name: regridder(ds[name]) for name in ds.data_vars if name in config.CMIP6_RENAME}
    )

    # zg (m) -> geopotential (m²/s²) so physics_pipeline.eady_growth_rate's
    # /g step inverts correctly.
    if "zg" in regridded.data_vars:
        regridded["zg"] = regridded["zg"] * physics_pipeline._G

    renamed = regridded.rename(config.CMIP6_RENAME)

    # CMIP6 uses ``plev`` for the pressure axis (Pa). Map to the ERA5 ``level``
    # name in hPa so the physics_pipeline level selectors work.
    if "plev" in renamed.coords:
        renamed = renamed.assign_coords(plev=renamed["plev"] / 100.0).rename({"plev": "level"})

    return renamed


# =============================================================================
# Stage 1 prediction over a lazy feature cube
# =============================================================================


def load_stage1_boosters(seed: int = config.RANDOM_SEED) -> list[lgb.Booster]:
    pattern = f"fold_*_seed{seed}.txt"
    files = sorted((config.CACHE_DIR / "stage1_models").glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Stage 1 boosters at {config.CACHE_DIR / 'stage1_models'} matching {pattern}."
        )
    return [lgb.Booster(model_file=str(p)) for p in files]


def load_stage2_boosters(seed: int = config.RANDOM_SEED) -> list[dict[float, lgb.Booster]]:
    out_dir = config.CACHE_DIR / "stage2_models"
    folds: dict[int, dict[float, lgb.Booster]] = {}
    for p in sorted(out_dir.glob(f"fold_*_alpha*_seed{seed}.txt")):
        # filename: fold_K_alphaA_seedS.txt
        parts = p.stem.split("_")
        k = int(parts[1])
        alpha = int(parts[2].replace("alpha", "")) / 100.0
        folds.setdefault(k, {})[alpha] = lgb.Booster(model_file=str(p))
    return [folds[k] for k in sorted(folds)]


def stack_features_to_matrix(features: xr.Dataset) -> tuple[np.ndarray, xr.DataArray]:
    """Materialise the lazy feature cube to a (N, F) NumPy matrix in the column
    order LightGBM expects. Returns the matrix and one of the original
    DataArrays (held back as a template for un-stacking the predictions)."""
    template = features["ivt"]
    lat_name = "latitude" if "latitude" in template.coords else "lat"
    lon_name = "longitude" if "longitude" in template.coords else "lon"

    base = features.transpose("time", lat_name, lon_name)
    # Add the auxiliary spatial/temporal columns the Stage 1 model was trained on.
    lat_grid = xr.broadcast(base[lat_name], base[lon_name])[0].astype("float32")
    lon_grid = xr.broadcast(base[lat_name], base[lon_name])[1].astype("float32")  # noqa: F841
    doy = base["time"].dt.dayofyear
    doy_sin = np.sin(2.0 * np.pi * doy / 365.25).astype("float32")
    doy_cos = np.cos(2.0 * np.pi * doy / 365.25).astype("float32")

    columns = []
    for name in _STAGE1_FEATURES:
        if name == "lat_deg":
            arr = xr.broadcast(lat_grid, base["time"])[0].transpose("time", lat_name, lon_name)
        elif name == "doy_sin":
            arr = xr.broadcast(doy_sin, base[lat_name], base[lon_name])[0].transpose(
                "time", lat_name, lon_name
            )
        elif name == "doy_cos":
            arr = xr.broadcast(doy_cos, base[lat_name], base[lon_name])[0].transpose(
                "time", lat_name, lon_name
            )
        else:
            arr = base[name]
        columns.append(arr.to_numpy().astype("float32").reshape(-1))

    matrix = np.column_stack(columns)
    return matrix, template


def predict_intensity(boosters: list[lgb.Booster], features: xr.Dataset) -> xr.DataArray:
    """Fold-mean intensity field. Stacks lazy features to a 2-D matrix, runs
    every booster, averages, then re-folds into ``(time, lat, lon)`` shape."""
    x, template = stack_features_to_matrix(features)
    preds = np.mean(
        [b.predict(x, num_iteration=getattr(b, "best_iteration", None)) for b in boosters], axis=0
    )
    out = preds.reshape(template.shape).astype("float32")
    return xr.DataArray(out, dims=template.dims, coords=template.coords, name="ar_intensity")


# =============================================================================
# Direct vs delta protocols
# =============================================================================


def _cmip6_features(
    catalog, source_id: str, experiment_id: str, period: tuple[str, str]
) -> xr.Dataset:
    """Open CMIP6, regrid + rename, slice to ``period``, compute physics features."""
    raw = resolve_zarr(catalog, source_id, experiment_id)
    raw = raw.sel(time=slice(*period))
    harmonised = regrid_and_rename(raw)
    return physics_pipeline.calculate_dynamics(harmonised)


def _climatology(features: xr.Dataset) -> xr.Dataset:
    """Per-month, per-pixel climatology over the period (one ``.compute()``)."""
    return features.groupby("time.month").mean(dim="time").compute()


def run_direct(
    catalog,
    source_id: str,
    experiment_id: str,
    stage1_boosters: list[lgb.Booster],
    stage2_boosters: list[dict[float, lgb.Booster]],
    era5_topo: xr.Dataset,
    period: tuple[str, str] = config.CMIP6_FUT_PERIOD,
) -> pd.DataFrame:
    """Raw CMIP6 features -> ML pipeline -> per-event predictions."""
    features = _cmip6_features(catalog, source_id, experiment_id, period)
    intensity = predict_intensity(stage1_boosters, features)
    return _events_to_stage2(
        intensity, features, stage2_boosters, era5_topo, source_id, experiment_id, "direct"
    )


def run_delta(
    catalog,
    source_id: str,
    experiment_id: str,
    era5_features: xr.Dataset,
    stage1_boosters: list[lgb.Booster],
    stage2_boosters: list[dict[float, lgb.Booster]],
    era5_topo: xr.Dataset,
    fut_period: tuple[str, str] = config.CMIP6_FUT_PERIOD,
    hist_period: tuple[str, str] = config.CMIP6_HIST_PERIOD,
) -> pd.DataFrame:
    """Delta-change protocol: feed Stage 1 a perturbed feature cube assembled
    from the ERA5 historical baseline plus the CMIP6 (future − historical)
    monthly anomaly. Cancels model bias by construction."""
    hist_clim = _climatology(_cmip6_features(catalog, source_id, experiment_id, hist_period))
    fut_clim = _climatology(_cmip6_features(catalog, source_id, experiment_id, fut_period))
    anomaly = fut_clim - hist_clim

    # Broadcast the monthly anomaly onto the ERA5 baseline timeline.
    era5_month = era5_features["time"].dt.month
    perturbed = era5_features + anomaly.sel(month=era5_month).drop_vars("month")

    intensity = predict_intensity(stage1_boosters, perturbed)
    return _events_to_stage2(
        intensity, perturbed, stage2_boosters, era5_topo, source_id, experiment_id, "delta"
    )


def _events_to_stage2(
    intensity: xr.DataArray,
    feature_cube: xr.Dataset,
    stage2_boosters: list[dict[float, lgb.Booster]],
    era5_topo: xr.Dataset,
    source_id: str,
    experiment_id: str,
    mode: str,
) -> pd.DataFrame:
    """Common tail: intensity field -> events -> per-event topo + Stage-1 physics
    features (for downstream SHAP) -> Stage 2 quantile predictions."""
    from src.features import event_features

    events, labels = event_post.extract_events_with_labels(
        intensity, land_sea_mask=era5_topo["land_sea_mask"], threshold=config.STAGE1_EVENT_THRESHOLD
    )
    if events.empty:
        return events.assign(source_id=source_id, experiment_id=experiment_id, mode=mode)

    # Materialise the Stage-1 physics features once over the inference window
    # and attach footprint×duration means per event. Required by Phase 5d SHAP.
    physics_cube = feature_cube[list(event_features._PHYSICS_FEATURES)].compute()
    events = event_features.attach_features(events, labels, physics_cube)

    topo_feats = topography.event_topo_features(events, era5_topo, intensity)
    dataset = events.merge(topo_feats, on="event_id")

    # Stage 2 feature order is hard-coded to match training.
    feat_cols = list(stage2_cascade._FEATURES)
    x = dataset[feat_cols].to_numpy(dtype="float32")
    preds = {alpha: [] for alpha in (0.05, 0.50, 0.95)}
    for fold_models in stage2_boosters:
        for alpha, b in fold_models.items():
            preds[alpha].append(b.predict(x, num_iteration=getattr(b, "best_iteration", None)))

    for alpha, arrs in preds.items():
        dataset[f"precip_pred_q{int(alpha * 100):02d}_mm"] = np.mean(arrs, axis=0)

    return dataset.assign(source_id=source_id, experiment_id=experiment_id, mode=mode)


# =============================================================================
# Orchestration + CLI
# =============================================================================


def write_events(df: pd.DataFrame, source_id: str, experiment_id: str, mode: str) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CACHE_DIR / f"cmip6_events_{source_id}_{experiment_id}_{mode}.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=config.CMIP6_QUERY_BASE["source_id"])
    parser.add_argument(
        "--experiment",
        type=str,
        default=config.HEADLINE_EXPERIMENT,
        choices=config.CMIP6_EXPERIMENTS,
    )
    parser.add_argument("--mode", type=str, default="direct", choices=("direct", "delta"))
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    catalog = open_pangeo_catalog()
    stage1 = load_stage1_boosters(seed=args.seed)
    stage2 = load_stage2_boosters(seed=args.seed)
    era5_topo = topography.open_era5_topography()

    if args.mode == "direct":
        df = run_direct(catalog, args.source, args.experiment, stage1, stage2, era5_topo)
    else:
        era5_features = physics_pipeline.calculate_dynamics(
            physics_pipeline.open_arco_era5()[list(physics_pipeline.required_era5_vars())].sel(
                time=slice(*config.TRAIN_PERIOD)
            )
        )
        df = run_delta(
            catalog, args.source, args.experiment, era5_features, stage1, stage2, era5_topo
        )

    out = write_events(df, args.source, args.experiment, args.mode)
    print(f"wrote {len(df)} events -> {out}")


if __name__ == "__main__":
    main()
