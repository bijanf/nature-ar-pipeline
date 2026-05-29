"""
Per-event topography aggregates for Stage 2.

The ERA5 ARCO Zarr already carries the model's own ``geopotential_at_surface``
and ``land_sea_mask``; using them keeps Stage 2's inputs self-consistent with
the ERA5 precip we will use as the Stage 2 target. The native 0.25° resolution
matches the grid Stage 1 predicts on, so no regridding is needed.

Surface elevation is recovered from the geopotential field via :math:`Z = \\Phi/g`
using the WMO standard gravity constant from
:mod:`src.features.physics_pipeline`.

Public surface
--------------
``open_era5_topography()`` — lazy ``xr.Dataset`` with two variables,
``elevation_m`` and ``land_sea_mask``, sliced to ``config.BBOX``.

``event_topo_features(events_df, topo, predicted_intensity)`` — for each row
in ``events_df`` (one labelled AR event), aggregate over the event's spatial
footprint (the union over time of the pixels above the Stage-1 threshold) and
return the following columns:

==================  =========================================================
``mean_elev_m``     mean surface elevation over the footprint (metres)
``max_elev_m``      max surface elevation over the footprint (metres)
``std_elev_m``      std-dev of surface elevation over the footprint (metres)
``land_fraction``   fraction of footprint pixels classified as land (``lsm > 0.5``)
==================  =========================================================

The function expects the same ``ar_intensity`` cube that produced the events,
so it can re-derive each event's footprint from the labelled connected
components without persisting them in the events Parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr

from src import config
from src.features import physics_pipeline


def open_era5_topography(zarr_url: str = config.ERA5_ZARR_URL) -> xr.Dataset:
    """Lazy ERA5 topography dataset: surface elevation + land-sea mask, bbox-sliced."""
    if zarr_url == config.ERA5_ZARR_URL:
        env_override = os.environ.get("ERA5_LOCAL_CACHE")
        if env_override:
            zarr_url = env_override

    if "://" not in zarr_url:
        path = Path(zarr_url)
        if path.is_dir() and (path / "static.nc").exists():
            return _open_topography_from_static_nc(path / "static.nc")

    ds = xr.open_zarr(zarr_url, consolidated=True, storage_options={"token": "anon"}, chunks={})
    z_sfc = ds[config.ERA5_SURFACE_VARS["z_sfc"]] / physics_pipeline._G
    lsm = ds[config.ERA5_SURFACE_VARS["lsm"]]

    # ARCO-ERA5 v3 stores the static fields on the full hourly time axis
    # (1900..2050) but only populates them inside the ERA5 record itself
    # (1940..2024). Selecting time=0 (1900-01-01) returns NaN. Pick a date
    # known to lie in the satellite era; the fields are time-invariant.
    static_t = "1990-01-01"
    if "time" in z_sfc.dims:
        z_sfc = z_sfc.sel(time=static_t, method="nearest").drop_vars("time", errors="ignore")
    if "time" in lsm.dims:
        lsm = lsm.sel(time=static_t, method="nearest").drop_vars("time", errors="ignore")

    topo = xr.Dataset(
        {"elevation_m": z_sfc.astype("float32"), "land_sea_mask": lsm.astype("float32")}
    )
    return physics_pipeline._apply_regional_bbox(topo)


def _open_topography_from_static_nc(static_nc: Path) -> xr.Dataset:
    """Build the topography dataset from the CDS static.nc fetched by
    ``src.data.fetch_cds_era5``. The file carries ``lsm`` and surface
    geopotential ``z`` (m^2/s^2) over the West-Coast bbox.
    """
    ds = xr.open_dataset(str(static_nc), engine="netcdf4")
    if "valid_time" in ds.dims:
        ds = ds.isel(valid_time=0).drop_vars("valid_time", errors="ignore")
    if "time" in ds.dims:
        ds = ds.isel(time=0).drop_vars("time", errors="ignore")
    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)

    z_sfc = (ds["z"] / physics_pipeline._G).astype("float32")
    lsm = ds["lsm"].astype("float32")
    topo = xr.Dataset({"elevation_m": z_sfc, "land_sea_mask": lsm})

    lon_name = "longitude" if "longitude" in topo.coords else "lon"
    topo = topo.assign_coords({lon_name: topo[lon_name] % 360.0}).sortby(lon_name)
    return physics_pipeline._apply_regional_bbox(topo)


def event_topo_features(
    events_df: pd.DataFrame,
    topo: xr.Dataset,
    predicted_intensity: xr.DataArray,
    threshold: float = config.STAGE1_EVENT_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate elevation + land fraction over each event's spatial footprint.

    ``predicted_intensity`` must be the same cube that produced ``events_df``,
    in the same ``(time, lat, lon)`` order; we re-label it here rather than
    plumb the labels through the events Parquet.
    """
    lat_name = "latitude" if "latitude" in predicted_intensity.coords else "lat"
    lon_name = "longitude" if "longitude" in predicted_intensity.coords else "lon"

    arr = predicted_intensity.transpose("time", lat_name, lon_name).to_numpy()
    labels, _ = ndi.label(arr > threshold, structure=np.ones((3, 3, 3), dtype=bool))

    elev = topo["elevation_m"].transpose(lat_name, lon_name).to_numpy()
    land = topo["land_sea_mask"].transpose(lat_name, lon_name).to_numpy()

    rows = []
    for event_id in events_df["event_id"].to_numpy():
        obj = labels == int(event_id)
        if not obj.any():
            rows.append(_nan_row(int(event_id)))
            continue
        footprint = obj.any(axis=0)
        elev_in = elev[footprint]
        land_in = land[footprint]
        rows.append(
            {
                "event_id": int(event_id),
                "mean_elev_m": float(np.mean(elev_in)),
                "max_elev_m": float(np.max(elev_in)),
                "std_elev_m": float(np.std(elev_in)),
                "land_fraction": float(np.mean(land_in > 0.5)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _nan_row(event_id: int) -> dict[str, float | int]:
    return {
        "event_id": event_id,
        "mean_elev_m": float("nan"),
        "max_elev_m": float("nan"),
        "std_elev_m": float("nan"),
        "land_fraction": float("nan"),
    }
