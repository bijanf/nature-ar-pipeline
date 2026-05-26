"""
Stage 2 dataset assembly: join event-level Stage 1 outputs with topography
aggregates and integrated ERA5 precipitation to produce
``data/cache/events_dataset.parquet``.

Pipeline per event
------------------
1. Re-derive footprint and event-time labels from the predicted-intensity cube
   via ``scipy.ndimage.label`` (3-D 26-connectivity, same as
   :mod:`src.models.event_post`).
2. ``event_topo_features`` -> mean / max / std elevation, land fraction.
3. ``integrate_event_precip`` -> spatially-averaged total ERA5 precipitation
   depth over the event's footprint × event-time window, in mm.

The precipitation target is the **mean depth across the event's spatial
footprint, summed over the event's hourly time window** (units: mm). That
quantity scales with both event size and intensity in a hydrologist-friendly
way and is what Stage 2's quantile regression predicts.

ERA5 ``total_precipitation`` is hourly accumulation in metres; we open it at
native hourly cadence (no 6-hourly subsample) so the time integral preserves
all accumulated precip during the event.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr

from src import config
from src.features import physics_pipeline, topography

# =============================================================================
# Precip integration
# =============================================================================


def integrate_event_precip(
    events_df: pd.DataFrame,
    predicted_intensity: xr.DataArray,
    era5_tp: xr.DataArray,
    threshold: float = config.STAGE1_EVENT_THRESHOLD,
) -> pd.DataFrame:
    """For each event, return spatially-averaged ERA5 precip depth integrated
    over the event's footprint × hourly time window (mm)."""
    lat_name = "latitude" if "latitude" in predicted_intensity.coords else "lat"
    lon_name = "longitude" if "longitude" in predicted_intensity.coords else "lon"

    arr = predicted_intensity.transpose("time", lat_name, lon_name).to_numpy()
    labels, _ = ndi.label(arr > threshold, structure=np.ones((3, 3, 3), dtype=bool))

    rows = []
    for _, ev in events_df.iterrows():
        eid = int(ev["event_id"])
        obj = labels == eid
        if not obj.any():
            rows.append({"event_id": eid, "precip_total_mm": float("nan")})
            continue

        footprint = obj.any(axis=0)
        i_idx, j_idx = np.where(footprint)
        n_cells = i_idx.size

        # Event hourly window: ERA5 tp at t encodes precip accumulated in the
        # hour ending at t, so we sweep the full closed interval [start, end+6h).
        t_start = pd.Timestamp(ev["start_time"])
        t_end = pd.Timestamp(ev["end_time"]) + pd.Timedelta(hours=6)

        tp_window = era5_tp.sel(time=slice(t_start, t_end)).isel(
            {
                lat_name: xr.DataArray(i_idx, dims="cell"),
                lon_name: xr.DataArray(j_idx, dims="cell"),
            }
        )

        # Spatial mean per hour, summed in time -> total depth in metres.
        depth_m = float(tp_window.sum().compute() / n_cells)
        rows.append({"event_id": eid, "precip_total_mm": depth_m * 1000.0})

    return pd.DataFrame.from_records(rows)


# =============================================================================
# Top-level join
# =============================================================================


def build_dataset(
    events_df: pd.DataFrame,
    predicted_intensity: xr.DataArray,
    era5_topo: xr.Dataset,
    era5_tp: xr.DataArray,
    threshold: float = config.STAGE1_EVENT_THRESHOLD,
) -> pd.DataFrame:
    """Join Stage 1 events + topography + ERA5 precip into one Parquet-ready table."""
    topo_feats = topography.event_topo_features(
        events_df, era5_topo, predicted_intensity, threshold=threshold
    )
    precip_feats = integrate_event_precip(
        events_df, predicted_intensity, era5_tp, threshold=threshold
    )
    return events_df.merge(topo_feats, on="event_id").merge(precip_feats, on="event_id")


def write_dataset(df: pd.DataFrame, name: str = "events_dataset") -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CACHE_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    return out


# =============================================================================
# CLI
# =============================================================================


def _open_era5_tp() -> xr.DataArray:
    """Open ERA5 hourly total_precipitation from the project Zarr, bbox-sliced."""
    ds = xr.open_zarr(
        config.ERA5_ZARR_URL,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks={},
    )
    tp = ds[config.ERA5_SURFACE_VARS["tp"]]
    return physics_pipeline._apply_regional_bbox(tp.to_dataset())[config.ERA5_SURFACE_VARS["tp"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=str,
        default=str(config.CACHE_DIR / "events.parquet"),
        help="Path to events Parquet from src.models.event_post.",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        required=True,
        help="Path to predicted-intensity Zarr or NetCDF (dims: time, lat, lon).",
    )
    parser.add_argument("--threshold", type=float, default=config.STAGE1_EVENT_THRESHOLD)
    parser.add_argument("--name", type=str, default="events_dataset")
    args = parser.parse_args()

    events_df = pd.read_parquet(args.events)
    intensity = (
        xr.open_zarr(args.intensity)["ar_intensity"]
        if args.intensity.endswith(".zarr")
        else xr.open_dataset(args.intensity)["ar_intensity"]
    )

    topo = topography.open_era5_topography()
    tp = _open_era5_tp()

    df = build_dataset(events_df, intensity, topo, tp, threshold=args.threshold)
    out = write_dataset(df, name=args.name)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
