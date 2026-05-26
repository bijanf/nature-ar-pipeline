"""
Event post-processing: convert a continuous AR-intensity field into discrete
AR-event records.

Pipeline
--------
1. Threshold the predicted intensity field at ``STAGE1_EVENT_THRESHOLD``
   (100 kg m⁻¹ s⁻¹). Below that, the predicted signal is too weak to be an AR.
2. Run **3-D connected-component labelling** over ``(time, lat, lon)`` so a
   single physical AR that persists across multiple 6-hourly steps gets *one*
   event id. The 26-connectivity structuring element joins pixels that share a
   face *or* an edge in the space-time cube — appropriate because AR landfall
   tracks typically translate by ≲ 250 km between 6-hour steps, well within a
   one-pixel diagonal at 0.25°.
3. Per-event aggregates:

   ====================  ======================================================
   ``mean_intensity``    mean predicted IVT over event pixels (kg m⁻¹ s⁻¹)
   ``max_intensity``     max predicted IVT over event pixels (kg m⁻¹ s⁻¹)
   ``footprint_area_km2`` area of the event's spatial union (km²)
   ``duration_hours``    time span from first to last 6-hourly slice + 6 h
   ``landfall_lat``      mean latitude of *land* pixels in the event (NaN if
                         none — i.e. an over-ocean AR)
   ``start_time``        first timestamp in the event
   ``end_time``          last timestamp in the event
   ====================  ======================================================

The footprint area is computed on the actual 0.25° lat/lon grid with
``cos(lat)`` weighting so the cells at 60° N don't get the same area as those
at 25° N.

I/O
---
Caller passes a lazy or in-memory ``xr.DataArray`` of predicted intensity
plus an ``xr.DataArray`` of the ERA5 ``land_sea_mask`` (used for landfall
classification). The labelled cube is materialised once via ``.compute()`` —
event extraction is fundamentally a global operation, no way to keep it lazy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr

from src import config

_KM_PER_DEG_LAT = 111.0


def extract_events(
    intensity: xr.DataArray,
    land_sea_mask: xr.DataArray | None = None,
    threshold: float = config.STAGE1_EVENT_THRESHOLD,
) -> pd.DataFrame:
    """Threshold + 3-D label + per-event aggregate. Returns a DataFrame indexed
    on a synthetic ``event_id``."""
    lat_name = "latitude" if "latitude" in intensity.coords else "lat"
    lon_name = "longitude" if "longitude" in intensity.coords else "lon"

    arr = intensity.transpose("time", lat_name, lon_name).to_numpy()
    binary = arr > threshold

    # 3-D 26-connectivity: any of 26 neighbours (faces+edges+corners) joins.
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, n_events = ndi.label(binary, structure=structure)
    if n_events == 0:
        return _empty_events_frame()

    times = pd.to_datetime(intensity["time"].to_numpy())
    lats = intensity[lat_name].to_numpy().astype("float64")
    lons = intensity[lon_name].to_numpy().astype("float64")

    # Per-cell area on the actual grid (km²), broadcast to (lat, lon).
    dlat = np.abs(np.diff(lats)).mean()
    dlon = np.abs(np.diff(lons)).mean()
    cell_area = (
        (dlat * _KM_PER_DEG_LAT) * (dlon * _KM_PER_DEG_LAT) * np.cos(np.deg2rad(lats))[:, None]
    )  # (lat, lon)

    if land_sea_mask is not None:
        land = (
            land_sea_mask.transpose(lat_name, lon_name)
            .to_numpy()
            .astype("float32")
            .reshape(1, lats.size, lons.size)
        )
    else:
        land = np.zeros((1, lats.size, lons.size), dtype="float32")

    records: list[dict[str, float | str]] = []
    for event_id in range(1, n_events + 1):
        obj = labels == event_id
        if obj.sum() < 2:
            # Single-pixel-time blip — drop without recording.
            continue
        intensities = arr[obj]

        # Spatial footprint = union over time of the (lat, lon) cells touched.
        footprint = obj.any(axis=0)
        area_km2 = float((cell_area * footprint).sum())

        # Time span. Step is 6 h; duration = last - first + 6h.
        time_idx = np.where(obj.any(axis=(1, 2)))[0]
        start_time = pd.Timestamp(times[time_idx[0]])
        end_time = pd.Timestamp(times[time_idx[-1]])
        duration_h = float((end_time - start_time).total_seconds() / 3600.0 + 6.0)

        # Landfall lat: mean lat of the land pixels inside the footprint.
        # 'land' is (1, lat, lon); footprint is (lat, lon).
        land_in_footprint = (land[0] > 0.5) & footprint
        if land_in_footprint.any():
            i_land, _ = np.where(land_in_footprint)
            landfall_lat = float(lats[i_land].mean())
        else:
            landfall_lat = float("nan")

        records.append(
            {
                "event_id": int(event_id),
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": duration_h,
                "mean_intensity": float(intensities.mean()),
                "max_intensity": float(intensities.max()),
                "footprint_area_km2": area_km2,
                "landfall_lat": landfall_lat,
            }
        )

    return pd.DataFrame.from_records(records)


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": pd.Series(dtype="int64"),
            "start_time": pd.Series(dtype="datetime64[ns]"),
            "end_time": pd.Series(dtype="datetime64[ns]"),
            "duration_hours": pd.Series(dtype="float64"),
            "mean_intensity": pd.Series(dtype="float64"),
            "max_intensity": pd.Series(dtype="float64"),
            "footprint_area_km2": pd.Series(dtype="float64"),
            "landfall_lat": pd.Series(dtype="float64"),
        }
    )


def write_events(events: pd.DataFrame, name: str = "events") -> Path:
    out = config.CACHE_DIR / f"{name}.parquet"
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out, index=False, compression="zstd")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intensity",
        type=str,
        required=True,
        help="Path to a Zarr or NetCDF intensity field with dims (time, lat, lon).",
    )
    parser.add_argument(
        "--land-sea-mask",
        type=str,
        default="",
        help="Optional path to an ERA5 land-sea-mask Zarr/NetCDF for landfall classification.",
    )
    parser.add_argument("--threshold", type=float, default=config.STAGE1_EVENT_THRESHOLD)
    parser.add_argument("--name", type=str, default="events")
    args = parser.parse_args()

    intensity = (
        xr.open_zarr(args.intensity)["ar_intensity"]
        if args.intensity.endswith(".zarr")
        else xr.open_dataset(args.intensity)["ar_intensity"]
    )
    lsm = None
    if args.land_sea_mask:
        ds = (
            xr.open_zarr(args.land_sea_mask)
            if args.land_sea_mask.endswith(".zarr")
            else xr.open_dataset(args.land_sea_mask)
        )
        lsm = ds["land_sea_mask"] if "land_sea_mask" in ds else ds[next(iter(ds.data_vars))]

    events = extract_events(intensity, land_sea_mask=lsm, threshold=args.threshold)
    out = write_events(events, name=args.name)
    print(f"wrote {len(events)} events -> {out}")


if __name__ == "__main__":
    main()
