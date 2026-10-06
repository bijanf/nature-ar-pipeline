"""Write a placeholder IVT climatology Zarr for one period.

The real `compute_ivt_climatology` triggers a heavy GCS stream of the
full-period IVT and is OOM-killed inside the io partition's 20 GB cap.
This stub creates a tiny on-disk Zarr at the cached path so the
existence check inside `ar_detection.compute_ivt_climatology` short-
circuits and downstream event extraction runs against the real data
without ever recomputing the climatology.

Values are a constant 250 kg m^-1 s^-1 (the GW absolute floor) so the
GW threshold reduces to the floor everywhere — i.e. detector behaves
as a pure absolute-threshold + geometry filter. Adequate for
*pipeline-debugging* runs; do NOT publish numbers produced from a
stubbed climatology. Replace with the real cache before any results.

Usage:
    python slurm/fake_clim.py recent_2015_2024
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from src import config
from src.features import physics_pipeline


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python slurm/fake_clim.py <period_name>", file=sys.stderr)
        sys.exit(2)
    name = sys.argv[1]
    period = dict(config.OBSERVATIONAL_PERIODS).get(name)
    if period is None:
        print(f"unknown period {name!r}", file=sys.stderr)
        sys.exit(2)

    # Reuse the project's bbox-clamp on the lat/lon coords by opening any
    # ERA5 surface field — cheap, no time-step compute.
    ds = xr.open_zarr(
        config.ERA5_ZARR_URL, consolidated=True, storage_options={"token": "anon"}, chunks={}
    )
    ds = physics_pipeline._apply_regional_bbox(ds[[config.ERA5_SURFACE_VARS["lsm"]]])
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    clim = xr.DataArray(
        np.full((12, lat.size, lon.size), 250.0, dtype="float32"),
        dims=("month", "latitude", "longitude"),
        coords={"month": np.arange(1, 13), "latitude": lat, "longitude": lon},
        name="ivt_climatology",
    )

    cache_path = (
        config.CACHE_DIR
        / f"ivt_climatology_{period[0][:4]}_{period[1][:4]}_q{int(config.GW_CLIM_QUANTILE * 100):02d}.zarr"
    )
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clim.to_dataset().to_zarr(cache_path, mode="w", consolidated=True)
    print(f"wrote stub climatology -> {cache_path}")


if __name__ == "__main__":
    main()
