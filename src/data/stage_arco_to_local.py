"""Pre-stage one year of ARCO-ERA5 bbox to local Zarr.

The io-partition cgroup OOMs the streaming pipeline at 20 GB regardless
of what Python does (see ``slurm/STATUS.md`` for the diagnostic chain).
This module bypasses that wall by writing a *local* Zarr cache of the
bbox-sliced ERA5, one year at a time, so subsequent jobs read from disk
on the ``standard`` partition (no internet, no cgroup limit) instead of
streaming from GCS.

Per-year volume (at 6-hourly cadence, the West Coast bbox, the five
pressure-level variables we use):

    1,461 timesteps × 22,701 cells × 6 levels × 5 vars × 4 bytes ≈ 4 GB

So a full 1940-2024 stage is ≈ 350 GB on ``/p/tmp/$USER`` or ``data/cache``.

Each year is written independently; partial progress survives a restart.
Resumes by skipping years whose output zarr already exists. The streaming
load per year is bounded (~4 GB raw downloaded), small enough that the
io partition's 20 GB cap is not in danger.

Usage
-----

    python -m src.data.stage_arco_to_local --year 1985 \\
        --out /p/tmp/$USER/era5_bbox_local

A simple SLURM submission helper would iterate ``--year`` over 1940-2024.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dask
import xarray as xr

from src import config
from src.features import physics_pipeline

dask.config.set(scheduler="synchronous")


_VARS = [*list(physics_pipeline.required_era5_vars()), config.ERA5_SURFACE_VARS["tp"]]


def stage_year(year: int, out_root: Path) -> Path:
    """Write one year's bbox + 6-hourly ERA5 to a local Zarr.

    The output is a self-contained Zarr group at ``out_root / f"{year}.zarr"``
    carrying all variables this project consumes. Skipped (no-op) if the
    target already exists, so the function is idempotent.
    """
    target = out_root / f"{year}.zarr"
    if target.exists():
        return target

    ds = xr.open_zarr(
        config.ERA5_ZARR_URL,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks={},
    )
    ds = physics_pipeline._apply_regional_bbox(ds[_VARS])
    ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    ds = ds.isel(time=slice(None, None, 6))

    out_root.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(target, mode="w", consolidated=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.CACHE_DIR / "era5_bbox_local",
        help="Root directory for per-year zarr stores.",
    )
    args = parser.parse_args()

    out = stage_year(args.year, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
