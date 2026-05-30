"""Fetch GLOBAL ERA5 monthly-mean pressure-level fields for the three windows.

The 6-hourly regional bbox cache powers AR detection (a corridor problem). The
thermodynamic/dynamic IVT decomposition, by contrast, needs only per-window
time-mean q, u, v, T, z per level — so we can compute it *globally* from monthly
means at a coarse grid. This is ~100x smaller than the 6-hourly stream and
never touches the io-partition OOM wall.

One file per window (global, 1 deg, 5 variables x 3 levels x all months of the
window), written to a project-disk directory (NOT $HOME).
"""

from __future__ import annotations

import os
from pathlib import Path

import cdsapi

OUT = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_MIN_BYTES = 4096

WINDOWS = {
    "presat": range(1940, 1960),
    "modern": range(1980, 2000),
    "recent": range(2015, 2025),
}
_VARS = [
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
]
_LEVELS = ["850", "500", "250"]


def fetch_window(name: str, years: range) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}_monthly_global.nc"
    if target.exists() and target.stat().st_size >= _MIN_BYTES:
        print(f"skip {target} (exists)")
        return target
    tmp = target.with_suffix(".nc.tmp")
    cdsapi.Client().retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": _VARS,
            "pressure_level": _LEVELS,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "grid": [1.0, 1.0],
            "data_format": "netcdf",
        },
        str(tmp),
    )
    if tmp.stat().st_size < _MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise OSError(f"undersized download for {name}")
    os.replace(tmp, target)
    print(f"wrote {target} ({target.stat().st_size/1e6:.0f} MB)")
    return target


def main() -> None:
    for name, years in WINDOWS.items():
        fetch_window(name, years)


if __name__ == "__main__":
    main()
