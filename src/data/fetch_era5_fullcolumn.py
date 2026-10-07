"""Fetch ERA5 monthly means for the full-column moisture-transport decomposition.

Two products, one file per year, global 1 degree grid, 1940-2024:

* single levels: vertical integrals of eastward and northward water-vapour flux
  (monthly means of hourly values, so they include sub-monthly transient
  transport), total column water vapour, surface pressure, sea-surface
  temperature, 2 m temperature and mean sea-level pressure;
* pressure levels: specific humidity, zonal and meridional wind and temperature
  on all 23 ERA5 levels from 1000 to 200 hPa, used for the mean-flow,
  level-by-level split of the transport change.

Usage: python -m src.data.fetch_era5_fullcolumn --kind {sl,pl} --years 1940 2024
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi

ROOT = Path("/p/projects/poem/fallah/nature_ar_data/era5_fullcolumn")
_MIN_BYTES = 4096

SL_VARS = [
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
    "total_column_water_vapour",
    "surface_pressure",
    "sea_surface_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
]
PL_VARS = [
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
]
PL_LEVELS = [
    "200", "225", "250", "300", "350", "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800", "825", "850", "875", "900", "925",
    "950", "975", "1000",
]


def _request(kind: str, year: int) -> tuple[str, dict]:
    req = {
        "product_type": "monthly_averaged_reanalysis",
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "grid": [1.0, 1.0],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if kind == "sl":
        req["variable"] = SL_VARS
        return "reanalysis-era5-single-levels-monthly-means", req
    req["variable"] = PL_VARS
    req["pressure_level"] = PL_LEVELS
    return "reanalysis-era5-pressure-levels-monthly-means", req


def fetch(kind: str, year: int) -> str:
    out = ROOT / kind
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"era5_{kind}_monthly_{year}.nc"
    if target.exists() and target.stat().st_size >= _MIN_BYTES:
        return f"skip {target.name}"
    tmp = target.with_suffix(".nc.tmp")
    dataset, req = _request(kind, year)
    cdsapi.Client(quiet=True).retrieve(dataset, req, str(tmp))
    if tmp.stat().st_size < _MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise OSError(f"undersized download {kind} {year}")
    os.replace(tmp, target)
    return f"wrote {target.name} ({target.stat().st_size / 1e6:.0f} MB)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sl", "pl"], required=True)
    ap.add_argument("--years", nargs=2, type=int, default=[1940, 2024])
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    years = range(a.years[0], a.years[1] + 1)
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = {y: ex.submit(fetch, a.kind, y) for y in years}
        for y, f in futs.items():
            try:
                print(y, f.result(), flush=True)
            except Exception as e:  # keep going; rerun fills gaps
                print(y, "FAILED", e, flush=True)


if __name__ == "__main__":
    main()
