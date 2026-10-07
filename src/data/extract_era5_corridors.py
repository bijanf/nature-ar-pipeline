"""Cut the yearly global ERA5 files into one file per corridor, and join the
global single-level fields into one file.

Inputs: era5_fullcolumn/{sl,pl}/era5_{sl,pl}_monthly_YYYY.nc (1940-2024).
Outputs:
* era5_corridors/<key>.nc: q, u, v, t on 23 levels and viwve, viwvn, tcwv, sp
  over the corridor box, monthly 1940-2024;
* era5_fullcolumn/era5_sl_monthly_1940_2024.nc: all single-level fields, global.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from src.figures.fig_global_ar_regions import _REGIONS

DATA = Path("/p/projects/poem/fallah/nature_ar_data")
SRC = DATA / "era5_fullcolumn"
OUT = DATA / "era5_corridors"
YEARS = range(1940, 2025)


def corridor_key(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _clean(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])


def _box(ds, box):
    lon0, lon1, lat0, lat1 = box
    return ds.sel(longitude=slice(lon0, min(lon1, 359.0)), latitude=slice(lat1, lat0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sl = xr.open_mfdataset([SRC / "sl" / f"era5_sl_monthly_{y}.nc" for y in YEARS],
                           combine="by_coords", preprocess=_clean)
    sl.to_netcdf(SRC / "era5_sl_monthly_1940_2024.nc")
    pl = xr.open_mfdataset([SRC / "pl" / f"era5_pl_monthly_{y}.nc" for y in YEARS],
                           combine="by_coords", preprocess=_clean)
    for name, *box in _REGIONS:
        ds = xr.merge([_box(pl[["q", "u", "v", "t"]], box),
                       _box(sl[["viwve", "viwvn", "tcwv", "sp"]], box)])
        ds = ds.astype("float32").load()
        ds.to_netcdf(OUT / f"{corridor_key(name)}.nc")
        print(name, dict(ds.sizes), flush=True)


if __name__ == "__main__":
    main()
