"""Fetch GLOBAL ERA5 monthly-mean CONVECTIVE single-level fields (CAPE, CIN).

Proof-of-concept for the severe-convective-storm efficiency direction. The
thermodynamic half of a CAPE x shear severe-environment proxy is CAPE/CIN, which
ERA5 carries as *single-level* fields (back to 1940). The dynamic half (deep-layer
bulk shear) is derived from the existing pressure-level monthly files in the
sibling ``era5_global_monthly`` directory, so no extra wind download is needed.

One file per window (global, 1 deg, CAPE + CIN, all months), written to a
project-disk directory (NOT $HOME). Mirrors fetch_cds_era5_monthly_global.py
(atomic .tmp -> rename, size gate) so the two products live side by side and the
five windows tile into a continuous 1940-2024 record.

CAVEAT: a *monthly mean* of CAPE strongly understates convective potential
(CAPE is intermittent and right-skewed). This is adequate for a first-look
trend + infrastructure-transfer test; the production study needs sub-daily CAPE.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import cdsapi

OUT = Path("/p/projects/poem/fallah/nature_ar_data/era5_convective_monthly")
_MIN_BYTES = 4096

# Same five windows as the pressure-level monthly product, so CAPE/CIN tiles
# exactly onto the existing winds for a continuous 1940-2024 series.
WINDOWS = {
    "presat": range(1940, 1960),
    "gap1_1960_1979": range(1960, 1980),
    "modern": range(1980, 2000),
    "gap2_2000_2014": range(2000, 2015),
    "recent": range(2015, 2025),
}
_VARS = [
    "convective_available_potential_energy",
    "convective_inhibition",
]


def _flatten(path: Path) -> Path:
    """The single-levels product returns a ZIP bundling one netCDF per variable
    (CAPE, CIN). Merge the inner files into one flat netCDF at ``path``, so
    downstream code sees a single ``cape``/``cin`` dataset like the pressure-level
    product. Idempotent: a non-zip file is left untouched."""
    import xarray as xr
    if not zipfile.is_zipfile(path):
        return path
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(path) as z:
            z.extractall(td)
        inner = sorted(Path(td).glob("*.nc"))
        dss = [xr.open_dataset(p) for p in inner]
        # The inner CAPE/CIN files carry the SAME months but with subtly
        # different time-coordinate metadata (different internal stepType), so a
        # plain outer-join merge unions them and DOUBLES the time axis. Force a
        # common time axis positionally (all windows have identical month sets)
        # before merging.
        ref = next((d for d in dss if "cape" in d.data_vars), dss[0])
        tref = "valid_time" if "valid_time" in ref.dims else "time"
        aligned = []
        for d in dss:
            tn = "valid_time" if "valid_time" in d.dims else "time"
            if d.sizes[tn] == ref.sizes[tref]:
                d = d.rename({tn: tref}) if tn != tref else d
                d = d.assign_coords({tref: ref[tref].values})
            aligned.append(d)
        ds = xr.merge(aligned, join="override", compat="override")
        flat = path.with_suffix(".flat.nc")
        ds.to_netcdf(flat)
        ds.close()
    os.replace(flat, path)
    return path


def fetch_window(name: str, years: range) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}_conv_monthly_global.nc"
    if target.exists() and target.stat().st_size >= _MIN_BYTES:
        print(f"skip {target} (exists)")
        return target
    tmp = target.with_suffix(".nc.tmp")
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": _VARS,
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
    _flatten(target)
    print(f"wrote {target} ({target.stat().st_size/1e6:.0f} MB)")
    return target


def main() -> None:
    for name, years in WINDOWS.items():
        fetch_window(name, years)


def reprocess() -> None:
    """Flatten any zip-archived window files already on disk (one-off repair for
    downloads made before _flatten was wired into fetch_window)."""
    for name in WINDOWS:
        p = OUT / f"{name}_conv_monthly_global.nc"
        if p.exists():
            _flatten(p)
            print(f"flattened {p.name}")


if __name__ == "__main__":
    main()
