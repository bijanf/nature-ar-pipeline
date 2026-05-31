"""Fetch one year of ERA5 from CDS into a local NetCDF cache.

Why CDS instead of the ARCO Zarr on GCS:
the io-partition cgroup includes page-cache + aiohttp/gcsfs buffers in
``memory.current`` and saturates the 20 GB cap in ~2 min when streaming
ARCO-ERA5 (see ``slurm/STATUS.md``). CDS instead returns a single
server-side bbox+time subset as a small NetCDF (~2 GB / year), which
fits comfortably under the cap.

Layout written by this module:

    data/cache/era5_bbox_local/
        static.nc                # land_sea_mask + geopotential_at_surface (once)
        2018_01_pl.nc            # u, v, T, q, z at config.PRESSURE_LEVELS, 6-hourly
        2018_01_sfc.nc           # surface_pressure, 6-hourly
        2018_02_pl.nc
        ...

Requests are sliced per month because a per-year query exceeds the CDS
"cost limit" (1 year × 5 vars × 3 levels × 365 days × 4 timesteps =
~21,900 fields, rejected by CDS as too large). Per-month is ~1,800
fields per pressure-level request and ~120 per surface request — safely
under the cap.

Variable names in the resulting NetCDFs follow ECMWF GRIB shortNames
(``u``, ``v``, ``t``, ``q``, ``z``, ``sp``, ``tp``, ``lsm``).
``physics_pipeline.open_arco_era5`` renames them to ARCO long names at the
load boundary so the rest of the pipeline is unchanged.

Usage
-----

    python -m src.data.fetch_cds_era5 --year 2018
    python -m src.data.fetch_cds_era5 --static-only

Each (year, month) NetCDF is fetched independently and skipped if it
already exists, so a job can resume across restarts / OOMs / network
hiccups without re-asking CDS for what it already wrote.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

from src import config

# CDS allows multiple concurrent retrievals per user; the operational guidance
# is up to 8-16, but conservative parallelism avoids polite-citizen issues.
_DEFAULT_CONCURRENCY = 8
_RETRY_DELAY_S = 20

_BBOX = config.BBOX
_AREA = [
    _BBOX["lat_max"],
    _BBOX["lon_min"] - 360.0 if _BBOX["lon_min"] > 180.0 else _BBOX["lon_min"],
    _BBOX["lat_min"],
    _BBOX["lon_max"] - 360.0 if _BBOX["lon_max"] > 180.0 else _BBOX["lon_max"],
]

_PL_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
]
_SFC_VARS = ["surface_pressure"]
_TP_VARS = ["total_precipitation"]
_STATIC_VARS = ["land_sea_mask", "geopotential"]

_TIMES = ["00:00", "06:00", "12:00", "18:00"]
_DAYS = [f"{d:02d}" for d in range(1, 32)]


def _pl_dataset_name(year: int) -> str:
    # The preliminary back-extension was merged into the main ERA5 dataset
    # in late 2024 / early 2025; the standalone back-extension endpoints now
    # return 404. The unified endpoint covers 1940-present.
    return "reanalysis-era5-pressure-levels"


def _sl_dataset_name(year: int) -> str:
    return "reanalysis-era5-single-levels"


def _client() -> cdsapi.Client:
    return cdsapi.Client(quiet=False, verify=True)


# A real CDS NetCDF for our bbox is megabytes; anything tiny is a failed or
# truncated write. A bare ``target.exists()`` let 0-byte files (left by a
# crashed retrieve) masquerade as complete, so they were never re-fetched and
# later crashed the loader with "NetCDF: Unknown file format".
_MIN_VALID_BYTES = 4096


def _already_complete(target: Path) -> bool:
    return target.exists() and target.stat().st_size >= _MIN_VALID_BYTES


def _retrieve_atomic(dataset: str, request: dict, target: Path) -> Path:
    """Retrieve to a sibling ``.tmp`` and atomically rename on success, so an
    interrupted download never leaves a file that looks complete. Any stale
    ``.tmp`` (or undersized ``target``) is cleared first."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    _client().retrieve(dataset, request, str(tmp))
    if tmp.stat().st_size < _MIN_VALID_BYTES:
        tmp.unlink(missing_ok=True)
        raise OSError(f"CDS retrieve wrote an undersized file for {target.name}")
    os.replace(tmp, target)
    return target


def fetch_pressure_level_month(year: int, month: int, out: Path) -> Path:
    target = out / f"{year}_{month:02d}_pl.nc"
    if _already_complete(target):
        return target
    return _retrieve_atomic(
        _pl_dataset_name(year),
        {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": _PL_VARS,
            "pressure_level": [str(p) for p in config.PRESSURE_LEVELS],
            "year": str(year),
            "month": f"{month:02d}",
            "day": _DAYS,
            "time": _TIMES,
            "area": _AREA,
        },
        target,
    )


def fetch_surface_month(year: int, month: int, out: Path) -> Path:
    target = out / f"{year}_{month:02d}_sfc.nc"
    if _already_complete(target):
        return target
    return _retrieve_atomic(
        _sl_dataset_name(year),
        {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": _SFC_VARS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": _DAYS,
            "time": _TIMES,
            "area": _AREA,
        },
        target,
    )


def fetch_tp_month(year: int, month: int, out: Path) -> Path:
    """Fetch hourly total_precipitation for one month. tp is an accumulated
    variable (precip during the preceding hour), so we fetch the full hourly
    record; Stage 2 sums it over each AR event's footprint × duration.
    """
    target = out / f"{year}_{month:02d}_tp.nc"
    if _already_complete(target):
        return target
    hourly_times = [f"{h:02d}:00" for h in range(24)]
    return _retrieve_atomic(
        _sl_dataset_name(year),
        {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": _TP_VARS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": _DAYS,
            "time": hourly_times,
            "area": _AREA,
        },
        target,
    )


_KIND_FN = {
    "pl": fetch_pressure_level_month,
    "sfc": fetch_surface_month,
    "tp": fetch_tp_month,
}


def _fetch_one(spec: tuple[str, int, int, Path]) -> Path:
    """Worker entry point: ``("pl" | "sfc" | "tp", year, month, out)`` with one retry.

    CDS requests fail transiently (queue overload, 5xx). One in-process retry
    after a short sleep covers the common case without compounding load.
    """
    kind, year, month, out = spec
    fn = _KIND_FN[kind]
    try:
        return fn(year, month, out)
    except Exception:
        time.sleep(_RETRY_DELAY_S)
        return fn(year, month, out)


def fetch_year(
    year: int,
    out: Path,
    concurrency: int = _DEFAULT_CONCURRENCY,
    kinds: tuple[str, ...] = ("pl", "sfc"),
) -> list[Path]:
    specs: list[tuple[str, int, int, Path]] = []
    for month in range(1, 13):
        for kind in kinds:
            specs.append((kind, year, month, out))

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_fetch_one, spec) for spec in specs]
        for fut in as_completed(futures):
            paths.append(fut.result())
    return sorted(paths)


def fetch_static_fields(out: Path) -> Path:
    target = out / "static.nc"
    if _already_complete(target):
        return target
    return _retrieve_atomic(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": _STATIC_VARS,
            "year": "1990",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "area": _AREA,
        },
        target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=config.CACHE_DIR / "era5_bbox_local",
        help="Output directory for the per-year NetCDFs.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help="Parallel CDS requests in flight at once (CDS allows up to ~8).",
    )
    parser.add_argument(
        "--kinds",
        type=str,
        nargs="+",
        default=["pl"],
        choices=("pl", "sfc", "tp"),
        help="Subset of fetch kinds: pl (pressure-level), sfc (surface pressure — "
        "not consumed by any feature, kept as an opt-in), tp (precip — read "
        "from PIK climate_data_central by default, opt in only if PIK mirror "
        "is unavailable).",
    )
    args = parser.parse_args()

    if args.static_only:
        path = fetch_static_fields(args.out)
        print(f"static -> {path}")
        return

    if args.year is None:
        parser.error("Pass either --year YYYY or --static-only.")

    static_path = fetch_static_fields(args.out)
    print(f"static -> {static_path}")
    paths = fetch_year(args.year, args.out, concurrency=args.concurrency, kinds=tuple(args.kinds))
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
