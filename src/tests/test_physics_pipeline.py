"""Network-free unit tests for physics_pipeline ingest paths.

The remote ARCO-Zarr path is exercised by the production smoke runs and is
network-bound; here we exercise the local CDS-NetCDF cache path because it
is the one the manuscript actually consumes (the io-partition cgroup blocks
the streaming path; see ``slurm/STATUS.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("metpy")
pytest.importorskip("netCDF4")

from src import config  # noqa: E402
from src.features import physics_pipeline  # noqa: E402


def _fake_pl(tmp_path, year: int, month: int) -> None:
    """Write a CDS-style pressure-level NetCDF for one month."""
    lat = np.linspace(60.0, 25.0, 8, dtype="float32")
    lon = np.linspace(-150.0, -110.0, 10, dtype="float32")  # CDS convention
    plev = np.array(config.PRESSURE_LEVELS, dtype="int32")
    n_t = 4
    valid_time = pd.date_range(f"{year}-{month:02d}-01", periods=n_t, freq="6h")
    shape = (n_t, plev.size, lat.size, lon.size)
    rng = np.random.default_rng(seed=year * 100 + month)
    ds = xr.Dataset(
        {
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), rng.standard_normal(shape).astype("float32")),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), rng.standard_normal(shape).astype("float32")),
            "t": (("valid_time", "pressure_level", "latitude", "longitude"), 250 + 30 * rng.standard_normal(shape).astype("float32")),
            "q": (("valid_time", "pressure_level", "latitude", "longitude"), (1e-3 * rng.uniform(size=shape)).astype("float32")),
            "z": (("valid_time", "pressure_level", "latitude", "longitude"), (1e5 * rng.uniform(size=shape)).astype("float32")),
        },
        coords={
            "valid_time": valid_time,
            "pressure_level": plev,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds.to_netcdf(tmp_path / f"{year}_{month:02d}_pl.nc")


def _fake_sfc(tmp_path, year: int, month: int) -> None:
    lat = np.linspace(60.0, 25.0, 8, dtype="float32")
    lon = np.linspace(-150.0, -110.0, 10, dtype="float32")
    n_t = 4
    valid_time = pd.date_range(f"{year}-{month:02d}-01", periods=n_t, freq="6h")
    ds = xr.Dataset(
        {
            "sp": (("valid_time", "latitude", "longitude"), np.full((n_t, lat.size, lon.size), 1.0e5, dtype="float32")),
        },
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / f"{year}_{month:02d}_sfc.nc")


def _fake_static(tmp_path) -> None:
    lat = np.linspace(60.0, 25.0, 8, dtype="float32")
    lon = np.linspace(-150.0, -110.0, 10, dtype="float32")
    ds = xr.Dataset(
        {
            "lsm": (("latitude", "longitude"), np.full((lat.size, lon.size), 0.3, dtype="float32")),
            "z": (("latitude", "longitude"), np.full((lat.size, lon.size), 1.0e4, dtype="float32")),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "static.nc")


def test_open_local_netcdf_cache_renames_and_shifts(tmp_path) -> None:
    _fake_pl(tmp_path, 2018, 1)
    _fake_pl(tmp_path, 2018, 2)
    _fake_sfc(tmp_path, 2018, 1)
    _fake_sfc(tmp_path, 2018, 2)
    _fake_static(tmp_path)

    ds = physics_pipeline._open_local_netcdf_cache(tmp_path)

    # GRIB short names renamed to ARCO long names.
    for arco_name in [
        config.ERA5_VARS["u"],
        config.ERA5_VARS["v"],
        config.ERA5_VARS["t"],
        config.ERA5_VARS["q"],
        config.ERA5_VARS["z"],
        config.ERA5_VARS["sp"],
        config.ERA5_SURFACE_VARS["lsm"],
        config.ERA5_SURFACE_VARS["z_sfc"],
    ]:
        assert arco_name in ds.data_vars, f"missing {arco_name}"

    # Static z and pressure-level z must not collide.
    assert ds[config.ERA5_VARS["z"]].dims == ("time", "level", "latitude", "longitude") or \
           ds[config.ERA5_VARS["z"]].dims == ("valid_time", "level", "latitude", "longitude") or \
           "level" in ds[config.ERA5_VARS["z"]].dims
    assert "level" not in ds[config.ERA5_SURFACE_VARS["z_sfc"]].dims

    # valid_time -> time, pressure_level -> level.
    assert "time" in ds.dims
    assert "level" in ds.dims

    # Longitudes shifted to 0..360 convention, sorted ascending.
    lons = ds["longitude"].to_numpy()
    assert float(lons.min()) >= 210.0 - 1e-3
    assert float(lons.max()) <= 250.0 + 1e-3
    assert np.all(np.diff(lons) > 0)

    # Two months of pressure-level files concatenated along time.
    assert ds.sizes["time"] == 8


def _fake_pik_tp(pik_dir, year: int, month: int) -> None:
    """Write a PIK-style total_precipitation_{YYYY}{MM}.nc with global var name `tp`.
    Hourly to mirror the real PIK store; the loader resamples to 6-hourly."""
    lat = np.linspace(60.0, 25.0, 8, dtype="float32")
    lon = np.linspace(-150.0, -110.0, 10, dtype="float32")
    valid_time = pd.date_range(f"{year}-{month:02d}-01", periods=24, freq="1h")
    rng = np.random.default_rng(seed=year * 1000 + month)
    ds = xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"), (1e-4 * rng.uniform(size=(24, lat.size, lon.size))).astype("float32"))},
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(pik_dir / f"total_precipitation_{year}{month:02d}.nc")


def test_open_local_cache_tp_pik_fallback(tmp_path, monkeypatch) -> None:
    """No ``*_tp.nc`` in local cache, but ERA5_TP_PIK_DIR points at a PIK-style
    directory — tp must be loaded from there, renamed to ``total_precipitation``,
    resampled to 6-hourly, and merged into the dataset."""
    cache = tmp_path / "cache"
    cache.mkdir()
    pik = tmp_path / "pik_tp"
    pik.mkdir()

    _fake_pl(cache, 2018, 1)
    _fake_sfc(cache, 2018, 1)
    _fake_static(cache)
    _fake_pik_tp(pik, 2018, 1)
    # 2017 must be ignored — only years present in pl_files should pull from PIK.
    _fake_pik_tp(pik, 2017, 1)

    monkeypatch.setenv("ERA5_TP_PIK_DIR", str(pik))
    ds = physics_pipeline._open_local_netcdf_cache(cache)

    tp_name = config.ERA5_SURFACE_VARS["tp"]
    assert tp_name in ds.data_vars
    # 24 hourly steps resampled to 6h => 4 windows; merged via inner-join on time
    # against the 4-step pl/sfc, so result must align on those 4 timestamps.
    assert ds.sizes["time"] == 4
    # Sanity: PIK 2017 entry was not pulled in (only 1 month of pl, year 2018).
    times = ds["time"].to_numpy().astype("datetime64[Y]")
    assert np.all(times == np.datetime64("2018", "Y"))


def test_open_local_cache_via_open_arco_era5(tmp_path, monkeypatch) -> None:
    _fake_pl(tmp_path, 2018, 1)
    _fake_sfc(tmp_path, 2018, 1)
    _fake_static(tmp_path)

    # The env-var override path must trigger when the default URL is in effect.
    monkeypatch.setenv("ERA5_LOCAL_CACHE", str(tmp_path))
    ds = physics_pipeline.open_arco_era5()

    # After _apply_regional_bbox (lon 210..250), all longitudes inside bbox.
    lons = ds["longitude"].to_numpy()
    assert lons.min() >= 210.0 - 1e-3
    assert lons.max() <= 250.0 + 1e-3
    # Variables are in ARCO long-name form ready for downstream physics.
    assert config.ERA5_VARS["u"] in ds.data_vars
    assert config.ERA5_VARS["t"] in ds.data_vars
