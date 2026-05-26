"""Invariants of the project configuration and physics pipeline contract.

These tests are network-free and run in CI on every push. They guard the
load-bearing constants and the lazy-graph chunk policy.
"""

from __future__ import annotations

from src import config


def test_bbox_covers_us_west_coast_landfall() -> None:
    assert config.BBOX == {
        "lat_min": 25.0,
        "lat_max": 60.0,
        "lon_min": 210.0,
        "lon_max": 250.0,
    }


def test_pressure_levels_holton_triad() -> None:
    # 850 / 500 / 250 hPa cover the lower-troposphere moisture, mid-troposphere
    # steering flow, and upper-troposphere jet / QG-PV layers respectively.
    assert config.PRESSURE_LEVELS == [850, 500, 250]


def test_required_era5_vars() -> None:
    # Lazy import: the physics module pulls metpy at import time. Keeping the
    # import inside the test lets the other invariants run in environments that
    # only have stdlib + the project (e.g. a stripped-down lint job).
    from src.features import physics_pipeline

    needed = set(physics_pipeline.required_era5_vars())
    assert needed == {
        config.ERA5_VARS["u"],
        config.ERA5_VARS["v"],
        config.ERA5_VARS["t"],
        config.ERA5_VARS["q"],
        config.ERA5_VARS["z"],
    }


def test_chunks_keep_vertical_and_spatial_whole() -> None:
    # Dask parallelism is along time only. Splitting level / latitude / longitude
    # would corrupt MetPy spherical derivatives and the vertical IVT integral.
    chunks = config.DEFAULT_CHUNKS
    assert chunks["time"] == 240
    assert chunks["level"] == -1
    assert chunks["latitude"] == -1
    assert chunks["longitude"] == -1


def test_cmip6_query_pins_ssp585() -> None:
    assert config.CMIP6_QUERY["experiment_id"] == "ssp585"
    assert "ua" in config.CMIP6_QUERY["variable_id"]
    assert "va" in config.CMIP6_QUERY["variable_id"]
