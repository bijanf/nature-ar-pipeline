"""Network-free unit tests for event post-processing.

Build small synthetic ``(time, lat, lon)`` intensity cubes and verify that
``extract_events`` produces sensible event records: single events count as
one, persistent events get one id across time, and landfall classification
respects the land/sea mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from src.models.event_post import extract_events  # noqa: E402

# A small 0.5° grid keeps the synthetic cube cheap.
_LATS = np.arange(30.0, 50.0, 0.5)
_LONS = np.arange(220.0, 240.0, 0.5)
_TIMES = pd.date_range("2010-01-01", periods=8, freq="6h")


def _intensity_cube(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": _TIMES, "latitude": _LATS, "longitude": _LONS},
        name="ar_intensity",
    )


def test_no_events_returns_empty_frame() -> None:
    cube = np.zeros((_TIMES.size, _LATS.size, _LONS.size), dtype="float32")
    df = extract_events(_intensity_cube(cube), threshold=100.0)
    assert df.empty
    assert {"event_id", "start_time", "mean_intensity"}.issubset(df.columns)


def test_one_persistent_event_gets_one_id() -> None:
    cube = np.zeros((_TIMES.size, _LATS.size, _LONS.size), dtype="float32")
    # Same spatial blob for 4 consecutive 6-hourly steps.
    cube[1:5, 10:20, 10:25] = 400.0

    df = extract_events(_intensity_cube(cube), threshold=100.0)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["mean_intensity"] == pytest.approx(400.0)
    assert row["max_intensity"] == pytest.approx(400.0)
    # 4 steps -> duration = (3 * 6) + 6 = 24 hours.
    assert row["duration_hours"] == pytest.approx(24.0)


def test_two_disjoint_events_get_two_ids() -> None:
    cube = np.zeros((_TIMES.size, _LATS.size, _LONS.size), dtype="float32")
    cube[0:2, 2:8, 2:8] = 300.0  # AR-like blob in the bottom-left, t=0-1
    cube[5:7, 30:35, 30:35] = 500.0  # disjoint blob top-right at t=5-6

    df = extract_events(_intensity_cube(cube), threshold=100.0)

    assert len(df) == 2
    assert set(df["max_intensity"].astype(int).tolist()) == {300, 500}


def test_below_threshold_drops() -> None:
    cube = np.zeros((_TIMES.size, _LATS.size, _LONS.size), dtype="float32")
    cube[1:3, 10:15, 10:15] = 50.0  # below the 100 floor

    df = extract_events(_intensity_cube(cube), threshold=100.0)
    assert df.empty


def test_landfall_lat_uses_land_sea_mask() -> None:
    cube = np.zeros((_TIMES.size, _LATS.size, _LONS.size), dtype="float32")
    cube[2:4, 5:15, 5:15] = 300.0

    # Only the upper half of the (lat, lon) plane is land.
    lsm = np.zeros((_LATS.size, _LONS.size), dtype="float32")
    lsm[20:, :] = 1.0
    lsm_da = xr.DataArray(
        lsm, dims=("latitude", "longitude"), coords={"latitude": _LATS, "longitude": _LONS}
    )

    df = extract_events(_intensity_cube(cube), land_sea_mask=lsm_da, threshold=100.0)
    assert len(df) == 1
    # Event sits in lat indices 5-14, which are below the land threshold ->
    # there are no land pixels in the footprint, landfall_lat must be NaN.
    assert np.isnan(df.iloc[0]["landfall_lat"])
