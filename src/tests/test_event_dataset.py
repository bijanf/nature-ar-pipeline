"""Network-free unit tests for Stage 2 dataset assembly.

Synthetic-cube coverage of ``integrate_event_precip``: confirm the spatial-mean
× time-integral semantics, and that missing event_ids return NaN cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("metpy")  # event_dataset imports topography (-> physics_pipeline -> metpy)

from src.data.event_dataset import integrate_event_precip  # noqa: E402

_LATS = np.arange(30.0, 40.0, 0.5)
_LONS = np.arange(220.0, 230.0, 0.5)


def _intensity_cube(values: np.ndarray, t0: str = "2010-01-01") -> xr.DataArray:
    times = pd.date_range(t0, periods=values.shape[0], freq="6h")
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": _LATS, "longitude": _LONS},
        name="ar_intensity",
    )


def _tp_cube(constant_m_per_hour: float, t0: str = "2009-12-31", n_hours: int = 72) -> xr.DataArray:
    times = pd.date_range(t0, periods=n_hours, freq="1h")
    return xr.DataArray(
        np.full((n_hours, _LATS.size, _LONS.size), constant_m_per_hour, dtype="float32"),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": _LATS, "longitude": _LONS},
        name="total_precipitation",
    )


def test_integrate_event_precip_constant_field() -> None:
    # One event covering 2 consecutive 6-hourly steps (so 12 hours).
    cube = np.zeros((4, _LATS.size, _LONS.size), dtype="float32")
    cube[1:3, 5:10, 5:10] = 400.0
    intensity = _intensity_cube(cube)

    # ERA5 tp: 0.001 m/hour everywhere -> 1 mm/hour. Event window spans
    # event_start to event_end + 6h = 18 hours of accumulation -> 18 mm.
    tp = _tp_cube(0.001, t0="2009-12-31", n_hours=72)

    events = pd.DataFrame(
        {
            "event_id": [1],
            "start_time": [pd.Timestamp("2010-01-01 06:00")],
            "end_time": [pd.Timestamp("2010-01-01 12:00")],
        }
    )

    out = integrate_event_precip(events, intensity, tp, threshold=100.0)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == 1
    # 13 hourly accumulations between 06:00 and 12:00+6h = [06:00, 18:00] inclusive.
    # spatial mean is the constant, sum over hours = 13 * 1 mm.
    assert row["precip_total_mm"] == pytest.approx(13.0, abs=0.1)


def test_integrate_missing_event_id_returns_nan() -> None:
    cube = np.zeros((4, _LATS.size, _LONS.size), dtype="float32")
    intensity = _intensity_cube(cube)
    tp = _tp_cube(0.001)

    events = pd.DataFrame(
        {
            "event_id": [42],
            "start_time": [pd.Timestamp("2010-01-01 00:00")],
            "end_time": [pd.Timestamp("2010-01-01 06:00")],
        }
    )

    out = integrate_event_precip(events, intensity, tp, threshold=100.0)
    assert np.isnan(out.iloc[0]["precip_total_mm"])


def test_integrate_zero_precip() -> None:
    cube = np.zeros((4, _LATS.size, _LONS.size), dtype="float32")
    cube[1:3, 5:10, 5:10] = 400.0
    intensity = _intensity_cube(cube)
    tp = _tp_cube(0.0)  # dry

    events = pd.DataFrame(
        {
            "event_id": [1],
            "start_time": [pd.Timestamp("2010-01-01 06:00")],
            "end_time": [pd.Timestamp("2010-01-01 12:00")],
        }
    )

    out = integrate_event_precip(events, intensity, tp, threshold=100.0)
    assert out.iloc[0]["precip_total_mm"] == pytest.approx(0.0)
