"""Network-free unit tests for the per-event topography aggregator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("metpy")  # topography imports physics_pipeline (-> metpy)

from src.features.topography import event_topo_features  # noqa: E402

_LATS = np.arange(30.0, 50.0, 0.5)
_LONS = np.arange(220.0, 240.0, 0.5)
_TIMES = pd.date_range("2010-01-01", periods=4, freq="6h")


def _intensity(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": _TIMES, "latitude": _LATS, "longitude": _LONS},
        name="ar_intensity",
    )


def _topo(elev: np.ndarray, lsm: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {
            "elevation_m": xr.DataArray(elev, dims=("latitude", "longitude")),
            "land_sea_mask": xr.DataArray(lsm, dims=("latitude", "longitude")),
        },
        coords={"latitude": _LATS, "longitude": _LONS},
    )


def test_topo_aggregates_one_event() -> None:
    cube = np.zeros((4, _LATS.size, _LONS.size), dtype="float32")
    cube[1:3, 5:15, 5:15] = 400.0  # one persistent event

    elev = np.full((_LATS.size, _LONS.size), 100.0, dtype="float32")
    elev[5:15, 5:15] = np.linspace(0.0, 2000.0, 10 * 10).reshape(10, 10).astype("float32")
    lsm = np.zeros_like(elev)
    lsm[5:15, 5:15] = 1.0  # entire footprint is land

    events = pd.DataFrame({"event_id": [1]})
    out = event_topo_features(events, _topo(elev, lsm), _intensity(cube), threshold=100.0)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == 1
    # Elevation spans 0 .. 2000 linearly over the 100 footprint cells.
    assert row["mean_elev_m"] == pytest.approx(1000.0, rel=0.1)
    assert row["max_elev_m"] == pytest.approx(2000.0, rel=0.01)
    assert row["std_elev_m"] > 100.0
    assert row["land_fraction"] == pytest.approx(1.0)


def test_partial_land_fraction() -> None:
    cube = np.zeros((4, _LATS.size, _LONS.size), dtype="float32")
    cube[1:3, 10:20, 10:20] = 300.0

    elev = np.full((_LATS.size, _LONS.size), 200.0, dtype="float32")
    lsm = np.zeros_like(elev)
    # Half of the event's footprint (left half) is land, half is ocean.
    lsm[10:20, 10:15] = 1.0

    events = pd.DataFrame({"event_id": [1]})
    out = event_topo_features(events, _topo(elev, lsm), _intensity(cube), threshold=100.0)

    assert out.iloc[0]["land_fraction"] == pytest.approx(0.5, abs=0.05)


def test_missing_event_id_yields_nan_row() -> None:
    cube = np.zeros((4, _LATS.size, _LONS.size), dtype="float32")
    elev = np.full((_LATS.size, _LONS.size), 100.0, dtype="float32")
    lsm = np.zeros_like(elev)

    # event_id 7 does not exist in the labelled cube.
    events = pd.DataFrame({"event_id": [7]})
    out = event_topo_features(events, _topo(elev, lsm), _intensity(cube), threshold=100.0)

    assert len(out) == 1
    assert np.isnan(out.iloc[0]["mean_elev_m"])
    assert np.isnan(out.iloc[0]["land_fraction"])
