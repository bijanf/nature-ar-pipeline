"""Network-free unit tests for the per-event Stage-1 feature reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.features import event_features


def _toy_inputs() -> tuple[pd.DataFrame, np.ndarray, xr.Dataset]:
    times = pd.date_range("2000-01-15", periods=4, freq="6h")
    lats = np.array([30.0, 35.0, 40.0])
    lons = np.array([230.0, 235.0])

    # Two events:
    #   event 1: t=0..1, all lat/lon cells (so a 2 x 3 x 2 brick of 12 pixels)
    #   event 2: t=2..3, just the (lat=40, lon=235) pixel for each timestep
    labels = np.zeros((4, 3, 2), dtype=int)
    labels[0:2, :, :] = 1
    labels[2:4, 2, 1] = 2

    # Constant-per-event physics fields, easy to reason about.
    def constants_for(value: float) -> np.ndarray:
        return np.full((4, 3, 2), value, dtype="float32")

    physics = {
        "ivt": constants_for(0.0),
        "ivt_u": constants_for(0.0),
        "ivt_v": constants_for(0.0),
        "theta_e_850": constants_for(0.0),
        "pv_250": constants_for(0.0),
        "eady_growth_rate": constants_for(0.0),
    }
    # Event 1 cells -> 100; event 2 cells -> 200; background stays 0.
    for name in physics:
        physics[name][labels == 1] = 100.0
        physics[name][labels == 2] = 200.0

    features = xr.Dataset(
        {name: (("time", "latitude", "longitude"), arr) for name, arr in physics.items()},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    events = pd.DataFrame(
        {
            "event_id": [1, 2],
            "start_time": [times[0], times[2]],
            "end_time": [times[1], times[3]],
        }
    )
    return events, labels, features


def test_attach_features_recovers_constants() -> None:
    events, labels, features = _toy_inputs()
    out = event_features.attach_features(events, labels, features)

    for col in ("ivt", "ivt_u", "theta_e_850", "pv_250", "eady_growth_rate"):
        assert out.loc[0, col] == 100.0
        assert out.loc[1, col] == 200.0


def test_attach_features_assigns_geometric_columns() -> None:
    events, labels, features = _toy_inputs()
    out = event_features.attach_features(events, labels, features)

    # event 1 touches all three latitudes -> mean = 35
    assert out.loc[0, "lat_deg"] == 35.0
    # event 2 only at lat=40
    assert out.loc[1, "lat_deg"] == 40.0
    # doy_sin/doy_cos must lie on the unit circle for both events.
    np.testing.assert_allclose(out["doy_sin"] ** 2 + out["doy_cos"] ** 2, 1.0, atol=1e-5)


def test_attach_features_handles_empty_input() -> None:
    _, labels, features = _toy_inputs()
    out = event_features.attach_features(pd.DataFrame({"event_id": []}), labels, features)
    assert out.empty
    for col in event_features.stage1_feature_columns():
        assert col in out.columns


def test_stage1_feature_columns_match_explainability() -> None:
    from src.models.explainability import _FEATURES

    assert set(event_features.stage1_feature_columns()) == set(_FEATURES)
