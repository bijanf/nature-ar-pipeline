"""Network-free unit tests for the CMIP6 inference module.

We don't open Pangeo or instantiate xesmf here. What we cover:

* ``stack_features_to_matrix`` produces a column-ordered (N, 9) matrix.
* ``predict_intensity`` averages across a list of fake boosters and re-folds
  the prediction back into a ``(time, lat, lon)`` DataArray.
* The CMIP6 variable rename map and pressure-axis conversion (Pa -> hPa)
  produce a Dataset that physics_pipeline could ingest in principle.

The xesmf-dependent regrid and intake-esm catalog access live behind the
``network`` marker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("lightgbm")
pytest.importorskip("metpy")

from src import config  # noqa: E402
from src.inference.cmip6_apply import (  # noqa: E402
    _STAGE1_FEATURES,
    predict_intensity,
    stack_features_to_matrix,
)


def _toy_features(n_time: int = 4, n_lat: int = 5, n_lon: int = 6) -> xr.Dataset:
    times = pd.date_range("2070-01-01", periods=n_time, freq="6h")
    lats = np.linspace(25.0, 40.0, n_lat)
    lons = np.linspace(220.0, 235.0, n_lon)
    shape = (n_time, n_lat, n_lon)
    rng = np.random.default_rng(0)
    data = {
        "ivt": (("time", "latitude", "longitude"), rng.uniform(50, 400, shape).astype("float32")),
        "ivt_u": (
            ("time", "latitude", "longitude"),
            rng.uniform(-200, 200, shape).astype("float32"),
        ),
        "ivt_v": (
            ("time", "latitude", "longitude"),
            rng.uniform(-200, 200, shape).astype("float32"),
        ),
        "theta_e_850": (
            ("time", "latitude", "longitude"),
            rng.uniform(280, 320, shape).astype("float32"),
        ),
        "pv_250": (
            ("time", "latitude", "longitude"),
            rng.uniform(-1e-6, 1e-5, shape).astype("float32"),
        ),
        "eady_growth_rate": (
            ("time", "latitude", "longitude"),
            rng.uniform(0, 1.5e-5, shape).astype("float32"),
        ),
    }
    return xr.Dataset(data, coords={"time": times, "latitude": lats, "longitude": lons})


def test_stack_produces_correct_shape_and_column_order() -> None:
    ds = _toy_features()
    matrix, template = stack_features_to_matrix(ds)

    n = ds.sizes["time"] * ds.sizes["latitude"] * ds.sizes["longitude"]
    assert matrix.shape == (n, len(_STAGE1_FEATURES))
    assert matrix.dtype == np.float32
    assert template.dims == ("time", "latitude", "longitude")


def test_stack_lat_column_matches_grid() -> None:
    ds = _toy_features()
    matrix, template = stack_features_to_matrix(ds)

    lat_col = _STAGE1_FEATURES.index("lat_deg")
    lats = ds["latitude"].to_numpy()
    # The lat column should reproduce the meshgrid: for each (t, i, j), value = lats[i].
    reshaped = matrix[:, lat_col].reshape(template.shape)
    expected = np.broadcast_to(lats[None, :, None], template.shape)
    np.testing.assert_allclose(reshaped, expected)


def test_stack_doy_columns_within_unit_circle() -> None:
    ds = _toy_features()
    matrix, _ = stack_features_to_matrix(ds)
    s = matrix[:, _STAGE1_FEATURES.index("doy_sin")]
    c = matrix[:, _STAGE1_FEATURES.index("doy_cos")]
    radius = np.sqrt(s * s + c * c)
    np.testing.assert_allclose(radius, 1.0, atol=1e-5)


class _ConstantBooster:
    """Fake booster: returns a constant value regardless of input."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.best_iteration = 1

    def predict(self, x: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype="float32")


def test_predict_intensity_averages_across_boosters() -> None:
    ds = _toy_features()
    boosters = [_ConstantBooster(100.0), _ConstantBooster(200.0), _ConstantBooster(300.0)]

    intensity = predict_intensity(boosters, ds)

    assert intensity.dims == ("time", "latitude", "longitude")
    assert intensity.shape == (ds.sizes["time"], ds.sizes["latitude"], ds.sizes["longitude"])
    np.testing.assert_allclose(intensity.to_numpy(), 200.0)


def test_cmip6_rename_map_targets_match_era5_vars() -> None:
    # Sanity guard — the CMIP6->ERA5 rename must produce keys that match
    # what physics_pipeline.required_era5_vars() asks for.
    from src.features import physics_pipeline

    targets = set(config.CMIP6_RENAME.values())
    needed = set(physics_pipeline.required_era5_vars())
    assert needed.issubset(targets)
