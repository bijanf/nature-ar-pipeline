"""Network-free unit tests for the SHAP bucket/band aggregation.

We don't fire up a real LightGBM model here — instead we feed synthetic SHAP
matrices through ``bucket_attribution`` and ``aggregate_by_landfall_band`` and
assert the buckets sum and bin correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("lightgbm")

from src.models.explainability import (
    _DYNAMIC,
    _FEATURES,
    _OTHER,
    _THERMODYNAMIC,
    aggregate_by_landfall_band,
    bucket_attribution,
)


def _shap_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n).astype("float32") for col in _FEATURES}
    return pd.DataFrame(data)


def test_bucket_attribution_sums_correctly() -> None:
    shap_df = pd.DataFrame(
        {
            "theta_e_850": [1.0, -2.0],
            "ivt": [0.5, 1.0],
            "pv_250": [-1.0, 2.0],
            "eady_growth_rate": [0.5, -0.5],
            "ivt_u": [-1.0, 1.0],
            "ivt_v": [1.0, -1.0],
            "lat_deg": [0.0, 0.0],
            "doy_sin": [0.0, 0.0],
            "doy_cos": [0.0, 0.0],
        }
    )
    out = bucket_attribution(shap_df)

    assert out.loc[0, "thermodynamic"] == pytest.approx(abs(1.0) + abs(0.5))
    assert out.loc[1, "thermodynamic"] == pytest.approx(abs(-2.0) + abs(1.0))
    assert out.loc[0, "dynamic"] == pytest.approx(abs(-1.0) + abs(0.5) + abs(-1.0) + abs(1.0))
    assert out.loc[0, "other"] == pytest.approx(0.0)


def test_band_aggregation_groups_landfall_correctly() -> None:
    buckets = pd.DataFrame(
        {
            "thermodynamic": [10.0, 20.0, 30.0],
            "dynamic": [5.0, 15.0, 25.0],
            "other": [0.0, 0.0, 0.0],
        }
    )
    landfall = pd.Series([28.0, 40.0, 55.0])  # south, central, north

    table = aggregate_by_landfall_band(buckets, landfall)

    assert list(table.index) == ["south_25_35N", "central_35_45N", "north_45_60N"]
    assert table.loc["south_25_35N", "thermodynamic"] == pytest.approx(10.0)
    assert table.loc["central_35_45N", "thermodynamic"] == pytest.approx(20.0)
    assert table.loc["north_45_60N", "dynamic"] == pytest.approx(25.0)


def test_band_aggregation_handles_repeated_lats() -> None:
    buckets = pd.DataFrame(
        {
            "thermodynamic": [10.0, 30.0, 50.0],
            "dynamic": [1.0, 3.0, 5.0],
            "other": [0.0, 0.0, 0.0],
        }
    )
    # Two events in the south band, one in central.
    landfall = pd.Series([26.0, 32.0, 40.0])

    table = aggregate_by_landfall_band(buckets, landfall)

    assert table.loc["south_25_35N", "thermodynamic"] == pytest.approx(20.0)  # mean of 10 & 30
    assert table.loc["central_35_45N", "thermodynamic"] == pytest.approx(50.0)


def test_features_constant_covers_three_buckets() -> None:
    # Sanity guard — the feature-set constants must exhaustively cover the buckets.
    assert set(_THERMODYNAMIC).isdisjoint(set(_DYNAMIC))
    assert set(_THERMODYNAMIC).isdisjoint(set(_OTHER))
    assert set(_DYNAMIC).isdisjoint(set(_OTHER))
    assert set(_FEATURES) == set(_THERMODYNAMIC) | set(_DYNAMIC) | set(_OTHER)
