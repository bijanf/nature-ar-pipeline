"""Network-free unit tests for the Stage 2 fold assignment and pinball metric.

The full quantile-regression training is gated behind the ``slow`` marker and
not run in CI. What we test here is the deterministic fold-assignment logic
and the metric calculation, both of which are pure numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.models.stage2_cascade import _fold_metrics, assign_event_folds


def _events_frame(n: int = 200, ocean_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(30.0, 50.0, size=n)
    lon = rng.uniform(225.0, 245.0, size=n)
    # Sprinkle some entirely-oceanic events with NaN landfall coords.
    ocean = rng.random(n) < ocean_frac
    lat[ocean] = np.nan
    lon[ocean] = np.nan
    return pd.DataFrame(
        {
            "event_id": np.arange(n, dtype="int64"),
            "landfall_lat": lat.astype("float32"),
            "landfall_lon": lon.astype("float32"),
            "precip_total_mm": rng.gamma(shape=2.0, scale=40.0, size=n).astype("float32"),
        }
    )


def test_ocean_events_get_minus_one_fold() -> None:
    df = _events_frame(n=500, ocean_frac=0.2)
    plan = assign_event_folds(df, n_folds=5)
    is_ocean = df["landfall_lat"].isna().to_numpy()
    # Every ocean event sits in -1; no land event does.
    assert (plan.fold[is_ocean] == -1).all()
    assert (plan.fold[~is_ocean] != -1).all()


def test_assignment_is_deterministic() -> None:
    df = _events_frame(seed=1)
    a = assign_event_folds(df, seed=123)
    b = assign_event_folds(df, seed=123)
    np.testing.assert_array_equal(a.fold, b.fold)


def test_folds_cover_full_range() -> None:
    df = _events_frame(n=1000, ocean_frac=0.0)
    plan = assign_event_folds(df, n_folds=5)
    assert set(np.unique(plan.fold).tolist()) <= set(range(5))
    # All five folds should appear with 1000 land events spread across the bbox.
    assert len(set(plan.fold.tolist())) == 5


def test_pinball_loss_perfect_predictions_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    preds = {0.05: y.copy(), 0.50: y.copy(), 0.95: y.copy()}
    m = _fold_metrics(y, preds)
    assert m["pinball_05"] == pytest.approx(0.0)
    assert m["pinball_50"] == pytest.approx(0.0)
    assert m["pinball_95"] == pytest.approx(0.0)
    assert m["pi90_coverage"] == pytest.approx(1.0)


def test_pi90_coverage_drops_when_band_is_too_narrow() -> None:
    y = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    lo = np.array([4.0] * 5)
    mid = np.array([10.0] * 5)
    hi = np.array([16.0] * 5)
    m = _fold_metrics(y, {0.05: lo, 0.50: mid, 0.95: hi})
    # Only y = 5, 10, 15 fall inside [4, 16]; coverage = 3/5 = 0.6
    assert m["pi90_coverage"] == pytest.approx(0.6)
