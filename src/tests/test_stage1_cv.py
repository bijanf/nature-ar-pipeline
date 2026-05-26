"""Network-free unit tests for the Stage 1 spatial-block CV machinery.

We exercise ``assign_folds`` and ``_buffer_mask`` on synthetic dataframes:
they are pure numpy/pandas and don't need lightgbm at import time, so the
file imports lightgbm lazily inside the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.models.stage1_ar_emulator import (
    _buffer_mask,
    assign_folds,
)


def _toy_frame(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = rng.uniform(25.0, 60.0, size=n)
    lons = rng.uniform(210.0, 250.0, size=n)
    years = rng.integers(1980, 2015, size=n)
    times = pd.to_datetime(years.astype(str) + "-06-15")
    ar_mask = rng.random(size=n) < 0.05
    return pd.DataFrame(
        {
            "time": times,
            "lat_deg": lats.astype("float32"),
            "longitude": lons.astype("float32"),
            "ar_mask": ar_mask,
            "ar_intensity": np.where(ar_mask, rng.normal(500.0, 100.0, size=n), 0.0).astype(
                "float32"
            ),
        }
    )


def test_fold_assignment_covers_all_rows() -> None:
    df = _toy_frame()
    plan = assign_folds(df, n_folds=5)

    assert plan.spatial_fold.shape == (len(df),)
    assert plan.year_fold.shape == (len(df),)
    assert set(np.unique(plan.spatial_fold).tolist()) <= set(range(5))
    assert set(np.unique(plan.year_fold).tolist()) <= set(range(5))


def test_fold_assignment_is_deterministic() -> None:
    df = _toy_frame(seed=1)
    a = assign_folds(df, n_folds=5, seed=123)
    b = assign_folds(df, n_folds=5, seed=123)
    np.testing.assert_array_equal(a.spatial_fold, b.spatial_fold)
    np.testing.assert_array_equal(a.year_fold, b.year_fold)


def test_stratified_folds_balance_ar_climatology() -> None:
    # With stratification, each fold's mean AR rate should be within ~30% of
    # the global rate. (Loose bound — small synthetic sample.)
    df = _toy_frame(n=20_000, seed=2)
    plan = assign_folds(df, n_folds=5, stratify=True)
    global_rate = df["ar_mask"].mean()
    for k in range(5):
        sel = plan.spatial_fold == k
        rate = df.loc[sel, "ar_mask"].mean()
        assert abs(rate - global_rate) < 0.30 * global_rate + 0.005


def test_buffer_mask_picks_rows_near_test_blocks() -> None:
    lat = np.array([30.0, 30.0, 30.0])
    lon = np.array([212.0, 218.0, 240.0])  # 212 is in/near, 218 is far, 240 is far
    test_block_lat = np.array([30.0])
    test_block_lon = np.array([212.5])  # block centred there with half-width 2.5

    out = _buffer_mask(lat, lon, test_block_lat, test_block_lon, block_deg=5.0, buffer_deg=1.0)

    # 212 is inside the block (dist 0 < 1.0): buffered.
    # 218: |lon - 212.5| = 5.5, beyond half-width 2.5 -> dist 3.0 > 1.0: NOT buffered.
    # 240: very far -> NOT buffered.
    assert out.tolist() == [True, False, False]


def test_buffer_mask_zero_buffer_only_in_block() -> None:
    lat = np.array([30.0, 30.0])
    lon = np.array([212.0, 220.0])
    out = _buffer_mask(lat, lon, np.array([30.0]), np.array([212.5]), block_deg=5.0, buffer_deg=0.0)
    # First row is inside the block (dist == 0); buffer_deg=0 rejects equality.
    # Use a tiny positive buffer to assert containment more cleanly.
    out_pos = _buffer_mask(
        lat, lon, np.array([30.0]), np.array([212.5]), block_deg=5.0, buffer_deg=0.01
    )
    assert out_pos.tolist() == [True, False]
    assert out.tolist() == [False, False]
