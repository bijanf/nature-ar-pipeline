"""Network-free unit tests for the period-contrast helpers.

We don't stream ERA5 here — only the deterministic numerical helpers that
won't suddenly start working differently across pandas/numpy versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("metpy")  # analysis.period_contrast imports physics_pipeline

from src import config
from src.analysis.period_contrast import (
    _bootstrap_mean,
    summarise_period,
)


def test_bootstrap_mean_handles_empty() -> None:
    out = _bootstrap_mean(np.array([], dtype="float32"))
    assert all(np.isnan(v) for v in out.values())


def test_bootstrap_mean_recovers_true_mean() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=100.0, scale=10.0, size=500)
    bands = _bootstrap_mean(sample, n=500, seed=42)
    # 90% bootstrap band should contain the sample mean.
    assert bands["q05"] < float(sample.mean()) < bands["q95"]


def test_summarise_period_aggregates_correctly() -> None:
    events = pd.DataFrame(
        {
            "mean_intensity": [300.0, 400.0, 500.0],
            "max_intensity": [350.0, 480.0, 600.0],
            "footprint_area_km2": [100000.0, 200000.0, 150000.0],
            "duration_hours": [12.0, 18.0, 24.0],
        }
    )
    s = summarise_period(events, "demo_period")
    assert s.period_name == "demo_period"
    assert s.n_events == 3
    assert s.mean_intensity == pytest.approx(400.0)
    assert s.mean_duration_h == pytest.approx(18.0)
    # Bootstrap bands should bracket the mean.
    assert s.intensity_q05 < s.mean_intensity < s.intensity_q95


def test_observational_periods_cover_full_record() -> None:
    # Sanity guard — the three periods cover 1940..2024 contiguously.
    names = [n for n, _ in config.OBSERVATIONAL_PERIODS]
    assert names == ["pre_sat_1940_1979", "modern_1980_2014", "recent_2015_2024"]
    # MODERN and TRAIN must alias each other (Story A's ML scope).
    assert config.MODERN_PERIOD == config.TRAIN_PERIOD
    assert config.RECENT_PERIOD == config.HOLDOUT_PERIOD
