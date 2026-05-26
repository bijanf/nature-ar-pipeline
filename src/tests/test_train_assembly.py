"""Network-free unit tests for the training-set assembly helpers.

The streaming + GW path lives behind the ``network`` marker (not run here).
What we test is the deterministic helper logic that future bugs would corrupt:
period parsing, the 6-hourly time selector, and the stratified subsampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("metpy")  # era5_train_assembly imports physics_pipeline -> metpy

from src.data.era5_train_assembly import (  # noqa: E402
    _parse_period,
    _select_6h_subsample,
    _stratified_subsample,
)


def test_parse_period_range() -> None:
    assert _parse_period("1980-2014") == ("1980-01-01", "2014-12-31")


def test_parse_period_single_year() -> None:
    assert _parse_period("1998") == ("1998-01-01", "1998-12-31")


def test_select_6h_subsample_keeps_only_synoptic_hours() -> None:
    # 24 consecutive hours -> we should retain hours 0, 6, 12, 18 only.
    times = pd.date_range("2010-01-01", periods=24, freq="1h")
    ds = xr.Dataset(
        {"ivt": (("time", "lat", "lon"), np.zeros((24, 3, 3), dtype="float32"))},
        coords={"time": times, "lat": [25.0, 25.25, 25.5], "lon": [210.0, 210.25, 210.5]},
    )
    rng = np.random.default_rng(0)

    out = _select_6h_subsample(ds, timestep_subsample=1, rng=rng)
    hours = out["time"].dt.hour.values

    assert set(hours.tolist()) == {0, 6, 12, 18}
    assert out.sizes["time"] == 4


def test_select_6h_subsample_further_subsamples_when_requested() -> None:
    times = pd.date_range("2010-01-01", periods=24 * 30, freq="1h")  # one month
    ds = xr.Dataset(
        {"ivt": (("time",), np.zeros(24 * 30, dtype="float32"))},
        coords={"time": times},
    )
    rng = np.random.default_rng(0)

    out = _select_6h_subsample(ds, timestep_subsample=4, rng=rng)

    # After 6-hourly select: 4*30 = 120 timesteps. Subsample by 4 -> 30.
    assert out.sizes["time"] == 30


def test_stratified_subsample_keeps_all_positives_and_caps_negatives() -> None:
    rng = np.random.default_rng(0)
    n_pos, n_neg = 100, 10_000
    df = pd.DataFrame(
        {
            "ivt": rng.normal(size=n_pos + n_neg),
            "ar_intensity": np.concatenate([rng.normal(500, 50, n_pos), np.zeros(n_neg)]),
            "ar_mask": np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)]),
        }
    )

    out = _stratified_subsample(df, n_neg_per_pos=3, rng=rng)

    assert (out["ar_mask"]).sum() == n_pos
    assert (~out["ar_mask"]).sum() == 3 * n_pos
    assert len(out) == 4 * n_pos


def test_stratified_subsample_no_positives_returns_empty_positive_set() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "ar_intensity": np.zeros(50),
            "ar_mask": np.zeros(50, dtype=bool),
        }
    )

    out = _stratified_subsample(df, n_neg_per_pos=3, rng=rng)

    assert len(out) == 0
