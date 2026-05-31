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
from src.analysis import period_contrast as pc
from src.analysis.period_contrast import (
    _bootstrap_mean,
    concat_periods,
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


def test_concat_periods_roundtrips_two_suffixes(tmp_path, monkeypatch) -> None:
    # Stand-in events tables for two per-period SLURM jobs.
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    common_cols = {
        "mean_intensity": [310.0, 420.0],
        "max_intensity": [360.0, 500.0],
        "footprint_area_km2": [120000.0, 180000.0],
        "duration_hours": [12.0, 18.0],
    }
    pd.DataFrame({"period": ["modern_1980_1999"] * 2, **common_cols}).to_parquet(
        tmp_path / "observational_events_modern.parquet", index=False
    )
    pd.DataFrame({"period": ["recent_2015_2024"] * 2, **common_cols}).to_parquet(
        tmp_path / "observational_events_recent.parquet", index=False
    )

    out = concat_periods(["modern", "recent"])

    assert out == tmp_path / "period_contrast.parquet"
    merged_events = pd.read_parquet(tmp_path / "observational_events.parquet")
    assert sorted(merged_events["period"].unique()) == ["modern_1980_1999", "recent_2015_2024"]
    assert len(merged_events) == 4

    summary = pd.read_parquet(out)
    assert sorted(summary["period_name"]) == ["modern_1980_1999", "recent_2015_2024"]
    assert (summary["n_events"] == 2).all()
    # Bootstrap bands must bracket the means.
    assert (summary["intensity_q05"] < summary["mean_intensity"]).all()
    assert (summary["mean_intensity"] < summary["intensity_q95"]).all()


def test_run_propagates_time_range_to_events(tmp_path, monkeypatch) -> None:
    # Capture the sub_window passed to _events_for_period, verify it equals
    # the time_range the caller passed.
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    captured: dict[str, object] = {}

    def fake_events(period_name, period, era5_topo, sub_window=None, fixed_threshold=None):
        captured["period"] = period
        captured["sub_window"] = sub_window
        captured["fixed_threshold"] = fixed_threshold
        return pd.DataFrame(
            {
                "period": [period_name],
                "mean_intensity": [300.0],
                "max_intensity": [350.0],
                "footprint_area_km2": [100000.0],
                "duration_hours": [12.0],
            }
        )

    class FakeTopo:
        def compute(self):
            return object()

    monkeypatch.setattr(pc, "_events_for_period", fake_events)
    monkeypatch.setattr("src.features.topography.open_era5_topography", lambda *a, **k: FakeTopo())

    pc.run(
        periods=[("modern_1980_1999", config.MODERN_PERIOD)],
        out_suffix="modern_1990s",
        time_range=("1990-01-01", "1999-12-31"),
    )

    assert captured["period"] == config.MODERN_PERIOD
    assert captured["sub_window"] == ("1990-01-01", "1999-12-31")
    assert captured["fixed_threshold"] is None  # default = period-internal climatology
    assert (tmp_path / "observational_events_modern_1990s.parquet").exists()
    assert (tmp_path / "period_contrast_modern_1990s.parquet").exists()


def test_open_period_features_detects_6hourly_cadence(monkeypatch) -> None:
    """When the upstream source is already 6-hourly (the local CDS cache),
    _open_period_features must NOT re-subsample (would cut data 6x)."""
    import numpy as np
    import xarray as xr

    lat = np.linspace(60.0, 25.0, 4)
    lon = np.linspace(210.0, 250.0, 5)
    levels = [850, 500, 250]
    times = pd.date_range("2018-01-01", periods=28, freq="6h")
    shape = (times.size, len(levels), lat.size, lon.size)

    def _f32(seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype("float32")

    ds = xr.Dataset(
        {
            "u_component_of_wind": (("time", "level", "latitude", "longitude"), _f32(1)),
            "v_component_of_wind": (("time", "level", "latitude", "longitude"), _f32(2)),
            "temperature": (("time", "level", "latitude", "longitude"), 270 + _f32(3)),
            "specific_humidity": (
                ("time", "level", "latitude", "longitude"),
                1e-3 * np.abs(_f32(4)),
            ),
            "geopotential": (("time", "level", "latitude", "longitude"), 1e5 * np.abs(_f32(5))),
        },
        coords={"time": times, "level": levels, "latitude": lat, "longitude": lon},
    )

    monkeypatch.setattr(pc.physics_pipeline, "open_arco_era5", lambda *a, **k: ds)
    monkeypatch.setattr(
        pc.physics_pipeline,
        "calculate_dynamics",
        lambda d: d.assign(ivt=d["u_component_of_wind"].isel(level=0)),
    )

    feats, _ = pc._open_period_features(("2018-01-01", "2018-01-08"))
    # 6-hourly source × 7 days = 28 timesteps. No re-subsampling.
    assert feats.sizes["time"] == 28


def test_open_period_features_subsamples_hourly_cadence(monkeypatch) -> None:
    """When the upstream is hourly (legacy ARCO Zarr path), we sub-sample
    by 6× to match the GW-detector cadence."""
    import numpy as np
    import xarray as xr

    lat = np.linspace(60.0, 25.0, 4)
    lon = np.linspace(210.0, 250.0, 5)
    levels = [850, 500, 250]
    times = pd.date_range("2018-01-01", periods=24, freq="1h")
    shape = (times.size, len(levels), lat.size, lon.size)

    def _f32(seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype("float32")

    ds = xr.Dataset(
        {
            "u_component_of_wind": (("time", "level", "latitude", "longitude"), _f32(1)),
            "v_component_of_wind": (("time", "level", "latitude", "longitude"), _f32(2)),
            "temperature": (("time", "level", "latitude", "longitude"), 270 + _f32(3)),
            "specific_humidity": (
                ("time", "level", "latitude", "longitude"),
                1e-3 * np.abs(_f32(4)),
            ),
            "geopotential": (("time", "level", "latitude", "longitude"), 1e5 * np.abs(_f32(5))),
        },
        coords={"time": times, "level": levels, "latitude": lat, "longitude": lon},
    )

    monkeypatch.setattr(pc.physics_pipeline, "open_arco_era5", lambda *a, **k: ds)
    monkeypatch.setattr(
        pc.physics_pipeline,
        "calculate_dynamics",
        lambda d: d.assign(ivt=d["u_component_of_wind"].isel(level=0)),
    )

    feats, _ = pc._open_period_features(("2018-01-01", "2018-01-02"))
    # 24 hourly → 4 6-hourly samples.
    assert feats.sizes["time"] == 4


def test_observational_periods_cover_full_record() -> None:
    # Sanity guard — the three periods cover 1940..2024 contiguously.
    names = [n for n, _ in config.OBSERVATIONAL_PERIODS]
    assert names == ["pre_sat_1940_1959", "modern_1980_1999", "recent_2015_2024"]
    # MODERN and TRAIN must alias each other (Story A's ML scope).
    assert config.MODERN_PERIOD == config.TRAIN_PERIOD
    assert config.RECENT_PERIOD == config.HOLDOUT_PERIOD
