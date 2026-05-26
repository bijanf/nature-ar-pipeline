"""Smoke tests for the four Nature-spec figure scripts.

Each test calls a figure's ``plot`` / ``build_figure`` function with synthetic
inputs, writes the result to a temp PDF, and verifies it is a valid vector
PDF (magic bytes + non-trivial size). The TrueType ``pdf.fonttype=42`` rule
is verified independently from rcParams.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
matplotlib = pytest.importorskip("matplotlib")

from src.figures import (  # noqa: E402
    fig1_pipeline_schematic,
    fig2_validation,
    fig3_trajectory,
    fig4_shap_drivers,
)


def _assert_vector_pdf(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 1024
    head = path.read_bytes()[:4]
    assert head == b"%PDF"


def test_fig1_schematic_writes_vector_pdf(tmp_path: Path) -> None:
    out = tmp_path / "fig1.pdf"
    fig = fig1_pipeline_schematic.build_figure()
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_fig2_validation_renders(tmp_path: Path) -> None:
    stage1_report = {"metrics": [{"rmse": 90.0 + i, "r2": 0.6 - 0.01 * i} for i in range(5)]}
    stage2_report = {
        "metrics": [
            {"pi90_coverage": 0.85 + 0.02 * i, "pinball_50": 10.0 - 0.5 * i} for i in range(5)
        ]
    }
    rng = np.random.default_rng(0)
    n = 300
    actual = rng.uniform(0.0, 800.0, size=n).astype("float32")
    predicted = actual + rng.normal(0.0, 40.0, size=n).astype("float32")
    holdout = pd.DataFrame({"actual": actual, "predicted": predicted, "ar_mask": actual > 250.0})

    out = tmp_path / "fig2.pdf"
    fig = fig2_validation.plot(stage1_report, stage2_report, holdout)
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_fig3_trajectory_renders(tmp_path: Path) -> None:
    # Observed: three periods with bootstrap bands.
    observed = pd.DataFrame(
        [
            {
                "period_name": "pre_sat_1940_1979",
                "n_events": 220,
                "mean_intensity": 320.0,
                "mean_max_intensity": 410.0,
                "mean_footprint_km2": 180000.0,
                "mean_duration_h": 30.0,
                "intensity_q05": 300.0,
                "intensity_q95": 340.0,
                "duration_q05": 27.0,
                "duration_q95": 33.0,
            },
            {
                "period_name": "modern_1980_2014",
                "n_events": 245,
                "mean_intensity": 360.0,
                "mean_max_intensity": 470.0,
                "mean_footprint_km2": 200000.0,
                "mean_duration_h": 32.0,
                "intensity_q05": 345.0,
                "intensity_q95": 375.0,
                "duration_q05": 30.0,
                "duration_q95": 34.0,
            },
            {
                "period_name": "recent_2015_2024",
                "n_events": 275,
                "mean_intensity": 395.0,
                "mean_max_intensity": 510.0,
                "mean_footprint_km2": 215000.0,
                "mean_duration_h": 34.0,
                "intensity_q05": 380.0,
                "intensity_q95": 410.0,
                "duration_q05": 31.0,
                "duration_q95": 37.0,
            },
        ]
    )
    rng = np.random.default_rng(1)

    def syn(n: int, scale: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precip_pred_q05_mm": rng.uniform(0, scale * 0.5, n),
                "precip_pred_q50_mm": rng.uniform(scale * 0.5, scale, n),
                "precip_pred_q95_mm": rng.uniform(scale, scale * 2.0, n),
            }
        )

    projected = {}
    for ssp, n, sc in [
        ("ssp245", 250, 35.0),
        ("ssp370", 290, 50.0),
        ("ssp460", 260, 42.0),
        ("ssp585", 330, 70.0),
    ]:
        projected[(ssp, "direct")] = syn(n, sc)
        projected[(ssp, "delta")] = syn(n - 20, sc * 0.85)

    out = tmp_path / "fig3.pdf"
    fig = fig3_trajectory.plot(observed, projected)
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_fig4_shap_drivers_renders(tmp_path: Path) -> None:
    bands = ["south_25_35N", "central_35_45N", "north_45_60N"]
    rng = np.random.default_rng(2)

    def synth_table(thermo_scale: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "thermodynamic": rng.uniform(0.5, 2.0, size=len(bands)) * thermo_scale,
                "dynamic": rng.uniform(0.3, 1.5, size=len(bands)),
                "other": rng.uniform(0.05, 0.4, size=len(bands)),
            },
            index=bands,
        )

    observed = {
        "pre_sat_1940_1979": synth_table(0.85),
        "modern_1980_2014": synth_table(1.0),
        "recent_2015_2024": synth_table(1.15),
    }
    projected = {
        "ssp245": synth_table(1.10),
        "ssp370": synth_table(1.25),
        "ssp460": synth_table(1.20),
        "ssp585": synth_table(1.45),
    }

    out = tmp_path / "fig4.pdf"
    fig = fig4_shap_drivers.plot(observed, projected)
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_fig5_landfall_density_renders(tmp_path: Path) -> None:
    pytest.importorskip("cartopy")
    from src.figures import fig5_landfall_density

    rng = np.random.default_rng(3)

    def synth_events(n: int, lat_centre: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                # ERA5 events store lon in 0..360
                "landfall_lon": rng.normal(loc=235.0, scale=2.0, size=n),
                "landfall_lat": rng.normal(loc=lat_centre, scale=2.5, size=n),
            }
        )

    observed = {
        "pre_sat_1940_1979": synth_events(220, 35.0),
        "modern_1980_2014": synth_events(245, 37.0),
        "recent_2015_2024": synth_events(275, 39.0),
    }
    projected = {
        "ssp245": synth_events(260, 39.5),
        "ssp370": synth_events(290, 40.5),
        "ssp460": synth_events(265, 40.0),
        "ssp585": synth_events(330, 42.0),
    }

    out = tmp_path / "fig5.pdf"
    fig = fig5_landfall_density.plot(observed, projected)
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_pdf_fonttype_is_truetype() -> None:
    from src.figures._style import apply_nature_style

    apply_nature_style()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
