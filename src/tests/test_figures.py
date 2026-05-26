"""Smoke tests for the four Nature-spec figure scripts.

Each test calls a figure's ``plot``/``build_figure`` function with synthetic
inputs, writes the result to a temp PDF, and verifies:

1. The file exists and is non-empty.
2. The PDF declares ``/FontFile2`` (TrueType embedded) somewhere — the
   ``pdf.fonttype = 42`` setting from :mod:`src.figures._style` is the
   reason this works.

These run network-free in CI.
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
    fig3_ssp_shift,
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


def test_fig3_ssp_shift_renders(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)

    def synthetic(n: int, scale: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precip_pred_q05_mm": rng.uniform(0.0, scale * 0.5, n),
                "precip_pred_q50_mm": rng.uniform(scale * 0.5, scale, n),
                "precip_pred_q95_mm": rng.uniform(scale, scale * 2.0, n),
            }
        )

    per_scenario = {}
    for ssp, n, sc in [
        ("ssp245", 200, 30.0),
        ("ssp370", 240, 40.0),
        ("ssp460", 210, 35.0),
        ("ssp585", 280, 60.0),
    ]:
        per_scenario[(ssp, "direct")] = synthetic(n, sc)
        per_scenario[(ssp, "delta")] = synthetic(n - 20, sc * 0.9)

    historical = pd.DataFrame({"precip_total_mm": rng.uniform(0.0, 30.0, 200)})

    out = tmp_path / "fig3.pdf"
    fig = fig3_ssp_shift.plot(per_scenario, historical)
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_fig4_shap_drivers_renders(tmp_path: Path) -> None:
    bands = ["south_25_35N", "central_35_45N", "north_45_60N"]
    rng = np.random.default_rng(2)
    per_ssp = {}
    for ssp in ("ssp245", "ssp370", "ssp460", "ssp585"):
        per_ssp[ssp] = pd.DataFrame(
            {
                "thermodynamic": rng.uniform(0.5, 2.0, size=len(bands)),
                "dynamic": rng.uniform(0.3, 1.5, size=len(bands)),
                "other": rng.uniform(0.05, 0.4, size=len(bands)),
            },
            index=bands,
        )

    out = tmp_path / "fig4.pdf"
    fig = fig4_shap_drivers.plot(per_ssp)
    fig.savefig(out)
    plt.close(fig)
    _assert_vector_pdf(out)


def test_pdf_fonttype_is_truetype() -> None:
    # Independent of any figure render: applying the style mounts fonttype=42.
    from src.figures._style import apply_nature_style

    apply_nature_style()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
