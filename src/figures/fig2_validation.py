"""
Figure 2 — Stage 1 + Stage 2 validation.

Three panels (double-column, ≈ 60 mm tall):

a) Stage 1 5-fold spatial-block CV: per-fold RMSE bars + mean ± std.
b) Stage 1 holdout (2015–2024): scatter of predicted vs ERA5 AR-conditional
   IVT, with the 1:1 line. Points colour-coded by AR-mask hit / miss.
c) Stage 2 quantile-regression PI coverage and pinball loss per fold.

Reads:
* ``data/cache/stage1_cv_report.json``
* ``data/cache/stage1_holdout_predictions.parquet`` (predicted, actual, ar_mask)
* ``data/cache/stage2_cv_report.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style


def plot(
    stage1_report: dict,
    stage2_report: dict,
    holdout: pd.DataFrame,
) -> plt.Figure:
    """Build the three-panel validation figure from the in-memory inputs."""
    apply_nature_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.30), constrained_layout=True
    )

    # ----- panel (a): Stage 1 CV RMSE -----
    ax = axes[0]
    metrics = stage1_report["metrics"]
    rmse = np.array([m["rmse"] for m in metrics])
    ax.bar(np.arange(len(rmse)), rmse, color="#4a6fa5", edgecolor="black", linewidth=0.4)
    ax.axhline(rmse.mean(), color="black", linestyle="--", lw=0.5)
    ax.set_xticks(np.arange(len(rmse)))
    ax.set_xlabel("fold")
    ax.set_ylabel("RMSE (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title("(a) Stage 1 CV")

    # ----- panel (b): holdout scatter -----
    ax = axes[1]
    pred = holdout["predicted"].to_numpy()
    actual = holdout["actual"].to_numpy()
    hit = holdout["ar_mask"].to_numpy().astype(bool)
    ax.scatter(actual[~hit], pred[~hit], s=1.5, color="#888", alpha=0.4, label="non-AR")
    ax.scatter(actual[hit], pred[hit], s=2.0, color="#c0392b", alpha=0.6, label="AR")
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.5)
    ax.set_xlabel("ERA5 AR-conditional IVT")
    ax.set_ylabel("predicted IVT")
    ax.set_title("(b) 2015–2024 holdout")
    ax.legend(loc="upper left")

    # ----- panel (c): Stage 2 PI coverage + pinball -----
    ax = axes[2]
    s2_metrics = stage2_report["metrics"]
    folds = np.arange(len(s2_metrics))
    coverage = np.array([m["pi90_coverage"] for m in s2_metrics])
    pinball50 = np.array([m["pinball_50"] for m in s2_metrics])

    ax.bar(
        folds - 0.18,
        coverage,
        width=0.36,
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.4,
        label="PI₉₀ coverage",
    )
    ax2 = ax.twinx()
    ax2.bar(
        folds + 0.18,
        pinball50,
        width=0.36,
        color="#e67e22",
        edgecolor="black",
        linewidth=0.4,
        label="pinball 0.50",
    )
    ax.axhline(0.90, color="black", linestyle="--", lw=0.5)
    ax.set_xticks(folds)
    ax.set_xlabel("fold")
    ax.set_ylabel("PI$_{90}$ coverage")
    ax2.set_ylabel("pinball (mm)")
    ax.set_title("(c) Stage 2 calibration")
    ax2.spines["top"].set_visible(False)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-report", type=str, default=str(config.CACHE_DIR / "stage1_cv_report.json")
    )
    parser.add_argument(
        "--stage2-report", type=str, default=str(config.CACHE_DIR / "stage2_cv_report.json")
    )
    parser.add_argument(
        "--holdout", type=str, default=str(config.CACHE_DIR / "stage1_holdout_predictions.parquet")
    )
    parser.add_argument("--out", type=str, default="figures/fig2_validation.pdf")
    args = parser.parse_args()

    stage1_report = json.loads(Path(args.stage1_report).read_text())
    stage2_report = json.loads(Path(args.stage2_report).read_text())
    holdout = pd.read_parquet(args.holdout)

    fig = plot(stage1_report, stage2_report, holdout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
