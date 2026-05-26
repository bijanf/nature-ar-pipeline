"""
Figure 4 — Thermodynamic vs dynamic driver attribution per landfall band.

Stacked bar chart (single-column): for each of the three landfall bands
(south 25–35 N, central 35–45 N, north 45–60 N) and each future SSP, the
mean |SHAP| contribution attributed to:

* **thermodynamic**: θ_e_850, IVT magnitude (q-driven, Clausius-Clapeyron axis)
* **dynamic**: PV_250, Eady σ_BI, IVT components (storm-track shifts)
* **other**: lat / day-of-year context

Reads ``data/cache/shap_attribution_<exp>.parquet`` for each
``exp`` in :data:`config.CMIP6_EXPERIMENTS`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.figures._style import COL_SINGLE_IN, apply_nature_style

_BUCKETS = ("thermodynamic", "dynamic", "other")
_BUCKET_COLOURS = {
    "thermodynamic": "#c0392b",  # warm
    "dynamic": "#2980b9",  # cool
    "other": "#bdc3c7",  # neutral
}
_BAND_ORDER = ("south_25_35N", "central_35_45N", "north_45_60N")
_BAND_LABEL = {
    "south_25_35N": "25–35 °N",
    "central_35_45N": "35–45 °N",
    "north_45_60N": "45–60 °N",
}


def plot(per_ssp: dict[str, pd.DataFrame]) -> plt.Figure:
    """Stacked bars: x = (SSP, band), height = bucket magnitude."""
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, COL_SINGLE_IN * 0.85), constrained_layout=True)

    ssps = [s for s in config.CMIP6_EXPERIMENTS if s in per_ssp]
    n_bands = len(_BAND_ORDER)
    n_ssps = len(ssps)

    x_centres = []
    x_labels = []
    width = 0.8 / n_bands

    for i, ssp in enumerate(ssps):
        table = per_ssp[ssp].reindex(_BAND_ORDER)
        for j, band in enumerate(_BAND_ORDER):
            x = i + (j - (n_bands - 1) / 2) * width
            bottom = 0.0
            for bucket in _BUCKETS:
                h = float(table.loc[band, bucket])
                ax.bar(
                    x,
                    h,
                    width=width * 0.95,
                    bottom=bottom,
                    color=_BUCKET_COLOURS[bucket],
                    edgecolor="black",
                    linewidth=0.3,
                )
                bottom += h
            x_centres.append(x)
            x_labels.append(_BAND_LABEL[band])

    # Per-SSP group label below the band-level ticks.
    for i, ssp in enumerate(ssps):
        ax.text(
            i,
            ax.get_ylim()[0] - 0.07 * ax.get_ylim()[1],
            ssp.upper(),
            ha="center",
            va="top",
            fontsize=6,
        )

    ax.set_xticks(np.arange(n_ssps))
    ax.set_xticklabels(["" for _ in ssps])
    ax.set_ylabel("mean |SHAP|")
    ax.set_title("Driver attribution per landfall band")

    # Manual legend (avoid double-counting from stack).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_BUCKET_COLOURS[b], edgecolor="black", linewidth=0.3)
        for b in _BUCKETS
    ]
    ax.legend(handles, list(_BUCKETS), loc="upper right")

    # Secondary x-axis labels for the bands.
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(x_centres)
    sec.set_xticklabels(x_labels, rotation=45, ha="left", fontsize=5)
    sec.tick_params(length=1.0)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=str(config.CACHE_DIR))
    parser.add_argument("--out", type=str, default="figures/fig4_shap_drivers.pdf")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    per_ssp: dict[str, pd.DataFrame] = {}
    for ssp in config.CMIP6_EXPERIMENTS:
        path = cache_dir / f"shap_attribution_{ssp}.parquet"
        if path.exists():
            per_ssp[ssp] = pd.read_parquet(path)

    fig = plot(per_ssp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
