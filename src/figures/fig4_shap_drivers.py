"""
Figure 4 — Thermodynamic vs dynamic driver attribution along the
observed-then-projected trajectory of Story A.

Two stacked panels (double-column):

a) **Observed**: Pre-sat | Modern | Recent — three groups of stacked bars per
   landfall band (25–35 N, 35–45 N, 45–60 N). Bar height = mean |SHAP|
   contribution, stacked into thermodynamic / dynamic / other buckets.

b) **Projected**: SSP2-4.5 | SSP3-7.0 | SSP4-6.0 | SSP5-8.5 — four groups
   per band, same colour code as (a). Same y-axis as (a) so a reviewer can
   read the bucket mix along the same |SHAP| ruler in both halves.

The story: the bucket-mix at each landfall band shifts continuously from the
left half (observed) into the right half (projected) — the projected SSP
attribution is not a different kind of evidence, just the same explainer
evaluated further along the trajectory.

Inputs
------
Per-source attribution Parquet files written by
:mod:`src.models.explainability`:

* observed:  ``shap_attribution_<period_name>.parquet`` for each
  ``period_name`` in :data:`src.config.OBSERVATIONAL_PERIODS`.
* projected: ``shap_attribution_<ssp>.parquet`` for each ``ssp`` in
  :data:`src.config.CMIP6_EXPERIMENTS`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_BUCKETS = ("thermodynamic", "dynamic", "other")
_BUCKET_COLOURS = {
    "thermodynamic": "#c0392b",
    "dynamic": "#2980b9",
    "other": "#bdc3c7",
}
_BAND_ORDER = ("south_25_35N", "central_35_45N", "north_45_60N")
_BAND_LABEL = {
    "south_25_35N": "25-35°N",
    "central_35_45N": "35-45°N",
    "north_45_60N": "45-60°N",
}
_OBS_ORDER = ("pre_sat_1940_1979", "modern_1980_2014", "recent_2015_2024")
_OBS_LABEL = {
    "pre_sat_1940_1979": "Pre-sat",
    "modern_1980_2014": "Modern",
    "recent_2015_2024": "Recent",
}
_SSP_LABEL = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp460": "SSP4-6.0",
    "ssp585": "SSP5-8.5",
}


def _stacked_group(ax: plt.Axes, x: float, table: pd.DataFrame, width: float) -> None:
    """Draw one (group_x, 3-band) stacked-bar trio at ``x``."""
    for j, band in enumerate(_BAND_ORDER):
        bx = x + (j - (len(_BAND_ORDER) - 1) / 2) * width
        bottom = 0.0
        for bucket in _BUCKETS:
            h = float(table.loc[band, bucket]) if band in table.index else 0.0
            ax.bar(
                bx,
                h,
                width=width * 0.95,
                bottom=bottom,
                color=_BUCKET_COLOURS[bucket],
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += h


def plot(
    observed: dict[str, pd.DataFrame],
    projected: dict[str, pd.DataFrame],
) -> plt.Figure:
    """2-row driver attribution along the Story-A trajectory."""
    apply_nature_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.6), constrained_layout=True
    )

    # Shared y-limit so observed and projected sit on the same |SHAP| ruler.
    all_heights: list[float] = []
    for src in (*observed.values(), *projected.values()):
        for band in _BAND_ORDER:
            if band in src.index:
                all_heights.append(float(src.loc[band, list(_BUCKETS)].sum()))
    ymax = max(all_heights) * 1.05 if all_heights else 1.0

    # ----- (a) observed -----
    ax = axes[0]
    width = 0.22
    obs_present = [p for p in _OBS_ORDER if p in observed]
    for i, p in enumerate(obs_present):
        _stacked_group(ax, float(i), observed[p].reindex(_BAND_ORDER), width)
    ax.set_xticks(np.arange(len(obs_present)))
    ax.set_xticklabels([_OBS_LABEL[p] for p in obs_present])
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("mean |SHAP|")
    ax.set_title("(a) Observed driver attribution (1940-2024)")

    # ----- (b) projected -----
    ax = axes[1]
    ssps_present = [s for s in config.CMIP6_EXPERIMENTS if s in projected]
    for i, s in enumerate(ssps_present):
        _stacked_group(ax, float(i), projected[s].reindex(_BAND_ORDER), width)
    ax.set_xticks(np.arange(len(ssps_present)))
    ax.set_xticklabels([_SSP_LABEL[s] for s in ssps_present])
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("mean |SHAP|")
    ax.set_title("(b) Projected driver attribution (2070-2099)")

    # Shared legend (drawn on the bottom panel only).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_BUCKET_COLOURS[b], edgecolor="black", linewidth=0.3)
        for b in _BUCKETS
    ]
    axes[1].legend(handles, list(_BUCKETS), loc="upper right", ncol=3)

    # Band-trio annotation under each group on both axes.
    for ax in axes:
        for x_centre in ax.get_xticks():
            for j, band in enumerate(_BAND_ORDER):
                bx = x_centre + (j - (len(_BAND_ORDER) - 1) / 2) * width
                ax.text(bx, -0.03 * ymax, _BAND_LABEL[band], ha="center", va="top", fontsize=4.5)

    return fig


def _read(cache_dir: Path, key: str) -> pd.DataFrame | None:
    path = cache_dir / f"shap_attribution_{key}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=str(config.CACHE_DIR))
    parser.add_argument("--out", type=str, default="figures/fig4_shap_drivers.pdf")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    observed = {p: _read(cache_dir, p) for p in _OBS_ORDER}
    observed = {p: df for p, df in observed.items() if df is not None}
    projected = {s: _read(cache_dir, s) for s in config.CMIP6_EXPERIMENTS}
    projected = {s: df for s, df in projected.items() if df is not None}

    fig = plot(observed, projected)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
