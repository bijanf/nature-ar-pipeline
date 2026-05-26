"""
Figure 3 — Observed-then-projected AR trajectory (Story A's headline figure).

A single x-axis ordered:

    Pre-sat | Modern | Recent | SSP2-4.5 | SSP3-7.0 | SSP4-6.0 | SSP5-8.5
    └─────── observed ──────┘ │ └────────────── projected ─────────────┘
                              ↑
                            vertical separator

Two stacked panels (double-column):

a) AR-event count per 30-year window.
b) Mean event-total precipitation (mm) with bootstrap / quantile bands.

The observed half (left of the separator) comes from
``data/cache/period_contrast.parquet`` produced by
:mod:`src.analysis.period_contrast`. The projected half (right of separator)
comes from ``data/cache/cmip6_events_<source>_<exp>_<mode>.parquet`` for each
SSP × {direct, delta}.

This figure is the empirical+projective spine of the paper: a reviewer who
distrusts the ML can still see the *observed* shift on the left half.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_OBS_ORDER = ("pre_sat_1940_1979", "modern_1980_2014", "recent_2015_2024")
_OBS_LABEL = {
    "pre_sat_1940_1979": "Pre-sat\n1940-79",
    "modern_1980_2014": "Modern\n1980-2014",
    "recent_2015_2024": "Recent\n2015-24",
}
_SSP_ORDER = ("ssp245", "ssp370", "ssp460", "ssp585")
_SSP_LABEL = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",  # headline
    "ssp460": "SSP4-6.0",
    "ssp585": "SSP5-8.5",
}
_OBS_COLOUR = "#34495e"  # slate
_HEADLINE_SSP = "ssp370"


def _ssp_colour(ssp: str) -> str:
    palette = {
        "ssp245": "#2980b9",
        "ssp370": "#e67e22",  # headline emphasis
        "ssp460": "#9b59b6",
        "ssp585": "#c0392b",
    }
    return palette[ssp]


def _read_projected(cache_dir: Path, source_id: str) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for ssp in _SSP_ORDER:
        for mode in ("direct", "delta"):
            path = cache_dir / f"cmip6_events_{source_id}_{ssp}_{mode}.parquet"
            if path.exists():
                out[(ssp, mode)] = pd.read_parquet(path)
    return out


def plot(
    observed: pd.DataFrame,
    projected: dict[tuple[str, str], pd.DataFrame],
) -> plt.Figure:
    """Build the trajectory figure from in-memory inputs."""
    apply_nature_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.55), constrained_layout=True
    )

    obs_df = observed.set_index("period_name").reindex(_OBS_ORDER)
    n_obs = len(_OBS_ORDER)
    n_ssp = len(_SSP_ORDER)
    x = np.arange(n_obs + n_ssp)

    # ----- panel (a): event count -----
    ax = axes[0]
    obs_counts = obs_df["n_events"].to_numpy()
    ax.bar(x[:n_obs], obs_counts, color=_OBS_COLOUR, edgecolor="black", linewidth=0.4)

    width = 0.36
    for i, mode in enumerate(("direct", "delta")):
        counts = np.array(
            [float(len(projected.get((ssp, mode), pd.DataFrame()))) for ssp in _SSP_ORDER]
        )
        ax.bar(
            x[n_obs:] + (i - 0.5) * width,
            counts,
            width=width,
            color=[_ssp_colour(s) for s in _SSP_ORDER],
            alpha=1.0 if mode == "direct" else 0.55,
            edgecolor="black",
            linewidth=0.4,
            label=mode,
        )

    ax.axvline(n_obs - 0.5, color="black", lw=0.6, linestyle="--")
    ax.text(n_obs - 0.6, ax.get_ylim()[1] * 0.95, "observed", ha="right", va="top", fontsize=5)
    ax.text(n_obs - 0.4, ax.get_ylim()[1] * 0.95, "projected", ha="left", va="top", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([_OBS_LABEL[p] for p in _OBS_ORDER] + [_SSP_LABEL[s] for s in _SSP_ORDER])
    ax.set_ylabel("events / 30 years")
    ax.set_title("(a) AR-event frequency")
    ax.legend(title="protocol", loc="upper left", ncol=2)

    # ----- panel (b): mean event-total precip with bands -----
    ax = axes[1]
    # Observed: use the period bootstrap bands.
    obs_med = obs_df["mean_intensity"].to_numpy()
    obs_lo = obs_df["intensity_q05"].to_numpy()
    obs_hi = obs_df["intensity_q95"].to_numpy()
    ax.errorbar(
        x[:n_obs],
        obs_med,
        yerr=[obs_med - obs_lo, obs_hi - obs_med],
        fmt="o",
        markersize=3,
        color=_OBS_COLOUR,
        capsize=2,
        lw=0.6,
    )

    for i, mode in enumerate(("direct", "delta")):
        meds = []
        los = []
        his = []
        for ssp in _SSP_ORDER:
            df = projected.get((ssp, mode), pd.DataFrame())
            if df.empty or "precip_pred_q50_mm" not in df.columns:
                meds.append(float("nan"))
                los.append(float("nan"))
                his.append(float("nan"))
                continue
            meds.append(float(df["precip_pred_q50_mm"].median()))
            los.append(float(df["precip_pred_q05_mm"].median()))
            his.append(float(df["precip_pred_q95_mm"].median()))
        meds = np.array(meds)
        los = np.array(los)
        his = np.array(his)
        marker = "o" if mode == "direct" else "s"
        ax.errorbar(
            x[n_obs:] + (i - 0.5) * 0.18,
            meds,
            yerr=[np.maximum(meds - los, 0), np.maximum(his - meds, 0)],
            fmt=marker,
            markersize=3,
            color="black",
            ecolor="black",
            capsize=2,
            lw=0.6,
            label=mode,
        )

    ax.axvline(n_obs - 0.5, color="black", lw=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([_OBS_LABEL[p] for p in _OBS_ORDER] + [_SSP_LABEL[s] for s in _SSP_ORDER])
    ax.set_ylabel("mean AR intensity / precip\n(observed / projected)")
    ax.set_title("(b) Trajectory of AR intensity")
    ax.legend(title="proj. protocol", loc="upper left")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observed",
        type=str,
        default=str(config.CACHE_DIR / "period_contrast.parquet"),
    )
    parser.add_argument("--source", type=str, default=config.CMIP6_QUERY_BASE["source_id"])
    parser.add_argument("--out", type=str, default="figures/fig3_trajectory.pdf")
    args = parser.parse_args()

    observed = pd.read_parquet(args.observed)
    projected = _read_projected(config.CACHE_DIR, args.source)

    fig = plot(observed, projected)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
