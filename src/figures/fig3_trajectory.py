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

_OBS_ORDER = ("pre_sat_1940_1959", "modern_1980_1999", "recent_2015_2024")
_OBS_LABEL = {
    "pre_sat_1940_1959": "Pre-sat\n1940-59",
    "modern_1980_1999": "Modern\n1980-99",
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


def _window_years(period_name: str) -> float:
    """Length in calendar years of an observational window, from config."""
    for name, (start, end) in config.OBSERVATIONAL_PERIODS:
        if name == period_name:
            return float(int(end[:4]) - int(start[:4]) + 1)
    return float("nan")


def _plot_observed_only(observed: pd.DataFrame) -> plt.Figure:
    """ERA5-only headline figure: four panels across the three observational
    windows — (a) event frequency normalised to events per decade (the
    windows are 20/20/10 yr, so raw counts are not directly comparable),
    (b) mean intensity, (c) mean footprint, (d) mean duration, with 90 %
    bootstrap bands where ``period_contrast`` provides them (intensity,
    duration)."""
    apply_nature_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.62), constrained_layout=True
    )
    obs = observed.set_index("period_name").reindex(_OBS_ORDER)
    x = np.arange(len(_OBS_ORDER))
    labels = [_OBS_LABEL[p] for p in _OBS_ORDER]
    years = np.array([_window_years(p) for p in _OBS_ORDER])

    # (a) frequency, normalised to events per decade to remove window-length bias
    ax = axes[0, 0]
    per_decade = obs["n_events"].to_numpy() / years * 10.0
    ax.bar(x, per_decade, color=_OBS_COLOUR, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("AR events per decade")
    ax.set_title("(a) AR-event frequency")

    # (b) mean intensity with bootstrap band
    ax = axes[0, 1]
    med = obs["mean_intensity"].to_numpy()
    lo = obs["intensity_q05"].to_numpy()
    hi = obs["intensity_q95"].to_numpy()
    ax.errorbar(
        x,
        med,
        yerr=[np.maximum(med - lo, 0), np.maximum(hi - med, 0)],
        fmt="o",
        markersize=3,
        color=_OBS_COLOUR,
        capsize=2,
        lw=0.6,
    )
    ax.set_ylabel("mean AR intensity (kg m⁻¹ s⁻¹)")
    ax.set_title("(b) AR intensity")

    # (c) mean footprint (no bootstrap band in the summary)
    ax = axes[1, 0]
    ax.bar(
        x,
        obs["mean_footprint_km2"].to_numpy() / 1e6,
        color=_OBS_COLOUR,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_ylabel("mean footprint (10⁶ km²)")
    ax.set_title("(c) AR footprint")

    # (d) mean duration with bootstrap band
    ax = axes[1, 1]
    dmed = obs["mean_duration_h"].to_numpy()
    dlo = obs["duration_q05"].to_numpy()
    dhi = obs["duration_q95"].to_numpy()
    ax.errorbar(
        x,
        dmed,
        yerr=[np.maximum(dmed - dlo, 0), np.maximum(dhi - dmed, 0)],
        fmt="o",
        markersize=3,
        color=_OBS_COLOUR,
        capsize=2,
        lw=0.6,
    )
    ax.set_ylabel("mean duration (h)")
    ax.set_title("(d) AR duration")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    return fig


def plot(
    observed: pd.DataFrame,
    projected: dict[tuple[str, str], pd.DataFrame],
    observed_only: bool = False,
) -> plt.Figure:
    """Build the trajectory figure from in-memory inputs.

    ``observed_only=True`` renders just the three observational windows
    (Pre-sat / Modern / Recent) with no projected CMIP6 half — the form
    used by the ERA5-only manuscript, where the projected panels would be
    synthetic and must not appear.
    """
    if observed_only:
        return _plot_observed_only(observed)

    apply_nature_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.55), constrained_layout=True
    )

    obs_df = observed.set_index("period_name").reindex(_OBS_ORDER)
    n_obs = len(_OBS_ORDER)
    n_ssp = 0 if observed_only else len(_SSP_ORDER)
    x = np.arange(n_obs + n_ssp)

    # ----- panel (a): event count -----
    ax = axes[0]
    obs_counts = obs_df["n_events"].to_numpy()
    ax.bar(x[:n_obs], obs_counts, color=_OBS_COLOUR, edgecolor="black", linewidth=0.4)

    width = 0.36
    if not observed_only:
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
    obs_ticks = [_OBS_LABEL[p] for p in _OBS_ORDER]
    ssp_ticks = [] if observed_only else [_SSP_LABEL[s] for s in _SSP_ORDER]
    ax.set_xticklabels(obs_ticks + ssp_ticks)
    ax.set_ylabel("events / window")
    ax.set_title("(a) AR-event frequency")
    if not observed_only:
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

    if not observed_only:
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
    ax.set_xticklabels(obs_ticks + ssp_ticks)
    ax.set_ylabel(
        "mean AR intensity (kg m⁻¹ s⁻¹)"
        if observed_only
        else "mean AR intensity / precip\n(observed / projected)"
    )
    ax.set_title("(b) Trajectory of AR intensity")
    if not observed_only:
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
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Render only the three observational windows (ERA5-only manuscript); "
        "omit the synthetic projected CMIP6 half.",
    )
    args = parser.parse_args()

    observed = pd.read_parquet(args.observed)
    projected = {} if args.observed_only else _read_projected(config.CACHE_DIR, args.source)

    fig = plot(observed, projected, observed_only=args.observed_only)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
