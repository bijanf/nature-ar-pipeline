"""
Figure 3 — Four-SSP shift in AR-event count and mean intensity,
direct vs delta side-by-side.

Two panels (double-column):

a) Per-SSP **event count** under each protocol, compared to the ERA5
   historical 1980–2014 baseline (horizontal grey reference line).
b) Per-SSP **median predicted event-total precip (mm)** with 5–95 % IPI bars
   from the Stage 2 quantile predictions.

Reads ``data/cache/cmip6_events_<source>_<exp>_<mode>.parquet`` for every
``exp`` in :data:`config.CMIP6_EXPERIMENTS` and ``mode`` ∈ {direct, delta}.
The historical baseline is read from
``data/cache/historical_events.parquet`` (produced by event_post on the
ERA5 train+holdout intensity field).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.figures._style import COL_DOUBLE_IN, apply_nature_style

_SSP_COLOURS = {
    "ssp245": "#2980b9",  # blue — middle-of-road
    "ssp370": "#e67e22",  # orange — CMIP7-aligned headline
    "ssp460": "#9b59b6",  # purple — inequality pathway
    "ssp585": "#c0392b",  # red — upper-bound stress test
}
_MODES = ("direct", "delta")


def _summarise(events: pd.DataFrame) -> dict[str, float]:
    """Per-frame summary statistics used by both panels."""
    return {
        "n_events": float(len(events)),
        "median_q50": float(events["precip_pred_q50_mm"].median()) if len(events) else float("nan"),
        "p05": float(events["precip_pred_q05_mm"].median()) if len(events) else float("nan"),
        "p95": float(events["precip_pred_q95_mm"].median()) if len(events) else float("nan"),
    }


def plot(
    per_scenario: dict[tuple[str, str], pd.DataFrame],
    historical: pd.DataFrame,
) -> plt.Figure:
    apply_nature_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.35), constrained_layout=True
    )

    ssps = list(config.CMIP6_EXPERIMENTS)
    x = np.arange(len(ssps))
    width = 0.36

    hist_count = float(len(historical))
    hist_precip = float(historical["precip_total_mm"].median()) if len(historical) else float("nan")

    # ----- panel (a): event count -----
    ax = axes[0]
    for i, mode in enumerate(_MODES):
        counts = np.array([_summarise(per_scenario[(ssp, mode)])["n_events"] for ssp in ssps])
        ax.bar(
            x + (i - 0.5) * width,
            counts,
            width=width,
            color=[_SSP_COLOURS[s] for s in ssps],
            edgecolor="black",
            linewidth=0.4,
            alpha=1.0 if mode == "direct" else 0.55,
            label=mode,
        )
    ax.axhline(hist_count, color="grey", linestyle="--", lw=0.5)
    ax.text(len(ssps) - 0.5, hist_count, "  ERA5 hist", color="grey", va="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in ssps])
    ax.set_ylabel("events / 30 years")
    ax.set_title("(a) AR-event frequency")
    ax.legend(title="protocol", loc="upper left")

    # ----- panel (b): median precip with 5/95 bands -----
    ax = axes[1]
    for i, mode in enumerate(_MODES):
        meds = np.array([_summarise(per_scenario[(ssp, mode)])["median_q50"] for ssp in ssps])
        los = np.array([_summarise(per_scenario[(ssp, mode)])["p05"] for ssp in ssps])
        his = np.array([_summarise(per_scenario[(ssp, mode)])["p95"] for ssp in ssps])
        ax.bar(
            x + (i - 0.5) * width,
            meds,
            width=width,
            color=[_SSP_COLOURS[s] for s in ssps],
            edgecolor="black",
            linewidth=0.4,
            alpha=1.0 if mode == "direct" else 0.55,
            label=mode,
        )
        # 5–95 % error bars per (SSP, mode).
        yerr_lo = np.maximum(meds - los, 0)
        yerr_hi = np.maximum(his - meds, 0)
        ax.errorbar(
            x + (i - 0.5) * width,
            meds,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor="black",
            lw=0.4,
            capsize=1.5,
        )
    ax.axhline(hist_precip, color="grey", linestyle="--", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in ssps])
    ax.set_ylabel("event-total precip (mm)")
    ax.set_title("(b) Median event precip with 5–95 % band")

    return fig


def _read_per_scenario(cache_dir: Path, source_id: str) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for ssp in config.CMIP6_EXPERIMENTS:
        for mode in _MODES:
            path = cache_dir / f"cmip6_events_{source_id}_{ssp}_{mode}.parquet"
            out[(ssp, mode)] = pd.read_parquet(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=config.CMIP6_QUERY_BASE["source_id"])
    parser.add_argument(
        "--historical", type=str, default=str(config.CACHE_DIR / "historical_events.parquet")
    )
    parser.add_argument("--out", type=str, default="figures/fig3_ssp_shift.pdf")
    args = parser.parse_args()

    per_scenario = _read_per_scenario(config.CACHE_DIR, args.source)
    historical = pd.read_parquet(args.historical)

    fig = plot(per_scenario, historical)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
