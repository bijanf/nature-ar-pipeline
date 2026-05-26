"""
Per-period observational statistics — the empirical core of Story A.

For each window in :data:`config.OBSERVATIONAL_PERIODS` we compute:

* Holton-dynamics features from ARCO-ERA5 (lazy);
* Guan-Waliser AR mask using a *period-internal* climatology so trends in
  IVT do not artificially erode the detection threshold;
* discrete events via :func:`src.models.event_post.extract_events`;
* per-event aggregates (count, mean / max intensity, footprint area, ERA5
  precip integral).

Output: one row per period plus bootstrap 5/50/95 % bands on the means.

Why a *period-internal* climatology matters:
A single 1940-2024 climatology would set the 85th-percentile threshold by the
*long-term* IVT distribution. If late-period IVT has risen, a fixed threshold
will pick up more pixels and we'd be confounding genuine intensification with
threshold drift. Period-internal climatology measures shifts *relative to each
period's own normal* — the cleanest empirical statement.

Caveat (must surface in the paper):
ERA5 1940-1978 (HRES + ERA5-BE back-extension) has weaker observational
constraint, particularly on moisture. We treat the pre-satellite period as
a *climatological reference* — adequate for trend detection at the basin
scale, inadequate for individual-event validation. The ML never trains on
pre-satellite data.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.features import ar_detection, physics_pipeline
from src.models import event_post

_BOOT_N = 1000
_BOOT_QUANTILES = (0.05, 0.50, 0.95)


@dataclass(frozen=True)
class PeriodSummary:
    period_name: str
    n_events: int
    mean_intensity: float
    mean_max_intensity: float
    mean_footprint_km2: float
    mean_duration_h: float
    # Bootstrap bands on the mean of each event-level statistic.
    intensity_q05: float
    intensity_q95: float
    duration_q05: float
    duration_q95: float


def _open_period_features(period: tuple[str, str]) -> tuple[xr.Dataset, xr.DataArray]:
    """Return (lazy Holton-features dataset, lazy IVT-only DataArray) for ``period``."""
    ds = physics_pipeline.open_arco_era5()[list(physics_pipeline.required_era5_vars())]
    ds = ds.sel(time=slice(*period))
    feats = physics_pipeline.calculate_dynamics(ds)
    return feats, feats["ivt"]


def _events_for_period(
    period_name: str, period: tuple[str, str], era5_topo: xr.Dataset
) -> pd.DataFrame:
    """Stream + compute one period's events. One materialisation pass."""
    feats, ivt = _open_period_features(period)

    # Period-internal climatology cached under a period-specific filename.
    clim = ar_detection.compute_ivt_climatology(ivt.chunk({"time": -1}), period=period)
    mask_lazy = ar_detection.compute_ar_mask(feats["ivt"], feats["ivt_u"], feats["ivt_v"], clim)
    intensity = (mask_lazy.astype("float32") * feats["ivt"]).rename("ar_intensity")

    intensity = intensity.compute()
    events = event_post.extract_events(
        intensity,
        land_sea_mask=era5_topo["land_sea_mask"],
        threshold=config.STAGE1_EVENT_THRESHOLD,
    )
    events.insert(0, "period", period_name)
    return events


def _bootstrap_mean(values: np.ndarray, n: int = _BOOT_N, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in _BOOT_QUANTILES}
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    qs = np.quantile(means, _BOOT_QUANTILES)
    return {f"q{int(q * 100):02d}": float(v) for q, v in zip(_BOOT_QUANTILES, qs, strict=True)}


def summarise_period(events: pd.DataFrame, period_name: str) -> PeriodSummary:
    """Per-period scalar aggregates plus bootstrap bands on the means."""
    ints = events["mean_intensity"].to_numpy()
    maxs = events["max_intensity"].to_numpy()
    foot = events["footprint_area_km2"].to_numpy()
    dur = events["duration_hours"].to_numpy()
    boot_int = _bootstrap_mean(ints)
    boot_dur = _bootstrap_mean(dur)
    return PeriodSummary(
        period_name=period_name,
        n_events=len(events),
        mean_intensity=float(np.mean(ints)) if ints.size else float("nan"),
        mean_max_intensity=float(np.mean(maxs)) if maxs.size else float("nan"),
        mean_footprint_km2=float(np.mean(foot)) if foot.size else float("nan"),
        mean_duration_h=float(np.mean(dur)) if dur.size else float("nan"),
        intensity_q05=boot_int["q05"],
        intensity_q95=boot_int["q95"],
        duration_q05=boot_dur["q05"],
        duration_q95=boot_dur["q95"],
    )


def run(periods: Iterable[tuple[str, tuple[str, str]]] = config.OBSERVATIONAL_PERIODS) -> Path:
    """Materialise per-period events + summaries, write Parquet."""
    from src.features import topography

    era5_topo = topography.open_era5_topography().compute()

    all_events: list[pd.DataFrame] = []
    summaries: list[PeriodSummary] = []
    for period_name, period in periods:
        events = _events_for_period(period_name, period, era5_topo)
        all_events.append(events)
        summaries.append(summarise_period(events, period_name))

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    events_df = pd.concat(all_events, ignore_index=True)
    events_df.to_parquet(config.CACHE_DIR / "observational_events.parquet", index=False)

    summary_df = pd.DataFrame([vars(s) for s in summaries])
    summary_df.to_parquet(config.CACHE_DIR / "period_contrast.parquet", index=False)

    return config.CACHE_DIR / "period_contrast.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--periods",
        type=str,
        nargs="*",
        default=None,
        help="Subset of period names (default: all of config.OBSERVATIONAL_PERIODS).",
    )
    args = parser.parse_args()

    if args.periods:
        periods = [(n, p) for n, p in config.OBSERVATIONAL_PERIODS if n in args.periods]
    else:
        periods = list(config.OBSERVATIONAL_PERIODS)

    out = run(periods)
    df = pd.read_parquet(out)
    print(json.dumps(df.to_dict(orient="records"), indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
