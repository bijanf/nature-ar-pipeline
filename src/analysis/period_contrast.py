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

import dask
import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.features import ar_detection, event_features, physics_pipeline
from src.models import event_post

# Force a single-threaded synchronous Dask scheduler. The io partition caps
# each job at ~20 GB and 4 CPUs; 4 parallel workers each pulling a full
# upstream pressure-level tile blows past the cap. The single-threaded
# scheduler keeps the in-flight working set to one chunk at a time, at the
# cost of wall-clock time — acceptable given a node-hour-cheap io job.
dask.config.set(scheduler="synchronous")

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
    """Return (lazy Holton-features dataset, lazy IVT-only DataArray) for ``period``.

    The Guan-Waliser detector is calibrated for 6-hourly cadence (GW §3.2).
    The ARCO-ERA5 v3 Zarr is hourly on disk and is sub-sampled by 6× here;
    the local CDS NetCDF cache is already 6-hourly (one timestep per
    request slot), so we detect the native cadence at the load boundary
    and only sub-sample when it's still hourly.
    """
    ds = physics_pipeline.open_arco_era5()[list(physics_pipeline.required_era5_vars())]
    ds = ds.sel(time=slice(*period))
    if ds.sizes["time"] >= 2:
        dt = np.asarray(ds["time"][1] - ds["time"][0], dtype="timedelta64[h]")
        if dt < np.timedelta64(6, "h"):
            stride = max(1, int(np.timedelta64(6, "h") / dt))
            ds = ds.isel(time=slice(None, None, stride))
    feats = physics_pipeline.calculate_dynamics(ds)
    return feats, feats["ivt"]


def _events_for_period(
    period_name: str,
    period: tuple[str, str],
    era5_topo: xr.Dataset,
    sub_window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Stream + compute one period's events. One materialisation pass.

    ``sub_window`` lets a memory-capped job materialise a decade-sized slice
    while still using the full-period climatology, so the GW threshold stays
    period-consistent across chunked runs.
    """
    feats, ivt = _open_period_features(period)

    # Period-internal climatology cached under a period-specific filename.
    clim = ar_detection.compute_ivt_climatology(ivt.chunk({"time": -1}), period=period)

    if sub_window is not None and sub_window != period:
        feats = feats.sel(time=slice(*sub_window))

    mask_lazy = ar_detection.compute_ar_mask(feats["ivt"], feats["ivt_u"], feats["ivt_v"], clim)
    intensity = (mask_lazy.astype("float32") * feats["ivt"]).rename("ar_intensity")

    # One materialisation pass over the (sub-)window: intensity + the six
    # physics features SHAP will consume. Without this, Phase 5d attribution
    # against the observational record has no row-per-event to operate on.
    feats_for_events = feats[list(event_features._PHYSICS_FEATURES)]
    materialised = xr.merge([intensity, feats_for_events]).compute()

    events, labels = event_post.extract_events_with_labels(
        materialised["ar_intensity"],
        land_sea_mask=era5_topo["land_sea_mask"],
        threshold=config.STAGE1_EVENT_THRESHOLD,
    )
    events = event_features.attach_features(events, labels, materialised)
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


def run(
    periods: Iterable[tuple[str, tuple[str, str]]] = config.OBSERVATIONAL_PERIODS,
    out_suffix: str = "",
    time_range: tuple[str, str] | None = None,
) -> Path:
    """Materialise per-period events + summaries, write Parquet.

    ``out_suffix`` lets a SLURM job own a single period's output without racing
    other periods that target the same unified file. Example: with
    ``out_suffix="modern"`` the writes go to
    ``observational_events_modern.parquet`` and ``period_contrast_modern.parquet``.
    Use :func:`concat_periods` to merge the per-suffix parquets into the unified
    files that the figure scripts read.

    ``time_range`` sub-slices each period's event extraction to a decade-sized
    window so a memory-capped SLURM job can fit one chunk; the period-internal
    climatology is still computed over the full period (cached by
    :func:`ar_detection.compute_ivt_climatology`), so chunked runs share the
    same threshold and trend-detection invariant holds.
    """
    from src.features import topography

    era5_topo = topography.open_era5_topography().compute()

    all_events: list[pd.DataFrame] = []
    summaries: list[PeriodSummary] = []
    for period_name, period in periods:
        sub_window = time_range if time_range is not None else period
        events = _events_for_period(period_name, period, era5_topo, sub_window=sub_window)
        all_events.append(events)
        summaries.append(summarise_period(events, period_name))

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    events_df = pd.concat(all_events, ignore_index=True)
    summary_df = pd.DataFrame([vars(s) for s in summaries])

    tag = f"_{out_suffix}" if out_suffix else ""
    events_df.to_parquet(config.CACHE_DIR / f"observational_events{tag}.parquet", index=False)
    summary_df.to_parquet(config.CACHE_DIR / f"period_contrast{tag}.parquet", index=False)

    return config.CACHE_DIR / f"period_contrast{tag}.parquet"


def concat_periods(suffixes: Iterable[str], out_suffix: str = "") -> Path:
    """Merge per-suffix event parquets into a single unified Story-A output.

    Reads ``observational_events_{suffix}.parquet`` for each suffix, concatenates
    the rows, recomputes :class:`PeriodSummary` per ``period`` column, and writes
    ``observational_events{tag}.parquet`` + ``period_contrast{tag}.parquet``,
    where ``tag = f"_{out_suffix}"`` if non-empty (else ``""`` — the unified
    Story-A files that ``fig3_trajectory`` and ``fig5_landfall_density`` consume).
    """
    parts = [
        pd.read_parquet(config.CACHE_DIR / f"observational_events_{s}.parquet")
        for s in suffixes
    ]
    events_df = pd.concat(parts, ignore_index=True)
    tag = f"_{out_suffix}" if out_suffix else ""
    events_df.to_parquet(config.CACHE_DIR / f"observational_events{tag}.parquet", index=False)

    summaries = [
        summarise_period(g.reset_index(drop=True), period_name)
        for period_name, g in events_df.groupby("period", sort=False)
    ]
    summary_df = pd.DataFrame([vars(s) for s in summaries])
    summary_df.to_parquet(config.CACHE_DIR / f"period_contrast{tag}.parquet", index=False)

    return config.CACHE_DIR / f"period_contrast{tag}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--periods",
        type=str,
        nargs="*",
        default=None,
        help="Subset of period names (default: all of config.OBSERVATIONAL_PERIODS).",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Per-period output mode: write to observational_events_{suffix}.parquet.",
    )
    parser.add_argument(
        "--concat",
        type=str,
        nargs="+",
        default=None,
        help="Concat suffixes into unified parquets and exit (no streaming).",
    )
    parser.add_argument(
        "--time-range",
        type=str,
        default=None,
        help="Sub-slice each period to YYYY-MM-DD:YYYY-MM-DD (climatology stays full-period).",
    )
    args = parser.parse_args()

    if args.concat:
        out = concat_periods(args.concat, out_suffix=args.out_suffix)
    else:
        if args.periods:
            periods = [(n, p) for n, p in config.OBSERVATIONAL_PERIODS if n in args.periods]
        else:
            periods = list(config.OBSERVATIONAL_PERIODS)

        sub: tuple[str, str] | None = None
        if args.time_range:
            start, end = args.time_range.split(":", 1)
            sub = (start, end)

        out = run(periods, out_suffix=args.out_suffix, time_range=sub)

    df = pd.read_parquet(out)
    print(json.dumps(df.to_dict(orient="records"), indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
