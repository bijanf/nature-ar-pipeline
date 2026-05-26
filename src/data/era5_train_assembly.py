"""
Stream ARCO-ERA5, derive Holton-dynamics features + Guan-Waliser AR mask, and
write a per-pixel stratified Parquet table for Stage 1.

Layout
------
Output is a Hive-partitioned directory of Parquet files::

    data/cache/train_rows/year=1980/part.parquet
    data/cache/train_rows/year=1981/part.parquet
    ...

(`holdout_rows/...` when ``holdout=True``.) Per-year partitioning lets the
ingest crash and resume cleanly, and lets PyArrow read selected years without
loading the rest.

Sampling
--------
* Source cadence ARCO-ERA5 is 1-hourly; we sub-select to 6-hourly (00/06/12/18 UTC)
  to match the canonical Guan-Waliser application cadence.
* Optionally further sub-sample timesteps by ``timestep_subsample`` for training.
  Holdout keeps every 6-hourly step.
* Per year we keep **all GW-positive pixels** + a random ``n_neg_per_pos × n_pos``
  sample of GW-negative pixels (within that year). All positives ensures the
  AR-conditional IVT distribution is fully represented; the negative cap keeps
  the table tractable for LightGBM.

Engineering notes
-----------------
* Each year is one materialised pass through the lazy graph: ``ds.compute()``
  pulls features + GW mask + target together so the GCS stream is read only
  once per year.
* The IVT climatology is loaded once at the top of the loop and shared across
  years (it depends on the full training window, not the year being assembled).
* Latitude and ``doy_sin/doy_cos`` are added after materialisation — they are
  cheap NumPy ops and don't justify being in the Dask graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.features import ar_detection, physics_pipeline

# Columns written to Parquet (order matters for downstream stability).
_FEATURE_COLS = (
    "ivt",
    "ivt_u",
    "ivt_v",
    "theta_e_850",
    "pv_250",
    "eady_growth_rate",
    "lat_deg",
    "doy_sin",
    "doy_cos",
)
_TARGET_COL = "ar_intensity"
_MASK_COL = "ar_mask"


# =============================================================================
# Helpers
# =============================================================================


def _lazy_features_and_climatology() -> tuple[xr.Dataset, xr.DataArray]:
    """Open ARCO-ERA5, compute Holton features lazily, return them alongside
    the cached training-period IVT climatology."""
    ds = physics_pipeline.open_arco_era5()[list(physics_pipeline.required_era5_vars())]
    feats = physics_pipeline.calculate_dynamics(ds)
    clim = ar_detection.compute_ivt_climatology(feats["ivt"])
    return feats, clim


def _select_6h_subsample(
    feats: xr.Dataset,
    timestep_subsample: int,
    rng: np.random.Generator,
) -> xr.Dataset:
    """Restrict to 00/06/12/18 UTC and optionally sub-sample further."""
    hours = feats["time"].dt.hour
    keep = (hours == 0) | (hours == 6) | (hours == 12) | (hours == 18)
    feats = feats.isel(time=np.flatnonzero(keep.values))
    if timestep_subsample > 1:
        n_t = feats.sizes["time"]
        idx = np.sort(rng.choice(n_t, size=n_t // timestep_subsample, replace=False))
        feats = feats.isel(time=idx)
    return feats


def _to_long_dataframe(loaded: xr.Dataset, lat_name: str, lon_name: str) -> pd.DataFrame:
    """Flatten the (time, lat, lon) cube into long-form rows. NaN rows from
    MetPy edge effects (Coriolis-at-equator etc.) are dropped at the boundary."""
    df = loaded.to_dataframe().reset_index()
    df = df.dropna(subset=[*_FEATURE_COLS, _TARGET_COL])

    # Time-of-year cyclic encoding. Use day-of-year on the actual timestamp.
    doy = df["time"].dt.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype("float32")
    df["lat_deg"] = df[lat_name].astype("float32")

    return df


def _stratified_subsample(
    df: pd.DataFrame, n_neg_per_pos: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Keep every positive row; draw ``n_neg_per_pos × n_pos`` negatives at random."""
    is_pos = df[_MASK_COL].to_numpy()
    pos = df[is_pos]
    neg = df[~is_pos]
    n_neg = min(len(neg), len(pos) * n_neg_per_pos)
    if n_neg == 0:
        return pos
    idx = rng.choice(len(neg), size=n_neg, replace=False)
    return pd.concat([pos, neg.iloc[idx]], ignore_index=True)


# =============================================================================
# Public entry point
# =============================================================================


def build_table(
    period: tuple[str, str] = config.TRAIN_PERIOD,
    n_neg_per_pos: int = 3,
    timestep_subsample: int = 4,
    holdout: bool = False,
    overwrite: bool = False,
) -> Path:
    """Assemble the per-pixel Parquet table for Stage 1.

    Returns the partitioned root directory.
    """
    out_dir = config.CACHE_DIR / ("holdout_rows" if holdout else "train_rows")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.RANDOM_SEED)
    feats_all, clim = _lazy_features_and_climatology()

    lat_name = "latitude" if "latitude" in feats_all.coords else "lat"
    lon_name = "longitude" if "longitude" in feats_all.coords else "lon"

    start_year = int(period[0][:4])
    end_year = int(period[1][:4])

    for year in range(start_year, end_year + 1):
        out_path = out_dir / f"year={year}" / "part.parquet"
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        f_year = feats_all.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        f_year = _select_6h_subsample(
            f_year, timestep_subsample=1 if holdout else timestep_subsample, rng=rng
        )

        mask = ar_detection.compute_ar_mask(f_year["ivt"], f_year["ivt_u"], f_year["ivt_v"], clim)
        target = (mask.astype("float32") * f_year["ivt"]).rename(_TARGET_COL)

        merged = xr.merge([f_year, mask.rename(_MASK_COL), target])
        loaded = merged.compute()
        df = _to_long_dataframe(loaded, lat_name, lon_name)

        if not holdout:
            df = _stratified_subsample(df, n_neg_per_pos, rng)

        cols = ["time", lat_name, lon_name, *_FEATURE_COLS, _MASK_COL, _TARGET_COL]
        df = df[cols]
        df.to_parquet(out_path, index=False, compression="zstd")

    return out_dir


# =============================================================================
# CLI
# =============================================================================


def _parse_period(s: str) -> tuple[str, str]:
    """``"1980-2014"`` -> ``("1980-01-01", "2014-12-31")``. Also accepts a
    single year ``"1998"``."""
    if "-" in s and len(s) > 4:
        start, end = s.split("-", 1)
        return (f"{start}-01-01", f"{end}-12-31")
    return (f"{s}-01-01", f"{s}-12-31")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period",
        type=str,
        default="1980-2014",
        help="Year range (e.g. 1980-2014) or single year (e.g. 1998).",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Build the held-out evaluation table (no subsampling).",
    )
    parser.add_argument("--n-neg-per-pos", type=int, default=3)
    parser.add_argument(
        "--timestep-subsample",
        type=int,
        default=4,
        help="Keep every Nth 6-hourly timestep in training mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = build_table(
        period=_parse_period(args.period),
        n_neg_per_pos=args.n_neg_per_pos,
        timestep_subsample=args.timestep_subsample,
        holdout=args.holdout,
        overwrite=args.overwrite,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
