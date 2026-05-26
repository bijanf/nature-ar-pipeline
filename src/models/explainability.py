"""
SHAP attribution: which atmospheric process drives the projected AR-intensity
shift under SSP2-4.5 / SSP3-7.0 / SSP4-6.0 / SSP5-8.5?

We use :class:`shap.TreeExplainer` on the Stage 1 LightGBM boosters and sum
the feature-level SHAP values into two physically motivated buckets:

* **Thermodynamic** — features dominated by low-level moisture and its
  scaling with temperature:
  ``theta_e_850`` (moist static energy at 850 hPa) and ``ivt`` (column
  vapor-transport magnitude, which by Clausius-Clapeyron rises ≈ 7 % per K).

* **Dynamic** — features describing the upper-level synoptic forcing and
  the kinematic structure of the AR axis:
  ``pv_250`` (Ertel PV at 250 hPa), ``eady_growth_rate`` (baroclinic
  instability proxy), and the zonal / meridional IVT components
  ``ivt_u`` / ``ivt_v``.

``lat_deg``, ``doy_sin``, ``doy_cos`` are auxiliary spatiotemporal context;
their SHAP contributions are reported under an ``other`` bucket but kept
separate from the physics buckets in the headline figure.

The aggregation reports per-landfall-band (25-35 N, 35-45 N, 45-60 N) means
of the absolute SHAP contribution per bucket. That partition matches the
geographic regions used in the manuscript (south, central, north US-West-Coast
AR landfall zones) and lets us answer the manuscript's central question:

    "Is the future change in AR intensity driven by thermodynamic moisture
    amplification, by dynamic shifts in the storm track, or by both?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import shap

from src import config

_THERMODYNAMIC = ("theta_e_850", "ivt")
_DYNAMIC = ("pv_250", "eady_growth_rate", "ivt_u", "ivt_v")
_OTHER = ("lat_deg", "doy_sin", "doy_cos")
_FEATURES = (*_THERMODYNAMIC, *_DYNAMIC, *_OTHER)

# Landfall-band edges (south, central, north). Inclusive on the low end.
_BAND_EDGES = (25.0, 35.0, 45.0, 60.0)
_BAND_LABELS = ("south_25_35N", "central_35_45N", "north_45_60N")


def compute_shap(booster: lgb.Booster, feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values for one Stage 1 booster on a feature DataFrame.

    Returns a DataFrame with one SHAP column per feature, same index as input.
    """
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(feature_rows[list(_FEATURES)].to_numpy(dtype="float32"))
    return pd.DataFrame(values, columns=list(_FEATURES), index=feature_rows.index)


def bucket_attribution(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Sum SHAP per row into thermodynamic / dynamic / other buckets.

    Uses mean of absolute SHAP within each bucket — the convention for
    relative-importance attribution. Signed sums (which can cancel) are kept
    out of the headline figure on purpose.
    """
    return pd.DataFrame(
        {
            "thermodynamic": shap_df[list(_THERMODYNAMIC)].abs().sum(axis=1),
            "dynamic": shap_df[list(_DYNAMIC)].abs().sum(axis=1),
            "other": shap_df[list(_OTHER)].abs().sum(axis=1),
        }
    )


def aggregate_by_landfall_band(buckets: pd.DataFrame, landfall_lat: pd.Series) -> pd.DataFrame:
    """Average bucket SHAP magnitudes within each landfall-latitude band."""
    bands = pd.cut(
        landfall_lat,
        bins=_BAND_EDGES,
        labels=list(_BAND_LABELS),
        include_lowest=True,
    )
    out = buckets.assign(band=bands).groupby("band", observed=True).mean()
    out.index.name = "landfall_band"
    return out


def write_attribution(table: pd.DataFrame, name: str = "shap_attribution") -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CACHE_DIR / f"{name}.parquet"
    table.to_parquet(out, compression="zstd")
    return out


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--booster",
        type=str,
        required=True,
        help="Path to a Stage 1 fold booster (e.g. data/cache/stage1_models/fold_0_seed42.txt).",
    )
    parser.add_argument(
        "--events",
        type=str,
        required=True,
        help=(
            "Per-event feature rows with the Stage-1 feature set + a landfall_lat "
            "column (typically the CMIP6 future events with their mean physics features)."
        ),
    )
    parser.add_argument("--name", type=str, default="shap_attribution")
    args = parser.parse_args()

    booster = lgb.Booster(model_file=args.booster)
    events = pd.read_parquet(args.events).dropna(subset=["landfall_lat"])

    shap_vals = compute_shap(booster, events)
    buckets = bucket_attribution(shap_vals)
    table = aggregate_by_landfall_band(buckets, events["landfall_lat"])
    out = write_attribution(table, name=args.name)
    print(table)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
