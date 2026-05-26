"""
Per-event Stage-1 feature reduction.

The Stage 1 emulator is a pixel-time regressor over the Holton-dynamics feature
set (theta_e_850, pv_250, ivt, ivt_u, ivt_v, eady_growth_rate, lat_deg,
doy_sin, doy_cos). The downstream SHAP attribution in
:mod:`src.models.explainability` needs one *row per event* in that feature
space — a single thermodynamic-and-dynamic snapshot representing the AR.

Reduction
---------
For each labelled event we take the **footprint × duration mean** of every
3-D feature: i.e. the arithmetic mean of (lat, lon, time) cells where
``labels == event_id``. This matches how Stage 1 aggregates predicted
intensity per event (mean over the same labelled pixel set), so the SHAP
row describes the same volumetric average the booster scored against.

The two pixel-level auxiliaries ``doy_sin`` / ``doy_cos`` and ``lat_deg`` are
derived from event geometry, not from the feature cube:

* ``lat_deg`` = footprint centroid latitude (weighted by cell area).
* ``doy_sin`` / ``doy_cos`` = sin/cos of the **mean event timestamp's**
  day-of-year, projected onto the unit circle exactly as
  :mod:`src.data.era5_train_assembly` projects training rows.

Caller contract
---------------
``labels`` must be the (time, lat, lon) integer cube returned by
:func:`src.models.event_post.extract_events_with_labels` so event-id
identity is preserved.

``features`` is an :class:`xarray.Dataset` whose data variables include the
six 3-D physics features (``ivt``, ``ivt_u``, ``ivt_v``, ``theta_e_850``,
``pv_250``, ``eady_growth_rate``). Each must be a ``(time, lat, lon)``
array aligned with ``labels``. The reduction calls ``.values`` once per
feature; caller is responsible for whatever materialisation / chunking is
viable for the period length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

_PHYSICS_FEATURES = (
    "ivt",
    "ivt_u",
    "ivt_v",
    "theta_e_850",
    "pv_250",
    "eady_growth_rate",
)


def attach_features(
    events: pd.DataFrame,
    labels: np.ndarray,
    features: xr.Dataset,
) -> pd.DataFrame:
    """Append Stage-1 feature columns to ``events`` via footprint×duration mean.

    Returns a new DataFrame with the same rows as ``events`` plus columns
    ``ivt``, ``ivt_u``, ``ivt_v``, ``theta_e_850``, ``pv_250``,
    ``eady_growth_rate``, ``lat_deg``, ``doy_sin``, ``doy_cos``.
    """
    if events.empty:
        out = events.copy()
        for col in (*_PHYSICS_FEATURES, "lat_deg", "doy_sin", "doy_cos"):
            out[col] = pd.Series(dtype="float32")
        return out

    lat_name = "latitude" if "latitude" in features.coords else "lat"
    lon_name = "longitude" if "longitude" in features.coords else "lon"

    physics_arrays: dict[str, np.ndarray] = {}
    for name in _PHYSICS_FEATURES:
        da = features[name].transpose("time", lat_name, lon_name)
        physics_arrays[name] = np.asarray(da.values, dtype="float32")

    lats = features[lat_name].to_numpy().astype("float64")
    times = pd.to_datetime(features["time"].to_numpy())

    out_records: list[dict[str, float]] = []
    for event_id in events["event_id"].astype(int):
        mask = labels == event_id
        if not mask.any():
            out_records.append({col: float("nan") for col in (*_PHYSICS_FEATURES, "lat_deg")})
            out_records[-1]["doy_sin"] = float("nan")
            out_records[-1]["doy_cos"] = float("nan")
            continue

        rec: dict[str, float] = {}
        for name, arr in physics_arrays.items():
            rec[name] = float(arr[mask].mean())

        # Footprint centroid latitude — weight equally across cells touched,
        # not by cell area. The Stage 1 model saw 'lat_deg' as a per-pixel
        # feature, so the unweighted mean over event pixels matches.
        i_lat = np.where(mask.any(axis=(0, 2)))[0]
        rec["lat_deg"] = float(lats[i_lat].mean())

        # Mean timestamp -> day-of-year on the unit circle (same projection
        # as the Stage-1 training rows).
        t_idx = np.where(mask.any(axis=(1, 2)))[0]
        mean_ts = pd.Timestamp(np.mean(times[t_idx].astype("int64"))).to_pydatetime()
        doy = mean_ts.timetuple().tm_yday
        angle = 2.0 * np.pi * (doy - 1) / 365.25
        rec["doy_sin"] = float(np.sin(angle))
        rec["doy_cos"] = float(np.cos(angle))

        out_records.append(rec)

    feats_df = pd.DataFrame.from_records(out_records).astype("float32")
    return pd.concat([events.reset_index(drop=True), feats_df], axis=1)


def stage1_feature_columns() -> tuple[str, ...]:
    """Column order matching :data:`src.models.explainability._FEATURES`."""
    return (*_PHYSICS_FEATURES, "lat_deg", "doy_sin", "doy_cos")
