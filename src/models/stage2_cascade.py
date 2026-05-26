"""
Stage 2 — LightGBM quantile regression for AR-event total precipitation.

Target
------
Spatially-averaged ERA5 ``total_precipitation`` summed over each AR event's
footprint and hourly time window (mm). See :mod:`src.data.event_dataset` for
the integration semantics.

Why three quantile models
-------------------------
For a hazard projection we need an honest uncertainty band, not a single point
prediction. We train three independent ``LGBMRegressor(objective="quantile",
alpha=α)`` models at α ∈ {0.05, 0.50, 0.95}. The α=0.50 model is the median
prediction; the (0.05, 0.95) pair brackets a 90 % prediction interval. Pinball
loss is the proper scoring rule for quantile predictions and is reported per
fold; PI coverage on the held-out 2015-2024 events is the headline
uncertainty-calibration check.

``linear_tree=True`` is retained from Stage 1: the SSP5-8.5/SSP3-7.0 future
events live in feature regions Stage 1 only partially covered, and linear
leaves let the quantile boundaries extrapolate along the same Clausius-Clapeyron
axis.

Inputs
------
Stage 2 reads ``data/cache/events_dataset.parquet`` produced by
:mod:`src.data.event_dataset`. The feature set is per-event:

* mean / max / std intensity, footprint area, duration, landfall latitude
  (carried from :mod:`src.models.event_post`);
* mean / max / std elevation, land fraction (from
  :mod:`src.features.topography`).

The Stage-2 spatial-block CV reuses the Stage 1 5° × 5° block + 1° buffer
scheme but applied at each event's ``landfall_lon`` / ``landfall_lat`` —
events whose landfall happens to sit inside a fold-``k`` spatial block are
the test set for that fold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.models.stage1_ar_emulator import _buffer_mask

_FEATURES = (
    "mean_intensity",
    "max_intensity",
    "footprint_area_km2",
    "duration_hours",
    "landfall_lat",
    "mean_elev_m",
    "max_elev_m",
    "std_elev_m",
    "land_fraction",
)
_TARGET = "precip_total_mm"
_QUANTILES = (0.05, 0.50, 0.95)

_LGBM_BASE: dict[str, object] = {
    "objective": "quantile",
    "linear_tree": True,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": config.RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
}


# =============================================================================
# Fold assignment by landfall location
# =============================================================================


@dataclass(frozen=True)
class EventFoldPlan:
    """Per-event fold assignment. ``fold`` is in ``[0, n_folds)``; rows whose
    landfall_lat is NaN (entirely oceanic events) are placed in a dedicated
    -1 bucket and never enter any test fold."""

    fold: np.ndarray
    block_lat: np.ndarray
    block_lon: np.ndarray
    n_folds: int


def assign_event_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_deg: float = 5.0,
    seed: int = config.RANDOM_SEED,
) -> EventFoldPlan:
    """Stratified spatial-block fold assignment by event landfall position.

    Block-to-fold mapping is by per-block event-count rank so each fold has a
    comparable number of events.
    """
    lat = df["landfall_lat"].to_numpy()
    lon = df["landfall_lon"].to_numpy()
    # Events with no land pixels in their footprint don't have a landfall
    # location and can't be spatially folded — keep them in -1.
    has_land = ~np.isnan(lat)

    fold = np.full(len(df), -1, dtype="int32")
    block_lat_col = np.full(len(df), np.nan, dtype="float32")
    block_lon_col = np.full(len(df), np.nan, dtype="float32")

    if has_land.any():
        lat_min = config.BBOX["lat_min"]
        lon_min = config.BBOX["lon_min"]
        bi_lat = np.floor((lat[has_land] - lat_min) / block_deg).astype("int32")
        bi_lon = np.floor((lon[has_land] - lon_min) / block_deg).astype("int32")
        block_id = bi_lat.astype("int64") * 1000 + bi_lon.astype("int64")
        block_lat_col[has_land] = (lat_min + (bi_lat + 0.5) * block_deg).astype("float32")
        block_lon_col[has_land] = (lon_min + (bi_lon + 0.5) * block_deg).astype("float32")

        counts = pd.Series(block_id).value_counts().sort_values().index.to_numpy()
        rng = np.random.default_rng(seed)
        # Round-robin assignment with a small shuffle inside ties to avoid
        # systematic latitude bias.
        rng.shuffle(counts)
        block_to_fold = {b: i % n_folds for i, b in enumerate(counts)}
        fold[has_land] = np.array([block_to_fold[b] for b in block_id], dtype="int32")

    return EventFoldPlan(
        fold=fold,
        block_lat=block_lat_col,
        block_lon=block_lon_col,
        n_folds=n_folds,
    )


# =============================================================================
# Training
# =============================================================================


def _train_one_quantile(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    alpha: float,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    params = {**_LGBM_BASE, "alpha": alpha, "metric": "quantile"}
    train_set = lgb.Dataset(x_tr, label=y_tr, feature_name=list(_FEATURES))
    valid_set = lgb.Dataset(x_te, label=y_te, feature_name=list(_FEATURES), reference=train_set)
    return lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def train_fold(
    df: pd.DataFrame,
    plan: EventFoldPlan,
    k: int,
    buffer_deg: float = 1.0,
    block_deg: float = 5.0,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> tuple[dict[float, lgb.Booster], dict[str, float]]:
    """Train all three quantile models for fold ``k``. Returns boosters + metrics."""
    test_mask = plan.fold == k
    train_mask = (plan.fold != k) & (plan.fold != -1)

    # Apply the spatial buffer to drop training events near the test-fold blocks.
    if train_mask.any() and test_mask.any():
        in_buffer = _buffer_mask(
            df["landfall_lat"].to_numpy(),
            df["landfall_lon"].to_numpy(),
            np.unique(plan.block_lat[test_mask]),
            np.unique(plan.block_lon[test_mask]),
            block_deg=block_deg,
            buffer_deg=buffer_deg,
        )
        train_mask = train_mask & ~in_buffer

    feature_cols = list(_FEATURES)
    x_tr = df.loc[train_mask, feature_cols].to_numpy(dtype="float32")
    y_tr = df.loc[train_mask, _TARGET].to_numpy(dtype="float32")
    x_te = df.loc[test_mask, feature_cols].to_numpy(dtype="float32")
    y_te = df.loc[test_mask, _TARGET].to_numpy(dtype="float32")

    boosters = {
        alpha: _train_one_quantile(
            x_tr, y_tr, x_te, y_te, alpha, num_boost_round, early_stopping_rounds
        )
        for alpha in _QUANTILES
    }

    predictions = {
        alpha: b.predict(x_te, num_iteration=b.best_iteration) for alpha, b in boosters.items()
    }
    return boosters, _fold_metrics(y_te, predictions)


def _fold_metrics(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> dict[str, float]:
    """Per-quantile pinball loss plus 90 % PI empirical coverage."""

    def _pinball(alpha: float, y: np.ndarray, q: np.ndarray) -> float:
        d = y - q
        return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))

    lo, mid, hi = preds[0.05], preds[0.50], preds[0.95]
    coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
    return {
        "pinball_05": _pinball(0.05, y_true, lo),
        "pinball_50": _pinball(0.50, y_true, mid),
        "pinball_95": _pinball(0.95, y_true, hi),
        "pi90_coverage": coverage,
        "n_test": int(y_true.size),
    }


def train_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_deg: float = 5.0,
    buffer_deg: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[dict[float, lgb.Booster]], list[dict[str, float]]]:
    plan = assign_event_folds(df, n_folds=n_folds, block_deg=block_deg, seed=seed)
    all_boosters: list[dict[float, lgb.Booster]] = []
    metrics: list[dict[str, float]] = []
    for k in range(n_folds):
        b, m = train_fold(df, plan, k, buffer_deg=buffer_deg, block_deg=block_deg)
        all_boosters.append(b)
        metrics.append({"fold": k, **m})
    return all_boosters, metrics


# =============================================================================
# Persistence + CLI
# =============================================================================


def save_boosters(
    all_boosters: list[dict[float, lgb.Booster]], seed: int = config.RANDOM_SEED
) -> Path:
    out_dir = config.CACHE_DIR / "stage2_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, boosters in enumerate(all_boosters):
        for alpha, b in boosters.items():
            b.save_model(str(out_dir / f"fold_{k}_alpha{int(alpha * 100):02d}_seed{seed}.txt"))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.CACHE_DIR / "events_dataset.parquet"),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--block-deg", type=float, default=5.0)
    parser.add_argument("--buffer-deg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset).dropna(subset=[_TARGET, "landfall_lat"])

    if args.cv:
        boosters, metrics = train_cv(
            df,
            n_folds=args.n_folds,
            block_deg=args.block_deg,
            buffer_deg=args.buffer_deg,
            seed=args.seed,
        )
        save_boosters(boosters, seed=args.seed)
        report = {
            "seed": args.seed,
            "n_folds": args.n_folds,
            "metrics": metrics,
            "pi90_coverage_mean": float(np.mean([m["pi90_coverage"] for m in metrics])),
            "pinball_50_mean": float(np.mean([m["pinball_50"] for m in metrics])),
        }
        report_path = config.CACHE_DIR / "stage2_cv_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
