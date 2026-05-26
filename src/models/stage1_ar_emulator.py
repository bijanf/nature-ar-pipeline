"""
Stage 1 — LightGBM emulator of Guan-Waliser AR intensity.

Target
------
``ar_intensity = GW_mask × IVT_ERA5``  (kg m⁻¹ s⁻¹, zero outside detected ARs).

Why this target
---------------
Predicting AR-conditional IVT (rather than a binary mask) keeps the regression
loss aligned with a continuous quantity that scales with low-level moisture and
therefore with Clausius-Clapeyron under SSP5-8.5 warming. ``linear_tree=True``
LightGBM leaves then admit linear extrapolation of intensity into climate
states the training data never visits, without forcing an out-of-support tree
descent on the categorical mask.

Spatial Block Cross-Validation
------------------------------
* 5° × 5° spatial blocks across the bbox -> 56 blocks; assigned to K = 5 folds
  stratified by per-block AR climatology so each fold has comparable base rate.
* Year-blocking: each calendar year is independently assigned to one of K
  year-folds. A row is held out for fold ``k`` iff its spatial-fold == k AND
  its year-fold == k. This is the (5 × 5) cross-product of GW (2015) blocking
  recommendations adapted for an AR-extreme regressor.
* 1° outer buffer: training rows within ``buffer_deg`` degrees of any spatial
  block belonging to the test fold are dropped, to break the synoptic-scale
  autocorrelation that would otherwise inflate validation skill.

Outputs
-------
* Per-fold ``lightgbm.Booster`` persisted to
  ``CACHE_DIR / "stage1_models" / "fold_{k}_seed{seed}.txt"``.
* JSON CV report at ``CACHE_DIR / "stage1_cv_report.json"`` with per-fold RMSE,
  R², AR-mask F1 (predicted intensity thresholded at
  ``STAGE1_EVENT_THRESHOLD``), and metadata.
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

_FEATURES = (
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
_TARGET = "ar_intensity"
_MASK = "ar_mask"

_LGBM_PARAMS: dict[str, object] = {
    "objective": "regression_l1",  # L1 is robust to the zero-inflated target
    "linear_tree": True,  # enables linear extrapolation in leaves
    "metric": "l1",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
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
# Fold assignment
# =============================================================================


@dataclass(frozen=True)
class FoldPlan:
    """Per-row fold assignments. ``spatial_fold`` and ``year_fold`` are integer
    arrays in ``[0, n_folds)``; ``block_lat`` / ``block_lon`` are the centres
    of each row's 5° × 5° spatial block, used downstream for buffer logic."""

    spatial_fold: np.ndarray
    year_fold: np.ndarray
    block_lat: np.ndarray
    block_lon: np.ndarray
    n_folds: int


def assign_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_deg: float = 5.0,
    stratify: bool = True,
    seed: int = config.RANDOM_SEED,
) -> FoldPlan:
    """Stratify rows into spatial and year folds. Spatial folds are formed by
    ranking 5° × 5° blocks by their AR climatology and round-robining the
    sorted blocks across the folds; year folds are a random permutation."""
    lat_min = config.BBOX["lat_min"]
    lon_min = config.BBOX["lon_min"]
    # Block lower-left integer index; centre = (idx + 0.5) * block_deg + min.
    bi_lat = np.floor((df["lat_deg"].to_numpy() - lat_min) / block_deg).astype("int32")
    bi_lon = np.floor((df["lon_deg"].to_numpy() - lon_min) / block_deg).astype("int32")
    block_id = bi_lat.astype("int64") * 1000 + bi_lon.astype("int64")
    block_lat = lat_min + (bi_lat + 0.5) * block_deg
    block_lon = lon_min + (bi_lon + 0.5) * block_deg

    if stratify:
        # Per-block AR climatology = fraction of rows with the mask set.
        ar_rate = pd.Series(df[_MASK].to_numpy(), index=block_id).groupby(level=0).mean()
        sorted_blocks = ar_rate.sort_values().index.to_numpy()
        block_to_fold = {b: i % n_folds for i, b in enumerate(sorted_blocks)}
    else:
        rng = np.random.default_rng(seed)
        unique_blocks = np.unique(block_id)
        perm = rng.permutation(unique_blocks)
        block_to_fold = {b: i % n_folds for i, b in enumerate(perm)}

    spatial_fold = np.array([block_to_fold[b] for b in block_id], dtype="int32")

    # Year-fold via deterministic shuffle.
    years = pd.to_datetime(df["time"]).dt.year.to_numpy()
    rng = np.random.default_rng(seed + 1)
    unique_years = np.unique(years)
    perm_years = rng.permutation(unique_years)
    year_to_fold = {int(y): i % n_folds for i, y in enumerate(perm_years)}
    year_fold = np.array([year_to_fold[int(y)] for y in years], dtype="int32")

    return FoldPlan(
        spatial_fold=spatial_fold,
        year_fold=year_fold,
        block_lat=block_lat.astype("float32"),
        block_lon=block_lon.astype("float32"),
        n_folds=n_folds,
    )


def _buffer_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    test_block_lat: np.ndarray,
    test_block_lon: np.ndarray,
    block_deg: float,
    buffer_deg: float,
) -> np.ndarray:
    """Return ``True`` for rows within ``buffer_deg`` of the test-fold blocks.

    Distance to a block is the Chebyshev distance from the row's (lat, lon) to
    the block's extent ``[centre ± block_deg/2]``. Adequate at the buffer scale
    (1° ≪ block_deg) and avoids spherical-geodesy noise.
    """
    half = block_deg / 2.0
    d_lat = np.maximum(0.0, np.abs(lat[:, None] - test_block_lat[None, :]) - half)
    d_lon = np.maximum(0.0, np.abs(lon[:, None] - test_block_lon[None, :]) - half)
    dist = np.sqrt(d_lat * d_lat + d_lon * d_lon)
    return (dist < buffer_deg).any(axis=1)


# =============================================================================
# Training
# =============================================================================


def train_fold(
    df: pd.DataFrame,
    plan: FoldPlan,
    k: int,
    buffer_deg: float = 1.0,
    block_deg: float = 5.0,
    params: dict[str, object] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, dict[str, float]]:
    """Train one fold of the spatial-block CV. Returns the booster and a metrics dict."""
    test_mask = (plan.spatial_fold == k) & (plan.year_fold == k)
    candidate_train = (plan.spatial_fold != k) & (plan.year_fold != k)

    test_block_lat = np.unique(plan.block_lat[plan.spatial_fold == k])
    test_block_lon = np.unique(plan.block_lon[plan.spatial_fold == k])
    in_buffer = _buffer_mask(
        df["lat_deg"].to_numpy(),
        df["lon_deg"].to_numpy(),
        test_block_lat,
        test_block_lon,
        block_deg=block_deg,
        buffer_deg=buffer_deg,
    )
    train_mask = candidate_train & ~in_buffer

    feature_cols = list(_FEATURES)
    x_tr = df.loc[train_mask, feature_cols].to_numpy(dtype="float32")
    y_tr = df.loc[train_mask, _TARGET].to_numpy(dtype="float32")
    x_te = df.loc[test_mask, feature_cols].to_numpy(dtype="float32")
    y_te = df.loc[test_mask, _TARGET].to_numpy(dtype="float32")
    mask_te = df.loc[test_mask, _MASK].to_numpy()

    train_set = lgb.Dataset(x_tr, label=y_tr, feature_name=feature_cols)
    valid_set = lgb.Dataset(x_te, label=y_te, feature_name=feature_cols, reference=train_set)

    booster = lgb.train(
        params=params or _LGBM_PARAMS,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    pred = booster.predict(x_te, num_iteration=booster.best_iteration)
    return booster, _fold_metrics(y_te, pred, mask_te)


def _fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask_true: np.ndarray
) -> dict[str, float]:
    """RMSE, R², and AR-mask F1 from thresholded intensity predictions."""
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual * residual)))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mask_pred = y_pred > config.STAGE1_EVENT_THRESHOLD
    tp = float(np.sum(mask_pred & mask_true))
    fp = float(np.sum(mask_pred & ~mask_true))
    fn = float(np.sum(~mask_pred & mask_true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"rmse": rmse, "r2": r2, "precision": precision, "recall": recall, "f1": f1}


def train_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_deg: float = 5.0,
    buffer_deg: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[lgb.Booster], list[dict[str, float]]]:
    """End-to-end CV: assign folds, train K boosters, return all of them plus per-fold metrics."""
    plan = assign_folds(df, n_folds=n_folds, block_deg=block_deg, seed=seed)
    boosters: list[lgb.Booster] = []
    metrics: list[dict[str, float]] = []
    for k in range(n_folds):
        b, m = train_fold(df, plan, k, buffer_deg=buffer_deg, block_deg=block_deg)
        boosters.append(b)
        metrics.append({"fold": k, **m})
    return boosters, metrics


# =============================================================================
# Persistence + CLI
# =============================================================================


def save_boosters(boosters: list[lgb.Booster], seed: int = config.RANDOM_SEED) -> Path:
    out_dir = config.CACHE_DIR / "stage1_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, b in enumerate(boosters):
        b.save_model(str(out_dir / f"fold_{k}_seed{seed}.txt"))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", action="store_true", help="Run 5-fold spatial-block CV.")
    parser.add_argument(
        "--train-rows",
        type=str,
        default=str(config.CACHE_DIR / "train_rows"),
        help="Path to the Hive-partitioned training Parquet directory.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--block-deg", type=float, default=5.0)
    parser.add_argument("--buffer-deg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    df = pd.read_parquet(args.train_rows)

    if args.cv:
        boosters, metrics = train_cv(
            df,
            n_folds=args.n_folds,
            block_deg=args.block_deg,
            buffer_deg=args.buffer_deg,
            seed=args.seed,
        )
        save_boosters(boosters, seed=args.seed)
        report_path = config.CACHE_DIR / "stage1_cv_report.json"
        report = {
            "seed": args.seed,
            "n_folds": args.n_folds,
            "block_deg": args.block_deg,
            "buffer_deg": args.buffer_deg,
            "metrics": metrics,
            "rmse_mean": float(np.mean([m["rmse"] for m in metrics])),
            "rmse_std": float(np.std([m["rmse"] for m in metrics])),
            "r2_mean": float(np.mean([m["r2"] for m in metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in metrics])),
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
