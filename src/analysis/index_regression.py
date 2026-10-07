"""Regression of the yearly wind-change contribution on six climate indices.

For every corridor, annually and for each season, the yearly wind-change
contribution D_y (from the decomposition) is regressed on a constant, a linear
time term and the standardised indices (relative Nino3.4, PDO, AMV, NAO, SAM,
PNA) averaged over the same months. Standard errors of the coefficients are
inflated for the lag-1 autocorrelation of the residuals (effective sample size
n (1 - r1) / (1 + r1), capped at n). The indices enter linearly detrended and
standardised, so that they carry their variability only and the secular change of
D_y is attributed to the time term. The part of D_y congruent with the indices is removed and
the change of the remainder between the two headline periods is reported next to
the raw change, together with simple correlations.

Seasons follow the decomposition: DJF is January, February and December of the
same calendar year.

Output: results_esd/index_regression.json
"""

from __future__ import annotations

import json

import numpy as np
import xarray as xr
from scipy import stats

from src.analysis import fullcolumn_decomp as fd
from src.figures.fig_global_ar_regions import _REGIONS

INDICES = ["rnino34", "pdo", "amv", "nao", "sam", "pna"]
SEASON_MONTHS = {"annual": list(range(1, 13)), "DJF": [1, 2, 12], "MAM": [3, 4, 5],
                 "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def index_table(ds: xr.Dataset, months: list[int]) -> dict[str, np.ndarray]:
    """Mean of each index over the given calendar months of every year 1940-2024."""
    t = ds["valid_time"]
    sel = ds.sel(valid_time=t.dt.month.isin(months))
    ann = sel.groupby("valid_time.year").mean("valid_time")
    ann = ann.sel(year=slice(int(fd.YEARS[0]), int(fd.YEARS[-1])))
    assert ann.sizes["year"] == len(fd.YEARS), ann.sizes
    return {k: ann[k].values.astype("float64") for k in INDICES}


def _yearly(label: str) -> dict:
    tag = "headline" if label == "annual" else label
    z = np.load(fd.OUT_DIR / f"yearly_{tag}.npz")
    out = {}
    for name, *_ in _REGIONS:
        k = fd.corridor_key(name)
        out[name] = {t: z[f"{k}__{t}"] for t in fd.TERMS + ("total",)}
    return out


def regress(y: np.ndarray, idx: dict[str, np.ndarray], years: np.ndarray, w0, w1) -> dict:
    names = list(idx)
    t = (years - years.mean()) / 10.0
    # indices are linearly detrended and standardised, so each carries its variability only and the
    # secular change of the wind term is attributed to the time term
    cols = []
    for nme in names:
        v = idx[nme] - np.polyval(np.polyfit(t, idx[nme], 1), t)
        cols.append(v / v.std(ddof=1))
    X = np.column_stack([np.ones_like(y), t] + cols)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    n_eff = float(np.clip(n * (1 - r1) / (1 + r1), k + 2, n))
    s2 = resid @ resid / (n_eff - k)
    cov = s2 * np.linalg.inv(X.T @ X) * (n / n_eff)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), df=n_eff - k)
    idx_part = X[:, 2:] @ beta[2:]
    m0 = (years >= w0[0]) & (years <= w0[1])
    m1 = (years >= w1[0]) & (years <= w1[1])
    resid_noidx = y - idx_part
    labels = ["const", "trend_per_decade"] + names
    return {
        "coef": {l: round(float(b), 3) for l, b in zip(labels, beta)},
        "se": {l: round(float(s), 3) for l, s in zip(labels, se)},
        "p": {l: round(float(p), 4) for l, p in zip(labels, pval)},
        "r1_resid": round(float(r1), 3), "n_eff": round(float(n_eff), 1),
        "r2_full": round(float(1 - resid.var() / y.var()), 3),
        "r2_indices": round(float(1 - (y - idx_part - beta[0]).var() / y.var()), 3),
        "corr": {nm: [round(float(stats.pearsonr(y, idx[nm])[0]), 3), round(float(stats.pearsonr(y, idx[nm])[1]), 4)]
                 for nm in names},
        "wind_diff_raw": round(float(y[m1].mean() - y[m0].mean()), 3),
        "wind_diff_index_removed": round(float(resid_noidx[m1].mean() - resid_noidx[m0].mean()), 3),
    }


def main():
    ds = xr.open_dataset(fd.OUT_DIR / "indices_era5.nc")
    out = {}
    for label, months in SEASON_MONTHS.items():
        idx = index_table(ds, months)
        ys = _yearly(label)
        out[label] = {}
        for name, series in ys.items():
            out[label][name] = {"wind": regress(series["wind"], idx, fd.YEARS, *fd.HEADLINE),
                                "moist": regress(series["moist"], idx, fd.YEARS, *fd.HEADLINE)}
    p = fd.OUT_DIR / "index_regression.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    for name in out["annual"]:
        r = out["annual"][name]["wind"]
        sig = [k for k in INDICES if r["p"][k] < 0.05]
        print(f"{name:18s} r2_idx {r['r2_indices']:5.2f} trend {r['coef']['trend_per_decade']:+6.2f} "
              f"(p {r['p']['trend_per_decade']:.3f}) sig: {sig} | diff raw {r['wind_diff_raw']:+6.2f} "
              f"idx-removed {r['wind_diff_index_removed']:+6.2f}")


if __name__ == "__main__":
    main()
