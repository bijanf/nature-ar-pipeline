"""Mode attribution of the dynamic (circulation) term — does a data-driven set of
large-scale circulation modes explain each corridor's dynamic intensification?

The two-window decomposition (``fig_global_ar_regions``) isolates a dynamic term
as the residual of the IVT change after removing the moisture (thermodynamic)
scaling. Here we test what that residual *is*, continuously over 1940-2024, with
an unsupervised + supervised machine-learning pipeline:

1.  We build, for every corridor, a continuous ANNUAL series of the moisture-
    weighted transport efficiency  Vhat(t) = IVT(t) / IWV(t).  Holding moisture
    fixed, an anomaly in Vhat is exactly the per-year analogue of the dynamic
    term: it is the circulation's contribution to transport, with the column-
    moisture amount divided out.

2.  We extract data-driven modes of the large-scale circulation as the leading
    EOFs (PCA / SVD) of the global annual-mean 500 hPa geopotential-height
    anomaly field (cos-latitude weighted).  These principal components are an
    objective, rotation-free basis for the interannual circulation — no hand
    chosen teleconnection station pairs.

3.  For each corridor we regress its Vhat anomaly on the leading PCs with both a
    ridge (linear, L2) and a random-forest (non-linear) model, scored by
    leave-one-block-out cross-validation, and rank the modes by permutation
    importance.  A corridor whose dynamic term is genuinely circulation-organised
    is *predictable* from the modes (positive CV skill) and loads on a small
    number of them; a corridor whose dynamic term is bootstrap-indistinguishable
    from zero (e.g. the US West Coast) should be poorly predictable.

The point is mechanism: the dynamic intensification is not numerical noise in a
residual but a projection onto recognised, secularly-trending modes of the
general circulation.  Everything here runs from the five ERA5 monthly windows
already on disk; no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
# Time-ordered windows that together tile 1940-2024 with no gap.
_FILES = [
    "presat_monthly_global.nc",
    "gap1_1960_1979_monthly_global.nc",
    "modern_monthly_global.nc",
    "gap2_2000_2014_monthly_global.nc",
    "recent_monthly_global.nc",
]
_N_PC = 6          # leading EOFs retained as circulation modes
_SEED = 0


# --------------------------------------------------------------------------- #
# Corridor transport efficiency Vhat(t) = IVT(t) / IWV(t)
# --------------------------------------------------------------------------- #
def _corridor_annual(ds, box):
    """Per-year area-mean IVT and IWV over the corridor box (one window)."""
    lon0, lon1, lat0, lat1 = box
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    sub = ds.sel(longitude=slice(lon0, lon1))
    sub = sub.sel(latitude=slice(lat1, lat0)) if float(ds.latitude[0]) > float(ds.latitude[-1]) \
        else sub.sel(latitude=slice(lat0, lat1))
    q = sub["q"].sortby(lev); u = sub["u"].sortby(lev); v = sub["v"].sortby(lev)
    pa = q[lev].astype("float64") * 100.0
    q = q.assign_coords({lev: pa}); u = u.assign_coords({lev: pa}); v = v.assign_coords({lev: pa})
    ivt = np.hypot((q * u).integrate(lev) / _G, (q * v).integrate(lev) / _G)
    iwv = q.integrate(lev) / _G
    w = np.cos(np.deg2rad(sub["latitude"]))
    ivt_m = ivt.weighted(w).mean(("latitude", "longitude"))
    iwv_m = iwv.weighted(w).mean(("latitude", "longitude"))
    years = ivt_m[tdim].dt.year
    ivt_y = ivt_m.groupby(years).mean(tdim)
    iwv_y = iwv_m.groupby(years).mean(tdim)
    yr = ivt_y["year"].to_numpy()
    return yr, ivt_y.to_numpy(), iwv_y.to_numpy()


def corridor_records():
    """Continuous 1940-2024 annual IVT, IWV, Vhat for every corridor."""
    out = {name: dict(year=[], ivt=[], iwv=[]) for name, *_ in _REGIONS}
    for f in _FILES:
        ds = xr.open_dataset(_DIR / f)
        for name, *box in _REGIONS:
            yr, ivt, iwv = _corridor_annual(ds, box)
            out[name]["year"].append(yr)
            out[name]["ivt"].append(ivt)
            out[name]["iwv"].append(iwv)
        ds.close()
    recs = {}
    for name, d in out.items():
        yr = np.concatenate(d["year"]); ivt = np.concatenate(d["ivt"]); iwv = np.concatenate(d["iwv"])
        order = np.argsort(yr)
        yr, ivt, iwv = yr[order], ivt[order], iwv[order]
        vhat = ivt / iwv
        recs[name] = dict(year=yr, ivt=ivt, iwv=iwv, vhat=vhat)
    return recs


# --------------------------------------------------------------------------- #
# Circulation modes: EOFs of the global annual-mean z500 anomaly field
# --------------------------------------------------------------------------- #
def circulation_modes(n_pc: int = _N_PC):
    """Leading EOFs/PCs of the cos-lat-weighted global annual z500 anomaly."""
    fields = []
    years = []
    lat = lon = None
    for f in _FILES:
        ds = xr.open_dataset(_DIR / f)
        lev = "pressure_level" if "pressure_level" in ds.dims else "level"
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        z = ds["z"].sel({lev: 500}) / _G
        zy = z.groupby(z[tdim].dt.year).mean(tdim)
        if lat is None:
            lat = ds["latitude"].to_numpy(); lon = ds["longitude"].to_numpy()
        fields.append(zy.transpose("year", "latitude", "longitude").to_numpy())
        years.append(zy["year"].to_numpy())
        ds.close()
    Z = np.concatenate(fields, axis=0)               # (T, ny, nx)
    yr = np.concatenate(years)
    order = np.argsort(yr); Z = Z[order]; yr = yr[order]
    T, ny, nx = Z.shape
    Zanom = Z - Z.mean(axis=0, keepdims=True)         # remove time mean
    wlat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0, None))   # area weight for EOF
    Zw = Zanom * wlat[None, :, None]
    X = Zw.reshape(T, ny * nx)
    # SVD of the (time x space) anomaly matrix -> PCs and EOFs.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :n_pc] * S[:n_pc]                       # (T, n_pc) principal components
    eofs = Vt[:n_pc].reshape(n_pc, ny, nx) / wlat[None, :, None]
    varexp = (S**2 / np.sum(S**2))[:n_pc]
    # Standardise PCs (zero mean already; unit variance for regression).
    pcs_std = (pcs - pcs.mean(0)) / pcs.std(0)
    return dict(year=yr, pcs=pcs_std, eofs=eofs, varexp=varexp, lat=lat, lon=lon)


# --------------------------------------------------------------------------- #
# Supervised attribution: ridge + random forest, block CV, permutation importance
# --------------------------------------------------------------------------- #
def _block_cv_r2(model_factory, X, y, n_blocks=5):
    """Leave-one-contiguous-block-out CV R^2 (blocks respect serial structure)."""
    from sklearn.metrics import r2_score
    n = len(y)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    yhat = np.full(n, np.nan)
    for b in range(n_blocks):
        te = np.zeros(n, bool); te[edges[b]:edges[b + 1]] = True
        tr = ~te
        if tr.sum() < 3 or te.sum() < 1:
            continue
        m = model_factory()
        m.fit(X[tr], y[tr])
        yhat[te] = m.predict(X[te])
    ok = ~np.isnan(yhat)
    return float(r2_score(y[ok], yhat[ok])) if ok.sum() > 3 else float("nan")


def _perm_importance(model, X, y, rng, n_rep=20):
    from sklearn.metrics import r2_score
    base = r2_score(y, model.predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - r2_score(y, model.predict(Xp)))
        imp[j] = np.mean(drops)
    return imp


def attribute(n_pc: int = _N_PC):
    """For each corridor, predict its Vhat anomaly from the circulation PCs."""
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(_SEED)
    modes = circulation_modes(n_pc)
    recs = corridor_records()
    myr, X = modes["year"], modes["pcs"]

    rows = []
    for name, *_ in _REGIONS:
        r = recs[name]
        # align years (should be identical 1940-2024)
        _, ia, ib = np.intersect1d(r["year"], myr, return_indices=True)
        vhat = r["vhat"][ia]
        y = (vhat - vhat.mean()) / vhat.std()         # standardized dynamic index
        Xc = X[ib]

        ridge_factory = lambda: RidgeCV(alphas=np.logspace(-2, 3, 20))
        rf_factory = lambda: RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=_SEED, n_jobs=1)

        r2_ridge = _block_cv_r2(ridge_factory, Xc, y)
        r2_rf = _block_cv_r2(rf_factory, Xc, y)

        ridge = ridge_factory(); ridge.fit(Xc, y)
        rf = rf_factory(); rf.fit(Xc, y)
        imp = _perm_importance(ridge, Xc, y, rng)
        coefs = ridge.coef_.tolist()

        # Continuous dynamic signal: Hamed-Rao-corrected trend in Vhat over 1940-2024
        from src.analysis.trend_stats import hamed_rao_mk
        tr = hamed_rao_mk(r["vhat"], r["year"].astype(float))
        vhat_pct_dec = 100 * tr.slope * 10 / r["vhat"].mean()
        ivt_tr = hamed_rao_mk(r["ivt"], r["year"].astype(float))

        rows.append(dict(
            region=name,
            r2_ridge=round(r2_ridge, 3),
            r2_rf=round(r2_rf, 3),
            ridge_coef=[round(c, 3) for c in coefs],
            perm_importance=[round(float(v), 4) for v in imp],
            top_mode=int(np.argmax(imp)) + 1,
            vhat_trend_pct_dec=round(vhat_pct_dec, 2),
            vhat_p=round(tr.p, 4),
            vhat_p_plain=round(tr.p_plain, 4),
            vhat_neff_ratio=round(tr.n_eff_ratio, 2),
            ivt_trend_pct_dec=round(100 * ivt_tr.slope * 10 / r["ivt"].mean(), 2),
            ivt_p=round(ivt_tr.p, 4),
        ))

    result = dict(
        n_pc=n_pc,
        varexp=[round(float(v), 4) for v in modes["varexp"]],
        years=[int(y) for y in myr],
        pcs=modes["pcs"].round(4).tolist(),
        corridors=rows,
    )
    return result, modes, recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pc", type=int, default=_N_PC)
    args = ap.parse_args()
    result, _, _ = attribute(args.n_pc)
    out = config.CACHE_DIR / "mode_attribution.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    print(f"variance explained by leading {args.n_pc} modes: "
          + ", ".join(f"{100*v:.1f}%" for v in result["varexp"]))
    print(f"{'corridor':20s} {'R2_ridge':>9s} {'R2_rf':>7s} {'topmode':>8s} "
          f"{'Vhat%/dec':>10s} {'p_HR':>7s} {'p_plain':>8s} {'n/n*':>6s}")
    for r in result["corridors"]:
        print(f"{r['region']:20s} {r['r2_ridge']:9.3f} {r['r2_rf']:7.3f} "
              f"{r['top_mode']:8d} {r['vhat_trend_pct_dec']:10.2f} {r['vhat_p']:7.3f} "
              f"{r['vhat_p_plain']:8.3f} {r['vhat_neff_ratio']:6.2f}")


if __name__ == "__main__":
    main()
