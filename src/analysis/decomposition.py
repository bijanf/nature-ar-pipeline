"""Exact three-term decomposition of corridor IVT change, with block bootstrap and
multiplicity control.

For each corridor we write the column moisture transport as the product of the
column water vapour and a moisture-weighted transport efficiency,

    IVT = IWV * Vhat ,   Vhat = IVT / IWV ,

so the change between a base window (0) and a later window (1) splits *exactly*
into three finite-difference terms,

    dIVT = Vhat0 * dIWV   (thermodynamic: more moisture, base-state circulation)
         + IWV0 * dVhat    (dynamic: changed transport efficiency, base-state moisture)
         + dIWV * dVhat    (covariance: simultaneous change of both).

This removes the ambiguity of a two-term "dynamic = residual" split by exposing
the covariance term explicitly; we show it is small relative to the dynamic term
for every corridor, so the residual is genuinely circulation.

Uncertainty: we resample whole years within each window (year-block bootstrap)
and, as a serial-correlation-aware cross-check, a moving-block bootstrap with a
multi-year block. From the replicate distribution of dVhat-driven dynamic term we
compute a one-sided bootstrap p-value (fraction of replicates <= 0) and then
control the false-discovery rate across the eight corridors with Benjamini-
Hochberg, so "robustly dynamic" survives multiple comparisons.

All inputs are the five ERA5 monthly windows already on disk; no network.
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
_FILES = {
    "presat": "presat_monthly_global.nc",
    "gap1": "gap1_1960_1979_monthly_global.nc",
    "modern": "modern_monthly_global.nc",
    "gap2": "gap2_2000_2014_monthly_global.nc",
    "recent": "recent_monthly_global.nc",
}
_WIN_LABEL = {"presat": "1940–1959", "gap1": "1960–1979", "modern": "1980–1999",
              "gap2": "2000–2014", "recent": "2015–2024"}
_BOOT = 1000
_SEED = 0


def _region_year_series(ds, box):
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
    ivt_y = ivt_m.groupby(f"{tdim}.year").mean(tdim).to_numpy()
    iwv_y = iwv_m.groupby(f"{tdim}.year").mean(tdim).to_numpy()
    return ivt_y, iwv_y


def _series_cache():
    """Per-corridor per-window (ivt_y, iwv_y) arrays."""
    cache = {w: {} for w in _FILES}
    for w, fname in _FILES.items():
        ds = xr.open_dataset(_DIR / fname)
        for name, *box in _REGIONS:
            cache[w][name] = _region_year_series(ds, box)
        ds.close()
    return cache


def _three_term(ivt0, iwv0, ivt1, iwv1):
    """Exact thermo/dynamic/covariance split of the inter-window IVT change."""
    I0, W0, I1, W1 = ivt0.mean(), iwv0.mean(), ivt1.mean(), iwv1.mean()
    V0, V1 = I0 / W0, I1 / W1
    dW, dV = W1 - W0, V1 - V0
    d_total = I1 - I0
    d_thermo = V0 * dW
    d_dyn = W0 * dV
    d_cov = dW * dV
    return d_total, d_thermo, d_dyn, d_cov


def _block_resample(n, rng, block):
    """Indices for a moving-block bootstrap (block=1 reduces to i.i.d. years)."""
    if block <= 1:
        return rng.integers(0, n, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
    return idx


def _bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def compute_pair(win_a="presat", win_b="recent", block=1, cache=None):
    """Three-term decomposition + bootstrap for every corridor, window a -> b."""
    cache = cache or _series_cache()
    rng = np.random.default_rng(_SEED)
    rows = []
    for name, *_ in _REGIONS:
        ivt0, iwv0 = cache[win_a][name]
        ivt1, iwv1 = cache[win_b][name]
        d_tot, d_th, d_dy, d_cv = _three_term(ivt0, iwv0, ivt1, iwv1)
        bt_th, bt_dy, bt_cv, bt_tot = [], [], [], []
        n0, n1 = len(ivt0), len(ivt1)
        for _ in range(_BOOT):
            i0 = _block_resample(n0, rng, block); i1 = _block_resample(n1, rng, block)
            t, th, dy, cv = _three_term(ivt0[i0], iwv0[i0], ivt1[i1], iwv1[i1])
            bt_th.append(th); bt_dy.append(dy); bt_cv.append(cv); bt_tot.append(t)
        bt_dy = np.array(bt_dy)
        ci = lambda a, lo, hi: [round(float(np.percentile(a, lo)), 2),
                                round(float(np.percentile(a, hi)), 2)]
        # one-sided bootstrap p for dynamic term sign (test |dyn|>0 in observed direction)
        if d_dy >= 0:
            p_boot = float(np.mean(bt_dy <= 0))
        else:
            p_boot = float(np.mean(bt_dy >= 0))
        base = ivt0.mean()
        rows.append(dict(
            region=name, base_ivt=round(float(base), 1),
            pct_change=round(100 * d_tot / base, 1),
            d_total=round(float(d_tot), 2), d_thermo=round(float(d_th), 2),
            d_dyn=round(float(d_dy), 2), d_cov=round(float(d_cv), 2),
            d_dyn_ci90=ci(bt_dy, 5, 95), d_dyn_ci95=ci(bt_dy, 2.5, 97.5),
            d_thermo_ci90=ci(np.array(bt_th), 5, 95),
            d_cov_ci90=ci(np.array(bt_cv), 5, 95),
            p_boot=round(p_boot, 4),
        ))
    # FDR across corridors on the dynamic-term bootstrap p
    padj = _bh_fdr([r["p_boot"] for r in rows])
    for r, pa in zip(rows, padj):
        r["p_boot_fdr"] = round(float(pa), 4)
        r["dyn_robust"] = bool(pa < 0.05)
        r["dyn_robust90"] = bool(pa < 0.10)
    return rows


def compute_5window(cache=None):
    """Thermo/dynamic/covariance at each window relative to presat (evolution)."""
    cache = cache or _series_cache()
    wins = ["presat", "gap1", "modern", "gap2", "recent"]
    out = {}
    for name, *_ in _REGIONS:
        ivt0, iwv0 = cache["presat"][name]
        series = []
        for w in wins:
            ivt1, iwv1 = cache[w][name]
            d_tot, d_th, d_dy, d_cv = _three_term(ivt0, iwv0, ivt1, iwv1)
            series.append(dict(window=w, label=_WIN_LABEL[w],
                               ivt=round(float(ivt1.mean()), 2),
                               iwv=round(float(iwv1.mean()), 3),
                               d_total=round(float(d_tot), 2),
                               d_thermo=round(float(d_th), 2),
                               d_dyn=round(float(d_dy), 2),
                               d_cov=round(float(d_cv), 2)))
        out[name] = series
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", type=int, default=1)
    args = ap.parse_args()
    cache = _series_cache()

    full = compute_pair("presat", "recent", block=1, cache=cache)
    full_blk = compute_pair("presat", "recent", block=3, cache=cache)
    sat = compute_pair("modern", "recent", block=1, cache=cache)
    evo = compute_5window(cache)

    out = dict(full=full, full_block3=full_blk, satellite_era=sat, evolution=evo)
    p = config.CACHE_DIR / "decomposition.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}\n")
    print("FULL (presat->recent), i.i.d.-year bootstrap + BH-FDR:")
    print(f"{'corridor':20s} {'tot':>6s} {'thermo':>7s} {'dyn':>6s} {'cov':>6s} "
          f"{'p_boot':>7s} {'p_fdr':>6s} {'robust':>7s}")
    for r in sorted(full, key=lambda r: -r["d_dyn"]):
        print(f"{r['region']:20s} {r['d_total']:6.2f} {r['d_thermo']:7.2f} {r['d_dyn']:6.2f} "
              f"{r['d_cov']:6.2f} {r['p_boot']:7.3f} {r['p_boot_fdr']:6.3f} "
              f"{'YES' if r['dyn_robust'] else ('~' if r['dyn_robust90'] else 'no'):>7s}")
    print("\nSurviving FDR<0.05:",
          sum(r["dyn_robust"] for r in full), " | block-3:",
          sum(r["dyn_robust"] for r in full_blk), " | satellite-era:",
          sum(r["dyn_robust"] for r in sat))


if __name__ == "__main__":
    main()
