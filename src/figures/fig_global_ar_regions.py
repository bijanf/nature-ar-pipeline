"""Global atmospheric-river corridors — is the observed intensification
thermodynamic or dynamic, and does the partition vary by region?

For each major moisture-transport corridor we take the ERA5 monthly fields
(already on disk), build per-YEAR area-mean column moisture transport (IVT) and
column water vapour (IWV), and decompose the recent-minus-pre-satellite change
into a thermodynamic term (more moisture, fixed circulation) and a dynamic term
(changed circulation). A YEAR-BLOCK BOOTSTRAP (resample whole years within each
window) puts a 90% CI on each term, so we report which corridors are robustly
dynamically driven rather than a fragile point share.

We plot the ABSOLUTE thermodynamic and dynamic changes (kg m^-1 s^-1) with their
CIs — robust where percentage shares explode (when the two terms cancel). The
mean-flow (monthly) limitation applies uniformly, so the inter-region comparison
is internally consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_BOOT = 1000

_REGIONS = [
    ("US West Coast",      210, 245, 32, 52),
    ("SE South America",   295, 315, -35, -20),
    ("Amazon outflow",     285, 307, -15, 2),
    ("Western Europe",     340, 360, 44, 58),
    ("East Asia",          125, 148, 27, 43),
    ("South Africa",        15,  35, -36, -24),
    ("SE Australia / NZ",  148, 178, -45, -33),
    ("SE US / Gulf",       263, 285, 27, 38),
]


def _region_year_series(ds, box):
    """Per-year area-mean (IVT, IWV) over the region box."""
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


def _decomp(ivt_p, iwv_p, ivt_r, iwv_r):
    i0, w0, i1, w1 = ivt_p.mean(), iwv_p.mean(), ivt_r.mean(), iwv_r.mean()
    d_total = i1 - i0
    d_thermo = i0 * (w1 / w0 - 1.0)
    return d_total, d_thermo, d_total - d_thermo


def compute():
    dp = xr.open_dataset(_DIR / "presat_monthly_global.nc")
    dr = xr.open_dataset(_DIR / "recent_monthly_global.nc")
    rng = np.random.default_rng(0)
    rows = []
    for name, *box in _REGIONS:
        ivt_p, iwv_p = _region_year_series(dp, box)
        ivt_r, iwv_r = _region_year_series(dr, box)
        d_tot, d_th, d_dy = _decomp(ivt_p, iwv_p, ivt_r, iwv_r)
        # year-block bootstrap
        bt_th, bt_dy, bt_tot = [], [], []
        np_, nr = len(ivt_p), len(ivt_r)
        for _ in range(_BOOT):
            ip = rng.integers(0, np_, np_); ir = rng.integers(0, nr, nr)
            t, th, dy = _decomp(ivt_p[ip], iwv_p[ip], ivt_r[ir], iwv_r[ir])
            bt_th.append(th); bt_dy.append(dy); bt_tot.append(t)
        ci = lambda a: (float(np.percentile(a, 5)), float(np.percentile(a, 95)))
        base = ivt_p.mean()
        rows.append(dict(
            region=name, base_ivt=round(float(base), 1),
            pct_change=round(100 * d_tot / base, 1),
            d_total=round(float(d_tot), 2), d_thermo=round(float(d_th), 2), d_dyn=round(float(d_dy), 2),
            d_thermo_ci=[round(x, 2) for x in ci(bt_th)],
            d_dyn_ci=[round(x, 2) for x in ci(bt_dy)],
            dyn_robust=bool(ci(bt_dy)[0] > 0 or ci(bt_dy)[1] < 0),   # CI excludes 0
            thermo_share=round(100 * d_th / d_tot) if d_tot else None,
            dyn_share=round(100 * d_dy / d_tot) if d_tot else None,
        ))
    return rows


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from src.analysis.decomposition import compute_pair
    apply_nature_style()
    # Rigorous three-term decomposition with FDR-controlled bootstrap p-values.
    rows = compute_pair("presat", "recent", block=1)
    rr = sorted(rows, key=lambda r: r["d_dyn"])
    names = [r["region"] for r in rr]; yy = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.52))
    th = np.array([r["d_thermo"] for r in rr]); dy = np.array([r["d_dyn"] for r in rr])
    cv = np.array([r["d_cov"] for r in rr])
    dy_err = np.array([[r["d_dyn"] - r["d_dyn_ci95"][0], r["d_dyn_ci95"][1] - r["d_dyn"]] for r in rr]).T
    th_err = np.array([[r["d_thermo"] - r["d_thermo_ci90"][0], r["d_thermo_ci90"][1] - r["d_thermo"]] for r in rr]).T
    ax.barh(yy - 0.26, th, height=0.26, color="#e67e22", label="thermodynamic")
    ax.errorbar(th, yy - 0.26, xerr=th_err, fmt="none", ecolor="#7a4410", lw=0.6, capsize=1.5)
    ax.barh(yy, dy, height=0.26, color="#2980b9", label="dynamic")
    ax.errorbar(dy, yy, xerr=dy_err, fmt="none", ecolor="#16415f", lw=0.6, capsize=1.5)
    ax.barh(yy + 0.26, cv, height=0.26, color="#95a5a6", label="covariance")
    for i, r in enumerate(rr):    # mark FDR-robust dynamic terms
        mark = "*" if r["dyn_robust"] else ("†" if r["dyn_robust90"] else "")
        if mark:
            ax.text(max(r["d_dyn_ci95"][1], 0) + 0.3, i, mark, va="center",
                    fontsize=10, color="#16415f")
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("contribution to IVT change (kg m$^{-1}$ s$^{-1}$)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    (config.CACHE_DIR / "global_ar_regions.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_path}")
    for r in sorted(rows, key=lambda r: -r["d_dyn"]):
        flag = "DYN-robust(FDR<.05)" if r["dyn_robust"] else (
            "FDR<.10" if r["dyn_robust90"] else "ns")
        print(f"  {r['region']:18s} +{r['pct_change']:5.1f}%  thermo {r['d_thermo']:+5.2f} "
              f"dyn {r['d_dyn']:+5.2f} cov {r['d_cov']:+5.2f} "
              f"p_fdr={r['p_boot_fdr']:.3f} {flag}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_global_ar_regions.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
