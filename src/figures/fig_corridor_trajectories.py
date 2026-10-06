"""Continuous 1940-2024 corridor-IVT trajectories for all eight corridors.

Shows that the intensification is not an artefact of the two-window contrast: in
every corridor the annual-mean column IVT rises gradually across the full record.
Small multiples, each with its Theil-Sen trend and Mann-Kendall significance,
built from the global ERA5 monthly fields (windows + gap decades) already on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_FILES = ["presat_monthly_global.nc", "gap1_1960_1979_monthly_global.nc",
          "modern_monthly_global.nc", "gap2_2000_2014_monthly_global.nc",
          "recent_monthly_global.nc"]


def _corridor_annual(ds, box):
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
    w = np.cos(np.deg2rad(sub["latitude"]))
    cm = ivt.weighted(w).mean(("latitude", "longitude"))
    return cm.groupby(f"{tdim}.year").mean(tdim)


def _series_all():
    dss = [xr.open_dataset(_DIR / f) for f in _FILES if (_DIR / f).exists()]
    out = {}
    for name, *box in _REGIONS:
        parts = [_corridor_annual(ds, box) for ds in dss]
        ann = xr.concat(parts, dim="year").sortby("year")
        out[name] = pd.Series(ann.to_numpy(), index=ann["year"].to_numpy().astype(int))
    return out


def _corridor_trends(series):
    """Per-corridor Theil-Sen %/decade with Hamed-Rao autocorrelation-corrected
    Mann-Kendall p (and the uncorrected p) -- the honest trend table."""
    from scipy.stats import theilslopes
    from src.analysis.trend_stats import hamed_rao_mk
    rows = []
    for name, *_ in _REGIONS:
        s = series[name]; yy = s.index.to_numpy().astype(float); vv = s.to_numpy()
        sl, b, _, _ = theilslopes(vv, yy)
        tr = hamed_rao_mk(vv, yy)
        rows.append(dict(region=name, pct_dec=round(100 * sl * 10 / vv.mean(), 2),
                         p_hr=round(tr.p, 4), p_plain=round(tr.p_plain, 4),
                         n_eff_ratio=round(tr.n_eff_ratio, 2),
                         slope=sl, intercept=b))
    return rows


def plot(out_path: Path) -> Path:
    import json
    import matplotlib.pyplot as plt
    from src import config
    apply_nature_style()
    series = _series_all()
    trends = {r["region"]: r for r in _corridor_trends(series)}

    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.5), sharex=True)
    order = [r[0] for r in _REGIONS]
    for ax, name in zip(axes.flat, order):
        s = series[name]; yy = s.index.to_numpy(); vv = s.to_numpy()
        tr = trends[name]; sl, b, p = tr["slope"], tr["intercept"], tr["p_hr"]
        pct = tr["pct_dec"]
        ax.plot(yy, vv, color="#5a6b7b", lw=0.5)
        ax.plot(yy, pd.Series(vv, index=yy).rolling(7, center=True, min_periods=3).mean().to_numpy(),
                color="#2c3e50", lw=1.0)
        ax.plot(yy, b + sl * yy, color="#c0392b", lw=1.2, ls="--")
        sig = "*" if p < 0.05 else ("†" if p < 0.10 else "")
        ax.set_title(f"{name}  {pct:+.0f}%/dec{sig}", fontsize=7)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("year", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=7.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    # Honest trend table: HR-corrected vs plain MK (asterisks use HR-corrected p).
    tbl = [{k: v for k, v in r.items() if k not in ("slope", "intercept")}
           for r in _corridor_trends(series)]
    (config.CACHE_DIR / "corridor_trends.json").write_text(json.dumps(tbl, indent=2))
    print(f"wrote {out_path}")
    for r in tbl:
        print(f"  {r['region']:18s} {r['pct_dec']:+5.1f}%/dec  p_HR={r['p_hr']:.3f}  "
              f"p_plain={r['p_plain']:.3f}  n/n*={r['n_eff_ratio']:.2f}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_corridor_trajectories.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
