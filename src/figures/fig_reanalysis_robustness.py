"""Robustness — is the corridor IVT trend reproduced in an INDEPENDENT reanalysis?

The single biggest objection to an ERA5-only trend is that it could be an
ERA5-internal artefact (model + assimilation), especially in the pre-satellite
back-extension. NCEP/NCAR R1 (1948-present, 2.5deg) is produced by a different
model and a different assimilation system, so an agreeing trend cannot be an
ERA5 artefact. We compute the same corridor-mean IVT from NCEP R1 monthly
pressure-level humidity and winds and compare its Theil-Sen trend with ERA5.

NCEP R1 carries specific humidity on 8 levels to 300 hPa (more than ERA5's three
retained levels), so its IVT integral is, if anything, better resolved; the test
is on the SIGN and significance of the multidecadal trend, not absolute
magnitude (the two reanalyses are not expected to match in level).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_G = 9.80665
_NCEP = Path("/p/projects/poem/fallah/nature_ar_data/ncep_r1")
_BOX = config.BBOX  # 25-60N, 210-250E


def _ncep_corridor_ivt() -> pd.Series:
    """Annual corridor-mean |IVT| (kg m^-1 s^-1) from NCEP R1 monthly fields."""
    q = xr.open_dataset(_NCEP / "shum.mon.mean.nc")["shum"]
    u = xr.open_dataset(_NCEP / "uwnd.mon.mean.nc")["uwnd"]
    v = xr.open_dataset(_NCEP / "vwnd.mon.mean.nc")["vwnd"]
    lev = next(c for c in q.dims if "level" in c.lower())
    lat = next(c for c in q.dims if c.lower().startswith("lat"))
    lon = next(c for c in q.dims if c.lower().startswith("lon"))
    # NCEP shum is g/kg on some mirrors, kg/kg on others; normalise to kg/kg.
    if float(np.nanmax(q.isel({d: 0 for d in q.dims if d not in (lat, lon, lev)}))) > 1.0:
        q = q / 1000.0
    # corridor box (NCEP lon 0..357.5 ascending, lat 90..-90 descending)
    sel = dict()
    box = lambda da: da.sel({lon: slice(_BOX["lon_min"], _BOX["lon_max"]),
                             lat: slice(_BOX["lat_max"], _BOX["lat_min"])})
    q, u, v = box(q), box(u), box(v)
    # integrate q*V over pressure (levels descending in hPa -> ascending Pa for -dp/g)
    q = q.sortby(lev); u = u.sortby(lev); v = v.sortby(lev)
    pa = q[lev].astype("float64") * 100.0
    q = q.assign_coords({lev: pa}); u = u.assign_coords({lev: pa}); v = v.assign_coords({lev: pa})
    ivt_u = (q * u).integrate(lev) / _G
    ivt_v = (q * v).integrate(lev) / _G
    ivt = np.hypot(ivt_u, ivt_v)
    w = np.cos(np.deg2rad(ivt[lat]))
    corridor = ivt.weighted(w).mean((lat, lon))
    tname = next(c for c in corridor.dims if "time" in c.lower())
    ann = corridor.groupby(f"{tname}.year").mean(tname)
    return pd.Series(ann.to_numpy(), index=ann["year"].to_numpy().astype(int), name="ncep")


def _theil(x, y):
    from scipy.stats import theilslopes, kendalltau
    s, b, lo, hi = theilslopes(y, x)
    _, p = kendalltau(x, y)
    return s, b, p


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from src.figures.fig_timeseries import _annual_series

    apply_nature_style()
    era = _annual_series()                      # ERA5 corridor IVT, 1940-2024
    ncep = _ncep_corridor_ivt()                 # NCEP R1 corridor IVT, 1948-2024

    out = {}
    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42))
    for name, s, col in [("ERA5", era, "#c0392b"), ("NCEP R1 (independent)", ncep, "#2980b9")]:
        yy = s.index.to_numpy().astype(int); vv = s.to_numpy()
        m = np.isfinite(vv)
        yy, vv = yy[m], vv[m]
        # z-score each series so the two reanalyses overlay despite different levels
        z = (vv - vv.mean()) / vv.std()
        sl, b, p = _theil(yy, vv)
        pct_dec = 100 * sl * 10 / vv.mean()
        ax.plot(yy, z, color=col, lw=0.9, marker="o", ms=2,
                label=f"{name}: {pct_dec:+.1f}%/decade (p={p:.1e})")
        zb = ((b + sl * yy) - vv.mean()) / vv.std()
        ax.plot(yy, zb, color=col, lw=1.4, ls="--")
        out[name.split()[0].lower()] = {"pct_per_decade": round(float(pct_dec), 2),
                                        "mk_p": float(f"{p:.2e}"),
                                        "years": f"{yy.min()}-{yy.max()}"}
    ax.set_xlabel("year", fontsize=8.5)
    ax.set_ylabel("corridor IVT anomaly (z-score)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    (config.CACHE_DIR / "reanalysis_robustness.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_reanalysis_robustness.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
