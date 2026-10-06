"""Seasonality of the corridor IVT change.

The recent-minus-pre-satellite change in corridor-mean IVT, split by season
(DJF / MAM / JJA / SON), for each corridor. This shows whether a corridor
intensifies year-round or in a particular season — and underpins the
reconciliation of conflicting basin trends (e.g. an annual-mean increase that
coexists with a wintertime decline). Built from the global ERA5 monthly fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src import config
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
_SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
_COL = {"DJF": "#2980b9", "MAM": "#27ae60", "JJA": "#c0392b", "SON": "#e67e22"}


def _corridor_monthly_ivt(ds, box):
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
    return cm.rename({tdim: "time"}).assign_coords(time=ds[tdim].values)


def compute():
    dp = xr.open_dataset(_DIR / "presat_monthly_global.nc")
    dr = xr.open_dataset(_DIR / "recent_monthly_global.nc")
    rows = []
    for name, *box in _REGIONS:
        sp = _corridor_monthly_ivt(dp, box); sr = _corridor_monthly_ivt(dr, box)
        seas = {}
        for s, months in _SEASONS.items():
            mp = sp.sel(time=sp["time"].dt.month.isin(months)).mean().item()
            mr = sr.sel(time=sr["time"].dt.month.isin(months)).mean().item()
            seas[s] = mr - mp
        rows.append(dict(region=name, **seas))
    return rows


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    rows = compute()
    names = [r["region"] for r in rows]
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42))
    for i, s in enumerate(_SEASONS):
        ax.bar(x + (i - 1.5) * w, [r[s] for r in rows], width=w, color=_COL[s], label=s)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("seasonal ΔIVT, recent − pre-sat\n(kg m$^{-1}$ s$^{-1}$)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    (config.CACHE_DIR / "seasonality.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_path}")
    for r in rows:
        print(f"  {r['region']:18s} DJF {r['DJF']:+5.1f}  MAM {r['MAM']:+5.1f}  "
              f"JJA {r['JJA']:+5.1f}  SON {r['SON']:+5.1f}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_seasonality.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
