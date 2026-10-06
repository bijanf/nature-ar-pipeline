"""SI figure: sensitivity of the dynamic term to the corridor box definition.

The corridors are fixed rectangles, so a referee may ask whether the verdict depends
on the exact edges. For each corridor we generate a family of perturbed boxes --
each edge shifted by +/-5 deg, the whole box translated in latitude and longitude,
and the area scaled by 0.7-1.4x -- recompute the point dynamic term for every
perturbation, and plot the spread against the headline value. The robust corridors
keep a clearly positive dynamic term across the family, so the result is not an
artefact of the particular rectangles chosen.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis.decomposition import _DIR, _FILES, _three_term
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665


def _box_means(ds, box):
    lon0, lon1, lat0, lat1 = box
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    sub = ds.sel(longitude=slice(lon0, lon1))
    sub = sub.sel(latitude=slice(lat1, lat0)) if float(ds.latitude[0]) > float(ds.latitude[-1]) \
        else sub.sel(latitude=slice(lat0, lat1))
    if sub.latitude.size == 0 or sub.longitude.size == 0:
        return None
    q = sub["q"].sortby(lev); u = sub["u"].sortby(lev); v = sub["v"].sortby(lev)
    pa = q[lev].astype("float64") * 100.0
    q = q.assign_coords({lev: pa}); u = u.assign_coords({lev: pa}); v = v.assign_coords({lev: pa})
    ivt = np.hypot((q * u).integrate(lev) / _G, (q * v).integrate(lev) / _G)
    iwv = q.integrate(lev) / _G
    w = np.cos(np.deg2rad(sub["latitude"]))
    ivt_y = ivt.weighted(w).mean(("latitude", "longitude")).groupby(f"{tdim}.year").mean(tdim).to_numpy()
    iwv_y = iwv.weighted(w).mean(("latitude", "longitude")).groupby(f"{tdim}.year").mean(tdim).to_numpy()
    return ivt_y, iwv_y


def _perturbations(box):
    lon0, lon1, lat0, lat1 = box
    out = [box]
    for d in (-5, 5):
        out += [(lon0 + d, lon1, lat0, lat1), (lon0, lon1 + d, lat0, lat1),
                (lon0, lon1, lat0 + d, lat1), (lon0, lon1, lat0, lat1 + d),
                (lon0 + d, lon1 + d, lat0, lat1), (lon0, lon1, lat0 + d, lat1 + d)]
    clon = (lon0 + lon1) / 2; clat = (lat0 + lat1) / 2
    for s in (0.7, 1.4):
        out.append((clon + (lon0 - clon) * s, clon + (lon1 - clon) * s,
                    clat + (lat0 - clat) * s, clat + (lat1 - clat) * s))
    return out


def plot(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    apply_nature_style()
    dp = xr.open_dataset(_DIR / _FILES["presat"])
    dr = xr.open_dataset(_DIR / _FILES["recent"])

    spreads = []; heads = []; names = []
    for name, *box in _REGIONS:
        vals = []
        for pb in _perturbations(box):
            mp = _box_means(dp, pb); mr = _box_means(dr, pb)
            if mp is None or mr is None:
                continue
            _, _, d_dy, _ = _three_term(mp[0], mp[1], mr[0], mr[1])
            vals.append(float(d_dy))
        spreads.append(vals); heads.append(vals[0]); names.append(name)

    order = np.argsort(heads)
    names = [names[i] for i in order]; spreads = [spreads[i] for i in order]
    heads = [heads[i] for i in order]; yy = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.42))
    for i, v in enumerate(spreads):
        ax.scatter(v, np.full(len(v), i), s=8, color="#7fb3d5", alpha=0.7, zorder=2)
    ax.scatter(heads, yy, s=26, color="#16415f", marker="D", zorder=3, label="headline box")
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("dynamic term over perturbed corridor boxes (kg m$^{-1}$ s$^{-1}$)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    for n, v in zip(names, spreads):
        print(f"  {n:18s} headline {v[0]:+5.2f}  range [{min(v):+5.2f}, {max(v):+5.2f}]")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_box_sensitivity.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
