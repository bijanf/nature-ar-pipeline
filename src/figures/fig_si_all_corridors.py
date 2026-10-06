"""SI figure: ΔIVT and Δz500 for all eight corridors.

To show the two main-text worked examples are not cherry-picked, we map the
recent-minus-pre-satellite change in column IVT (shading) and the accompanying
500 hPa geopotential-height change (contours) for every corridor on a common
layout, each zoomed to its box. The circulation-driven corridors show coherent
height changes co-located with the transport increase; the thermodynamic corridors
show a transport increase with little organised height change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.analysis.decomposition import _DIR, _FILES
from src.figures._style import apply_nature_style, COL_DOUBLE_IN
from src.figures.fig_global_ar_regions import _REGIONS

_G = 9.80665


def _mean_fields(win):
    ds = xr.open_dataset(_DIR / _FILES[win])
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds[["q", "u", "v", "z"]].mean(tdim)
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180)).sortby("longitude")
    z500 = ds["z"].sel({lev: 500}) / _G
    q = ds["q"].sortby(lev); u = ds["u"].sortby(lev); v = ds["v"].sortby(lev)
    pa = q[lev].astype("float64") * 100.0
    q = q.assign_coords({lev: pa}); u = u.assign_coords({lev: pa}); v = v.assign_coords({lev: pa})
    ivt = np.hypot((q * u).integrate(lev) / _G, (q * v).integrate(lev) / _G)
    return ivt, z500


def _wrap(lon):
    return (lon + 180) % 360 - 180


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    apply_nature_style()

    ivt_r, z_r = _mean_fields("recent")
    ivt_p, z_p = _mean_fields("presat")
    d_ivt = (ivt_r - ivt_p); d_z = (z_r - z_p)
    lon = d_ivt["longitude"].to_numpy(); lat = d_ivt["latitude"].to_numpy()
    pc = ccrs.PlateCarree()

    # Square panels (uniform shape, no thin strips) with a per-corridor window so
    # each region reads with proper geographic context. The SE Australia/NZ corridor
    # box sits offshore in the Tasman Sea, so its window is widened and shifted west
    # to show the whole continent plus New Zealand.
    _VIEW = {"SE Australia / NZ": (147.0, -30.0, 32.0)}
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.54))
    for k, (name, lo0, lo1, la0, la1) in enumerate(_REGIONS):
        if name in _VIEW:
            clon, clat, half = _VIEW[name]
        else:
            clon = (_wrap(lo0) + _wrap(lo1)) / 2.0
            clat = (la0 + la1) / 2.0
            half = 19.0
        ext = [clon - half, clon + half,
               max(clat - half, -88.0), min(clat + half, 88.0)]
        ax = fig.add_subplot(2, 4, k + 1, projection=pc)
        ax.set_extent(ext, crs=pc)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#eeeae0", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, edgecolor="0.4", zorder=4)
        dmax = float(np.nanpercentile(np.abs(d_ivt.sel(
            longitude=slice(ext[0], ext[1]), latitude=slice(ext[3], ext[2])).to_numpy()), 98))
        cf = ax.pcolormesh(lon, lat, d_ivt.to_numpy(), transform=pc, cmap="RdBu_r",
                           vmin=-dmax, vmax=dmax, shading="auto", zorder=1)
        dzb = d_z.sel(longitude=slice(ext[0], ext[1]), latitude=slice(ext[3], ext[2]))
        zmax = float(np.nanpercentile(np.abs(dzb.to_numpy()), 95)) or 1.0
        levs = np.array([-3, -2, -1, 1, 2, 3]) * zmax / 3
        ax.contour(lon, lat, d_z.to_numpy(), levels=levs, colors="k", linewidths=0.5,
                   linestyles=["--", "--", "--", "-", "-", "-"], transform=pc, zorder=3)
        ax.plot([_wrap(lo0), _wrap(lo1), _wrap(lo1), _wrap(lo0), _wrap(lo0)],
                [la0, la0, la1, la1, la0], color="#16a085", lw=0.8, transform=pc, zorder=5)
        ax.set_title(name, fontsize=7, pad=2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04, hspace=0.18, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_si_all_corridors.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
