"""Worked example of a DYNAMICALLY driven corridor: Western Europe.

Unlike the US West Coast (pure thermodynamic moistening), Western Europe's
observed IVT intensification carries a large, robust circulation contribution
(Figure 2). This figure shows what that looks like: the mean moisture-transport
corridor into Western Europe, where it has intensified (red ΔIVT contours), and
the 500 hPa geopotential-height change (black contours) — the circulation
reorganisation that drives the dynamic term. Built from the global ERA5 monthly
fields already on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")
# Western Europe / NE Atlantic, in -180..180 after rolling longitude.
_EXTENT = [-45, 25, 33, 66]


def _fields(path: Path):
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds[["q", "u", "v", "z"]].mean(tdim)
    # roll longitude to -180..180 so Europe + Atlantic are contiguous
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180)).sortby("longitude")
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    q = ds["q"].sortby(lev); u = ds["u"].sortby(lev); v = ds["v"].sortby(lev)
    pa = q[lev].astype("float64") * 100.0
    qP = q.assign_coords({lev: pa}); uP = u.assign_coords({lev: pa}); vP = v.assign_coords({lev: pa})
    ivt_u = (qP * uP).integrate(lev) / _G
    ivt_v = (qP * vP).integrate(lev) / _G
    ivt = np.hypot(ivt_u, ivt_v)
    z500 = ds["z"].sel({lev: 500}) / _G
    return ivt, ivt_u, ivt_v, z500


def plot(out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    apply_nature_style()
    ivt_r, u_r, v_r, z_r = _fields(_DIR / "recent_monthly_global.nc")
    ivt_p, _, _, z_p = _fields(_DIR / "presat_monthly_global.nc")
    d_ivt = ivt_r - ivt_p
    d_z = z_r - z_p

    lon = ivt_r["longitude"].to_numpy(); lat = ivt_r["latitude"].to_numpy()
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.62))
    ax = fig.add_subplot(111, projection=pc)
    ax.set_extent(_EXTENT, crs=pc)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f4f8fb", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e9e4d8", zorder=1)

    vmax = float(np.nanpercentile(ivt_r.sel(longitude=slice(_EXTENT[0], _EXTENT[1]),
                                            latitude=slice(_EXTENT[3], _EXTENT[2])).to_numpy(), 99))
    cf = ax.contourf(lon, lat, ivt_r.to_numpy(), levels=np.linspace(0, vmax, 11),
                     cmap="YlGnBu", extend="max", transform=pc, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="#46505c", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#8a939d", zorder=5)

    st = 3
    ax.quiver(lon[::st], lat[::st], u_r.to_numpy()[::st, ::st], v_r.to_numpy()[::st, ::st],
              transform=pc, scale=2600, width=0.003, color="#33404d", alpha=0.8, zorder=6)

    # intensification (red) and the circulation change driving it (black z500 contours)
    cs = ax.contour(lon, lat, d_ivt.to_numpy(), levels=[10, 20, 30], colors="#c0392b",
                    linewidths=[0.8, 1.1, 1.4], transform=pc, zorder=7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="+%.0f")
    zmax = float(np.nanpercentile(np.abs(d_z.sel(longitude=slice(_EXTENT[0], _EXTENT[1]),
                                                  latitude=slice(_EXTENT[3], _EXTENT[2])).to_numpy()), 98))
    levs = np.array([-3, -2, -1, 1, 2, 3]) * zmax / 3
    cz = ax.contour(lon, lat, d_z.to_numpy(), levels=levs, colors="k", linewidths=0.7,
                    linestyles=["--", "--", "--", "-", "-", "-"], transform=pc, zorder=8)
    ax.clabel(cz, inline=True, fontsize=6, fmt="%.0f")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5, linestyle=":")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
    cb = fig.colorbar(cf, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("mean IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=8); cb.ax.tick_params(labelsize=7.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures") / "fig_europe_example.pdf")
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
