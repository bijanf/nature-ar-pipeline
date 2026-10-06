"""Worked examples of DYNAMICALLY driven corridors.

The two corridors with the largest robust circulation (dynamic) contributions to
their observed IVT intensification (Figure 2) — Western Europe and southeastern
South America. For each we show the mean moisture corridor, where it has
intensified (red ΔIVT contours), and the accompanying 500 hPa geopotential-height
change (black contours) — the circulation reorganisation that drives the dynamic
term. Built from the global ERA5 monthly fields already on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.figures._style import apply_nature_style, COL_DOUBLE_IN

_G = 9.80665
_DIR = Path("/p/projects/poem/fallah/nature_ar_data/era5_global_monthly")

# extent [lon_min, lon_max, lat_min, lat_max] in -180..180; dlev levels for ΔIVT
_REGIONS = {
    "europe": dict(extent=[-45, 25, 33, 66], dlev=[10, 20, 30], label="Western Europe"),
    "southamerica": dict(extent=[-80, -35, -45, -8], dlev=[6, 12, 18],
                         label="southeastern South America"),
    "amazon": dict(extent=[-85, -40, -25, 12], dlev=[10, 20, 30], label="Amazon outflow"),
}


def _fields(path: Path):
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds[["q", "u", "v", "z"]].mean(tdim)
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180)).sortby("longitude")
    lev = "pressure_level" if "pressure_level" in ds.dims else "level"
    q = ds["q"].sortby(lev); u = ds["u"].sortby(lev); v = ds["v"].sortby(lev)
    pa = q[lev].astype("float64") * 100.0
    qP = q.assign_coords({lev: pa}); uP = u.assign_coords({lev: pa}); vP = v.assign_coords({lev: pa})
    ivt = np.hypot((qP * uP).integrate(lev) / _G, (qP * vP).integrate(lev) / _G)
    z500 = ds["z"].sel({lev: 500}) / _G
    return ivt, u.sel({lev: 850}), v.sel({lev: 850}), z500


def plot(region_key: str, out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    spec = _REGIONS[region_key]
    ext = spec["extent"]
    apply_nature_style()
    ivt_r, u_r, v_r, z_r = _fields(_DIR / "recent_monthly_global.nc")
    ivt_p, _, _, z_p = _fields(_DIR / "presat_monthly_global.nc")
    d_ivt = ivt_r - ivt_p
    d_z = z_r - z_p
    lon = ivt_r["longitude"].to_numpy(); lat = ivt_r["latitude"].to_numpy()
    pc = ccrs.PlateCarree()

    fig = plt.figure(figsize=(COL_DOUBLE_IN, COL_DOUBLE_IN * 0.6))
    ax = fig.add_subplot(111, projection=pc)
    ax.set_extent(ext, crs=pc)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f4f8fb", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e9e4d8", zorder=1)

    inbox = ivt_r.sel(longitude=slice(ext[0], ext[1]), latitude=slice(ext[3], ext[2]))
    vmax = float(np.nanpercentile(inbox.to_numpy(), 99))
    cf = ax.contourf(lon, lat, ivt_r.to_numpy(), levels=np.linspace(0, vmax, 11),
                     cmap="YlGnBu", extend="max", transform=pc, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="#46505c", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#8a939d", zorder=5)

    st = 3
    ax.quiver(lon[::st], lat[::st], u_r.to_numpy()[::st, ::st], v_r.to_numpy()[::st, ::st],
              transform=pc, scale=240, width=0.003, color="#33404d", alpha=0.8, zorder=6)

    cs = ax.contour(lon, lat, d_ivt.to_numpy(), levels=spec["dlev"], colors="#c0392b",
                    linewidths=[0.8, 1.1, 1.4], transform=pc, zorder=7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="+%.0f")
    dz_box = d_z.sel(longitude=slice(ext[0], ext[1]), latitude=slice(ext[3], ext[2]))
    zmax = float(np.nanpercentile(np.abs(dz_box.to_numpy()), 98))
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
    print(f"wrote {out_path} ({spec['label']})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", choices=list(_REGIONS), default=None)
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    keys = [args.region] if args.region else list(_REGIONS)
    for k in keys:
        plot(k, args.outdir / f"fig_example_{k}.pdf")


if __name__ == "__main__":
    main()
